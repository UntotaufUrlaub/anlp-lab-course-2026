import requests
import time
import json
from datasets import load_dataset
import os
from dotenv import load_dotenv
import random
from openai import OpenAI
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed


#--------------------------calls the Semantic Scholar API to gather paper information--------------------------
# Send the API request only for one request to try out
def send_api_request(url, query_params):
    # exponential backoff
    delay = 1

    # Check r status
    for i in range(10):
        r = requests.get(url, params=query_params)

        if r.status_code == 429:
            print(f"429 -> {delay} seconds waiting")
            time.sleep(delay)
            delay *= 2
        elif r.status_code == 200:
            # get r data as needed
            data = r.json()
            print("data acquired")
            # print(json.dumps(data, indent=2))
            return data
        else:
            raise Exception(
                f"Request failed:{r.status_code}: {r.text}"
            )  # can also change to return None instead


# extracts all corpus_ids and puts them in separate cache file for calling later, has to be run first
def get_list_of_corpus_ids():
    corpus_ids = []
    corpus_clean_data = load_dataset(
        "princeton-nlp/LitSearch", "corpus_clean", split="full", streaming=True
    )

    for i, paper in enumerate(corpus_clean_data):
        corpus_ids.append(paper["corpusid"])
        if i % 10000 == 0:
            print(f"{i} paper_ids processed already.")

    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "corpus_ids.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)


# call the API with backoff
load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # get api key


# does exponential backoff
def post_with_backoff(url, *, params, json_payload, headers, max_retries=6):
    delay = 2

    for attempt in range(max_retries):
        response = requests.post(
            url, params=params, json=json_payload, headers=headers, timeout=30
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code in {429, 500, 502, 503, 504}:
            wait = delay + random.uniform(0, 0.5)
            print(
                f"Retry {attempt + 1}: status {response.status_code}, waiting {wait:.1f}s"
            )
            time.sleep(wait)
            delay *= 2
            continue

        print(response.text)
        response.raise_for_status()

    response.raise_for_status()


# calls the batch api and requests information
def get_paper_batch(paper_url, paper_ids):
    url = paper_url

    params = {
        "fields": "corpusId,publicationVenue,fieldsOfStudy,publicationTypes,journal,authors,authors.name,"
        "authors.affiliations"
    }

    payload = {"ids": [f"CorpusId:{cid}" for cid in paper_ids]}

    headers = {"x-api-key": API_KEY}

    return post_with_backoff(url, params=params, json_payload=payload, headers=headers)


# out of the whole corpus_ids file create queryable chunks
def chunks(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_json_if_exists(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# extracts data with a cache, so if something crashes can continue from previous endpoint
def data_extraction(corpus_ids_file, batch_size=250):
    with open(corpus_ids_file, "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)

    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, "paper_cache.json")
    processed_path = os.path.join(output_dir, "processed_ids.json")

    failed_path = os.path.join(output_dir, "failed_ids.json")
    failed_ids = set(load_json_if_exists(failed_path, []))

    # load previous progress
    all_results = load_json_if_exists(cache_path, {})
    processed_ids = set(load_json_if_exists(processed_path, []))

    for batch in chunks(corpus_ids, batch_size):

        # skip already processed ids
        batch = [cid for cid in batch if str(cid) not in processed_ids]
        print(f"Processing batch length: {len(batch)}")

        if len(batch) == 0:
            continue
        try:
            papers = get_paper_batch(
                "https://api.semanticscholar.org/graph/v1/paper/batch", batch
            )
        except requests.exceptions.HTTPError as e:
            print("Failed batch:", batch[:10])
            failed_ids.update(str(cid) for cid in batch)
            save_json(failed_path, list(failed_ids))

            processed_ids.update(str(cid) for cid in batch)
            continue

        for paper in papers:

            if paper is None:
                continue

            corpus_id = str(paper["corpusId"])

            all_results[corpus_id] = paper
            processed_ids.add(corpus_id)

        # SAVE AFTER EVERY BATCH
        save_json(cache_path, all_results)
        save_json(processed_path, list(processed_ids))

        print(f"Processed: {len(processed_ids)}")

        time.sleep(3)


# determine papers that have not been found during the enrichment process
def detect_missing_ids():
    missing_ids = []
    with open("cache/corpus_ids.json", "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open("cache/paper_cache.json", "r", encoding="utf-8") as f:
        cache = json.load(f)

    keys = cache.keys()

    for id in corpus_ids:
        if str(id) not in keys:
            missing_ids.append(str(id))
    print(f"Number of missing ids: {len(missing_ids)}.")

    with open("cache/missing_corpus_ids.json", "w", encoding="utf-8") as f:
        json.dump(missing_ids, f, indent=2, ensure_ascii=False)

def run_semantic_scholar_api_call_program(batch_size):
    print("🔍 Finding all paper ids in the papers dataset...")
    get_list_of_corpus_ids()

    print("\n Starting API calling, please be patient...")
    data_extraction("cache/corpus_ids.json", batch_size=batch_size)

    print("\n API call complete!")
    print("\n 🔍 Detecting missing ids...")
    detect_missing_ids()
    print("\n✅ Done!")


#--------------------------OpenAI cost estimations--------------------------
# the number is dollars per 1M tokens

MODEL_MAPPING = {
    "gpt-5-nano":{
        "input":0.05,
        "cached":0.005,
        "output":0.4,
    },
    "gpt-5-mini":{
        "input": 0.25,
        "cached": 0.025,
        "output": 2

    }
}
# helper function for pretty printing
def pretty_cost_report(result):
    return f"""
    Model: {result['model']}
    Number of Queries: {result['number_of_queries']:,}

    Input Tokens : {result['input_tokens']:,}
    Output Tokens: {result['output_tokens']:,}

    Input Cost   : ${result['input_cost']:}
    Output Cost  : ${result['output_cost']:}

    Total Cost   : ${result['total_cost']:}
    """

# estimates the cost for queries
def api_cost_estimator(model, number_of_queries, avg_input_tokens, avg_output_tokens):
    prices = MODEL_MAPPING.get(model)

    if prices is None:
        raise ValueError(f"{model} is not supported.")

    total_input_tokens = avg_input_tokens * number_of_queries
    total_output_tokens = avg_output_tokens * number_of_queries

    input_cost = total_input_tokens * prices["input"] / 1e6
    output_cost = total_output_tokens * prices["output"] / 1e6

    result = {
        "model": model,
        "number_of_queries": number_of_queries,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }
    return result, pretty_cost_report(result)

# estimate number of average input and output tokens
SECTORS = """["Academic","Industry","Government","Healthcare","Nonprofit","Independent"]"""

FIELDS = """FIELDS = ["Computer Science","Mathematics","Engineering","Medicine","Biology","Physics",
    "Economics","Psychology","Social Sciences","Other"]"""

GENERAL_RESEARCH_AREA = """["Machine Learning","Natural Language Processing","Computer Vision",
    "Information Retrieval","Knowledge Representation and Reasoning","Speech and Audio Processing",
    "Robotics and Control","Human-Computer Interaction","Software Engineering","Databases","Security and Privacy",
    "Optimization","Statistics","Theoretical Computer Science","Computational Biology","Biomedical AI",
    "Multimodal Learning","Data Mining","Distributed Systems","Other"]"""

METHOD_FAMILY = """["Rule-Based Methods","Statistical Methods","Probabilistic Models","Machine Learning","Deep Learning",
    "Neural Networks","Transformer Models","Graph Neural Networks","Knowledge Graph Methods","Embedding Methods",
    "Representation Learning","Retrieval Methods","Ranking Methods","Sequence Labeling","Classification",
    "Clustering","Structured Prediction","Parsing Algorithms","Ontology-Based Methods","Lexicon-Based Methods",
    "Corpus-Based Methods","Active Learning","Semi-Supervised Learning","Self-Supervised Learning",
    "Transfer Learning","Multi-Task Learning","Meta Learning","Reinforcement Learning","Evaluation Frameworks",
    "Dataset Construction","Annotation Frameworks","Other"]"""

JSON_SCHEMA = """{
  "affiliations": [
    {"country": "", "sector": "", "organization": ""}
  ],
  "field_of_study_path": {
    "field": "",
    "research_area": "",
    "topic_family": "",
    "specific_topic": ""
  },
  "method_path": {
    "method_family": "",
    "method_category": "",
    "specific_method": ""
  }
}"""

def construct_user_prompt(title, abstract, affiliation, fields_of_study):
    return f"""Fill this schema {JSON_SCHEMA} with following  paper metadata:

                        Title: {title}
                        
                        Abstract: {abstract}
                        
                        Affiliations: {affiliation}
                        
                        Fields of Study: {fields_of_study}
                        
                        Guidelines:
                        - If Affiliations is empty, return exactly: "affiliations": []
                          Never infer affiliations from title, abstract, venue, acknowledgements, 
                          or field of study.
                          Do not create null affiliation objects.
                        - If Affiliations not empty, use affiliations exactly as organization names.
                        - Create one affiliation entry per affiliation string.
                        - Fields of Study are strong evidence for field.
                        - Choose the primary topic and primary method.
                        - Do not invent categories outside the allowed lists.
                        - Keep specific_topic, method_category, and specific_method short: maximum 6 words each.
                        - Prefer reusable labels over paper-specific descriptions.
                        -Hierarchy rule:
                        Each level must be a more specific subcategory of the previous level.
                        Do not repeat the same category at two levels.
                        Do not put a broader category after a narrower category.
                        -Do not assign Knowledge Graphs, Information Retrieval, or Knowledge Representation unless the 
                        title or abstract explicitly mentions retrieval, search, ranking, knowledge graphs, 
                        entities, relations, ontologies, or reasoning.
                        - research_area must describe the main research area of the paper.
                        - topic_family must describe what problem/domain the paper studies.
                        - specific_topic must describe the paper’s concrete research focus.
                        - method_family must describe the technical approach family.
                        - method_category must describe the type of method used.
                        - specific_method must name the concrete method, algorithm, model, framework, or technique.
                        """

def construct_system_prompt():
    return f"""
            You are a scientific metadata extraction system.
            Use null if unknown.
            Use the closest valid category when uncertain.
            
            Allowed sectors: {SECTORS}
            Allowed field: {FIELDS}
            Allowed research_area: {GENERAL_RESEARCH_AREA}
            Allowed method_family: {METHOD_FAMILY}"""

def pretty_print_token_report(result):
    return f"""
        Model: {result['model']}

        AVG Input Tokens : {result['avg_input_tokens']:,}
        AVG Output Tokens: {result['avg_output_tokens']:,}

        Total_Input_Tokens: {result['total_input_tokens']:,}
        Total_Output_Tokens : {result['total_output_tokens']:,}

        Total Tokens: {result['total_tokens']:,}
        """

def estimate_avg_input_output_tokens(
        model,
        sample_size=1000

):
    enc = tiktoken.encoding_for_model(model)

    docs = []
    with open("output/documents_enriched_02.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("source_type")=="paper":
                docs.append(doc)

    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for paper in random.sample(docs, sample_size):
        title=(paper["title"])
        abstract=(paper["main_text"])
        affiliation=(paper.get("entities").get("affiliations"))
        fields_of_study=(paper.get("structured_fields").get("categorical").get("fields_of_study"))

        user_prompt = construct_user_prompt(title, abstract, affiliation, fields_of_study)

        system_prompt =construct_system_prompt()

        output_example = """{
                      "hierarchical": {
                        "affiliations": [
                          {"country": "United States", "sector": "Academic", "organization": "University of Michigan"},
                          {"country": "Uruguay", "sector": "Academic", "organization": "Universidad de la República"}
                        ],
                        "field_of_study_path": {
                          "field": "Computer Science",
                          "research_area": "Machine Learning",
                          "topic_family": "Knowledge Representation",
                          "specific_topic": "Video Representation"
                        },
                        "method_path": {
                          "method_family": "Deep Learning",
                          "method_category": "Transformer Models",
                          "specific_method": "Video Tokenization"
                        }
                      }
                    }"""
        input_tokens = len(enc.encode(user_prompt)) + len(enc.encode(system_prompt))
        output_tokens = len(enc.encode(output_example))

        total_tokens += input_tokens + output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    result = {
        "model": model,
        "avg_input_tokens": total_input_tokens/sample_size,
        "avg_output_tokens": total_output_tokens/sample_size,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens
    }

    return result, pretty_print_token_report(result)

def run_cost_estimation(model, number_of_queries, MAX_BUDGET):
    print("Calculating average input and output tokens ...")
    token_info, token_report = estimate_avg_input_output_tokens(model, 1000)
    print("\nCalculating costs...")
    cost_info, cost_report = api_cost_estimator(model, number_of_queries,
                                        token_info["avg_input_tokens"], token_info["avg_output_tokens"])
    print(token_report)
    print("\n")
    print(cost_report)

    total_cost = cost_info['total_cost']
    if total_cost > MAX_BUDGET:
        raise ValueError( f"❌ Estimated cost ${total_cost:} exceeds budget.")
    else:
        print(f"✅ Run within budget.")



#--------------------------calling OpenAI API for hierarchy information one request at a time------------------------


def extract_paper_fields(doc):
    s = ";"

    title = doc.get("title") or ""
    abstract = doc.get("main_text") or ""

    authors = s.join(
        doc.get("entities", {}).get("people", []) or []
    )

    affiliations = s.join(
        doc.get("entities", {}).get("organizations", []) or []
    )

    fields_of_study = s.join(
        doc.get("structured_fields", {})
           .get("categorical", {})
           .get("fields_of_study", []) or []
    )

    paper_id = doc.get("raw_source", {}).get("native_id")

    return paper_id, title, abstract, authors, affiliations, fields_of_study


def process_single_paper(doc, model):
    client = OpenAI()

    paper_id, title, abstract, authors, affiliations, fields_of_study = extract_paper_fields(doc)

    system_prompt = construct_system_prompt()
    user_prompt = construct_user_prompt(
        title,
        abstract,
        affiliations,
        fields_of_study
    )

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        reasoning={"effort": "minimal"},
        max_output_tokens=300,
    )

    return paper_id, {
        "base_info": {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "affiliations": affiliations,
            "fields_of_study": fields_of_study,
        },
        "hierarchy_info": response.output_text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    }


def load_paper_docs(corpus_path, processed_ids, batch_size):
    docs = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            if doc.get("source_type") != "paper":
                continue

            paper_id = doc.get("raw_source", {}).get("native_id")

            if paper_id is None:
                continue

            if paper_id in processed_ids:
                continue

            docs.append(doc)

            if len(docs) >= batch_size:
                break

    return docs


def hierarchy_information_extraction_parallel(
    model,
    corpus_path,
    batch_size=100,
    max_workers=10,
    output_cache_path="cache/hierarchy_cache_nano.json",
):
    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)

    if os.path.exists(output_cache_path):
        with open(output_cache_path, "r", encoding="utf-8") as f:
            hierarchy_cache = json.load(f)
    else:
        hierarchy_cache = {}

    processed_ids = set(hierarchy_cache.keys())

    docs = load_paper_docs(
        corpus_path=corpus_path,
        processed_ids=processed_ids,
        batch_size=batch_size,
    )

    print(f"Loaded {len(docs)} unprocessed papers.")
    print(f"Starting API calls with {max_workers} workers...")

    start = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_paper, doc, model)
            for doc in docs
        ]

        for future in as_completed(futures):
            try:
                paper_id, result = future.result()

                hierarchy_cache[paper_id] = result
                completed += 1

                with open(output_cache_path, "w", encoding="utf-8") as f:
                    json.dump(hierarchy_cache, f, ensure_ascii=False, indent=4)

            except Exception as e:
                failed += 1
                print(f"\nRequest failed: {e}")

            elapsed = time.time() - start
            avg_time = elapsed / max(completed + failed, 1)
            remaining = avg_time * (len(docs) - completed - failed)

            print(
                f"\rDone: {completed}/{len(docs)} | "
                f"Failed: {failed} | "
                f"Elapsed: {elapsed/60:.1f} min | "
                f"ETA: {remaining/60:.1f} min",
                end=""
            )

    print("\n✅ Finished batch.")
    return hierarchy_cache

#--------------------------calling OpenAI Batch API for hierarchy information in batches------------------------

def build_hierarchy_batch_request(doc, model):
    paper_id, title, abstract, authors, affiliations, fields_of_study = extract_paper_fields(doc)

    if paper_id is None:
        return None

    return {
        "custom_id": str(paper_id),
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": construct_system_prompt(),
            "input": construct_user_prompt(
                title,
                abstract,
                affiliations,
                fields_of_study
            ),
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": 300,
        }
    }


def load_submitted_batch_ids(manifest_path):
    submitted_ids = set()

    if not os.path.exists(manifest_path):
        return submitted_ids

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for batch in manifest.get("batches", []):
        submitted_ids.update(str(paper_id) for paper_id in batch.get("paper_ids", []))

    return submitted_ids


def append_batch_manifest(manifest_path, batch_info):
    manifest = load_json_if_exists(manifest_path, {"batches": []})
    manifest["batches"].append(batch_info)
    save_json(manifest_path, manifest)


def write_openai_batch_input_file(requests, path):
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def submit_openai_batch_file(client, input_path, metadata=None):
    with open(input_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    return client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata or {}
    )


def submit_hierarchy_information_batches(
    model,
    corpus_path,
    request_batch_size=1000,
    output_dir="cache/openai_hierarchy_batches",
    manifest_path=None,
    max_batches=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if manifest_path is None:
        manifest_path = os.path.join(output_dir, "batch_manifest.json")

    submitted_ids = load_submitted_batch_ids(manifest_path)
    client = OpenAI()

    requests_to_submit = []
    paper_ids = []
    submitted_batches = []

    def submit_current_batch(batch_number):
        input_path = os.path.join(output_dir, f"hierarchy_batch_{batch_number:05d}.jsonl")
        write_openai_batch_input_file(requests_to_submit, input_path)

        batch = submit_openai_batch_file(
            client,
            input_path,
            metadata={
                "description": "paper hierarchy extraction",
                "model": model,
                "batch_number": str(batch_number),
            }
        )

        batch_info = {
            "batch_id": batch.id,
            "input_file_id": batch.input_file_id,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "status": batch.status,
            "input_path": input_path,
            "paper_ids": paper_ids.copy(),
        }
        append_batch_manifest(manifest_path, batch_info)
        submitted_batches.append(batch_info)
        print(f"Submitted batch {batch.id} with {len(paper_ids)} requests.")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            if doc.get("source_type") != "paper":
                continue

            paper_id = doc.get("raw_source", {}).get("native_id")

            if paper_id is None or str(paper_id) in submitted_ids:
                continue

            request = build_hierarchy_batch_request(doc, model)

            if request is None:
                continue

            requests_to_submit.append(request)
            paper_ids.append(str(paper_id))

            if len(requests_to_submit) >= request_batch_size:
                submit_current_batch(len(submitted_batches) + 1)
                submitted_ids.update(paper_ids)
                requests_to_submit = []
                paper_ids = []

                if max_batches is not None and len(submitted_batches) >= max_batches:
                    return submitted_batches

    if requests_to_submit and (max_batches is None or len(submitted_batches) < max_batches):
        submit_current_batch(len(submitted_batches) + 1)

    return submitted_batches

def refresh_batch_manifest(manifest_path):
    client = OpenAI()
    manifest = load_json_if_exists(manifest_path, {"batches": []})

    for batch_info in manifest["batches"]:
        batch_id = batch_info["batch_id"]

        batch = client.batches.retrieve(batch_id)

        batch_info["status"] = batch.status
        batch_info["output_file_id"] = batch.output_file_id
        batch_info["error_file_id"] = batch.error_file_id

        print(
            f"Batch {batch_id}: {batch.status} | "
            f"output={batch.output_file_id} | error={batch.error_file_id}"
        )

    save_json(manifest_path, manifest)
    return manifest

def extract_output_text(response_body):
    # For /v1/responses output, this is usually available directly
    if "output_text" in response_body:
        return response_body["output_text"]

    # fallback for nested output structure
    output_parts = []
    for item in response_body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                output_parts.append(content.get("text", ""))

    return "\n".join(output_parts)


def download_completed_batch_outputs(
    manifest_path,
    output_cache_path="cache/hierarchy_cache_nano_batch.json",
    downloaded_dir="cache/openai_hierarchy_batches/downloaded_outputs",
):
    os.makedirs(downloaded_dir, exist_ok=True)

    client = OpenAI()
    manifest = load_json_if_exists(manifest_path, {"batches": []})
    hierarchy_cache = load_json_if_exists(output_cache_path, {})

    for batch_info in manifest["batches"]:
        if batch_info.get("status") != "completed":
            print(f"Skipping {batch_info['batch_id']} because status is {batch_info.get('status')}")
            continue

        output_file_id = batch_info.get("output_file_id")

        if not output_file_id:
            print(f"No output file for {batch_info['batch_id']}")
            continue

        local_output_path = os.path.join(
            downloaded_dir,
            f"{batch_info['batch_id']}_output.jsonl"
        )

        # Download only if not already downloaded
        if not os.path.exists(local_output_path):
            content = client.files.content(output_file_id)
            content.write_to_file(local_output_path)

        with open(local_output_path, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)

                paper_id = result["custom_id"]

                if result.get("error") is not None:
                    hierarchy_cache[paper_id] = {
                        "error": result["error"]
                    }
                    continue

                response_body = result["response"]["body"]
                output_text = extract_output_text(response_body)

                hierarchy_cache[paper_id] = {
                    "hierarchy_info": output_text,
                    "raw_response": response_body,
                }

        save_json(output_cache_path, hierarchy_cache)
        print(f"Downloaded and merged {batch_info['batch_id']}")

    print(f"Total cached results: {len(hierarchy_cache)}")
    return hierarchy_cache


#--------------------------main program-----------------------------------------------------------------
if __name__ == "__main__":
    # choose the batch size with wich semantic scholar responds
    # batch size has to be < 500 as to not overload the api
    # semantic_scholar_batch_size = 250
    # run_semantic_scholar_api_call_program(semantic_scholar_batch_size)
    model = "gpt-5-nano"
    # run_cost_estimation(model, 5000, 10)
    hierarchy_information_extraction_parallel(
        model=model,
        corpus_path="output/documents_enriched_01.jsonl",
        batch_size=5000,
        max_workers=5,
        output_cache_path="cache/hierarchy_cache_nano_4.json",
    )
