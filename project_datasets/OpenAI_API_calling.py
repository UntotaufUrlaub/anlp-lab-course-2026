import time
import json
import os
from dotenv import load_dotenv
import random
from openai import OpenAI
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from project_datasets.SemanticScholar_API_calling import load_json_if_exists, save_json

load_dotenv()

#--------------------------OpenAI cost estimations--------------------------
# the number is dollars per 1M tokens

MODEL_MAPPING = {
    "gpt-5-nano":{
        "input":0.05,
        "cached":0.005,
        "output":0.4
    },
    "gpt-5-mini":{
        "input": 0.25,
        "cached": 0.025,
        "output": 2
    }
}

TIER_TOKEN_LIMITS = {
    1: 2_000_000,
    2: 20_000_000
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
        """

def estimate_avg_input_output_tokens(
        model,
        sample_size=1000,
        document = "output/documents_enriched_01.jsonl"

):
    enc = tiktoken.encoding_for_model(model)

    docs = []
    with open(document, "r", encoding="utf-8") as f:
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
    }

    return result, pretty_print_token_report(result)

def run_cost_estimation(
    model,
    number_of_queries,
    MAX_BUDGET,
    batch_size=1000,
    tier=1,
    using_batch_api=False,
):
    print("\n" + "=" * 70)
    print("📊 COST ESTIMATION")
    print("=" * 70)

    print("\n🔍 Estimating average token usage...")
    token_info, token_report = estimate_avg_input_output_tokens(model, 1000)

    print("💰 Estimating costs...")
    cost_info, cost_report = api_cost_estimator(
        model,
        number_of_queries,
        token_info["avg_input_tokens"],
        token_info["avg_output_tokens"],
    )

    total_cost = cost_info["total_cost"]

    print("\n" + "-" * 70)
    print("TOKEN ESTIMATION")
    print("-" * 70)

    print(f"Model               : {model}")
    print(f"Queries             : {number_of_queries:,}")
    print(f"Avg Input Tokens    : {token_info['avg_input_tokens']:,.2f}")
    print(f"Avg Output Tokens   : {token_info['avg_output_tokens']:,.2f}")

    print("\n" + "-" * 70)
    print("COST ESTIMATION")
    print("-" * 70)

    print(f"Input Cost          : ${cost_info['input_cost']:,.4f}")
    print(f"Output Cost         : ${cost_info['output_cost']:,.4f}")
    print(f"Total Cost          : ${total_cost:,.4f}")

    if using_batch_api:
        total_cost /= 2

        enqueued_tokens = batch_size * (
            token_info["avg_input_tokens"]
            + token_info["avg_output_tokens"]
        )

        token_limit = TIER_TOKEN_LIMITS.get(tier)

        print("\n" + "-" * 70)
        print("BATCH API ANALYSIS")
        print("-" * 70)

        print(f"Batch Size                : {batch_size:,}")
        print(f"Tier                      : {tier}")
        print(f"Enqueued Tokens per batch : {enqueued_tokens:,.0f}")

        if token_limit is not None:
            print(f"Tier Limit          : {token_limit:,.0f}")

            if enqueued_tokens > float(token_limit):
                print("Status              : ❌ EXCEEDS TOKEN LIMIT")
            else:
                print("Status              : ✅ WITHIN TOKEN LIMIT")

        print(f"\nBatch API Cost      : ${total_cost:,.4f}")
        print("(50% discount applied)")

    print("\n" + "-" * 70)
    print("BUDGET CHECK")
    print("-" * 70)

    print(f"Budget              : ${MAX_BUDGET:,.2f}")
    print(f"Estimated Cost      : ${total_cost:,.4f}")

    if total_cost > MAX_BUDGET:
        print("\n❌ ESTIMATED COST EXCEEDS BUDGET")
        raise ValueError(
            f"Estimated cost ${total_cost:,.2f} exceeds budget ${MAX_BUDGET:,.2f}"
        )

    print("\n✅ RUN IS WITHIN BUDGET")
    print("=" * 70)




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

ACTIVE_BATCH_STATUSES = {
    "validating",
    "in_progress",
    "finalizing",
}

VALID_BATCH_STATUSES = {
        "validating",
        "in_progress",
        "finalizing",
        "completed",
}


FINISHED_BATCH_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "expired",
}

# build one request with all the info of a batch
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

# loads batch ids already submitted and whose status is valid
def load_submitted_batch_ids(manifest_path):
    submitted_ids = set()

    # if the manifest does not exist yet, return an empty set
    if not os.path.exists(manifest_path):
        return submitted_ids

    # if the manifest already exists we load the document
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # go through each batch and add all paper_ids of each batch to get the set of already submitted, valid ids
    for batch in manifest.get("batches", []):
        status = batch.get("status")

        if status not in VALID_BATCH_STATUSES:
            continue

        submitted_ids.update(str(paper_id) for paper_id in batch.get("paper_ids", []))

    return submitted_ids

# append the batch_info dictionary to the manifest.json object
def append_batch_manifest(manifest_path, batch_info):
    # loads the manifest json file if it exists
    manifest = load_json_if_exists(manifest_path, {"batches": []})
    # appends the batch_info under batches
    manifest["batches"].append(batch_info)
    # saves the json file
    save_json(manifest_path, manifest)

# writes a jsonl file with the information in requests
def write_openai_batch_input_file(requests, path):
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

# submits batch to OpenAI
def submit_openai_batch_file(client, input_path, metadata=None):
    with open(input_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    # OpenAI returns a file with an id: batch_input_file.id
    return client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata or {}
    )

# 1. Create output folder
# 2. Load already submitted paper IDs from manifest
# 3. Open corpus JSONL
# 4. For each paper:
#       - skip if not paper
#       - skip if already submitted
#       - build one OpenAI request
#       - add request to current list
# 5. Once list reaches 1000:
#       - write local JSONL file
#       - upload it to OpenAI
#       - create batch job
#       - save batch metadata in manifest
#       - reset list
# 6. At end, submit leftover papers

# total submission function
def submit_hierarchy_information_batches(
    model,
    corpus_path,
    request_batch_size=1000,
    output_dir="cache/openai_hierarchy_batches",
    manifest_path=None,
    max_batches=None,
    verbose=True,
):
    # make output_idr files for storing the json files for submitting to the API
    os.makedirs(output_dir, exist_ok=True)

    # if manifest not provided we create the manifest file, where all submitted batches are stored by ID
    if manifest_path is None:
        manifest_path = os.path.join(output_dir, "batch_manifest.json")

    # loads already submitted ids of the manifest
    submitted_ids = load_submitted_batch_ids(manifest_path)
    client = OpenAI()

    requests_to_submit = []
    paper_ids = []
    submitted_batches = []

    def submit_current_batch(batch_number):
        #access file with number
        input_path = os.path.join(output_dir, f"hierarchy_batch_{batch_number:05d}.jsonl")
        write_openai_batch_input_file(requests_to_submit, input_path)

        # submit batch to OpenAI and get a reply back containing a batch object:
        # batch.id = batch_xyz
        # batch.status = validating
        # batch.input_file_id = file - abc123
        # batch.output_file_id = None
        # batch.error_file_id = None

        batch = submit_openai_batch_file(
            client,
            input_path,
            metadata={
                "description": "paper hierarchy extraction",
                "model": model,
                "batch_number": str(batch_number),
            }
        )

        # creates a dictionary summarizing the batch
        batch_info = {
            "batch_id": batch.id,
            "input_file_id": batch.input_file_id,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "status": batch.status,
            "input_path": input_path,
            "paper_ids": paper_ids.copy(),
        }

        # appends batch_info to the manifest dictionary
        append_batch_manifest(manifest_path, batch_info)
        # appends the batch to submitted batches
        submitted_batches.append(batch_info)
        if verbose:
            print(f"Submitted batch {batch.id} with {len(paper_ids)} requests.")

    # loads the main corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            # loads each json document
            doc = json.loads(line)

            # if source type is not a paper skip it
            if doc.get("source_type") != "paper":
                continue

            # get native id
            paper_id = doc.get("raw_source", {}).get("native_id")

            # if the paper_id was already submitted we continue
            if paper_id is None or str(paper_id) in submitted_ids:
                continue

            # builds the request object with system and user prompt
            request = build_hierarchy_batch_request(doc, model)

            if request is None:
                continue

            # add to requests to submit array
            requests_to_submit.append(request)
            # add the paper_id to mark processed ids
            paper_ids.append(str(paper_id))

            #if we have enough for a batch
            if len(requests_to_submit) >= request_batch_size:
                submit_current_batch(len(submitted_batches) + 1)
                submitted_ids.update(paper_ids)
                # reset arrays to empty
                requests_to_submit = []
                paper_ids = []

                # stop if max_batches reached
                if max_batches is not None and len(submitted_batches) >= max_batches:
                    return submitted_batches

    # handles the last partial batch
    if requests_to_submit and (max_batches is None or len(submitted_batches) < max_batches):
        submit_current_batch(len(submitted_batches) + 1)

    return submitted_batches

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

# inspect error messages of batch
def inspect_failed_batch(batch_id):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    print("\n===== FAILED BATCH INSPECTION =====")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Input file ID: {batch.input_file_id}")
    print(f"Output file ID: {batch.output_file_id}")
    print(f"Error file ID: {batch.error_file_id}")

    print("\nFull batch object:")
    print(batch.model_dump_json(indent=4))

# remove failed batches from manifest
def remove_failed_batches_from_manifest(manifest_path):
    manifest = load_json_if_exists(manifest_path, {"batches": []})

    before = len(manifest["batches"])

    manifest["batches"] = [
        batch for batch in manifest["batches"]
        if batch.get("status") != "failed"
    ]

    after = len(manifest["batches"])

    save_json(manifest_path, manifest)

    print(f"Removed failed batches: {before - after}")
    print(f"Remaining batches: {after}")

#-------------- batch manager which submits and manages batches automatically--------------------------------
class OpenAIBatchManager:
    LOG_WIDTH = 72
    ACTIVE_STATUSES = ACTIVE_BATCH_STATUSES
    VALID_SUBMITTED_STATUSES = VALID_BATCH_STATUSES

    def __init__(
            self,
            model="gpt-5-nano",
            corpus_path="output/documents_enriched_01.jsonl",
            corpus_ids_path="cache/corpus_ids.json",
            output_dir="cache/openai_hierarchy_batches",
            output_cache_path="cache/hierarchy_cache_nano_batch.json",
            request_batch_size=800,
            max_active_batches=2,
            check_interval_seconds=30,
    ):
        self.model = model

        # path variables
        self.corpus_path = corpus_path
        self.corpus_ids_path = corpus_ids_path
        self.output_dir = output_dir
        self.output_cache_path = output_cache_path

        self.manifest_path = os.path.join(output_dir, "batch_manifest.json")
        self.downloaded_dir = os.path.join(output_dir, "downloaded_outputs")

        # API calling variables
        self.request_batch_size = request_batch_size
        self.max_active_batches = max_active_batches
        self.check_interval_seconds = check_interval_seconds

        self.client = OpenAI()

        # making the directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.downloaded_dir, exist_ok=True)

    #---------------------------helper methods-------------------------------------------
    def print_section(self, title):
        print("=" * self.LOG_WIDTH)
        print(title)
        print("=" * self.LOG_WIDTH)

    def print_key_value(self, label, value):
        print(f"{label:<24}: {value}")

    def format_number(self, value):
        return f"{value:,}"

    def clear_console(self):
        os.system("cls" if os.name == "nt" else "clear")

    # to load the output_cache
    def load_output_cache(self):
        return load_json_if_exists(self.output_cache_path, {})

    # to save the output_cache
    def save_output_cache(self, cache_data):
        save_json(self.output_cache_path, cache_data)

    # to load the manifest
    def load_manifest(self):
        return load_json_if_exists(self.manifest_path, {"batches": []})

    # to save the manifest
    def save_manifest(self, manifest):
        save_json(self.manifest_path, manifest)

    # print summary of the current manifest
    def get_manifest_status_counts(self):
        manifest = self.load_manifest()
        return Counter(
            batch.get("status", "unknown")
            for batch in manifest.get("batches", [])
        )

    # print summary of the current manifest
    def print_manifest_summary(self):
        manifest = self.load_manifest()
        manifest_counter = self.get_manifest_status_counts()

        print("\nBatch status summary")
        print("-" * self.LOG_WIDTH)
        if manifest_counter:
            for status, count in sorted(manifest_counter.items()):
                self.print_key_value(status, count)
        else:
            print("No batches in manifest.")

        self.print_key_value("Total batches", len(manifest.get("batches", [])))

    def print_manager_cycle_summary(self, cycle_number, new_downloads, next_check=None):
        manifest = self.load_manifest()
        manifest_counter = self.get_manifest_status_counts()

        self.print_section(f"MANAGER CYCLE {cycle_number}")
        print()
        self.print_key_value("Model", self.model)
        self.print_key_value("Active batches", f"{self.count_active_batches()}/{self.max_active_batches}")
        self.print_key_value("Cached results", self.format_number(self.count_cached_results()))
        self.print_key_value("Unsubmitted papers", self.format_number(self.count_unsubmitted_papers()))
        self.print_key_value("New downloads", self.format_number(new_downloads))

        print("\nBatch status summary")
        print("-" * self.LOG_WIDTH)
        for status in ["completed", "in_progress", "failed"]:
            self.print_key_value(status, self.format_number(manifest_counter.get(status, 0)))

        for status, count in sorted(manifest_counter.items()):
            if status not in {"completed", "in_progress", "failed"}:
                self.print_key_value(status, self.format_number(count))

        print()
        self.print_key_value("Total batches", self.format_number(len(manifest.get("batches", []))))

        if next_check is not None:
            print()
            self.print_key_value("Next check", f"in {next_check} seconds")

    # returns submitted_ids that did not fail
    def get_submitted_ids(self):
        manifest = self.load_manifest()
        submitted_ids = set()

        for batch in manifest.get("batches", []):
            if batch.get("status") in self.VALID_SUBMITTED_STATUSES:
                submitted_ids.update(str(x) for x in batch.get("paper_ids", []))

        return submitted_ids

    # returns all ids that have not been submitted yet
    def get_unsubmitted_ids(self):
        with open(self.corpus_ids_path, "r", encoding="utf-8") as f:
            all_ids = {str(x) for x in json.load(f)}

        return all_ids - self.get_submitted_ids()

    # counts the number of unsubmitted papers
    def count_unsubmitted_papers(self):
        return len(self.get_unsubmitted_ids())

    # counts the number of batches whose status is active
    def count_active_batches(self):
        manifest = self.load_manifest()
        return sum(
            1 for batch in manifest["batches"]
            if batch.get("status") in self.ACTIVE_STATUSES
        )

    # counts the number of elements in the output cache
    def count_cached_results(self):
        if not os.path.exists(self.output_cache_path):
            return 0

        with open(self.output_cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        return len(cache)

    # ---------------------------running methods-------------------------------------------
    # just load all the batches in the manifest and check for status and output_file_id and save again
    def refresh_batch_manifest(self):
        manifest = self.load_manifest()

        for batch_info in manifest["batches"]:
            if batch_info.get("status") not in self.ACTIVE_STATUSES:
                continue
            batch_id = batch_info["batch_id"]

            batch = self.client.batches.retrieve(batch_id)

            batch_info["status"] = batch.status
            batch_info["output_file_id"] = batch.output_file_id
            batch_info["error_file_id"] = batch.error_file_id

        self.save_manifest(manifest)

    def download_completed_batch_outputs(self):
        manifest = self.load_manifest()
        hierarchy_cache = self.load_output_cache()
        new_downloads = 0

        # go through each batch in the manifest
        for batch_info in manifest["batches"]:
            # if status is not completed skip
            if batch_info.get("status") != "completed":
                continue
            # if downloaded is true skip, because don't have to download again
            if batch_info.get("downloaded"):
                continue

            # get output file id
            output_file_id = batch_info.get("output_file_id")

            if not output_file_id:
                continue

            # create local_output_path so one file per batch
            local_output_path = os.path.join(
                self.downloaded_dir,
                f"{batch_info['batch_id']}_output.jsonl"
            )

            # Download only if not already downloaded
            if not os.path.exists(local_output_path):
                content = self.client.files.content(output_file_id)
                content.write_to_file(local_output_path)

            # open the downloaded file
            with open(local_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    # load the json response file
                    result = json.loads(line)

                    paper_id = result["custom_id"]

                    # continue if paper id already exists
                    if paper_id in hierarchy_cache:
                        continue

                    # if there was an error for this paper write down the error message
                    if result.get("error") is not None:
                        hierarchy_cache[paper_id] = {
                            "error": result["error"]
                        }
                        continue

                    # return only the needed json response text
                    response_body = result["response"]["body"]
                    output_text = extract_output_text(response_body)

                    hierarchy_cache[paper_id] = {
                        "hierarchy_info": output_text,
                    }

            self.save_output_cache(hierarchy_cache)
            batch_info["downloaded"] = True
            self.save_manifest(manifest)
            new_downloads += 1

        return new_downloads

    def run_batch_submission(self, verbose=True):
        if verbose:
            self.print_section("OPENAI BATCH SUBMISSION")

        manifest_path = self.manifest_path

        try:
            submitted_batches = submit_hierarchy_information_batches(
                model=self.model,
                corpus_path=self.corpus_path,
                request_batch_size=self.request_batch_size,
                output_dir=self.output_dir,
                manifest_path=self.manifest_path,
                max_batches=1,
                verbose=verbose,
            )

            if verbose:
                total_requests = sum(len(batch["paper_ids"]) for batch in submitted_batches)
                self.print_key_value("New batches", len(submitted_batches))
                self.print_key_value("New paper requests", total_requests)
                self.print_key_value("Manifest", manifest_path)

                if submitted_batches:
                    print("\nSubmitted batches")
                    print("-" * self.LOG_WIDTH)
                    for batch in submitted_batches:
                        print(
                            f"{batch['batch_id']} | "
                            f"status: {batch['status']} | "
                            f"papers: {len(batch['paper_ids'])}"
                        )
                else:
                    print("No new batches submitted.")

            return submitted_batches

        except Exception as e:
            print("\nBatch submission failed.")
            self.print_key_value("Error type", type(e).__name__)
            self.print_key_value("Error message", e)
            print("Check OPENAI_API_KEY, corpus_path, JSONL validity, and batch request objects.")
            raise

    def manager_run(self,run_once=False):
        cycle_number = 1
        while True:
            # 1. Reload the manifest
            self.refresh_batch_manifest()

            # 2. Download completed outputs
            new_downloads = self.download_completed_batch_outputs()

            # 3. Estimate how many batches are active
            active_batches = self.count_active_batches()

            # 4. Submit new batches until active queue is full
            while active_batches < self.max_active_batches:
                submitted_batches = self.run_batch_submission(verbose=False)

                if not submitted_batches:
                    break

                active_batches += len(submitted_batches)

            self.refresh_batch_manifest()

            # 5. Stop condition for one-cycle testing
            if run_once:
                self.clear_console()
                self.print_manager_cycle_summary(cycle_number, new_downloads)
                print()
                self.print_key_value("Reason", "run_once=True")
                break

            # 6. Finished condition
            active_batches = self.count_active_batches()

            if active_batches == 0 and len(self.get_unsubmitted_ids()) == 0:
                self.clear_console()
                self.print_manager_cycle_summary(cycle_number, new_downloads)
                print()
                self.print_key_value("Reason", "all papers submitted and processed")
                break

            self.clear_console()
            self.print_manager_cycle_summary(
                cycle_number,
                new_downloads,
                next_check=self.check_interval_seconds,
            )
            time.sleep(self.check_interval_seconds)
            cycle_number += 1

#--------------------------main program-----------------------------------------------------------------
if __name__ == "__main__":
    model = "gpt-5-nano"
    request_batch_size = 800
    max_active_batches = 2
    check_interval_seconds = 120


    manager = OpenAIBatchManager(
        model="gpt-5-nano",
        request_batch_size=request_batch_size,
        max_active_batches=max_active_batches,
        check_interval_seconds=check_interval_seconds,
    )

    # manager.manager_run(run_once=True)
    # manager.print_manifest_summary()

    # run_cost_estimation(model, 64183-2600, 4.18, 800, 1, True)
    manager.print_manifest_summary()
    print("cached:", manager.count_cached_results())
    print("Unsubmitted:", manager.count_unsubmitted_papers())
