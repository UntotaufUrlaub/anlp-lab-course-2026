import pandas as pd
import requests
import time
import json
from datasets import load_dataset
import os
from dotenv import load_dotenv
import random
from openai import OpenAI
import tiktoken

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


#--------------------------calls the OpenAI Responses API to gather hierarchy information--------------------------
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

GENERAL_RESEARCH_AREA = """["Machine Learning","Artificial Intelligence","Natural Language Processing","Computer Vision",
    "Information Retrieval","Knowledge Representation and Reasoning","Graph Learning","Robotics and Control",
    "Human-Computer Interaction","Security and Privacy","Software Engineering","Distributed Systems","Data Mining",
    "Scientific Machine Learning","Computational Biology","Medical AI","Signal Processing","Speech and Audio",
    "Recommendation Systems","Optimization","Theoretical Machine Learning","Quantum Computing","Multimodal AI",
    "Generative AI"]"""

METHOD_FAMILY = """["Transformer Models","Graph Neural Networks","Convolutional Neural Networks",
    "Recurrent Neural Networks","Diffusion Models","Variational Autoencoders","Generative Adversarial Networks",
    "Representation Learning","Contrastive Learning","Self-Supervised Learning","Retrieval Methods",
    "Ranking Methods","Reinforcement Learning","Imitation Learning","Probabilistic Models","Bayesian Methods",
    "Optimization Methods","Knowledge-Based Methods","Neuro-Symbolic Methods","Ensemble Methods",
    "Foundation Models","Multimodal Models"]"""

BROAD_DOMAIN = """["Computer Science","Mathematics","Medicine","Engineering","Biology", "Physics","Economics",
    "Psychology","Political Science","Art","History","Philosophy","Chemistry","Materials Science","Sociology",
    "Geography","Business","Environmental Science","Geology"]"""

FIELDS = """["Artificial Intelligence","Natural Language Processing","Computer Vision","Information Retrieval",
    "Graph Learning","Robotics","Security and Privacy","Software Engineering","Systems","Optimization",
    "Probability and Statistics","Applied Mathematics","Theoretical Machine Learning","Robotics and Control",
    "Signal Processing","Hardware and Systems","Medical AI","Neuroscience","Computational Biology",
    "Scientific Machine Learning","Quantum Computing","Computational Chemistry","Decision Systems",
    "Algorithmic Economics","Cognitive Modeling"]"""

JSON_SCHEMA = """{
  "hierarchical": {
    "affiliations": [
        {"country", "sector", "organization"},
        ...
    ],
    "field_of_study_path": {
      "broad_domain":,
      "field":,
      "subfield":,
    },
    "topic_path": {
      "general_research_area":,
      "topic_family":,
      "specific_topic":,
    },
    "method_path": {
      "method_family":,
      "method_category":,
      "specific_method":,
    }
  }
}"""

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
        fields_of_study=(paper.get("structured_fields").get("multi_label").get("fields_of_study"))

        user_prompt = (f"Fill this schema {JSON_SCHEMA} with following paper metadata:Title:"
                       f"{title}"
                       f"Abstract: {abstract}"
                        f"Affiliations: {affiliation}"
                        f"Fields of Study: {fields_of_study}"
                        f"Guidelines:"
                        f"- Fields of Study are strong evidence for broad_domain."
                        f"- Use affiliations exactly as organization names."
                        f"- Create one affiliation entry per affiliation string."
                        f"- Choose the primary topic and primary method."
                        f"- Do not invent categories outside the allowed lists."
                        f"- Keep topic_family, specific_topic, method_category, and specific_method short: "
                        f"maximum 6 words each."
                        f"- Prefer reusable labels over paper-specific descriptions."
                        f"-Hierarchy rule:"
                        f"Each level must be a more specific subcategory of the previous level."
                        f"Do not repeat the same category at two levels."
                        f"Do not put a broader category after a narrower category.")

        system_prompt =f"""
                        You are a scientific metadata extraction system.
                        Use null if unknown.
                        Use the closest valid category when uncertain.
                        
                        Allowed sectors: {SECTORS}
                        Allowed broad_domain: {BROAD_DOMAIN}
                        Allowed fields: {FIELDS}
                        Allowed general_research_area: {GENERAL_RESEARCH_AREA}
                        Allowed method_family: {METHOD_FAMILY}"""

        output_example = """{
                  "hierarchical": {
                    "affiliations": [
                      {"country": "United States", "sector": "Independent", "organization": "University of Michigan"},
                      {"country": "Uruguay", "sector": "Independent", "organization": "Universidad de la República"}
                    ],
                    "field_of_study_path": {
                      "broad_domain": "Computer Science",
                      "field": "Artificial Intelligence",
                      "subfield": "Machine Learning"
                    },
                    "topic_path": {
                      "general_research_area": "Machine Learning",
                      "topic_family": "Quantization",
                      "specific_topic": "Learned quantization for indexing"
                    },
                    "method_path": {
                      "method_family": "Foundation Models",
                      "method_category": "Self-Supervised Learning",
                      "specific_method": "Uniform spherical quantizer training"
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

#--------------------------main program-----------------------------------------------------------------
if __name__ == "__main__":
    # choose the batch size with wich semantic scholar responds
    # batch size has to be < 500 as to not overload the api
    # semantic_scholar_batch_size = 250
    # run_semantic_scholar_api_call_program(semantic_scholar_batch_size)
    model = "gpt-5-nano"
    print("Calculating average input and output tokens ...")
    token_info, token_report = estimate_avg_input_output_tokens(model, 1000)
    print("\nCalculating costs...")
    _, cost_report = api_cost_estimator(model, 64183,
                                        token_info["avg_input_tokens"], token_info["avg_output_tokens"])
    print(token_report)
    print("\n")
    print(cost_report)