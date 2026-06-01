import requests
import time
import json
from datasets import load_dataset
import os
from dotenv import load_dotenv
import random


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
            raise Exception(f"Request failed:{r.status_code}: {r.text}")  # can also change to return None instead


# extracts all corpus_ids and puts them in separate cache file for calling later, has to be run first
def get_list_of_corpus_ids():
    corpus_ids = []
    corpus_clean_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full", streaming=True)

    for i, paper in enumerate(corpus_clean_data):
        corpus_ids.append(paper["corpusid"])
        if i % 10000 == 0:
            print(f"{i}")

    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "corpus_ids.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)


# --------------------------call the api which can respond in batches--------------------------
load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # get api key


# does exponential backoff
def post_with_backoff(url, *, params, json_payload, headers, max_retries=6):
    delay = 2

    for attempt in range(max_retries):
        response = requests.post(
            url,
            params=params,
            json=json_payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code in {429, 500, 502, 503, 504}:
            wait = delay + random.uniform(0, 0.5)
            print(f"Retry {attempt + 1}: status {response.status_code}, waiting {wait:.1f}s")
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

    payload = {
        "ids": [f"CorpusId:{cid}" for cid in paper_ids]
    }

    headers = {
        "x-api-key": API_KEY
    }

    return post_with_backoff(
        url,
        params=params,
        json_payload=payload,
        headers=headers
    )


# out of the whole corpus_ids file create queryable chunks
def chunks(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_json_if_exists(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# extracts data with a cache, so if something crashes can continue from previous endpoint
def data_extraction(corpus_ids_file, batch_size=10000):
    with open(corpus_ids_file, "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)

    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, "paper_cache.json")
    processed_path = os.path.join(output_dir, "missing_processed_ids.json")

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
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                batch
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


# ----------------------- add correction logic for missing entries -------
def detect_missing_ids():
    missing_ids = []
    with open("cache/corpus_ids.json", "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open("cache/paper_cache.json", "r", encoding="utf-8") as f:
        cache = json.load(f)

    keys = cache.keys()

    for id in corpus_ids:
        if str(id) not in keys:
            missing_ids.append(id)
    print(len(missing_ids))

    with open("cache/missing_corpus_ids.json", "w", encoding="utf-8") as f:
        json.dump(missing_ids, f)


# --------------------------------- adds info of paper_cache into common schema ------------
def ensure_paper_cache_exists(cache_path="cache/paper_cache.json"):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} does not exist. Put paper_cache into the cache folder."
        )


def enrich_document_with_paper_info():
    ensure_paper_cache_exists()
    i = 0
    with open("cache/paper_cache.json", "r", encoding="utf-8") as f:
        paper_cache = json.load(f)

    with open("output/documents.jsonl", "r", encoding="utf-8") as infile, \
            open("output/documents_enriched_01.jsonl", "w", encoding="utf-8") as outfile:

        for line in infile:
            doc = json.loads(line)

            if (
                    doc["source_type"] == "paper"
                    and doc["raw_source"]["native_id"] is not None
            ):

                corpus_id = str(doc["raw_source"]["native_id"])

                if corpus_id in paper_cache.keys():
                    i += 1
                    metadata = paper_cache[corpus_id]

                    venue = metadata.get("publicationVenue") or {}
                    journal = metadata.get("journal") or {}

                    doc["entities"]["people"] = [
                        author.get("name")
                        for author in metadata.get("authors", [])
                        if author.get("name")
                    ]

                    doc["entities"]["organizations"] = list({
                        affiliation
                        for author in metadata.get("authors", [])
                        for affiliation in author.get("affiliations", [])
                    })

                    doc["entities"]["topics"] = metadata.get("fieldsOfStudy") or []

                    doc["structured_fields"]["categorical"]["venue_name"] = venue.get("name")
                    doc["structured_fields"]["categorical"]["venue_type"] = venue.get("type")
                    doc["structured_fields"]["categorical"]["journal"] = journal.get("name")
                    doc["structured_fields"]["categorical"]["journal_volume"] = journal.get("volume")
                    doc["structured_fields"]["multi_label"]["fields_of_study"] = metadata.get("fieldsOfStudy") or []
                    doc["structured_fields"]["multi_label"]["publication_types"] = metadata.get(
                        "publicationTypes") or []

            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"modified_entries: {i}")


# --------------------------------- add paper to paper queries ------------
def add_paper_to_paper_queries(
        documents_input_path="output/documents_enriched_01.jsonl",
        documents_output_path="output/documents_enriched_02.jsonl",
        qrels_input_path="output/qrels.jsonl",
        qrels_output_path="output/qrels_01.jsonl",
        random_number=200
):
    """
    Adds paper to paper query records.

    For every randomly selected paper with explicit_related_ids, create a new query document:
    - query text = title + abstract
    - gold labels = related paper IDs
    - one qrel row per related paper
    """

    docs = []

    with open(documents_input_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    # randomly select a random_amount of paper_docs
    paper_docs = [
        doc for doc in docs
        if doc["source_type"] == "paper"
           and len(doc["relations"]["explicit_related_ids"]) > 0
    ]

    selected_papers = random.Random(42).sample(
        paper_docs,
        min(random_number, len(paper_docs))
    )
    # find the maximum corpus_id
    max_id = max(int(doc["id"]) for doc in docs if str(doc["id"]).isdigit())  #
    next_id = max_id + 1

    new_query_docs = []
    new_qrels = []

    for doc in selected_papers:

        related_ids = doc.get("relations", {}).get("explicit_related_ids", [])

        if not related_ids:
            continue

        title = doc.get("title") or ""
        abstract = doc.get("main_text") or ""

        query_text = f"{title}\n\n{abstract}".strip()

        if not query_text:
            continue

        query_id = str(next_id)
        next_id += 1

        query_doc = {
            "id": query_id,
            "source_dataset": doc.get("source_dataset", "semantic_scholar"),
            "source_type": "query",
            "title": title,
            "main_text": query_text,
            "secondary_texts": [],
            "structured_fields": doc.get("structured_fields", {
                "categorical": {},
                "multi_label": {},
                "hierarchical": {}
            }),
            "entities": doc.get("entities", {
                "people": [],
                "organizations": [],
                "projects": [],
                "topics": []
            }),
            "relations": {
                "explicit_related_ids": related_ids
            },
            "retrieval_metadata": {
                "is_queryable": True,
                "is_candidate": False
            },
            "raw_source": {
                "native_id": doc.get("raw_source", {}).get("native_id"),
                "url": doc.get("raw_source", {}).get("url")
            }
        }

        new_query_docs.append(query_doc)
        new_qrels.append({
            "query_id": query_id,
            "candidate_id": [str(x) for x in related_ids],
            "relation_type": "citation"
        })

    with open(documents_output_path, "w", encoding="utf-8") as outfile:
        for doc in docs + new_query_docs:
            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(qrels_output_path, "w", encoding="utf-8") as outfile:
        if os.path.exists(qrels_input_path):
            with open(qrels_input_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

        for qrel in new_qrels:
            outfile.write(json.dumps(qrel, ensure_ascii=False) + "\n")

    print(f"Eligible paper docs: {len(paper_docs)}")
    print(f"Selected paper docs: {len(selected_papers)}")
    print(f"Added {len(new_query_docs)} paper-to-paper query docs")
    print(f"Added {len(new_qrels)} paper-to-paper qrels")
    print(f"Wrote documents to {documents_output_path}")
    print(f"Wrote qrels to {qrels_output_path}")


if __name__ == "__main__":
    # enrich_document_with_paper_info()
    add_paper_to_paper_queries()
