import json
import os
import random
import re
import sys
import time

def ensure_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")

def read_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def already_done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

# --------------------------------- adds info of paper_cache into common schema ------------
def make_paper_metadata_enricher(paper_cache_path):
    ensure_file_exists(paper_cache_path)

    with open(paper_cache_path, "r", encoding="utf-8") as f:
        paper_cache = json.load(f)

    def enrich(doc):
        if doc["source_type"] != "paper":
            return doc

        corpus_id = str(doc["raw_source"].get("native_id"))
        metadata = paper_cache.get(corpus_id)
        if not metadata:
            return doc

        venue = metadata.get("publicationVenue") or {}
        journal = metadata.get("journal") or {}

        doc["entities"]["people"] = [
            author.get("name")
            for author in metadata.get("authors", [])
            if author.get("name")
        ]
        doc["entities"]["organizations"] = sorted({
            affiliation
            for author in metadata.get("authors", [])
            for affiliation in author.get("affiliations", [])
        })
        doc["entities"]["topics"] = metadata.get("fieldsOfStudy") or []

        categorical = doc["structured_fields"]["categorical"]
        categorical.update({
            "venue_name": venue.get("name"),
            "venue_type": venue.get("type"),
            "journal": journal.get("name"),
            "journal_volume": journal.get("volume"),
            "fields_of_study": metadata.get("fieldsOfStudy") or [],
            "publication_types": metadata.get("publicationTypes") or [],
        })

        return doc

    return enrich

# --------------------------------- adds queried hierarchy info into common schema ------------
#Json preprocessing functions to clean invalid Json:
def remove_comments_json_text(text):

    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return text.strip()

def remove_babble(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return text

    return text[start:end + 1]

def repair_fragmented_schema(data):
    if not isinstance(data, list):
        return data

    merged = {}

    for item in data:
        if isinstance(item, list):
            merged["affiliations"] = item
        elif isinstance(item, dict):
            merged.update(item)

    return merged


def make_hierarchy_enricher(hierarchy_cache_path):
    ensure_file_exists(hierarchy_cache_path)

    with open(hierarchy_cache_path, "r", encoding="utf-8") as f:
        hierarchy_cache = json.load(f)

    def enrich(doc):
        if doc["source_type"] != "paper":
            return doc

        hierarchical = doc["structured_fields"].get("hierarchical", {})

        # already successfully enriched
        if hierarchical and "errors" not in hierarchical:
            return doc

        corpus_id = str(doc["raw_source"].get("native_id"))
        cache_entry = hierarchy_cache.get(corpus_id)

        # not available yet, leave untouched
        if cache_entry is None:
            return doc

        # now we know we tried this document
        doc["structured_fields"]["hierarchical"]["errors"] = []

        metadata = cache_entry.get("hierarchy_info")

        # try cleaning approaches
        if not metadata:
            doc["structured_fields"]["hierarchical"]["errors"].append(
                "missing_hierarchy_info"
            )
            return doc

        try:
            parsed_metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            try:
                clean_metadata = remove_comments_json_text(metadata)
                clean_metadata = remove_babble(clean_metadata)

                try:
                    parsed_metadata = json.loads(clean_metadata)

                except json.JSONDecodeError:
                    fragment_list = json.loads("[" + clean_metadata + "]")
                    parsed_metadata = repair_fragmented_schema(fragment_list)

            except (json.JSONDecodeError, TypeError):
                doc["structured_fields"]["hierarchical"]["errors"].append(
                    "JSONDecodeError"
                )
                return doc

        if not isinstance(parsed_metadata, dict):
            doc["structured_fields"]["hierarchical"]["errors"].append(
                "hierarchy_info_not_dict"
            )
            return doc

        doc["structured_fields"]["hierarchical"].pop("errors", None)
        doc["structured_fields"]["hierarchical"].update(parsed_metadata)

        return doc

    return enrich

# --------------------------------- add paper to paper queries ------------
def make_paper_to_paper_query_enricher(random_number=200, seed=42):
    def enrich(docs, qrels):
        paper_docs = [
            doc
            for doc in docs
            if doc["source_type"] == "paper"
            and len(doc["relations"]["explicit_related_ids"]) > 0
        ]

        selected_papers = random.Random(seed).sample(
            paper_docs,
            min(random_number, len(paper_docs)),
        )

        max_id = max(int(doc["id"]) for doc in docs if str(doc["id"]).isdigit())
        next_id = max_id + 1

        new_docs = []
        new_qrels = []

        for doc in selected_papers:
            related_ids = doc["relations"]["explicit_related_ids"]
            title = doc.get("title") or ""
            abstract = doc.get("main_text") or ""
            query_text = f"{title}\n\n{abstract}".strip()

            if not query_text:
                continue

            query_id = str(next_id)
            next_id += 1

            new_docs.append({
                "id": query_id,
                "source_dataset": doc.get("source_dataset", "semantic_scholar"),
                "source_type": "query",
                "title": title,
                "main_text": query_text,
                "secondary_texts": [],
                "structured_fields": doc.get("structured_fields", {
                    "categorical": {},
                    "hierarchical": {},
                }),
                "entities": doc.get("entities", {
                    "people": [],
                    "organizations": [],
                    "projects": [],
                    "topics": [],
                }),
                "relations": {"explicit_related_ids": related_ids},
                "retrieval_metadata": {
                    "is_queryable": True,
                    "is_candidate": False,
                },
                "raw_source": doc.get("raw_source", {
                    "native_id": None,
                    "url": None,
                }),
            })

            new_qrels.append({
                "query_id": query_id,
                "candidate_ids": [str(x) for x in related_ids],
                "relation_type": "citation",
            })

        return docs + new_docs, qrels + new_qrels

    return enrich

# ------------------------define the general enrichment functions------------------------------------

def run_document_enrichment(input_path, output_path, enrichers):
    ensure_file_exists(input_path)
    if len(enrichers)==0:
        raise ValueError("Cannot enrich when nothing given.")

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            doc = json.loads(line)

            for enricher in enrichers:
                doc = enricher(doc)

            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")

def run_dataset_addition(documents_input_path,
                         documents_output_path,
                         qrels_input_path,
                         qrels_output_path,
                         enrichers):
    ensure_file_exists(documents_input_path)
    ensure_file_exists(qrels_input_path)
    if len(enrichers)==0:
        raise ValueError("Cannot enrich when nothing given.")

    docs = read_jsonl(documents_input_path)
    qrels = read_jsonl(qrels_input_path)

    for enricher in enrichers:
        docs, qrels = enricher(docs, qrels)

    write_jsonl(documents_output_path, docs)
    write_jsonl(qrels_output_path, qrels)

def run_metadata_enrichment(force_overwrite=False):
    output_path = "output/documents_enriched_01.jsonl"

    if already_done(output_path) and not force_overwrite:
        print(f"Skipping metadata enrichment. Already exists: {output_path}")
        print("Use --overwrite-metadata to rebuild.\n")
        return

    if force_overwrite:
        print("\nForce rebuilding metadata enrichment...")
    else:
        print("\nRunning metadata enrichment...")
    run_document_enrichment(
        "output/documents.jsonl",
        output_path,
        [make_paper_metadata_enricher("cache/paper_cache.json")],
    )
    print(f"Done. Wrote {output_path}")

def run_paper_to_paper_enrichment(force_overwrite=False):
    documents_output_path = "output/documents_enriched_02.jsonl"
    qrels_output_path = "output/qrels_enriched_02.jsonl"

    if already_done(documents_output_path) and already_done(qrels_output_path) and not force_overwrite:
        print(f"Skipping paper_to_paper query enrichment. "
              f"Already exists: {documents_output_path}, {qrels_output_path}")
        print("Use --readd-paper-to-paper to rebuild.\n")
        return

    if force_overwrite:
        print("\nForce running paper_to_paper enrichment...")
    else:
        print("\nRunning paper_to_paper enrichment...")
    run_dataset_addition(
        "output/documents_enriched_01.jsonl",
        documents_output_path,
        "output/qrels.jsonl",
        qrels_output_path,
        [make_paper_to_paper_query_enricher(200, 42)],
    )
    print(f"Done. Wrote {documents_output_path} and {qrels_output_path}")

def run_hierarchy_enrichment(force_overwrite=False):
    output_path = "output/documents_enriched_03.jsonl"

    if already_done(output_path) and not force_overwrite:
        print(f"Skipping hierarchy data enrichment. Already exists: {output_path}")
        print("Use --overwrite-hierarchy to rebuild.\n")
        return

    if force_overwrite:
        print("\nForce rebuilding hierarchy enrichment...")
    else:
        print("\nRunning hierarchy data enrichment...")
    run_document_enrichment(
        "output/documents_enriched_02.jsonl",
        output_path,
        [make_hierarchy_enricher("cache/hierarchy_cache_nano_batch.json")],
    )
    print(f"Done. Wrote {output_path}")

if __name__ == "__main__":
    force_overwrite_metadata = "--overwrite-metadata" in sys.argv
    force_readd_paper = "--readd-paper-to-paper" in sys.argv
    force_overwrite_hierarchy = "--overwrite-hierarchy" in sys.argv


    run_metadata_enrichment(force_overwrite_metadata)
    run_paper_to_paper_enrichment(force_readd_paper)
    run_hierarchy_enrichment(force_overwrite_hierarchy)