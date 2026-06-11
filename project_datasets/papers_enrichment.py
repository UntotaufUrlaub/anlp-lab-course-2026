import json
import os
import random

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
def make_hierarchy_enricher(hierarchy_cache_path):
    ensure_file_exists(hierarchy_cache_path)

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

    docs = read_jsonl(documents_input_path)
    qrels = read_jsonl(qrels_input_path)

    for enricher in enrichers:
        docs, qrels = enricher(docs, qrels)

    write_jsonl(documents_output_path, docs)
    write_jsonl(qrels_output_path, qrels)


if __name__ == "__main__":
    info_enrichers = [make_paper_metadata_enricher("cache/paper_cache.json")]
    dataset_enrichers=[make_paper_to_paper_query_enricher(200,42)]

    run_document_enrichment("output/documents.jsonl", "output/documents_enriched.jsonl",
                            info_enrichers)
    run_dataset_addition("output/documents_enriched.jsonl",
                         "output/documents_enriched_with_paper_to_paper.jsonl",
                         "output/qrels.jsonl",
                         "output/qrels_enriched_with_paper_to_paper.jsonl",
                         dataset_enrichers)