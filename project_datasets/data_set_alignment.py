"""
IR Benchmark preprocessing pipeline.
Transforms GitHub issues and papers dataframes into the canonical schema.
Output: documents.jsonl, qrels.jsonl
"""

import json
import pandas as pd
from pathlib import Path


# ── ID helpers ────────────────────────────────────────────────────────────────

def github_id(repository, issue_no):
    return f"github:{repository}:issue:{issue_no}"


def paper_id(corpusid):
    return f"paper:{corpusid}"


def query_id(query_set, idx):
    return f"query:{query_set}:{str(idx).zfill(4)}"


# ── Core document builder ─────────────────────────────────────────────────────

def make_doc(id, source_dataset, source_type, title, main_text,
             secondary_texts=None, created_at=None, closed_at=None,
             categorical=None, multi_label=None, hierarchical=None,
             people=None, organizations=None, projects=None, topics=None,
             explicit_related_ids=None, split="unlabeled",
             is_queryable=True, is_candidate=True,
             native_id=None, url=None):
    return {
        "id": id,
        "source_dataset": source_dataset,
        "source_type": source_type,
        "title": title,
        "main_text": main_text,
        "secondary_texts": secondary_texts or [],
        "timestamps": {"created_at": created_at, "closed_at": closed_at},
        "structured_fields": {
            "categorical": categorical or {},
            "multi_label": multi_label or {},
            "hierarchical": hierarchical or {},
        },
        "entities": {
            "people": people or [],
            "organizations": organizations or [],
            "projects": projects or [],
            "topics": topics or [],
        },
        "relations": {"explicit_related_ids": explicit_related_ids or []},
        "retrieval_metadata": {
            "split": split,
            "is_queryable": is_queryable,
            "is_candidate": is_candidate,
        },
        "raw_source": {"native_id": native_id, "url": url},
    }


def make_qrel(query_id, candidate_id, relation_type, source="explicit",
              relevance=3, split="unlabeled"):
    return {
        "query_id": query_id,
        "candidate_id": candidate_id,
        "relevance": relevance,
        "relation_type": relation_type,
        "source": source,
        "split": split,
    }


# ── GitHub ────────────────────────────────────────────────────────────────────

def process_github(issues_df, linked_df, split="unlabeled"):
    # Build relation lookup: issue_no -> [related_issue_nos]
    # modified because linked_df already has lists
    relations = linked_df.set_index("issue_no")["related_issue_nos"].to_dict()

    docs, qrels = [], []
    for _, r in issues_df.iterrows():
        repo = str(r["repository"])
        no = str(r["issue_no"])
        canon_id = github_id(repo, no)

        # code should work now
        related = [github_id(repo, str(x)) for x in json.loads(str(relations.get(int(no), [])))]

        docs.append(make_doc(
            id=canon_id,
            source_dataset=repo,
            source_type="github_issue",
            title=r.get("issue_title"),
            main_text=r.get("issue_body") or "",
            secondary_texts=r["comments"] if isinstance(r.get("comments"), list) else [],
            created_at=_iso(r.get("created_at")),
            closed_at=_iso(r.get("closed_at")),
            categorical={"status": "closed" if pd.notna(r.get("closed_at")) else "open"},
            multi_label={"labels": r["labels"] if isinstance(r.get("labels"), list) else []},
            projects=[repo],
            explicit_related_ids=related,
            split=split,
            native_id=no,
            url=r.get("issue_url"),
        ))
        for cid in related:
            qrels.append(make_qrel(canon_id, cid, "linked_issue", split=split))

    return docs, qrels


# ── Papers ────────────────────────────────────────────────────────────────────

def process_papers(queries_df, corpus_df, split="unlabeled"):
    docs, qrels = [], []

    # Corpus
    for _, r in corpus_df.iterrows():
        cid = str(r["corpusid"])
        cited = [paper_id(str(x)) for x in (r["citations"] if isinstance(r.get("citations"), list) else [])]
        full = r.get("full_paper")
        docs.append(make_doc(
            id=paper_id(cid),
            source_dataset="semantic_scholar",
            source_type="paper",
            title=r.get("title"),
            main_text=r.get("abstract") or "",
            secondary_texts=[full] if isinstance(full, str) and full else [],
            explicit_related_ids=cited,
            split=split,
            is_queryable=False,
            native_id=cid,
        ))

    # Queries
    for query_set, group in queries_df.groupby("query_set"):
        for idx, (_, r) in enumerate(group.iterrows()):
            qid = query_id(query_set, idx)
            gold = [paper_id(str(x)) for x in r["corpusids"].tolist()]
            docs.append(make_doc(
                id=qid,
                source_dataset=str(query_set),
                source_type="query",
                title=None,
                main_text=r.get("query") or "",
                categorical={
                    "specificity": str(r["specificity"]) if pd.notna(r.get("specificity")) else "",
                    "quality": str(r["quality"]) if pd.notna(r.get("quality")) else "",
                },
                explicit_related_ids=gold,
                split=split,
                is_candidate=False,
            ))
            for cid in gold:
                qrels.append(make_qrel(qid, cid, "corpusid_match", split=split))

    return docs, qrels


# ── IO ────────────────────────────────────────────────────────────────────────

def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records):>6} records → {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.isoformat() if isinstance(value, pd.Timestamp) else str(value)


# ── Main ──────────────────────────────────────────────────────────────────────

from datasets import load_dataset
import kagglehub
from json_processing import load_json_dataframes, construct_issue_links

if __name__ == "__main__":
    path = kagglehub.dataset_download("zakareaalshara/android-closed-issues-20110101-20210101-clean")
    (issues, prs, links) = load_json_dataframes(path)
    issue_links = construct_issue_links(issues, links)

    issues_df = issues
    linked_df = issue_links
    queries_df = load_dataset("princeton-nlp/LitSearch", "query", split="full").to_pandas()
    corpus_df = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full", streaming=True).to_pandas()

    gh_docs, gh_qrels = process_github(issues_df, linked_df, split="unlabeled")
    pap_docs, pap_qrels = process_papers(queries_df, corpus_df, split="unlabeled")

    write_jsonl(gh_docs + pap_docs, "output/documents.jsonl")
    write_jsonl(gh_qrels + pap_qrels, "output/qrels.jsonl")
