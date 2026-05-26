"""
IR Benchmark preprocessing pipeline.
Transforms GitHub issues and papers dataframes into the canonical schema.
Output: documents.jsonl, qrels.jsonl
"""

import json
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

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

def process_github(issues_df, linked_df, id_counter, gh_id_map, split="unlabeled"):
    # Build relation lookup: issue_no -> [related_issue_nos]
    # modified because linked_df already has lists
    relations = linked_df.set_index("issue_no")["related_issue_nos"].to_dict()

    docs, qrels = [], []
    for _, r in issues_df.iterrows():
        repo = str(r["repository"])
        issue_no = int(r["issue_no"])
        id = str(id_counter["count"])
        id_counter["count"] += 1
        gh_id_map[issue_no] = id  # Map original issue_no to global ID

        # code should work now
        related_issues = json.loads(str(relations.get(issue_no, [])))
        # Map original issue numbers to global IDs (they may not exist yet, keep as-is for now)
        related = [gh_id_map.get(int(x), str(x)) for x in related_issues]

        # Parse pipe-separated labels
        label_str = r.get("labels") if isinstance(r.get("labels"), str) else ""
        multi_label_data, hierarchical_data = parse_pipe_separated_labels(label_str)

        docs.append(make_doc(
            id=id,
            source_dataset=repo,
            source_type="github_issue",
            title=r.get("issue_title"),
            main_text=r.get("issue_body") or "",
            secondary_texts=r["comments"] if isinstance(r.get("comments"), list) else [],
            categorical={"status": "closed" if pd.notna(r.get("closed_at")) else "open"},
            multi_label=multi_label_data,
            hierarchical=hierarchical_data,
            projects=[repo],
            explicit_related_ids=related,
            split=split,
            native_id=str(issue_no),
            url=r.get("issue_url"),
        ))
        for cid in related:
            qrels.append(make_qrel(id, cid, "linked_issue", split=split))

    return docs, qrels


# ── Papers ────────────────────────────────────────────────────────────────────

def process_papers(queries_df, corpus_df, id_counter, paper_id_map, split="unlabeled"):
    docs, qrels = [], []

    # Corpus
    for _, r in corpus_df.iterrows():
        corpusid = str(r["corpusid"])
        id = str(id_counter["count"])
        id_counter["count"] += 1
        paper_id_map[corpusid] = id  # Map original corpusid to global ID
        
        cited = [str(x) for x in (r["citations"] if isinstance(r.get("citations"), list) else [])]
        full = r.get("full_paper")
        docs.append(make_doc(
            id=id,
            source_dataset="semantic_scholar",
            source_type="paper",
            title=r.get("title"),
            main_text=r.get("abstract") or "",
            secondary_texts=[full] if isinstance(full, str) and full else [],
            explicit_related_ids=cited,
            split=split,
            is_queryable=False,
            native_id=corpusid,
        ))

    # Queries
    for query_set, group in queries_df.groupby("query_set"):
        for idx, (_, r) in enumerate(group.iterrows()):
            qid = str(id_counter["count"])
            id_counter["count"] += 1
            gold = [str(x) for x in r["corpusids"].tolist()]
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

def parse_pipe_separated_labels(label_str):
    """
    Parse pipe-separated canonical label string into categorical and hierarchical components.
    e.g., 'bug|priority:high|layer:frontend' →
        multi_label: {"labels": ["bug|priority:high|layer:frontend"]},
        hierarchical: {
            "all_labels": ["bug", "priority:high", "layer:frontend"],
            "categories": ["bug"],
            "priority": "high",
            "layer": "frontend"
        }
    """
    if not isinstance(label_str, str) or not label_str.strip():
        return {}, {}
    
    labels = label_str.split("|")
    categories = []
    hierarchical = {"all_labels": labels}
    
    for label in labels:
        if ":" in label:
            key, value = label.split(":", 1)
            hierarchical[key.strip()] = value.strip()
        else:
            categories.append(label.strip())
    
    if categories:
        hierarchical["categories"] = categories
    
    return {"labels": [label_str]}, hierarchical


def load_or_cache_datasets(use_cache=False, cache_dir="cache"):
    """
    Load datasets from cache (parquet) if available, otherwise download and cache.
    
    Args:
        use_cache: If True, try to load from cache first
        cache_dir: Directory to store cached parquet files
    
    Returns:
        Tuple of (issues_df, issue_links, queries_df, corpus_df)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    issues_cache = cache_path / "issues.parquet"
    linked_cache = cache_path / "issue_links.parquet"
    queries_cache = cache_path / "queries.parquet"
    corpus_cache = cache_path / "corpus.parquet"
    
    # Try loading from cache if requested and all files exist
    if use_cache and all([f.exists() for f in [issues_cache, linked_cache, queries_cache, corpus_cache]]):
        print("\n📦 Loading from cache...")
        issues = pd.read_parquet(issues_cache)
        issue_links = pd.read_parquet(linked_cache)
        queries_df = pd.read_parquet(queries_cache)
        corpus_df = pd.read_parquet(corpus_cache)
        print(f"✓ Loaded {len(issues)} issues, {len(issue_links)} links, {len(queries_df)} queries, {len(corpus_df)} papers")
        return issues, issue_links, queries_df, corpus_df
    
    # Download datasets with progress indication
    print("\n📥 Downloading datasets...")
    
    # GitHub issues
    print("  [1/4] Fetching GitHub issues...")
    path = kagglehub.dataset_download("zakareaalshara/android-closed-issues-20110101-20210101-clean")
    (issues, prs, links) = load_json_dataframes(path)
    print(f"      ✓ Loaded {len(issues)} issues")
    
    print("  [2/4] Normalizing labels...")
    issues = normalise_labels(issues)
    issue_links = construct_issue_links(issues, links)
    print(f"      ✓ Constructed {len(issue_links)} links")
    
    print("  [3/4] Fetching papers (queries)...")
    queries_df = load_dataset("princeton-nlp/LitSearch", "query", split="full").to_pandas()
    print(f"      ✓ Loaded {len(queries_df)} queries")
    
    print("  [4/4] Fetching papers (corpus)...")
    corpus_df = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full", streaming=True).to_pandas()
    print(f"      ✓ Loaded {len(corpus_df)} papers")
    
    # Cache for next run
    print("\n💾 Caching datasets to parquet...")
    issues.to_parquet(issues_cache, compression="snappy")
    issue_links.to_parquet(linked_cache, compression="snappy")
    queries_df.to_parquet(queries_cache, compression="snappy")
    corpus_df.to_parquet(corpus_cache, compression="snappy")
    print("✓ Cache saved. Use --use-cache flag next run for faster loading")
    
    return issues, issue_links, queries_df, corpus_df

# ── Main ──────────────────────────────────────────────────────────────────────

from datasets import load_dataset
import kagglehub
from json_processing import load_json_dataframes, construct_issue_links, normalise_labels

if __name__ == "__main__":
    # Check for --use-cache flag
    use_cache = "--use-cache" in sys.argv
    
    # Load datasets with caching
    issues_df, linked_df, queries_df, corpus_df = load_or_cache_datasets(use_cache=use_cache)

    # Initialize global counter and ID mappings
    id_counter = {"count": 0}
    gh_id_map = {}  # Maps original GitHub issue_no -> global ID
    paper_id_map = {}  # Maps original paper corpusid -> global ID

    print("\n🔄 Processing GitHub issues...")
    gh_docs, gh_qrels = process_github(issues_df, linked_df, id_counter, gh_id_map, split="unlabeled")
    print(f"✓ Processed {len(gh_docs)} issues and {len(gh_qrels)} qrels")
    
    print("\n🔄 Processing papers...")
    pap_docs, pap_qrels = process_papers(queries_df, corpus_df, id_counter, paper_id_map, split="unlabeled")
    print(f"✓ Processed {len(pap_docs)} papers/queries and {len(pap_qrels)} qrels")

    print("\n📝 Writing output files...")
    write_jsonl(gh_docs + pap_docs, "output/documents.jsonl")
    write_jsonl(gh_qrels + pap_qrels, "output/qrels.jsonl")
    print("\n✅ Done!")
