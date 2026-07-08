"""
Baseline Benchmarking Script for Semi-Structured Information Retrieval

This script evaluates two baseline methods:
1. BM25 (lexical/sparse retrieval)
2. Sentence Transformers (dense embeddings)

Metrics computed:
- NDCG@k (Normalized Discounted Cumulative Gain)
- MAP@k (Mean Average Precision)
- Recall@k
- Precision@k
- MRR (Mean Reciprocal Rank)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging
import random

try:
    from methods import (
        BM25Baseline,
        DenseEmbeddingBaseline,
        CHARMInspiredMethod,
        SAGEGraphExpansionMethod,
        GNNRet,
        NovelGATMethod,
    )
    from utils import parse_args, _flatten_hierarchical
    from benchmarkRunner import BenchmarkRunner
except ImportError:
    from project_code.methods import (
        BM25Baseline,
        DenseEmbeddingBaseline,
        GNNRet,
        NovelGATMethod,
        CHARMInspiredMethod,
        SAGEGraphExpansionMethod,
    )
    from project_code.utils import parse_args, _flatten_hierarchical
    from project_code.benchmarkRunner import BenchmarkRunner

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    # level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def has_structured_fields(doc: Dict) -> bool:
    """Return True if doc has any non-empty structured_fields content."""
    source_type = doc.get("source_type")
    if source_type == "query":
        return True  # never filter queries

    structured = doc.get("structured_fields", {}) or {}
    categorical = structured.get("categorical", {}) or {}
    hierarchical = structured.get("hierarchical", []) or []

    has_categorical = any(
        (v if not isinstance(v, (list, dict)) else any(v))
        for v in categorical.values()
        if v is not None
    )
    has_hierarchical = bool(_flatten_hierarchical(hierarchical))
    return has_categorical or has_hierarchical


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_documents_and_qrels(
    docs_path: str, qrels_path: str, batch_size: int = None, random_seed: int = None
) -> Tuple[Dict, Dict[str, Set[int]]]:
    """
    Load documents and qrels, organize by dataset.

    Args:
    - docs_path: Path to documents JSONL file
    - qrels_path: Path to qrels JSONL file
    - batch_size: If set, randomly sample this many queries (default: None = use all)
    - random_seed: Random seed for reproducibility

    Returns:
    - documents: Dict[doc_id -> doc]
    - qrels: Dict[query_id -> Set[candidate_id]]
    """
    logger.info(f"Loading documents from {docs_path}")
    docs_list = load_jsonl(docs_path)
    documents = {doc["id"]: doc for doc in docs_list}
    logger.info(f"Loaded {len(documents)} documents")

    # Drop non-query docs with empty structured fields
    before = len(documents)
    documents = {
        did: doc for did, doc in documents.items() if has_structured_fields(doc)
    }
    logger.info(
        f"Dropped {before - len(documents)} docs with empty structured_fields ({len(documents)} remaining)"
    )

    logger.info(f"Loading qrels from {qrels_path}")
    qrels_list = load_jsonl(qrels_path)
    qrels = defaultdict(set)

    for qrel in qrels_list:
        query_id = qrel["query_id"]
        if "candidate_ids" in qrel:
            qrels[query_id].update(qrel["candidate_ids"])
        elif "candidate_id" in qrel:
            qrels[query_id].add(qrel["candidate_id"])
        else:
            raise ValueError(
                "Unsupported qrel format: expected 'candidate_ids' or 'candidate_id'."
            )

    qrels = {qid: set(cids) for qid, cids in qrels.items()}

    # Remove qrels that reference missing documents.
    if documents:
        available_doc_ids = set(documents.keys())
        filtered_qrels = {}
        missing_queries = 0
        missing_candidates = 0

        for qid, cids in qrels.items():
            if qid not in available_doc_ids:
                missing_queries += 1
                continue

            filtered_candidates = {cid for cid in cids if cid in available_doc_ids}
            if filtered_candidates:
                filtered_qrels[qid] = filtered_candidates
            else:
                missing_candidates += 1

        if missing_queries or missing_candidates:
            logger.warning(
                f"Filtered qrels: {missing_queries} queries missing from documents, "
                f"{missing_candidates} queries had no valid candidates"
            )
        qrels = filtered_qrels

    logger.info(f"Loaded {len(qrels)} queries with hard ground truth candidates")

    # Sample batch if batch_size is specified
    if batch_size is not None and batch_size > 0:
        if random_seed:
            random.seed(random_seed)
        query_ids = list(qrels.keys())
        batch_size = min(batch_size, len(query_ids))
        sampled_query_ids = random.sample(query_ids, batch_size)
        qrels = {qid: qrels[qid] for qid in sampled_query_ids}
        logger.info(
            f"Sampled {len(qrels)} queries (batch_size={batch_size}, seed={random_seed})"
        )

    return documents, qrels


def sample_documents(
    documents: Dict, qrels: Dict[str, Set[int]], sample_size: int, random_seed: int = 42
) -> Tuple[Dict, Dict[str, Set[int]]]:
    """Sample documents while preserving at least some query/qrel pairs."""
    if sample_size is None or sample_size <= 0:
        return documents, qrels
    if random_seed:
        random.seed(random_seed)
    doc_ids = list(documents.keys())
    available_doc_ids = set(doc_ids)
    sample_size = min(sample_size, len(doc_ids))

    if not qrels:
        sampled_doc_ids = set(random.sample(doc_ids, sample_size))
        sampled_documents = {doc_id: documents[doc_id] for doc_id in sampled_doc_ids}
        logger.info(f"Sampled {len(sampled_documents)} documents (no qrels available)")
        return sampled_documents, qrels

    # Sample by query groups so qrels stay meaningful.
    query_ids = [qid for qid in qrels.keys() if qid in available_doc_ids]
    random.shuffle(query_ids)

    sampled_doc_ids = set()
    sampled_qrels = {}

    for query_id in query_ids:
        candidate_ids = {cid for cid in qrels[query_id] if cid in available_doc_ids}
        if not candidate_ids:
            continue

        group_ids = {query_id} | candidate_ids

        if len(sampled_doc_ids) + len(group_ids - sampled_doc_ids) > sample_size:
            # Stop before we exceed the requested sample budget.
            break

        sampled_doc_ids.update(group_ids)
        sampled_qrels[query_id] = candidate_ids

    if not sampled_qrels:
        # If nothing could be sampled within the requested budget, keep one
        # valid query group.
        valid_query_id = None
        for query_id in query_ids:
            candidate_ids = {cid for cid in qrels[query_id] if cid in available_doc_ids}
            if candidate_ids:
                valid_query_id = query_id
                break

        if valid_query_id is not None:
            candidate_ids = {
                cid for cid in qrels[valid_query_id] if cid in available_doc_ids
            }
            sampled_doc_ids = {valid_query_id} | candidate_ids
            sampled_qrels = {valid_query_id: candidate_ids}
            logger.warning(
                "Requested sample_size too small to preserve qrels; "
                "sampling one valid query group instead."
            )
        else:
            logger.warning(
                "No valid qrels remain after filtering against document IDs; "
                "falling back to random document sampling."
            )
            sampled_doc_ids = set(random.sample(doc_ids, sample_size))

    if len(sampled_doc_ids) < sample_size:
        remaining_ids = [doc_id for doc_id in doc_ids if doc_id not in sampled_doc_ids]
        extra_count = min(sample_size - len(sampled_doc_ids), len(remaining_ids))
        if extra_count > 0:
            sampled_doc_ids.update(random.sample(remaining_ids, extra_count))

    sampled_documents = {doc_id: documents[doc_id] for doc_id in sampled_doc_ids}
    logger.info(
        f"Sampled {len(sampled_documents)} documents and preserved {len(sampled_qrels)} qrels"
    )

    return sampled_documents, sampled_qrels


# Baseline methods (user can choose which one to use as baseline)
# These are now defined in utils.py and imported above
# But we need to define the builders here since they use our local classes
BASELINE_METHODS = {
    "bm25": {
        "builder": lambda args: BM25Baseline(),
        "label": "BM25",
        "include_labels": False,
    },
    "dense": {
        "builder": lambda args: DenseEmbeddingBaseline(model_name=args.embedding_model),
        "label": "DenseEmbedding",
        "include_labels": False,
    },
}

# Experimental methods to compare against baseline
EXPERIMENTAL_METHODS = {
    "dense_labels": {
        "builder": lambda args: DenseEmbeddingBaseline(
            model_name=args.embedding_model,
            include_labels=True,
        ),
        "label": "DenseEmbedding+Labels",
    },
    "charm": {"builder": lambda args: CHARMInspiredMethod(), "label": "charm"},
    "graph_sage": {
        "builder": lambda args: SAGEGraphExpansionMethod(),
        "label": "GraphSage",
    },
    "gnn_ret": {
        "builder": lambda args: GNNRet(model_name=args.embedding_model, epochs=20, lr=0.01),
        "label": "GNNRet",
    },
    "novel_gat": {
        "builder": lambda args: NovelGATMethod(model_name=args.embedding_model, epochs=60, lr=0.1),
        "label": "NovelGAT",
    },
}


def build_benchmark_config(
    args, documents, qrels, baseline_config, experimental_method_keys
):
    """Create a serializable benchmark configuration block for result outputs."""
    return {
        "docs_path": args.docs_path,
        "qrels_path": args.qrels_path,
        "output_path": args.output_path,
        "embedding_model": args.embedding_model,
        "baseline": args.baseline,
        "baseline_label": baseline_config["label"],
        "methods": args.methods,
        "experimental_method_labels": {
            method_key: EXPERIMENTAL_METHODS[method_key]["label"]
            for method_key in experimental_method_keys
        },
        "k_values": args.k_values,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "debug": args.debug,
        "metadata_boost": getattr(args, "metadata_boost", None),
        "hyperparam_search": getattr(args, "hyperparam_search", False),
        "n_trials": getattr(args, "n_trials", None),
        "document_count": len(documents),
        "query_count": len(qrels),
    }


def hyperparam_search(args, documents, qrels, n_trials=20, seed=42):
    random.seed(seed)

    search_space = {
        "metadata_boost": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        "title_boost": [0.5, 1.0, 1.5, 2.0, 3.0],
        "main_text_boost": [0.5, 1.0, 1.5, 2.0],
    }

    primary_metric = f"ndcg@{max(args.k_values)}_mean"
    results = []
    seen = set()

    trial = 0
    while trial < n_trials:
        params = {k: random.choice(v) for k, v in search_space.items()}
        key = tuple(params[k] for k in sorted(params))
        if key in seen:
            continue
        seen.add(key)
        trial += 1

        logger.info(f"\nTrial {trial}/{n_trials}: {params}")

        method = CHARMInspiredMethod(
            field_weights={
                "metadata": params["metadata_boost"],
                "title": params["title_boost"],
                "main_text": params["main_text_boost"],
            }
        )
        label = f"CHARM_mb{params['metadata_boost']}_tb{params['title_boost']}_mt{params['main_text_boost']}"

        runner = BenchmarkRunner(
            documents,
            qrels,
            k_values=args.k_values,
            debug=False,
            cache_dir=Path(args.output_path).parent / "cache",
        )
        runner.run_method(label, method, use_cache=True)

        score = runner.results[label].get(primary_metric, 0.0)
        results.append(
            {**params, "label": label, "score": score, "metrics": runner.results[label]}
        )
        logger.info(f"  → {primary_metric}: {score:.4f}")

    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 80)
    print(f"HYPERPARAM SEARCH RESULTS (ranked by {primary_metric})")
    print("=" * 80)
    for r in results[:5]:
        print(
            f"  metadata={r['metadata_boost']} title={r['title_boost']} "
            f"main_text={r['main_text_boost']} → {primary_metric}={r['score']:.4f}"
        )

    best = results[0]
    logger.info(f"\nBest config: {best}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    use_cache = args.sample_size is None and not args.debug
    if not use_cache:
        logger.info("Caching disabled (sample_size / debug mode active)")

    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Enable debug logging when requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
        if args.sample_size is None:
            args.sample_size = 500
            logger.info(
                "No sample size provided; using sample_size=500 for debug checks"
            )

    # Load data
    documents, qrels = load_documents_and_qrels(
        args.docs_path,
        args.qrels_path,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    if args.sample_size is not None:
        documents, qrels = sample_documents(
            documents, qrels, args.sample_size, random_seed=args.seed
        )

    if getattr(args, "hyperparam_search", False):
        hyperparam_search(
            args, documents, qrels, n_trials=args.n_trials, seed=args.seed
        )
        return

    # Initialize benchmark runner
    runner = BenchmarkRunner(documents, qrels, k_values=args.k_values, debug=args.debug)

    # Always run baseline first
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING BASELINE METHOD ({args.baseline.upper()})")
    logger.info("=" * 80)
    baseline_config = BASELINE_METHODS[args.baseline]
    baseline_method = baseline_config["builder"](args)
    runner.run_method(
        baseline_config["label"],
        baseline_method,
        include_labels=baseline_config["include_labels"],
        use_cache=use_cache,
    )

    # Run experimental methods
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EXPERIMENTAL METHODS")
    logger.info("=" * 80)
    for method_key in args.methods:
        method_config = EXPERIMENTAL_METHODS[method_key]
        method = method_config["builder"](args)
        if isinstance(method, GNNRet) or isinstance(method, NovelGATMethod):
            method.build_index(documents)
            qrels_list = [
                {"query_id": qid, "candidate_ids": list(cids)}
                for qid, cids in qrels.items()
            ]
            method.train(qrels_list)
        runner.run_method(
            method_config["label"],
            method,
            use_cache=use_cache,
        )

    benchmark_config = build_benchmark_config(
        args,
        documents,
        qrels,
        baseline_config,
        args.methods,
    )

    runner.print_results()
    runner.print_comparison_summary(baseline_config["label"])
    runner.save_results(args.output_path, benchmark_config=benchmark_config)

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
