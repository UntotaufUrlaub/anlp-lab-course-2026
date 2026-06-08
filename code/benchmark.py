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
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging
from tqdm import tqdm
import argparse
import random

try:
    from methods import (
        BaseMethod,
        BM25Baseline,
        DenseEmbeddingBaseline,
        MetadataAwareMethod,
        prepare_text,
    )
    from metrics import evaluate
except ImportError:
    from code.methods import (
        BaseMethod,
        BM25Baseline,
        DenseEmbeddingBaseline,
        MetadataAwareMethod,
        prepare_text,
    )
    from code.metrics import evaluate

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
        logger.info(f"Sampled {
                len(qrels)} queries (batch_size={batch_size}, seed={random_seed})")

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
    logger.info(f"Sampled {
            len(sampled_documents)} documents and preserved {
            len(sampled_qrels)} qrels")

    return sampled_documents, sampled_qrels


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Run benchmarks on retrieval methods."""

    def __init__(
        self,
        documents: Dict,
        qrels: Dict,
        k_values: List[int] = [10],
        debug: bool = False,
    ):
        self.documents = documents
        self.qrels = qrels
        self.k_values = k_values
        self.results = {}
        self.debug = debug

    def run_method(self, method_name: str, method: BaseMethod) -> Dict:
        """
        Run a single method on all queries.

        Args:
        - method_name: Name of the method
        - method: Retrieval method object with retrieve() method

        Returns:
        - Dictionary with aggregated metrics
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running {method_name}...")
        logger.info(f"{'=' * 60}")

        query_metrics = []
        query_details = []
        debug_count = 0

        for query_id, ground_truth in tqdm(self.qrels.items(), desc=method_name):
            # Get query document
            if query_id not in self.documents:
                logger.warning(f"Query {query_id} not found in documents")
                continue

            query_doc = self.documents[query_id]
            query_text = prepare_text(
                query_doc, include_title=True, include_labels=False
            )

            # Extract metadata from query document for metadata-aware methods
            query_metadata = None
            structured = query_doc.get("structured_fields", {})
            if isinstance(structured, dict):
                query_metadata = {}
                if "multi_label" in structured:
                    query_metadata["multi_label"] = structured.get("multi_label")
                if "categorical" in structured:
                    query_metadata["categorical"] = structured.get("categorical")

            # Retrieve results (exclude query itself)
            # Use max of k_values to ensure we get enough results
            retrieve_top_k = max(self.k_values) if self.k_values else 10

            # Pass metadata to retrieve if method supports it
            if (
                hasattr(method, "retrieve")
                and "metadata" in method.retrieve.__code__.co_varnames
            ):
                results = method.retrieve(
                    query_text, top_k=retrieve_top_k, metadata=query_metadata
                )
            else:
                results = method.retrieve(query_text, top_k=retrieve_top_k)
            rankings = [doc_id for doc_id, _ in results if doc_id != query_id]

            relevant_count = len(ground_truth)
            matched_relevant = sum(1 for doc_id in rankings if doc_id in ground_truth)

            # Debug: print first few queries
            max_debug_queries = 5 if self.debug else 2
            if debug_count < max_debug_queries:
                logger.info(
                    f"  Query {query_id}: {
                        len(rankings)} retrieved, {relevant_count} total relevant, {matched_relevant} matched"
                )
                if len(rankings) > 0:
                    logger.info(f"    Top retrieved: {rankings[:5]}")
                debug_count += 1

            # Capture query-level summary
            query_details.append(
                {
                    "query_id": query_id,
                    "retrieved": len(rankings),
                    "top_retrieved": rankings[:5],
                    "matched": matched_relevant,
                }
            )

            # Compute metrics
            metrics = evaluate(rankings, ground_truth, self.k_values)
            query_metrics.append(metrics)

        # Aggregate metrics
        aggregated = self._aggregate_metrics(query_metrics)
        aggregated["query_details"] = query_details
        self.results[method_name] = aggregated

        return aggregated

    def _aggregate_metrics(self, query_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all queries."""
        if not query_metrics:
            return {}

        aggregated = {}
        for metric_name in query_metrics[0].keys():
            values = [m[metric_name] for m in query_metrics]
            aggregated[f"{metric_name}_mean"] = np.mean(values)

        return aggregated

    def print_results(self):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("BASELINE BENCHMARK RESULTS")
        print("=" * 80 + "\n")

        for method_name, metrics in self.results.items():
            print(f"\n{method_name}")
            print("-" * 80)

            # Group by metric type
            metric_types = {}
            for metric_name, value in metrics.items():
                base_metric = metric_name.rsplit("_", 1)[0]
                if base_metric not in metric_types:
                    metric_types[base_metric] = {}
                metric_types[base_metric][metric_name] = value

            for metric_type in sorted(metric_types.keys()):
                if metric_type == "query":
                    continue
                # print(f"\n  {metric_type.upper()}:")
                for stat_name in sorted(metric_types[metric_type].keys()):
                    value = metric_types[metric_type][stat_name]
                    try:
                        print(f"    {stat_name:30s}: {value:.4f}")
                    except (TypeError, ValueError):
                        print(f"    {stat_name:30s}: {value}")

            if "query_details" in metrics:
                print("\n  QUERY DETAILS:")
                for detail in metrics["query_details"][:5]:
                    print(f"    query_id={
                            detail['query_id']} retrieved={
                            detail['retrieved']} " f"matched={
                            detail['matched']} top={
                            detail['top_retrieved']}")
                if len(metrics["query_details"]) > 5:
                    print(
                        f"    ...and {len(metrics['query_details']) - 5} more queries"
                    )

        print("\n" + "=" * 80)

    def print_comparison_summary(self, baseline_label: str):
        """Print a comparison summary of experimental methods vs baseline."""
        if len(self.results) < 2:
            logger.info("Not enough methods to compare (need at least 2)")
            return

        if baseline_label not in self.results:
            logger.warning(f"Baseline '{baseline_label}' not found in results")
            return

        baseline_metrics = self.results[baseline_label]
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"\nBaseline: {baseline_label}\n")

        # Get only the metric names (exclude query_details)
        metric_names = [
            m for m in baseline_metrics.keys() if m != "query_details" and "_mean" in m
        ]

        # Compare each other method against baseline
        for method_name, method_metrics in self.results.items():
            if method_name == baseline_label:
                continue

            print(f"\n{method_name}")
            print("-" * 80)

            improvements = []
            degradations = []
            no_change = []

            for metric_name in sorted(metric_names):
                if metric_name not in method_metrics:
                    continue

                baseline_value = baseline_metrics[metric_name]
                method_value = method_metrics[metric_name]

                # Calculate percentage change
                if baseline_value != 0:
                    pct_change = (
                        (method_value - baseline_value) / baseline_value
                    ) * 100
                else:
                    pct_change = 0

                # Format output
                metric_display = metric_name.replace("_mean", "")
                symbol = "↑" if pct_change > 0 else "↓" if pct_change < 0 else "→"
                color_code = (
                    "\033[92m"
                    if pct_change > 0
                    else "\033[91m" if pct_change < 0 else ""
                )
                reset_code = "\033[0m"

                output = f"  {metric_display:25s}: {method_value:.4f} (baseline: {baseline_value:.4f}) {symbol} {pct_change:+.2f}%"

                print(output)

                if pct_change > 0:
                    improvements.append((metric_display, pct_change))
                elif pct_change < 0:
                    degradations.append((metric_display, pct_change))
                else:
                    no_change.append(metric_display)

            # Summary statistics
            if improvements or degradations:
                print("\n  Summary:")
                if improvements:
                    avg_improvement = np.mean([p[1] for p in improvements])
                    print(
                        f"    ✓ Improved {len(improvements)}/{len(metric_names)} metrics (avg: {avg_improvement:+.2f}%)"
                    )
                if degradations:
                    avg_degradation = np.mean([d[1] for d in degradations])
                    print(
                        f"    ✗ Degraded {len(degradations)}/{len(metric_names)} metrics (avg: {avg_degradation:+.2f}%)"
                    )

        print("\n" + "=" * 80)

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


# Baseline methods (user can choose which one to use as baseline)
BASELINE_METHODS = {
    "bm25": {
        "builder": lambda args: BM25Baseline(),
        "label": "BM25 (Baseline)",
    },
    "dense": {
        "builder": lambda args: DenseEmbeddingBaseline(model_name=args.embedding_model),
        "label": "DenseEmbedding (Sentence Transformers)",
    },
}

# Experimental methods to compare against baseline
EXPERIMENTAL_METHODS = {
    "metadata_aware": {
        "builder": lambda args: MetadataAwareMethod(
            metadata_boost=getattr(args, "metadata_boost", 2.0)
        ),
        "label": "MetadataAware (BM25 + metadata)",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmarking Script for Semi-Structured Retrieval"
    )
    parser.add_argument(
        "--docs-path",
        default="output/documents.jsonl",
        help="Path to documents JSONL file",
    )
    parser.add_argument(
        "--qrels-path", default="output/qrels.jsonl", help="Path to qrels JSONL file"
    )
    parser.add_argument(
        "--output-path",
        default="results/method_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence Transformer model name",
    )
    parser.add_argument(
        "--metadata-boost",
        type=float,
        default=2.0,
        help="Boost factor applied to metadata matches in MetadataAwareMethod",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10],
        help="K values for metrics (e.g., --k-values 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Approximate number of documents to sample for a smaller index; preserves sampled query/qrel groups when possible",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for a smaller sampled pipeline and additional logging",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of random queries to benchmark (default: None = all queries)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
    )
    parser.add_argument(
        "--baseline",
        default="bm25",
        choices=list(BASELINE_METHODS.keys()),
        help=f"Baseline method to use for comparison (default: bm25). Available: {', '.join(BASELINE_METHODS.keys())}",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(EXPERIMENTAL_METHODS.keys()),
        choices=list(EXPERIMENTAL_METHODS.keys()),
        help=f"Experimental methods to benchmark against baseline. Available: {', '.join(EXPERIMENTAL_METHODS.keys())}",
    )

    args = parser.parse_args()

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

    # Initialize benchmark runner
    runner = BenchmarkRunner(documents, qrels, k_values=args.k_values, debug=args.debug)

    # Always run baseline first
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING BASELINE METHOD ({args.baseline.upper()})")
    logger.info("=" * 80)
    baseline_config = BASELINE_METHODS[args.baseline]
    baseline_method = baseline_config["builder"](args)
    baseline_method.build_index(documents)
    runner.run_method(baseline_config["label"], baseline_method)

    # Run experimental methods
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EXPERIMENTAL METHODS")
    logger.info("=" * 80)
    for method_key in args.methods:
        method_config = EXPERIMENTAL_METHODS[method_key]
        method = method_config["builder"](args)
        method.build_index(documents, include_labels=True)
        runner.run_method(method_config["label"], method)

    # Print and save results
    runner.print_results()
    runner.print_comparison_summary(baseline_config["label"])
    runner.save_results(args.output_path)

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
