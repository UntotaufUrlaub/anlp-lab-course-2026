import logging
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import json

try:
    from methods import (
        BaseMethod,
        prepare_text,
    )
    from metrics import evaluate
except ImportError:
    from project_code.methods import (
        BaseMethod,
        prepare_text,
    )
    from project_code.metrics import evaluate

logger = logging.getLogger(__name__)


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
                query_doc, include_title=True, include_categorical=False, include_hierarchical=False
            )

            # Extract metadata from query document for metadata-aware methods
            query_metadata = None
            structured = query_doc.get("structured_fields", {})
            if isinstance(structured, dict):
                query_metadata = {}
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
                    f"  Query {query_id}: {len(rankings)} "
                    f"retrieved, {relevant_count} "
                    f"total relevant, {matched_relevant} matched"
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
                    print(f"    query_id={detail['query_id']} "
                          f"retrieved={detail['retrieved']} " 
                          f"matched={detail['matched']} "
                          f"top={detail['top_retrieved']}")
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
