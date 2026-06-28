"""Utility functions for the benchmark script."""

import argparse
from typing import Any


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmarking Script for Semi-Structured Retrieval"
    )
    parser.add_argument(
        "--docs-path",
        default="project_datasets/output/documents_enriched_03.jsonl",
        help="Path to documents JSONL file",
    )
    parser.add_argument(
        "--qrels-path",
        default="project_datasets/output/qrels_enriched_02.jsonl",
        help="Path to qrels JSONL file",
    )
    parser.add_argument(
        "--output-path",
        default="project_code/results/method_results.json",
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
        choices=["bm25", "dense"],
        help="Baseline method to use for comparison (default: bm25). Available: bm25, dense"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        choices=["metadata_aware"],
        help="Experimental methods to benchmark against baseline. Available: metadata_aware",
    )
    parser.add_argument(
        "--include-categorical",
        action="store_true")

    parser.add_argument(
        "--include-hierarchical",
        action="store_true")

    return parser


def parse_args(args: Any = None) -> argparse.Namespace:
    """Parse and return command-line arguments.

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed arguments as argparse.Namespace object.
    """
    parser = create_parser()
    return parser.parse_args(args)
