"""Utility functions for the benchmark script."""

import argparse
from typing import Any, Dict, List, Tuple

DEFAULT_RESULTS_FILENAME = "method_results.json"
DEFAULT_RESULTS_PATH = f"results/{DEFAULT_RESULTS_FILENAME}"


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmarking Script for Semi-Structured Retrieval"
    )
    parser.add_argument(
        "--docs-path",
        default="output/documents_enriched_03.jsonl",
        help="Path to documents JSONL file",
    )
    parser.add_argument(
        "--qrels-path",
        default="output/qrels_enriched_02.jsonl",
        help="Path to qrels JSONL file",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_RESULTS_PATH,
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
        default=3.0,
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
        default="dense",
        choices=["bm25", "dense"],
        help="Baseline method to use for comparison (default: dense). Available: bm25, dense",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dense_labels", "charm", "graph_sage"],
        choices=["dense_labels", "charm", "graph_sage"],
        help="Experimental methods to benchmark against baseline. Available: dense_labels, charm, graph_sage",
    )
    parser.add_argument("--hyperparam-search", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)

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


def _flatten_hierarchical(hierarchical) -> List[str]:
    if isinstance(hierarchical, list):
        return [v for v in hierarchical if v]
    elif isinstance(hierarchical, dict):
        parts = []
        for v in hierarchical.values():
            if isinstance(v, dict):
                parts.extend(str(x) for x in v.values() if x)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v if x)
            elif v:
                parts.append(str(v))
        return parts
    return []


# TODO refactor categorical extract to method
def extract_metadata_text(doc: Dict) -> str:
    structured = doc.get("structured_fields", {}) or {}
    categorical = structured.get("categorical", {}) or {}
    hierarchical = structured.get("hierarchical", []) or []

    parts = []
    for v in categorical.values():
        if isinstance(v, list):
            parts.extend(str(x) for x in v if x is not None)
        elif v is not None:
            parts.append(str(v))

    parts += _flatten_hierarchical(hierarchical)
    return " ".join(parts)


# TODO does order of concatenating matters?
def prepare_text_dennis(
    doc: Dict, include_title: bool = True, include_labels: bool = False
) -> str:
    text_parts = []
    if include_labels:
        text_parts.append(extract_metadata_text(doc))
    if include_title and doc.get("title"):
        text_parts.append(doc["title"])
    if doc.get("main_text"):
        text_parts.append(doc["main_text"])
    return " ".join(text_parts)
