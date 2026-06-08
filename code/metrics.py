"""
Evaluation metrics for retrieval benchmarking.

Metrics computed:
- NDCG@k (Normalized Discounted Cumulative Gain)
- MAP@k (Mean Average Precision)
- Recall@k
- Precision@k
- MRR (Mean Reciprocal Rank)
"""

import numpy as np
from typing import Dict, List, Set


def ndcg_at_k(rankings: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Calculate NDCG@k for binary ground truth relevance.
    """
    dcg = 0.0
    for i, doc_id in enumerate(rankings[:k]):
        rel = 1 if doc_id in ground_truth else 0
        dcg += rel / np.log2(i + 2)

    ideal_rels = [1] * min(len(ground_truth), k)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(rankings: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Calculate MAP@k for binary ground truth relevance.
    """
    score = 0.0
    num_relevant = 0

    for i, doc_id in enumerate(rankings[:k]):
        if doc_id in ground_truth:
            num_relevant += 1
            score += num_relevant / (i + 1)

    num_relevant_total = len(ground_truth)
    return score / num_relevant_total if num_relevant_total > 0 else 0.0


def recall_at_k(rankings: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Calculate Recall@k.
    """
    num_relevant_retrieved = sum(1 for doc_id in rankings[:k] if doc_id in ground_truth)
    num_relevant_total = len(ground_truth)
    return (
        num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0.0
    )


def precision_at_k(rankings: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """
    Calculate Precision@k.
    """
    num_relevant_retrieved = sum(1 for doc_id in rankings[:k] if doc_id in ground_truth)
    return num_relevant_retrieved / k


def mrr(rankings: List[int], ground_truth: Set[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    Finds the rank of the first relevant document.
    """
    for i, doc_id in enumerate(rankings):
        if doc_id in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    rankings: List[int], ground_truth: Set[int], k_values: List[int] = [10]
) -> Dict[str, float]:
    """
    Compute all metrics for a single query.
    """
    metrics = {"mrr": mrr(rankings, ground_truth)}

    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(rankings, ground_truth, k)
        metrics[f"map@{k}"] = map_at_k(rankings, ground_truth, k)
        metrics[f"recall@{k}"] = recall_at_k(rankings, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(rankings, ground_truth, k)

    return metrics
