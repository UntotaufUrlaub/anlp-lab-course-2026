"""Retrieval method implementations for the baseline benchmark.

This module defines the base method contract and current retrieval methods.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple
from utils import prepare_text, extract_metadata_text

import numpy as np
from sklearn.decomposition import PCA

# try:
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# except ImportError:
# import subprocess

# subprocess.check_call(
#     ["pip", "install", "rank-bm25", "sentence-transformers", "scikit-learn"]
# )
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """Base class that all retrieval methods must derive from."""

    @abstractmethod
    def build_index(self, documents: Dict, include_labels: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query_doc: Dict, top_k: int = 100) -> List[Tuple[str, float]]:
        raise NotImplementedError

    def cache_key(self) -> str:
        """Filename stem used for the cache file. Defaults to the class name."""
        return type(self).__name__

    def save_cache(self, cache_dir: Path) -> None:
        """Persist index to disk. No-op by default."""

    def load_cache(self, cache_dir: Path) -> bool:
        """Load index from disk. Return True on hit, False on miss. No-op by default."""
        return False


class BM25Baseline(BaseMethod):
    """BM25 lexical retrieval method."""

    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []

    def build_index(self, documents: Dict, include_labels: bool = False) -> None:
        logger.info("Building BM25 index...")
        self.corpus = []
        self.doc_ids = []

        for doc_id, doc in documents.items():
            text = prepare_text(doc, include_title=True, include_labels=include_labels)
            tokens = text.lower().split()
            self.corpus.append(tokens)
            self.doc_ids.append(doc_id)

        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25 index built with {len(self.corpus)} documents")

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query_text = prepare_text(query_doc, include_title=True, include_labels=False)
        query_tokens = query_text.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices]


class DenseEmbeddingBaseline(BaseMethod):
    """Dense embedding retrieval method using Sentence Transformers."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", include_labels: bool = False
    ):
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        self.embeddings = []
        self.doc_ids = []
        self.include_labels = include_labels

    def build_index(self, documents: Dict, **kwargs) -> None:
        logger.info(
            f"Building embedding index... with include_labels {self.include_labels}"
        )
        self.embeddings = []
        self.doc_ids = []

        doc_list = list(documents.items())
        texts = []
        for doc_id, doc in doc_list:
            text = prepare_text(
                doc, include_title=True, include_labels=self.include_labels
            )
            texts.append(text)
            self.doc_ids.append(doc_id)

        logger.info(f"Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info(f"Embedding index built with {len(self.embeddings)} documents")

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query_text = prepare_text(
            query_doc, include_title=True, include_labels=self.include_labels
        )
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(similarities[i])) for i in top_k_indices]


class CHARMInspiredMethod(BaseMethod):
    """
    Simplified CHARM: separate per-field embeddings aggregated with static weights,
    two-stage retrieval via aggregated shortlist + max-field reranking.
    No fine-tuning; uses frozen sentence transformer.
    Field hierarchy: metadata -> title -> main_text (coarse to fine).
    """

    _CACHE_FIELDS = ("doc_ids", "agg_embeddings", "field_embeddings")

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        field_weights: Dict[str, float] = None,
        first_stage_k: int = 10,
    ):
        self.model = SentenceTransformer(model_name)
        self.field_weights = field_weights or {
            "metadata": 2.0,
            "title": 1.5,
            "main_text": 0.5,
        }
        self.first_stage_k = first_stage_k
        self.doc_ids = []
        self.agg_embeddings = None  # (N, D)
        self.field_embeddings = {}  # {field: (N, D)}

    def cache_key(self):
        w = "_".join(f"{k}{v}" for k, v in self.field_weights.items())
        return f"{type(self).__name__}_{w}_k{self.first_stage_k}"

    def _extract_fields(self, doc: Dict) -> Dict[str, str]:
        return {
            "metadata": extract_metadata_text(doc),
            "title": doc.get("title") or "",
            "main_text": doc.get("main_text") or "",
        }

    def _encode(self, texts):
        return self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True
        )

    def _cosine(self, matrix, vec):
        return (
            matrix
            @ vec
            / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec) + 1e-10)
        )

    def build_index(self, documents: Dict, **kwargs):
        self.doc_ids = list(documents.keys())
        all_fields = [self._extract_fields(doc) for doc in documents.values()]
        total_weight = sum(self.field_weights.values())

        weighted_sum = None
        for field, w in self.field_weights.items():
            texts = [f[field] for f in all_fields]
            embs = self._encode(texts)
            self.field_embeddings[field] = embs
            weighted_sum = (
                (weighted_sum + w * embs) if weighted_sum is not None else w * embs
            )

        self.agg_embeddings = weighted_sum / total_weight

    def retrieve(self, query_doc: Dict, top_k: int = 10, **kwargs):

        fields = self._extract_fields(query_doc)
        query_text = " ".join(v for v in fields.values() if v)
        q = self.model.encode(query_text, convert_to_numpy=True)

        # Stage 1: shortlist via aggregated embedding
        k1 = min(self.first_stage_k, len(self.doc_ids))
        agg_scores = self._cosine(self.agg_embeddings, q)
        shortlist = np.argsort(agg_scores)[-k1:][::-1]

        # Stage 2: rerank by max similarity across any field
        field_scores = np.stack(
            [
                self._cosine(self.field_embeddings[f][shortlist], q)
                for f in self.field_embeddings
            ],
            axis=1,
        ).max(axis=1)

        order = np.argsort(field_scores)[-(top_k + 1) :][::-1]
        final = shortlist[order]
        scores = field_scores[order]

        return [(self.doc_ids[i], float(s)) for i, s in zip(final, scores)]
