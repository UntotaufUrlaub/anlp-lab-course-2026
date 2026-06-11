"""Retrieval method implementations for the baseline benchmark.

This module defines the base method contract and current retrieval methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

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


def prepare_text(
    doc: Dict, include_title: bool = True, include_labels: bool = False
) -> str:
    """Prepare document text for retrieval."""
    text_parts = []

    if include_title and doc.get("title"):
        text_parts.append(doc["title"])

    if include_labels:
        structured = doc.get("structured_fields", {})
        multi_label = structured.get("multi_label", {})
        for key, values in multi_label.items():
            if isinstance(values, list):
                text_parts.append(" ".join(values))
            else:
                text_parts.append(str(values))

    if doc.get("main_text"):
        text_parts.append(doc["main_text"])

    return " ".join(text_parts)


class BaseMethod(ABC):
    """Base class that all retrieval methods must derive from."""

    @abstractmethod
    def build_index(self, documents: Dict, include_labels: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        raise NotImplementedError


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

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices]


class DenseEmbeddingBaseline(BaseMethod):
    """Dense embedding retrieval method using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        self.embeddings = []
        self.doc_ids = []

    def build_index(self, documents: Dict, include_labels: bool = False) -> None:
        logger.info("Building embedding index...")
        self.embeddings = []
        self.doc_ids = []

        doc_list = list(documents.items())
        texts = []
        for doc_id, doc in doc_list:
            text = prepare_text(doc, include_title=True, include_labels=include_labels)
            texts.append(text)
            self.doc_ids.append(doc_id)

        logger.info(f"Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info(f"Embedding index built with {len(self.embeddings)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(similarities[i])) for i in top_k_indices]


# TODO hierarchical data wehen real available somehow
class MetadataAwareMethod(BaseMethod):
    """Metadata-aware retrieval with weighted field aggregation and optional two-stage reranking."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        metadata_boost: float = 2.0,
        first_stage_k: int = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.metadata_boost = metadata_boost
        self.first_stage_k = first_stage_k

        self.doc_ids = []
        self.agg_embeddings = None  # (N, D) for first-stage
        self.field_embeddings = {}  # {field: (N, D)} for reranking

    def _extract_fields(self, doc: Dict) -> Dict[str, str]:
        """Return ordered fields: metadata → title → main_text."""
        structured = doc.get("structured_fields", {}) or {}
        multi_label = structured.get("multi_label", {})

        metadata = multi_label or multi_label.get("fields_of_study") or []

        return {
            "metadata": (
                " ".join(metadata) if isinstance(metadata, list) else str(metadata)
            ),
            "title": doc.get("title") or "",
            "main_text": doc.get("main_text") or "",
        }

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def build_index(self, documents: Dict, **kwargs) -> None:
        self.doc_ids = list(documents.keys())
        all_fields = [self._extract_fields(doc) for doc in documents.values()]

        field_weights = {
            "metadata": self.metadata_boost,
            "title": 2.0,
            "main_text": 1.0,
        }

        # Encode each field as a batch
        for field in field_weights:
            texts = [f[field] for f in all_fields]
            self.field_embeddings[field] = self._encode(texts)  # (N, D)

        # Aggregated embedding: weighted average over fields
        total_weight = sum(field_weights.values())
        self.agg_embeddings = (
            sum(w * self.field_embeddings[f] for f, w in field_weights.items())
            / total_weight
        )  # (N, D)

    def retrieve(
        self, query: str, top_k: int = 10, **kwargs
    ) -> List[Tuple[str, float]]:
        q = self._encode([query])[0]  # (D,)

        def cosine(matrix, vec):
            return (
                matrix
                @ vec
                / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec) + 1e-10)
            )

        top_k += 1
        # Stage 1: shortlist via aggregated embeddings
        k1 = self.first_stage_k or len(self.doc_ids)
        scores = cosine(self.agg_embeddings, q)
        shortlist = np.argsort(scores)[-k1:][::-1]

        # Stage 2: rerank by max field similarity over shortlist
        if self.first_stage_k:
            field_scores = np.stack(
                [
                    cosine(self.field_embeddings[f][shortlist], q)
                    for f in self.field_embeddings
                ],
                axis=1,
            ).max(
                axis=1
            )  # (k1,)
            order = np.argsort(field_scores)[-top_k:][::-1]
            shortlist = shortlist[order]
            scores = field_scores[order]
        else:
            scores = scores[shortlist[:top_k]]
            shortlist = shortlist[:top_k]

        return [(self.doc_ids[i], float(s)) for i, s in zip(shortlist, scores)]
