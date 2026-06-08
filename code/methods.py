"""Retrieval method implementations for the baseline benchmark.

This module defines the base method contract and current retrieval methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess

    subprocess.check_call(
        ["pip", "install", "rank-bm25", "sentence-transformers", "scikit-learn"]
    )
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def prepare_text(
    doc: Dict, include_title: bool = True, include_labels: bool = False
) -> str:
    """Prepare document text for retrieval."""
    text_parts = []

    if include_title and doc.get("title"):
        text_parts.append(doc["title"])

    if doc.get("main_text"):
        text_parts.append(doc["main_text"])

    if include_labels:
        structured = doc.get("structured_fields", {})
        multi_label = structured.get("multi_label", {})
        for key, values in multi_label.items():
            if isinstance(values, list):
                text_parts.append(" ".join(values))
            else:
                text_parts.append(str(values))

    return " ".join(text_parts)


# TODO hierarchical data wehen available


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

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
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

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.doc_ids[i], float(similarities[i])) for i in top_k_indices]


class MetadataAwareMethod(BaseMethod):
    """Metadata-aware retrieval that combines BM25 text scores with metadata matching.

    Usage:
      - During `build_index`, the method collects BM25 tokenized corpus and a simple
        metadata set per document composed from `structured_fields` and `entities`.
      - At `retrieve` time, pass an optional `metadata` dict (e.g. `{"labels": ["bug"]}`)
        to boost documents that match the provided metadata values.

    This is intended for cases where hierarchical fields are not available yet but
    categorical/multi-label metadata exists.
    """

    def __init__(self, metadata_boost: float = 2.0):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.doc_metadata_sets = []
        self.metadata_boost = float(metadata_boost)

    def _build_metadata_set(self, doc: Dict) -> set:
        s = set()
        structured = doc.get("structured_fields", {})
        categorical = (
            structured.get("categorical", {}) if isinstance(structured, dict) else {}
        )
        multi_label = (
            structured.get("multi_label", {}) if isinstance(structured, dict) else {}
        )

        for k, v in categorical.items():
            if v is None:
                continue
            s.add(str(v).lower())

        for k, values in multi_label.items():
            if isinstance(values, list):
                for vv in values:
                    s.add(str(vv).lower())
            else:
                s.add(str(values).lower())

        entities = doc.get("entities", {}) or {}
        for k, values in entities.items():
            if isinstance(values, list):
                for vv in values:
                    s.add(str(vv).lower())
            else:
                s.add(str(values).lower())

        # also include raw_source fields if present (provenance)
        raw = doc.get("raw_source", {}) or {}
        for k, v in raw.items():
            if v is None:
                continue
            if isinstance(v, list):
                for vv in v:
                    s.add(str(vv).lower())
            else:
                s.add(str(v).lower())

        return s

    def build_index(self, documents: Dict, include_labels: bool = False) -> None:
        logger.info("Building Metadata-aware BM25 index...")
        self.corpus = []
        self.doc_ids = []
        self.doc_metadata_sets = []

        for doc_id, doc in documents.items():
            text = prepare_text(doc, include_title=True, include_labels=include_labels)
            tokens = text.lower().split()
            self.corpus.append(tokens)
            self.doc_ids.append(doc_id)
            self.doc_metadata_sets.append(self._build_metadata_set(doc))

        self.bm25 = BM25Okapi(self.corpus)
        logger.info(
            f"Metadata-aware BM25 index built with {len(self.corpus)} documents"
        )

    def retrieve(
        self, query: str, top_k: int = 5, metadata: Dict = None
    ) -> List[Tuple[str, float]]:
        # BM25 text scores
        query_tokens = query.lower().split()
        text_scores = self.bm25.get_scores(query_tokens)

        # normalize text scores to [0,1]
        max_score = float(np.max(text_scores)) if len(text_scores) > 0 else 1.0
        if max_score <= 0:
            max_score = 1.0
        text_scores_norm = text_scores / max_score

        # compute metadata match scores
        metadata_score = np.zeros_like(text_scores_norm)
        if metadata:
            # collect query metadata values as a set of lowercased strings
            query_values = set()
            for k, v in (metadata.items() if isinstance(metadata, dict) else []):
                if isinstance(v, list):
                    for vv in v:
                        query_values.add(str(vv).lower())
                else:
                    query_values.add(str(v).lower())

            if len(query_values) > 0:
                for i, meta_set in enumerate(self.doc_metadata_sets):
                    if not meta_set:
                        continue
                    # simple match fraction
                    matches = len(meta_set.intersection(query_values))
                    metadata_score[i] = matches / float(len(query_values))

        combined = text_scores_norm + (self.metadata_boost * metadata_score)
        top_k_indices = np.argsort(combined)[-top_k:][::-1]
        return [(self.doc_ids[i], float(combined[i])) for i in top_k_indices]
