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
    subprocess.check_call(["pip", "install", "rank-bm25", "sentence-transformers", "scikit-learn"])
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def prepare_text(doc: Dict, include_title: bool = True, include_labels: bool = True) -> str:
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


class BaseMethod(ABC):
    """Base class that all retrieval methods must derive from."""

    @abstractmethod
    def build_index(self, documents: Dict, include_labels: bool = True) -> None:
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

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        self.embeddings = []
        self.doc_ids = []

    def build_index(self, documents: Dict, include_labels: bool = True) -> None:
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
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Embedding index built with {len(self.embeddings)} documents")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.doc_ids[i], float(similarities[i])) for i in top_k_indices]
