"""Retrieval method implementations for the baseline benchmark.

This module defines the base method contract and current retrieval methods.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib
import hashlib
import json
import torch

# try:
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pathlib import Path

# except ImportError:
# import subprocess

# subprocess.check_call(
#     ["pip", "install", "rank-bm25", "sentence-transformers", "scikit-learn"]
# )
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"

# helper method to rewrite text format into one nice string
def _as_text_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, dict):
        items = []
        for v in value.values():
            items.extend(_as_text_list(v))
        return items
    return [str(value)]

def prepare_text(
    doc: Dict, include_title: bool = True, include_hierarchical: bool = False, include_categorical: bool = False
) -> str:
    """Prepare document text for retrieval."""
    text_parts = []

    if include_title and doc.get("title"):
        text_parts.append(doc["title"])

    structured = doc.get("structured_fields") or {}

    if include_categorical:
        categorical = structured.get("categorical")
        text_parts.append(" ".join(_as_text_list(categorical)))


    if include_hierarchical:
        hierarchical = structured.get("hierarchical")
        text_parts.append(" ".join(_as_text_list(hierarchical)))

    if doc.get("main_text"):
        text_parts.append(doc["main_text"])

    return " ".join(text_parts)

# create hash function to create a bm25 and dense embedding index depending on configuration of dataset
# to later load with joblib efficiently
def make_cache_key(documents: Dict, method:str,model: str = None,
                   include_categorical: bool = False, include_hierarchical: bool=False,) -> str:
    """Create a stable hash for the BM25 index input."""
    hasher = hashlib.sha256()

    if method=="dense":
        config = {
            "method": method,
            "model": model,
            "include_categorical": include_categorical,
            "include_hierarchical": include_hierarchical,
            "num_documents": len(documents),
        }
    else:
        config = {
            "method": method,
            "include_categorical": include_categorical,
            "include_hierarchical": include_hierarchical,
            "num_documents": len(documents),
        }
    hasher.update(json.dumps(config, sort_keys=True).encode("utf-8"))

    for doc_id in sorted(documents.keys()):
        doc = documents[doc_id]

        relevant_data = {
            "id": doc_id,
            "title": doc.get("title"),
            "main_text": doc.get("main_text"),
        }

        structured = doc.get("structured_fields", {})

        if include_categorical:
            relevant_data["categorical"] = structured.get("categorical")

        if include_hierarchical:
            relevant_data["hierarchical"] = structured.get("hierarchical")

        hasher.update(
            json.dumps(relevant_data, sort_keys=True, default=str).encode("utf-8")
        )

    return hasher.hexdigest()[:12]


class BaseMethod(ABC):
    """Base class that all retrieval methods must derive from."""

    @abstractmethod
    def build_index(self, documents: Dict, include_hierarchical: bool = False, include_categorical: bool=False)\
            -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
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
    """BM25 lexical retrieval method with joblib caching."""

    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.cache_dir = CACHE_DIR

    def build_index(self, documents: Dict, include_hierarchical: bool = False, include_categorical: bool = False) \
            -> None:
        # create unique cache_key for this document configuration
        cache_key = make_cache_key(documents, method = "bm25", include_hierarchical=include_hierarchical,
                                        include_categorical=include_categorical)
        # determine the cache path for this configuration
        cache_path = self.cache_dir / f"bm25_{cache_key}.joblib"

        # if the cache_path exists we load it
        if cache_path.exists():
            logger.info(f"Loading BM25 index from cache: {cache_path}")
            cache = joblib.load(cache_path)

            self.bm25 = cache["bm25"]
            self.corpus = cache["corpus"]
            self.doc_ids = cache["doc_ids"]

            logger.info(f"Loaded BM25 index with {len(self.corpus)} documents")
            return

        # build the index
        logger.info("Building BM25 index...")
        self.corpus = []
        self.doc_ids = []

        for doc_id, doc in documents.items():
            text = prepare_text(doc, include_title=True,
                                include_hierarchical=include_hierarchical,
                                include_categorical=include_categorical)
            tokens = text.lower().split()
            self.corpus.append(tokens)
            self.doc_ids.append(doc_id)

        self.bm25 = BM25Okapi(self.corpus)

        # save it to the cache path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "bm25": self.bm25,
                "corpus": self.corpus,
                "doc_ids": self.doc_ids,
                "include_hierarchical": include_hierarchical,
                "include_categorical": include_categorical,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(f"BM25 index built with {len(self.corpus)} documents")
        logger.info(f"Saved BM25 index to cache: {cache_path}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices]


class DenseEmbeddingBaseline(BaseMethod):
    """Dense embedding retrieval method using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        # include gpu if possible for speed up
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        logger.info(f"Using device: {device}")

        self.embeddings = []
        self.doc_ids = []
        self.cache_dir = CACHE_DIR

    def build_index(self, documents: Dict, include_hierarchical: bool = False, include_categorical: bool = False)\
            -> None:
        # create unique cache_key for this document configuration
        cache_key = make_cache_key(documents, include_hierarchical=include_hierarchical,
                                    include_categorical=include_categorical,
                                    method="dense", model=self.model_name)
        # determine the cache path for this configuration
        cache_path = self.cache_dir / f"dense_{cache_key}.joblib"

        # if the cache_path exists we load it
        if cache_path.exists():
            logger.info(f"Loading dense index from cache: {cache_path}")
            cache = joblib.load(cache_path)

            self.embeddings = cache["embeddings"]
            self.doc_ids = cache["doc_ids"]

            logger.info(f"Loaded dense index with {len(self.embeddings)} embeddings")
            return

        logger.info("Building embedding index...")
        self.embeddings = []
        self.doc_ids = []

        doc_list = list(documents.items())
        texts = []
        for doc_id, doc in doc_list:
            text = prepare_text(doc, include_title=True, include_categorical=include_categorical,
                                include_hierarchical=include_hierarchical)
            texts.append(text)
            self.doc_ids.append(doc_id)

        logger.info(f"Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )

        # save it to the cache path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embeddings": self.embeddings,
                "doc_ids": self.doc_ids,
                "include_hierarchical": include_hierarchical,
                "include_categorical": include_categorical,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(f"Embedding index built with {len(self.embeddings)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(similarities[i])) for i in top_k_indices]


class MetadataAwareMethod(BaseMethod):
    """Metadata-aware retrieval with weighted field aggregation and optional two-stage reranking."""

    # Fields pickled to / restored from cache
    _CACHE_FIELDS = ("doc_ids", "agg_embeddings", "field_embeddings")

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        metadata_boost: float = 2.0,
        first_stage_k: int = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.metadata_boost = metadata_boost
        self.first_stage_k = first_stage_k

        self.doc_ids = []
        self.agg_embeddings = None  # (N, D) for first-stage
        self.field_embeddings = {}  # {field: (N, D)} for reranking

    def _as_text_list(self,value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v]
        if isinstance(value, dict):
            items = []
            for v in value.values():
                items.extend(self._as_text_list(v))
            return items
        return [str(value)]

    def _extract_fields(self, doc: Dict) -> Dict[str, str]:
        structured = doc.get("structured_fields", {}) or {}
        categorical = structured.get("categorical", {}) or {}
        hierarchical = structured.get("hierarchical", {}) or {}
        if isinstance(categorical, dict):
            fields_of_study = categorical.get("fields_of_study", [])
            metadata_parts = self._as_text_list(fields_of_study)
        else:
            metadata_parts = self._as_text_list(categorical)

        metadata_parts += self._as_text_list(hierarchical)

        return {
            "metadata": " ".join(metadata_parts),
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
            self.field_embeddings[field] = self._encode(texts)

        # Aggregated embedding: weighted average over fields
        total_weight = sum(field_weights.values())
        self.agg_embeddings = (
            sum(w * self.field_embeddings[f] for f, w in field_weights.items())
            / total_weight
        )

    def save_cache(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{self.cache_key()}.pkl"
        payload = {field: getattr(self, field) for field in self._CACHE_FIELDS}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Cache saved: {path}")

    def load_cache(self, cache_dir: Path) -> bool:
        path = cache_dir / f"{self.cache_key()}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            payload = pickle.load(f)
        for field, value in payload.items():
            setattr(self, field, value)
        logger.info(f"Cache loaded: {path} ({len(self.doc_ids)} docs)")
        return True

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

class SAGEGraphExpansionMethod(BaseMethod):

    def build_index(self, documents: Dict, **kwargs) -> None:
        pass

    def retrieve(self, query: str, **kwargs) -> List[Tuple[str, float]]:
        pass
