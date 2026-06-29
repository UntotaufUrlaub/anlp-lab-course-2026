"""Retrieval method implementations for the baseline benchmark.

This module defines the base method contract and current retrieval methods.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

from utils import (
    prepare_text,
    extract_metadata_text,
)

import numpy as np
from sklearn.decomposition import PCA
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


# create hash function to create a bm25 and dense embedding index depending on configuration of dataset
# to later load with joblib efficiently
def make_cache_key(
    documents: Dict,
    method: str,
    model: str = None,
    include_categorical: bool = False,
    include_hierarchical: bool = False,
) -> str:
    """Create a stable hash for the BM25 index input."""
    hasher = hashlib.sha256()

    if method == "dense" or method == "graph_sage" or method == "metadata_aware":
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
    def build_index(
        self,
        documents: Dict,
    ) -> None:
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
    """BM25 lexical retrieval method with joblib caching."""

    def __init__(
        self,
        include_categorical: bool = False,
        include_hierarchical: bool = False,
    ):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.cache_dir = CACHE_DIR
        self.include_hierarchical = include_hierarchical
        self.include_categorical = include_categorical

    def build_index(
        self,
        documents: Dict,
    ) -> None:
        # create unique cache_key for this document configuration
        cache_key = make_cache_key(
            documents,
            method="bm25",
            include_hierarchical=self.include_hierarchical,
            include_categorical=self.include_categorical,
        )
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
            text = prepare_text(
                doc,
                include_title=True,
                include_hierarchical=self.include_hierarchical,
                include_categorical=self.include_categorical,
            )
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
                "include_hierarchical": self.include_hierarchical,
                "include_categorical": self.include_categorical,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(f"BM25 index built with {len(self.corpus)} documents")
        logger.info(f"Saved BM25 index to cache: {cache_path}")

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query_text = prepare_text(query_doc, include_title=True, include_labels=False)
        query_tokens = query_text.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[-(top_k + 1) :][::-1]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_indices]


class DenseEmbeddingBaseline(BaseMethod):
    """Dense embedding retrieval method using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        include_categorical: bool = False,
        include_hierarchical: bool = False,
    ):
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        # include gpu if possible for speed up
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.include_categorical = include_categorical
        self.include_hierarchical = include_hierarchical
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        logger.info(f"Using device: {device}")

        self.embeddings = []
        self.doc_ids = []
        self.cache_dir = CACHE_DIR

    def build_index(
        self,
        documents: Dict,
    ) -> None:

        # create unique cache_key for this document configuration
        cache_key = make_cache_key(
            documents,
            include_hierarchical=self.include_hierarchical,
            include_categorical=self.include_categorical,
            method="dense",
            model=self.model_name,
        )
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

        logger.info(
            "Building embedding index... with include_labels {self.include_labels}"
        )
        self.embeddings = []
        self.doc_ids = []

        doc_list = list(documents.items())
        texts = []
        for doc_id, doc in doc_list:
            text = prepare_text(
                doc,
                include_title=True,
                include_categorical=self.include_categorical,
                include_hierarchical=self.include_hierarchical,
            )
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
                "include_hierarchical": self.include_hierarchical,
                "include_categorical": self.include_categorical,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(f"Embedding index built with {len(self.embeddings)} documents")

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query_text = prepare_text(
            query_doc,
            include_title=True,
            include_categorical=self.include_categorical,
            include_hierarchical=self.include_hierarchical,
        )
        query_embedding = self.model.encode(
            query_text, convert_to_numpy=True, show_progress_bar=False
        )
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
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


class SAGEGraphExpansionMethod(BaseMethod):

    def __init__(
        self,
        paper_graph_path=None,
        github_graph_path=None,
        model_name: str = "all-MiniLM-L6-v2",
        graph_weight: float = 0.15,
        expansion_factor: int = 5,
    ):
        graph_dir = PROJECT_ROOT / "graphs"
        self.graph_paths = {
            "paper": (
                Path(paper_graph_path)
                if paper_graph_path
                else graph_dir / "paper_graph.pkl"
            ),
            "github_issue": (
                Path(github_graph_path)
                if github_graph_path
                else graph_dir / "github_graph.pkl"
            ),
        }
        self.graph_weight = graph_weight
        self.expansion_factor = expansion_factor

        logger.info(f"Loading Sentence Transformer model: {model_name}")
        # include gpu if possible for speed up
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        logger.info(f"Using device: {device}")

        self.indexes = {
            "paper": {"embeddings": None, "doc_ids": [], "id_to_index": {}},
            "github_issue": {"embeddings": None, "doc_ids": [], "id_to_index": {}},
        }
        self.graphs = {}
        self.graph_node_lookup = {}
        self.query_source_by_text = {}
        self.cache_dir = CACHE_DIR

    def _dataset_key_for_doc(self, doc: Dict):
        source_type = doc.get("source_type")

        if source_type == "github_issue":
            return "github_issue"

        if source_type in {"paper", "query"}:
            return "paper"

        return None

    def _index_id_for_doc(self, doc_id, doc: Dict, dataset_key: str):
        if dataset_key == "paper":
            raw_source = doc.get("raw_source", {}) or {}
            native_id = raw_source.get("native_id")
            return str(native_id) if native_id else None

        if dataset_key == "github_issue":
            return str(doc_id)

        return None

    def _is_candidate(self, doc: Dict):
        return doc.get("retrieval_metadata", {}).get("is_candidate", True)

    def _is_queryable(self, doc: Dict):
        return doc.get("retrieval_metadata", {}).get("is_queryable", True)

    def _load_graphs(self):
        for dataset_key, graph_path in self.graph_paths.items():
            if dataset_key in self.graphs:
                continue

            if not graph_path.exists():
                logger.warning(f"Graph file not found for {dataset_key}: {graph_path}")
                self.graphs[dataset_key] = None
                self.graph_node_lookup[dataset_key] = {}
                continue

            logger.info(f"Loading {dataset_key} graph from {graph_path}")
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)

            self.graphs[dataset_key] = graph
            self.graph_node_lookup[dataset_key] = {
                str(node_id): node_id for node_id in graph.nodes
            }

    def _cosine_scores(self, embeddings, query_embedding):
        return (
            embeddings
            @ query_embedding
            / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
                + 1e-10
            )
        )

    def _graph_neighbor_scores(self, dataset_key, base_scores):
        graph = self.graphs.get(dataset_key)
        if graph is None:
            return {}

        index = self.indexes[dataset_key]
        id_to_index = index["id_to_index"]
        doc_ids = index["doc_ids"]
        graph_nodes = self.graph_node_lookup.get(dataset_key, {})
        expanded_scores = {}

        for doc_id, base_score in base_scores.items():
            graph_node = graph_nodes.get(str(doc_id))
            if graph_node is None:
                continue

            edge_iterators = [
                (
                    (target_id, data)
                    for _, target_id, data in graph.out_edges(graph_node, data=True)
                ),
                (
                    (source_id, data)
                    for source_id, _, data in graph.in_edges(graph_node, data=True)
                ),
            ]

            for edge_iterator in edge_iterators:
                for neighbor_id, edge_data in edge_iterator:
                    neighbor_key = str(neighbor_id)
                    if neighbor_key not in id_to_index:
                        continue

                    edge_weight = float(edge_data.get("weight", 1.0))
                    expanded_scores[neighbor_key] = expanded_scores.get(
                        neighbor_key, 0.0
                    ) + (self.graph_weight * base_score * edge_weight)

        return expanded_scores

    def build_index(
        self,
        documents: Dict,
    ) -> None:
        # create unique cache_key for this document configuration
        cache_key = make_cache_key(
            documents,
            include_hierarchical=True,
            include_categorical=True,
            method="graph_sage",
            model=self.model_name,
        )

        # determine the cache path for this configuration
        cache_path = self.cache_dir / f"graph_sage_{cache_key}.joblib"

        self._load_graphs()

        # if the cache_path exists we load it
        if cache_path.exists():
            logger.info(f"Loading graph_sage index from cache: {cache_path}")
            cache = joblib.load(cache_path)

            if "indexes" in cache:
                self.indexes = cache["indexes"]
                self.query_source_by_text = cache.get("query_source_by_text", {})
                for dataset_key, index in self.indexes.items():
                    index["id_to_index"] = {
                        doc_id: idx
                        for idx, doc_id in enumerate(index.get("doc_ids", []))
                    }

                total_embeddings = sum(
                    len(index.get("doc_ids", [])) for index in self.indexes.values()
                )
                logger.info(
                    f"Loaded graph_sage index with {total_embeddings} embeddings"
                )
                return

            logger.info("Ignoring old graph_sage cache format and rebuilding index.")

        logger.info("Building dataset-aware graph_sage embedding indexes...")
        texts_by_dataset = {"paper": [], "github_issue": []}
        doc_ids_by_dataset = {"paper": [], "github_issue": []}
        self.query_source_by_text = {}

        for doc_id, doc in documents.items():
            dataset_key = self._dataset_key_for_doc(doc)
            if dataset_key is None:
                continue

            query_text = prepare_text(
                doc,
                include_title=True,
                include_categorical=False,
                include_hierarchical=False,
            )
            self.query_source_by_text.setdefault(query_text, dataset_key)

            if not self._is_candidate(doc):
                continue

            index_id = self._index_id_for_doc(doc_id, doc, dataset_key)
            if not index_id:
                continue

            text = prepare_text(
                doc,
                include_title=True,
                include_categorical=True,
                include_hierarchical=True,
            )
            texts_by_dataset[dataset_key].append(text)
            doc_ids_by_dataset[dataset_key].append(index_id)

        for dataset_key, texts in texts_by_dataset.items():
            logger.info(f"Encoding {len(texts)} {dataset_key} documents...")

            if texts:
                embeddings = self.model.encode(
                    texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
                )
            else:
                embeddings = np.empty((0, 0))

            doc_ids = doc_ids_by_dataset[dataset_key]
            self.indexes[dataset_key] = {
                "embeddings": embeddings,
                "doc_ids": doc_ids,
                "id_to_index": {doc_id: idx for idx, doc_id in enumerate(doc_ids)},
            }

        # save it to the cache path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "indexes": self.indexes,
                "query_source_by_text": self.query_source_by_text,
                "include_hierarchical": True,
                "include_categorical": True,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(
            "Graph_sage index built with "
            f"{len(self.indexes['paper']['doc_ids'])} paper documents and "
            f"{len(self.indexes['github_issue']['doc_ids'])} GitHub documents"
        )

    def _dataset_key_for_query(self, query: str, metadata=None):
        if metadata:
            source_type = metadata.get("source_type") or metadata.get("source_dataset")
            if source_type == "github_issue" or (
                isinstance(source_type, str) and "/" in source_type
            ):
                return "github_issue"
            if source_type in {"paper", "query"}:
                return "paper"

        dataset_key = self.query_source_by_text.get(query)
        if dataset_key:
            return dataset_key

        raise ValueError(
            "Could not determine whether query belongs to the paper or GitHub dataset. "
            "Pass metadata with source_type, or build the index with queryable documents."
        )

    def retrieve(
        self, query: Dict, top_k: int = 10, metadata=None, **kwargs
    ) -> List[Tuple[str, float]]:
        prep_query = prepare_text(
            query,
            include_title=True,
            include_categorical=True,
            include_hierarchical=True,
        )
        dataset_key = self._dataset_key_for_query(prep_query, metadata=metadata)
        index = self.indexes[dataset_key]
        embeddings = index["embeddings"]
        doc_ids = index["doc_ids"]

        if embeddings is None or len(doc_ids) == 0:
            return []

        query_embedding = self.model.encode(
            prep_query, convert_to_numpy=True, show_progress_bar=False
        )
        similarities = self._cosine_scores(embeddings, query_embedding)

        shortlist_size = min(len(doc_ids), max(top_k * self.expansion_factor, top_k))
        shortlist_indices = np.argsort(similarities)[-shortlist_size:][::-1]

        combined_scores = {
            doc_ids[idx]: float(similarities[idx]) for idx in shortlist_indices
        }

        graph_scores = self._graph_neighbor_scores(dataset_key, combined_scores)
        for doc_id, graph_score in graph_scores.items():
            combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + graph_score

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[: top_k + 1]
