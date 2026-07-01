"""Retrieval method implementations for the baseline benchmark.

This module defines the base method contract and current retrieval methods.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple
from utils import prepare_text_dennis, extract_metadata_text

import numpy as np
import joblib
import hashlib
import json
import torch

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
    doc: Dict,
    include_title: bool = True,
    include_labels: bool = False,
) -> str:
    """Prepare document text for retrieval."""
    text_parts = []

    if include_title and doc.get("title"):
        text_parts.append(doc["title"])

    structured = doc.get("structured_fields") or {}

    if include_labels:
        categorical = structured.get("categorical")
        text_parts.append(" ".join(_as_text_list(categorical)))
        hierarchical = structured.get("hierarchical")
        text_parts.append(" ".join(_as_text_list(hierarchical)))

    if doc.get("main_text"):
        text_parts.append(doc["main_text"])

    return " ".join(text_parts)


# create hash function to create a bm25 and dense embedding index depending on configuration of dataset
# to later load with joblib efficiently
def make_cache_key(
    documents: Dict,
    method: str,
    model: str = None,
    include_labels: bool = True,
) -> str:
    """Create a stable hash for the BM25 index input."""
    hasher = hashlib.sha256()

    if method == "dense" or method == "graph_sage" or method == "metadata_aware":
        config = {
            "method": method,
            "model": model,
            "include_labels": include_labels,
            "num_documents": len(documents),
        }
    else:
        config = {
            "method": method,
            "include_labels": include_labels,
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

        if include_labels:
            relevant_data["categorical"] = structured.get("categorical")
            relevant_data["hierarchical"] = structured.get("hierarchical")

        hasher.update(
            json.dumps(relevant_data, sort_keys=True, default=str).encode("utf-8")
        )

    return hasher.hexdigest()[:12]


class BaseMethod(ABC):
    """Base class that all retrieval methods must derive from."""

    @abstractmethod
    def build_index(self, documents: Dict) -> None:
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

    def __init__(self, include_labels: bool = False):
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self.cache_dir = CACHE_DIR
        self.include_labels = include_labels

    def build_index(self, documents: Dict) -> None:
        # create unique cache_key for this document configuration
        cache_key = make_cache_key(
            documents, method="bm25", include_labels=self.include_labels
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
                doc, include_title=True, include_labels=self.include_labels
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
                "include_labels": self.include_labels,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(f"BM25 index built with {len(self.corpus)} documents")
        logger.info(f"Saved BM25 index to cache: {cache_path}")

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query_text = prepare_text(
            query_doc, include_title=True, include_labels=self.include_labels
        )
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
        # include gpu if possible for speed up
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        logger.info(f"Using device: {device}")
        self.model_name = model_name
        self.embeddings = []
        self.doc_ids = []
        self.include_labels = include_labels
        self.cache_dir = CACHE_DIR

    def build_index(self, documents: Dict, **kwargs) -> None:
        # create unique cache_key for this document configuration
        cache_key = make_cache_key(
            documents,
            include_labels=self.include_labels,
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

        logger.info("Building embedding index...")
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

        # save it to the cache path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embeddings": self.embeddings,
                "doc_ids": self.doc_ids,
                "include_labels": self.include_labels,
                "cache_key": cache_key,
            },
            cache_path,
        )

        logger.info(f"Embedding index built with {len(self.embeddings)} documents")

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query_text = prepare_text(
            query_doc, include_title=True, include_labels=self.include_labels
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
            "metadata": 1.0,
            "title": 1.0,
            "main_text": 2.0,
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

    def _is_useless_metadata(self, doc: Dict) -> bool:
        structured = doc.get("structured_fields") or {}
        categorical = structured.get("categorical", {}) or {}
        hierarchical = structured.get("hierarchical", {}) or {}
        keys = set(categorical.keys())
        return (
            bool(keys)
            and keys <= {"specificity", "quality"}
            and not hierarchical
            and all(
                isinstance(v, (int, float)) or str(v).strip()
                for v in categorical.values()
            )
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
        metadata_text = fields["metadata"].strip()

        active_weights = (
            self.field_weights
            if metadata_text and not self._is_useless_metadata(query_doc)
            else {
                "main_text": self.field_weights.get("main_text", 1.0),
            }
        )

        weighted_sum = None
        total_weight = sum(active_weights.values())
        for field, w in active_weights.items():
            text = fields[field]
            if not text:
                continue
            emb = self.model.encode(text, convert_to_numpy=True)
            weighted_sum = (
                (weighted_sum + w * emb) if weighted_sum is not None else w * emb
            )

        if weighted_sum is None:
            return []

        q = weighted_sum / total_weight
        scores = self._cosine(self.agg_embeddings, q)
        order = np.argsort(scores)[-(top_k + 1) :][::-1]

        return [(self.doc_ids[i], float(scores[i])) for i in order]


class SAGEGraphExpansionMethod(BaseMethod):

    def __init__(
        self,
        paper_graph_path=None,
        github_graph_path=None,
        model_name: str = "all-MiniLM-L6-v2",
        graph_weight: float = 0.15,
        expansion_factor: int = 5,
        include_labels: bool = True,
    ):
        self.include_labels = include_labels
        # directory where graphs are
        graph_dir = PROJECT_ROOT / "graphs"
        # individual graphs of paper and github_issue dataset
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
        # weight how strongly graph-neighbors influence final score
        self.graph_weight = graph_weight
        # controls how many seed nodes are used to then start graph expansion
        self.expansion_factor = expansion_factor

        logger.info(f"Loading Sentence Transformer model: {model_name}")
        # include gpu if possible for speed up
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        logger.info(f"Sentence Transformer model loaded: {model_name}")
        logger.info(f"Using device: {device}")

        # store embeddings and document_ids
        # id_to_index stores paper/github entry id and corresponding row in embedding matrix
        # id_to_dataset stores paper/github entry id and which dataset it belongs to
        self.indexes = {
            "embeddings": None,
            "doc_ids": [],
            "id_to_index": {},
            "id_to_dataset": {},
        }
        # hold the networkx graphs in "paper" and "github_issue"
        self.graphs = {}

        # enter all node_ids of each graph
        self.graph_node_lookup = {}
        #
        self.query_source_by_text = {}
        self.cache_dir = CACHE_DIR

    # given a document return which dataset it belongs to
    def _dataset_key_for_doc(self, doc: Dict):
        source_type = doc.get("source_type")

        if source_type == "github_issue":
            return "github_issue"

        if source_type == "paper":
            return "paper"

        return None

    # given a document index and document, return the native id for papers and document id for github issues
    def _index_id_for_doc(self, doc_id, doc: Dict, dataset_key: str):
        if dataset_key == "paper":
            raw_source = doc.get("raw_source", {}) or {}
            native_id = raw_source.get("native_id")
            return str(native_id) if native_id else None

        if dataset_key == "github_issue":
            return str(doc_id)

        return None

    # load the graphs from the directories in self.graph_paths
    def _load_graphs(self):
        for dataset_key, graph_path in self.graph_paths.items():
            # means we already have loaded this graph
            if dataset_key in self.graphs:
                continue

            if not graph_path.exists():
                logger.warning(f"Graph file not found for {dataset_key}: {graph_path}")
                self.graphs[dataset_key] = None
                self.graph_node_lookup[dataset_key] = {}
                continue

            # loaded graph can be put into self.graphs
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

        id_to_index = self.indexes["id_to_index"]
        expanded_scores = {}

        for doc_id, base_score in base_scores.items():
            graph_node = str(doc_id)
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
            method="graph_sage",
            model=self.model_name,
        )

        # determine the cache path for this configuration
        cache_path = self.cache_dir / f"graph_sage_{cache_key}.joblib"

        # load the graphs
        self._load_graphs()

        # if the cache_path exists we load it
        if cache_path.exists():
            logger.info(f"Loading graph_sage index from cache: {cache_path}")
            cache = joblib.load(cache_path)

            if "indexes" in cache:
                # reconstruct the cache
                self.indexes = cache["indexes"]

                # just give out info on how many doc_ids in cache and are loaded
                logger.info(
                    f"Loaded graph_sage index with {len(self.indexes['doc_ids'])} embeddings"
                )
                return

            logger.info(
                "Indexes which contains the embeddings was not in the cache, so cache has to be rebuilt."
            )

        logger.info("Building graph_sage embedding indexes...")

        texts = []
        # clear out old values just in case when rebuilding indexes
        self.indexes["embeddings"] = None
        self.indexes["doc_ids"].clear()
        self.indexes["id_to_index"].clear()
        self.indexes["id_to_dataset"].clear()

        for doc_id, doc in documents.items():
            dataset_key = self._dataset_key_for_doc(doc)
            if dataset_key is None:
                continue

            index_id = self._index_id_for_doc(doc_id, doc, dataset_key)
            if not index_id:
                continue

            text = prepare_text(
                doc, include_title=True, include_labels=self.include_labels
            )

            self.indexes["id_to_index"][index_id] = len(self.indexes["doc_ids"])
            self.indexes["doc_ids"].append(index_id)
            self.indexes["id_to_dataset"][index_id] = dataset_key
            texts.append(text)

        logger.info(f"Encoding {len(texts)} documents...")
        self.indexes["embeddings"] = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )

        # save it to the cache path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "indexes": self.indexes,
                "include_labels": self.include_labels,
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info(
            "Graph_sage index built with " f"{len(self.indexes['doc_ids'])} documents."
        )

    def retrieve(self, query: dict, top_k: int = 10) -> List[Tuple[str, float]]:
        query = prepare_text(
            query, include_title=True, include_labels=self.include_labels
        )
        index = self.indexes

        # get all the important info of the index cache
        embeddings = index["embeddings"]
        doc_ids = index["doc_ids"]
        id_to_index = index["id_to_index"]
        id_to_dataset = index["id_to_dataset"]

        if embeddings is None or len(doc_ids) == 0:
            return []

        # get the query_embedding
        query_embedding = self.model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        )
        # calculate the similarities
        similarities = self._cosine_scores(embeddings, query_embedding)

        # get the indices of the top_k*expansion_factor documents
        shortlist_size = min(len(doc_ids), max(top_k * self.expansion_factor, top_k))
        shortlist_indices = np.argsort(similarities)[-shortlist_size:][::-1]

        combined_scores: dict[str, dict[str, float]] = {
            "paper": {},
            "github_issue": {},
        }

        for idx in shortlist_indices:
            doc_id = str(doc_ids[idx])
            dataset = id_to_dataset.get(doc_id)

            if dataset == "paper":
                combined_scores["paper"][doc_id] = float(similarities[idx])
            elif dataset == "github_issue":
                combined_scores["github_issue"][doc_id] = float(similarities[idx])

        paper_graph_scores = self._graph_neighbor_scores(
            "paper", combined_scores["paper"]
        )
        github_graph_scores = self._graph_neighbor_scores(
            "github_issue", combined_scores["github_issue"]
        )

        for doc_id, graph_score in paper_graph_scores.items():
            combined_scores["paper"][doc_id] = (
                combined_scores["paper"].get(doc_id, 0.0) + graph_score
            )

        for doc_id, graph_score in github_graph_scores.items():
            combined_scores["github_issue"][doc_id] = (
                combined_scores["github_issue"].get(doc_id, 0.0) + graph_score
            )

        final_scores = {
            **combined_scores["paper"],
            **combined_scores["github_issue"],
        }

        ranked = sorted(
            final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[:top_k]

class GNNRet(BaseMethod):
    """Graph-propagation retrieval.

    Builds a document graph where edges connect documents that share at least
    one entity (person / organisation / project / topic).  At inference time,
    K seed nodes (closest to the query) spread their relevance score through
    the graph for L rounds; a per-round mixing weight α is trained with a
    hinge loss on the training split of the supplied qrels.

    Parameters
    ----------
    model_name : str
        Sentence Transformer model for encoding documents and queries.
    K : int
        Number of seed nodes selected per propagation round — the K documents
        with the smallest current distance h[i, l-1] to the query.
    L : int
        Number of message-passing (propagation) rounds.
    O : int
        Size of the candidate set ``so`` used when computing the training loss.
        The O nodes with the lowest h[i, L] are selected.
    lr : float
        SGD learning rate for the α parameter vector.
    margin : float
        Hinge margin ``r`` in loss = max(0, r + d_y^L − d_o^L).
    epochs : int
        Number of full passes over the training qrels during ``train()``.
    max_entity_cluster : int
        Entities shared by more than this many documents are ignored when
        building edges — prevents very common labels from creating dense cliques.
    train_ratio : float
        Fraction of (valid) qrels used for training.
    val_ratio : float
        Fraction of (valid) qrels used for validation; the remainder is test.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        K: int = 5,
        L: int = 5,
        O: int = 25,
        lr: float = 0.01,
        margin: float = 0.1,
        epochs: int = 10,
        max_entity_cluster: int = 50,
        train_ratio: float = 0.10,
        val_ratio: float = 0.15,
    ):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.K = K
        self.L = L
        self.O = O
        self.lr = lr
        self.margin = margin
        self.epochs = epochs
        self.max_entity_cluster = max_entity_cluster
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.doc_ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self.embeddings: np.ndarray = None
        self.doc_norms: np.ndarray = None
        # edge_index: LongTensor [2, E] (both directions stored)
        self.edge_index: "torch.Tensor" = None
        self._documents: Dict = {}
        # alpha[l] controls blending at round l
        self.alpha = torch.nn.Parameter(torch.full((L,), 0.5))
        self.train_qrels: List[Dict] = []
        self.val_qrels: List[Dict] = []
        self.test_qrels: List[Dict] = []

    # ── Index construction ────────────────────────────────────────────────────

    def load_cache(self, cache_dir) -> bool:
        return self.embeddings is not None

    def build_index(self, documents: Dict, include_labels: bool = False) -> None:
        """Encode all documents and build the entity-shared graph."""
        cache_key = make_cache_key(documents, method="gnn_ret", model=self.model_name)
        cache_path = CACHE_DIR / f"gnn_ret_{cache_key}.joblib"

        if cache_path.exists():
            logger.info("GNNRet: loading index from cache: %s", cache_path)
            cache = joblib.load(cache_path)
            self._documents = cache["_documents"]
            self.doc_ids = cache["doc_ids"]
            self._id_to_idx = cache["_id_to_idx"]
            self.embeddings = cache["embeddings"]
            self.doc_norms = cache["doc_norms"]
            self.edge_index = torch.tensor(cache["edge_index"], dtype=torch.long)
            logger.info("GNNRet: loaded index with %d documents", len(self.doc_ids))
            return

        logger.info("GNNRet: encoding %d documents...", len(documents))
        self._documents = documents
        self.doc_ids = list(self._documents.keys())
        self._id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

        texts = [prepare_text(doc, include_title=True) for doc in self._documents.values()]
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        self.doc_norms = np.linalg.norm(self.embeddings, axis=1)

        logger.info("GNNRet: building entity-shared graph...")
        entity_to_docs: Dict[str, List[int]] = {}
        for idx, doc in enumerate(documents.values()):
            for field in ("people", "organizations", "projects", "topics"):
                for entity in doc.get("entities", {}).get(field, []):
                    if entity:
                        key = f"{field}:{entity}"
                        entity_to_docs.setdefault(key, []).append(idx) #creates a dictionary with entries eg. key:"people":"XYZ" value:[0, 1, 2 ...](doc_inds)
        label_to_docs: Dict[str, List[int]] = {}
        for idx, doc in enumerate(documents.values()):
            structured = doc.get("structured_fields", {}) or {}
            for field_type in ("categorical", "hierarchical"):
                for label in _as_text_list(structured.get(field_type)):
                    if label:
                        key = f"label:{label}"
                        label_to_docs.setdefault(key, []).append(idx)

        #builds list of edges based on shared entities and shared labels
        src_list, dst_list = [], []
        for group in (entity_to_docs, label_to_docs):
            for indices in group.values():
                if len(indices) > self.max_entity_cluster:
                    continue
                for i in indices:
                    for j in indices:
                        if i != j:
                            src_list.append(i)
                            dst_list.append(j)

        #adds citation edges
        for idx, doc in enumerate(documents.values()):
            if doc.get("source_type") == "paper":
                for entity in doc.get("relations", {}).get("explicit_related_ids", []):
                    entity_str = str(entity)
                    if entity_str in self._id_to_idx:
                        src_list.append(idx)
                        dst_list.append(self._id_to_idx[entity_str])

        if src_list:
            self.edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)

        n_edges = self.edge_index.shape[1]
        logger.info(
            "GNNRet: graph ready — %d nodes, %d directed edges",
            len(self.doc_ids), n_edges,
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "_documents": self._documents,
                "doc_ids": self.doc_ids,
                "_id_to_idx": self._id_to_idx,
                "embeddings": self.embeddings,
                "doc_norms": self.doc_norms,
                "edge_index": self.edge_index.numpy(),
                "cache_key": cache_key,
            },
            cache_path,
        )
        logger.info("GNNRet: index cached to %s", cache_path)

    # ── Qrel split ────────────────────────────────────────────────────────────

    def split_qrels(self, qrels: List[Dict]) -> None:
        """Randomly partition resolvable qrels into train / val / test."""
        import random

        valid = [
            q for q in qrels
            if str(q["query_id"]) in self._id_to_idx
            and q.get("candidate_ids")
            and all(str(c) in self._id_to_idx for c in q["candidate_ids"])
        ]
        random.shuffle(valid)
        n = len(valid)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        self.train_qrels = valid[:n_train]
        self.val_qrels = valid[n_train : n_train + n_val]
        self.test_qrels = valid[n_train + n_val :]
        logger.info(
            "Qrels split: %d train / %d val / %d test",
            len(self.train_qrels), len(self.val_qrels), len(self.test_qrels),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_h0(self, q_emb: np.ndarray) -> np.ndarray:
        norms = self.doc_norms * np.linalg.norm(q_emb) + 1e-10
        return 1.0 - np.dot(self.embeddings, q_emb) / norms

    # ── Graph propagation ─────────────────────────────────────────────────────

    def _propagate(self, h0: np.ndarray, alpha: "torch.Tensor") -> "torch.Tensor":
        """Propagate relevance distances through the graph for L rounds.

        h0    : (N,) numpy array — h[i,0] = 1 − cosine_sim(doc_i, query).
        alpha : (L,) Parameter with grad.
        Returns (N,) torch tensor hL.
        """

        N = len(self.doc_ids)
        h = torch.tensor(h0, dtype=torch.float32)

        k_seeds = min(self.K, N)
        for l in range(self.L):
            # --- select K seeds (detached — seed selection has no gradient) ---
            seed_indices = h.detach().topk(k_seeds, largest=False).indices

            # --- filter edge_index to edges whose source is a seed -----------
            seed_mask = torch.zeros(N, dtype=torch.bool)
            seed_mask[seed_indices] = True
            active_mask = seed_mask[self.edge_index[0]]   # bool [E]

            if not active_mask.any():
                break

            active_src = self.edge_index[0][active_mask]  # [E_active]
            active_dst = self.edge_index[1][active_mask]  # [E_active]

            # --- scatter_min: min arriving score at each destination ----------
            # fill_value=+inf so nodes with no message keep inf
            src_scores = h[active_src]                    # [E_active], with grad
            min_msg = src_scores.new_full((N,), float("inf"))
            min_msg = torch.scatter_reduce(
                min_msg, 0, active_dst, src_scores, reduce="amin", include_self=True
            )                                             # [N]

            # --- update nodes that received a message ------------------------
            received = torch.isfinite(min_msg)            # [N] bool
            min_msg_safe = torch.where(received, min_msg, torch.zeros_like(min_msg))

            a = alpha[l]
            update = a * h + (1.0 - a) * min_msg_safe
            mask_f = received.float()
            h = mask_f * update + (1.0 - mask_f) * h

        return h

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, qrels: List[Dict]) -> None:
        """Learn α using hinge loss over the training qrel split.

        Loss per query: max(0, margin + d_y^L − d_o^L)
            d_y^L  = mean h[i,L] for gold docs sy
            d_o^L  = mean h[i,L] for non-gold docs in top-O  (so − sy)
        """
        self.split_qrels(qrels)
        optimizer = torch.optim.SGD([self.alpha], lr=self.lr)

        N = len(self.doc_ids)
        top_o = min(self.O, N)

        # Pre-encode all training queries once (reused across epochs)
        query_emb_cache: Dict[str, np.ndarray] = {}
        for qrel in self.train_qrels:
            qid = str(qrel["query_id"])
            if qid not in query_emb_cache:
                query_text = prepare_text(self._documents[qid])
                query_emb_cache[qid] = self.model.encode(query_text, convert_to_numpy=True)

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_updates = 0

            for qrel in self.train_qrels:
                qid = str(qrel["query_id"])
                gold_ids = {str(c) for c in qrel["candidate_ids"]}

                h0 = self._compute_h0(query_emb_cache[qid])

                hL = self._propagate(h0, self.alpha)

                # top-O set (so)
                topO_indices = hL.detach().topk(top_o, largest=False).indices.tolist()
                so_ids = {self.doc_ids[i] for i in topO_indices}

                sy_idx = [self._id_to_idx[g] for g in gold_ids if g in self._id_to_idx]
                so_minus_sy_idx = [
                    self._id_to_idx[d]
                    for d in (so_ids - gold_ids)
                    if d in self._id_to_idx
                ]
                if not sy_idx or not so_minus_sy_idx:
                    continue

                dyL = hL[sy_idx].mean()
                doyL = hL[so_minus_sy_idx].mean()
                loss = torch.clamp(self.margin + dyL - doyL, min=0.0)

                if loss.grad_fn is None:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.alpha.data.clamp_(0.0, 1.0)
                total_loss += loss.item()
                n_updates += 1

            alpha_vals = [f"{v:.3f}" for v in self.alpha.tolist()]
            logger.info(
                "Epoch %d/%d  loss=%.4f  over %d queries  alpha=%s",
                epoch + 1, self.epochs, total_loss, n_updates, alpha_vals,
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query_doc: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top_k documents ranked by propagated relevance score."""
        query = prepare_text(query_doc, include_title=True)
        q_emb = self.model.encode(query, convert_to_numpy=True)
        h0 = self._compute_h0(q_emb)

        with torch.no_grad():
            hL = self._propagate(h0, self.alpha)

        actual_k = min(top_k, len(self.doc_ids))
        topk_indices = hL.topk(actual_k, largest=False).indices.tolist()
        # Invert distance → similarity so higher score means more relevant
        return [(self.doc_ids[i], float(1.0 - hL[i].item())) for i in topk_indices]