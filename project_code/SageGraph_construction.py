import json
from abc import abstractmethod, ABC
from collections import defaultdict
from itertools import combinations

import networkx as nx
import pickle
import logging
from pathlib import Path

# --------------------------------- logger ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)



"""
loads the whole dataset and returns the documents of the specified dataset
dataset can be "github" or "paper"
"""
def load_entries(path_of_joint_dataset, dataset: str):
    docs = []
    with open(path_of_joint_dataset, "r", encoding="utf-8") as f:
        logger.info(f"Loading {dataset} dataset...")
        for line in f:
            doc = json.loads(line)

            if dataset == "paper":
                if doc.get("source_type") and doc.get("source_type") == "paper":
                    docs.append(doc)
            if dataset == "github":
                if doc.get("source_type") and doc.get("source_type") == "github_issue":
                    docs.append(doc)

    return docs

# helper function to extract values of corpus
def normalize_values(value):
    if value is None:
        return []

    if isinstance(value, dict):
        values = []
        for nested_value in value.values():
            values.extend(normalize_values(nested_value))
        return values

    if isinstance(value, list):
        values = []
        for nested_value in value:
            values.extend(normalize_values(nested_value))
        return values

    if isinstance(value, str):
        if not value.strip():
            return []
        if "|" in value:
            return [part.strip() for part in value.split("|") if part.strip()]
        return [value.strip()]

    return [value]


def get_values(entry, field_path):
    value = entry

    for key in field_path:
        if not isinstance(value, dict):
            return []
        value = value.get(key)

    return normalize_values(value)


def get_hierarchical_paths(entry):
    hierarchical = entry.get("structured_fields", {}).get("hierarchical", {})

    if isinstance(hierarchical, dict):
        paths = hierarchical.values()
    else:
        paths = normalize_values(hierarchical)

    normalized_paths = []
    for path in paths:
        if isinstance(path, str):
            levels = [level.strip() for level in path.replace("\\", "/").split("/") if level.strip()]
        elif isinstance(path, list):
            levels = [str(level).strip() for level in path if str(level).strip()]
        elif isinstance(path, dict):
            levels = [str(level).strip() for level in path.values() if str(level).strip()]
        else:
            continue

        if levels:
            normalized_paths.append(levels)

    return normalized_paths


def add_or_update_edge(graph, source_id, target_id, relation, weight):
    if source_id == target_id:
        return

    edge_data = graph.get_edge_data(source_id, target_id, default={})
    for edge_key, attributes in edge_data.items():
        if attributes.get("relation") == relation:
            attributes["weight"] = attributes.get("weight", 0.0) + weight
            return

    graph.add_edge(
        source_id,
        target_id,
        relation=relation,
        weight=weight,
    )
#----------------------- adding function --------------------------------------------------------------------------

# add one node per paper_entry/github_entry to the graph
def add_document_nodes(graph, entries, text_attribute="main_text"):
    entry_ids = set()

    for entry in entries:
        entry_id = entry.get("id")
        if entry_id is None:
            continue

        entry_ids.add(entry_id)
        graph.add_node(
            entry_id,
            title=entry.get("title", ""),
            source_dataset=entry.get("source_dataset", ""),
            source_type=entry.get("source_type", ""),
            main_text=entry.get("main_text", ""),
            secondary_texts=entry.get("secondary_texts", []),
            structured_fields=entry.get("structured_fields", {}),
            entities=entry.get("entities", {}),
            relations=entry.get("relations", {}),
            retrieval_metadata=entry.get("retrieval_metadata", {}),
            raw_source=entry.get("raw_source", {}),
        )

        if text_attribute != "main_text":
            graph.nodes[entry_id][text_attribute] = entry.get("main_text", "")

    return entry_ids

# add related edges
def add_explicit_relation_edges(graph, entries, valid_ids, relation, weight=1.0):
    edge_count = 0

    for entry in entries:
        source_id = entry.get("id")
        if source_id not in valid_ids:
            continue

        related_ids = entry.get("relations", {}).get("explicit_related_ids", [])
        for target_id in related_ids:
            target_id = str(target_id)
            if target_id in valid_ids:
                add_or_update_edge(graph, source_id, target_id, relation, weight)
                edge_count += 1

    return edge_count


def add_shared_field_edges(graph, entries, fields, weight=0.5, max_bucket_size=200):
    edge_count = 0

    for field_name, field_path in fields.items():
        buckets = defaultdict(list)

        for entry in entries:
            entry_id = entry.get("id")
            if entry_id is None:
                continue

            for value in get_values(entry, field_path):
                buckets[str(value)].append(entry_id)

        for _, entry_ids in buckets.items():
            unique_ids = sorted(set(entry_ids))

            # avoid huge noisy cliques
            if len(unique_ids) > max_bucket_size:
                continue

            for source_id, target_id in combinations(unique_ids, 2):
                add_or_update_edge(
                    graph,
                    source_id,
                    target_id,
                    relation=f"share {field_name}",
                    weight=weight,
                )
                add_or_update_edge(
                    graph,
                    target_id,
                    source_id,
                    relation=f"share {field_name}",
                    weight=weight,
                )
                edge_count += 2

    return edge_count


def add_hierarchical_path_edges(graph, entries, weight=0.75, max_bucket_size=200):
    buckets = defaultdict(list)
    edge_count = 0

    for entry in entries:
        entry_id = entry.get("id")
        if entry_id is None:
            continue

        for path in get_hierarchical_paths(entry):
            for level_index in range(1, len(path) + 1):
                prefix = " / ".join(path[:level_index])
                buckets[prefix].append(entry_id)

    for _, entry_ids in buckets.items():
        unique_ids = sorted(set(entry_ids))

        if len(unique_ids) > max_bucket_size:
            continue

        for source_id, target_id in combinations(unique_ids, 2):
            add_or_update_edge(
                graph,
                source_id,
                target_id,
                relation="share hierarchical path",
                weight=weight,
            )
            add_or_update_edge(
                graph,
                target_id,
                source_id,
                relation="share hierarchical path",
                weight=weight,
            )
            edge_count += 2

    return edge_count

class SAGEGraph(ABC):
    def __init__(self, graph_name):
        self.graph = nx.MultiDiGraph()

        GRAPH_DIR = Path("graphs")
        GRAPH_DIR.mkdir(exist_ok=True)

        self.save_path = GRAPH_DIR / f"{graph_name}.pkl"

    @abstractmethod
    def build_graph(self, doc_entries):
        pass

    def save_graph(self,):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def load_graph(self):
        with open(self.save_path, "rb") as f:
            return pickle.load(f)

class PaperGraph(SAGEGraph):
    """
    builds the networkx graph corresponding to the dataset
    """

    def __init__(self, graph_name):
        super().__init__(graph_name)

    def build_graph(self, paper_entries):

        logger.info(f"Building paper_graph nodes with {len(paper_entries)} documents...")

        # 1. make document nodes
        paper_ids = add_document_nodes(self.graph, paper_entries, text_attribute="abstract")

        logger.info(f"Built {len(paper_ids)} paper_graph nodes.")

        # 2. make citation relation edges with normal weight 1.0
        logger.info(f"\nBuilding graph citation relations with {len(paper_ids)} papers...")

        # 2. add relation edges
        citation_edges = add_explicit_relation_edges(
            self.graph,
            paper_entries,
            paper_ids,
            relation="citation",
            weight=1.0,
        )
        logger.info(f"Added {citation_edges} citation edges.")

        # 3. Add metadata edges efficiently using buckets
        logger.info(f"Building graph relations using metadata with {len(paper_ids)} papers...")
        fields = {
            "venue_name": ["structured_fields", "categorical", "venue_name"],
            "venue_type": ["structured_fields", "categorical", "venue_type"],
            "journal": ["structured_fields", "categorical", "journal"],
            "fields_of_study": ["structured_fields", "categorical", "fields_of_study"],
            "publication_types": ["structured_fields", "categorical", "publication_types"],
            "topics": ["entities", "topics"],
            "authors": ["entities", "people"],
            "organizations": ["entities", "organizations"],
        }

        metadata_edges = add_shared_field_edges(self.graph, paper_entries, fields)
        logger.info(f"Added {metadata_edges} paper metadata edges.")

        # 4. add hierarchical edges
        hierarchical_edges = add_hierarchical_path_edges(self.graph, paper_entries)
        logger.info(f"Added {hierarchical_edges} paper hierarchical edges.")

        return self.graph

class GitHubGraph(SAGEGraph):
    def __init__(self, graph_name):
        super().__init__(graph_name)

    def build_graph(self, github_entries):
        logger.info(f"Building github_graph nodes with {len(github_entries)} documents...")
        github_ids = add_document_nodes(self.graph, github_entries)

        logger.info(f"Building linked issue relations with {len(github_ids)} issues...")
        linked_issue_edges = add_explicit_relation_edges(
            self.graph,
            github_entries,
            github_ids,
            relation="linked_issue",
            weight=1.0,
        )
        logger.info(f"Added {linked_issue_edges} linked issue edges.")

        logger.info(f"Building GitHub graph relations using metadata with {len(github_ids)} issues...")
        fields = {
            "repository": ["source_dataset"],
            "labels": ["structured_fields", "categorical"],
            "projects": ["entities", "projects"],
            "people": ["entities", "people"],
            "organizations": ["entities", "organizations"],
            "topics": ["entities", "topics"],
        }

        metadata_edges = add_shared_field_edges(self.graph, github_entries, fields)
        logger.info(f"Added {metadata_edges} GitHub metadata edges.")

        hierarchical_edges = add_hierarchical_path_edges(self.graph, github_entries)
        logger.info(f"Added {hierarchical_edges} GitHub hierarchical edges.")

        return self.graph
