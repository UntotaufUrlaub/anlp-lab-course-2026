import json
import sys
from abc import abstractmethod, ABC
from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx
import pickle
import logging
import random
from pathlib import Path

import numpy as np

# import project_datasets

# --------------------------------- logger ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
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


# helper function to extract values of corpus in nice string format
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


def extract_path_levels(path_data):
    if isinstance(path_data, dict):
        return [
            str(value).strip()
            for value in path_data.values()
            if value and str(value).strip()
        ]

    if isinstance(path_data, list):
        return [
            str(value).strip() for value in path_data if value and str(value).strip()
        ]

    if isinstance(path_data, str):
        return [
            level.strip()
            for level in path_data.replace("\\", "/").split("/")
            if level.strip()
        ]

    return []


def get_hierarchical_paths(entry):
    hierarchical = entry.get("structured_fields", {}).get("hierarchical", {})

    paths = []

    if isinstance(hierarchical, dict):
        items = hierarchical.items()
    elif isinstance(hierarchical, list):
        items = enumerate(hierarchical)
    else:
        return []

    for key, value in items:
        if key == "affiliations":
            continue

        levels = extract_path_levels(value)

        if levels:
            paths.append(levels)

    return paths


def add_or_update_edge(graph, source_id, target_id, relation, weight):
    if source_id == target_id:
        return

    if source_id not in graph or target_id not in graph:
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


# ----------------------- adding function --------------------------------------------------------------------------


# add one node per paper_entry
def add_document_nodes(graph, entries):
    # set of entries that were already visited
    entry_ids = set()

    # go through every entry
    for entry in entries:
        # we add the normal id
        entry_id = entry.get("id")

        if entry_id is None:
            continue

        # skip entries with empty title and main_text
        title = entry.get("title", "")
        main_text = entry.get("main_text", "")

        if not title.strip() or not main_text.strip():
            continue

        # add to set and graph if a valid and non-empty paper entry
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
        )

    return entry_ids


# add related edges
# valid ids are all extracted ids that have been added to the graph
def add_explicit_relation_edges(graph, entries, valid_ids, relation, weight=1.0):
    edge_count = 0

    # go through each entry and if not a valid entry skip
    for entry in entries:
        # we add the normal id
        source_id = entry.get("id")

        if source_id is None:
            continue

        # get all the related ids of this entry
        related_ids = entry.get("relations", {}).get("explicit_related_ids", [])

        # go through each id of the related
        for target_id in related_ids:
            target_id = str(target_id)
            # if we have a related entry which is valid then we can directly add the edge
            if target_id in valid_ids and source_id in valid_ids:
                add_or_update_edge(graph, source_id, target_id, relation, weight)
                edge_count += 1

    # return number of edges
    return edge_count


def add_shared_field_edges(graph, entries, fields, weight=0.5, max_bucket_size=50):
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


def add_hierarchical_path_edges(graph, entries, weight=0.75, max_bucket_size=50):
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
        self.graph_name = graph_name

        GRAPH_DIR = PROJECT_DIR / "project_code" / "graphs"
        GRAPH_DIR.mkdir(exist_ok=True)

        self.save_path = GRAPH_DIR / f"{self.graph_name}.pkl"

    @abstractmethod
    def build_graph(self, doc_entries):
        pass

    def save_graph(
        self,
    ):
        logger.info(f"Saving graph {self.graph_name} ...")
        with open(self.save_path, "wb") as f:
            pickle.dump(self.graph, f)

    def load_graph(self):
        logger.info(f"Loading graph {self.graph_name} ...")
        with open(self.save_path, "rb") as f:
            self.graph = pickle.load(f)
            return self.graph

    def inspect_graph_densities_for_expansion(self):
        logger.info(f"Inspecting graph_densities for {self.graph_name} ...")
        sample_nodes = list(self.graph.nodes())

        neighbor_counts = []
        for node in sample_nodes:
            neighbors = set(self.graph.successors(node)) | set(
                self.graph.predecessors(node)
            )
            neighbor_counts.append(len(neighbors))

        avg_neighbor_count = (
            sum(neighbor_counts) / len(neighbor_counts)
            if len(neighbor_counts) > 0
            else 0
        )

        # inspect neighborhood
        print("avg neighbors:", avg_neighbor_count)
        print("max neighbors:", max(neighbor_counts))

        degrees = [self.graph.degree(node) for node in self.graph.nodes()]

        # inspect degree distribution
        print(f"Average Degree: {sum(degrees) / len(degrees):.2f}")
        print(f"Median Degree: {np.median(degrees)}")
        print(f"95th percentile: {np.percentile(degrees, 95)}")
        print(f"99th percentile: {np.percentile(degrees, 99)}")
        print(f"Maximum Degree: {max(degrees)}")

    def inspect_empty_nodes(self):
        empty_nodes = [n for n, attrs in self.graph.nodes(data=True) if not attrs]

        print("empty nodes:", len(empty_nodes))
        print("examples:", empty_nodes[:10])

        if len(empty_nodes) != 0:
            node = empty_nodes[0]
            print("node id:", node)
            print("has paper prefix:", str(node).startswith("paper:"))

        non_empty_nodes = [n for n, attrs in self.graph.nodes(data=True) if attrs]

        print("non-empty nodes:", len(non_empty_nodes))
        print("non-empty examples:", non_empty_nodes[:10])
        print(self.graph.nodes[non_empty_nodes[0]])


class PaperGraph(SAGEGraph):
    """
    builds the networkx graph corresponding to the paper dataset
    """

    def __init__(self, graph_name):
        super().__init__(graph_name)

    def build_graph(self, paper_entries):

        logger.info(f"Building paper_graph nodes...")

        # 1. make document nodes
        paper_ids = add_document_nodes(self.graph, paper_entries)

        logger.info(f"\nAdded {len(paper_ids)} paper_graph nodes to the graph.")

        logger.info(f"\nBuilding graph citation relation edges...")

        # 2. make citation relation edges with normal weight 1.0
        citation_edges = add_explicit_relation_edges(
            self.graph,
            paper_entries,
            paper_ids,
            relation="citation",
            weight=1.0,
        )
        logger.info(f"\nAdded {citation_edges} citation edges.")

        # 3. Add metadata edges efficiently using buckets
        logger.info(
            f"\nBuilding graph relations using metadata with {len(paper_ids)} papers..."
        )
        fields = {
            "venue_name": ["structured_fields", "categorical", "venue_name"],
            "publication_types": [
                "structured_fields",
                "categorical",
                "publication_types",
            ],
            "authors": ["entities", "people"],
            # "organizations": ["entities", "organizations"],
        }

        metadata_edges = add_shared_field_edges(self.graph, paper_entries, fields)
        logger.info(f"\nAdded {metadata_edges} paper metadata edges.")

        # 4. add hierarchical edges
        logger.info(f"\nBuilding graph relations using hierarchical edges...")
        hierarchical_edges = add_hierarchical_path_edges(self.graph, paper_entries)
        logger.info(f"\nAdded {hierarchical_edges} paper hierarchical edges.")

        return self.graph

    def inspect_unusually_high_neighbor_nodes(self):
        max_neighbor_node = max(
            self.graph.nodes,
            key=lambda n: len(
                set(self.graph.successors(n)) | set(self.graph.predecessors(n))
            ),
        )

        neighbors = set(self.graph.successors(max_neighbor_node)) | set(
            self.graph.predecessors(max_neighbor_node)
        )

        print(
            "Title of node with most unique neighbors",
            self.graph.nodes[max_neighbor_node].get("title"),
        )

        print("Node_id of node with most unique neighbors:", max_neighbor_node)
        print("Unique neighbor count:", len(neighbors))
        print("degree:", self.graph.degree(max_neighbor_node))

        relations = Counter()

        for _, _, data in self.graph.out_edges(max_neighbor_node, data=True):
            relations["OUT " + data.get("relation", "unknown")] += 1

        for _, _, data in self.graph.in_edges(max_neighbor_node, data=True):
            relations["IN " + data.get("relation", "unknown")] += 1

        print(relations)


class GitHubGraph(SAGEGraph):
    def __init__(self, graph_name):
        super().__init__(graph_name)

    def build_graph(self, github_entries):

        logger.info(f"Building graph...")
        github_ids = add_document_nodes(self.graph, github_entries)

        logger.info(f"\nAdded {len(github_ids)} graph nodes to the graph.")

        logger.info(f"\nBuilding linked issue relations ...")

        linked_issue_edges = add_explicit_relation_edges(
            self.graph,
            github_entries,
            github_ids,
            relation="linked_issue",
            weight=1.0,
        )
        logger.info(f"\nAdded {linked_issue_edges} linked issue edges.")

        logger.info(f"\nBuilding GitHub graph relations using metadata...")
        fields = {
            "repository": ["source_dataset"],
            "labels": ["structured_fields", "categorical"],
            "projects": ["entities", "projects"],
            "people": ["entities", "people"],
            "organizations": ["entities", "organizations"],
            "topics": ["entities", "topics"],
        }

        metadata_edges = add_shared_field_edges(self.graph, github_entries, fields)
        logger.info(f"\nAdded {metadata_edges} GitHub metadata edges.")

        logger.info(f"\nBuilding GitHub graph relations using hierarchical data...")
        hierarchical_edges = add_hierarchical_path_edges(self.graph, github_entries)
        logger.info(f"\nAdded {hierarchical_edges} GitHub hierarchical edges.")

        return self.graph


if __name__ == "__main__":
    total_dataset_path = (
        # PROJECT_DIR / "project_datasets" / "output" / "documents_enriched_03.jsonl"
        PROJECT_DIR
        / "output"
        / "documents_enriched_03.jsonl"
    )

    # paper_graph construction
    paper_entries = load_entries(total_dataset_path, "paper")

    paper_graph = PaperGraph("paper_graph")

    paper_graph.build_graph(paper_entries)

    paper_graph.save_graph()

    # github graph construction
    github_entries = load_entries(total_dataset_path, "github")

    github_graph = GitHubGraph("github_graph")

    github_graph.build_graph(github_entries)

    github_graph.save_graph()

    # inspect graph statistics
    print("Paper_graph statistics:")
    paper_graph = PaperGraph("paper_graph")
    paper_graph.load_graph()
    print("\n")
    paper_graph.inspect_unusually_high_neighbor_nodes()
    print("\n")
    paper_graph.inspect_empty_nodes()
    print("\n")
    paper_graph.inspect_graph_densities_for_expansion()

    print("Github_graph statistics:")
    github_graph = GitHubGraph("github_graph")
    github_graph.load_graph()
    print("\n")
    github_graph.inspect_empty_nodes()
    print("\n")
    github_graph.inspect_graph_densities_for_expansion()
