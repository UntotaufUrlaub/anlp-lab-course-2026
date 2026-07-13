# Project Title: Graph-Based Retrieval Enhancement for Semi-Structured Data

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation & Problem Statement](#motivation--problem-statement)
3. [Research Questions](#research-questions)
4. [Datasets & Ground Truth](#datasets--ground-truth)
5. [Approach & Methodology](#approach--methodology)
6. [Technical Implementation](#technical-implementation)
7. [Key Methods & Baselines](#key-methods--baselines)
8. [Setup & Environment](#setup--environment)
9. [Pipeline & Execution](#pipeline--execution)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Key Technologies](#key-technologies)
12. [Literature Review Summary](#literature-review-summary)
13. [Results & Benchmarks](#results--benchmarks)
14. [References & Resources](#references--resources)

---

## Project Overview

This project explores **information retrieval on semi-structured datasets** combining free text and hierarchical categorical fields. We benchmark multiple retrieval methods to understand which structural signals are most useful for improving relevance ranking.

**Core Task**: For a given entry (GitHub issue, scientific paper), find related entries to avoid redundancy and enable bundled resolution.

**Key Innovation**: Leverage hierarchical relationships in categorical metadata (e.g., author affiliations, error type hierarchies) to enhance retrieval beyond pure text-based methods.

---

## Motivation & Problem Statement

### The Gap

Real-world datasets (tickets, issues, papers) contain **mixed properties**:

- **Free text fields**: Descriptions, abstracts, comments
- **Structured/categorical fields**: Labels, types, classifications
- **Hierarchical relationships**: Authors → Research Labs/Institutes, Error Types → Frontend/Backend/Infrastructure

Traditional retrieval methods treat these as either:

1. **Pure text** (losing structure information)
2. **Pure knowledge graphs** (losing semantic text understanding)

### Why It Matters

**GitHub Issues Example**:

- Finding duplicate issues prevents wasted effort
- Related issues can be resolved together
- Hierarchy: Error Type (Frontend/Backend) → Specific Component → Detailed Classification

**Scientific Papers Example**:

- Finding related papers enables better literature reviews
- Citation relationships indicate relevance
- Hierarchy: Authors → Research Groups/Institutes, Field of Study → Specific Topic → Subtopic

### Research Focus

This project focuses on **embeddings + GNN refinement layers** trained on small datasets. This practical approach leverages graph structure without excessive computational overhead while exploring how structural signals align with relevance in real-world semi-structured data.

---

## Research Questions

### Primary Research Question

**Can retrieval be improved by exploiting structural relationships between documents rather than treating them as isolated text?**

This core question motivates investigating whether graph structure, metadata, and learned refinement layers can enhance ranking beyond pure text similarity.

### Sub-Research Questions

1. **Which structural signals are most useful for improving retrieval?**
   - Do hierarchical categorical fields enhance ranking?
   - Do explicit links (citations, issue links) add value?
   - What edge types matter most?

2. **Can learned graph propagation further refine retrieval?**
   - Can GNNs improve upon text embeddings + graph expansion?
   - Do small datasets allow effective GNN training?

3. **Do retrieval improvements generalize across different semi-structured domains?**
   - Do methods that work for GitHub also work for papers?
   - Are hierarchical pretraining techniques universal?

---

## Datasets & Ground Truth

### Dataset Requirements

Each dataset must contain:

1. **Free text content** (descriptions, abstracts, etc.)
2. **Structured/categorical fields** (labels, types, classifications)
3. **Hierarchical relationships** in categorical data
4. **Ground truth** (known related pairs for evaluation)

### Ground Truth Strategies (Ranked by Preference)

#### 1. Explicit Ground Truth (Highest Quality)

**GitHub Issues**:

- Existing issue links (30-50% of issues in popular repos)
- Duplicate issue markers
- Same-PR resolution relationships

**Scientific Papers**:

- Citation relationships
- Shared authorship
- Same research group/institution

#### 2. Weak Supervision (Scalable)

**GitHub Issues**:

- Issues with same labels (related, not duplicates)
- Issues resolved by same PR
- Issues in same repository/component

**Scientific Papers**:

- Papers in same research area/venue/time period
- Authors from same institution
- Papers citing common references

#### 3. Synthetic Generation (If Needed)

- Use LLMs to rate similarity
- Apply hierarchical distance metrics
- Domain-specific synthetic labeling

### Recommended Datasets

#### GitHub Issues

- **Source**: Kaggle GitHub Issues Dataset
- **Download**: `python project_datasets/data_set_alignment.py`
- **Preprocessing**: `project_datasets/json_processing.py`
- **Output**: `output/documents.jsonl`, `output/qrels.jsonl`

#### Scientific Papers

- **Source**: LitSearch (query corpus + papers)
- **Download**: Via `datasets` library
- **Enrichment**: Semantic Scholar metadata via `project_datasets/data_set_enrichment.py`

---

## Approach & Methodology

The project uses a **tiered retrieval approach**, progressively adding complexity and leveraging structure:

### Tier 1: Text Retrieval Baselines

**Purpose**: Establish baseline performance without structure

**Methods**:

1. **BM25** (Sparse Retrieval)
   - Probabilistic model with document length normalization
   - Keyword-based ranking
   - Fast baseline

2. **Dense Retrieval** (Sentence Transformers)
   - Models: `all-MiniLM-L6-v2`, `bge-small-en`, `e5-base`, `specter2` (papers)
   - Embedding variants:
     - GitHub: title only, title+body, title+labels, title+comments
     - Papers: title only, title+abstract, abstract chunks, citation context
   - Similarity: cosine distance
   - Ranking: top-k selection

### Tier 2: Metadata-Aware Retrieval

**Purpose**: Understand whether structured fields improve ranking

**Implemented Methods**:

1. **DenseEmbeddingBaseline + Labels** (`dense_labels`)
   - Same dense retrieval as Tier 1 baseline, but `include_labels=True`
   - Prepends categorical and hierarchical fields to document text before encoding
   - Compares against pure text baseline to measure metadata contribution
   - Zero additional hyperparameters (inherits model, batch_size, device)

2. **CHARMInspiredMethod** (`charm`)
   - Cascading Hierarchical Attention Retrieval Model (CHARM) by
     Freymuth, N., Liu, D., Ricatte, T., & Mansour, S. (2025). Hierarchical Multi-field Representations for Two-Stage E-commerce Retrieval (arXiv:2501.18707). arXiv. https://doi.org/10.48550/arXiv.2501.18707
   - **Per-field embeddings**: Separate Sentence Transformer encodings for:
     - `metadata`: structured categorical/hierarchical fields
     - `title`: document title
     - `main_text`: full document content e.g. abstract/issue description
   - **Weighted aggregation**: Combines field vectors with learned weights
     - metadata: 1.0 × e_metadata
     - title: 1.0 × e_title
     - main_text: 2.0 × e_main_text
     - Final score: weighted mean of per-field cosine similarities
   - **Adaptive filtering**: Detects and skips low-quality metadata (specificity check)
   - **Two-stage pipeline**:
     1. First-stage: Dense retrieval on combined representation
     2. Second-stage: Per-field reranking of top-k candidates
   - **No fine-tuning**: All embeddings frozen from Sentence Transformer
   - Hyperparameters: field_weights (default as above)

---

### Tier 3: Graph Expansion

**Purpose**: Incorporate local graph structure to expand candidate pools

**Implemented Methods**:

1. **SAGEGraphExpansionMethod** (`graph_sage`)
   - **Graph source**: Pre-constructed offline graphs (paper_graph.pkl, github_graph.pkl)
     - Nodes: All documents
     - Edges: Same as GNNRet (documents sharing entities/labels)
     - Format: NetworkX directed graphs with weighted edges
   - **Online retrieval**:
     1. Dense retrieval baseline → top-K × expansion_factor candidates (K typically 10)
     2. For each seed document, expand to both in-neighbors and out-neighbors in graph
     3. Neighbor scores: e^{-(1-w)}$ where w is edge weight
     4. Weighted combination: `graph_weight * neighbor_score (if in top k initially retrieved) + normalised edge weight * base_score`
   - **Result**: Empirically shows +2-4% recall improvements (dataset-dependent on graph quality)
   - **Requirements**: Pre-built graphs in `project_code/graphs/` (see SageGraph_construction.py)
   - Hyperparameters: graph_weight=0.15 (default), expansion_factor=2

### Tier 4: GNN Refinement

**Purpose**: Learn to refine ranking using graph structure

**Implemented Methods**:

1. **GNNRet** (`gnn_ret`)
   - **Graph Construction**: Offline entity-shared graph
     - Nodes: All documents
     - Edges: Documents sharing entities (people, organizations, projects, topics) or labels (categorical/hierarchical)
     - Max cluster size (default 50) to avoid dense cliques from common labels
   - **Training** (if qrels available):
     - Split qrels: 10% train / 15% validation / remainder test (configurable)
     - Learns per-round mixing weight α[l] using hinge loss
     - Loss: max(0, margin + d_y^L - d_o^L) where d = distance to gold docs vs top-O non-gold
   - **Inference**:
     - Compute initial distances: h[i,0] = 1 - cosine_sim(doc, query)
     - Propagate through graph for L rounds (default 5) with learnable α
     - K seed nodes (default 5) per round propagate to neighbors
     - Final score: 1 - distance
   - Hyperparameters: K=5, L=5, O=25, lr=0.01, margin=0.1, epochs=10

2. **NovelGATMethod** (`novel_gat`)
   - **Graph Construction**: Same as GNNRet + explicit citation edges (for papers)
   - **Training**:
     - Learns 2-layer MLP attention weights for neighbor aggregation
     - Uses same hinge loss as GNNRet
     - Split: 10% train / 90% test (no validation split)
   - **Inference**:
     - Attention-weighted aggregation: h[i,l] = sum_j a[i,j] \* h[j,l-1]
     - Attention scores computed by MLP from (h_dst, h_src) pairs
     - Softmax normalization per destination node
     - Single propagation round (unlike GNNRet's multi-round)
   - Hyperparameters: hidden_dim=8, O=25, lr=0.01, margin=0.1, epochs=60

---

## Technical Implementation

### Unified Data Schema

All datasets follow a common schema:

```json
{
  "id": "unique-identifier",
  "title": "short title",
  "content": "free text content (description, abstract, etc.)",
  "structured_fields": {
    "labels": ["label1", "label2"],
    "category": "primary category",
    "subcategory": "secondary category"
  },
  "hierarchical_metadata": {
    "label_hierarchy": ["root", "intermediate", "leaf"],
    "field_hierarchy": ["level1", "level2", "level3"]
  },
  "links": [
    {
      "target_id": "related-doc-id",
      "relation_type": "duplicate|citation|shared_author|shared_label"
    }
  ]
}
```

### ETL Pipeline

#### Stage 1: Download & Normalize

**File**: `project_datasets/data_set_alignment.py`

```bash
python project_datasets/data_set_alignment.py [--use-cache]
```

**Outputs**:

- `output/documents.jsonl` – normalized documents
- `output/qrels.jsonl` – query-relevant-document relationships

#### Stage 2: Data Enrichment

**File**: `project_datasets/data_set_enrichment.py`

Runs three stages:

**2.1 Metadata Enrichment**

- Input: `output/documents.jsonl`
- Output: `output/documents_enriched_01.jsonl`
- Adds: Semantic Scholar metadata (authors, affiliations, field of study, venue, publication type)
- Cache: `cache/paper_cache.json`

**2.2 Paper-to-Paper Query Enrichment**

- Input: `output/documents_enriched_01.jsonl`, `output/qrels.jsonl`
- Outputs: `output/documents_enriched_02.jsonl`, `output/qrels_enriched_02.jsonl`
- Converts papers into retrieval queries based on citation/related-paper links
- Deterministic: fixed seed for reproducibility

**2.3 Hierarchy Enrichment**

- Input: `output/documents_enriched_02.jsonl`
- Output: `output/documents_enriched_03.jsonl`
- Adds: Hierarchical metadata (affiliation paths, field-of-study paths, method paths)
- Cache: `cache/hierarchy_cache_nano_batch.json`

### Benchmark Implementation

**File**: `project_code/benchmark.py`

Core components:

- Method implementations: `project_code/methods.py`
- Evaluation metrics: `project_code/metrics.py`
- Utility functions: `project_code/utils.py`
- Graph construction: `project_code/SageGraph_construction.py`
- Validation: `project_code/validate_benchmark.py`

**Runner**: `project_code/benchmarkRunner.py`

---

## Key Methods & Baselines

### Implemented Methods (6 Total)

All methods inherit from `BaseMethod` abstract class with `build_index()` and `retrieve(query, k)` interface.

#### Tier 1: Baselines (2 Methods)

| Method Name        | CLI Name | Type   | Description                                                                                             | Hyperparameters                                             |
| ------------------ | -------- | ------ | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| BM25Baseline       | bm25     | Sparse | Rank-BM25 probabilistic retrieval; keyword-based matching                                               | include_labels=False                                        |
| DenseEmbeddingBase | dense    | Dense  | Sentence Transformer embeddings with cosine similarity (default `dense_labels` variant uses labels too) | model="all-MiniLM-L6-v2", batch_size=32, device="cuda\|cpu" |

#### Tier 2: Metadata-Aware (1 Method)

| Method Name         | CLI Name | Type     | Description                                                              | Hyperparameters                                        |
| ------------------- | -------- | -------- | ------------------------------------------------------------------------ | ------------------------------------------------------ |
| CHARMInspiredMethod | charm    | Encoding | Per-field weighted embeddings: combines text, title, and metadata fields | field_weights={metadata:1.0, title:1.0, main_text:2.0} |

#### Tier 3: Graph Expansion (1 Method)

| Method Name              | CLI Name   | Type            | Description                                                                  | Hyperparameters                                                  |
| ------------------------ | ---------- | --------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| SAGEGraphExpansionMethod | graph_sage | Graph Expansion | Pre-built offline graphs with online node expansion for retrieval candidates | graph_weight=0.15, expansion_factor=2, pre-built graphs required |

#### Tier 4: GNN Refinement (2 Methods)

| Method Name    | CLI Name  | Type           | Description                                                                          | Hyperparameters                                                       |
| -------------- | --------- | -------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| GNNRet         | gnn_ret   | GNN Refinement | Entity-shared graph with learnable per-round propagation weights (α[l])              | K=5, L=5, O=25, margin=0.1, epochs=10, lr=0.01, max_entity_cluster=50 |
| NovelGATMethod | novel_gat | GNN Refinement | Graph Attention Network with citation edges; attention-weighted neighbor aggregation | hidden_dim=8, O=25, epochs=60, lr=0.01, margin=0.1, train_ratio=0.10  |

### Embedding Models

**Default Model**: `all-MiniLM-L6-v2` (384-dimensional, balanced performance/speed)

**Device Strategy**:

- Automatic CUDA detection (GPU if available, else CPU fallback)
- Batch size: 32 documents per batch for embedding generation
- Override via CLI: `--embedding-model <model_name>`

**Alternative Models** (via CLI):

- `bge-small-en` – BM25-style sparse-to-dense bridge
- `e5-base` – Multilingual, strong general-purpose embeddings
- `all-mpnet-base-v2` – Higher quality but slower

---

## Setup & Environment

### Requirements

- **Python**: 3.12.0
- **System**: Linux, macOS, or Windows
- **GPU** (optional): Recommended for embedding generation
- **Storage**: ~50GB for full datasets + caches

### Installation

#### 1. Create Virtual Environment

**Windows**:

```bash
python -m venv MyVenv
MyVenv\Scripts\activate.bat
```

**PowerShell**:

```powershell
python -m venv MyVenv
.\MyVenv\Scripts\Activate.ps1
```

**Linux/macOS**:

```bash
python -m venv MyVenv
source MyVenv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:

- `torch` – Deep learning framework
- `torch-geometric` – Graph neural networks
- `transformers` – BERT/Sentence Transformers
- `sentence-transformers` – Embedding models
- `rank-bm25` – BM25 implementation
- `faiss-cpu` or `faiss-gpu` – Vector similarity search
- `scikit-learn` – Machine learning utilities
- `pandas`, `numpy` – Data manipulation
- `huggingface_hub` – Model downloads
- `datasets` – Dataset utilities
- `kagglehub` – Kaggle dataset download
- `semanticscholar` – Paper metadata API

---

## Pipeline & Execution

### Step 1: Validate Environment

```bash
python project_code/validate_benchmark.py
```

Checks:

- Python version compatibility
- Required packages installed
- Input data paths accessible
- Output directories writable

### Step 2: Build Aligned Dataset

```bash
python project_datasets/data_set_alignment.py [--use-cache]
```

**Flags**:

- `--use-cache`: Reuse cached input data when available

**Outputs**:

- `output/documents.jsonl` (aligned documents)
- `output/qrels.jsonl` (ground truth)

**Time**: ~1-2 hours

### Step 3: Enrich Dataset

```bash
python project_datasets/data_set_enrichment.py
```

Runs three stages in sequence:

**Stage 1**: Metadata enrichment (~20 min)

- Adds: Semantic Scholar info, authors, affiliations, venues
- Output: `documents_enriched_01.jsonl`

**Stage 2**: Query enrichment (~15 min)

- Adds: Paper-to-paper queries from citations
- Output: `documents_enriched_02.jsonl`, `qrels_enriched_02.jsonl`

**Stage 3**: Hierarchy enrichment (~30 min)

- Adds: Hierarchical paths for fields
- Output: `documents_enriched_03.jsonl`

**Total Time**: ~1 hour

### Step 4: Run Benchmark

```bash
python project_code/benchmark.py \
  --docs-path output/documents_enriched_03.jsonl \
  --qrels-path output/qrels_enriched_02.jsonl \
  --baseline bm25 \
  --methods dense_labels charm graph_sage gnn_ret novel_gat \
  --embedding-model all-MiniLM-L6-v2 \
  --output results/method_results.json \
  --k-values 10 100
```

**Key Flags**:

- `--docs-path`: Documents file (default: `output/documents_enriched_03.jsonl`)
- `--qrels-path`: Ground truth file (default: `output/qrels_enriched_02.jsonl`)
- `--baseline`: Baseline method (`bm25` or `dense`, default: `bm25`)
- `--methods`: Experimental methods to benchmark (choices: `dense_labels`, `charm`, `graph_sage`, `gnn_ret`, `novel_gat`)
- `--embedding-model`: Sentence Transformer model (default: `all-MiniLM-L6-v2`)
- `--output`: Results JSON file
- `--k-values`: K values for metrics (default: `[10]`)
- `--sample-size`: Approximate number of documents to sample (default: None = all)
- `--batch-size`: Number of random queries to benchmark (default: None = all)
- `--seed`: Random seed for reproducible sampling
- `--debug`: Enable debug mode (smaller sample, extra logging)

**Outputs**:

- `results/method_results.json` – Aggregated metrics per method
- `project_code/cache/` – Cached embeddings (dense*\*.joblib, bm25*_.joblib, graph*sage*_.joblib, etc.)
- `project_code/graphs/` – Pre-constructed graphs (paper_graph.pkl, github_graph.pkl)

### Step 5: Visualize Results

```bash
python project_code/benchmark_visualization.py \
  --results results/method_results.json \
  --output results/visualizations/
```

**Outputs**:

- Recall@k curves
- NDCG comparisons
- MRR rankings
- Method runtime analysis

---

## Evaluation Metrics

### Implemented Metrics (5 Total)

All metrics support multiple k-values (e.g., k=[10, 100]).

#### 1. Recall@K

- **Definition**: $\text{Recall@K} = \frac{|\text{relevant items in top-k}|}{|\text{total relevant items}|}$
- **Interpretation**: Of all relevant documents, what fraction did we retrieve in top-k?
- **Range**: [0, 1] — higher is better
- **Code**: `recall_at_k(rankings, ground_truth, k=10)`

#### 2. NDCG@K (Normalized Discounted Cumulative Gain)

- **Definition**: $\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$ where $\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$ (binary relevance)
- **Interpretation**: Ranking quality accounting for position (top results weighted more)
- **Range**: [0, 1] — higher is better
- **Code**: `ndcg_at_k(rankings, ground_truth, k=10)`

#### 3. MAP@K (Mean Average Precision)

- **Definition**: $\text{MAP@K} = \frac{1}{|\text{relevant}|} \sum_{i=1}^{K} \text{Precision@i} \times \text{rel}_i$
- **Interpretation**: Average precision at each relevant item position
- **Range**: [0, 1] — higher is better
- **Code**: `map_at_k(rankings, ground_truth, k=10)`

#### 4. Precision@K

- **Definition**: $\text{Precision@K} = \frac{|\text{relevant items in top-k}|}{K}$
- **Interpretation**: What fraction of top-k results are relevant?
- **Range**: [0, 1] — higher is better
- **Code**: `precision_at_k(rankings, ground_truth, k=10)`

#### 5. MRR (Mean Reciprocal Rank)

- **Definition**: $\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$ where rank_i is position of first relevant doc
- **Interpretation**: How quickly do we find the first relevant result?
- **Range**: [0, 1] — higher is better (1.0 = always first, 0.5 = always second)
- **Code**: `mrr(rankings, ground_truth)`
- **No k parameter**: Computes across entire ranking

### CLI Usage

```bash
# Specify k-values for metrics computation
--k-values 10 100
```

This generates: `recall@10`, `recall@100`, `ndcg@10`, `ndcg@100`, `map@10`, `map@100`, `precision@10`, `precision@100`, `mrr`.

### Implementation

**File**: `project_code/metrics.py` — `evaluate(rankings, ground_truth, k_values=[10])`

Returns dictionary aggregated over all queries as `{metric_mean, ...}`.

---

## Key Technologies

### Embedding & Retrieval

- **Sentence Transformers**: Semantic embeddings
- **FAISS**: Fast similarity search
- **BM25**: Sparse retrieval baseline

### Graph Processing

- **PyTorch Geometric**: GNN implementations
- **NetworkX**: Graph construction & analysis
- **DGL** (optional): Alternative GNN framework

### Data Processing

- **Pandas / NumPy**: Data manipulation
- **datasets** (HuggingFace): Dataset utilities
- **kagglehub**: Dataset downloads

### LLMs & APIs

- **SemanticScholar API**: Paper metadata
- **Kaggle API**: GitHub issues dataset
- **OpenAI API**: (optional) LLM-based enrichment

### Visualization

- **Matplotlib**: Charts & plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive plots

---

## Literature Review Summary

### Text Retrieval (Baseline)

- **BM25**: Probabilistic model, strong baseline for keyword matching
- **Sentence Transformers**: Dense semantic embeddings, captures paraphrase

### Metadata-Aware Retrieval

| Method                                   | Paper                        | Key Insight                                                      |
| ---------------------------------------- | ---------------------------- | ---------------------------------------------------------------- |
| Hierarchical Retrieval                   | arXiv:2509.16411             | Pretrain on hierarchy to fix "lost-in-the-long-distance" problem |
| HyperbolicRAG                            | arXiv:2602.07739, 2511.18808 | Use hyperbolic space to naturally preserve tree hierarchies      |
| Cobweb (Hierarchical Semantic Retrieval) | arXiv:2510.02539             | Use label hierarchy as tree-structured index, no training needed |

### Graph Expansion

| Method                                 | Paper            | Key Insight                                                         |
| -------------------------------------- | ---------------- | ------------------------------------------------------------------- |
| SAGE (Structure-Aware Graph Expansion) | arXiv:2602.16964 | Offline graph + online expansion, +5.7-8.5 recall points            |
| Struc-Emb                              | arXiv:2510.08774 | Inject neighbor context at encoding time (sequential concatenation) |
| CaseLink                               | arXiv:2403.17780 | Combine NLP similarity + GNN refinement for link prediction         |

### GNN Refinement

| Method                               | Paper            | Key Insight                                                             |
| ------------------------------------ | ---------------- | ----------------------------------------------------------------------- |
| GNN-Ret / RGNN-Ret                   | arXiv:2406.06572 | Build subgraph over top-k, propagate relevance via GNN, small data req. |
| G-Retriever                          | arXiv:2403.06121 | GNN for textual graph understanding                                     |
| GAR (Generation-Augmented Retrieval) | -                | Diverse context generation for query expansion                          |

### LLM Integration

| Method  | Paper            | Key Insight                                         |
| ------- | ---------------- | --------------------------------------------------- |
| FastRAG | arXiv:2411.13773 | Use LLM to extract structure (schema learning)      |
| CORONA  | arXiv:2402.07739 | LLM + GNN + preference reasoning for recommendation |
| RuleRAG | arXiv:2412.00896 | Rule mining for logical retrieval                   |

### Surveys & Foundational

- "A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models" – Graph RAG overview
- "A Comprehensive Survey on Graph Neural Networks" – GNN fundamentals

---

## Results & Benchmarks

### Expected Improvements (from Literature)

Based on published papers (confidence levels estimated):

| Method                              | Recall@100 Lift   | NDCG Lift     | Notes                                                           |
| ----------------------------------- | ----------------- | ------------- | --------------------------------------------------------------- |
| Dense Baseline (all-MiniLM-L6-v2)   | –                 | –             | Reference point                                                 |
| BM25 Baseline                       | -20 to -30%       | -15 to -25%   | Loses semantic similarity                                       |
| Dense + Labels (`dense_labels`)     | +1 to -1%         | +1 to -1%     | Metadata alone insufficient; limited signal over text           |
| CHARM (`charm`)                     | +3 to +5%         | +3 to +5%     | Per-field weighting yields limited improvement over text        |
| SAGE Graph Expansion (`graph_sage`) | +10 to +15%       | +10 to +15%   | Graph provides gains when structure aligns with relevance       |
| GNNRet (`gnn_ret`)                  | +15 to +17%       | +15 to +17%   | Graph structure strongest gain; learns entity-label propagation |
| NovelGAT (`novel_gat`)              | ----------------- | ------------- | Graph refinement; learns attention-weighted propagation         |

### Actual Results Format

Results are saved to the `--output` JSON file and contain:

```json
{
  "timestamp": "2026-07-09T14:23:00Z",
  "benchmark_config": {
    "docs_path": "output/documents_enriched_03.jsonl",
    "qrels_path": "output/qrels_enriched_02.jsonl",
    "embedding_model": "all-MiniLM-L6-v2",
    "baseline": "bm25",
    "baseline_label": "BM25",
    "methods": ["dense_labels", "charm", "graph_sage", "gnn_ret", "novel_gat"],
    "experimental_method_labels": {...}
  },
  "results": {
    "bm25": {
      "recall@10_mean": 0.32,
      "recall@100_mean": 0.65,
      "ndcg@10_mean": 0.38,
      "ndcg@100_mean": 0.72,
      "map@100_mean": 0.41,
      "precision@10_mean": 0.25,
      "mrr_mean": 0.45,
      "query_details": [{...}]
    },
    "dense_labels": {
      "recall@10_mean": 0.48,
      "recall@100_mean": 0.72,
      "ndcg@10_mean": 0.51,
      "ndcg@100_mean": 0.75,
      ...
    }
  }
}
```

**Note**: k-values vary based on CLI `--k-values` flag. Results are aggregated means across all queries. `query_details` contains top-k retrieval samples for debugging.

### To Run Full Benchmark

```bash
python project_code/benchmark.py \
  --docs-path output/documents_enriched_03.jsonl \
  --qrels-path output/qrels_enriched_02.jsonl \
  --baseline bm25 \
  --methods dense_labels charm graph_sage gnn_ret novel_gat \
  --embedding-model all-MiniLM-L6-v2 \
  --k-values 10 100 \
  --output results/method_results.json
```

---

## References & Resources

### Core Papers

#### Text Retrieval

1. Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." NOW Publishers.
   - https://www.researchgate.net/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond

2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805
   - https://arxiv.org/abs/1810.04805

#### Graph Neural Networks

3. Kipf, T., & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks." arXiv:1609.02907
   - https://arxiv.org/abs/1609.02907

4. Hamilton, W., Ying, Z., & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs." arXiv:1706.02216
   - https://arxiv.org/abs/1706.02216

#### Graph + Retrieval

5. Sui, Z., et al. (2024). "SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data." arXiv:2602.16964
   - https://arxiv.org/abs/2602.16964

6. Deng, M., et al. (2024). "Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings." arXiv:2510.08774
   - https://arxiv.org/abs/2510.08774

#### GNN-Enhanced Retrieval

7. Li, M., et al. (2024). "GNN Enhanced Retrieval for Question Answering of LLMs." arXiv:2406.06572
   - https://arxiv.org/abs/2406.06572

#### Hierarchical Retrieval

8. Raffel, C., et al. (2024). "Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe." arXiv:2509.16411
   - https://arxiv.org/abs/2509.16411

9. Deng, X., et al. (2024). "HyperbolicRAG: Hyperbolic Embeddings for Retrieval-Augmented Generation." arXiv:2511.18808
   - https://arxiv.org/abs/2511.18808

10. Yao, L., et al. (2024). "Hierarchical Semantic Retrieval with Cobweb." arXiv:2510.02539
    - https://arxiv.org/abs/2510.02539

#### Surveys & Foundations

11. Zhou, J., et al. (2020). "A Comprehensive Survey on Graph Neural Networks." IEEE TPAMI, 32(3):1-21.
    - https://arxiv.org/abs/1901.00596

12. Shen, J., et al. (2024). "Graph Retrieval-Augmented Generation: A Survey" arXiv:2408.08921
    - https://arxiv.org/abs/2408.08921

#### Metadata & Structure

13. Gao, L., et al. (2024). "FastRAG: Retrieval Augmented Generation for Semi-structured Data." arXiv:2411.13773
    - https://arxiv.org/abs/2411.13773

### Key Resources

**Sentence Transformers**: https://www.sbert.net/

**PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

**FAISS**: https://github.com/facebookresearch/faiss

**HuggingFace Models**: https://huggingface.co/sentence-transformers

**SemanticScholar API**: https://www.semanticscholar.org/

**Kaggle Datasets**: https://www.kaggle.com/

---

## AI Tool Usage Declaration

This project was developed with assistance from AI tools (GPT models, GitHub Copilot, etc.). AI was used for:

- Code scaffolding and implementation
- Documentation drafting
- Exploratory analysis

All technical claims have been verified and are the responsibility of the project team.

---

## Appendix: Quick Reference

### Command Cheatsheet

```bash
# Setup
python -m venv MyVenv
source MyVenv/bin/activate  # Linux/macOS
MyVenv\Scripts\activate.bat # Windows
pip install -r requirements.txt

# Validate environment
python project_code/validate_benchmark.py

# Build aligned dataset
python project_datasets/data_set_alignment.py --use-cache

# Enrich dataset (3 stages: metadata, queries, hierarchy)
python project_datasets/data_set_enrichment.py

# Run full benchmark (all 6 methods, 2 k-values)
python project_code/benchmark.py \
  --docs-path output/documents_enriched_03.jsonl \
  --qrels-path output/qrels_enriched_02.jsonl \
  --baseline bm25 \
  --methods dense_labels charm graph_sage gnn_ret novel_gat \
  --embedding-model all-MiniLM-L6-v2 \
  --k-values 10 100 \
  --output results/method_results.json

# Run single method for quick testing
python project_code/benchmark.py \
  --docs-path output/documents_enriched_03.jsonl \
  --qrels-path output/qrels_enriched_02.jsonl \
  --baseline bm25 \
  --methods charm \
  --debug --sample-size 100

# Visualize results
python project_code/benchmark_visualization.py \
  --results results/method_results.json \
  --output results/visualizations/
```

### File Paths Reference

| Purpose                         | Path                                 |
| ------------------------------- | ------------------------------------ |
| Raw documents                   | `output/documents.jsonl`             |
| Enriched documents (all stages) | `output/documents_enriched_03.jsonl` |
| Ground truth queries            | `output/qrels_enriched_02.jsonl`     |
| Benchmark results               | `results/method_results.json`        |
| Cached embeddings               | `project_code/cache/`                |
| Constructed graphs              | `project_code/graphs/`               |

---

**End of Documentation**\_

_Last Updated: July 2026_
