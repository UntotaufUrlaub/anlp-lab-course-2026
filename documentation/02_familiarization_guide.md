# Familiarization Guide: Basics for Semi-Structured Ticket Data Retrieval

> This guide provides an overview of foundational concepts (GNNs, RAG, GraphRAG, Information Retrieval, Metrics) and their connection to your project: **Finding related tickets in semi-structured GitHub issue data**.

## Table of Contents

1. [Information Retrieval Fundamentals](#information-retrieval-fundamentals)
2. [Graph Neural Networks (GNNs)](#graph-neural-networks-gnns)
3. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
4. [Graph RAG](#graph-rag)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Connection to Your Project](#connection-to-your-project)
7. [Key References](#key-references)

---

## Information Retrieval Fundamentals

### What is Information Retrieval?

Information retrieval (IR) is the task of finding relevant information from a collection of documents in response to a query. The core problem: **given a query, rank documents by relevance**.

**Classical IR Pipeline:**

1. **Query Processing**: Parse and normalize the user's query
2. **Document Indexing**: Build efficient indexes of documents
3. **Retrieval**: Find candidate documents (BM25, TF-IDF)
4. **Ranking**: Score and rank by relevance
5. **Result Presentation**: Return top-k results

### Traditional Approaches

#### TF-IDF (Term Frequency-Inverse Document Frequency)

- **Idea**: Documents are bags of words; relevance based on term frequency and rarity
- **Pros**: Fast, interpretable, baseline
- **Cons**: No semantic understanding, ignores word order and context

```
Score(query, doc) = Σ TF(term, doc) × IDF(term)
```

#### BM25 (Best Matching 25)

- **Idea**: Probabilistic model that improves on TF-IDF with document length normalization
- **Pros**: Strong baseline for lexical retrieval
- **Cons**: Still keyword-based, no semantic understanding

### Modern Approaches: Dense Retrieval

#### Vector Embeddings

- **Idea**: Represent text as dense vectors in high-dimensional space; similarity = proximity
- **Methods**:
  - Word embeddings: Word2Vec, GloVe, FastText
  - Sentence embeddings: Sentence-BERT, all-MiniLM-L6-v2
  - Dense retrievers: DPR (Dense Passage Retrieval), ColBERT

**Why embeddings matter for your project:**

- GitHub issues are text-rich (descriptions, comments) → perfect for semantic embeddings
- Can capture semantic similarity beyond keyword matching
- Example: "Cannot connect to database" and "DB connection error" would be similar in embedding space

```
Relevance(query, doc) = cos_similarity(embed(query), embed(doc))
```

---

## Graph Neural Networks (GNNs)

### Core Concept

GNNs are neural networks designed to learn representations from graph-structured data. They leverage both **node features** and **graph topology**.

![GNN-Overview](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1280px-Camponotus_flavomarginatus_ant.jpg)

_Source: Conceptual diagram - GNNs learn from interconnected structures_

### GNN Architecture: Message Passing

The fundamental operation in GNNs is **message passing**:

```
h^(l+1)_v = UPDATE(h^(l)_v, AGGREGATE({h^(l)_u : u ∈ N(v)}))
```

Where:

- `h^(l)_v`: hidden state of node v at layer l
- `N(v)`: neighbors of node v
- **AGGREGATE**: combine neighbor information (sum, mean, max)
- **UPDATE**: update node representation (neural network)

### Popular GNN Variants

| Model                                 | Key Feature                    | Use Case                 |
| ------------------------------------- | ------------------------------ | ------------------------ |
| **GCN** (Graph Convolutional Network) | Spectral convolution           | Node classification      |
| **GraphSAGE**                         | Sampling + aggregation         | Inductive learning       |
| **GAT** (Graph Attention Network)     | Attention weights on neighbors | Importance weighting     |
| **GIN** (Graph Isomorphism Network)   | Powerful discriminator         | Graph structure learning |

### Why GNNs for Your Project?

**GitHub issues form a natural graph:**

- **Nodes**: Issues/tickets
- **Edges**: "linked_to", "related_to", "blocks", "duplicates", etc.
- **Node features**: Issue text, labels, components
- **Graph structure**: Citations between issues, shared components/tags

**GNN Benefits:**

1. **Encode relational structure**: Learn which issues tend to be related
2. **Multi-hop reasoning**: Discover indirect relationships (e.g., "A links to B, B links to C")
3. **Flexible**: Can combine text features + graph structure

**Example Flow:**

```
GitHub Issue Graph:
  Issue #123 (text embedding) --> GCN/GAT --> Updated representation
    ├─ linked to Issue #456
    ├─ mentions component "auth"
    └─ has label "bug"
```

### Resources

- **Foundational Survey**: "A Comprehensive Survey on Graph Neural Networks" (2020)
  - https://arxiv.org/abs/1901.00596
- **Stanford Course**: https://web.stanford.edu/class/cs224w/

---

## Retrieval-Augmented Generation (RAG)

### Core Idea

RAG combines retrieval with generation: first retrieve relevant documents, then use them to generate better responses.

```
Query → [RETRIEVER] → Relevant Docs → [GENERATOR (LLM)] → Answer
```

### Traditional RAG Pipeline

1. **Offline Phase**:
   - Index all documents with embeddings
   - Build dense vector index (FAISS, Annoy, etc.)

2. **Online Phase** (at query time):
   - Embed query
   - Retrieve top-k similar documents using vector similarity
   - Pass retrieved documents + query to LLM
   - LLM generates answer conditioned on retrieved context

### Why RAG?

- **Grounding**: LLMs generate based on real documents, reducing hallucination
- **Up-to-date**: Easy to add new documents without retraining
- **Interpretability**: Can show which documents were retrieved
- **Cost**: More efficient than fine-tuning

### Connection to Your Project

For ticket retrieval, RAG principles suggest:

1. **Retrieve** similar tickets using embeddings or graph walks
2. **Use** that context to explain why tickets are related
3. Alternatively, **generate** synthetic links or summaries

---

## Graph RAG

### What is Graph RAG?

Graph RAG extends RAG by explicitly incorporating graph structure. Instead of flat document retrieval, it leverages knowledge graphs or issue graphs.

![GraphRAG-Concept](https://raw.githubusercontent.com/microsoft/graphrag/main/docs/img/graphrag_architecture.png)

_Source: Microsoft's GraphRAG architecture illustration_

### Key Components

#### 1. Graph Construction

Build a knowledge graph from unstructured text:

- Extract entities (issues, components, users, error codes)
- Extract relationships (links, mentions, co-occurrence)
- Result: Structured graph from text

**For your project:**

- **Entities**: Issues, components, error types, users
- **Relationships**: "issue_mentions_component", "issue_duplicates", "issue_caused_by_error"

#### 2. Graph-Aware Retrieval

Instead of only semantic similarity, also consider:

- **Direct connections**: Issues with explicit links
- **Proximity**: Issues within k-hops in the graph
- **Community structure**: Issues in the same cluster/component

#### 3. LLM-based Reasoning

Use LLMs to:

- Traverse the graph ("Starting from this issue, what other issues are affected?")
- Synthesize information ("Summarize all issues related to authentication")
- Generate insights ("Why are these 5 issues likely duplicates?")

### Graph RAG Advantages

✓ Captures both **semantic** (text) and **structural** (graph) information
✓ Supports **multi-hop reasoning** (not just direct similarity)
✓ More **interpretable** (can explain via graph paths)
✓ Better at finding **indirect relationships**

### Challenges

✗ Requires graph construction (entity/relation extraction)
✗ Computational cost increases with graph size
✗ Quality depends on extraction accuracy

### Connection to Your Project

**Your ideal retrieval system:**

```
GitHub Issue + Graph Structure
    ↓
Graph Construction: Extract entities & relationships
    ↓
Query: "Find issues related to #123"
    ↓
Multi-Strategy Retrieval:
  • Semantic similarity (text embedding)
  • Direct links (explicit connections)
  • Graph proximity (k-hop neighbors)
  • Community detection (component-based clustering)
    ↓
Ranked Results: Top-k related issues
```

### Key Papers

- **GraphRAG**: "Query 2.0 - GraphRAG by Microsoft Research"
  - https://arxiv.org/abs/2408.08921
- **LLM + Graph Navigation**: "LLMs as Autonomous Agents for Graph Exploration"
  - https://arxiv.org/abs/2407.09777

---

## Evaluation Metrics

### Ranking Metrics

When you retrieve a list of items, how do you measure quality?

#### 1. Precision@k

**What fraction of top-k results are relevant?**

```
Precision@k = (# relevant in top-k) / k
```

- Example: If you retrieve 5 issues and 3 are actually related → Precision@5 = 0.6
- Use when: False positives are costly

#### 2. Recall@k

**What fraction of all relevant items did you retrieve?**

```
Recall@k = (# relevant in top-k) / (total # relevant items)
```

- Example: If there are 10 related issues total and you retrieve 4 in top-5 → Recall@5 = 0.4
- Use when: Missing relevant items is costly

#### 3. Mean Reciprocal Rank (MRR)

**How highly ranked is the first relevant item?**

```
MRR = (1/N) Σ 1/rank_first_relevant
```

- Example: If first relevant issue is at rank 3 → contributes 1/3
- Use when: Position of first relevant result matters

#### 4. Normalized Discounted Cumulative Gain (NDCG)

**How good are the rankings, considering position?**

```
DCG@k = Σ (rel_i / log2(i+1))
NDCG@k = DCG@k / IDCG@k  (ideal ranking)
```

- Downweights items that appear later
- Ranges 0-1 (1 = perfect ranking)
- Use when: You have graded relevance (not just relevant/irrelevant)

#### 5. Mean Average Precision (MAP)

**Average precision across different recall levels**

```
MAP = (1/Q) Σ AP_q
AP_q = Σ P(k) × Δrecall(k)
```

- Comprehensive metric combining precision at all recall levels
- Ranges 0-1

### Which metrics for your project?

| Metric          | Why Use It                                          | Context                                   |
| --------------- | --------------------------------------------------- | ----------------------------------------- |
| **Precision@5** | Avoid showing wrong suggestions to users            | "Top 5 related issues" feature            |
| **Recall@20**   | Ensure you find most related issues                 | Bundling issues for duplicate detection   |
| **NDCG**        | Rank quality (some issues more related than others) | Ranking by relevance degree               |
| **MRR**         | Speed of finding first match                        | "Is the most similar issue ranked first?" |

### Implementation Example (for your project)

```python
def evaluate_retrieval(query_issues, retrieved_ranking, true_related):
    """
    query_issues: The issue you're querying
    retrieved_ranking: Ranked list of retrieved issues
    true_related: Ground truth related issues (from manual annotation)
    """
    retrieved_set = set(retrieved_ranking[:5])  # top-5
    true_set = set(true_related)

    # Metrics
    precision = len(retrieved_set & true_set) / len(retrieved_set)
    recall = len(retrieved_set & true_set) / len(true_set)

    # First relevant rank
    mrr = 0
    for rank, issue in enumerate(retrieved_ranking, 1):
        if issue in true_set:
            mrr = 1 / rank
            break

    return {
        "precision@5": precision,
        "recall@5": recall,
        "mrr": mrr
    }
```

### References

- **Metrics Survey**: https://www.ibm.com/think/topics/information-retrieval
- **NDCG Explanation**: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

---

## Connection to Your Project

### The Problem You're Solving

**Task**: Given a GitHub issue, find other related issues (duplicates, related bugs, blocking issues, etc.)

**Why it's hard:**

1. **Semi-structured**: Issues have text (descriptions, comments) + structure (links, labels, components)
2. **Heterogeneous relationships**: "duplicates", "blocks", "is-caused-by", "related-to", etc.
3. **Semantic gaps**: Issues can be related despite different wording
4. **Scale**: Thousands to millions of issues with complex interconnections

### Your Hybrid Approach

```
┌─────────────────────────────────────────────────┐
│     GitHub Issue Corpus + Relationship Graph     │
└─────────────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ↓             ↓             ↓
    ┌────────┐    ┌────────┐    ┌────────┐
    │ Dense  │    │  Graph │    │ Hybrid │
    │ Embeds │    │  Based │    │ Methods│
    │(RAG)   │    │ (GNN)  │    │(GR-RAG)│
    └────────┘    └────────┘    └────────┘
         │             │             │
         └─────────────┼─────────────┘
                       ↓
          ┌────────────────────────┐
          │  Ranking & Evaluation  │
          │  (NDCG, Recall, etc.)  │
          └────────────────────────┘
                       ↓
          ┌────────────────────────┐
          │   Related Issues       │
          │   (Ranked by relevance)│
          └────────────────────────┘
```

### Specific Connections

#### Information Retrieval

- Use **embeddings** to find semantically similar issues
- Use **BM25** as strong baseline
- Use **metrics** (NDCG, recall) to evaluate which method works best

#### GNNs

- **Graph**: Issue mentions → component graph → component similarity
- **Graph**: Issue links → direct relationships → propagate signals
- **Model**: GAT or GraphSAGE to learn which relationships matter
- **Output**: Issue embeddings that encode graph structure

#### RAG

- **Retrieve** similar issues + retrieve related issues via graph walks
- **Use**: Retrieved issues as context for explaining why they're related
- **Alternative**: Use LLM to generate synthetic relationships

#### Graph RAG

- **Best approach**: Combine all three
- **Query**: "What issues are related to #123?"
- **Process**:
  1. Extract entities/relations from #123
  2. Retrieve similar issues semantically
  3. Retrieve neighbor issues in graph
  4. Use GNN to score combined evidence
  5. Rank final results by relevance

---

## Key References

### Foundational Papers

#### Information Retrieval & Dense Retrieval

1. **Dense Passage Retrieval (DPR)** - Karpukhin et al., EMNLP 2020
   - https://arxiv.org/abs/2004.04906
   - _Foundational work on dense retrieval with neural networks_

2. **Information Retrieval Overview** - IBM
   - https://www.ibm.com/think/topics/information-retrieval

#### Graph Neural Networks

3. **A Comprehensive Survey on Graph Neural Networks** - Zhou et al., 2020
   - https://arxiv.org/abs/1901.00596
   - _Complete overview of GNN architectures and applications_

4. **Graph Attention Networks** - Velicković et al., ICLR 2018
   - https://arxiv.org/abs/1706.03762
   - _Attention mechanism for graphs_

5. **GraphSAGE** - Hamilton et al., NIPS 2017
   - https://arxiv.org/abs/1706.02216
   - _Inductive representation learning on large graphs_

#### Graph RAG & Combined Approaches

6. **Query2Vec: Graph RAG** - Microsoft Research, 2024
   - https://arxiv.org/abs/2408.08921
   - _Integrating LLMs with knowledge graphs_

7. **LLMs as Autonomous Agents for Graph Exploration** - 2024
   - https://arxiv.org/abs/2407.09777
   - _Using LLMs to traverse and reason over graphs_

8. **Graph-enhanced LLMs for Multi-hop Question Answering** - 2024
   - https://arxiv.org/abs/2501.00309

#### Ticket/Issue Related Work

9. **Ticket Resolution with GNNs** - Recent Survey
   - https://arxiv.org/abs/2505.23419
   - _Direct application to your use case_

### Recommended Learning Path

1. **Week 1-2: IR & Embeddings**
   - Read: DPR paper
   - Implement: Basic BM25 retrieval + embedding similarity
   - Resource: IBM IR overview

2. **Week 2-3: GNNs**
   - Read: GNN Survey (Sections 1-3)
   - Implement: Simple GCN on toy graph
   - Experiment: Different aggregation functions

3. **Week 3-4: Graph RAG**
   - Read: GraphRAG + Graph exploration papers
   - Understand: Entity extraction, graph construction
   - Design: How to build issue graph

4. **Week 4+: Project-Specific Implementation**
   - Combine all approaches
   - Build benchmark
   - Evaluate with metrics

### Tools & Libraries

- **Dense Retrieval**: PyTorch, HuggingFace Transformers
- **Graph Construction**: spaCy (NER), NetworkX
- **GNNs**: PyTorch Geometric, DGL
- **Embeddings**: sentence-transformers, all-MiniLM-L6-v2
- **Evaluation**: scikit-learn, RanKLists

### Additional Resources

- **Stanford CS224W (Graph ML Course)**: https://web.stanford.edu/class/cs224w/
- **Connected Papers** (find related papers): https://www.connectedpapers.com/
- **Semantic Scholar** (AI-powered search): https://www.semanticscholar.org/

---

## Next Steps

Once you've reviewed this guide:

1. **Understand the fundamentals** - Read at least one foundational paper from each section
2. **Identify your dataset** - Collect GitHub issues (or choose dataset from project description)
3. **Plan your graph** - Decide what nodes/edges represent in your issue graph
4. **Design your pipeline** - Sketch how you'll combine IR, GNNs, and Graph RAG
5. **Set up evaluation** - Define how you'll collect ground truth labels and choose metrics

---

_Last Updated: 2026-04-25_
_This guide is part of the "Exploring GNNs and other methods for retrieval of semi-structured ticket data" project_
