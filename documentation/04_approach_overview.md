# Main Questions

1. Which structural signals are most useful for improving retrieval?

2. Can learned graph propagation further refine retrieval?
   
3. Do retrieval improvements generalize across different semi-structured domains?


# Approach

## 1. Text Retrieval

Examples:
- BM25
- Sentence Transformers
- DPR (maybe less practical as baseline)

Use as baseline on how well does pure text retrieval work:

1.1 BM25 as sparse retrieval

1.2  Dense retrieval
  - use Sentence Transformers: https://huggingface.co/sentence-transformers
  - Example models:  
      all-MiniLM-L6-v2  
      bge-small-en  
      e5-base  
      gte-large    
  - use for embeddings:    
    - GitHub Issues Possible embeddings:    
        title only  
        title + body  
        title + labels  
        title + first comments  
    - Papers Possible embeddings:   
        title only  
        title + abstract  
        abstract chunks  
        citation context   
        embedding model for scientific papers: https://huggingface.co/allenai/specter2
  - cosine-similarity for ranking
  - select top-k

->see if title only or title with additional information works better  
->can fuse embeddings or embedded both in one

-----------------------------------------------------------------------------------------------

## 2. Metadata-Aware Retrieval

Examples:
- multi-field embeddings
- label-aware embeddings
- hierarchical retrieval

Answer Question 1: Does metadata/hierarchy improve retrieval?

2.1 Hierarchy-aware representation learning  
  - learn embeddings that preserve hierarchy
  - Hierarchy Paper: https://arxiv.org/abs/2509.16411

2.2 Hierarchy-aware indexing/navigation
  - use hierarchy itself as retrieval structure
  - Cobweb Paper: https://arxiv.org/abs/2510.02539
  - depends on having good hierarchies in the data, not really given in the paper dataset, but could be added
    with querying an LLM

-----------------------------------------------------------------------------------------------

## 3. Graph Expansion
 
Examples:
- SAGE
- citation expansion
- issue-link expansion

Answer Question 1: Does including local graph structure improve retrieval?

3.1 SAGE Paper:https://arxiv.org/abs/2503.01713
  - build graph offline:  
    - example edges:
        - GitHub    
        shared labels  
        same repository  
        same PR fix  
        duplicate link  
        shared error type  
        -  Papers  
        citation edge  
        shared author  
        same venue  
        same research group  
        topic overlap
  - Prune graph to keep only most meaningful relations to create sparse high quality graph (top-N neighbors/ similarity
    threshold)
  - do dense retrieval to find seed nodes in graph for query
  - do graph expansion of neighbours of seeds, but only meaningful ones
  - rerank using dense similarity/sparse similarity/ graph signals
-> uses graph during retrieval

3.2 Struc-Emb Paper: https://arxiv.org/abs/2510.08774
  - encode node + neighborhood context in offline phase and have to choose what counts as neighbour (top-N/ similarity
    threshold/ radius based)
  - Sequential concatenation: concatenate all to create embedding with neighborhood information
  - store vectors
  - normal dense retrieval
-> uses graph during encoding
-----------------------------------------------------------------------------------------------

## 4. Graph Neural Refinement

Examples: 
- GNN-Ret
- RGNN-Ret

Answer Question 2: Can learned graph propagation further refine retrieval?

Pipeline: https://arxiv.org/abs/2406.06572
- query and query embedding
- embed documents: e_text = Encoder(description) + e_hier = HierEmbed(label_path)
- do dense retrieval and select top-k candidates
- build subgraph
- do n-layer GNN
- update representations
- get relevance scores
- re-rank candidates

-----------------------------------------------------------------------------------------------

## 5. Additional Interesting Methods

5.1 LLM-based metadata extraction
- FastRAG: https://arxiv.org/abs/2411.13773






