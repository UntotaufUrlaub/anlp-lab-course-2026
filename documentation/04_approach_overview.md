# Main Questions

1. Which structural signals are most useful for improving retrieval?

2. Can learned graph propagation further refine retrieval?
   
3. Do retrieval improvements generalize across different semi-structured domains?


# Approach

1. Text Retrieval

Examples:
- BM25
- Sentence Transformers
- DPR

Use as baseline on how well does pure text retrieval work

2. Metadata-Aware Retrieval

Examples:
- multi-field embeddings
- label-aware embeddings
- hierarchical retrieval

Answer Question 1: Does metadata/hierarchy improve retrieval?

3. Graph Expansion
 
Examples:
- SAGE
- citation expansion
- issue-link expansion

Answer Question 1: Does including local graph structure improve retrieval?

4.  Graph Neural Refinement

Examples:
- GNN-Ret
- RGNN-Ret

Answer Question 2: Can learned graph propagation further refine retrieval?



