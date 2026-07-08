# anlp-lab-course-2026

## Overview

This repository contains a retrieval benchmark pipeline for aligning a mixed corpus of GitHub issues and scientific papers, enriching it with metadata, and evaluating retrieval methods.

## Requirements

- Python 3.12.0
- A virtual environment such as MyVenv created with `python -m venv MyVenv`

## Core pipeline files

- `project_datasets/data_set_alignment.py` — download, normalize, align, and export the base documents and qrels files
- `project_datasets/json_processing.py` — preprocess and normalize GitHub issue metadata
- `project_datasets/data_set_enrichment.py` — run the three enrichment stages over the aligned dataset
- `project_code/benchmark.py` — run retrieval benchmarks on the aligned/enriched corpus
- `project_code/methods.py` — retrieval method implementations used by the benchmark
- `project_code/utils.py` — shared CLI parsing and text preparation helpers for the benchmark
- `project_code/validate_benchmark.py` — verify input data, dependencies, and output paths
- `requirements.txt` — required Python packages

## 1. Set up the environment

### 1) Activate the virtual environment

Windows:

```bat
MyVenv\Scripts\activate.bat
```

PowerShell:

```powershell
.\MyVenv\Scripts\Activate.ps1
```

Linux/Mac:

```bash
source MyVenv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Build the aligned dataset

Run:

```bash
python project_datasets/data_set_alignment.py
```

This script will:

- download the GitHub issues dataset through `kagglehub`
- normalize and align labels with the preprocessing helpers in `project_datasets/json_processing.py`
- download the LitSearch query and corpus datasets through `datasets`
- write the base files to:
  - `output/documents.jsonl`
  - `output/qrels.jsonl`

To reuse cached input data when available, run:

```bash
python project_datasets/data_set_alignment.py --use-cache
```

## 3. Enrich the dataset

Run:

```bash
python project_datasets/data_set_enrichment.py
```

The enrichment pipeline runs three stages in order and writes progressively richer versions of the dataset:

1. Metadata enrichment
   - input: `output/documents.jsonl`
   - output: `output/documents_enriched_01.jsonl`
   - adds Semantic Scholar metadata such as authors, affiliations, fields of study, venue/journal information, and publication types
   - uses the cache file `cache/paper_cache.json`

2. Paper-to-paper query enrichment
   - input: `output/documents_enriched_01.jsonl` and `output/qrels.jsonl`
   - outputs: `output/documents_enriched_02.jsonl` and `output/qrels_enriched_02.jsonl`
   - converts a subset of papers into retrieval queries based on citation/related-paper links
   - uses a fixed seed for reproducibility

3. Hierarchy enrichment
   - input: `output/documents_enriched_02.jsonl`
   - output: `output/documents_enriched_03.jsonl`
   - adds hierarchical scientific metadata such as affiliation paths, field-of-study paths, and method paths
   - uses the cache file `cache/hierarchy_cache_nano_batch.json`

You can rerun individual stages with:

```bash
python project_datasets/data_set_enrichment.py --overwrite-metadata
python project_datasets/data_set_enrichment.py --readd-paper-to-paper
python project_datasets/data_set_enrichment.py --overwrite-hierarchy
```

The order of the three stages should be preserved.

### Notes on the enrichment outputs

- The hierarchy step may leave some entries marked with an error object when the LLM output is invalid or unrecoverable.
- The enrichment caches allow interrupted runs to resume without re-querying data that was already collected.
- A validation helper for hierarchy outputs is available in `project_datasets/paper_validation.py`.

## 4. Validate the benchmark setup

```bash
python project_code/validate_benchmark.py
```

This checks for the presence and readability of the base data files, the benchmark script, and the expected output directory.

## 5. Experimental methods and baselines

The benchmark compares a small set of retrieval strategies that differ in how they use document structure, metadata, and graph information.

### Baselines

- `BM25Baseline` — a classical lexical baseline that ranks documents by term overlap using BM25.
- `DenseEmbeddingBaseline` — a dense retrieval baseline that embeds documents and queries with a sentence-transformer model and ranks by cosine similarity.

### Experimental methods

- `CHARMInspiredMethod` — a structure-aware retrieval method that combines metadata, title, and main-text signals with weighted field embeddings and a two-stage reranking strategy.
- `SAGEGraphExpansionMethod` — a graph-enhanced retrieval method that expands retrieval scores using neighborhood information from paper and issue graphs.
- `GNNRet` — a graph-propagation retrieval method that builds an entity-shared document graph, selects the closest seed nodes to the query, and spreads relevance through the graph for L rounds using a learned per-round mixing weight

### Notes

- The benchmark can be run with different combinations of baselines and experimental methods via the `--baseline` and `--methods` flags.
- The default experiment set includes the dense baseline and the two experimental methods above.

## 6. Run the retrieval benchmark

The benchmark now defaults to the enriched dataset files defined in `project_code/utils.py`:

- documents: `output/documents_enriched_03.jsonl`
- qrels: `output/qrels_enriched_02.jsonl`

Run the full benchmark with:

```bash
python project_code/benchmark.py
```

Useful variants:

```bash
python project_code/benchmark.py --batch-size 10
python project_code/benchmark.py --debug
python project_code/benchmark.py --sample-size 500 --batch-size 10
```

Common CLI flags include:

- `--docs-path` to override the input document file
- `--qrels-path` to override the input qrels file
- `--embedding-model` to switch the dense embedding model
- `--metadata-boost` to tune metadata-aware scoring
- `--k-values` to change evaluation cutoffs
- `--sample-size` and `--debug` for smaller test runs
- `--batch-size` and `--seed` for reproducible query subsampling
- `--baseline` and `--methods` to select the retrieval methods to evaluate
- `--hyperparam-search` and `--n-trials` for optional tuning runs

To run GNNRet specifically:

```bash
python project_code/benchmark.py --methods gnn_ret
```

## 7. View benchmark results

The benchmark writes its results to:

- `results/method_results.json`

To create a comparison plot from the saved benchmark results, run:

```bash
python project_code/benchmark_visualization.py
```

The generated chart will be written under `results/visualizations/`.

## 8. Notes

- `project_datasets/data_set_alignment.py` should be run before the enrichment and benchmark steps because the later stages rely on the generated JSONL files.
- If the dataset download step fails, verify that the required packages are installed and that any required credentials are configured.
- The benchmark can be pointed at different dataset versions with `--docs-path` and `--qrels-path` when needed.

## Quick start summary

1. Activate the virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python project_datasets/data_set_alignment.py`.
4. Run `python project_datasets/data_set_enrichment.py`.
5. Run `python project_code/validate_benchmark.py`.
6. Run `python project_code/benchmark.py`.
7. Generate a plot with `python project_code/benchmark_visualization.py`.
8. Inspect the output in `results/method_results.json` and `results/visualizations/`.