# anlp-lab-course-2026

## Requirements

Python 3.12.0
A virtual environment e.g. MyVenv with `python -m venv MyVenv`

## Benchmark Pipeline

This repository contains the benchmark pipeline for dataset acquisition, alignment, and retrieval evaluation.

## Core pipeline files

- `project_datasets/data_set_alignment.py` — download, normalize, align, and export documents/qrels
- `project_datasets/json_processing.py` — process raw GitHub issue JSON and normalize labels
- `code/benchmark.py` — run retrieval benchmarks on the aligned dataset
- `code/methods.py` — retrieval method implementations used by the benchmark
- `code/validate_benchmark.py` — verify data files, benchmark script, and environment
- `requirements.txt` — the required pip packages

## 1. Required Set up

### 1) Activate the virtual environment

Windows:

```bat
MyVenv\Scripts\activate.bat
```

or in powershell:

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

## 2. Download datasets and build aligned dataset files

```bash
python project_datasets/data_set_alignment.py
```

This script will:

- download the GitHub issues dataset via `kagglehub`
- normalize and align labels using `json_processing.normalise_labels()`
- download the LitSearch query and corpus dataset via `datasets`
- write aligned outputs to:
  - `project_datasets/output/documents.jsonl`
  - `project_datasets/output/qrels.jsonl`

Use `python project_datasets/data_set_alignment.py --use-cache` to reuse cached dataset files when available.

## 3. Dataset enrichment
```bash
python project_datasets/data_set_enrichment.py
```
This script will:

- run the following 3 data enrichment steps, provided the caches with collected API information exist
  (`project/datasets/cache/paper_cache.json`, `project/datasets/cache/hierarchy_cache_nano_batch.json`)
- enrichment steps can be done again if anything in the pipeline changes by running
`python project_datasets/data_set_enrichment.py --overwrite-metadata` or
`python project_datasets/data_set_enrichment.py --readd-paper-to-paper` or
`python project_datasets/data_set_enrichment.py --overwrite-hierarchy` respectively. 
It is advised that the order of enrichment of 1), 2) and 3) should
be kept.

The dataset enrichment pipeline incrementally extends the aligned corpus with additional metadata.
Each enrichment stage creates a new dataset version while preserving all information from previous stages.

### 1) Semantic Scholar metadata
documents.jsonl
→ documents_enriched_01.jsonl

The paper subset of the unified corpus is enriched using the Semantic Scholar Graph API.
Semantic Scholar enrichment was available for 62,641 of 64,183 paper records (~97.6% coverage).

The enrichment step adds:
- Authors
- Author affiliations
- Fields of study
- Publication venue
- Journal information
- Publication types

The retrieved metadata is stored locally in:
`project/datasets/cache/paper_cache.json`
to avoid repeated API calls.
The API querying process can be found in `project_datasets/SemanticScholar_API_calling.py`.

### 2) Paper to Paper retrieval queries
documents_enriched_01.jsonl
→ documents_enriched_02.jsonl

qrels.jsonl
→ qrels_02.jsonl

A subset of papers is converted into retrieval queries using citation relationships already present in the corpus.
For each selected paper:  
Query text = title + abstract  
Relevant documents = cited or explicitly related papers  
Only a random subset of papers is converted into queries by default. 
A fixed random seed (Random(42)) is used to ensure reproducibility.

### 3) Hierarchy enrichment
documents_enriched_02.jsonl
→ documents_enriched_03.jsonl

Hierarchical scientific metadata is extracted using an LLM.
The enrichment produces three hierarchy structures:
```
{
  "affiliations": [
    {
      "country": "",
      "sector": "",
      "organization": ""
    }
  ],
  "field_of_study_path": {
    "field": "",
    "research_area": "",
    "topic_family": "",
    "specific_topic": ""
  },
  "method_path": {
    "method_family": "",
    "method_category": "",
    "specific_method": ""
  }
}
```
A paper may contain multiple affiliations; therefore affiliations is represented as a list of affiliation objects.
The hierarchy extraction is performed using OpenAI GPT-5 Nano.
Hierarchy outputs are stored locally in:
`project/datasets/cache/hierarchy_cache_nano_batch.json`
to allow interrupted enrichment runs to be resumed without repeating previously completed API requests.
The API querying process can be found in `project_datasets/OpenAI_API_calling.py`

#### Notes on data quality
- 6,197 of 64,183 papers (~9.6%) contain neither title nor abstract and are excluded from hierarchy enrichment.
- Invalid or unrecoverable outputs are marked with an error entry and do not overwrite existing metadata.
- to get full statistics on errors and categories of the LLM output run `python project_datasets paper_validation.py`

## 4. Validate the benchmark setup

```bash
python project_code/validate_benchmark.py
```

## 5.  Run the retrieval benchmark (takes very... long)

```bash
python project_code/benchmark.py
```

better

```bash
python project_code/benchmark.py --batch-size 10
```

For quick debug testing use a smaller sampled catalog:

```bash
python project_code/benchmark.py --debug
```

Or explicitly sample a subset of documents while preserving query/qrel groups:

```bash
python project_code/benchmark.py --sample-size 500 --batch-size 10
```

#### other command flags:

| Task                            | Command                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| Run with defaults (all queries) | `python code/benchmark.py`                                                                      |
| Quick test (10 random queries)  | `python code/benchmark.py --batch-size 10`                                                      |
| Medium batch (50 queries)       | `python code/benchmark.py --batch-size 50`                                                      |
| Test with different seed        | `python code/benchmark.py --batch-size 10 --seed 123`                                           |
| Change embedding model          | `python code/benchmark.py --embedding-model all-mpnet-base-v2`                                  |
| Custom k-values                 | `python code/benchmark.py --k-values 5 20 100`                                                  |
| Debug mode (small sampled run)  | `python code/benchmark.py --debug`                                                              |
| Sample documents for testing    | `python code/benchmark.py --sample-size 500 --batch-size 10`                                    |
| Use specialized paper model     | `python code/benchmark.py --embedding-model allenai/specter2`                                   |
| Full custom command             | `python code/benchmark.py --batch-size 20 --embedding-model all-mpnet-base-v2 --k-values 10 50` |

## 6. View benchmark results

Results are written to:

- `results/method_results.json`

## 7. Notes

- `project_datasets/data_set_alignment.py` must run before `code/benchmark.py` because the benchmark reads the aligned dataset files.
- If the dataset download step fails, verify that `kagglehub`, `datasets`, `pandas`, and `tqdm` are installed and that Kaggle credentials are configured.

## Quick Start Summary

1. Activate virtual environment.
2. Install dataset and benchmark dependencies.
3. Run `python project_datasets/data_set_alignment.py`.
4. Run `python code/validate_benchmark.py`.
5. Run `python code/benchmark.py`.
6. Check results in `results/method_results.json`.
