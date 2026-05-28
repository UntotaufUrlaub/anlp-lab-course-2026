# anlp-lab-course-2026

## Dataset Enrichment

The paper subset of the unified corpus was enriched using the Semantic Scholar Graph API with additional metadata such as authors, affiliations, publication venue, and fields of study.

Semantic Scholar enrichment was available for 62,641 of 64,183 paper records (~97.6% coverage). The remaining 1,542 corpus IDs could not be resolved consistently through the API and were therefore left unchanged in the final enriched dataset.
The enriched dataset is around ~2.54 GB.

## Requirements

Python 3.12.0
A virtual environment e.g. MyVenv with `python -m venv MyVenv`

# Benchmark Pipeline

This repository contains the benchmark pipeline for dataset acquisition, alignment, and baseline retrieval evaluation.

## Core pipeline files

- `project_datasets/data_set_alignment.py` — download, normalize, align, and export documents/qrels
- `project_datasets/json_processing.py` — process raw GitHub issue JSON and normalize labels
- `code/baseline_benchmark.py` — run baseline retrieval benchmarks on the aligned dataset
- `code/methods.py` — baseline method implementations used by the benchmark
- `code/validate_benchmark.py` — verify data files, benchmark script, and environment
- `code/benchmark_requirements.txt` — benchmark dependencies
- `project_datasets/requirements.txt` — dataset preparation dependencies

## Execution steps

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
pip install -r project_datasets/requirements.txt
pip install -r code/benchmark_requirements.txt
```

### 3) Download datasets and build aligned dataset files

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

### 4) Validate the benchmark setup

```bash
python code/validate_benchmark.py
```

### 5) Run the baseline benchmark (takes very... long)

```bash
python code/baseline_benchmark.py
```

better

```bash
python code/baseline_benchmark.py --batch-size 10
```

#### other command flags:

| Task                            | Command                                                                                                  |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Run with defaults (all queries) | `python code/baseline_benchmark.py`                                                                      |
| Quick test (10 random queries)  | `python code/baseline_benchmark.py --batch-size 10`                                                      |
| Medium batch (50 queries)       | `python code/baseline_benchmark.py --batch-size 50`                                                      |
| Test with different seed        | `python code/baseline_benchmark.py --batch-size 10 --seed 123`                                           |
| Change embedding model          | `python code/baseline_benchmark.py --embedding-model all-mpnet-base-v2`                                  |
| Custom k-values                 | `python code/baseline_benchmark.py --k-values 5 20 100`                                                  |
| Use specialized paper model     | `python code/baseline_benchmark.py --embedding-model allenai/specter2`                                   |
| Full custom command             | `python code/baseline_benchmark.py --batch-size 20 --embedding-model all-mpnet-base-v2 --k-values 10 50` |

### 6) View benchmark results

Results are written to:

- `results/baseline_results.json`

## Notes

- `project_datasets/data_set_alignment.py` must run before `code/baseline_benchmark.py` because the benchmark reads the aligned dataset files.
- If the dataset download step fails, verify that `kagglehub`, `datasets`, `pandas`, and `tqdm` are installed and that Kaggle credentials are configured.

## Quick Start Summary

1. Activate virtual environment.
2. Install dataset and benchmark dependencies.
3. Run `python project_datasets/data_set_alignment.py`.
4. Run `python code/validate_benchmark.py`.
5. Run `python code/baseline_benchmark.py`.
6. Check results in `results/baseline_results.json`.

TODO: Check paper dataset add what is going wrong on my end
