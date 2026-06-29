# Unified Common Data Schema for Mixed-Property Retrieval Benchmarks

This schema combines the **GitHub issues** and **scientific papers** datasets into one unified benchmark format.
It supports free text retrieval, structured categorical fields, hierarchical categories, explicit relations for ground truth, and multiple datasets in one unified pipeline.

---

## 1. Design Goals

The schema should be:

1. **Unified** across datasets
   One record format for GitHub issues and papers (and future sources).

2. **Retrieval-friendly**
   Easy to index, embed, and evaluate.

3. **Extensible**
   New datasets and fields (e.g. entities for GNN) can be populated later without changing core code.

4. **Ground-truth aware**
   Explicit relevance judgments are stored separately from documents.

5. **Hierarchy-aware**
   Supports nested categorical structures such as:
   - label → sublabel
   - domain → subdomain → topic
   - institution → faculty → lab
   - component → subsystem → module

---

## 2. File Layout

qrels also known as _set relevance assessments_.

Use two main files:

- `documents.jsonl`
  One row per item to retrieve.

- `qrels.jsonl`
  One row per query–relevance pair.

Optional:

- `queries.jsonl`
  If queries are kept separate from the document corpus (relevant for the papers dataset).

- `relations.jsonl`
  For explicit graph edges beyond retrieval labels (useful for GNN construction).

---

## 3. Canonical Document Schema

Each dataset item is stored in a common document format.

```json
{
  "id": "string",
  "source_dataset": "string",
  "source_type": "github_issue | paper | query",

  "title": "string | null",
  "main_text": "string",
  "secondary_texts": ["string"],

  "structured_fields": {
    "categorical": {
      "field_name": "string"
    },
    "hierarchical": {
      "field_name": ["level_1", "level_2", "level_3"]
    }
  },

  "entities": {
    "people": ["string"],
    "organizations": ["string"],
    "projects": ["string"],
    "topics": ["string"]
  },

  "relations": {
    "explicit_related_ids": ["string"]
  },

  "retrieval_metadata": {
    "is_queryable": true,
    "is_candidate": true
  },

  "raw_source": {
    "native_id": "string | null",
    "url": "string | null"
  }
}
```

---

## 4. Field Definitions

### `id`

A unique identifier across all datasets.

Example:

- `1234`

---

### `source_dataset`

The dataset name or collection name.

Examples:

- `github_apache_airflow`
- `semantic_scholar`
- `set_a` (from `query_set`)

---

### `source_type`

The document type in the original source.

Values:

- `github_issue`
- `paper`
- `query`

For the papers dataset, queries and corpus papers are distinct record types. Queries carry `is_candidate: false` and papers carry `is_queryable: false`. This makes the pipeline logic clean and avoids special-casing downstream.

---

### `title`

Short textual title, or `null` if not applicable (e.g. for query records).

Examples:

- GitHub issue title
- paper title
- `null` for a natural language query

---

### `main_text`

The primary free-text content used for retrieval.

Examples:

- GitHub issue body
- paper abstract
- natural language query string

---

### `secondary_texts`

Additional free-text fields that may assist retrieval. Stored as a list so sources can be concatenated or selectively used later.

Examples:

- GitHub issue comments
- full paper text
- PR description or review discussion

---

### `structured_fields.categorical`

Fields with possible multiple categories that do not form a strict hierarchy.

Examples:

- GitHub issue `labels`: `["bug", "scheduler"]`
- keywords or tags

---

### `structured_fields.hierarchical`

The most important structured field for graph-based and GNN approaches.

Represent hierarchical paths as ordered lists from general to specific.

Examples:

GitHub:

```json
"hierarchical": {
  "bug_category": ["infrastructure", "scheduler", "parsing"]
}
```

Papers:

```json
"hierarchical": {
   "affiliations": [
    {"country": "USA", "sector": "Academic", "organization": "Carnegie Mellon University"},
    {"country": "USA", "sector": "Academic", "organization": "Dartmouth College"},
    {"country": "USA", "sector": "Industry", "organization": "Google"},
    {"country": "USA", "sector": "Industry", "organization": "Google DeepMind"}]
  "field_of_study_path": {
    "field": "Computer Science",
    "research_area": "Machine Learning",
    "topic_family": "Robustness and Evaluation",
    "specific_topic": "BERT reproducibility"
  }
}
```

This format makes it easy to compute prefix match, hierarchical distance, and shared ancestor similarity — all useful as GNN edge features.

---

### `entities`

Named entities and important linked objects. Currently empty lists for both datasets, kept as a forward-looking placeholder for GNN node and edge feature construction.

Fields:

- `people`: authors, assignees, reporters, reviewers
- `organizations`: institutions, companies, labs
- `projects`: repository names, venues, tracks
- `topics`: IR, GNN, retrieval, etc.

For GitHub issues, `people` (assignees) and `projects` (repository) are partially derivable already. For papers, `people` (authors) and `organizations` (affiliations) are the natural targets once author metadata is available. These map directly onto graph nodes in a GNN setting.

---

### `relations.explicit_related_ids`

The most important field for ground truth and supervision.

Contains IDs of items known to be directly related.

Examples:

- GitHub: related issue IDs (from `linked_issues.related_issue_nos`)
- Papers corpus: cited paper IDs (from `citations`)
- Queries: gold-standard paper IDs (from `corpusids`)

---

### `retrieval_metadata`

Controls benchmark splits and retrieval roles.

Fields:

- `is_queryable`: whether this item can be used as a query
- `is_candidate`: whether this item can be retrieved as a candidate

Future possible:

- `split`: `train` / `dev` / `test` / `unlabeled`

For the papers dataset:

- query records → `is_queryable: true`, `is_candidate: false`
- paper records → `is_queryable: false`, `is_candidate: true`

For GitHub issues:

- issue records → `is_queryable: true`, `is_candidate: true`

---

### `raw_source`

Keeps the original source reference for debugging and provenance.

Fields:

- `native_id`: original dataset ID (e.g. `issue_no`, `corpusid`)
- `url`: original URL if available

---

## 5. Field Mapping Tables

### GitHub Issues

| Schema field                           | Source column                           | Notes                                           |
|----------------------------------------|-----------------------------------------|-------------------------------------------------|
| `id`                                   | `issue_no` + `repository`               | e.g. `github:apache-airflow:issue:23145`        |
| `source_dataset`                       | `repository`                            | e.g. `github_apache_airflow`                    |
| `source_type`                          | —                                       | hardcode `github_issue`                         |
| `title`                                | `issue_title`                           |                                                 |
| `main_text`                            | `issue_body`                            |                                                 |
| `secondary_texts`                      | `comments`                              | list of comment strings                         |
| `structured_fields.categorical.status` | —                                        | empty                                           |
| `structured_fields.hierarchical`       | `labels`                                | list labels forming a hierarchy, e.g. `area\ui` |
| `entities.people`                      | —                                       | empty for now; assignees derivable later        |
| `entities.organizations`               | —                                       | empty for now                                   |
| `entities.projects`                    | `repository`                            | derivable now if desired                        |
| `entities.topics`                      | —                                       | empty for now                                   |
| `relations.explicit_related_ids`       | `related_issue_nos` from `linked_issues` | join on `issue_no`                              |
| `retrieval_metadata.is_queryable`      | —                                       | `true`                                          |
| `retrieval_metadata.is_candidate`      | —                                       | `true`                                          |
| `raw_source.native_id`                 | `issue_no`                              |                                                 |
| `raw_source.url`                       | `issue_url`                             |                                                 |

### Paper Corpus Records

| Schema field                      | Source column | Notes                                       |
| --------------------------------- | ------------- | ------------------------------------------- |
| `id`                              | `corpusid`    | e.g. `paper:12345678`                       |
| `source_dataset`                  | —             | hardcode e.g. `semantic_scholar`            |
| `source_type`                     | —             | hardcode `paper`                            |
| `title`                           | `title`       |                                             |
| `main_text`                       | `abstract`    |                                             |
| `secondary_texts`                 | `full_paper`  | wrap in list: `[full_paper]`                |
| `structured_fields.categorical`   | —             | empty `{}` for now                          |
| `structured_fields.hierarchical`  | —             | empty `{}` for now                          |
| `entities.people`                 | —             | empty for now; authors derivable later      |
| `entities.organizations`          | —             | empty for now; affiliations derivable later |
| `entities.projects`               | —             | empty for now                               |
| `entities.topics`                 | —             | empty for now                               |
| `relations.explicit_related_ids`  | `citations`   | list of cited `corpusid`s                   |
| `retrieval_metadata.is_queryable` | —             | `false`                                     |
| `retrieval_metadata.is_candidate` | —             | `true`                                      |
| `raw_source.native_id`            | `corpusid`    |                                             |
| `raw_source.url`                  | —             | `null`                                      |

### Query Records (Papers Dataset)

| Schema field                                | Source column | Notes                          |
| ------------------------------------------- | ------------- | ------------------------------ |
| `id`                                        | derived       | e.g. `query:papers:set_a:0042` |
| `source_dataset`                            | `query_set`   |                                |
| `source_type`                               | —             | hardcode `query`               |
| `title`                                     | —             | `null`                         |
| `main_text`                                 | `query`       | natural language query string  |
| `secondary_texts`                           | —             | `[]`                           |
| `structured_fields.categorical.specificity` | `specificity` |                                |
| `structured_fields.categorical.quality`     | `quality`     |                                |
| `structured_fields.hierarchical`            | —             | empty `{}` for now             |
| `entities`                                  | —             | all empty lists for now        |
| `relations.explicit_related_ids`            | `corpusids`   | gold-standard paper IDs        |
| `retrieval_metadata.is_queryable`           | —             | `true`                         |
| `retrieval_metadata.is_candidate`           | —             | `false`                        |
| `raw_source.native_id`                      | —             | `null`                         |
| `raw_source.url`                            | —             | `null`                         |

---

## 6. Concrete Record Examples

### GitHub Issue Record

```json
{
  "id": "github:apache-airflow:issue:23145",
  "source_dataset": "github_apache_airflow",
  "source_type": "github_issue",
  "title": "Scheduler crashes when parsing nested DAGs",
  "main_text": "The scheduler fails after upgrading to version X.Y. Nested DAG definitions trigger a parsing error ...",
  "secondary_texts": [
    "Comment 1: Happens only on Linux.",
    "Comment 2: Related to file path handling."
  ],
  "structured_fields": {
    "categorical": { "labels": ["bug", "scheduler"] },
    "hierarchical": {}
  },
  "entities": {
    "people": [],
    "organizations": [],
    "projects": ["apache-airflow"],
    "topics": []
  },
  "relations": {
    "explicit_related_ids": ["github:apache-airflow:issue:23099"]
  },
  "retrieval_metadata": {
    "is_queryable": true,
    "is_candidate": true
  },
  "raw_source": {
    "native_id": "23145",
    "url": "https://github.com/apache/airflow/issues/23145"
  }
}
```

### Paper Corpus Record

```json
{
  "id": "paper:12345678",
  "source_dataset": "semantic_scholar",
  "source_type": "paper",
  "title": "Graph Neural Methods for Mixed-Property Retrieval",
  "main_text": "We study retrieval over datasets containing both text and structured hierarchical metadata ...",
  "secondary_texts": ["Full paper text here ..."],
  "structured_fields": {
    "categorical": {"venue_name": None, "venue_type": None, "journal": None, "journal_volume": None, "fields_of_study": [], "publication_types":[]},
    "hierarchical": {"affiliations": [ {} ], "field_of_study_path":  {}, "method_path":  {}}
  },
  "entities": {
    "people": [list of authors],
    "organizations": [list of correponding organizations of authors],
    "projects": [],
    "topics": [semanticscholar topic, but only says ComputerScience]
  },
  "relations": {
    "explicit_related_ids": ["paper:87654321", "paper:11223344"]
  },
  "retrieval_metadata": {
    "is_queryable": false,
    "is_candidate": true
  },
  "raw_source": {
    "native_id": "12345678",
    "url": null
  }
}
```

### Query Record

```json
{
  "id": "query:papers:set_a:0042",
  "source_dataset": "set_a",
  "source_type": "query",
  "title": null,
  "main_text": "methods for hierarchical document retrieval using graph networks",
  "secondary_texts": [],
  "structured_fields": {
    "categorical": {
      "specificity": "high",
      "quality": "good"
    },
    "hierarchical": {}
  },
  "entities": {
    "people": [],
    "organizations": [],
    "projects": [],
    "topics": []
  },
  "relations": {
    "explicit_related_ids": ["paper:12345678", "paper:87654321"]
  },
  "retrieval_metadata": {
    "is_queryable": true,
    "is_candidate": false
  },
  "raw_source": {
    "native_id": null,
    "url": null
  }
}
```

---

## 7. Retrieval Ground Truth Schema (Qrels)

Store relevance judgments separately from documents.
Future idea:
source (type) e.g. explicit | weak
split e.g. train | dev | test

```json
{
  "query_id": "string",
  "candidate_id": "string",
  "relation_type": "linked_issue | citation | corpusid_match | manually_judged"
}
```

**Future task, not relevant for now**
Recommended relevance scale:

- `3` = direct positive / gold relation
- `2` = strong weak positive
- `1` = soft related
- `0` = not relevant / unjudged

For a strict IR benchmark keep only `3` as positive and `0` as negative.

### Example Qrels Records

Papers query to paper:

```json
{
  "query_id": "0042",
  "candidate_id": "12345678",
  "relation_type": "corpusid_match"
}
```

GitHub issue to linked issue:

```json
{
  "query_id": "23145",
  "candidate_id": "23099",
  "relation_type": "linked_issue"
}
```

---

## 8. Recommended Benchmark Setup

### Queries

- GitHub issues (`is_queryable: true`)
- Paper queries (`source_type: query`)

### Candidates

- GitHub issues (`is_candidate: true`)
- Paper corpus records (`is_candidate: true`)

### Ground Truth

- GitHub: explicit issue–issue links from `linked_issues.related_issue_nos`
- Papers: gold corpus IDs from `corpusids` in query frame; citations from `citations` in corpus frame

### Weak Supervision (future)

Use only as auxiliary labels or soft GNN edges:

- Same GitHub label
- Same repository component
- Same author group
- Same institution
- Same topic path

---

## 9. What Is Deferred

These fields are intentionally empty now but kept in the schema for future use:

| Field                            | Reason deferred                 | Future source                                |
| -------------------------------- | ------------------------------- | -------------------------------------------- |
| `structured_fields.hierarchical` | No hierarchy data available yet | Derived label trees, affiliation hierarchies |
| `entities.people`                | Not extracted yet               | NER on text, GitHub assignees, paper authors |
| `entities.organizations`         | Not extracted yet               | Author affiliations, company names           |
| `entities.topics`                | Not extracted yet               | NER, keyword extraction, topic models        |
| `relations` weak positives       | Extra derivation work           | Same label, same author, same institution    |
