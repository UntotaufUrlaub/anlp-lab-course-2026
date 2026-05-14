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

## 2. Recommended File Layout

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

  "timestamps": {
    "created_at": "string | null",
    "closed_at": "string | null"
  },

  "structured_fields": {
    "categorical": {
      "field_name": "string"
    },
    "multi_label": {
      "field_name": ["string"]
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
    "split": "train | dev | test | unlabeled",
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

A unique canonical identifier across all datasets.

Examples:

- `github:apache-airflow:issue:1234`
- `paper:12345678`
- `query:papers:set_a:0042`

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

### `timestamps`

ISO 8601 datetime strings, or `null` if not available.

Fields:

- `created_at`
- `closed_at`

---

### `structured_fields.categorical`

Flat single-value categorical metadata.

Examples:

- `status`: `open` / `closed` (derived from `closed_at` being null for GitHub)
- `specificity`: from query metadata in papers dataset
- `quality`: from query metadata in papers dataset

---

### `structured_fields.multi_label`

Fields with multiple labels that do not form a strict hierarchy.

Examples:

- GitHub issue `labels`: `["bug", "scheduler"]`
- keywords or tags

---

### `structured_fields.hierarchical`

The most important structured field for graph-based and GNN approaches.

Represent hierarchical paths as ordered lists from general to specific. Currently empty `{}` for both datasets — to be populated once label hierarchies or affiliation trees are derived.

Possible future examples:

GitHub:

```json
"hierarchical": {
  "bug_category": ["infrastructure", "scheduler", "parsing"]
}
```

Papers:

```json
"hierarchical": {
  "affiliation": ["Germany", "TUM", "Chair of Information Retrieval"],
  "topic_path": ["information retrieval", "graph retrieval", "GNNs"]
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

- `split`: `train` / `dev` / `test` / `unlabeled`
- `is_queryable`: whether this item can be used as a query
- `is_candidate`: whether this item can be retrieved as a candidate

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

| Schema field                           | Source column                            | Notes                                         |
| -------------------------------------- | ---------------------------------------- | --------------------------------------------- |
| `id`                                   | `issue_no` + `repository`                | e.g. `github:apache-airflow:issue:23145`      |
| `source_dataset`                       | `repository`                             | e.g. `github_apache_airflow`                  |
| `source_type`                          | —                                        | hardcode `github_issue`                       |
| `title`                                | `issue_title`                            |                                               |
| `main_text`                            | `issue_body`                             |                                               |
| `secondary_texts`                      | `comments`                               | list of comment strings                       |
| `timestamps.created_at`                | `created_at`                             | ISO 8601                                      |
| `timestamps.closed_at`                 | `closed_at`                              | ISO 8601, nullable                            |
| `structured_fields.categorical.status` | derived                                  | `open` / `closed` from `closed_at` null check |
| `structured_fields.multi_label.labels` | `labels`                                 |                                               |
| `structured_fields.hierarchical`       | —                                        | empty `{}` for now                            |
| `entities.people`                      | —                                        | empty for now; assignees derivable later      |
| `entities.organizations`               | —                                        | empty for now                                 |
| `entities.projects`                    | `repository`                             | derivable now if desired                      |
| `entities.topics`                      | —                                        | empty for now                                 |
| `relations.explicit_related_ids`       | `related_issue_nos` from `linked_issues` | join on `issue_no`                            |
| `retrieval_metadata.is_queryable`      | —                                        | `true`                                        |
| `retrieval_metadata.is_candidate`      | —                                        | `true`                                        |
| `raw_source.native_id`                 | `issue_no`                               |                                               |
| `raw_source.url`                       | `issue_url`                              |                                               |

### Paper Corpus Records

| Schema field                      | Source column | Notes                                       |
| --------------------------------- | ------------- | ------------------------------------------- |
| `id`                              | `corpusid`    | e.g. `paper:12345678`                       |
| `source_dataset`                  | —             | hardcode e.g. `semantic_scholar`            |
| `source_type`                     | —             | hardcode `paper`                            |
| `title`                           | `title`       |                                             |
| `main_text`                       | `abstract`    |                                             |
| `secondary_texts`                 | `full_paper`  | wrap in list: `[full_paper]`                |
| `timestamps.created_at`           | —             | `null`                                      |
| `timestamps.closed_at`            | —             | `null`                                      |
| `structured_fields.categorical`   | —             | empty `{}` for now                          |
| `structured_fields.multi_label`   | —             | empty `{}` for now                          |
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
| `timestamps.created_at`                     | —             | `null`                         |
| `timestamps.closed_at`                      | —             | `null`                         |
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
  "timestamps": {
    "created_at": "2025-03-14T10:22:00Z",
    "closed_at": "2025-03-18T08:40:00Z"
  },
  "structured_fields": {
    "categorical": { "status": "closed" },
    "multi_label": { "labels": ["bug", "scheduler"] },
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
    "split": "test",
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
  "timestamps": {
    "created_at": null,
    "closed_at": null
  },
  "structured_fields": {
    "categorical": {},
    "multi_label": {},
    "hierarchical": {}
  },
  "entities": {
    "people": [],
    "organizations": [],
    "projects": [],
    "topics": []
  },
  "relations": {
    "explicit_related_ids": ["paper:87654321", "paper:11223344"]
  },
  "retrieval_metadata": {
    "split": "test",
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
  "timestamps": {
    "created_at": null,
    "closed_at": null
  },
  "structured_fields": {
    "categorical": {
      "specificity": "high",
      "quality": "good"
    },
    "multi_label": {},
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
    "split": "test",
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

```json
{
  "query_id": "string",
  "candidate_id": "string",
  "relevance": 0,
  "relation_type": "linked_issue | citation | corpusid_match | manually_judged",
  "source": "explicit | weak",
  "split": "train | dev | test"
}
```

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
  "query_id": "query:papers:set_a:0042",
  "candidate_id": "paper:12345678",
  "relevance": 3,
  "relation_type": "corpusid_match",
  "source": "explicit",
  "split": "test"
}
```

GitHub issue to linked issue:

```json
{
  "query_id": "github:apache-airflow:issue:23145",
  "candidate_id": "github:apache-airflow:issue:23099",
  "relevance": 3,
  "relation_type": "linked_issue",
  "source": "explicit",
  "split": "test"
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
