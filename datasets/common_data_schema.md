# Unified Common Data Schema for Mixed-Property Retrieval Benchmarks

<span style="color:red">Depends on our approach, how the data sets look like. This is only a draft until we know for sure how the data sets look like and what is possible!!!</span>

This schema is designed to combine datasets such as **GitHub issues / pull requests** and **scientific papers** into one benchmark format.  
It supports:

- free text retrieval
- structured categorical fields
- hierarchical categories
- explicit relations for ground truth
- weakly derived relations for analysis
- multiple datasets in one unified pipeline

---

## 1. Design Goals

The schema should be:

1. **Unified** across datasets  
   One record format for GitHub, papers, tickets, and future sources.

2. **Retrieval-friendly**  
   Easy to index, embed, and evaluate.

3. **Extensible**  
   New datasets can be added without changing core code.

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

We can save it as a file temporary to then load it, so we save us the preprocessing all the time

Suggestions by ChatGPT <br>
Use two main files:

- `documents.jsonl`  
  One row per item to retrieve.

- `qrels.jsonl`  
  One row per query–relevance pair.

Optional:

- `queries.jsonl`  
  If queries are not simply the same as documents.

- `relations.jsonl`  
  For explicit graph edges beyond retrieval labels.

---

## 3. Canonical Document Schema

Each dataset item is stored in a common document format.

```json
{
  "id": "string",
  "source_dataset": "string",
  "source_type": "github_issue | github_pr | paper | ticket | other",
  "title": "string",
  "main_text": "string",
  "secondary_texts": ["string"],
  "language": "string",
  "timestamps": {
    "created_at": "string",
    "updated_at": "string",
    "closed_at": "string"
  },
  "structured_fields": {
    "categorical": {
      "field_name": "string"
    },
    "hierarchical": {
      "field_name": ["level_1", "level_2", "level_3"]
    },
    "multi_label": {
      "field_name": ["label_a", "label_b"]
    }
  },
  "entities": {
    "people": ["string"],
    "organizations": ["string"],
    "projects": ["string"],
    "topics": ["string"]
  },
  "relations": {
    "explicit_related_ids": ["string"],
    "metadata_related_ids": ["string"]
  },
  "retrieval_metadata": {
    "split": "train | dev | test | unlabeled",
    "is_queryable": true,
    "is_candidate": true
  },
  "raw_source": {
    "native_id": "string",
    "url": "string"
  }
}
```

---

## 4. Field Definitions

### `id`

A unique canonical identifier across all datasets.

Example:

- `github:apache-airflow:issue:1234`
- `paper:10.1145/1234567.8901234`

---

### `source_dataset`

The dataset name or collection name. TODO: We could also name this as source_dataset_type and make it categorical, hence `github | paper`. <br>
Or we do it on a finer granularity, e.g. github repo pytorch but we also have a raw_source field..

Examples:

- `github_apache`
- `pubmed`
- `acl_papers`
- `jira_company_x`

---

### `source_type`

The document type in the original source. <br>
Depending on how diverse our data sets are we should utilize this, e.g. github data set consists of issues, issue comment and pull requests or more. <br>
Status quo: paper data set uses queries in natural language and has paper ids mapped. So it is not a paper to papers data set afaik.

Recommended values:

- `github_issue`
- `github_pr`
- `paper`
- `query`
- `issue_comment`
- `commit`
- `other`

---

### `title`

Short textual title. Or query text depending on the data sets.

Examples:

- GitHub issue title
- paper title
- query text

---

### `main_text`

The primary free-text content used for retrieval.

Examples:

- GitHub issue description
- paper abstract

---

### `secondary_texts`

Additional free-text fields that may help retrieval.

Examples:

- comments
- PR description
- review discussion
- citation context
- notes

This field is a list so you can concatenate or selectively use text sources later.

---

### `timestamps`

If needed for later, maybe links are outdated.

Fields:

- `created_at`
- `updated_at`
- `closed_at`

Use ISO 8601 strings.

---

### `structured_fields.categorical`

Flat categorical metadata. Maybe a label is given by the meta data directly or indirectly could be utilized for the hierchical part too.

Examples:

- `status`: `open`
- `priority`: `high`
- `venue`: `acl`
- `component`: `frontend`

---

### `structured_fields.hierarchical`

The most important field for your project.

Represent hierarchical paths as ordered lists from general to specific. This could be inferred by a (self-constructed) graph,
this could be the key to get way better accuracy than the baseline through a GNN for instance. <br>

This depends completely and the data set and how we tackel inferring/categorizing it, Jakob already mentioned some possible labels/relations e.g. Front-/backend split, or University -> Insitute -> Group. <br>

Possible examples:

GitHub:

```json
"hierarchical": {
  "bug_category": ["infrastructure", "logging", "parsing"]
}
```

Papers:

```json
"hierarchical": {
  "affiliation": ["Germany", "TUM", "Chair of Information Retrieval"]
}
```

This format makes it easy to compute:

- prefix match
- hierarchical distance
- shared ancestor similarity

---

### `structured_fields.multi_label`

Use for fields with several labels that do not form a strict hierarchy.

Examples:

- tags
- keywords
- topics
- multiple authors’ areas

---

### `entities`

Normalized named entities or important linked objects.

Examples:

- people: authors, assignees, reviewers
- organizations: institutions, companies, labs
- projects: repo names, venues, tracks
- topics: IR, GNN, retrieval

This field is optional but useful for graph-style retrieval and GNN features.

---

### `relations.explicit_related_ids`

The most important field for ground truth or supervision.

Contains IDs of items known to be directly related.

Examples:

- issue ↔ duplicate issue
- issue ↔ fixing PR
- paper ↔ cited paper
- paper ↔ same-sample sibling paper
- query ↔ papers

---

### `relations.metadata_related_ids`

Relations derived from metadata rather than directly annotated.
Might need manual check, too much work?

Examples:

- same label
- same author
- same institution
- same repository
- same component

These should usually be treated as:

- weak positives
- auxiliary graph edges
- analysis-only relations

Not all of them should be used as hard ground truth.

---

### `retrieval_metadata`

Depends on how our data set is structured -> if it always query/issue -> linked stuff then we can drop this imho, except you see a big benefit by having this.
Controls benchmark splits and retrieval roles.

Suggested fields:

- `split`: train / dev / test / unlabeled
- `is_queryable`: whether this item can be used as a query
- `is_candidate`: whether this item can be retrieved as a candidate

This is helpful when queries and candidates differ by type.
For example, you may want:

- issues as queries
- issues + PRs as candidates

---

### `raw_source`

Keeps the original source reference.

Fields:

- `native_id`: original dataset ID
- `url`: original URL if available

This is useful for debugging and provenance.

---

## 5. Example: GitHub Issue Record

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
  "language": "en",
  "timestamps": {
    "created_at": "2025-03-14T10:22:00Z",
    "updated_at": "2025-03-15T12:10:00Z",
    "closed_at": "2025-03-18T08:40:00Z"
  },
  "structured_fields": {
    "categorical": {
      "status": "closed",
      "priority": "high",
      "component": "scheduler"
    },
    "hierarchical": {
      "bug_category": ["infrastructure", "scheduler", "parsing"]
    },
    "multi_label": {
      "labels": ["bug", "parsing", "linux"]
    }
  },
  "entities": {
    "people": ["maintainer_a"],
    "organizations": ["Apache Airflow"],
    "projects": ["airflow"],
    "topics": ["workflow orchestration", "parsing"]
  },
  "relations": {
    "explicit_related_ids": ["github:apache-airflow:pr:9123"],
    "metadata_related_ids": ["github:apache-airflow:issue:23099"]
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

---

## 6. Example: Paper Record

```json
{
  "id": "paper:10.1145/1234567.8901234",
  "source_dataset": "acl_papers",
  "source_type": "paper",
  "title": "Graph Neural Methods for Mixed-Property Retrieval",
  "main_text": "We study retrieval over datasets containing both text and structured hierarchical metadata ...",
  "secondary_texts": [
    "Citation context from related works section",
    "Author affiliations"
  ],
  "language": "en",
  "timestamps": {
    "created_at": "2024-06-01T00:00:00Z",
    "updated_at": "2024-06-10T00:00:00Z",
    "closed_at": null
  },
  "structured_fields": {
    "categorical": {
      "venue": "ACL",
      "track": "main"
    },
    "hierarchical": {
      "affiliation": ["Germany", "TUM", "Department of Computer Science"],
      "topic_path": ["information retrieval", "graph retrieval", "GNNs"]
    },
    "multi_label": {
      "keywords": ["retrieval", "graph neural networks", "mixed data"]
    }
  },
  "entities": {
    "people": ["author_1", "author_2"],
    "organizations": ["TUM"],
    "projects": ["ACL 2024"],
    "topics": ["information retrieval", "graph neural networks"]
  },
  "relations": {
    "explicit_related_ids": ["paper:10.1145/9876543.2109876"],
    "metadata_related_ids": ["paper:10.1145/5555555.6666666"]
  },
  "retrieval_metadata": {
    "split": "test",
    "is_queryable": true,
    "is_candidate": true
  },
  "raw_source": {
    "native_id": "10.1145/1234567.8901234",
    "url": "https://dl.acm.org/doi/10.1145/1234567.8901234"
  }
}
```

---

## 7. Retrieval Ground Truth Schema

Depends on our approach, how the data sets look like!!! Remember this is a draft, since the data sets are not final yet, we could not make final design decisions.
Store relevance judgments separately from documents.

```json
{
  "query_id": "string",
  "candidate_id": "string",
  "relevance": 0,
  "relation_type": "duplicate | citation | linked_issue | same_topic | same_author | same_component | manually_judged",
  "source": "explicit | weak | synthetic",
  "split": "train | dev | test"
}
```

---

## 8. Example Qrels Record

```json
{
  "query_id": "github:apache-airflow:issue:23145",
  "candidate_id": "github:apache-airflow:pr:9123",
  "relevance": 3,
  "relation_type": "linked_pr",
  "source": "explicit",
  "split": "test"
}
```

Recommended relevance scale:

- `3` = direct positive / gold relation
- `2` = strong weak positive
- `1` = soft related
- `0` = not relevant

If you want a strict IR benchmark, keep only:

- `3` as positive
- `0` as negative / unjudged

---

## 9. Recommended Benchmark Setup

A clean setup for your project is:

### Queries

Use:

- GitHub issues
- paper abstracts or titles
- optional tickets

### Candidates

Use:

- issues
- PRs
- papers

### Ground Truth

Use:

- explicit links when available
- citations for papers
- issue–PR links for GitHub
- duplicates or same-incident links where available

### Weak Supervision

Use only as auxiliary labels:

- same label
- same component
- same institution
- same topic path
- same author group

---

## 10. Suggested Minimal Common Schema

If you want the smallest practical version, use this:

```json
{
  "id": "string",
  "source_dataset": "string",
  "source_type": "string",
  "title": "string",
  "text": "string",
  "metadata": {
    "created_at": "string",
    "category": "string",
    "labels": ["string"]
  },
  "hierarchy": {
    "path": ["string", "string", "string"]
  },
  "relations": {
    "positive_ids": ["string"]
  }
}
```

This minimal version is enough to:

- unify datasets
- train baselines
- evaluate retrieval
- extend later without redesign

---

## 11. Practical Recommendation

For your project, I would use this structure:

- `documents.jsonl` for all items
- `qrels.jsonl` for retrieval labels
- `relations.jsonl` for optional graph edges
- `splits.json` for train/dev/test partitioning

That gives you:

- clean benchmark logic
- easy ETL per dataset
- easy evaluation
- compatibility with BM25, dense retrieval, and GNN-based reranking

---

## 12. Notes on Mapping GitHub and Papers

### GitHub mapping

- `title` → issue title / PR title
- `main_text` → issue body / PR description
- `structured_fields.categorical.component` → repo component label
- `structured_fields.hierarchical.bug_category` → label hierarchy if available
- `relations.explicit_related_ids` → linked issue / linked PR / duplicate

### Paper mapping

- `title` → paper title
- `main_text` → abstract
- `secondary_texts` → citations / author info / sections
- `structured_fields.hierarchical.affiliation` → institution hierarchy
- `relations.explicit_related_ids` → citations
