# Project Specification: Semi-Structured Information Retrieval

## Exploring GNNs and Other Methods for Retrieval of Mixed-Property Datasets

**Latest Update:** Supervisor feedback incorporated (April 25, 2026)

---

## Introduction

- Name, field of study, project related experience, confidence with handling papers

## Motivation

- Real-world datasets like tickets/issues have **mixed properties** combining free text and structured fields
- **Core Task**: For a given entry (ticket, paper, etc.), find related entries to avoid redundancy and enable bundled resolution
  - GitHub examples: issues within pull requests, issues within tags, linked issues
  - Paper examples: citing/cited relationships, shared research areas, author collaborations
- Need mixed approaches to handle heterogeneous data combining text + structure
- Key insight: **Hierarchical categorical fields** make underlying relationships more interesting and valuable
  - Example: Authors → Research Labs/Institutes (hierarchical structure)
  - Example: Error Types → Frontend/Backend/Infrastructure classifications
- Benchmark methods and gain experience with different techniques across diverse datasets

## Project Description

This project explores the extension of information retrieval beyond classical text or knowledge graph elements. In real-world use cases, information is often not present purely in text form nor purely in knowledge graph form; rather, some information is textual while some is relational and categorical.

**Key Insight**: Mixed-property datasets contain:

1. **Free text fields** (descriptions, abstracts, comments)
2. **Structured/categorical fields** (labels, types, classifications)
3. **Hierarchical relationships** in categorical data (the most interesting aspect)

Example structures:

- **GitHub Issues**: Title + Description (text) + Hierarchical error classification (structure)
- **Scientific Papers**: Abstract + Body (text) + Author affiliation hierarchy (structure)

This project leverages established techniques from KG retrieval and text retrieval:

- Vector embeddings (for semantic text similarity)
- Graph walking (multihop reasoning, LLM agents)
- Graph Neural Networks (GNNs) for relational structure refinement

## Project Plan (Refined with Supervisor Feedback)

### Phase 0+1: Familiarization & Dataset Identification (Complete by **May 14**)

**Familiarization:**

- Learn basics of GNNs, RAG, GraphRAG, information retrieval, metrics

**Dataset Identification - CRITICAL REQUIREMENTS:**

- **Identify 2 datasets** with mixed properties (diversity is key)
- Each dataset MUST have:
  - Free text content (descriptions, abstracts, etc.)
  - Structured/categorical fields
  - **IMPORTANTLY**: Hierarchical relationships in categorical fields
    - Example (GitHub): Error types → {Frontend, Backend, Infrastructure}
    - Example (Papers): Authors → {Research Labs, Institutes, Companies}
  - Ground truth (or ability to create it) (known related pairs for evaluation)
- **Recommended combination**: GitHub Issues + Scientific Papers
- Alternative: Other ticket systems (Jira, etc.) + Academic citations

### Phase 2: Literature Review (Complete by **May 28**)

- Identify promising retrieval techniques from recent papers
- Supervisor emphasis: **embeddings + GNN refinement layers** trained on small datasets
- Understand how hierarchical structures enhance retrieval
- Focus on papers referenced in project specification

### Phase 3: Benchmark Implementation (Complete by **July 9**)

**Implementation Requirements:**

- **Unified data schema** for both datasets (see implementation section below)
- **Dataset-specific ETL functions** converting raw data to common schema
- Implement at least:
  - **2 datasets** (GitHub + Papers recommended)
  - **2 SOTA baselines** (e.g., BM25, dense retrieval methods)
  - **2 pretrained models** (embedding models as needed)
  - **1 evaluation metric** (MRR, NDCG, Recall@k, etc.)
  - **1+ novel team method** (ideally combining embeddings + GNN)

**Time Allocation Recommendations:**

- Invest 2-3 days early on: Tech stack/architecture design
- Main phase: Dataset prep, baseline implementation, experiments
- Budget 3-5 days end buffer: LLM evaluation and final runs may take time

### Phase 4: Presentation (Complete by **July 14**)

- Create poster and presentation for poster session
- Tell a compelling story across 2 datasets
- Demonstrate method effectiveness on diverse data

### Phase 5: Final Report & Scope Extension (After July 14)

- Enhance analysis and broaden scope if time permits
- Additional datasets or methods beyond initial 2
- Final report with comprehensive evaluation

## Timeline at a Glance

- **Meetings**: Every 2 weeks, 45 minutes (Thursday, 12:15-13:00)
- **May 14**: X Familiarization complete + 2 diverse datasets identified with ground truth
- **May 28**: X Literature review with identified techniques
- **July 9**: X Benchmark implemented and ready for evaluation
- **July 14**: X Poster session (tell the first story)
- **End of exam period**: X Final report due (broadened evaluation)

**Note**: Timeline is flexible based on challenges discovered. Communicate conflicts (vacation, exams) that affect uniform pacing.

## Implementation Strategy

### Unified Data Schema & Modular Architecture

To efficiently handle 2+ datasets, implement modular data processing:

**Common Data Schema** (unified representation across all datasets):

```
{
  id: unique_identifier,
  text_fields: {
    main: string,           // primary content (issue desc, abstract, etc.)
    secondary: [string]     // comments, references, additional text
  },
  metadata: {
    title: string,
    created: timestamp,
    source: string          // dataset source identifier
  },
  structured_fields: {
    categorical: {          // simple categories
      field_name: value,
      field_name: value
    },
    hierarchical: {         // hierarchical structures
      category_name: [hierarchy_path]
    }
  },
  related_entities: {
    explicit: [ids],        // known related items (ground truth)
    metadata_based: [ids]   // derived relations (same category, author, etc.)
  }
}
```

**Dataset-Specific ETL Functions:**

```python
def prepare_github_issues(raw_data) -> CommonSchema
  # Map GitHub API response to common schema
  # Handle: labels→hierarchical categories, linked_issues→related_entities

def prepare_papers(raw_data) -> CommonSchema
  # Map paper metadata to common schema
  # Handle: authors→hierarchical affiliations, citations→related_entities
```

**Benefits:**

- Embedding, retrieval, and evaluation code works uniformly
- Minimal dataset-specific logic
- Easy to add new datasets later

### Ground Truth Strategy

**This is CRITICAL for evaluation.** You need known related pairs to test retrieval methods.

**Approaches (ranked by preference):**

1. **Explicit ground truth** (easiest, highest quality)
   - GitHub: Use existing issue links (30-50% of issues in popular repos)
   - Papers: Use citation relationships
   - Papers: Shared authors or research groups

2. **Weak supervision** (scalable, moderate quality)
   - GitHub: Issues with same labels (not duplicates, but related)
   - GitHub: Issues resolved by same PR
   - Papers: Papers in same research area/venue/time period
   - Papers: Authors from same institution

3. **Synthetic generation** (if needed)
   - Supervisor mentioned applying a "clever method" to generate expected outputs
   - Can discuss in next meeting if explicit/weak supervision insufficient
   - Examples: Using LLM to rate similarity, hierarchical distance metrics

## Communication & Project Management

- **Team Project**: Setup shared communication channel (Slack, Teams, Discord)
- **Advisor Contact**: BMW email for questions between meetings
- **Guidelines**:
  - Utilize each team member's strengths
  - Communicate weaknesses/blockers early
  - Flag conflicts (vacation, exams) that may affect timeline
  - Request timeline adjustments based on emerging challenges

## Setup & Resources

### Compute Resources

- Private compute resources or free options: Google Colab, SOC Server
- **TUM LLM Service**: https://morpheus.cit.tum.de/
  - Free credits available
  - SOC credits available
- Expected compute needs:
  - Dataset download and preprocessing: CPU sufficient (~1-2 hours)
  - Embedding generation: GPU helpful (~2-4 hours per dataset)
  - GNN training on small datasets: GPU recommended (varies by architecture)
  - Inference and evaluation: Flexible (can use CPU/GPU as available)

### AI Tool Usage Policy

- **Allowed**: Yes, you may use AI tools (per supervisor guidance)
- **Responsibility**: You are the supervisor of the work; take final responsibility
- **Consideration**: AI is an amplifier (for better or worse)
- **Best Practice**:
  - Use for ideation, code scaffolding, document drafting
  - Verify all technical claims yourself
  - Cite AI tool usage in methods
- **Report Requirement**: Include a "Declaration of AI Tool Usage" section in final report

### Expected Challenges & Mitigation

| Challenge                  | Impact                | Mitigation                                                   |
| -------------------------- | --------------------- | ------------------------------------------------------------ |
| Ground truth creation      | 2-3 days              | Start early; use explicit links first, then weak supervision |
| LLM evaluation             | Time-consuming        | Budget execution time at end of Phase 3                      |
| Dataset diversity          | Project scope         | Carefully select 2 datasets with different structures        |
| Tech stack decisions       | Late changes = rework | Invest 2-3 days early on Phase 3 for architecture            |
| Unforeseen blockers        | Deadline pressure     | Maintain 3-5 day buffer before milestones                    |
| Variable team availability | Timeline slippage     | Communicate conflicts early; adjust timeline as needed       |

## References

### Ticket Resolution

- Inspiration for dataset selection
- https://arxiv.org/abs/2505.23419

### Graph + LLM Surveys (Graph RAG Methods)

- https://arxiv.org/abs/2408.08921
- https://aclanthology.org/2022.aacl-main.46/
- https://arxiv.org/abs/2308.07107
- https://arxiv.org/abs/2501.00309
- https://arxiv.org/abs/2407.09777
- https://ojs.aaai.org/index.php/AAAI/article/view/5681
- https://ieeexplore.ieee.org/document/10915556

### Tables + LLMs

- https://arxiv.org/abs/2402.17944

### GNN + LLM

- https://aclanthology.org/2025.findings-acl.856/
- https://aclanthology.org/2021.naacl-main.45/

## Useful Tools

- Google Scholar: https://scholar.google.com/
- Connected Papers: https://www.connectedpapers.com/auth
- Semantic Scholar (Asta): https://asta.allen.ai/
