# Research Evidence Pipeline

A portfolio-ready Data Engineering and AI Engineering product that turns a bounded
Europe PMC literature query into a validated, observable corpus and an evaluated,
citation-bound evidence search interface.

The application is integrated into the repository's central Streamlit portfolio.
Once that portfolio has been connected to Streamlit Community Cloud, this page is
available through the shared navigation without a separate deployment.

## Problem

Scientific search results are not automatically an AI-ready data product. Source
records can be incomplete, duplicated or inconsistent; retrieval can silently fail
for unfamiliar terminology; and summaries can imply certainty that the underlying
evidence does not support.

This project demonstrates the complete engineering path:

1. ingest a bounded batch from a legitimate public API;
2. preserve raw records and content hashes in a Bronze layer;
3. normalize and validate a typed Silver contract;
4. create a Gold document product for retrieval;
5. build lexical and latent-semantic indexes;
6. evaluate index health with deterministic queries;
7. produce an extractive evidence brief whose statements retain source links;
8. expose pipeline, quality and AI diagnostics in one operational interface.

It is a literature-discovery aid, not medical advice and not a systematic review.

## What the application does

- searches one of five useful life-science domains or a safe custom topic;
- retrieves 50–200 recent records with abstracts;
- shows source hit count, retrieved batch size and data retention;
- records per-stage row counts, duration and content hashes;
- executes eight explicit data-contract and reconciliation checks;
- provides Silver and Gold previews plus CSV and JSON exports;
- measures publication-year, venue and open-access-flag coverage;
- embeds the validated corpus using TF-IDF plus Truncated SVD;
- ranks evidence with a configurable lexical/semantic hybrid score;
- blocks or warns on zero-vector and weak-match questions;
- creates an extractive brief from retrieved abstracts with numbered sources;
- reports Hit@5, MRR@10, median rank, zero-vector rate and vocabulary size.

## End-to-end architecture

```text
Europe PMC REST search
        │
        ▼
Extract ── HTTP status / bounded request / fallback state
        │
        ▼
Bronze ── raw payload + source ID + SHA-256 payload hash
        │
        ▼
Silver ── normalized schema + deduplication + contract filters
        │
        ▼
Gold ──── AI text + document hash + derived operational features
        │
        ├──────────────► pipeline ledger / quality checks / manifest
        │
        ▼
TF-IDF sparse index ──► Truncated-SVD latent-semantic index
        │                         │
        └──────── hybrid cosine ranking ────────┐
                                                ▼
                                  citation-bound evidence brief
                                                │
                                                ▼
                            evaluation + confidence/failure states
```

The hosted app intentionally executes in memory. The same stage boundaries and
hashes can be mapped to object storage, a warehouse or an orchestrator without
changing the data contracts.

## Data Engineering implementation

### Extract

`src/data.py` sends a bounded `GET` request with:

- `query`: selected terms plus `HAS_ABSTRACT:Y` and a first-publication-date range;
- `format=json`;
- `resultType=core` for abstracts and detailed metadata;
- `pageSize=50..200` in the interface, with a hard implementation ceiling of 250;
- a descriptive `User-Agent` and an explicit request timeout.

Network, HTTP, schema and empty-result failures switch to deterministic synthetic
metadata. The UI displays a warning and never presents those records as live.

### Bronze

`bronze_table()` preserves each response object as `raw_payload` and adds:

- `ingest_position`;
- `source_record_id`;
- `source`;
- a SHA-256 `payload_hash` from canonical, key-sorted JSON.

Running the same payload through the pipeline therefore produces identical content
hashes, making replay and idempotency checks possible.

### Silver contract

`silver_table()` normalizes nested API fields into one publication per row. It:

- constructs a stable `source:id` record key;
- parses publication dates and years;
- converts citations to non-negative integers;
- flattens MeSH descriptors and full-text links;
- normalizes DOI and Boolean source flags;
- removes duplicate record IDs;
- rejects rows without a usable ID, title, abstract or publication year.

Required contract columns are checked explicitly. A run fails rather than silently
continuing when no record satisfies the contract.

### Gold product

`gold_table()` adds:

- retrieval-ready `document_text` from title, abstract and MeSH terms;
- abstract word count;
- citation band;
- canonical Europe PMC article URL;
- a SHA-256 `document_hash` over AI-relevant content.

The run manifest reconciles Bronze, Silver and Gold row counts and records duration,
status, dropped rows and content hash for every stage.

### Data-quality rules

The UI and exported manifest report:

1. unique record IDs;
2. completeness of required ID, title and abstract fields;
3. minimum abstract length;
4. valid publication-year bounds;
5. non-negative citation counts;
6. Silver/Gold row reconciliation;
7. unique deterministic document hashes;
8. at least 50% Bronze-to-Silver retention.

## AI Engineering implementation

### Hybrid retrieval

`build_index()` uses a deterministic `TfidfVectorizer` with English stop-word
removal, unigram/bigram features, sublinear term frequency and a bounded vocabulary.
`TruncatedSVD(random_state=42)` projects the sparse matrix into at most 64 latent
dimensions. Both representations are normalized.

For a question, `search()` calculates:

```text
hybrid_score = semantic_weight × cosine(SVD query, SVD document)
             + (1 − semantic_weight) × cosine(TF-IDF query, TF-IDF document)
```

The user can set the semantic weight from 0 to 1 and inspect both components for
every result. This makes ranking behavior observable rather than opaque.

### Confidence and failure states

- an empty query produces no results;
- a query with no indexed vocabulary is marked as a zero vector;
- top hybrid score below 0.20 triggers a weak-match warning;
- the evidence brief is withheld when no evidence is retrieved;
- the interface states that relevance is not clinical validity.

### Citation-bound evidence brief

`evidence_brief()` is deliberately extractive. It selects up to two leading
sentences from each of the top three retrieved abstracts and attaches numbered
Europe PMC sources. It does not invoke an LLM, invent prose or make clinical
recommendations. This is a safe, keyless demonstration of retrieval-augmented
evidence assembly rather than a claim of generative reasoning.

### Retrieval evaluation

`evaluate_retrieval()` builds a separate abstract-only TF-IDF index. Up to 40
publication titles become deterministic evaluation queries; the document from which
each title originated is the relevance label. The production title is not included
in that evaluation document representation.

Reported metrics:

- **Hit@5**: share of title queries whose own abstract appears in the first five;
- **MRR@10**: mean reciprocal rank, with ranks beyond ten scored as zero;
- **median rank**;
- **zero-query rate**;
- **evaluation vocabulary size**.

This tests title-to-abstract alignment and index health. It is not a human relevance
benchmark, an answer-correctness evaluation or proof of systematic-review recall.

## Data source

### Provider and exact URLs

- Provider: [Europe PMC](https://europepmc.org/About), operated by EMBL-EBI and partners.
- Production endpoint: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
- [REST API documentation](https://europepmc.org/RestfulWebService)
- [Search syntax](https://europepmc.org/searchsyntax)

No API key or paid credential is used. The API supports JSON and the `core` result
type used by this application. Europe PMC updates its searchable corpus as provider
sources and its index are updated; the public service does not provide this project
with a fixed freshness SLA. The app displays its own exact retrieval timestamp.

### Fields used

| API field | Use |
|---|---|
| `id`, `source` | stable publication key and Europe PMC URL |
| `doi` | normalized external identifier |
| `title` | display and evaluation query |
| `abstractText` | retrieval corpus and extractive evidence |
| `authorString` | attribution |
| `journalTitle` | venue coverage |
| `pubYear`, `firstPublicationDate` | filtering and temporal coverage |
| `citedByCount` | descriptive citation context |
| `isOpenAccess`, `inEPMC` | source-provided availability flags |
| `publicationTypes` | audit metadata |
| `meshHeadingList` | controlled-vocabulary retrieval enrichment |
| `fullTextUrlList` | optional source link |

### Rights and usage assumptions

Europe PMC's REST API documentation is published as an Apache-licensed service
interface. Publication metadata, abstracts and linked full text originate from
multiple providers and individual reuse rights can differ by record. The application:

- retrieves a bounded batch on demand;
- does not package or redistribute full-text articles;
- displays short abstract excerpts for discovery;
- preserves source, author, DOI and Europe PMC links;
- treats `isOpenAccess` only as the provider's record flag, not as a universal
  permission statement.

Users should inspect the publication's own licence before redistributing its content.

## Repository modules

| Module | Responsibility |
|---|---|
| `app.py` | standalone Streamlit entrypoint |
| `ui.py` | controls, pipeline views, diagnostics, retrieval and exports |
| `src/data.py` | Europe PMC request, safe query construction and fallback |
| `src/pipeline.py` | Bronze/Silver/Gold contracts, hashes, QA and manifest |
| `src/retrieval.py` | indexing, hybrid search, evaluation and evidence brief |
| `tests/test_pipeline.py` | contracts, deduplication, types and idempotency |
| `tests/test_retrieval.py` | ranking, guardrails, reproducibility and evaluation |

## Local setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone mode:

```bash
streamlit run research_evidence_pipeline/app.py
```

Tests:

```bash
pytest -q
```

No `.env` file or secret is required.

## Operational assumptions and limitations

- Search uses one bounded result batch, not complete API pagination.
- Default source sorting favors recent results, creating selection bias.
- Citation counts are descriptive, field- and age-dependent, and not quality scores.
- TF-IDF/SVD captures corpus-relative language, not biomedical truth.
- The evaluation relevance labels are derived automatically, not judged by experts.
- Abstracts omit methods and caveats that may appear in full text.
- Extractive sentences can lose surrounding context.
- In-memory Streamlit execution demonstrates the contracts but is not a durable lakehouse.
- The deterministic fallback validates system behavior but contains no real evidence.

## Possible extensions

- persist content-addressed Bronze objects and partitioned Parquet Silver/Gold tables;
- add an orchestrator such as Dagster or Prefect and OpenLineage events;
- paginate incrementally using checkpoints and watermarks;
- add Great Expectations or Pandera contracts;
- add domain sentence embeddings behind a lazy optional dependency;
- create expert-labeled retrieval and answer-quality evaluation sets;
- add duplicate-study and retraction/status signals;
- compare corpus snapshots for concept and vocabulary drift;
- serve the index through FastAPI with latency and error SLOs.

