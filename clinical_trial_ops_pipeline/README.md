# Clinical Trial Operations ML Pipeline

An online Data Engineering and AI Engineering product that turns public ClinicalTrials.gov records into an observable, contract-tested feature snapshot and an evaluated model for trial-discontinuation signals.

> **Deployment:** this project is integrated into the repository's central `streamlit_app.py`. If Streamlit Community Cloud has not been connected yet, the only remaining manual step is to deploy that root entrypoint once at [share.streamlit.io](https://share.streamlit.io/). No public live URL is claimed until it has been verified.

## Problem and product behavior

Trial registries contain valuable operational signals, but the source is nested, mutable, incomplete and not directly safe for modeling. A useful product must make ingestion, rejection, lineage, leakage boundaries and model limitations visible—not merely draw charts from raw JSON.

The app therefore provides:

- bounded live ingestion for a selectable medical condition;
- a deterministic source-shaped fallback when the API is unavailable or too small;
- content-addressed snapshots with SHA-256 hashes and repeatable run IDs;
- nested JSON normalization, typed fields, deduplication and explicit contract rejection;
- reconciliation, completeness, range, label and leakage checks;
- an auditable feature view restricted to registration/design-time information;
- time-aware holdout evaluation for an explainable logistic classifier;
- ROC AUC, average precision, Brier score, precision, recall, F1 and confusion matrix;
- calibration diagnostics and population-stability monitoring;
- coefficient-based explainability and a non-causal scenario workbench;
- model, pipeline, prediction and manifest exports.

This is an operations-research portfolio demonstration. It is **not** a medical, ethical, regulatory, investment or protocol decision tool.

## End-to-end architecture

```text
ClinicalTrials.gov API v2
        │ bounded GET, timeout, status handling
        ▼
Content-addressed snapshot
        │ canonical payload hashes, source metadata
        ▼
Typed trial contract
        │ flatten, coerce, deduplicate, reject, reconcile
        ▼
Leakage-aware feature view
        │ registration/design-time fields only
        ▼
Time-aware model validation
        │ preprocess → logistic regression → holdout metrics
        ▼
Monitoring and workbench
        └ calibration, PSI drift, coefficients, scenario and audit exports
```

The hosted version executes in memory so it works on Streamlit Community Cloud. The run ID and content hashes are intentionally suitable as object-store keys or orchestration idempotency keys in a production extension.

## Data Engineering implementation

### Extraction and resilience

`src/data.py` requests `GET https://clinicaltrials.gov/api/v2/studies` with:

- `query.cond`: selected condition;
- `filter.overallStatus`: `COMPLETED|TERMINATED|WITHDRAWN|SUSPENDED`;
- `pageSize`: bounded to 80–500;
- `countTotal=true`;
- `format=json`;
- `sort=StudyFirstPostDate:desc`.

The request has separate connect/read timeouts, a two-attempt retry budget for transport failures and `429/502/503/504`, checks HTTP status and JSON shape, and rejects batches with fewer than 50 records. Permanent 4xx failures are not retried. Any request, decoding or minimum-volume failure activates deterministic synthetic records with the same nested v2 shape. The UI always labels fallback mode and exposes the reason. `config.example.toml` documents safe operational defaults and contains no secrets.

### Snapshot and idempotency

Each unmodified record receives a canonical SHA-256 `payload_hash`. Dictionary keys are sorted before hashing, so logically identical payloads generate identical hashes. The `run_id` combines the cleaned condition, complete source hash and feature hash; rerunning identical content is therefore deterministic.

### Data contract and quality

`src/pipeline.py` extracts and validates:

| Module | Fields used |
|---|---|
| `identificationModule` | `nctId`, `briefTitle` |
| `statusModule` | `overallStatus`, first-post, start, completion and update dates |
| `designModule` | study type, phase, enrollment, allocation, masking, primary purpose |
| `sponsorCollaboratorsModule` | lead-sponsor name and class |
| `conditionsModule` | condition count |
| `armsInterventionsModule` | intervention count |
| `contactsLocationsModule` | distinct country count |
| `eligibilityModule` | minimum/maximum age and healthy-volunteer flag |
| top-level | `hasResults` for audit only, never as a model feature |

Records are deduplicated by NCT ID. Invalid IDs, short titles, non-terminal statuses, missing posting dates, non-positive enrollment or implausibly large enrollment are rejected. Quality checks cover uniqueness, allowed statuses, ranges, completeness, row reconciliation, numeric finiteness, class availability, retention and outcome-leakage exclusion.

The operational event ledger records stage, status, input/output/rejected rows, execution time and content hash for extract, snapshot, contract and feature-view stages.

## AI Engineering implementation

### Target and leakage boundary

The label maps `COMPLETED` to 0 and `TERMINATED`, `WITHDRAWN` or `SUSPENDED` to 1. These statuses describe different situations and remain sponsor-reported.

The model only uses fields available from study design/registration:

- phase, study type, sponsor class, allocation, masking and primary purpose;
- log enrollment, condition/intervention/country counts;
- minimum age, age span and healthy-volunteer flag.

Completion date, last update, results availability, status text and any post-outcome field are explicitly excluded.

### Model lifecycle

The pipeline uses median numeric imputation, standardization, most-frequent categorical imputation, unknown-safe one-hot encoding and L2-regularized logistic regression with balanced class weights. The newest 25% of records form the preferred temporal holdout. If either class disappears from that split, a deterministic stratified fallback is used and disclosed in model metadata.

Evaluation includes:

- **ROC AUC:** ranking quality across thresholds;
- **average precision:** ranking performance focused on discontinued records;
- **Brier score:** mean squared probabilistic error (lower is better);
- **accuracy, precision, recall and F1:** threshold behavior at 0.50;
- **confusion matrix:** error counts;
- **calibration bins:** mean score versus observed holdout frequency;
- **PSI drift:** distribution shift from training to holdout for numeric features;
- **outcome shift:** absolute label-rate difference.

Coefficients show conditional associations in model space, not causal effects. Balanced class weights and bounded sampling mean the score must not be interpreted as a population-calibrated probability.

## Important modules and functions

| File | Responsibility |
|---|---|
| `src/data.py` | safe query, API request, bounded batch and deterministic fallback |
| `src/pipeline.py` | hashes, snapshot, normalization, feature view, quality checks and run ledger |
| `src/model.py` | preprocessing, temporal split, model, metrics, calibration, PSI and scenario scoring |
| `ui.py` | responsive control plane, states, charts, tables, controls and exports |
| `config.example.toml` | safe example defaults without credentials |
| `app.py` | standalone Streamlit entrypoint |
| `tests/` | contracts, idempotency, fallback, leakage, reproducibility, metrics, drift and guards |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone:

```bash
streamlit run clinical_trial_ops_pipeline/app.py
```

Tests:

```bash
pytest -q clinical_trial_ops_pipeline/tests
```

No secrets or paid credentials are required.

## Exact source, cadence and usage terms

- **Provider:** ClinicalTrials.gov, maintained by the U.S. National Library of Medicine at the National Institutes of Health.
- **API:** [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api), endpoint `https://clinicaltrials.gov/api/v2/studies`.
- **Structure:** [official study-data structure](https://clinicaltrials.gov/data-api/about-api/study-data-structure).
- **Retrieval:** keyless HTTPS GET, JSON, bounded terminal-study snapshot, no full-text or participant-level data.
- **Update cadence:** records are submitted and updated by responsible parties; the application requests the current registry snapshot and caches it for six hours. Individual records do not share one guaranteed update interval.
- **Terms:** [ClinicalTrials.gov terms and conditions](https://clinicaltrials.gov/about-site/terms-conditions). ClinicalTrials.gov asks users to cite the source; international copyright may apply. The repository stores code and synthetic fallback records, not a republished registry dump.
- **Assumption:** source fields and status reflect the current submitted registry record and may be incomplete, revised or inconsistently reported.

## Operational considerations and limitations

- The hosted job is bounded and in-memory; production should persist raw payloads, manifests and model artifacts in immutable storage.
- API schema changes require contract-version updates and a quarantined-record path.
- Production orchestration should add exponential backoff with jitter, deadlines, alerts, retry budgets and service-level objectives.
- The recent condition-specific snapshot may have selection and label imbalance; it does not estimate global trial prevalence.
- Status categories collapse heterogeneous operational and scientific reasons.
- Registry updates can introduce temporal leakage that is difficult to reconstruct without historical snapshots.
- Drift thresholds are monitoring heuristics, not automated retraining commands.
- Model evaluation measures performance on the retrieved sample only.

## Possible extensions

1. Persist daily immutable snapshots to S3/GCS and query them with DuckDB or Iceberg.
2. Add schema-version negotiation and a dead-letter table for rejected records.
3. Backtest on true historical snapshots instead of current record states.
4. Add calibrated gradient boosting and compare it through a model registry.
5. Add subgroup evaluation by phase, sponsor class, geography and condition.
6. Serve the validated feature contract and model behind a versioned FastAPI endpoint.
7. Add orchestration telemetry, data SLAs, alert routing and retraining approval gates.
