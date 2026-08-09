# NYC 311 Resolution Operations Pipeline

An end-to-end Data Engineering and AI Engineering product that turns mature
NYC 311 service-request records into a contracted data product and an evaluated,
calibrated estimate of resolution-time uncertainty.

> The model describes historical closed cases. It does not define an official
> service-level agreement, prioritize emergencies or promise a completion time.

## Problem and product behavior

Resolution-time analytics are deceptively difficult. The raw system contains many
agencies, changing complaint vocabularies, duplicated snapshots, incomplete labels,
long-running cases and fields created only after work is finished. A useful product
must prevent those post-outcome fields from leaking into an intake-time model.

The hosted application provides:

- bounded, deterministic sampling across a complete historical window;
- paginated Socrata API ingestion with timeouts and bounded retries;
- an atomic, reproducible source-shaped fallback;
- canonical Bronze payload hashes and content-addressed layers;
- a typed Silver contract with stable request keys;
- reason-coded quarantine and exact row reconciliation;
- a Gold view containing only information available when a request is created;
- nine schema, label, reconciliation and temporal-coverage gates;
- stage lineage, timings, hashes and a deterministic run ID;
- conditional median and 90th-percentile resolution models;
- strict chronological train, calibration and untouched test windows;
- a transparent train-only agency/complaint median baseline;
- split calibration of the upper resolution estimate;
- MAE, median absolute error, RMSE, pinball loss, coverage and baseline skill;
- agency-level scorecards and population-stability monitoring;
- an interactive request-intake simulator;
- Silver, holdout-audit and run-manifest exports.

## Architecture and data flow

```text
NYC Open Data / Socrata
        │ mature closed records + deterministic key-modulo sample
        │ pagination, timeout, retry, bounded row budget
        ├──────── failure ───────> deterministic source-shaped fallback
        ▼
Bronze
        │ canonical source JSON + SHA-256 payload hash + ingest sequence
        ▼
Silver contract ────────────────> reason-coded quarantine
        │ typed timestamps, unique key, label bounds, reconciliation
        ▼
Gold intake feature view
        │ only request-time categories and calendar context
        ▼
AI lifecycle
        ├─ older training period
        ├─ 45-day upper-quantile calibration period
        ├─ newest 45-day untouched test period
        └─ baseline, drift and per-agency evaluation
        ▼
Median estimate + calibrated upper estimate + audit manifest
```

## Data Engineering implementation

### Extraction and deterministic sampling

`src/data.py` reads the official Socrata resource:

```text
GET https://data.cityofnewyork.us/resource/erm2-nwe9.json
```

The default query covers 365 historical days but ends 35 days before retrieval.
Only records with a closure timestamp, agency and complaint type are requested.
A stable source-key condition spreads a small reproducible sample across the entire
window:

```sql
(unique_key::number % 1999) < 2
```

This avoids the common mistake of requesting only the newest few thousand rows,
which would cover merely a narrow time slice. Pages are ordered by creation time and
key, fetched sequentially and capped at 6,000 rows. HTTP 429, HTTP 5xx and transport
errors receive one bounded exponential-backoff retry.

The source hash depends only on canonical records, not retrieval time. If the API
fails or returns fewer than 1,200 rows, the entire run switches to deterministic
source-shaped demo data. Live and synthetic records are never mixed.

### Bronze contract

Bronze preserves all selected source fields and adds:

- `payload_hash`: SHA-256 of canonical source JSON;
- `ingest_sequence`: deterministic order within this extraction;
- source-level metadata: query bounds, sample rule, retrieval mode and source hash.

Payload hashes support idempotent replay, unchanged-record detection and later
migration to immutable object storage.

### Silver contract and quarantine

| Field | Contract |
|---|---|
| `unique_key` | Required and unique 311 request identifier |
| `payload_hash` | 64-character lowercase SHA-256 |
| `created_at` | Valid UTC request timestamp |
| `closed_at` | Valid UTC closure timestamp |
| `agency` | Required responding agency code |
| `agency_name` | Human-readable agency name when supplied |
| `complaint_type` | Required request category |
| `descriptor` | Detail available at intake; missing becomes `Unknown` |
| `location_type` | General location category, never a street address |
| `borough` | Borough or explicit `UNSPECIFIED` |
| `open_data_channel_type` | Intake channel or explicit `UNKNOWN` |
| `resolution_hours` | Closure minus creation; 0–720 hours |

Missing keys, invalid timestamps, negative durations, durations over 30 days and
duplicate keys enter quarantine with the first applicable reason. Silver plus
quarantine must reconcile exactly to Bronze.

### Gold feature view and leakage boundary

Gold derives only request-time features:

- agency, complaint, descriptor, location type, borough and intake channel;
- creation hour, weekday and month;
- weekend and overnight flags.

The closure timestamp, current status, resolution description and resolution-action
timestamp are never model inputs. `resolution_hours` is retained only as the target.
Street addresses, coordinates and free-text resolution descriptions are not ingested.

### Quality and observability

Nine gates validate payload hashes, exact row reconciliation, key uniqueness,
required dimensions, event ordering, the 30-day label boundary, intake-only
features, minimum row count and at least 150 distinct creation dates.

The stage ledger records input, output, rejection count, execution time and layer
content hash. The run ID combines the source and layer hashes; replaying identical
records produces the same run and layer identities.

## AI Engineering implementation

### Conditional quantile models

Two `HistGradientBoostingRegressor` pipelines are trained on `log1p` resolution
hours:

- quantile 0.50 estimates the conditional median;
- quantile 0.90 estimates an upper planning boundary.

Categorical fields are one-hot encoded with infrequent-category grouping and safe
unknown-category handling. Calendar fields remain numeric. Predictions are
transformed back with `expm1`, clipped at zero, and guarded so the upper estimate can
never be below the median estimate.

These are conditional estimates for similar historical requests—not a causal model.

### Chronological evaluation

Random splitting would allow future operating patterns into training. Instead:

1. older eligible requests form training data;
2. the next 45 days form the calibration set;
3. the newest 45 days remain untouched until final scoring.

The explicit baseline is the training-period median for the matching agency and
complaint. Unseen pairs fall back to agency median and then global median. No test
labels contribute to the baseline.

### Upper-bound calibration

The raw 90th-percentile model is evaluated on the separate calibration window. The
90th percentile of positive residuals `actual − predicted_q90` becomes a non-negative
correction. That correction is frozen before the untouched holdout is evaluated.

This is a one-sided split-conformal-style correction. It targets empirical upper
coverage but cannot guarantee future coverage after distribution shift.

### Evaluation metrics

- **MAE:** mean absolute median-estimate error in hours.
- **Median AE:** typical absolute error, robust to extreme durations.
- **RMSE:** emphasizes costly large misses.
- **Baseline MAE:** agency/complaint group-median benchmark.
- **Skill:** `1 − model MAE / baseline MAE`; negative means baseline wins.
- **Pinball loss q50/q90:** quantile-specific asymmetric loss.
- **Upper coverage:** share of holdout labels below the calibrated upper estimate.
- **≤24-hour accuracy:** agreement on a simple operational duration band.

All core behavior is reported globally and per agency. Poor slices remain visible.

### Drift and failure states

Population Stability Index compares training and holdout distributions for hour,
agency, complaint, borough and channel:

- PSI below 0.10: stable;
- 0.10–0.25: watch;
- above 0.25: investigate.

PSI is a diagnostic, not an automatic retraining trigger. Unknown simulator
categories are accepted safely by the encoder, but the interface warns that rare or
unseen combinations can still be unreliable. Runs with insufficient temporal split
sizes or failed data gates publish no prediction.

## Exact data source

- **Provider:** City of New York, NYC Open Data.
- **Dataset:** [311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data).
- **API documentation:** [Socrata resource `erm2-nwe9`](https://dev.socrata.com/foundry/data.cityofnewyork.us/erm2-nwe9).
- **Endpoint:** `https://data.cityofnewyork.us/resource/erm2-nwe9.json`.
- **Update cadence:** the dataset is operational and continuously receives new and updated service-request records; the app retrieves a bounded snapshot and caches it for six hours.
- **Credentials:** no API key or paid account is required for this bounded public query.
- **Fields used:** `unique_key`, `created_date`, `closed_date`, `agency`, `agency_name`, `complaint_type`, `descriptor`, `location_type`, `borough`, `open_data_channel_type`, `status`, `resolution_action_updated_date`.
- **Policy context:** [NYC Open Data overview](https://opendata.cityofnewyork.us/overview/) and the dataset page govern public access and source attribution.

The application deliberately does not ingest incident address, cross streets,
coordinates, caller details or resolution-description text.

## Modules and important functions

| File | Responsibility |
|---|---|
| `src/data.py` | Date bounds, key sampling, pagination, retry, schema-shaped fallback |
| `src/pipeline.py` | Bronze hashes, Silver contract, quarantine, Gold features, DQ and lineage |
| `src/model.py` | Temporal split, quantile pipelines, calibration, baseline, metrics, PSI and simulator |
| `ui.py` | Responsive control plane, operational views, evaluation and exports |
| `tests/test_pipeline.py` | Contract, quarantine, reconciliation, idempotency, retry and fallback tests |
| `tests/test_model.py` | Reproducibility, shapes, metric domains, unknown-category and edge-case tests |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Choose **NYC 311 Resolution Operations Pipeline** in the central navigation. For
standalone execution:

```bash
streamlit run nyc_311_resolution_pipeline/app.py
```

No secrets are required. `config.example.toml` documents safe defaults.

## Testing

```bash
python -m pytest -q nyc_311_resolution_pipeline/tests
```

Tests cover contract types, reconciliation, duplicate and invalid-duration
quarantine, deterministic layer hashes, fallback reproducibility, transient retry,
configuration failure, model output shape, metric domains, quantile ordering,
reproducibility, unknown categories and insufficient-data guards.

## Operational considerations

- Run after the chosen label-maturity delay and store query metadata with every snapshot.
- Persist Bronze records immutably by payload hash and Silver/Gold as partitioned Parquet.
- Upsert source snapshots by `unique_key`; preserve changed payload hashes as history.
- Monitor row volume, quarantine rate, content-hash changes and temporal coverage.
- Register feature schema, training hash, calibration correction and metrics together.
- Gate promotion on baseline skill, upper coverage and minimum agency slice size.
- Recalibrate before retraining automatically; investigate PSI alongside source changes.
- Keep official emergency and service workflows completely outside this advisory model.

## Limitations

- Only closed cases receive labels; still-open cases are absent and can create selection bias.
- Ending 35 days earlier and excluding durations over 30 days reduces but does not remove censoring.
- Deterministic key sampling is reproducible but may differ slightly from a truly random sample.
- Agency processes and complaint vocabularies change over time.
- Resolution timestamps measure recorded closure, not necessarily physical completion or satisfaction.
- Rare category combinations may have weak support.
- A single global model can hide agency-specific behavior despite visible scorecards.
- The 90% upper estimate is empirical and may miss under operational change.
- The hosted runtime does not persist state between sessions.

## Extensions

1. Persist source change-data capture in object storage with Iceberg history.
2. Add open-request survival analysis rather than excluding censored observations.
3. Train agency-specific models when minimum slice sizes are satisfied.
4. Use hierarchical target encoding learned inside each training fold.
5. Add repeated rolling-origin backtests and promotion thresholds.
6. Monitor delayed labels, interval coverage and calibration by agency in production.
7. Serve versioned estimates through FastAPI with feature and model registries.
8. Add model cards, approval workflow and human outcome feedback.

## Hosted use

The page is integrated into the stable root `streamlit_app.py`. An existing
Streamlit Community Cloud deployment updates automatically from `main`. Because no
confirmed public URL is stored in the repository, no live URL is invented. If the
portfolio is not connected yet, the single manual step is to deploy
`streamlit_app.py` once at [share.streamlit.io](https://share.streamlit.io/).
