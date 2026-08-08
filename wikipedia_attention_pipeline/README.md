# Wikipedia Attention Forecast Pipeline

An end-to-end Data Engineering and AI Engineering product that replays public
Wikipedia attention aggregates through an event-time pipeline and produces an
evaluated multi-series forecast with uncertainty, anomalies and failure states.

> Pageviews describe observed attention. They do not measure sentiment, truth,
> importance, intent or the cause of a traffic change.

## Problem and product behavior

Public pageview series look simple, but a production workflow must handle source
delay, repeated delivery, out-of-order events, changing scale and forecast
uncertainty. This product makes those responsibilities visible instead of hiding
them behind a line chart.

The hosted application provides:

- a selectable portfolio of eight technology articles;
- bounded daily pageview ingestion from the Wikimedia Analytics API;
- a compliant identifying user agent, timeout, retry budget and atomic fallback;
- deterministic event replay with duplicates and late arrivals;
- Bronze payload hashes, stable event IDs and micro-batch IDs;
- event-time watermarks and an operational batch ledger;
- a typed Silver contract with quarantine and reconciliation;
- a leakage-safe Gold forecast feature view;
- nine data-quality and temporal-integrity gates;
- a global gradient-boosted forecast across multiple article series;
- strict train/calibration/test windows in chronological order;
- weekly-naive baseline comparison and skill measurement;
- split-conformal 90% prediction intervals;
- WAPE, SMAPE, MAE, RMSE, coverage and per-series evaluation;
- an outside-interval anomaly review queue;
- a recursive 7-, 14- or 21-day planning forecast;
- CSV exports and a complete operational JSON manifest;
- a deterministic source-shaped fallback when the API is unavailable.

## Architecture and data flow

```text
Wikimedia Analytics API
        │ sequential bounded GETs, timeout, retry, user agent
        ├──────── failure ───────> deterministic source-shaped fallback
        ▼
Bronze event replay
        │ event ID + payload SHA-256 + arrival delay + micro-batch
        ▼
Event-time processor
        │ two-day watermark, late-event flags, duplicate delivery
        ▼
Silver daily contract ───────────> reason-coded quarantine
        │ typed fields, dedupe, reconciliation
        ▼
Gold forecast view
        │ shifted lags + rolling history + calendar features
        ▼
Rolling-origin AI lifecycle
        ├─ train window
        ├─ 28-day conformal calibration window
        ├─ untouched 28-day test window
        └─ full-history production refit
        ▼
Forecast, intervals, anomaly queue and manifest
```

The hosted app uses deterministic in-memory replay so streaming behavior is
demonstrable on Streamlit Community Cloud. A production version can persist the
same content hashes, batch ledger and watermarks in an object store and stream
processor without changing the contracts.

## Data Engineering implementation

### Extraction

`src/data.py` calls one official endpoint per selected article:

```text
GET https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
    en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}
```

The query fixes the following semantics:

- project: English Wikipedia;
- access: all access methods;
- agent: user traffic;
- granularity: daily;
- date range: 120–300 days in the UI, ending two days before retrieval.

Requests are sequential, limited to the selected articles and cached for six
hours. Each uses separate connect/read timeouts and an identifying user agent.
Transport errors, HTTP 429 and HTTP 5xx receive one bounded exponential-backoff
retry. Missing or less than 90% complete history triggers one atomic fallback:
live and synthetic series are never mixed.

### Event-time micro-batch replay

The source provides daily aggregates, not an event bus. The application says so
explicitly and replays those aggregates deterministically:

1. `event_id = project:article:timestamp`.
2. Canonical JSON produces a SHA-256 payload hash.
3. A stable hash-derived delay assigns `arrival_time` after `event_time`.
4. Selected events are deliberately replayed a second time.
5. Arrival-ordered records are divided into configurable micro-batches.
6. Each batch advances `watermark = max(previous watermark, max event time − 2 days)`.
7. Events older than the previous watermark are marked late.

This demonstrates at-least-once delivery, out-of-order arrival and event-time
processing without pretending that Wikimedia exposes a live pageview stream.

### Silver contract and quarantine

| Field | Meaning / validation |
|---|---|
| `event_id` | Required unique project/article/timestamp identity |
| `payload_hash` | Canonical raw-payload SHA-256 |
| `event_time` | Parsed UTC source timestamp |
| `arrival_time` | Deterministic replay arrival timestamp |
| `batch_id` | Replay micro-batch identifier |
| `late_beyond_watermark` | Event-time observability flag |
| `project` | Must be `en.wikipedia` |
| `article` | Required page title |
| `granularity` | Must be `daily` |
| `access`, `agent` | Source traffic dimensions |
| `views` | Integer and non-negative |

Duplicate IDs, missing article names, wrong projects/granularity, invalid event
times and negative/non-numeric views enter quarantine with an explicit reason.
Silver plus quarantine must reconcile exactly to Bronze.

### Gold feature contract

Within each article series, the Gold layer creates:

- previous-day views (`lag_1`);
- weekly and two-week lags (`lag_7`, `lag_14`);
- shifted seven- and 28-day medians;
- shifted 28-day standard deviation;
- weekday sine/cosine cycle;
- deterministic series trend;
- article identity used by the global model.

Every rolling window is shifted by one day before calculation. The current target
is therefore never included in its own features. The first 28 article-days are
removed only after features are constructed.

### Quality and observability

Nine gates cover Bronze identity, Silver uniqueness, non-negative counts, daily
source semantics, event-time completeness, Bronze/Silver/quarantine reconciliation,
complete Gold lags, minimum article history and future-leakage prevention.

The stage ledger records input/output/rejected rows, execution duration and content
hash. The batch ledger records unique IDs, duplicates, late events, maximum event
time and watermark. Replaying identical source records produces identical layer
hashes and the same run ID.

## AI Engineering implementation

### Forecast model

One global `HistGradientBoostingRegressor` learns all selected series jointly.
The target is `log1p(views)`, reducing the dominance of large spikes and
high-volume pages. Predictions are transformed back with `expm1` and clipped at
zero. The model uses fixed hyperparameters and random seed 42 for reproducibility.

The article index allows shared temporal patterns while retaining series-specific
splits. Because it is ordinal rather than one-hot, the index should be replaced by
an embedding or categorical-native estimator in a larger production model.

### Chronological evaluation

No random split is used:

1. all older eligible rows form training data;
2. the next 28 days form the calibration window;
3. the newest 28 days remain untouched until final evaluation;
4. only after evaluation is complete is a production estimator refit on all Gold rows.

The explicit baseline predicts each day with its value seven days earlier.
`skill_vs_weekly_naive = 1 − model WAPE / baseline WAPE`; positive skill means the
model improves on the weekly-naive baseline.

### Metrics

- **MAE:** average absolute view-count error.
- **RMSE:** square-root mean squared error, emphasizing large misses.
- **WAPE:** total absolute error divided by total actual views.
- **SMAPE:** symmetric percentage error per observation.
- **Weekly-naive WAPE:** transparent lag-7 benchmark.
- **Skill:** relative WAPE improvement over that benchmark.
- **Interval coverage:** share of test observations inside the 90% interval.
- **Mean interval width:** operational cost of uncertainty.

Metrics are reported globally and per article. A negative skill score is not hidden:
it means the model failed to beat a simple seasonal baseline on that slice.

### Conformal intervals and anomalies

Absolute residuals on the separate calibration window produce the empirical 90th
percentile radius. The radius is added to/subtracted from test predictions without
using test labels. Coverage is then measured on the untouched holdout.

Any actual observation outside that interval enters the anomaly queue with direction,
absolute residual and severity. This means "unexpected under the recent model", not
"important", "positive", "negative" or "caused by a particular event".

### Forward forecast

After evaluation, the production model refits on all validated Gold history. Future
days are generated recursively: each predicted value becomes history for the next
step. The same calibration radius is shown around the forecast. Because recursive
error compounds and the radius is fixed, longer-horizon uncertainty is approximate
and clearly documented.

## Exact data source

- **Provider:** Wikimedia Foundation.
- **Dataset/API:** [Wikimedia Analytics API – Page view analytics](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html).
- **Endpoint:** `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/...`.
- **Coverage:** the official documentation states that pageview endpoints serve data from July 1, 2015 onward.
- **Retrieval cadence:** daily aggregates; the app requests the current bounded history and caches it for six hours.
- **Completeness assumption:** recent aggregates can require a full day or more, so this app deliberately ends two days before retrieval and never converts missing days to zero.
- **Credentials:** no API key or paid account required.
- **Fields used:** `project`, `article`, `granularity`, `timestamp`, `access`, `agent`, `views`.
- **Usage requirements:** the app follows the official [API Usage Guidelines](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines), including identification, rate-limit compliance and sequential bounded access.
- **Terms:** [Wikimedia Foundation Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

The application uses aggregate counts and article titles, not article text, user
identifiers or personal data. It identifies itself independently and does not imply
Wikimedia Foundation sponsorship or endorsement.

## Modules and important functions

| File | Responsibility |
|---|---|
| `src/data.py` | Allowlist, date bounds, respectful requests, retries, schema check and fallback |
| `src/pipeline.py` | Bronze replay, watermarks, Silver contract, quarantine, Gold lags, DQ and hashes |
| `src/model.py` | Chronological splits, gradient boosting, baseline, conformal intervals, anomalies and future forecast |
| `ui.py` | Responsive control plane, evaluation, operations views, states and exports |
| `tests/test_pipeline.py` | Schema, idempotency, watermarks, duplicates, lateness, retry and fallback tests |
| `tests/test_model.py` | Reproducibility, output shape, metrics, intervals, anomaly and edge-case tests |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Select **Wikipedia Attention Forecast Pipeline** in the central navigation.
Standalone execution is also supported:

```bash
streamlit run wikipedia_attention_pipeline/app.py
```

No secrets are required. `config.example.toml` documents safe defaults.

## Testing

```bash
python -m pytest -q wikipedia_attention_pipeline/tests
```

Tests cover source allowlisting, deterministic fallback, content hashes, replayed
duplicates, monotonic watermarks, lateness reconciliation, typed contracts, shifted
feature correctness, idempotency, retry recovery, failure-closed input, model
reproducibility, output shapes, metric domains, interval breaches and small-data guards.

## Operational considerations

- Schedule one daily run after the source's expected publication delay.
- Persist Bronze payloads and batch metadata immutably by content hash.
- Store the Silver contract and quarantine as separate partitioned datasets.
- Maintain durable source watermarks rather than recomputing replay state.
- Alert on missing dates, quarantine spikes, stalled watermarks and API retry exhaustion.
- Register model code, article map, feature schema, training hash, interval radius and metrics together.
- Gate promotion on baseline skill, interval coverage and article-level minimum volume.
- Recalibrate intervals and review anomaly outcomes before retraining automatically.

## Limitations

- The hosted replay is a deterministic simulation, not a real message broker.
- Pageview counts can be revised, delayed or affected by source definitions.
- Article renames and redirects can split attention history.
- A bounded portfolio is not representative of Wikipedia overall.
- Attention cannot explain sentiment, causality or real-world importance.
- Gradient boosting cannot extrapolate sudden unprecedented events reliably.
- One global conformal radius may over-cover small series and under-cover large ones.
- Recursive intervals do not fully capture increasing multi-step uncertainty.
- The hosted runtime does not persist state between Streamlit sessions.

## Extensions

1. Replace replay with Kafka/Redpanda and Flink event-time windows.
2. Persist Bronze/Silver/Gold Parquet in S3 with Iceberg table metadata.
3. Add article-rename and redirect resolution through the MediaWiki API.
4. Use per-series or normalized conformal intervals.
5. Add blocked rolling-origin backtests across several historical cutoffs.
6. Compare LightGBM, probabilistic boosting and temporal foundation models.
7. Serve versioned forecasts through FastAPI with a feature/model registry.
8. Add delayed-label monitoring, alert acknowledgement and incident annotations.

## Hosted use

The page is integrated into the repository's stable root `streamlit_app.py`.
An existing Streamlit Community Cloud deployment updates automatically from `main`.
Because no confirmed public URL is stored in the repository, no live URL is invented.
If deployment has not yet been connected, the single remaining manual step is to
deploy `streamlit_app.py` once at [share.streamlit.io](https://share.streamlit.io/).
