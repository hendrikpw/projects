# Household Load Forecast Control

A deployment-ready Data Engineering and AI Engineering product that turns more
than two million minute-level household electricity readings into a governed
hourly data product and an evaluated 24-hour-ahead forecasting service.

The project is integrated into the repository's central Streamlit application.
When that application is deployed, select **Household Load Forecast Control** in
the sidebar. If no deployment exists, choose this repository, branch `main` and
root entrypoint `streamlit_app.py` in Streamlit Community Cloud.

## Problem

Load forecasting is not just a regression problem. Before a prediction can be
trusted, an operational system must answer:

- Did every source delivery arrive exactly once?
- Is the event timestamp valid and ordered?
- Were missing minutes silently converted to zero?
- Which gaps were imputed, and which were withheld?
- Does the published data reconcile with input, quarantine and replay counts?
- Was the model evaluated on genuinely later history?
- Does it beat a strong same-hour persistence baseline?
- Does the uncertainty interval achieve its claimed coverage?
- What happens when recent load features are missing or outside training range?

This application makes those questions visible alongside the forecast itself.

## Product behavior

### Pipeline control

- safely downloads and verifies the official ZIP archive;
- parses the 126.8 MB text member in bounded chunks;
- aggregates minute readings into hourly weighted means and energy sums;
- assigns stable event IDs and SHA-256 payload hashes;
- simulates weekly micro-batches, late events and duplicate deliveries;
- applies typed contracts and reasoned quarantine;
- interpolates only short hourly gaps;
- withholds unresolved long-gap hours;
- reconciles every delivery through Bronze, Silver and Gold;
- publishes stage, batch, quality and lineage ledgers.

### Forecast evaluation

- predicts household active power 24 hours ahead;
- uses only information available at forecast issue time;
- keeps train, calibration and test periods strictly chronological;
- compares against same-hour persistence;
- calibrates an empirical 80% forecast interval;
- reports MAE, RMSE, MAPE, R², pinball loss and interval coverage;
- measures peak capture under a 10% review budget;
- exposes permutation importance and PSI feature drift.

### Serving workbench

The hosted app can replay a recent forecast issue and inject:

- a recent-load scale shift;
- one to six missing lag features.

The serving route becomes:

- `auto-forecast` for complete in-domain inputs;
- `review` for a limited input or distribution warning;
- `forecast-withheld` when missingness or out-of-domain evidence exceeds the
  configured guardrail.

The workbench demonstrates failure behavior without requiring external
infrastructure or user credentials.

## Architecture and data flow

```text
UCI versioned ZIP
    |
    | retry + timeout + byte bounds + ZIP/path contract
    v
Chunked minute parser
    |
    | UTC timestamp parsing + numeric coercion
    | hourly weighted means and energy sums
    v
Bronze hourly deliveries
    |
    | event ID + payload hash + delivery sequence
    | weekly micro-batch + watermark + intentional replay
    v
Silver contracted events ---------> reasoned quarantine
    |
    | global replay suppression + temporal sort
    v
Gold continuous hourly product
    |
    | short-gap interpolation + long-gap withholding
    | lag, rolling and calendar features
    v
Chronological train / calibration / untouched test
    |
    | point residual model + lower/upper quantile models
    | blend selection + conformal interval adjustment
    v
Evaluation, drift, serving guardrails and audit exports
```

## Data Engineering implementation

### Safe extraction

`src/data.py` downloads the archive with:

- an explicit portfolio User-Agent;
- a 45-second timeout;
- three attempts with exponential backoff;
- accepted HTTP status handling;
- a 10–30 MB compressed-size contract;
- ZIP signature verification;
- path-traversal rejection;
- an exact allowlist for `household_power_consumption.txt`;
- a 100–150 MB expanded-member contract.

The parser reads 250,000 rows at a time. It never needs to materialize the full
text table in memory.

### Hourly transformation

For each chunk:

1. `Date` and `Time` are combined with the documented day-first format.
2. Measurement strings and `?` markers are coerced explicitly.
3. Rows without a valid timestamp or active-power measurement are counted as
   source-level missing rows, not zero consumption.
4. Power, voltage and intensity become reading-count-weighted hourly means.
5. The three sub-metering channels become hourly energy sums.
6. Partial aggregates at chunk boundaries are merged by timestamp.

### Event identity and idempotence

An hourly event ID is the first 24 hexadecimal characters of:

```text
SHA256(timestamp_utc_iso8601)
```

The payload hash contains timestamp, seven measurements and reading count.
Twenty-four events are intentionally delivered twice at the end of the stream.
Replay detection is global, so duplicates are suppressed even when the original
and repeated delivery occur in different batches.

### Event time and watermarks

- A micro-batch contains 168 deliveries, representing one nominal week.
- The system records the maximum event time seen before each delivery.
- Lateness is the positive difference between that maximum and event time.
- Events more than 48 hours behind the current maximum are marked as arriving
  after the watermark.

The watermark is observable metadata. It does not silently alter the timestamp.

### Data contracts

Silver accepts an observed hour only if:

- event ID and payload hash have the expected hexadecimal shape;
- timestamp and all seven measurements are finite;
- active power lies between 0 and 20 kW;
- voltage lies between 150 and 300 V;
- current lies between 0 and 100 A;
- all sub-metering values are non-negative;
- reading count lies between 1 and 60.

The first failed rule becomes the quarantine reason.

### Missing-hour policy

After Silver, the system constructs the complete hourly UTC index:

- gaps of at most six consecutive hours are interpolated in time;
- imputed rows are marked with `was_imputed = true` and `readings = 0`;
- longer unresolved gaps are withheld from Gold;
- no remaining model value may be missing or infinite.

This policy is an explicit portfolio assumption, not an official UCI procedure.

### Publication gates

Gold is published only when all ten checks pass:

1. at least one million source-minute deliveries;
2. at least 30,000 usable Gold hours;
3. at least 95% acceptance of observed hourly records;
4. unique stable event IDs;
5. exact suppression of all intentional replays;
6. complete row reconciliation;
7. finite Gold measurements;
8. missing-hour and unresolved-gap budgets below configured limits;
9. strictly increasing, non-duplicated event time;
10. at least 1,300 days of history.

Any failed gate stops publication and produces a visible Streamlit error state.

### Lineage and observability

The run manifest includes:

- archive SHA-256;
- Bronze, Silver, Gold and batch-ledger SHA-256 hashes;
- deterministic run ID;
- input, output, replay, quarantine and gap counts;
- stage runtime;
- first and last event time;
- per-batch delivery, uniqueness, replay and maximum-lateness metrics.

Reprocessing identical content produces identical layer hashes and run ID.

## AI Engineering implementation

### Forecast target

At hourly issue time `t`, the target is:

```text
y(t) = active_power(t + 24 hours)
```

The direct 24-hour horizon avoids recursively feeding model predictions back
into later horizons.

### Features

All features are known at issue time:

- current active power;
- active-power lags at 1, 2, 3, 24, 48 and 168 hours;
- 24-hour mean and standard deviation;
- 168-hour mean and standard deviation;
- current voltage and current intensity;
- share of energy represented by the three sub-meters;
- sine/cosine encodings for hour, weekday and day of year.

No future load, future sensor value or test-derived statistic is used.

### Residual forecasting

Same-hour persistence predicts tomorrow's load as today's current load:

```text
baseline(t + 24) = load(t)
```

The gradient-boosting models learn the residual:

```text
residual(t) = load(t + 24) - load(t)
forecast = load(t) + weight * predicted_residual
```

The blend weight is chosen only on the calibration period from values between
0 and 1. A weight of zero would safely fall back to persistence.

### Models

Three deterministic `HistGradientBoostingRegressor` models are fitted:

- squared-error point residual;
- 10th-percentile residual;
- 90th-percentile residual.

All use 170 boosting iterations, learning rate 0.06, at most 31 leaves,
minimum leaf size 30, L2 regularization and random seed 42.

### Chronological lifecycle

After lag construction, rows are divided without shuffling:

- oldest 70%: model training;
- next 15%: blend and interval calibration;
- newest 15%: untouched final test.

For the verified live run:

- 23,842 training hours, through 16 September 2009;
- 5,109 calibration hours, through 20 April 2010;
- 5,110 test hours, through 26 November 2010.

### Conformal interval adjustment

For every calibration row, nonconformity is:

```text
max(lower - actual, actual - upper, 0)
```

The empirical 80th percentile is added symmetrically to the quantile interval.
Coverage is then measured only on the later untouched test. This is marginal
historical coverage under the observed split, not a guarantee for new homes or
future structural changes.

### Promotion policy

The AI artifact is rejected unless:

- test MAE is no worse than same-hour persistence; and
- empirical interval coverage lies between 70% and 95%.

The upper coverage bound helps detect a trivially broad interval that would
pass only by being operationally uninformative.

### Drift and explainability

- Permutation importance measures the increase in test MAE after shuffling a
  feature.
- Population Stability Index compares train and test feature distributions.
- PSI above 0.25 is displayed as a strong distribution-shift signal.

High PSI is not hidden and does not itself prove model failure. In this static
household history it warns that later behavior and seasons differ from training.

### Serving guardrails

Training 0.1% and 99.9% feature quantiles define a broad operating envelope.

- Complete inputs with fewer than one OOD feature use `auto-forecast`.
- Limited missingness or OOD evidence uses `review`.
- More than 10% missing features or at least four OOD features uses
  `forecast-withheld`.

Missing values are passed to the model only for a demonstrable guarded result;
they never receive an unconditional production route.

## Verified live result

The live validation on 21 August 2026 produced:

### Pipeline

- 2,075,259 minute source rows;
- 25,979 source rows without a usable active-power measurement;
- 34,168 observed hourly aggregates;
- 34,253 published Gold hours after continuity processing;
- 24 of 24 intentional replays suppressed;
- zero observed hourly records quarantined;
- 85 short-gap hours interpolated;
- 336 long-gap hours withheld;
- all ten publication gates passed;
- deterministic run ID `1a5d79cad937`.

### Forecast

- test MAE: **0.427 kW**;
- persistence MAE: **0.514 kW**;
- RMSE: **0.581 kW**;
- R²: **0.329**;
- empirical 80% interval coverage: **81.98%**;
- mean interval width: **1.299 kW**;
- top-10% peak capture: **37.18%**;
- persistence peak capture: **30.72%**;
- mean inference time: approximately **0.002 ms per hour**.

MAPE is reported in the app but is unstable near very low loads, so MAE is the
primary promotion metric.

## Data source

### Provider

UCI Machine Learning Repository:
[Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

- DOI: [10.24432/C58K54](https://doi.org/10.24432/C58K54)
- Direct archive:
  [individual+household+electric+power+consumption.zip](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)
- Creators: Georges Hebrail and Alice Berard
- Donated to UCI: 29 August 2012
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Update cadence: static historical dataset; no scheduled refresh is stated
- Coverage: 16 December 2006 through 26 November 2010
- Location: one household in Sceaux, approximately 7 km from Paris, France
- Sampling: one-minute measurements

### Fields used

| Source field | Meaning | Unit | Transformation |
|---|---|---:|---|
| `Date` | local calendar date | dd/mm/yyyy | combined with time, represented as UTC for deterministic processing |
| `Time` | local clock time | hh:mm:ss | combined with date and floored to hour |
| `Global_active_power` | minute-averaged active power | kW | hourly reading-count-weighted mean and forecast target |
| `Global_reactive_power` | minute-averaged reactive power | kW | hourly weighted mean |
| `Voltage` | minute-averaged voltage | V | hourly weighted mean and model feature |
| `Global_intensity` | minute-averaged current | A | hourly weighted mean and model feature |
| `Sub_metering_1` | kitchen active energy | Wh | hourly sum |
| `Sub_metering_2` | laundry-room active energy | Wh | hourly sum |
| `Sub_metering_3` | water-heater / air-conditioner energy | Wh | hourly sum |

The source timestamps do not include an explicit timezone. The pipeline assigns
UTC to make event processing deterministic. Therefore displayed UTC should not
be interpreted as a verified conversion from French civil time, especially
around daylight-saving transitions.

## Reproducible fallback

If download, HTTP, ZIP, size, schema or parsing checks fail, `fallback_data()`
creates 34,344 deterministic hourly observations with:

- daily and twice-daily demand cycles;
- weekday/weekend behavior;
- annual seasonality;
- autoregressive noise;
- physically plausible voltage and sub-metering channels;
- the same data contract and AI lifecycle as the live source.

The UI clearly marks demo mode and includes the exact fallback reason. It never
presents generated values as UCI observations.

## Important modules

| Module | Responsibilities |
|---|---|
| `src/data.py` | retrying download, safe ZIP contract, chunked parsing, hourly aggregation and fallback |
| `src/pipeline.py` | hashes, micro-batches, watermarks, contracts, replay suppression, gaps, quality gates and lineage |
| `src/model.py` | feature engineering, temporal lifecycle, residual/quantile models, calibration, evaluation, drift and serving |
| `ui.py` | responsive pipeline control plane, forecast diagnostics, workbench, error states and exports |
| `tests/test_forecast.py` | source, contract, idempotence, model, reproducibility and guardrail tests |
| `app.py` | standalone Streamlit entrypoint |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone project page:

```bash
streamlit run household_load_forecast/app.py
```

Run project tests:

```bash
PYTHONPATH=. pytest -q household_load_forecast/tests
```

The application requires no API key, paid credential or secret.

## Test coverage

The project tests:

- deterministic fallback output;
- missing allowlisted archive member;
- ZIP path traversal rejection;
- chunk-boundary hourly aggregation;
- exact replay and watermark behavior;
- typed quarantine reasons;
- row reconciliation and publication gates;
- deterministic Gold hashes and run IDs;
- feature shape and 24-hour target alignment;
- chronological partition ordering;
- MAE improvement over persistence;
- interval ordering and coverage;
- deterministic model output;
- normal serving output;
- missing-feature and OOD fail-safe routes.

## Limitations

- One household cannot represent a neighborhood, utility territory or country.
- The history ends in 2010 and does not reflect current devices, tariffs,
  electrified heating, solar generation or electric vehicles.
- The source does not provide weather, occupancy, holidays or tariffs.
- Missing intervals may be non-random; interpolation can reduce true peaks.
- Assigning UTC is a deterministic engineering convention, not a verified
  source-timezone conversion.
- The chronological 70/15/15 split is one retrospective evaluation, not full
  rolling-origin cross-validation.
- PSI is sensitive to seasonality because train and test cover different dates.
- Forecast intervals quantify observed residual uncertainty, not data-source,
  equipment or societal uncertainty.
- Scenario controls perturb engineered features; they are diagnostics, not
  causal estimates.
- No automated retraining, artifact registry, durable database, queue broker or
  alerting backend runs inside Streamlit Community Cloud.

## Possible extensions

- rolling-origin evaluation across multiple seasons;
- a real streaming implementation with Kafka or Redpanda;
- Parquet/Iceberg layers and DuckDB validation queries;
- explicit source-timezone and daylight-saving reconciliation;
- weather and holiday features from a documented public source;
- multi-household hierarchical forecasts;
- probabilistic scoring with CRPS and coverage by load regime;
- experiment tracking and signed model artifacts;
- scheduled drift alerts and champion/challenger promotion;
- FastAPI forecast endpoint with typed request/response contracts;
- durable feature store and online/offline consistency tests.

## Responsible interpretation

This project demonstrates engineering and model-control patterns. It is not an
official UCI application, a utility forecast, an energy-saving recommendation or
a safety-critical control system. The source and creators are credited; no
endorsement is implied.
