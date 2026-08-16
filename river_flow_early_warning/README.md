# USGS River Flow Early-Warning Pipeline

A portfolio-ready Data Engineering and AI Engineering product that converts public U.S. river-gauge observations into a governed daily data product and an evaluated three-day high-flow review signal.

The application is integrated into the repository's central `streamlit_app.py` and requires no API key or secret. If the repository has not yet been connected to Streamlit Community Cloud, the only remaining deployment step is to select branch `main`, entrypoint `streamlit_app.py`, and click **Deploy**.

## Problem and product behavior

Streamflow observations arrive from many monitoring locations, can be provisional, may later be revised and have very different scales. A Mississippi reading cannot be compared directly with a smaller mountain river. A useful warning product therefore needs controlled ingestion, station-aware semantics, temporal evaluation and strong safety boundaries.

This application:

- retrieves more than eight years of daily mean discharge for six USGS stations in one bounded request;
- preserves station, observation date, unit semantics, qualifier, delivery key and record hash;
- deliberately replays two deliveries per station to demonstrate idempotent suppression;
- validates and quarantines invalid station IDs, dates, missing values and negative discharge;
- publishes Bronze, Silver and Gold layers only after ten quality and reconciliation gates pass;
- creates lag-only features and a future three-day high-flow label without leaking future observations into model inputs;
- separates 2018–2023 training, 2024 calibration and 2025 onward testing;
- calibrates probabilities and compares ranking with a station/month climatology baseline;
- exposes review-budget capture, calibration reliability, gauge-specific metrics, PSI drift and an interactive latest-flow scenario;
- labels source outages, model failures, provisional observations and interpretation limits.

The output is not an official flood forecast. “High flow” is a portfolio modeling definition, not flood stage or expected damage.

## Architecture and data flow

```text
USGS Daily Values RDB · parameter 00060 · statistic 00003
          │ identified User-Agent · timeout · retry · response bounds
          ▼
Bronze    station-day deliveries + delivery/event IDs + batch metadata
          │ typed contract · range rules · replay-safe deduplication
          ▼
Silver    one governed station-day + qualifier + record hash + location
          │ lag-only transformations · seasonality · future label
          ▼
Gold      model-ready station-days + three-day outcome
          │ 2018–2023 train ── 2024 calibrate ── 2025+ test
          ▼
AI        HistGradientBoosting + isotonic calibration + climatology baseline
          │ promotion gate · PSI · station audit · scenario serving
          ▼
UI        pipeline control plane · model evaluation · warning workbench
```

If the live request fails, the entire pipeline switches to one deterministic six-station demonstration dataset. Live and synthetic rows are never mixed.

## Data Engineering

### Bounded ingestion

`src/data.py` calls:

```text
https://waterservices.usgs.gov/nwis/dv/
  ?format=rdb
  &sites=01463500,01646500,02177000,07010000,09402500,12149000
  &startDT=2018-01-01
  &endDT=<current date>
  &parameterCd=00060
  &statCd=00003
  &siteStatus=all
```

The request has an identifying User-Agent, connect/read timeouts, exponential retry, a 3 MB response limit and an RDB contract check. The parser supports repeated per-site headers and dynamic internal time-series column IDs rather than assuming unstable column names.

### Layers and contracts

**Bronze** retains `agency`, `site_no`, `event_date`, original `discharge_cfs`, complete qualifier, deterministic `delivery_id`, `event_id` and monthly `batch_id`. Twelve replayed deliveries make idempotency observable.

**Silver** converts dates and numeric values, checks the station allowlist, rejects negative discharge, quarantines contract failures and keeps one event per station-day. It enriches station name and coordinates while preserving whether the qualifier contains a provisional marker. Each released row receives a SHA-256 record hash.

**Gold** creates only features known before the prediction boundary:

- log-discharge lags at 1, 2, 3, 7, 14 and 30 days;
- shifted 7-day mean and standard deviation;
- shifted 30-day mean;
- one- and seven-day log changes;
- day-of-year sine and cosine.

The target is the maximum observed daily mean in the following three days. It is never included as a feature.

### Quality, lineage and observability

Ten fail-closed gates verify:

1. all six stations are present;
2. station-day event keys are unique;
3. discharge is finite and nonnegative;
4. observation dates are plausible;
5. each station has at least 2,500 days;
6. exactly twelve replayed deliveries are suppressed;
7. input, quarantine, duplicate and output rows reconcile;
8. at least 15,000 rows are feature-ready;
9. record hashes are complete;
10. source size remains within bounds.

The stage ledger exposes input/output volume, rejected rows, elapsed source time and shortened content hashes. Bronze, Silver and Gold hashes plus the source hash form a deterministic run ID.

## AI Engineering

### Target and temporal lifecycle

For each gauge, the high-flow reference is the 90th percentile of discharge learned exclusively from 2018–2023 training data. A row is positive when any of the following three daily means reaches that threshold.

- **Training:** 2018–2023, fits thresholds and gradient boosting.
- **Calibration:** 2024, fits isotonic probability calibration.
- **Test:** 2025 through the newest fully labeled date, untouched until final evaluation.

This split prevents later hydrologic behavior from determining older features, thresholds or calibration.

### Model and baseline

The candidate is `HistGradientBoostingClassifier` with fixed random seed, bounded tree complexity and L2 regularization. The model captures nonlinear relationships among recent flow, trend, volatility and seasonal position.

Isotonic regression maps raw model scores to probabilities using only 2024. Ranking metrics use the pre-calibration score because isotonic plateaus can create ties; Brier score evaluates the calibrated probability.

The baseline is historical station/month event frequency learned from the training period. It knows which river and season are involved but cannot inspect the latest flow. The candidate must exceed baseline average precision or publication fails.

### Evaluation and monitoring

- **Average precision:** rare-event ranking quality.
- **ROC-AUC:** ordering of event and non-event days.
- **Brier score:** calibrated probability error.
- **Recall at 10%:** event capture when only 10% of station-days can be reviewed.
- **Reliability bins:** predicted versus observed holdout frequency.
- **Per-station AP:** reveals gauges where performance differs or cannot be computed.
- **PSI:** train-to-test feature distribution change, marked watch above 0.10 and high above 0.25.

PSI is diagnostic evidence, not an automatic retraining trigger.

## Verified live snapshot

On 16 August 2026, the live source returned 18,882 unique station-days from 1 January 2018 through 15 August 2026. The simulated replay added twelve deliveries, all of which were suppressed. No live row required quarantine and all ten quality gates passed.

The temporal model lifecycle contained:

| Split | Rows |
|---|---:|
| Training, 2018–2023 | 12,966 |
| Calibration, 2024 | 2,196 |
| Test, 2025–14 August 2026 | 3,534 |

Held-out results:

| Metric | Result |
|---|---:|
| Model average precision | 0.367 |
| Seasonal-climatology average precision | 0.088 |
| ROC-AUC | 0.887 |
| Brier score | 0.042 |
| Recall at 10% review budget | 53.3% |
| Baseline recall at 10% | 21.7% |
| Holdout high-flow event rate | 5.2% |

The Colorado River gauge had no positive event in this specific holdout under its training-period threshold, so a per-station AP is undefined there. The UI shows that as missing rather than inventing a score.

## Exact data source and usage

- **Provider:** U.S. Geological Survey, Water Data for the Nation.
- **Service:** [Daily Values service documentation](https://waterservices.usgs.gov/docs/dv-service/daily-values-service-details/).
- **Portal:** [Water Data for the Nation](https://waterdata.usgs.gov/).
- **Authentication:** none.
- **Parameter:** `00060`, discharge in cubic feet per second.
- **Statistic:** `00003`, daily mean.
- **Fields used:** agency, station number, observation date, discharge and data-value qualifier.
- **Retrieval:** one HTTPS RDB request covering six allowlisted stations and a bounded date range.
- **Cadence:** daily values are published as observations become available; provisional data can later be revised.
- **Rights:** USGS-authored data and information are generally public domain under the [USGS copyright and credits policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits). Separately credited material can have other rights. Attribution is retained and no USGS endorsement is implied.

Stations:

| Site | Monitoring location |
|---|---|
| `01463500` | Delaware River at Trenton, New Jersey |
| `01646500` | Potomac River near Washington, D.C. |
| `02177000` | Chattooga River near Clayton, Georgia |
| `07010000` | Mississippi River at St. Louis, Missouri |
| `09402500` | Colorado River near Grand Canyon, Arizona |
| `12149000` | Snoqualmie River near Carnation, Washington |

## Important modules

| Module | Responsibility |
|---|---|
| `src/data.py` | resilient USGS client, dynamic RDB parser and deterministic fallback |
| `src/pipeline.py` | Bronze/Silver/Gold, contracts, replay handling, quality, hashes and lineage |
| `src/model.py` | target creation, temporal split, boosting, calibration, baseline, drift and scenario scoring |
| `ui.py` | responsive operational dashboard, evaluation, warning audit and exports |
| `app.py` | standalone Streamlit entrypoint |
| `tests/test_control.py` | parsing, schema, quarantine, idempotency, fallback, leakage and model tests |

## Setup, usage and testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Select **USGS River Flow Early Warning** in the portfolio sidebar. For a standalone launch:

```bash
streamlit run river_flow_early_warning/app.py
```

Tests:

```bash
pytest -q river_flow_early_warning/tests
```

## Failure behavior and limitations

- Any live-source failure activates one completely deterministic fallback snapshot.
- An invalid response, malformed RDB body or oversized payload fails closed.
- Contract failures are quarantined with explicit reasons.
- Failed data gates or a candidate below baseline stop model publication.
- Daily means can hide dangerous intraday peaks.
- The model does not know precipitation, forecasts, upstream conditions, snowpack, reservoirs or flood defenses.
- Qualifiers can indicate provisional data that may later change.
- Six stations and one historical period cannot represent national hydrology.
- A percentile exceedance is not an official action stage or flood warning.
- Low probability is not proof of safety.
- Use official National Weather Service, USGS and local emergency instructions for real decisions.

## Extensions

- move ingestion to the modern USGS OGC APIs and persist immutable Parquet partitions;
- process 15-minute continuous values with event-time watermarks and late-arrival repair;
- join NOAA precipitation forecasts, snowpack and upstream gauge topology;
- replace the fixed target with official station-specific action/flood stages where available;
- use rolling-origin backtests across multiple hydrologic years;
- add conformal or Bayesian uncertainty around event probability;
- register model/data versions and reviewer outcomes in an operational store;
- serve approved warning scores through a versioned FastAPI endpoint.
