# Flight Delay Operations Control

A deployment-ready Data Engineering and AI Engineering product that turns the official U.S. airline on-time archive into a governed data product and a calibrated pre-departure review queue.

## Product

Operations teams cannot investigate every scheduled flight equally. This Streamlit application provides a reproducible control plane: it ingests a bounded official monthly archive, validates and reconciles every selected flight, then ranks a strictly later holdout with only schedule-known information. It includes pipeline lineage, quality gates, operational states, calibration, carrier-level evaluation, drift, a scenario workbench and CSV/JSON exports.

The model is a prioritization aid. It is not an official flight-status service, causal explanation, safety system or passenger promise.

## End-to-end architecture

```text
BTS monthly ZIP
  -> safe download + deterministic hash sampling
  -> Bronze deliveries + source/payload hashes + replay injection
  -> Silver typed contract + quarantine + event-level deduplication
  -> Gold operated flights + schedule-known features
  -> temporal train / calibration / test split
  -> histogram gradient boosting + isotonic calibration
  -> evaluation, drift, review queue and scenario scoring
```

### Data Engineering

- **Safe ingestion:** one bounded ZIP request with an explicit User-Agent, connect/read timeouts, HTTP validation, archive-size limits and safe-member checks.
- **Deterministic reduction:** natural flight keys are SHA-256 hashed and selected with a stable modulo rule. The hosted app processes at most 90,000 records instead of expanding the entire archive into memory.
- **Bronze:** preserves source columns, a stable `event_id`, per-row `payload_hash`, and 25 deterministic replay deliveries used to prove idempotency.
- **Silver:** coerces dates and numeric types, enforces airport, distance, schedule-time and operational-state contracts, quarantines failures, and reconciles replays by natural key.
- **Gold:** includes only operated flights with a usable binary arrival-delay label. Every row remains traceable through `event_id`.
- **Observability:** a run ledger records layer counts, rejects, duration and content hashes. Ten fail-closed publication gates cover volume, uniqueness, replay suppression, reconciliation, ranges, airport and label contracts, leakage, temporal coverage and source bounds.
- **Fallback:** a fixed-seed 72,000-flight generator activates atomically if the source cannot be retrieved or validated. The UI visibly marks demo mode; live and synthetic records are never mixed.

Repeated runs over identical input publish identical Bronze/Silver/Gold hashes and the same `run_id`; the retrieval timestamp is operational metadata and intentionally does not affect identity.

### AI Engineering

The target is `ArrDel15`: arrival at least 15 minutes after schedule, using the official BTS definition. Seven features are known from the schedule: reporting carrier, origin, destination, day of week, scheduled departure hour, scheduled elapsed time and distance. Actual arrival delay, cancellation and diversion outcomes are explicitly excluded.

The June 2026 sample is split chronologically:

- days 1–18: model training;
- days 19–24: isotonic probability calibration;
- days 25–30: untouched evaluation.

The classifier is histogram gradient boosting. Airport categories are learned from training only; infrequent or unseen codes map to `__OTHER__`. Promotion requires candidate average precision to beat the no-skill prevalence baseline. The UI also reports ROC-AUC, Brier score, recall within a 10% review budget, carrier-level average precision, reliability bins and Population Stability Index for numeric features. Scenario scoring uses the same category guardrail and transformation pipeline as evaluation.

On the verified live sample (17 August 2026), 86,273 unique flights passed Silver and 84,450 operated flights formed Gold. The time-held-out test contained 17,386 rows. Average precision was **0.397** versus a **0.220** prevalence baseline, ROC-AUC **0.705**, Brier score **0.163**, and the highest-risk 10% captured **22.0%** of delayed flights. These are sample-specific engineering checks, not future performance guarantees.

## Data source and contract

- **Provider:** U.S. Department of Transportation, Bureau of Transportation Statistics (BTS).
- **Exact archive:** [`On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_6.zip`](https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2026_6.zip).
- **Dataset documentation:** [Airline On-Time Performance database](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD).
- **Field definitions:** [BTS field reference](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ).
- **Delay definition:** [BTS delay explanation](https://www.transtats.bts.gov/ot_delay/OT_DelayCause1.asp?20=E).
- **Retrieval:** keyless HTTPS ZIP download; CSV is streamed from the archive in 60,000-row chunks.
- **Cadence:** monthly reporting; the latest archive observed during implementation was June 2026.
- **Usage:** U.S. federal statistical data is publicly accessible. Preserve BTS/DOT attribution and check source-specific terms before redistribution.

Fields used are `FlightDate`, `DayofMonth`, `DayOfWeek`, `Reporting_Airline`, `Flight_Number_Reporting_Airline`, `Origin`, `Dest`, `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance`, `Cancelled`, `Diverted`, `ArrDel15`, and `ArrDelayMinutes`. The last four operational/outcome fields are used for contracts, filtering, labels and audit only—not as predictors.

## Modules

- `src/data.py`: download, ZIP safety, bounded deterministic sampling and fallback.
- `src/pipeline.py`: Bronze/Silver/Gold contracts, quarantine, hashes, reconciliation and gates.
- `src/model.py`: time split, category guardrails, model, calibration, evaluation, drift and scenario service.
- `ui.py`: responsive control plane, charts, failure states, queue and exports.
- `tests/test_control.py`: deterministic ingestion, fallback, archive safety, contracts, idempotency, leakage, model metrics and edge cases.

## Run and test

From the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
pytest flight_delay_control/tests -q
```

No secret is required. `config.example.toml` documents the bounded operating assumptions. If the central Streamlit Community Cloud application is not deployed yet, connect this repository, select branch `main` and entrypoint `streamlit_app.py`, then click **Deploy** once.

## Limitations and extensions

The stable hash sample supports reproducibility but is not a weighted population estimate. One month cannot represent seasonality. The model lacks live weather, aircraft rotations, crew constraints, congestion forecasts and prior-leg state. BTS data can be revised; scheduled values are not real-time status. Calibration can degrade after deployment, and PSI is diagnostic rather than an automatic retraining order.

Useful extensions include multi-month backtesting, airport/time target encoding with nested validation, live weather joins with event-time correctness, delayed-label monitoring, model registry promotion, cost-based routing thresholds, shadow deployments and carrier-specific calibration where sample size permits.
