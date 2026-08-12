# NOAA Storm Impact Operations Pipeline

A portfolio-ready Data Engineering and AI Engineering product that converts NOAA's revision-prone annual Storm Events bulk data into a governed monetary-impact dataset and a time-separated two-stage prioritization model.

## What the application does

- discovers the newest NOAA revision for the contracted annual details file
- verifies filename, gzip signature, compressed and expanded size, schema and content hashes
- normalizes monetary strings such as `10.00K`, `1.50M` and `2B` into USD
- quarantines duplicate IDs, invalid locations/times and incomplete monetary labels
- publishes Bronze, Silver and Gold lineage with deterministic run identity
- estimates both probability of any reported damage and conditional damage magnitude
- calibrates a dedicated $250,000-or-more tail classifier for scarce-capacity review ranking
- evaluates calibration, ranking, monetary error, drift and explainability on later months
- compares the model with a transparent event-type historical-rate baseline
- shows how much observed damage is captured at a fixed review capacity
- serves safe what-if scoring and CSV/JSON audit exports

## Architecture

```text
NOAA bulk directory
  │ latest annual revision discovery
  ▼
bounded .csv.gz extraction
  │ retry · timeout · filename/gzip/size/schema guards
  ▼
Bronze source snapshot + source hash
  │ typing · K/M/B parsing · identity · geography · time
  ├──► reason-coded quarantine
  ▼
Silver complete monetary labels
  │ feature/target separation · leakage gate
  ▼
Gold serving contract + layer/run hashes
  ├──► damage classifier → isotonic probability calibration
  ├──► positive-only log-damage regressor → conformal residual
  └──► expected impact, baseline, ranking, drift and test audit
```

Publication is fail-closed: any mandatory quality gate withholds Gold. Reprocessing the same source revision produces identical layer hashes and run ID.

## Data Engineering implementation

`src/data.py` first reads the official directory and selects the greatest `cYYYYMMDD` revision for the 2025 details file. The downloader uses an explicit user agent, connect/read timeout, three attempts and exponential backoff. Only filenames matching NOAA's documented grammar are accepted. Compressed input is capped at 20 MB, expanded input at 90 MB, and the gzip magic bytes are checked before parsing a strict 12-column projection.

`src/pipeline.py` implements:

1. **Bronze:** source-shaped snapshot and immutable SHA-256 fingerprint.
2. **Silver:** normalized identifiers, categories, timestamps, coordinates and USD values; first-error quarantine semantics.
3. **Gold:** eight ordered pre-outcome features, a binary hurdle target and monetary targets.
4. **Observability:** input/output/rejection volume, stage time, status, stage hashes, revision, source mode, run ID and manifest hash.

Blank property or crop damage is treated as unknown—not zero. This avoids false certainty but creates a complete-case analysis with selection bias, which is shown as quarantine volume.

## AI Engineering implementation

The 2025 snapshot is separated by event start month:

- January–June: model training
- July–September: isotonic calibration and 90% positive-damage residual calibration
- October–December: untouched test evaluation

The hurdle system contains:

1. a class-weighted `HistGradientBoostingClassifier` for `P(any damage)`;
2. a positive-only `HistGradientBoostingRegressor` for `log(1 + damage)`;
3. isotonic calibration fitted only on the middle block;
4. `expected damage = calibrated probability × conditional amount`.

A separately class-weighted and calibrated tail classifier estimates `P(damage ≥ $250,000)`. Its score drives the operational review queue because median-oriented monetary regression predictably underweights rare, extremely costly events. The threshold and 10% review capacity are fixed before the final test period.

The explicit baseline multiplies each event type's training damage rate by its median positive training damage. Evaluation reports AUCPR, ROC-AUC, Brier score, MAE, WAPE, top-capacity damage capture, severe-event precision/recall, permutation importance and PSI. The conditional interval is calibrated only among positive events and is not presented as an unconditional loss interval.

## Leakage policy

Only state, event type, county/zone type, month, start hour, reported event-specific magnitude and start coordinates enter the model. The following fields are deliberately excluded because they are outcomes, post-event assessments or operationally unavailable at the intended decision boundary:

- property/crop damage inputs
- deaths and injuries
- event and episode narratives
- end time and duration
- tornado F/EF scale and path assessment

## Exact source, cadence and rights

- **Provider:** NOAA National Centers for Environmental Information; records submitted by National Weather Service offices
- **Database:** https://www.ncei.noaa.gov/stormevents/
- **Bulk directory:** https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
- **Contracted live revision at validation:** `StormEvents_details-ftp_v1.0_d2025_c20260728.csv.gz`
- **Format specification:** https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf
- **Update cadence:** monthly submissions commonly appear about 75–90 days after the end of a data month; annual files may be revised
- **Fields read:** EVENT_ID, STATE, YEAR, MONTH_NAME, EVENT_TYPE, CZ_TYPE, BEGIN_DATE_TIME, DAMAGE_PROPERTY, DAMAGE_CROPS, MAGNITUDE, BEGIN_LAT and BEGIN_LON
- **Rights:** U.S. federal-government works are generally public domain in the United States. NOAA attribution is retained. NCEI cautions that some archived externally submitted material can have separate rights; this application uses structured database fields only.

NOAA states that damage values are broad best estimates, are not adjusted for inflation and can originate from sources outside NWS. The database is not a complete census of all weather.

## Modules

- `src/data.py` — revision discovery, resilient extraction, decompression guards and deterministic fallback
- `src/pipeline.py` — contracts, monetary parser, quarantine, quality gates and lineage
- `src/model.py` — hurdle models, calibration, uncertainty, evaluation, drift and scoring
- `ui.py` — responsive control plane, model monitoring, workbench and exports
- `tests/test_pipeline.py` — Data Engineering and AI Engineering behavior tests

## Setup and testing

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q tests
```

No secret or paid credential is required. The deterministic fallback keeps pipeline and model behavior demonstrable when NOAA is temporarily unavailable.

Tests cover damage units, fallback determinism, contracts, quarantine, leakage, content idempotency, discovery revisions, decompression limits, failure fallback, reproducible model metrics, probability bounds and serving output.

## Operational limitations and extensions

The model prioritizes completed reports; it is not a live hazard forecast. Magnitude has different units by event type. Complete-case labels create selection bias. Monetary values are nominal and extremely skewed. Geographic and seasonal patterns can encode structural reporting differences rather than physical causality.

Production extensions should ingest current hazard feeds, join inflation and exposure layers, add forecast-time meteorology, maintain slowly changing revision history, use object storage and partitioned Parquet, orchestrate backfills, track delayed labels, introduce model registry/rollback, validate across multiple years and evaluate decisions with emergency-management experts.

## Hosted use

The page is registered in the repository's central `streamlit_app.py`. After the repository owner deploys that entrypoint once on Streamlit Community Cloud, the project appears as **NOAA Storm Impact Operations Pipeline** and future `main` commits update the same deployment.
