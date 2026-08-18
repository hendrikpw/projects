# Kepler Candidate Reliability Control

A portfolio-ready Data Engineering and AI Engineering product that turns the NASA Exoplanet Archive's cumulative Kepler Objects of Interest catalog into a governed data product and an evaluated, calibrated candidate-vetting workbench.

The project is integrated into the repository's central `streamlit_app.py`; it needs no API key or secret.

## Problem and product behavior

A transit-like brightness dip is not automatically a planet. Kepler Objects of Interest can be candidates, confirmed planets or false positives, and several KOIs can belong to the same target star. A credible engineering demonstration must therefore preserve source semantics, prevent sibling signals from crossing evaluation boundaries, block outcome-derived vetting fields, quantify uncertainty and withhold extreme out-of-distribution cases.

The application provides:

- a current, SQL-style TAP extraction from the official cumulative KOI table;
- Bronze/Silver/Gold lineage, payload hashes, quarantine and replay-proof processing;
- ten fail-closed data-quality and leakage gates;
- a deterministic star-group training, calibration and test split;
- calibrated planet-like probability and three review routes plus OOD withholding;
- average precision, ROC-AUC, Brier score and fixed-budget capture;
- calibration reliability, permutation importance and PSI drift;
- an interactive transit scenario using exactly the production transformation path;
- exportable governed data, model audit and JSON manifest;
- explicit live, loading, fallback, empty, pipeline-failure and model-failure states.

The score reproduces current catalog dispositions from selected physical/catalog measurements. It does not discover or confirm planets, replace analysis of light curves and follow-up observations, or establish causal scientific relationships.

## Architecture and data flow

```text
NASA Exoplanet Archive TAP / cumulative
  -> bounded CSV response + exact schema validation
  -> Bronze: source rows + event IDs + payload hashes + replay deliveries
  -> Silver: numeric contracts + physical/coordinate rules + quarantine + dedupe
  -> Gold: vetted label + measurement-only features
  -> star-group SHA-256 split (60% train / 20% calibrate / 20% test)
  -> histogram gradient boosting + isotonic calibration
  -> promotion gate + drift + importance + uncertainty/OOD routing
  -> Streamlit control plane and audit exports
```

## Data Engineering

### Ingestion and safety

`src/data.py` sends a keyless HTTPS request to the synchronous TAP endpoint with an explicit User-Agent and connect/read timeout. The query selects only 18 documented columns. Publication requires a response between 0.5 and 5 MB, 7,000–12,000 parsed rows and an exact ordered schema; an HTML error page, truncated response or upstream schema change therefore fails closed rather than silently changing the product.

If the live source is unavailable or invalid, a fixed-seed 7,800-row demonstration catalog is activated atomically. Live and synthetic rows are never mixed, and the UI displays the fallback reason.

### Bronze, Silver and Gold

- **Bronze:** preserves every selected source field, creates a stable SHA-256 `event_id` from `kepoi_name`, hashes the complete payload, and intentionally replays 20 valid deliveries.
- **Silver:** coerces types, validates KOI names and documented dispositions, enforces positive period/duration/depth and ICRS coordinate bounds, records quarantine reasons and keeps one deterministic payload per event.
- **Gold:** publishes the binary engineering target plus 13 measurement features, star identity for group isolation and source identifiers for auditing.

The Stage Ledger records input, output, rejection, duration and layer hash. The run ID depends on source and layer contents—not retrieval time—so identical input produces identical identities.

### Publication gates

Ten rules cover source volume, unique event IDs, exact replay suppression, row reconciliation, disposition states, positive physical quantities, coordinates, target balance, feature coverage and leakage. A failed gate prevents Gold and model publication.

## AI Engineering

### Target and leakage boundary

`CONFIRMED` and `CANDIDATE` form the portfolio class `planet_like=1`; `FALSE POSITIVE` is zero. This groups archive states for an engineering classification exercise and does not imply candidates are confirmed planets.

Features are transit period, duration, depth, fitted radius, equilibrium temperature, insolation, stellar effective temperature, surface gravity and radius, Kepler magnitude, KOI multiplicity, number of observed transits and transit-model signal-to-noise ratio. The archive disposition, pipeline disposition, Robovetter score, false-positive flags, names and any disposition-confidence field are blocked.

### Evaluation design

All KOIs from the same `kepid` star are assigned to one split by a stable SHA-256 bucket:

- buckets 0–5: training;
- buckets 6–7: isotonic probability calibration;
- buckets 8–9: untouched testing.

This prevents multiple signals from a single star from leaking across boundaries. Median imputation is learned inside the training pipeline. The candidate is promoted only if test average precision exceeds the test prevalence baseline by at least five percentage points.

On the verified live run of 18 August 2026:

- 9,200 unique KOIs reached Silver and Gold;
- 364 non-positive-depth rows were quarantined;
- 20 replay deliveries were exactly suppressed;
- 5,526 rows trained the model, 1,814 calibrated it and 1,860 remained in test;
- average precision: **0.928** versus **0.525** prevalence baseline;
- ROC-AUC: **0.922**;
- Brier score: **0.112**;
- planet-like recall in the highest-scoring 10%: **19.0%**;
- 238 test KOIs were routed as uncertain.

The test-set permutation view reports the decrease in average precision after one feature is shuffled. PSI compares train and test feature distributions. Neither is causal evidence. Serving also compares scenario values with training 1st/99th-percentile bounds; two or more violations force `ood-review`, even if the classifier appears confident.

## Exact source documentation

- **Provider:** NASA Exoplanet Archive, operated by Caltech/IPAC under contract with NASA's Exoplanet Exploration Program.
- **Endpoint:** [`https://exoplanetarchive.ipac.caltech.edu/TAP/sync`](https://exoplanetarchive.ipac.caltech.edu/TAP/sync).
- **Table:** `cumulative`, the cumulative Kepler Objects of Interest delivery.
- **Retrieval:** synchronous TAP query with `format=csv`; no API key.
- **Documentation:** [TAP guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html), [KOI documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html), [column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html), and [purpose of the cumulative table](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html).
- **Update behavior:** the cumulative table represents the archive's current preferred dispositions and parameters assembled from KOI activity tables; values and dispositions may change when the preferred source changes.
- **Attribution:** follow the archive's [official acknowledgment guidance](https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html) and cite underlying literature when applicable.

Selected fields are `kepid`, `kepoi_name`, `koi_disposition`, `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`, `koi_count`, `koi_num_transits`, `koi_model_snr`, `ra`, and `dec`.

## Modules and tests

- `src/data.py`: TAP request, safety bounds, schema validation and deterministic fallback.
- `src/pipeline.py`: contracts, Bronze/Silver/Gold, quarantine, hashes, reconciliation and gates.
- `src/model.py`: group-isolated split, preprocessing, model, calibration, evaluation, drift, importance and OOD serving.
- `ui.py`: responsive operational and scientific workbench.
- `tests/test_control.py`: determinism, fallback, schema failure, replay safety, quarantine, idempotency, leakage, group isolation, evaluation shape and OOD behavior.

Run from repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
pytest kepler_candidate_control/tests -q
```

## Limitations and extensions

The cumulative KOI catalog combines current results from multiple activity tables and is not automatically appropriate for occurrence-rate studies. Current disposition is a retrospective archive label, not a prospective confirmation process. The selected features omit raw light curves, centroid tests, contamination diagnostics, imaging, spectroscopy and human review reports. Median imputation can hide missingness structure. A random star-group split tests independence between stars but not historical concept drift.

Extensions include release-aware backtesting across DR24/DR25 tables, explicit missingness models, conformal set prediction, cost-sensitive triage, raw light-curve representation learning, model cards per stellar regime, delayed-label monitoring and a registry-backed shadow deployment.
