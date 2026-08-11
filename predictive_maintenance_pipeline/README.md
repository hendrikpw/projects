# Predictive Maintenance Decision Pipeline

A deployment-ready Data Engineering and AI Engineering mini-product that turns the UCI AI4I 2020 machine-cycle benchmark into a governed feature product and an evaluated, cost-sensitive maintenance decision system.

## Product behavior

The Streamlit control plane exposes the full lifecycle instead of showing only a classifier score:

- Bronze, Silver and Gold volumes, latency and SHA-256 lineage
- schema, domain, uniqueness, reconciliation, scale and leakage gates
- rejected-record quarantine and layer inspection
- ordered train/calibration/test evaluation
- AUCPR, ROC-AUC, Brier score, precision, recall, F1 and balanced accuracy
- probability reliability, permutation importance and PSI drift monitoring
- configurable false-negative versus false-positive cost and threshold optimization
- single-cycle what-if serving with explicit guardrails
- exportable Gold data, evaluation decisions and run manifest

## Architecture and data flow

```text
UCI ZIP
  │ retry · timeout · archive size/member allowlist
  ▼
Bronze source table + source/content hashes
  │ typed contract · domains · ranges · identity
  ├──► quarantine + reason
  ▼
Silver validated cycles + failure-mode audit fields
  │ derived temperature gap and power proxy
  │ target-derived TWF/HDF/PWF/OSF/RNF removed
  ▼
Gold serving contract
  ├──► weighted gradient boosting
  ├──► isotonic probability calibration
  ├──► cost-sensitive threshold
  └──► untouched test evaluation + drift + explainability
```

Every run ID is derived from the source hash and Gold hash. Reprocessing identical input produces identical layer hashes and run identity. Publication stops if any mandatory gate fails.

## Data Engineering

`src/data.py` performs bounded extraction with connect/read timeouts, three attempts, exponential backoff, archive-size validation and an exact ZIP-member allowlist. It falls back atomically to a deterministic, source-shaped synthetic sample when the public host is unavailable.

`src/pipeline.py` implements:

1. **Bronze:** source-shaped rows and immutable fingerprint.
2. **Silver:** normalized names and types; UDI uniqueness; L/M/H product domain; physical range checks; target validation; reason-coded quarantine.
3. **Gold:** ordered model contract with two engineered features. It deliberately excludes all five failure-mode columns because they encode the target.
4. **Observability:** row reconciliation, per-stage latency, rejected counts, status, layer hashes, run ID and manifest hash.

## AI Engineering

The dataset has no timestamp. The application therefore uses UDI order as a reproducible process-order proxy and labels it honestly as an **ordered split**, not a temporal split:

- first 60%: training
- next 20%: isotonic calibration and operating-threshold selection
- final 20%: untouched evaluation

A `HistGradientBoostingClassifier` receives inverse-prevalence positive weights. Isotonic regression maps raw scores to probabilities. A threshold grid minimizes:

`decision cost = false negatives × missed-failure cost + false positives × inspection cost`

AUCPR is primary because failure is rare. ROC-AUC, Brier score, precision, recall, F1, balanced accuracy and confusion counts cover discrimination, calibration and decisions. A prevalence-only predictor is the explicit AUCPR baseline. Permutation importance measures test-set AUCPR loss; population stability index (PSI) monitors feature distribution shift (`<0.10` stable, `0.10–0.25` watch, `≥0.25` high).

## Source, fields, cadence and rights

- **Provider:** UC Irvine Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Exact archive:** https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
- **Retrieval:** one ZIP download; `ai4i2020.csv` parsed from an exact member allowlist
- **Cadence:** static benchmark, donated/published in 2020; no live operational update cadence
- **Scale:** 10,000 synthetic machine-cycle records
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the UCI dataset page
- **Model fields:** Type, air temperature, process temperature, rotational speed, torque, tool wear, derived temperature gap and power proxy
- **Target:** Machine failure
- **Audit-only fields:** UDI, Product ID and TWF/HDF/PWF/OSF/RNF; the failure modes never enter Gold or the model

## Important modules

- `src/data.py` — resilient extraction, archive validation and fallback
- `src/pipeline.py` — typed transformations, quarantine, contracts, hashes and ledger
- `src/model.py` — training, calibration, cost policy, evaluation, PSI and serving
- `ui.py` — responsive Streamlit product, charts, states, controls and exports
- `tests/test_pipeline.py` — DE and AI behavior tests

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q tests
```

No secrets or paid credentials are required. For the central portfolio, run `streamlit run ../streamlit_app.py`.

## Testing and operational considerations

Tests cover deterministic fallback, schema shape, quarantine reasons, row reconciliation, content idempotency, source failure, unsafe archives, retry recovery, leakage exclusion, model reproducibility, probability bounds, evaluation baseline, cost response and serving output.

Production extensions should add asset/time identity, streaming telemetry, a feature store, orchestrated backfills, delayed-label joins, maintenance-action logging, group/time cross-validation, alert queues, human overrides, model registry, rollback, service-level objectives and shadow deployment.

## Limitations

AI4I is synthetic and contains one row per process observation rather than a longitudinal asset history. UDI ordering is not event time. Failure-mode labels are constructed from known rules. Offline performance therefore demonstrates engineering behavior, not real plant safety or economic value. The tool is not a remaining-useful-life estimator and must never operate machinery.

## Hosted use

The project is registered in the repository's central `streamlit_app.py`. Once the repository owner connects that entrypoint to Streamlit Community Cloud, it is available under **Predictive Maintenance Decision Pipeline** and future `main` commits update the same deployment automatically.
