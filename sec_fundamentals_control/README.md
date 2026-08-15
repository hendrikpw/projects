# SEC Fundamentals Control

A deployment-ready Data Engineering and AI Engineering product for tracing, reconciling and reviewing public SEC XBRL fundamentals. It turns revision-prone Company Facts feeds into a governed quarterly data product, then evaluates a peer-local anomaly detector before allowing it to produce a review queue.

The application is integrated into the repository's central `streamlit_app.py`. It requires no API key or secret. Until the owner connects the repository to Streamlit Community Cloud, the only manual deployment step is to select this repository, branch `main`, entrypoint `streamlit_app.py`, and click **Deploy**.

## Problem and product behavior

Financial facts are not a clean rectangular dataset. Companies can use alternative US-GAAP concepts, later filings can revise earlier values, duration and instant facts use different frames, and a statistically unusual ratio is not evidence of wrongdoing. The product therefore treats provenance and evaluation as first-class behavior:

- loads eight bounded SEC Company Facts feeds with an identified User-Agent, retries, timeouts and atomic fallback;
- preserves raw fact lineage through CIK, concept, unit, frame, filing date, form and accession number;
- quarantines invalid rows and selects the latest filing for each company, standardized calendar frame and metric;
- reconciles revenue, net income, assets and equity into feature-ready company-quarters;
- records content hashes, row reconciliation, revision counts, stage timings and ten fail-closed quality gates;
- trains Local Outlier Factor only on older quarters, calibrates its threshold on the next period and evaluates on the newest period;
- compares the model with a transparent maximum robust-distance baseline using deterministic controlled stresses;
- exposes drift, a natural review queue, scenario scoring, nearest historical peers and downloadable audit artifacts.

“Review” means locally unusual within this small technology-company benchmark. It does **not** mean fraud, accounting error, investment risk or future underperformance.

## Architecture and data flow

```text
SEC Company Facts JSON (8 CIKs)
        │ identified User-Agent · retry · timeout · atomic fallback
        ▼
Bronze  standardized-frame USD facts + accession lineage + record hashes
        │ typed parsing · quarantine · concept precedence
        ▼
Silver  latest fact per ticker/frame/metric + revision count
        │ four-metric reconciliation · derived ratios · content hash
        ▼
Gold    one governed company-quarter per row
        │ earlier train ── next 8 frames calibrate ── latest 8 frames test
        ▼
AI      RobustScaler + novelty LOF + controlled-stress evaluation
        │ threshold, baseline, drift, nearest-peer evidence
        ▼
UI      pipeline ledger · quality gates · evaluation · review · scenario
```

The pipeline is idempotent: identical source payloads produce identical Bronze, Silver and Gold content hashes and the same run identifier. If any live feed fails, the entire run switches to one deterministic demonstration snapshot rather than mixing live and synthetic companies.

## Data Engineering implementation

### Extraction and resilience

`src/data.py` requests `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` sequentially for AAPL, MSFT, GOOGL, AMZN, META, NVDA, INTC and IBM. Requests use a descriptive User-Agent, bounded response size, connect/read timeouts, retry with backoff and response validation. The UI caches successful work for six hours.

### Contract and transformations

`src/pipeline.py` accepts standardized SEC frames only:

- duration facts: `CYyyyyQq`;
- instant facts: `CYyyyyQqI`;
- forms: `10-Q` and `10-K`;
- unit: `USD`;
- required fields: ticker, CIK, metric, concept, frame, period end, filed date, accession, form and finite numeric value.

Concept precedence is explicit. Revenue can resolve from `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet` or `Revenues`; income from `NetIncomeLoss` or `ProfitLoss`; assets from `Assets`; equity from `StockholdersEquity` or `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`.

For the same ticker, frame and metric, the most recently filed fact wins. Earlier facts remain observable through `revision_count`, rejected/superseded reconciliation and accession lineage. Gold features are:

- net margin = net income / revenue;
- implied liability ratio = (assets − equity) / assets;
- quarterly asset turnover = revenue / assets;
- quarterly return on assets = net income / assets;
- year-over-year revenue growth;
- year-over-year margin change.

The implied liability value and all ratios are application-derived analytical fields, not SEC-published metrics.

### Quality, lineage and observability

Ten publication gates cover non-empty layers, required columns, unique fact identifiers, finite values, positive assets/revenue, ratio bounds, Gold uniqueness, accession completeness, stage reconciliation and feature readiness. A failing gate raises an exception and prevents an analytical release. The stage ledger records input/output volume, quarantined or superseded rows, elapsed milliseconds and SHA-256 content hashes.

## AI Engineering implementation

`src/model.py` uses six features: revenue growth, net margin, liability ratio, quarterly asset turnover, quarterly return on assets and margin change. A `RobustScaler` is fitted on the earlier training frames. A novelty-enabled `LocalOutlierFactor` learns local historical neighborhoods without using the holdout.

The temporal lifecycle is deliberately separated:

1. older standardized reported frames train preprocessing and LOF;
2. the next eight frames calibrate the review threshold at the 95th percentile;
3. the latest eight frames are an untouched test set.

Because no reliable public anomaly/fraud label accompanies these facts, evaluation uses a clearly marked sensitivity test. Each held-out row receives a deterministic stress that moves two features by 2.5 and 2.0 training robust-scale units. The candidate is compared with a maximum robust-distance baseline using average precision, ROC-AUC and recall within a 10% review budget. Promotion fails if candidate average precision is below the baseline. These metrics assess detector sensitivity and ranking only—not real-world anomaly correctness.

Population Stability Index compares training with recent test features. PSI above 0.10 is shown as watch and above 0.25 as high. It is a monitoring signal, not an automatic retraining command. Scenario explanations report each feature's deviation from eight nearest historical reference quarters; this is local evidence, not a causal explanation.

## Verified live snapshot

On 15 August 2026, the live validation processed 1,214 current facts into 273 complete company-quarters and identified 154 superseded fact records. All ten publication gates passed. The time split produced 146 training, 48 calibration and 47 test rows.

Controlled-stress evaluation produced:

| Metric | Result |
|---|---:|
| Peer-local LOF average precision | 0.848 |
| Robust-distance baseline average precision | 0.714 |
| Stress ROC-AUC | 0.915 |
| Recall at 10% review budget | 17.0% |
| Baseline recall at 10% | 10.6% |
| Calibration alert rate | 6.25% |

Several recent features showed high PSI. The UI exposes that prominently because this benchmark is small and recent company economics differ materially from older training history.

## Exact data source and usage

- **Provider:** U.S. Securities and Exchange Commission, EDGAR.
- **Endpoint:** `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`.
- **Documentation:** [SEC EDGAR application programming interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
- **Access guidance:** [SEC developer resources and fair access](https://www.sec.gov/about/developer-resources).
- **Authentication:** none; SEC states that these JSON APIs do not require an API key.
- **Update cadence:** SEC states that submissions and XBRL APIs update throughout the day in real time as filings are disseminated.
- **Fields used:** entity name, CIK, taxonomy/concept, USD unit, value, start/end dates, standardized calendar frame, filed date, accession number and form.
- **Retrieval:** HTTPS JSON GET with identified User-Agent, sequential bounded requests, retries, timeout and caching.
- **Terms:** automated access must follow SEC privacy, security and fair-access guidance. The application uses structured public filing facts, does not download filing text, preserves source attribution and does not imply SEC endorsement. Rights in issuer materials can vary.

## Modules

| Module | Responsibility |
|---|---|
| `src/data.py` | SEC client, bounded retries, source hashes and deterministic fallback |
| `src/pipeline.py` | Bronze/Silver/Gold contracts, revision resolution, quality gates, lineage and features |
| `src/model.py` | temporal split, scaling, LOF, baseline, stress evaluation, drift and scenario evidence |
| `ui.py` | responsive control plane, charts, states, exports and source/limitations panel |
| `app.py` | standalone Streamlit entrypoint |
| `tests/test_control.py` | schema, idempotency, failure, model, evaluation and edge-case tests |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Select **SEC Fundamentals Control** in the sidebar. A standalone launch is also possible with `streamlit run sec_fundamentals_control/app.py`.

Run the project tests with:

```bash
pytest -q sec_fundamentals_control/tests
```

## Failure behavior and limitations

- A source outage activates a deterministic whole-run fallback; the UI labels it.
- A malformed or oversized response fails closed.
- Contract failures go to quarantine; failed quality or model promotion gates stop publication.
- SEC calendar frames improve alignment but cannot make different fiscal calendars and policies identical.
- Concept alternatives can still represent economically different disclosures.
- Eight large technology companies are not a representative market universe.
- LOF has no fraud labels, no causal interpretation and no accounting judgment.
- Controlled stresses are synthetic and may not represent real restatements or reporting errors.
- High drift can reduce reliability; analysts must inspect filings and accession-linked evidence.
- This is educational software, not accounting, legal or investment advice.

## Possible extensions

- persist immutable snapshots to object storage and query them with DuckDB or Iceberg;
- ingest filing timestamps incrementally and add late-arrival service-level objectives;
- expand sector-specific cohorts and learn concept mappings per industry;
- add filed-document links and human review outcomes without redistributing filing text;
- evaluate with a curated restatement or accounting-review dataset;
- serve versioned scores through FastAPI with model cards and approval workflow;
- monitor cohort drift, alert stability and reviewer agreement over time.
