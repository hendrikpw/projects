# FDA Recall Triage Pipeline

An observable Data Engineering and AI Engineering product that contracts three
public FDA enforcement streams into one versioned dataset and evaluates a
confidence-aware NLP model for recall-class triage.

> This application is a portfolio demonstration. It does not establish a health
> hazard, classify a real recall, or replace FDA and qualified safety review.

## Problem and product behavior

FDA recall reports for food, drugs and devices share important concepts but live
behind separate endpoints. Operational teams need consistent records, visible
data quality and honest model behavior—not a chart that silently trains on
whatever an API happened to return.

The hosted product therefore provides:

- bounded ingestion from three official enforcement endpoints;
- a class-stratified sample so Class I, II and III are all testable;
- content-addressed raw snapshots and deterministic run IDs;
- a unified typed contract, deduplication and reason-coded quarantine;
- nine quality, reconciliation and leakage checks;
- an auditable run ledger with row counts, latency and content hashes;
- a word- and character-level NLP classifier with untouched holdout evaluation;
- accuracy, balanced accuracy, macro F1, log loss and calibration error;
- per-class evaluation, confusion matrix and learned-term inspection;
- drift sentinels for vocabulary, source mix, label mix and text length;
- adjustable confidence abstention and a failure-closed scoring workbench;
- CSV prediction/data exports and a JSON run manifest;
- deterministic, clearly marked fallback data when live ingestion is unavailable.

## End-to-end architecture

```text
openFDA food ─┐
openFDA drug ─┼─> stratified extract ─> content-addressed snapshot
openFDA device┘         │                         │
                       retry                 SHA-256 lineage
                         └─> source-shaped fallback

snapshot ─> normalize ─> typed contract ─┬─> validated recall product
                                        └─> reason-coded quarantine

validated product ─> leakage-safe NLP view ─> time-aware holdout
                                            ├─> TF-IDF word + char features
                                            ├─> balanced logistic regression
                                            ├─> evaluation + explanations
                                            ├─> drift sentinels
                                            └─> selective prediction / abstain
```

All hosted computation is bounded and in memory. In production, the content
hashes are natural object-store partition keys for immutable Bronze payloads,
typed Silver records and the Gold model view.

## Data Engineering

### 1. Extract

For every selected domain, the loader requests a bounded stratum for each target
class. With all domains selected this is nine requests. Each HTTPS request has a
20-second timeout, one exponential-backoff retry for transport, HTTP 429 and
HTTP 5xx failures, an explicit user agent and response-contract validation.

The entire run switches to the deterministic fallback if any required stratum is
unavailable. Live and fallback records are never mixed, avoiding a partially
synthetic corpus that could be mistaken for FDA data.

### 2. Content-addressed snapshot

The untouched response object is canonicalized as sorted JSON and hashed with
SHA-256. Payload order, source domain and recall-number hints remain available
for lineage. Replaying identical inputs yields the same raw hash, feature hash
and run ID—an explicit idempotency property tested in the suite.

### 3. Unified contract and quarantine

Food, drug and device payloads map to one schema:

| Field | Type / use |
|---|---|
| `record_id` | Stable `domain:recall_number` key |
| `domain` | `food`, `drug` or `device` |
| `recall_number`, `event_id` | FDA identifiers |
| `report_date`, `recall_initiation_date`, `termination_date` | Parsed dates |
| `classification` | Supervised target: Class I, II or III |
| `status` | Recall lifecycle state |
| `recalling_firm` | Firm named in the report |
| `product_description` | FDA product text |
| `reason_for_recall` | FDA reason text |
| `product_quantity`, `distribution_pattern` | Scope context |
| `country`, `state` | Reported geography |
| `voluntary_mandated`, `initial_firm_notification` | Recall process fields |

Contract violations are not silently discarded. Unsupported domains, missing or
duplicate IDs, invalid classes/dates and insufficient product/reason text receive
a reason code in quarantine. A run cannot train if fewer than 60 valid records or
fewer than all three target classes survive.

### 4. Gold NLP view and quality gates

The Gold view contains identity, domain, report date, class label, source link and
one document assembled from domain, product, reason and firm. The `classification`
field is kept only as the separate target and never inserted into model text.

Quality gates validate payload and record uniqueness, supported sources, target
coverage, dates, text completeness, row reconciliation, target leakage and at
least 80% contract retention. Failed validation suppresses model output.

## AI Engineering

### Representation and classifier

The model combines two sparse representations:

- word TF-IDF with unigrams and bigrams for phrases such as product and hazard terms;
- character TF-IDF with 3–5 character windows for morphology, spelling and domain notation.

Balanced multinomial logistic regression turns the union into Class I/II/III
probabilities. The fixed random seed makes evaluation reproducible. Linear
coefficients are displayed by class as associations for inspection; they are not
causal explanations or FDA rules.

### Evaluation

The newest 25% of records form the holdout when all three classes remain present.
Otherwise a deterministic stratified split is explicitly reported. The UI shows:

- accuracy and balanced accuracy;
- macro F1, giving each class equal importance;
- multi-class log loss for probability quality;
- expected calibration error (ECE) across confidence bins;
- class-level precision, recall, F1 and support;
- a confusion matrix and row-level prediction audit.

Because ingestion intentionally balances the nine source/class strata, scores
measure behavior on that bounded evaluation corpus. They do not represent the
natural prevalence of FDA recall classes, and predicted probabilities are not
real-world hazard probabilities.

### Abstention and failure states

The workbench accepts a configurable confidence threshold. If maximum predicted
probability is below it, the model returns `Deferred` instead of forcing a class.
The coverage curve shows how many holdout rows would be accepted and the accuracy
among those accepted. Insufficient input, failed contracts and failed evaluation
all stop scoring completely.

### Drift and monitoring

Four reference-vs-holdout signals are produced:

- word out-of-vocabulary share;
- total-variation distance for domain mix;
- total-variation distance for class mix;
- absolute median text-length shift.

Each has a visible portfolio threshold. A production system should compare weekly
candidate data with the registered training baseline, add segment-level sample
minimums and block automatic promotion when agreed gates fail.

## Exact data sources

Provider: **U.S. Food and Drug Administration, openFDA / Recall Enterprise
System (RES)**.

| Stream | API endpoint | Official documentation |
|---|---|---|
| Food enforcement | `https://api.fda.gov/food/enforcement.json` | [Food Enforcement Overview](https://open.fda.gov/apis/food/enforcement/) |
| Drug enforcement | `https://api.fda.gov/drug/enforcement.json` | [Drug Enforcement Overview](https://open.fda.gov/apis/drug/enforcement/) |
| Device enforcement | `https://api.fda.gov/device/enforcement.json` | [Device Enforcement Overview](https://open.fda.gov/apis/device/enforcement/) |

The enforcement pages describe publicly releasable RES records from 2004 onward
and weekly updates. Retrieval uses `classification:"Class I|II|III"`, descending
`report_date` sorting and a bounded `limit`. openFDA supports a maximum `limit` of
1,000 per request; this product uses much smaller strata.

No paid credentials or user-supplied secrets are required. The official
[authentication page](https://open.fda.gov/apis/authentication/) documents unauthenticated
limits of 240 requests per minute and 1,000 requests per day per IP. A normal
three-domain run makes nine requests and is cached for six hours in Streamlit.

Usage is governed by the [openFDA Terms of Service](https://open.fda.gov/terms/).
The API itself warns that results should be treated as unvalidated and must not be
used to make medical-care decisions. Provider attribution and direct official
documentation links are preserved throughout the product.

## Preprocessing and assumptions

1. Preserve every raw response object and source domain.
2. Normalize whitespace, identifiers, classes and three date fields.
3. Deduplicate on `domain:recall_number`.
4. Quarantine invalid IDs, sources, classes, dates or short core text.
5. Assemble model text from domain, product, reason and firm only.
6. Hold out evaluation rows before vectorizer fitting.
7. Fit TF-IDF vocabulary and classifier only on training text.
8. Evaluate probabilities, class metrics, abstention and drift on holdout text.

The class-stratified sample is an engineering choice for bounded demonstrability,
not an estimate of the true recall distribution. Report language may contain
proxies for internal classification decisions. Historical association is not a
regulatory rule, clinical conclusion or future prediction.

## Important modules and functions

| Module | Responsibility |
|---|---|
| `src/data.py` | Allowlisted domains, retry-aware API ingestion, metadata and deterministic fallback |
| `src/pipeline.py` | Snapshot hashes, normalization, typed contract, quarantine, Gold view, DQ and ledger |
| `src/model.py` | Split, TF-IDF, model training, evaluation, explainability, drift and selective scoring |
| `ui.py` | Responsive Streamlit control plane, model lifecycle, workbench, states and exports |
| `tests/test_pipeline.py` | Contract, schema, idempotency, retries, fallback and reconciliation tests |
| `tests/test_model.py` | Output, reproducibility, metrics, abstention, edge and monitoring tests |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then select **FDA Recall Triage Pipeline** in the central navigation. The same
module can run independently with:

```bash
streamlit run fda_recall_nlp_pipeline/app.py
```

No environment variables are required. `config.example.toml` documents bounded
defaults and contains no secrets.

## Testing

```bash
python -m pytest -q
```

The tests cover allowlisting, deterministic fallback generation, stable hashes,
typed validation, quarantine, leakage boundaries, idempotent replay, atomic
fallback, transient retry, reproducible training, metric domains, output shape,
probability normalization, abstention, drift output and small-data failure.

## Operational considerations

- Cache live ingestion to respect public shared infrastructure.
- Schedule weekly after the documented FDA refresh rather than polling continuously.
- Persist raw payloads immutably and keep contract/model manifests together.
- Alert on retry exhaustion, DQ failures, quarantine spikes and hash anomalies.
- Register model, vectorizer, training hash, evaluation slice and thresholds together.
- Require qualified review for all real cases, including high-confidence outputs.
- Audit accepted and deferred cases separately to prevent silent coverage loss.

## Limitations

- The bounded snapshot is not the complete RES database.
- Class balancing changes source prevalence and probability interpretation.
- OpenFDA data can be amended, delayed or incomplete.
- Free-text shortcuts and temporal changes can degrade generalization.
- Coefficients are associations, not causality or official decision logic.
- The hosted runtime does not persist artifacts between sessions.
- The model does not use local exposure, patient, inventory or business context.

## Extensions

- materialize Bronze/Silver/Gold Parquet partitions in object storage;
- orchestrate weekly incremental watermarks with a durable scheduler;
- validate the schema with a formal Pydantic or Arrow contract registry;
- add rolling temporal backtests and bootstrap confidence intervals;
- calibrate probabilities on a representative non-stratified reference set;
- add human-reviewed error taxonomies and fairness/slice audits;
- expose a versioned FastAPI scoring contract with model registry integration;
- monitor post-deployment labels, abstention queues and delayed feedback.

## Hosted use

This page is integrated into the repository's central `streamlit_app.py`. Once
the owner connects the repository to Streamlit Community Cloud, every `main`
commit updates the persistent browser app automatically. Until a public URL is
confirmed, the only manual step is to deploy `streamlit_app.py` once at
[share.streamlit.io](https://share.streamlit.io/).
