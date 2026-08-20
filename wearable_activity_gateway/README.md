# Wearable Activity Inference Gateway

A deployment-ready Data Engineering and AI Engineering product that turns a
versioned smartphone-sensor study into a replay-safe window data product and a
confidence-aware six-class inference service.

> Hosted use: this page is integrated into the repository's central
> `streamlit_app.py`. If Streamlit Community Cloud is not connected yet, deploy
> that root entrypoint once; future commits to `main` update the same app.

## Problem and product behavior

Wearable inference is not only a classification problem. Sensor windows can be
replayed, malformed, missing, shifted between users or produced outside the
device and wearing conditions seen during training. A high-confidence label is
unsafe if the pipeline cannot identify the window or the model has never been
evaluated on a different person.

This application therefore provides one end-to-end gateway:

- safe download and validation of the versioned UCI archive;
- event-time micro-batches with stable window and payload hashes;
- Bronze/Silver/Gold layers, quarantine and replay suppression;
- ten fail-closed publication gates and a batch/run ledger;
- strict subject-level train, calibration and test isolation;
- multiclass Extra Trees inference with temperature scaling;
- confidence-based automatic inference versus review;
- missing-sensor and out-of-distribution withholding;
- majority baseline, class metrics, confusion matrix, calibration error,
  feature importance, PSI drift and latency measurement;
- an interactive failure-injection workbench and audit exports;
- explicit loading, live/demo, quarantine, empty and failure states.

It is a model-engineering demonstration, not a medical device, fall detector or
proof of a person's behavior.

## Architecture and data flow

```text
UCI repository ZIP
    │ retry · timeout · byte bounds · ZIP signature
    ▼
safe nested-archive reader
    │ path traversal check · expanded-size limit · required-file contract
    ▼
Bronze deliveries
    │ stable window ID · payload hash · event time · micro-batch · replay
    ▼
Silver sensor contract ─────► reasoned quarantine
    │ finite/range/subject/activity checks · idempotent window key
    ▼
Gold model windows
    │ layer hashes · run ID · batch and quality ledgers
    ▼
subject-isolated lifecycle
    ├─ 17 people: fit
    ├─ 4 people: probability/decision calibration
    └─ 9 people: untouched test
    ▼
Extra Trees + temperature scaling
    │ confidence + missingness + feature-domain guardrails
    ▼
auto-inference OR human/sensor review
```

## Data Engineering lifecycle

### Safe acquisition

`src/data.py` downloads the official archive with a descriptive user agent,
45-second timeout and at most three attempts with bounded exponential backoff.
The response must be a 10–80 MB ZIP payload. UCI currently wraps the original
dataset ZIP inside a repository ZIP; both layers are checked before reading.

Every archive member is rejected if it is absolute or contains `..`. Total
expanded content is capped at 300 MB. The inner archive must contain exactly the
required feature dictionary, label dictionary and train/test feature, target
and subject files. The matrix parser verifies row and column reconciliation.

### Source schema

The live schema contains:

- 561 normalized time- and frequency-domain features;
- `subject_id` from 1 to 30;
- activity ID and one of six documented labels;
- source partition and source row;
- no missing values in the published dataset.

Feature names are sanitized and prefixed with their official numeric position,
so duplicate or punctuation-heavy labels cannot silently overwrite columns.

### Bronze and simulated streaming

Each source window receives:

- a 24-character SHA-256 `window_id` from partition, row, subject and activity;
- a full SHA-256 `payload_hash` over the source values;
- a deterministic UTC event timestamp for replay demonstration;
- monotonically increasing delivery sequence;
- a 256-delivery `batch_id`.

Twenty valid windows are deliberately replayed at the end of the delivery
stream. The batch ledger marks them with global duplicate history even when the
original and replay occur in different batches.

### Silver contract and quarantine

The contract rejects a delivery with a precise first-failure reason when:

- the stable window key is malformed;
- subject ID is outside 1–30;
- activity ID is undocumented;
- numeric activity and label disagree;
- any feature is missing or non-finite;
- any normalized feature exceeds `[-1.05, 1.05]`;
- event time is invalid.

Valid duplicate window IDs are suppressed with first-delivery semantics. Gold
contains only unique, finite, bounded windows and the lineage fields required
for an audit.

### Publication gates and observability

Publication stops if any of ten gates fail: minimum volume, feature count,
accepted-row ratio, unique window key, exact replay count, row reconciliation,
finite features, bounded domain, six-class coverage or 30-subject/class-share
coverage. Bronze, Silver, Gold and batch ledgers receive deterministic hashes;
the source and layer hashes form `run_id`.

## AI Engineering lifecycle

### Leakage-resistant subject isolation

The original UCI test subjects remain untouched. Within the original training
partition, subject IDs divisible by five form calibration; the remaining
subjects form training. Thus all windows from one person remain in exactly one
stage:

- 5,930 windows / 17 subjects for fitting;
- 1,422 windows / 4 subjects for calibration;
- 2,947 windows / 9 subjects for final testing.

This is stricter than a random window split, which would let highly correlated
windows from one wearer appear in training and evaluation.

### Model, probability calibration and decision policy

A deterministic `ExtraTreesClassifier` uses 220 trees, class-balanced weights,
minimum leaf size two and square-root feature subsampling. It captures nonlinear
interactions across the 561 published features while remaining CPU-friendly for
Streamlit Community Cloud.

Raw multiclass probabilities are calibrated with one scalar temperature. The
temperature is selected only on calibration subjects by minimizing multiclass
log loss over a fixed grid. A confidence threshold is then selected on the same
calibration group to maximize coverage subject to 95% automatic-inference
accuracy. The final test set is not consulted when selecting either parameter.

### Evaluation and promotion gates

The untouched test reports:

- ordinary and balanced accuracy;
- macro F1 and majority-class macro-F1 baseline;
- Top-2 accuracy;
- multiclass log loss;
- expected calibration error (ECE);
- automatic-inference coverage and selective accuracy;
- review/withholding rate;
- per-window batch inference latency;
- recall, confidence and review rate for every activity.

Promotion fails if macro F1 does not beat the majority baseline by at least 0.20,
balanced accuracy is below 0.75 or Top-2 accuracy is below 0.90.

### OOD and sensor-failure guardrails

Training medians are used only to make a computational score when a feature is
missing. Missingness above 5% always produces `sensor-fault-review`. Feature
z-scores are calculated from training-subject distributions; inference is also
withheld when more than 2% of features exceed four standard deviations or any
feature exceeds eight. Low calibrated confidence produces `human-review`.

These are conservative engineering guardrails, not proof that a sensor is
correctly calibrated. PSI on the 15 most important features describes
train-to-test shift with watch levels at 0.10 and 0.25.

## Verified live run

The release run on 20 August 2026 processed the complete official archive:

- 10,299 accepted sensor windows and 561 features;
- 30 subjects and all six activities;
- 10,319 delivered rows after replay injection;
- 20/20 replay deliveries suppressed;
- zero live rows quarantined;
- all ten data publication gates passed.

On the nine untouched test subjects:

- accuracy: **93.65%**;
- balanced accuracy: **93.22%**;
- macro F1: **0.9331** versus **0.0514** majority baseline;
- Top-2 accuracy: **98.41%**;
- multiclass log loss: **0.2222**;
- ECE: **0.0530**;
- automatic-inference coverage: **81.74%**;
- selective accuracy: **98.26%**;
- review route: **18.26%**;
- batch inference: approximately **0.030 ms per window**.

Walking downstairs was the hardest class at 80.24% recall. Two important
gravity-axis features crossed the PSI 0.25 watch boundary. These are diagnostic
signals, not automatic reasons to retrain. Results apply to this static study
and exact controlled evaluation, not arbitrary consumer devices.

## Exact data source, fields and license

| Item | Documentation |
|---|---|
| Provider | UCI Machine Learning Repository |
| Dataset page | https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones |
| Versioned archive | https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip |
| DOI | https://doi.org/10.24432/C54S4K |
| Published | Donated 9 December 2012; static research dataset |
| Subjects | 30 volunteers aged 19–48 |
| Device and placement | Samsung Galaxy S II worn at the waist |
| Sampling | Accelerometer and gyroscope at 50 Hz; filtered 2.56-second windows, 50% overlap, 128 readings per window |
| Activities | Walking, walking upstairs, walking downstairs, sitting, standing and laying |
| Fields used | 561 published normalized time/frequency features, activity ID/label, subject ID and source partition |
| License | Creative Commons Attribution 4.0 International |

Credit: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X.
(2013), *Human Activity Recognition Using Smartphones*, UCI Machine Learning
Repository, DOI 10.24432/C54S4K.

## Reproducible fallback

If download, ZIP safety, file schema, dimensions or minimum live contracts fail,
`fallback_data(seed=42)` creates 1,800 clearly labeled demonstration windows for
30 subjects, six activities and 48 bounded features. It passes through the same
replay, contract, split, evaluation and UI paths. Demo mode and the captured
failure reason are always visible; it never impersonates the UCI data.

## Important modules

| Module | Responsibility |
|---|---|
| `src/data.py` | retries, archive safety, nested ZIP handling, matrix parsing, feature dictionary and fallback |
| `src/pipeline.py` | micro-batches, IDs/hashes, Bronze/Silver/Gold, contracts, quarantine, gates and ledgers |
| `src/model.py` | subject splits, Extra Trees, temperature scaling, confidence policy, metrics, drift, OOD and serving |
| `ui.py` | responsive operations dashboard, evaluation views, failure workbench and exports |
| `tests/test_gateway.py` | archive, fallback, schema, replay, lineage, isolation, model and guardrail tests |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone page:

```bash
streamlit run wearable_activity_gateway/app.py
```

No key, paid service or secret is required. Safe default parameters are shown in
`config.example.toml`.

## Tests

```bash
pytest -q wearable_activity_gateway/tests
pytest -q
```

The project tests cover deterministic fallback, incomplete and path-traversal
archives, replay idempotency, hash invariance, publication gates, batch
reconciliation, reasoned quarantine, subject isolation, baseline promotion,
class/output shape, reproducibility, ordinary inference, missing-sensor failure
and insufficient-subject failure.

## Limitations and extensions

- The data come from 30 people, one 2012 phone model and one waist position.
- Published engineered features assume UCI's filtering/window pipeline; this app
  does not claim byte-identical feature generation from arbitrary raw sensors.
- The six labels omit transitions, falls, vehicles, sports and uncontrolled
  everyday behavior. Overlapping windows are statistically correlated.
- Temperature scaling and thresholds need periodic reevaluation with labeled
  target-device data. PSI alone does not diagnose the reason for drift.
- A production extension could add a versioned raw-signal feature service,
  device-specific data contracts, Kafka event timestamps/watermarks, a model
  registry, shadow deployment, per-device drift, reviewed outcomes and a
  latency-budgeted FastAPI/ONNX edge serving path.
