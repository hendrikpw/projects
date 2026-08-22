# Handwritten Digit Recognition Gateway

> A replay-safe image-data pipeline and calibrated computer-vision gateway for reading isolated handwritten digits from previously unseen writers.

This project is both a Data Engineering product and an AI Engineering product. It does not stop at a classifier notebook: it safely ingests the official UCI archive, preserves the original writer boundary, validates every image, records content lineage, evaluates the model on untouched writers, deliberately damages test images, and withholds unsafe inputs at serving time.

## Product behavior

The hosted Streamlit page provides three connected control planes:

1. **Pipeline control** — extraction safety, micro-batch replay, typed contracts, quarantine, class balance, reconciliation, quality gates and layer lineage.
2. **Model evaluation** — untouched-writer metrics, normalized confusion matrix, per-digit recall, confidence routing, pixel importance and controlled corruption curves.
3. **Robustness lab** — select a held-out image, inject Gaussian noise or missing pixels, inspect the route, compare candidate probabilities and view local occlusion sensitivity.

The central portfolio app exposes the project as **Handwritten Digit Recognition Gateway**.

## Why this problem matters

A model can score well while the surrounding system remains unsafe. Duplicate deliveries can leak into evaluation, malformed images can silently enter training, random splits can place the same writer on both sides, confidence can be uncalibrated, and an API can return a prediction for an input unlike anything seen during training.

This gateway makes those failure modes visible. The product boundary is deliberately narrow: one normalized, isolated digit represented by an 8×8 matrix. It is not general document OCR, signature verification, identity evidence or a safety-critical reader.

## Architecture and data flow

```text
UCI versioned ZIP
       │ retry · timeout · size · ZIP signature · exact allowlist
       ▼
Bronze deliveries ── stable sample/image hashes ── replay detection
       │
       ▼
Silver contract ──── typed label/pixels ────────── quarantine
       │
       ▼
Gold tensors ─────── normalized 0..1 ───────────── SHA-256 lineage
       │
       ├── official 30-writer train → fit
       ├── disjoint train slice → temperature + confidence policy
       └── official 13-writer test → final metrics + corruption suite
                                                   │
                                                   ▼
                                  auto-read / human-review / withheld
```

### Bronze

- Preserves the 64 integer pixels, label, official source split and row number.
- Creates a stable `sample_id` from split, source row and label.
- Creates an `image_hash` from the 64 pixel bytes.
- Assigns deterministic 128-row micro-batches.
- Adds twenty intentional duplicate deliveries to demonstrate replay suppression.
- Records deliveries, unique samples, replays and hashes per batch.

### Silver

Every record is checked against the image contract:

- exactly one stable sample identifier and image hash;
- source split is `train` or `test`;
- label is an integer from 0 through 9;
- all 64 pixels are present, finite and integral;
- every pixel lies in the documented range 0 through 16;
- duplicate sample identifiers are rejected after the first delivery.

Rejected records enter quarantine with a machine-readable reason. They never reach Gold or the model.

### Gold

- Pixels are normalized from `0..16` to `0..1`.
- The official train/test partition is unchanged.
- All identifiers and hashes remain attached for audit.
- Canonically ordered content receives a deterministic layer hash and run ID.

## Data quality and observability

The pipeline fails closed if any publication gate fails:

1. source rows reconcile with accepted rows and quarantine;
2. deliveries reconcile with accepted, quarantined and replay rows;
3. sample identifiers are unique;
4. image hashes are present;
5. exactly 64 normalized pixel features are published;
6. all Gold pixels remain within `0..1`;
7. all ten labels are represented;
8. the official train partition is present;
9. the official test partition is present;
10. replay suppression matches the injected deliveries.

The run manifest includes source mode, source and layer SHA-256 hashes, row counts, replay and quarantine counts, durations, run ID and fallback reason. Stage and batch ledgers make the transformation path inspectable.

## AI lifecycle

### Split strategy

The dataset's natural author boundary is more important than a convenient random split:

- the official training file contains samples from 30 people;
- a deterministic, label-stratified slice of that file is reserved for calibration;
- the official test file contains samples from 13 different people and is untouched until final evaluation.

This tests generalization to new writers and prevents the most obvious writer leakage.

### Model

An RBF support-vector classifier learns nonlinear boundaries over the 64 normalized pixels. The compact UCI dataset makes this model fast enough for a hosted demonstration while still providing a meaningful nonlinear image-recognition workflow.

The decision scores are converted to probabilities with multiclass softmax temperature scaling. Temperature is selected only on the calibration slice by minimizing log loss. A separate calibrated confidence policy routes a sample to automatic reading or human review.

### Evaluation

The untouched test evaluation includes:

- accuracy and macro F1;
- majority-class macro-F1 baseline;
- top-3 accuracy;
- multiclass log loss;
- expected calibration error;
- selective accuracy, automatic coverage and review rate;
- per-digit recall and review rate;
- normalized confusion matrix;
- latency per image.

The promotion gate requires macro F1 above 0.90, a material improvement over the majority baseline, and corrupted-image macro F1 above 0.70.

### Robustness and failure states

The same untouched test set is evaluated at six corruption levels. Each level combines deterministic Gaussian noise with pixel dropout. The dashboard reports accuracy, macro F1, automatic coverage and selective accuracy as the input deteriorates.

Serving has three routes:

- `auto-read`: input looks plausible and confidence clears the calibrated threshold;
- `human-review`: the image remains plausible but confidence is insufficient;
- `input-withheld`: missing pixels exceed the safety bound or ink density falls outside the training envelope.

These are explicit product states, not hidden exceptions.

### Explainability

Global permutation importance measures the reduction in test accuracy when one pixel is shuffled. The interactive local explanation removes one pixel at a time and measures how much support for the winning class falls. These views describe model sensitivity; they do not prove a causal reason for a person's handwriting.

## Data source

| Item | Documentation |
|---|---|
| Provider | UCI Machine Learning Repository |
| Dataset | [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) |
| Direct archive | [Versioned UCI ZIP](https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip) |
| DOI | [10.24432/C50P49](https://doi.org/10.24432/C50P49) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Cadence | Static research dataset; donated 30 June 1998 |
| Retrieval | HTTPS ZIP download without an API key |

### Origin and fields

UCI documents 5,620 instances and 64 integer features. The source began with 32×32 normalized bitmaps derived from NIST forms. Each image was divided into non-overlapping 4×4 blocks; counts of foreground pixels became an 8×8 matrix with values from 0 to 16.

The two files used are:

- `optdigits.tra`: 3,823 images from 30 contributors;
- `optdigits.tes`: 1,797 images from 13 different contributors.

Fields used:

- `px_0_0` through `px_7_7`: block-level foreground-pixel counts;
- `label`: handwritten class from 0 through 9;
- `source_split` and `source_row`: pipeline provenance added during ingestion.

No names, raw handwriting forms or personal identifiers are present in the processed files.

## Resilient ingestion and fallback

The downloader uses a descriptive user agent, timeout, three attempts with exponential backoff, compressed and expanded size limits, ZIP-signature validation, path-traversal protection and an exact member allowlist. Parsing rejects an unexpected column count.

If the official source is unavailable or invalid as a whole, the app atomically switches to a deterministic seven-segment-style sample generator. The UI labels this mode and stores the source failure in the manifest. Live and fallback samples are never mixed.

## Important modules

| Module | Responsibility |
|---|---|
| `src/data.py` | Safe archive retrieval, member validation, matrix parsing and deterministic fallback |
| `src/pipeline.py` | Bronze/Silver/Gold, hashes, replay suppression, contracts, quarantine, ledgers and quality gates |
| `src/model.py` | Writer-safe split, classifier, temperature scaling, evaluation, corruption suite and serving guardrails |
| `ui.py` | Responsive pipeline control, model evaluation, robustness lab and audit exports |
| `tests/test_gateway.py` | Data, idempotence, failure-state, reproducibility, model and serving tests |

## Run locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the project tests:

```bash
pytest -q digit_recognition_gateway/tests
```

No secret or paid credential is needed. `config.example.toml` documents non-secret operational defaults.

## Hosted use

The project is registered in the central app at `streamlit_app.py`. If a Streamlit Community Cloud deployment already follows `main`, the new page appears automatically after the commit. If no deployment exists, the sole manual step is to create one for repository `hendrikpw/projects`, branch `main`, entry point `streamlit_app.py`.

A public live-demo URL is intentionally not invented. Add the confirmed URL here and in the root README once deployment is connected.

## Operational considerations

- The source is static, so caching avoids repeated downloads and model fits.
- All transformations and corruption tests use fixed seeds.
- Fitting and inference are bounded for Community Cloud resources.
- One project page is isolated by the central router's error boundary.
- Manifests, held-out predictions and quality ledgers can be downloaded from the UI.
- A production service would persist artifacts in object storage and publish model/data versions to a registry rather than relying on process cache.

## Limitations

- The images are heavily normalized and only 8×8 pixels; modern phone photos and scanned documents are out of distribution.
- The training population and collection process are historical and limited.
- UCI provides the aggregate writer split but not a writer identifier per row, so calibration can be label-stratified but not person-grouped within the training file.
- Synthetic corruption covers noise and missing pixels, not rotations, translations, adversarial examples or every device failure.
- Temperature scaling improves probabilistic behavior on the calibration slice but cannot guarantee future calibration.
- Occlusion sensitivity is local model behavior, not a human explanation or causal attribution.

## Possible extensions

- retain the original 32×32 bitmaps and compare convolutional or transformer encoders;
- add rotation, blur, translation and contrast corruption suites;
- evaluate conformal prediction sets for explicit coverage guarantees;
- serve the model behind FastAPI with schema validation and telemetry;
- store OpenTelemetry spans and artifact manifests in durable infrastructure;
- monitor class, confidence, pixel and route drift against a production reference window;
- add human-review feedback without contaminating the untouched benchmark.

## Skills demonstrated

Secure public-data ingestion, ZIP hardening, typed data contracts, micro-batching, idempotency, quarantine, lineage, observability, reconciliation, fail-closed publication, image preprocessing, leakage prevention, multiclass classification, probability calibration, selective prediction, robustness testing, OOD detection, explainability, Streamlit product design and practical automated testing.
