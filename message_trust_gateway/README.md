# Message Trust Gateway

A privacy-aware Data Engineering and AI Engineering mini-product that transforms the UCI SMS Spam Collection into an idempotent message data product and a calibrated allow/review/block gateway with duplicate-group isolation, adversarial evaluation and drift monitoring.

## Product capabilities

- bounded, retried ZIP ingestion with exact member allowlist and path/size guards
- deterministic micro-batch audit with event and payload hashes
- Unicode/whitespace normalization, typed labels and reason-coded quarantine
- exact-text duplicate groups that cannot cross model splits
- URL, email, phone and money tokenization before Gold publication and inference
- combined word- and character-ngram classification
- separate Platt calibration and policy-threshold selection
- allow, block and explicit human-review outcomes
- word-only benchmark, AUCPR/ROC-AUC/Brier/precision/recall/F1 evaluation
- deterministic obfuscation challenge and clean-versus-adversarial comparison
- PSI drift, global coefficient view and local contribution evidence
- operational exports for Gold, test decisions and the run manifest

## Architecture and lifecycle

```text
UCI ZIP
  │ retry · timeout · compressed/expanded bounds · exact allowlist
  ▼
Bronze source events + source/payload hashes
  │ deterministic 500-row replay · event IDs
  ▼
Silver normalized text + label contract + duplicate group hash
  ├──► reason-coded quarantine
  ▼
Gold tokenized text + drift features + layer/run hashes
  │ group hash split: 70% train / 15% calibrate / 15% test
  ├──► word TF-IDF baseline
  ├──► word + character TF-IDF logistic model
  ├──► Platt calibration and allow/review/block policy
  └──► clean, adversarial, drift and evidence audit
```

Reprocessing identical input creates the same Gold hash and run ID. Any failed mandatory quality or privacy gate prevents publication.

## Data Engineering

`src/data.py` downloads the static archive with an explicit user agent, connect/read timeout, three attempts and exponential backoff. It accepts only `SMSSpamCollection` and `readme`, prevents path traversal, and limits compressed plus expanded size. A deterministic source-shaped fallback demonstrates the entire application when UCI is unavailable.

`src/pipeline.py` provides:

1. **Bronze:** source row, original label/message and immutable source lineage.
2. **Replay audit:** stable 500-row batches, payload hashes, received/accepted/duplicate-delivery counters.
3. **Silver:** NFKC normalization, whitespace cleanup, label contract, event identity and SHA-256 exact-text group identity.
4. **Gold:** privacy-tokenized text and length, token, digit, uppercase and sensitive-pattern counts.
5. **Observability:** stage volumes, quarantine, duration, status, hashes, run ID, manifest and quality results.

Tokenization replaces detected URLs, emails, phone numbers and monetary strings with typed placeholders. It reduces direct exposure but is not complete de-identification.

## AI Engineering

UCI explicitly states the collection is not chronologically sorted. The application therefore uses deterministic group-hash splitting and does not describe it as temporal. Every identical normalized message stays wholly within train, calibration or test.

The candidate model unions:

- word TF-IDF with unigrams and bigrams;
- character-boundary TF-IDF with 3–5 character grams;
- class-weighted logistic regression with a fixed seed.

A word-only logistic model is the explicit benchmark. A one-dimensional logistic (Platt) calibrator is fitted on candidate decision scores from calibration groups. Two independent policy thresholds target at least 99% ham precision for auto-allow and 95% spam precision for auto-block where feasible; messages between them are deferred to review.

Evaluation includes AUCPR as primary rare-class metric, ROC-AUC, Brier score, precision, recall, F1, confusion counts, automatic coverage and review rate. An adversarial suite deterministically changes common spam tokens (`free → fr33`, `prize → pr!ze`, `claim → cla1m`) and reports degradation rather than claiming general robustness. PSI monitors input-shape drift; n-gram coefficients and per-message contributions provide bounded linear-model evidence.

### Validated reference run

Against the official 5,574-row archive, the duplicate-group holdout contained 882 messages from 802 groups. The candidate achieved 0.9902 AUCPR versus 0.9864 for the word-only baseline, 0.9975 ROC-AUC and a 0.0091 Brier score. At the calibration-selected policy it reached 97.4% precision and 94.1% recall; the deterministic obfuscation challenge retained 0.9904 AUCPR and 91.6% block recall. These figures validate this fixed benchmark split, not future production traffic. The data run also suppressed all 25 intentionally replayed deliveries and passed all ten publication gates.

## Exact source and rights

- **Provider:** UC Irvine Machine Learning Repository
- **Dataset:** https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection
- **Exact archive:** https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- **DOI:** https://doi.org/10.24432/C5CC84
- **Files used:** `SMSSpamCollection`; each line contains `ham`/`spam`, a tab and raw message text
- **Scale:** 5,574 labeled messages; no missing values reported by UCI
- **Cadence:** static research corpus donated 21 June 2012; no live update schedule
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Collection provenance:** combined from Grumbletext, the NUS SMS Corpus, Caroline Tagg's thesis collection and SMS Spam Corpus v0.1 Big, as documented by UCI

The application preserves UCI attribution and does not redistribute a persisted raw corpus inside the repository; it retrieves the public archive at runtime and uses a generated fallback when unavailable.

## Modules

- `src/data.py` — safe extraction and deterministic fallback
- `src/pipeline.py` — replay audit, contracts, privacy transforms, quarantine and lineage
- `src/model.py` — group splits, baseline/candidate, Platt calibration, policy, attack suite, drift and serving
- `ui.py` — responsive control plane, operational/model views and inference workbench
- `tests/test_gateway.py` — Data Engineering and AI Engineering behavior tests

## Setup and tests

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q tests
```

No secrets or paid credentials are required. Tests cover fallback determinism, normalization, privacy tokens, quarantine, duplicate-group isolation, idempotent hashes, source failure, unsafe archives, reproducibility, baseline promotion, probability bounds, adversarial transforms and inference edge cases.

## Limitations and production extensions

The corpus is old, English-only and assembled from multiple sources. Its prevalence does not match a specific production environment. Exact duplicate grouping does not catch paraphrases. Regex tokenization can miss personal data or over-redact benign text. The attack suite covers only simple substitutions. Coefficients are not causal explanations.

A production version should add consent and retention controls, encrypted raw storage, access audit, language routing, near-duplicate clustering, current abuse labels, delayed feedback joins, active learning, reviewer disagreement, fairness analysis, policy/version registry, canary release, rollback, rate limiting, appeals and red-team evaluation.

## Hosted use

The project is registered in the central `streamlit_app.py`. Once that entrypoint is connected to Streamlit Community Cloud, it appears under **Message Trust Gateway** and subsequent `main` commits update the same hosted portfolio.
