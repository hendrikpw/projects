# Stack Overflow Tag Routing Pipeline

A portfolio-ready Data Engineering and AI Engineering product that turns bounded Stack Overflow question snapshots into a contracted relational data product and a time-evaluated multi-label tag recommendation service.

## Problem and product

Technical communities need tags to route questions toward relevant experts, search indexes and review queues. Manual tagging is inconsistent, while forced automated predictions hide uncertainty. This application therefore combines observable ingestion with a recommendation policy that can abstain.

The hosted interface provides pipeline stages, micro-batch replay, quality gates, quarantine, question–tag reconciliation, chronological model evaluation, popularity comparison, confidence routing, drift, per-tag performance, local evidence and CSV/JSON exports.

## Architecture

```text
Stack Exchange API /questions
  │ bounded paging · retry · timeout · API backoff · quota metadata
  ▼
Bronze API snapshot + source hash
  │ deterministic 200-row delivery replay
  ▼
Silver contracts + UTC timestamps + sanitized HTML + event identity
  ├── reason-coded quarantine
  ├── question ↔ tag bridge
  ▼
Gold privacy-tokenized text + operational features + layer/run hashes
  │ oldest 70% train / next 15% policy / newest 15% test
  ├── popularity baseline
  ├── word + character TF-IDF one-vs-rest logistic model
  ├── confidence abstention
  └── Precision@3 · Recall@3 · F1 · Brier · drift · evidence
```

Identical input produces the same Gold hash and run ID. A failed mandatory contract, reconciliation or model-promotion gate withholds publication.

## Data Engineering

`src/data.py` requests at most twelve 100-row pages, uses explicit timeouts and retries, honors API-provided `backoff`, records remaining quota, and stops when `has_more` is false. The application caches a run for six hours. A deterministic 1,800-row fallback demonstrates the entire system when the live API is unavailable.

`src/pipeline.py` implements:

1. **Bronze:** source payload with retrieval and content hashes.
2. **Replay audit:** stable 200-row batches plus twenty intentional duplicate deliveries that must be suppressed.
3. **Silver contract:** numeric question IDs, UTC creation timestamps, sanitized title/body, one-to-five normalized tags and stable event identity.
4. **Quarantine:** missing IDs, duplicate IDs, invalid dates, empty text, invalid tag counts and extreme text sizes receive explicit reasons.
5. **Relational Gold:** a reconciled question–tag bridge plus privacy-tokenized model text and length/code/question-mark features.
6. **Observability:** input/output/rejected volumes, timings, payload/layer/run hashes, batch counters, quality results and manifest metadata.

URL and email patterns are tokenized before Gold publication. This reduces direct exposure but is not complete anonymization.

## AI Engineering

The API is ordered by creation date. The oldest 70% of valid questions train the model, the next 15% select the confidence-abstention threshold, and the newest 15% remain untouched until final evaluation. Only training data chooses the twelve modeled tags.

The candidate combines word unigrams/bigrams and character-boundary 3–5 grams with twelve one-vs-rest, class-weighted logistic classifiers. Character features help with code tokens, framework spellings and naming variants. A transparent baseline always predicts the three most prevalent training tags.

Evaluation reports:

- **Precision@3:** correct tags among the three recommendation slots;
- **Recall@3:** share of modeled true tags recovered in those slots;
- micro/macro F1 at a documented probability threshold;
- multilabel Brier score for probability error;
- auto-suggestion coverage and human-review rate;
- per-tag support, F1 and prevalence movement;
- PSI for observable input-shape drift.

The candidate is published only if its Precision@3 exceeds the popularity baseline. Serving returns five ranked suggestions, a review route below the policy threshold, privacy-safe input and positive n-gram evidence for the leading tag.

### Reproducible fallback validation

The forced-offline reference run processed 1,800 questions and 3,600 question–tag edges, suppressed all twenty replayed deliveries and passed all ten publication gates. On its newest 270-row holdout, candidate Precision@3 was 38.0% versus 13.8% for the popularity baseline; Recall@3 was 65.6%, micro F1 0.783, macro F1 0.777 and multilabel Brier score 0.022. The policy automatically routed 80.7% and deferred 19.3%. These synthetic results validate software behavior and failure handling—not expected performance on live Stack Overflow traffic. Live metrics are recomputed and shown whenever the API is reachable.

## Exact source, fields and rights

- **Provider:** Stack Exchange, Stack Overflow site
- **Endpoint:** https://api.stackexchange.com/2.3/questions
- **Method documentation:** https://api.stackexchange.com/docs/questions
- **Paging documentation:** https://api.stackexchange.com/docs/paging
- **Parameters:** `site=stackoverflow`, `sort=creation`, `order=desc`, `pagesize=100`, `filter=withbody`
- **Fields used:** `question_id`, `creation_date`, `last_activity_date`, `title`, `body`, `tags`, `score`, `view_count`, `answer_count`, `is_answered`, `link`
- **Retrieval:** anonymous HTTPS JSON API; no paid credentials or secret required
- **Cadence:** continuously changing source; app snapshot cached for six hours
- **API limits:** maximum page size 100; anonymous access is limited to page 25; response `backoff` must be honored
- **License:** public contributions after 2 May 2018 are CC BY-SA 4.0; see https://stackoverflow.com/help/licensing

Every published question retains its direct source link for attribution. The application displays bounded excerpts, does not post or edit content, and does not claim Stack Overflow endorsement.

## Modules and operation

- `src/data.py` — API paging, retry/backoff, quota capture and fallback
- `src/pipeline.py` — contracts, replay audit, HTML/privacy transforms, bridge, quarantine and lineage
- `src/model.py` — temporal splits, multi-label model, baseline, evaluation, abstention, drift and serving
- `ui.py` — responsive control plane, evaluation views and routing workbench
- `tests/test_router.py` — Data and AI Engineering behavior tests

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q tests
```

No secrets are required. Tests cover fallback reproducibility, HTML/privacy transforms, relational reconciliation, quarantine, temporal isolation, replay/idempotency, source failure, candidate promotion, model reproducibility, output shape and empty input.

## Operational limits and extensions

A recent bounded sample is not representative of all Stack Overflow history. Rare or emerging tags outside the top-twelve vocabulary cannot be recommended. Tags can be edited after retrieval. HTML sanitization intentionally discards presentation. Confidence is not correctness, and n-gram weights are associations rather than causal explanations.

A production extension should use cursor-based persisted snapshots, revision ingestion, delayed-label joins, a tag taxonomy, rare-label retrieval, multilingual routing, reviewer feedback, per-language fairness analysis, stronger code-aware encoders, model/version registries, canary deployment, rollback, access logs and explicit retention controls.

## Hosted use

The page is registered in the repository's central `streamlit_app.py`. Once that entrypoint is connected to Streamlit Community Cloud, **Stack Overflow Tag Routing Pipeline** is available in the same browser-based portfolio and future `main` commits update it automatically.
