# Federal Procurement Entity Resolution Control

A deployment-ready Data Engineering and AI Engineering product that turns a
bounded USAspending contract snapshot into governed award/recipient tables and
an evaluated supplier-name resolution service.

> Hosted use: this page is integrated into the repository's central
> `streamlit_app.py`. If Streamlit Community Cloud is not connected yet, deploy
> that root entrypoint once; future `main` commits update the same application.

## Problem and product behavior

Supplier names arrive with missing punctuation, changed legal suffixes, token
reordering, abbreviations and typographical loss. Exact joins either miss those
records or encourage unsafe manual fixes. At the same time, the reference data
must be replay-safe, traceable and structurally valid before it is allowed into
an identity service.

The application therefore provides one end-to-end control plane:

- a bounded, retrying extraction of current federal procurement awards;
- Bronze deliveries with payload hashes and intentional replay records;
- typed Silver contracts, reasoned quarantine and idempotent deduplication;
- Gold award and UEI-keyed recipient tables with referential checks;
- a character-level top-k supplier candidate service;
- separately calibrated similarity/margin thresholds and human-review routing;
- held-out Top-1, Hit@5, MRR@5, coverage, selective accuracy, false-merge and
  unknown-name rejection metrics;
- operational charts, row-level evaluation audit, interactive name workbench
  and CSV/JSON exports;
- loading, live/demo, empty, quarantine and failure states.

The match result is candidate-search assistance. It is not legal entity
verification. The official Unique Entity Identifier (UEI) remains authoritative.

## Architecture and data flow

```text
USAspending Award Search API
    │ bounded FY pages · timeout · retry · response check
    ▼
Bronze deliveries
    │ immutable payload hash · delivery ID · replay injection
    ▼
Silver procurement contract
    │ types · UEI/name/key/date/amount rules · quarantine · dedupe
    ├──────────────► rejected-delivery audit
    ▼
Gold award table ─────────► agency/value operations view
    │ foreign key
    ▼
Gold recipient reference (UEI + canonical name)
    │
    ├─► controlled corruptions ─► calibration/test isolation
    │
    ▼
char 3–5 gram TF-IDF index
    │ cosine top-k · score + runner-up margin
    ▼
auto-link candidate OR human review
```

Every layer has a deterministic content hash. `run_id` combines the source hash
and published layer hashes. The app exposes row counts, rejections and hashes in
the run ledger; the downloadable manifest preserves the same evidence.

## Data Engineering lifecycle

### Extract

`src/data.py` calls the official keyless `POST
/api/v2/search/spending_by_award/` endpoint. The default run requests six pages
of 100 awards and stops early when `page_metadata.hasNext` is false. It uses a
25-second timeout, a descriptive user agent and at most three attempts with
bounded exponential backoff for HTTP 429 and transient 5xx responses. The
service's `awards/last_updated` value sets the snapshot end date; the start is
October 1 of the corresponding federal fiscal year.

### Bronze

Each delivery gets:

- `delivery_id`: first 24 hexadecimal characters of a SHA-256 hash over the raw
  flattened values;
- `payload_hash`: full SHA-256 row fingerprint;
- all source fields without semantic type coercion.

Fifteen rows are deliberately replayed. Their removal in Silver is an executable
idempotency assertion, not a claim that the upstream source sent duplicates.

### Silver contract and quarantine

The contract parses amounts and timestamps, standardizes names/UEIs and rejects
rows with:

- missing/short award key;
- recipient name shorter than three characters;
- UEI that is not 12 alphanumeric characters;
- non-numeric, non-finite or implausibly large absolute amount;
- missing awarding agency;
- invalid start or last-modified timestamp.

The first failed rule becomes `quarantine_reason`. Valid replayed `delivery_id`s
are suppressed. Signed award values are retained because modifications can
decrease the current award total.

### Gold products and publication gates

The award product contains one stable award key plus recipient identity, current
award value, dates, agency, NAICS, PSC, description and lineage hash. The
recipient product uses UEI as its primary key, selects the most frequently
observed name as `canonical_name`, and aggregates award count, award value,
agency count and latest modification.

Publication fails closed unless all eight gates pass: source volume, typed-row
acceptance, unique award key, UEI shape, replay suppression, complete row
reconciliation, award/recipient foreign-key equality and finite amounts.

## AI Engineering lifecycle

### Verified live run

The release validation on 19 August 2026 used the USAspending snapshot reported
through 18 August 2026. Six pages produced 600 accepted awards and 465 distinct
UEIs; no live row was quarantined and all 15 injected replay deliveries were
suppressed. On 1,415 held-out corrupted names the matcher achieved 97.88% Top-1,
100% Hit@5 and 0.9888 MRR@5. The calibrated policy auto-linked 94.28% of test
queries at 99.25% selective accuracy, produced a 0.71% false-merge rate across
all test queries, and rejected all four unrelated probes. These numbers measure
the controlled test harness, not general production accuracy.

### Retrieval representation

`src/resolution.py` normalizes case, punctuation and whitespace. Each name is
represented by its original token sequence, a sorted-token view and an acronym
view. `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))` learns local
character patterns and the sparse cosine product retrieves five candidates.
This is deterministic, light enough for hosted execution and robust to small
spelling changes without an external model or secret.

### Controlled reliability evaluation

The official UEI supplies the identity label. Each canonical name produces five
deterministic failure modes:

1. case and spacing loss;
2. removal of the legal suffix;
3. first-token swap;
4. one vowel deletion;
5. first-token plus acronym abbreviation.

The UEI hash assigns the entire identity to calibration or test, so variants of
one supplier cannot cross the evaluation boundary. Calibration searches
similarity and best-vs-second-candidate margin thresholds subject to at least
95% selective accuracy and hard non-trivial floors. The untouched test set then
reports:

- **Top-1 accuracy:** correct UEI ranked first;
- **Hit@5:** correct UEI appears in the candidate set;
- **MRR@5:** reciprocal position of the correct candidate;
- **exact baseline accuracy:** normalized exact equality;
- **coverage:** test queries eligible for automatic linkage;
- **selective accuracy:** accuracy among eligible queries;
- **false-merge rate:** incorrect auto-links divided by all test queries;
- **unknown rejection:** unrelated probe names deferred by the guardrails.

Promotion fails if matching is worse than exact equality, Hit@5 is below 90%,
selective accuracy is below 90%, or fewer than 75% of unknown probes are
rejected. The UI shows accuracy and coverage separately for every corruption
type and exposes every held-out decision.

### Serving and failure states

`resolve_name()` returns up to five candidates. Automatic linkage requires both
the calibrated similarity and separation margin. Empty inputs return no result;
weak, ambiguous and zero-information names route to `human-review`. The product
does not infer ownership, parent/subsidiary relationships, responsibility,
fraud, or the legality of an entity.

## Exact source documentation

| Item | Documentation |
|---|---|
| Provider | U.S. Department of the Treasury, USAspending.gov |
| API | `https://api.usaspending.gov/api/v2/search/spending_by_award/` |
| Endpoint index | https://api.usaspending.gov/docs/endpoints |
| Exact request contract | https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/contracts/v2/search/spending_by_award.md |
| Data disclosures | https://www.usaspending.gov/data/about-the-data-download.pdf |
| Retrieval | Keyless HTTPS POST; contract award type codes A, B, C and D; current FY time filter; newest modified rows first |
| Requested fields | Award ID, recipient name, UEI and recipient ID, amount, start/end dates, awarding agency/subagency, contract award type, NAICS, PSC, description and last-modified timestamp |
| Update cadence | USAspending states that procurement records are generally available within five days of FPDS submission; certain Defense and USACE data can be delayed 90 days |
| Usage | Public federal spending information made available under the DATA Act; retain source attribution and observe the site's disclosures rather than assuming every field is complete or contemporaneous |

USAspending is the official open data source of U.S. federal spending. Data are
available from fiscal year 2001, but reporting obligations and quality vary over
time. This app uses only a bounded current-FY search result, not the entire award
universe. The sort date can reflect a modification to an older award. The amount
is the current award amount, not necessarily current-year obligations or cash
outlays.

## Reproducible fallback

If the live API, schema or minimum-volume check fails, `fallback_data(seed=42)`
creates 180 clearly labeled demonstration awards for 60 deterministic suppliers.
It uses the same fields, contracts, hashes, replay checks, evaluation and UI.
Fallback mode is always shown at the top of the app with the captured failure
reason; it never masquerades as live federal data.

## Important modules and functions

| Module | Responsibility |
|---|---|
| `src/data.py` | retries, last-updated lookup, bounded pagination, flattening and deterministic fallback |
| `src/pipeline.py` | Bronze/Silver/Gold transformations, contracts, quarantine, reconciliation, hashes and run ledger |
| `src/resolution.py` | normalization, corruption harness, identity isolation, TF-IDF retrieval, threshold selection, metrics and serving |
| `ui.py` | modern responsive control plane, charts, audits, workbench and exports |
| `tests/test_control.py` | deterministic data, contract, replay, lineage, reference, evaluation, reproducibility and fail-closed tests |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone project page:

```bash
streamlit run federal_procurement_resolution/app.py
```

No API key, paid service or secret is required. Settings are documented in
`config.example.toml`; the hosted portfolio intentionally uses bounded safe
defaults.

## Tests

```bash
pytest -q federal_procurement_resolution/tests
pytest -q
```

The project tests cover deterministic fallback, content-hash invariance,
idempotent replay behavior, publication gates, foreign-key reconciliation,
reasoned quarantine, normalization/corruptions, performance against the exact
baseline, selective metrics, reproducibility, output schema, empty-input failure
and minimum-reference protection.

## Limitations and possible extensions

- Evaluation defects are synthetic and may not match the real production error
  distribution; collect human-reviewed linkage outcomes before deployment.
- A bounded recent-modification sample is useful for demonstration but not a
  representative spending estimate or complete supplier master.
- One UEI can have legitimate name changes; parent/subsidiary and successor
  relationships require authoritative reference data beyond string similarity.
- Model thresholds must be re-evaluated when the supplier universe or error mix
  changes. The per-failure-mode panel is a starting monitor, not a substitute
  for labeled production drift data.
- A production version could persist Parquet layers in object storage, schedule
  incremental extraction from a watermark, maintain slowly changing recipient
  dimensions, add reviewed decisions to a registry, and expose a versioned
  FastAPI service with latency/error/service-level telemetry.
