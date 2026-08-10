# MovieLens Recommendation Serving Pipeline

An end-to-end Data Engineering and AI Engineering product that converts the
MovieLens `latest-small` archive into a contracted interaction data product and an
evaluated, diversity-aware recommendation service with an explicit cold-start path.

> Recommendations estimate preference from historical ratings. They do not measure
> artistic quality, guarantee enjoyment or infer demographic characteristics.

## Problem and product behavior

A recommender is more than a similarity function. Production-oriented work must
verify source files, preserve lineage, prevent a user's future interaction from
entering training, compare against a simple baseline, define the candidate catalog,
measure catalog effects and handle users with no history.

The hosted application provides:

- versioned archive download with timeout, retry and byte-size guard;
- strict ZIP-member allowlisting and schema validation before extraction;
- Bronze manifests and SHA-256 hashes for source files, archive and layers;
- typed Silver contracts for movies, ratings, tags and external links;
- reason-coded rating quarantine and exact reconciliation;
- foreign-key validation between interactions, tags and the movie catalog;
- a Gold movie feature product with rating, tag, genre and novelty measures;
- ten data-quality, scale and referential-integrity gates;
- deterministic run IDs, stage timings and audit metadata;
- an implicit-feedback latent-factor recommender using Truncated SVD;
- chronological leave-last-positive-out evaluation;
- full-unseen-catalog ranking without sampled easy negatives;
- a non-personalized popularity baseline;
- HitRate@20, MRR@20, NDCG@20, catalog coverage and novelty;
- a diversity/novelty-aware serving workbench for known users;
- an explicit genre-plus-popularity fallback for cold-start users;
- model-promotion status, empty states, failure states and audit exports;
- deterministic source-shaped demo data if the archive is unavailable.

## Architecture and data flow

```text
GroupLens ml-latest-small.zip
        │ timeout, retry, size guard, strict member allowlist
        ├──────── failure ───────> deterministic source-shaped fallback
        ▼
Bronze manifest
        │ movies / ratings / tags / links + SHA-256 content hashes
        ▼
Silver contracts ───────────────> reason-coded rating quarantine
        │ types, UTC events, dedupe, domains, foreign keys
        ▼
Gold movie feature product
        │ popularity, rating activity, tags, genres, novelty
        ▼
Chronological recommendation lifecycle
        ├─ hide each eligible user's last positive interaction
        ├─ train latent factors on all earlier interactions
        ├─ rank against every unseen catalog item
        └─ compare with train-only popularity
        ▼
Known-user serving + explicit cold-start fallback + evaluation manifest
```

## Data Engineering implementation

### Safe batch ingestion

`src/data.py` downloads exactly one public artifact:

```text
GET https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
```

The request uses a descriptive user agent, separate connection/read timeouts and one
bounded exponential-backoff retry for transport errors, HTTP 429 and HTTP 5xx. An
archive below 100 KB is rejected before parsing.

ZIP extraction is in memory. Only these members are accepted:

- `ml-latest-small/movies.csv`
- `ml-latest-small/ratings.csv`
- `ml-latest-small/tags.csv`
- `ml-latest-small/links.csv`

Every member receives a 50 MB uncompressed-size guard. Absolute paths and parent
directory traversal are rejected. Required headers are checked before a table can
enter Bronze.

If download, ZIP integrity or source schema fails, the complete run changes to a
deterministic source-shaped fallback. Live and synthetic interactions are never
mixed. The UI clearly identifies fallback mode.

### Bronze manifest and idempotency

Bronze preserves the four source tables and creates a manifest containing table
name, row count and deterministic content hash. The downloaded ZIP also receives an
archive SHA-256 hash.

Layer hashes depend on canonical table contents rather than retrieval time. The run
ID combines archive and layer identities; identical input therefore produces the
same output identity, enabling idempotent orchestration and change detection.

### Silver data contracts

#### Ratings

| Field | Contract |
|---|---|
| `userId` | Required anonymized integer user identity |
| `movieId` | Required integer resolving to the movie catalog |
| `rating` | 0.5–5.0 in exact half-star increments |
| `rated_at` | Unix timestamp parsed to timezone-aware UTC |
| `timestamp` | Preserved original epoch seconds |

Invalid identities, out-of-range ratings, invalid timestamps, unknown movie
references and duplicate `(userId, movieId, timestamp)` events enter quarantine.

#### Movies, tags and links

- movies require unique `movieId`, non-empty title and source genre string;
- tags require user, movie, non-empty tag and valid UTC timestamp;
- links preserve one `movieId` mapping to IMDb and TMDB identifiers when present;
- all rating and tag movie IDs must resolve to the Silver movie dimension.

Free-text tags are preserved for product diagnostics but are not used by the
collaborative model, avoiding accidental leakage from tags written after a rating.

### Gold feature product

The Gold movie table includes:

- rating count, mean rating and unique users;
- positive-rating count using the four-star threshold;
- first and last rating time;
- tag and unique-tag counts;
- source genre string and genre count;
- parsed release year when available;
- positive-interaction popularity share;
- self-information novelty in bits.

Gold is used by the interface, serving explanations, novelty reranking and data
exports. Model training continues to use the contracted event table directly.

### Quality and observability

Ten gates cover required tables, rating reconciliation, movie and rating identities,
the rating domain, rating and tag foreign keys, UTC timestamps, minimum dataset
scale and exact Gold/movie reconciliation.

The stage ledger records input rows, output rows, rejected records, execution time,
status and content hash. The manifest export combines source identity, every gate,
stage lineage, model configuration and evaluation metrics.

## AI Engineering implementation

### Feedback transformation

MovieLens contains explicit ratings. The collaborative model converts them to a
non-negative preference strength:

```text
implicit_weight = clip((rating − 2.5) / 2.5, 0, 1)
```

Examples:

- 5.0 stars → weight 1.0
- 4.0 stars → weight 0.6
- 3.0 stars → weight 0.2
- 2.5 stars or below → weight 0

This treats low ratings as an absence of positive evidence rather than inventing
negative exposure information. It is a modeling assumption, not a GroupLens rule.

### Latent-factor model

A sparse user-by-movie matrix is decomposed with deterministic `TruncatedSVD`.
The default 48 latent factors capture shared rating patterns without using titles,
genres, tags or user demographics. User-factor and item-factor multiplication
produces a score for every catalog movie.

Already seen training movies receive `−∞` before ranking, so they can never be
recommended again. The random seed is fixed at 42 and SVD uses nine power iterations
for stable output.

### Chronological holdout

For each user with at least 20 ratings and a rating of four stars or higher:

1. interactions are sorted by event time;
2. the chronologically last positive interaction is removed;
3. every earlier interaction remains available for training;
4. the hidden movie becomes the single relevant evaluation item;
5. ranking considers the full catalog minus training-seen movies.

This is harder and more honest than evaluating against a small sampled set of random
negatives. It still measures historical re-ranking, not online satisfaction.

### Popularity baseline

The baseline ranks movies by `log1p` of their positive training interaction count.
Seen items are excluded per user. The baseline never reads holdout labels.

The promotion gate compares candidate and baseline HitRate@20. A model that does not
beat popularity is explicitly marked as failed rather than silently presented as an
improvement.

### Evaluation metrics

- **HitRate@20:** share of users whose hidden movie appears in the top 20.
- **MRR@20:** reciprocal rank of the hidden movie, zero if absent.
- **NDCG@20:** logarithmically discounted rank quality for the hidden item.
- **Hit lift:** absolute HitRate difference from popularity.
- **Catalog coverage:** unique recommended movies divided by the full catalog.
- **Novelty:** average self-information of recommended items in bits.
- **Explained variance:** variance represented by the selected SVD factors.

HitRate and MRR measure relevance recovery. Coverage and novelty expose whether the
system concentrates attention on only a few popular films. None alone is sufficient.

### Serving and diversity reranking

For a known anonymized user, serving combines normalized latent relevance with a
configurable novelty weight. A small genre-coverage bonus diversifies the final
candidate set. Explanations show shared genres from highly rated training history or
state that the result primarily comes from latent taste-neighbor evidence.

This explanation describes serving features; it is not a causal explanation of why
the user will like a film.

### Cold start and failure behavior

Collaborative factors do not exist for a user with no history. The cold-start path
therefore asks for explicit genres and combines genre overlap with training
popularity. It is clearly labeled as a different algorithm.

Unknown programmatic user IDs raise an error directing the caller to cold start.
Empty genre selection produces an informative UI state. Insufficient eligible users,
failed data gates or invalid archive contents withhold recommendations.

## Exact data source and license

- **Provider:** GroupLens Research, University of Minnesota.
- **Dataset:** [MovieLens latest-small](https://grouplens.org/datasets/movielens/).
- **Download:** `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`.
- **Official README and license:** [ml-latest-small README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).
- **Version:** generated September 26, 2018.
- **Scale documented by GroupLens:** 100,836 ratings, 3,683 tag applications, 9,742 movies and 610 anonymized users.
- **Observation period:** March 29, 1996 through September 24, 2018.
- **Update cadence:** this exact artifact is static; the `latest-small` development dataset may be replaced by GroupLens over time.
- **Credentials:** no API key or paid account required.
- **Fields used:** rating user/movie IDs, half-star rating, UTC timestamp; movie title/genres; tag user/movie IDs, tag and timestamp; IMDb/TMDB link IDs.
- **Dataset paper:** [Harper & Konstan, 2015](https://doi.org/10.1145/2827872).

The official license permits research use with attribution and redistribution under
the same conditions, forbids implied endorsement, disclaims warranties, and requires
prior permission for commercial or revenue-bearing use. This repository provides a
non-commercial portfolio demonstration and does not imply endorsement by GroupLens
or the University of Minnesota.

## Modules and important functions

| File | Responsibility |
|---|---|
| `src/data.py` | Download, retry, ZIP guards, header contracts and fallback |
| `src/pipeline.py` | Bronze manifests, Silver contracts, quarantine, Gold features, DQ and hashes |
| `src/model.py` | Temporal split, sparse factors, baseline, full-catalog evaluation and serving |
| `ui.py` | Responsive control plane, model evaluation, known-user/cold-start workbench and exports |
| `tests/test_pipeline.py` | Schemas, references, quarantine, idempotency, archive and fallback behavior |
| `tests/test_model.py` | Shapes, metrics, reproducibility, seen-item exclusion, cold start and edge cases |

## Setup and usage

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Choose **MovieLens Recommendation Serving Pipeline** in the central navigation.
Standalone execution is also supported:

```bash
streamlit run movielens_recommendation_pipeline/app.py
```

No secrets are required. `config.example.toml` documents safe defaults.

## Testing

```bash
python -m pytest -q movielens_recommendation_pipeline/tests
```

Tests cover fallback determinism, source-shaped schemas, UTC typing, foreign keys,
invalid and duplicate quarantine, exact reconciliation, idempotent layer hashes,
archive allowlisting, retry recovery, model output shapes, metric domains,
reproducibility, seen-item exclusion, unique recommendations, cold-start output,
unknown users and minimum-data guards.

## Operational considerations

- Pin an archive checksum in a production release rather than tracking a mutable URL.
- Persist the original archive immutably and publish versioned Silver/Gold Parquet.
- Upsert interaction events by composite identity and retain rejected records separately.
- Register factor count, split policy, training hash, candidate policy and metrics together.
- Monitor interaction volume, foreign-key failures, catalog churn and recommendation coverage.
- Require baseline lift and minimum coverage before promotion.
- Re-evaluate by user-activity and catalog-popularity slices, not only globally.
- Run online experiments before claiming satisfaction, retention or revenue impact.
- Maintain a documented deletion and retraining workflow for production user data.

## Limitations

- The dataset is static and ends in 2018; it does not represent current viewing behavior.
- It is a development dataset and GroupLens says it may change over time.
- Users all have at least 20 ratings, so true new-user behavior is absent.
- Missing ratings mean unknown exposure, not dislike.
- Historical rating data contains popularity and exposure bias.
- Titles and genres are manually entered or imported and can contain errors.
- One hidden positive item provides a narrow relevance definition.
- Offline full-catalog metrics do not measure satisfaction or long-term diversity.
- SVD scores are not calibrated probabilities.
- Genre-based cold start is intentionally simple and non-personalized.
- The hosted runtime does not persist state across Streamlit sessions.

## Extensions

1. Add weighted regularized matrix factorization optimized for implicit feedback.
2. Compare Bayesian Personalized Ranking and two-tower retrieval.
3. Add repeated temporal holdouts and user-activity confidence intervals.
4. Evaluate calibration, serendipity, intra-list diversity and popularity bias by slice.
5. Build an approximate-nearest-neighbor item index for two-stage retrieval.
6. Add feature-store snapshots and a versioned online/offline parity test.
7. Serve recommendations through FastAPI with a model registry and request tracing.
8. Add online feedback, consent, deletion, experiment and rollback workflows.

## Hosted use

The page is integrated into the stable root `streamlit_app.py`. An existing
Streamlit Community Cloud deployment updates automatically from `main`. Because no
confirmed public URL is stored in the repository, no live URL is invented. If the
portfolio is not connected yet, the single manual step is to deploy
`streamlit_app.py` once at [share.streamlit.io](https://share.streamlit.io/).
