# Food Label & Product Choice Intelligence

A polished Streamlit mini-product for comparing packaged food labels, auditing
data quality and finding nutritionally similar products through an explainable
preference model.

The project is integrated into the repository's central `streamlit_app.py`.
Once that app is deployed on Streamlit Community Cloud, commits to `main`
appear at the same URL.

## Problem

Food databases contain useful nutrient labels, but comparisons are easily
distorted by missing fields, incompatible priorities and small brand samples.
This app separates three ideas:

1. official or source-provided fields such as Nutri-Score and NOVA;
2. a transparent, user-weighted within-sample comparison score;
3. data completeness and similarity diagnostics.

It does not diagnose health, prescribe a diet or label a product as universally
healthy.

## Features

- eight selectable product categories and six markets;
- one bounded structured API search per category/market;
- filters for nutrient-field coverage and official Nutri-Score;
- adjustable priorities for sugar, salt, saturated fat, fibre and protein;
- an explainable 0–100 Choice Fit percentile score;
- sugar/protein/fibre product landscape;
- brand-level sample-aware comparison;
- product ingredient, additive and allergen inspection;
- z-score nutrient similarity search across eight fields;
- field-by-field missingness report;
- CSV export and deterministic synthetic fallback.

## Data source and reuse

| Item | Detail |
|---|---|
| Provider | Open Food Facts |
| Database | Community-contributed packaged food product database |
| Production API | <https://world.openfoodfacts.org/api/v2/search> |
| API documentation | <https://openfoodfacts.github.io/openfoodfacts-server/api/> |
| Terms | <https://world.openfoodfacts.org/terms-of-use> |
| Retrieval | Structured v2 search, JSON, read-only, no account or API key |
| Search rate limit | 10 requests per minute per IP |
| Cache | Six hours for each category/market combination |
| Database licence | Open Database License (ODbL): attribution and share-alike |
| Individual contents | Database Contents License |
| Images | CC BY-SA with possible additional rights; not used by this project |

The API documentation currently recommends v3 for individual product reads,
but structured filter search remains available only through `/api/v2/search`.
The app always sends a custom User-Agent as required by Open Food Facts.

### Fields used

- identification: barcode, product name, brand and quantity;
- source classifications: `nutrition_grades` and `nova_group`;
- nutrients per 100 g/ml: energy, fat, saturated fat, carbohydrates, sugars,
  fibre, protein and salt;
- label context: ingredients, additives, allergens and labels;
- quality metadata: source completeness and last modification time.

No product images or personal data are retrieved.

## Pipeline

```text
Category + market controls
        │
        ▼
Open Food Facts structured search · max 100 products
        │
        ▼
Nested JSON / nutriment flattening
        │
        ▼
Schema, range, duplicate and coverage validation
        │
        ├── live product sample
        └── explicitly labelled deterministic fallback
        │
        ▼
Percentile scoring · brand aggregation · similarity · missingness
        │
        ▼
Streamlit controls · Plotly charts · audit tables · CSV
```

## Analytical methods

### Choice Fit

For each selected priority, the nutrient is converted to a percentile within
the active product sample:

- lower values rank higher for sugars, salt and saturated fat;
- higher values rank higher for fibre and protein.

For product `i`:

```text
Choice Fitᵢ = Σ(weightⱼ × available percentileᵢⱼ) / Σ(available weightsⱼ)
```

Missing nutrients are not converted to zero. The score is renormalised over
the fields available for that product. The result is sample-relative and will
change with category, market, filters and weights.

### Similarity

Eight nutrient fields are:

1. median-imputed within the active sample;
2. standardised to zero mean and unit variance;
3. compared using root-mean-square Euclidean distance.

This identifies nearby recorded nutrition profiles. It is not a substitute,
health recommendation or taste prediction.

### Brand comparison

Brands must meet a user-selected minimum product count. The app then reports
median Choice Fit, sugar, protein, data coverage and NOVA group 4 share. Medians
reduce sensitivity to one extreme product but do not make the sample
representative.

### Data quality

The app reports the share of source products with each important field.
Nutrition coverage is the percentage of eight nutrient fields present on a
product. Crowdsourced missingness remains visible throughout the interface.

## Code guide

- `src/data.py`
  - `fetch_products()` performs the bounded, identified API request.
  - `parse_products()` flattens product and nutriment objects.
  - `_prepare_frame()` validates numeric ranges, coverage and uniqueness.
  - `build_demo_data()` creates stable category-shaped fallback records.
- `src/analytics.py`
  - `choice_fit_score()` implements the transparent percentile model.
  - `similar_products()` calculates standardised nutrient distance.
  - `brand_summary()` and `missingness_report()` create portfolio diagnostics.
- `ui.py`
  - renders controls, loading/error states, KPIs, charts, product workbench,
    quality audit and export.
- `tests/test_food_analytics.py`
  - validates parsing, scoring direction, filtering, similarity, aggregation,
    missingness and deterministic fallback behavior.

## Run

From the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone:

```bash
streamlit run food_label_intelligence/app.py
```

## Limitations

- Open Food Facts is crowdsourced and provides no assurance that every record is
  accurate, complete or current.
- The first 100 scan-sorted results are not a representative market sample.
- Nutrients per 100 g and per 100 ml are not always directly comparable.
- Nutri-Score applicability and formulas can vary by product type and revision.
- NOVA can be missing when ingredients are incomplete.
- Choice Fit is category-relative and is not an official or medical score.
- Similarity ignores serving size, price, taste, dietary needs and ingredient
  interactions.
- API outages or global rate limits activate synthetic demo mode.

## Possible extensions

- barcode lookup through the current v3 product endpoint;
- price data and value-for-money comparison;
- saved comparison profiles;
- serving-size normalisation;
- longitudinal label-change tracking;
- separate beverage and solid-food models;
- user-visible ODbL export attribution metadata.
