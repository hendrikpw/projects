# EU Inflation & Household Basket Intelligence

A deployment-ready Streamlit mini-product for comparing official harmonised
inflation across Europe and translating category pressure into a transparent,
user-defined household basket.

The project is available from the repository's central `streamlit_app.py`
portfolio. Once that application is deployed on Streamlit Community Cloud,
future commits to `main` are reflected at the same URL.

## Problem

Headline inflation is comparable across countries, but it rarely matches one
household's spending mix. This application keeps both concepts separate:

1. it explores the official Harmonised Index of Consumer Prices (HICP);
2. it creates an explicitly non-official household estimate from user weights;
3. it exposes the categories and percentage-point contributions behind that
   estimate.

## What the application does

- compares monthly all-items annual inflation across up to eight economies;
- displays the latest category pressure as a country/category heatmap;
- lets users weight food, housing, transport and other household categories;
- calculates an expenditure-weighted personal basket estimate;
- decomposes that estimate into percentage-point contributions;
- translates the estimate into illustrative monthly and annual euro pressure;
- measures the share of categories with inflation above 2%;
- compares countries by level, month-to-month acceleration of the annual rate,
  12-month volatility and a robust median/MAD distance;
- retains exact observation periods in an audit table;
- exports the selected official observations to CSV;
- uses a clearly labelled deterministic fallback if Eurostat is unavailable.

## Data source

| Item | Detail |
|---|---|
| Provider | Eurostat, Statistical Office of the European Union |
| Dataset | `prc_hicp_minr` — Harmonised index of consumer prices (HICP), ECOICOP version 2, monthly data |
| Dataset page | <https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_minr/default/table> |
| Persistent identifier | <https://doi.org/10.2908/PRC_HICP_MINR> |
| API endpoint | <https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_minr> |
| Retrieval | Eurostat Statistics API, JSON-stat response, no API key |
| Frequency | Monthly; publication timing can differ by country/category |
| Measure | `unit=RCH_A`, change in one month compared with the same month one year earlier, percent |
| Geography | Euro area plus 16 selected EU economies |
| Classification | `coicop18`: `TOTAL` and the 13 ECOICOP version 2 divisions |

The selected fields are `freq`, `unit`, `coicop18`, `geo`, `time` and `value`
(renamed to `rate`). The API's dimension metadata is decoded rather than relying
on a hard-coded array position.

Eurostat's [copyright and reuse
notice](https://ec.europa.eu/eurostat/help/copyright-notice) generally permits
reuse of Eurostat material free of charge with source attribution, subject to
the listed conditions and third-party exceptions. This project adapts and
visualises Eurostat data; Eurostat is not responsible for the adaptation.

## Pipeline

```text
Eurostat Statistics API
        │
        ▼
JSON-stat dimension decoder
        │
        ▼
Schema, type and range validation
        │
        ├── live observations
        └── labelled deterministic fallback on failure
        │
        ▼
Latest-series / momentum / volatility features
        │
        ├── official cross-country comparison
        ├── category heatmap
        └── user-weighted basket decomposition
        │
        ▼
Streamlit controls, Plotly charts, audit table and CSV
```

## Analytical method

### Annual rate and acceleration

The official `RCH_A` observation is already calculated by Eurostat. The app
defines monthly acceleration as:

```text
latest annual rate − preceding month's annual rate
```

This is a change in the year-on-year rate, expressed in percentage points. It
does not represent the one-month price change.

### Household basket

For available categories, input weights are normalised to 100%:

```text
normalised weightᵢ = input weightᵢ / Σ input weights
personal rate = Σ(normalised weightᵢ × category annual rateᵢ)
```

The contribution for a category is its normalised weight multiplied by its
annual rate. Contributions reconcile exactly to the personal basket estimate.
This calculation is illustrative and does not replace official national HICP
weights or a household cost-of-living calculation.

### Breadth, volatility and robust distance

- Inflation breadth is the share of non-total divisions whose latest rate is
  above 2%.
- Volatility is the standard deviation of the last 12 published annual rates.
- Robust distance is `(rate − cross-country median) / (1.4826 × MAD)`. It is a
  descriptive comparison, not an alert or prediction.

### Spending pressure

```text
monthly pressure = monthly basket spend × personal annual rate / 100
annual pressure = monthly pressure × 12
```

This assumes the latest annual rate applied to an unchanged basket and is shown
only as a like-for-like illustration.

## Code guide

- `src/data.py`
  - `parse_jsonstat()` decodes arbitrary JSON-stat dimension ordering.
  - `fetch_hicp()` submits a bounded, multi-country API query.
  - `_prepare_frame()` validates types, dimensions, ranges and uniqueness.
  - `build_demo_data()` supplies reproducible, visibly synthetic fallback data.
  - `load_data()` selects live or fallback mode and returns source metadata.
- `src/analytics.py`
  - `latest_observations()` aligns current and previous published rates.
  - `country_summary()` computes momentum, rolling volatility and robust distance.
  - `personal_basket()` normalises weights and reconciles contributions.
  - `inflation_breadth()` and `spending_pressure()` calculate explainable KPIs.
- `ui.py`
  - renders controls, state messages, metrics, charts, audit table and export;
  - caches the bounded source response for six hours;
  - applies the portfolio's shared responsive Audi-inspired design system.
- `tests/test_inflation_analytics.py`
  - validates JSON-stat ordering, acceleration, basket reconciliation,
    spending arithmetic and deterministic fallback generation.

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The standalone page can also be started with:

```bash
streamlit run eu_inflation_intelligence/app.py
```

No secret or paid credential is required.

## Limitations

- HICP is designed for harmonised macroeconomic comparison, not one person's
  exact cost of living.
- Category rates can have different latest publication periods.
- The personal basket uses division-level rates and omits product-level choices,
  quality changes, regional variation, substitution and individual contracts.
- Negative contributions are possible when a category is in deflation.
- The app is descriptive; it does not forecast future inflation.
- The selected geography set is intentionally bounded for predictable hosted
  performance.

## Possible extensions

- official country-specific HICP weights alongside personal weights;
- contributions from month-on-month index movements;
- energy and food subcategory drill-down;
- purchasing-power scenarios with income growth;
- revision tracking and publication-calendar metadata;
- downloadable calculation report with saved basket profiles.
