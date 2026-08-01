# FX Market Regime Intelligence

A deployment-ready Streamlit product for exploring euro reference exchange rates,
market regimes, unusual multi-currency sessions, correlation structure and simple
exposure scenarios.

## Problem

An exchange rate alone says little about whether a move is routine, volatile or
unusual across the wider currency market. Analysts also need to understand
co-movement, concentration and the financial meaning of a hypothetical rate
change. This application turns official daily ECB observations into a transparent
monitoring and scenario workflow.

It is an analytical portfolio project—not trading advice, a price feed for
transactions, a forecast or an official ECB risk product.

## What the application does

- compares up to ten daily euro reference-rate series;
- rebases currencies to 100 for comparable paths;
- calculates 30-session changes, rolling annualised volatility, historical VaR,
  worst session and maximum drawdown;
- classifies calm, normal and stress volatility regimes without look-ahead;
- detects unusual multi-currency sessions with Isolation Forest;
- shows daily-return correlations and hierarchical behavior groups;
- creates a transparent inverse-volatility reference allocation;
- translates a hypothetical rate shock into foreign-currency exposure values;
- exports a complete currency risk audit;
- uses an unmistakably labelled deterministic fallback when the source fails.

## Data source

- **Provider:** European Central Bank (ECB)
- **Dataset:** Exchange Rates (`EXR`)
- **Dataset page:** <https://data.ecb.europa.eu/data/datasets/EXR>
- **API documentation:** <https://data.ecb.europa.eu/help/api/data>
- **REST endpoint:** <https://data-api.ecb.europa.eu/service/data/EXR>
- **Series pattern:** `EXR.D.<CURRENCY>.EUR.SP00.A`
- **Retrieval:** one SDMX REST query requesting `format=csvdata` and observations
  since 1 January 2021; no key, registration or paid credential is required
- **Update cadence:** daily reference-rate observations on ECB publication days
- **Reuse:** public ESCB statistics may be reused free of charge when the source
  is quoted and modifications are stated; see the
  [official reuse policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)

Currencies loaded: AUD, CAD, CHF, CNY, GBP, JPY, NOK, PLN, SEK and USD. The
reference quote is foreign-currency units per euro.

### Fields used

| ECB field | App field | Use |
|---|---|---|
| `KEY` | `series_key` | source-series audit identifier |
| `FREQ` | validation | daily observations only (`D`) |
| `CURRENCY` | `currency` | comparison and grouping |
| `CURRENCY_DENOM` | validation | euro denominator only (`EUR`) |
| `EXR_TYPE` | validation | spot reference-rate type (`SP00`) |
| `EXR_SUFFIX` | validation | average series (`A`) |
| `TIME_PERIOD` | `date` | daily observation date |
| `OBS_VALUE` | `rate_per_eur` | positive quoted rate |
| `OBS_STATUS` | `observation_status` | source status retained for auditability |

## End-to-end pipeline

1. `src/data.py` builds one bounded SDMX query for all ten currency series.
2. `parse_ecb_csv()` validates dimensions, parses dates and numeric values,
   removes invalid/duplicate observations and retains official source keys.
3. `rate_matrix()` aligns the selected currencies by date and forward-fills at
   most three missing publication days.
4. `src/analytics.py` calculates log returns, normalized paths and risk measures.
5. Expanding historical volatility thresholds classify each session's regime.
6. Isolation Forest scores unusual combinations of currency returns and market
   conditions.
7. Hierarchical clustering groups currencies by return correlation.
8. `ui.py` renders responsive controls, metrics, Plotly charts, explanations,
   scenario outputs, fallback/error states and CSV export.

## Analytical methods

### Returns and volatility

For rate `P`:

```text
daily log return = ln(P_t / P_(t-1))
annualised volatility = std(daily log returns) × sqrt(252)
```

A positive return means one euro buys more units of that foreign currency.

### Historical Value at Risk

The displayed 95% historical VaR is the negative fifth percentile of observed
daily log returns. It is a descriptive tail statistic, not a maximum possible
loss and not a parametric forecast.

### Regimes

The daily market volatility feature is the median rolling annualised volatility
across selected currencies. For each date, an expanding history with at least 60
observations determines thresholds:

- at or below the historical 35th percentile: `Calm`;
- at or above the historical 80th percentile: `Stress`;
- otherwise: `Normal`.

Because thresholds use expanding data, earlier states never use future values.

### Isolation Forest

The model uses every selected currency's daily return plus median absolute move,
cross-currency dispersion and rolling market volatility. It builds 250 trees with
`random_state=42`. The user-selected contamination rate controls the approximate
share flagged as unusual. A percentile rank converts the model decision function
to a readable 0–100 anomaly score.

### Correlation groups and reference allocation

Agglomerative clustering uses average linkage and distance `1 - correlation`.
The inverse-volatility view assigns weights proportional to `1 / volatility` and
normalizes them to 100%. It intentionally ignores return expectations, portfolio
liabilities, transaction costs and correlations in the final weighting step.

### Exposure scenario

```text
foreign value = EUR amount × quoted rate
shocked rate = current rate × (1 + shock percentage)
```

This is a static sensitivity—not a prediction or hedging recommendation.

## Architecture and important functions

```text
fx_regime_intelligence/
├── app.py
├── ui.py
├── src/
│   ├── data.py
│   └── analytics.py
├── tests/
│   └── test_fx_analytics.py
└── README.md
```

- `fetch_rates()` performs the bounded official API request.
- `parse_ecb_csv()` validates the source schema.
- `build_demo_data()` generates reproducible correlated fallback paths.
- `risk_summary()` produces reconciled per-currency metrics.
- `market_regimes()` implements expanding, no-look-ahead classification.
- `detect_anomalies()` runs multivariate Isolation Forest.
- `correlation_and_clusters()` discovers behavior groups.
- `inverse_volatility_allocation()` builds the reference weighting.
- `shock_scenario()` reconciles exposure values.
- `render_dashboard()` owns the complete interaction flow.

## Run and validate

Use the central portfolio:

```bash
streamlit run streamlit_app.py
```

Or run this project independently:

```bash
streamlit run fx_regime_intelligence/app.py
```

Install and test:

```bash
pip install -r requirements.txt
pytest -q
```

No secrets, API keys or local downloads are required.

## Limitations

- ECB reference rates are informative rates, not executable transaction prices.
- Weekends and ECB non-publication days have no new observation.
- Forward-filling is limited to three rows for alignment and does not create a
  new official observation.
- Volatility regimes depend on the selected currencies and history.
- Isolation Forest identifies statistical rarity, not causes or future direction.
- VaR and drawdown are historical and can understate future extremes.
- Currency exposure direction depends on the real asset or liability context.
- Synthetic fallback values are only for interface demonstration.

## Possible extensions

- user-uploaded invoice or revenue exposure mapped locally to currencies;
- hedging-ratio and cash-flow-at-risk scenarios;
- event annotation for ECB decisions and macroeconomic releases;
- walk-forward regime stability and anomaly validation;
- transaction-cost-aware portfolio optimization;
- scheduled alerts when the market enters a stress state.
