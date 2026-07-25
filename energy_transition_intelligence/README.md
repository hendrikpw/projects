# Energy Transition Intelligence

An interactive, browser-based data science product for comparing European
energy-transition profiles. It combines public World Bank indicators with a
transparent composite score, unsupervised learning and a scenario simulator.

The project is integrated into the repository's central Streamlit portfolio.
Once that app is connected to Streamlit Community Cloud, no local setup or
download is required.

## Problem

Energy-transition comparisons are often reduced to a single measure such as
renewable generation. That misses important trade-offs: a country can have a
large renewable share while still emitting substantial CO2 per person or using
energy inefficiently.

This mini-product answers four practical questions:

1. Which selected country has the strongest relative transition profile?
2. Which countries have structurally similar profiles?
3. How have renewable-electricity shares changed over time?
4. How would explicit improvements affect a country's relative score?

## What the application does

- compares up to 20 European countries;
- calculates a documented 0-100 Energy Transition Score;
- exposes coverage and every input used in the score;
- groups countries with reproducible K-means clustering;
- projects the three model dimensions into a two-dimensional PCA map;
- charts annual renewable-electricity trajectories;
- simulates renewable, emissions and energy-intensity improvements;
- exports the reviewed comparison as CSV;
- falls back to clearly labelled deterministic demo data if the API fails.

The design uses the portfolio's shared Audi-inspired editorial system: dark
anthracite surfaces, restrained red accents, large typography, responsive
layouts, scroll-driven reveals and reduced-motion support.

## Data source

**Provider:** World Bank  
**Dataset:** World Development Indicators (WDI)  
**API version:** Indicators API V2  
**Authentication:** none  
**Dataset cadence:** annual; release timing and the latest year vary by
indicator and country  
**Query window:** 2010-2024  
**Retrieval:** HTTPS GET requests, JSON response, up to 20,000 rows per
indicator request  
**Retrieved fields:** `country.id`, `country.value`, `countryiso3code`,
`indicator.id`, `indicator.value`, `date`, `value`, `unit`  
**License:** Creative Commons Attribution 4.0 for WDI, subject to the World
Bank's dataset terms

Exact resources:

- [World Bank Indicators API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)
- [World Development Indicators catalog entry](https://datacatalog.worldbank.org/search/dataset/0037712/world-development-indicators)
- [World Bank data licensing](https://datacatalog.worldbank.org/public-licenses)
- [Renewable electricity output API](https://api.worldbank.org/v2/country/all/indicator/EG.ELC.RNEW.ZS?format=json)
- [CO2 emissions per-capita API](https://api.worldbank.org/v2/country/all/indicator/EN.ATM.CO2E.PC?format=json)
- [Primary-energy intensity API](https://api.worldbank.org/v2/country/all/indicator/EG.EGY.PRIM.PP.KD?format=json)
- [Electric power consumption API](https://api.worldbank.org/v2/country/all/indicator/EG.USE.ELEC.KH.PC?format=json)

### Indicators

| Code | Field used | Unit | Analytical role |
|---|---|---|---|
| `EG.ELC.RNEW.ZS` | Renewable electricity output | % of total output | 50% of score, higher is better |
| `EN.ATM.CO2E.PC` | CO2 emissions | metric tons per capita | 30% of score, lower is better |
| `EG.EGY.PRIM.PP.KD` | Primary-energy intensity | MJ per 2021 PPP dollar GDP | 20% of score, lower is better |
| `EG.USE.ELEC.KH.PC` | Electric power consumption | kWh per capita | contextual audit field |

The CO2 indicator may be published with a longer delay than other WDI series.
For this reason, the application retains and displays the latest non-null value
per country and indicator instead of silently discarding countries that do not
share one common reporting year.

## Data pipeline

1. `src/data.py` requests each indicator for the supported country set.
2. The response parser keeps valid country-level observations and converts
   years and values to numeric types.
3. `latest_snapshot` chooses the most recent non-null observation separately
   for each country and indicator.
4. `score_countries` median-imputes missing score fields, records how many
   values were imputed and min-max normalizes each component.
5. `cluster_countries` standardizes the three score inputs, trains K-means with
   `random_state=42` and `n_init=20`, then runs two-component PCA for display.
6. `ui.py` renders controls, KPIs, charts, the scenario model, caveats and CSV
   export.

API results are cached in Streamlit for 24 hours. A new cloud session or cache
expiry triggers a fresh request. Because WDI itself is annual, more frequent
polling would not add analytical value.

## Score methodology

For every selected comparison set:

```text
Score =
    0.50 × normalized renewable electricity output
  + 0.30 × inverse-normalized CO2 emissions per capita
  + 0.20 × inverse-normalized primary-energy intensity
```

The result is multiplied by 100. Min-max normalization makes the score easy to
inspect but also means it is **relative to the selected countries**. It is not
an official World Bank index. If an input is missing, the comparison median is
used and the imputation is surfaced in the audit table.

The scenario lab keeps the comparison range fixed, applies only the chosen
changes and recalculates the selected country's score. It is a sensitivity
analysis, not a forecast.

## Architecture

```text
energy_transition_intelligence/
├── app.py                  # standalone Streamlit entrypoint
├── ui.py                   # complete interactive product
├── src/
│   ├── data.py             # API client, schema and demo fallback
│   └── analytics.py        # score, clustering, PCA and scenario logic
├── tests/
│   └── test_energy_analytics.py  # analytical invariants
└── README.md
```

Important functions:

- `fetch_world_bank_data()` performs source retrieval and validation.
- `_parse_indicator_payload()` converts the nested API response to tidy data.
- `build_demo_data()` creates a fixed synthetic fallback; its observations are
  always marked `is_demo=True`.
- `latest_snapshot()` retains both latest values and reporting years.
- `score_countries()` calculates components, coverage, score and rank.
- `cluster_countries()` provides reproducible K-means and PCA output.
- `scenario_score()` calculates current and user-defined scenario scores.
- `render_dashboard()` owns controls, empty/error states and presentation.

## Local setup

The preferred route is the central hosted portfolio. For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Run tests from the repository root:

```bash
pytest energy_transition_intelligence/tests -q
```

## Error and fallback behaviour

Network timeouts, HTTP errors and structurally invalid API responses trigger a
deterministic synthetic dataset. The interface displays a warning and never
describes fallback observations as World Bank measurements. The fallback exists
only so the mini-product remains demonstrable during an upstream outage.

## Limitations

- Country scores depend on the selected comparison group.
- Indicator reporting years can differ.
- Median imputation reduces information and is explicitly disclosed.
- K-means assumes roughly spherical clusters and does not establish causality.
- PCA axes are mathematical combinations, not policy concepts.
- National averages hide sectoral and regional inequalities.
- The scenario lab does not model costs, feasibility, time or rebound effects.
- The latest API year is capped at 2024 in this version for a stable comparison
  window and can be extended in one configuration parameter.

## Possible extensions

- dynamically ingest all World Bank economies and income classifications;
- include installed renewable capacity and energy-security measures;
- add confidence bands for delayed or revised indicators;
- create sector-specific power, transport and heating views;
- estimate transition pathways with time-series forecasting;
- persist historical score snapshots for change monitoring.
