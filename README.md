# Data Science Portfolio Projects

A growing collection of portfolio-ready data products built from public,
source-documented data. The projects combine ingestion, analytics, visualization,
testing and deployment-ready user interfaces.

## Live portfolio

The repository is prepared as one central Streamlit Community Cloud application.
Deploy `streamlit_app.py` once at [share.streamlit.io](https://share.streamlit.io/)
to use every project in the browser. Future commits to `main` update that same
app automatically.

## Projects

| Project | Description | Stack |
|---|---|---|
| [Job Market Analytics Dashboard](./job-market-analytics-dashboard) | Streamlit dashboard that collects job postings from the Arbeitnow API and analyzes labor-market demand by keyword, location, company, remote status, and skills. | Python, Streamlit, Pandas, Plotly, API ingestion |
| [European Air Quality Intelligence](./air_quality_intelligence) | Live seven-day city comparison for European AQI, pollutant drivers, poor-air hours and exportable forecasts. | Python, Streamlit, Pandas, Plotly, Open-Meteo/CAMS |
| [Energy Transition Intelligence](./energy_transition_intelligence) | European transition benchmark with a transparent composite score, K-means peer groups, PCA projection and an interactive scenario lab. | Python, Streamlit, World Bank API, scikit-learn, Plotly |
| [NYC Collision Risk Intelligence](./nyc_collision_intelligence) | Daily geospatial safety explorer with outcome-aware hotspots, robust median/MAD anomaly alerts and reported-factor severity analysis. | Python, Streamlit, NYC Open Data, Pandas, Plotly |
| [Global Seismic Activity Intelligence](./earthquake_intelligence) | Near-real-time earthquake explorer with physical energy features, haversine DBSCAN sequences, robust activity anomalies and Gutenberg-Richter analysis. | Python, Streamlit, USGS GeoJSON, scikit-learn, Plotly |
| [EU Inflation & Household Basket Intelligence](./eu_inflation_intelligence) | Harmonised European inflation comparison with category pressure, a transparent personal-basket estimator and contribution analysis. | Python, Streamlit, Eurostat JSON-stat API, Pandas, Plotly |
| [Orbital Launch Reliability Intelligence](./orbital_launch_intelligence) | Global launch operations explorer with Wilson reliability intervals, cadence, provider concentration, launch-pad geography and upcoming missions. | Python, Streamlit, Launch Library 2, Pandas, Plotly |
| [Food Label & Product Choice Intelligence](./food_label_intelligence) | Packaged-food comparison with an explainable preference score, nutrient similarity, brand analytics and explicit missing-data audits. | Python, Streamlit, Open Food Facts, Pandas, Plotly |
| [Cyber Vulnerability Prioritization Intelligence](./cyber_vulnerability_intelligence) | Confirmed-exploitation triage with current EPSS, transparent urgency scoring, remediation-capacity scenarios and TF-IDF related-record retrieval. | Python, Streamlit, CISA KEV, FIRST EPSS, scikit-learn, Plotly |
| [FX Market Regime Intelligence](./fx_regime_intelligence) | Daily euro reference-rate monitor with expanding volatility regimes, Isolation Forest anomalies, behavior clusters and exposure scenarios. | Python, Streamlit, ECB Data API, scikit-learn, Plotly |
| [Biodiversity Observation Intelligence](./biodiversity_observation_intelligence) | European species-occurrence explorer with GBIF taxonomy, full-query reporting facets, haversine DBSCAN and provenance-aware quality audits. | Python, Streamlit, GBIF, Darwin Core, scikit-learn, Plotly |
| [Open Source Repository Health Intelligence](./open_source_health_intelligence) | Public GitHub delivery workbench with censoring-aware resolution curves, backlog aging, contributor concentration and release cadence. | Python, Streamlit, GitHub REST API, Kaplan–Meier, Plotly |

## Repository structure

```text
projects/
├── .streamlit/
│   └── config.toml
├── air_quality_intelligence/
├── biodiversity_observation_intelligence/
├── cyber_vulnerability_intelligence/
├── energy_transition_intelligence/
├── earthquake_intelligence/
├── eu_inflation_intelligence/
├── food_label_intelligence/
├── fx_regime_intelligence/
├── job-market-analytics-dashboard/
├── nyc_collision_intelligence/
├── open_source_health_intelligence/
├── orbital_launch_intelligence/
├── portfolio_pages/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Goal

The goal is to demonstrate practical data science and analytics engineering
through useful, transparent mini-products: API ingestion, transformation,
exploratory analysis, metric design, dashboarding, resilient error handling,
testing and cloud-ready delivery.
