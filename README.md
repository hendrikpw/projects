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

## Repository structure

```text
projects/
├── .streamlit/
│   └── config.toml
├── air_quality_intelligence/
├── energy_transition_intelligence/
├── job-market-analytics-dashboard/
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
