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
| [European Air Quality](./air_quality_intelligence) | Compare seven-day air-quality forecasts for European cities and see which hours and pollutants stand out. | Python, Streamlit, Pandas, Plotly, Open-Meteo/CAMS |
| [Energy Transition](./energy_transition_intelligence) | Compare European countries on renewable electricity, emissions and energy use, then try a few what-if changes. | Python, Streamlit, World Bank API, scikit-learn, Plotly |
| [NYC Crash Map](./nyc_collision_intelligence) | Explore New York City collision records by borough, time, location and reported outcome. | Python, Streamlit, NYC Open Data, Pandas, Plotly |
| [Recent Earthquakes](./earthquake_intelligence) | Browse recent USGS earthquakes by location, magnitude and depth, with nearby events grouped on the map. | Python, Streamlit, USGS GeoJSON, scikit-learn, Plotly |
| [Prices Across Europe](./eu_inflation_intelligence) | Compare official inflation rates across Europe and estimate how they would affect a household spending mix. | Python, Streamlit, Eurostat JSON-stat API, Pandas, Plotly |
| [Space Launches](./orbital_launch_intelligence) | Explore launch history, compare provider outcomes with uncertainty ranges and browse upcoming missions. | Python, Streamlit, Launch Library 2, Pandas, Plotly |
| [Food Label Comparison](./food_label_intelligence) | Compare packaged foods by their recorded nutrition labels and see where product information is missing. | Python, Streamlit, Open Food Facts, Pandas, Plotly |
| [Vulnerability Priorities](./cyber_vulnerability_intelligence) | Sort known-exploited vulnerabilities using exploit likelihood, deadlines and available remediation capacity. | Python, Streamlit, CISA KEV, FIRST EPSS, scikit-learn, Plotly |
| [Euro Exchange Rates](./fx_regime_intelligence) | Explore daily ECB exchange rates and compare how currencies have moved against the euro. | Python, Streamlit, ECB Data API, scikit-learn, Plotly |
| [Species Observations](./biodiversity_observation_intelligence) | Explore GBIF species observations by location and season, with reminders about gaps and reporting bias. | Python, Streamlit, GBIF, Darwin Core, scikit-learn, Plotly |
| [Repository Activity](./open_source_health_intelligence) | Look at public GitHub activity, including issue age, pull requests, contributors and releases. | Python, Streamlit, GitHub REST API, Kaplan–Meier, Plotly |
| [London Cycle Stations](./london_bike_rebalancing_intelligence) | Check live Santander Cycles station availability and explore a simple bike-transfer scenario. | Python, Streamlit, TfL BikePoint API, DBSCAN, Plotly |
| [Research Evidence Pipeline](./research_evidence_pipeline) | Observable Bronze/Silver/Gold literature pipeline with data contracts, deterministic lineage, hybrid semantic retrieval and citation-bound evidence briefs. | Python, Streamlit, Europe PMC, TF-IDF/SVD, scikit-learn |
| [Clinical Trial Operations ML Pipeline](./clinical_trial_ops_pipeline) | Content-addressed trial-registry pipeline with typed contracts, leakage-aware features, time-aware classification, calibration and drift monitoring. | Python, Streamlit, ClinicalTrials.gov API v2, scikit-learn, Plotly |
| [FDA Recall Triage Pipeline](./fda_recall_nlp_pipeline) | Multi-source enforcement pipeline with content hashes, typed contracts, quarantine, evaluated multi-class NLP, drift monitoring and confidence abstention. | Python, Streamlit, openFDA/RES, TF-IDF, scikit-learn, Plotly |
| [Wikipedia Attention Forecast Pipeline](./wikipedia_attention_pipeline) | Event-time pageview pipeline with micro-batch replay, watermarks, quarantine, rolling-origin forecasting, conformal intervals and anomaly review. | Python, Streamlit, Wikimedia Analytics, gradient boosting, Plotly |
| [NYC 311 Resolution Operations Pipeline](./nyc_311_resolution_pipeline) | Content-addressed civic-service pipeline with deterministic window sampling, leakage-safe resolution quantiles, calibration, baseline evaluation and drift monitoring. | Python, Streamlit, NYC Open Data, quantile boosting, Plotly |
| [MovieLens Recommendation Serving Pipeline](./movielens_recommendation_pipeline) | Contracted interaction pipeline with temporal holdout, full-catalog latent-factor evaluation, popularity baseline, novelty reranking and explicit cold start. | Python, Streamlit, GroupLens MovieLens, sparse SVD, Plotly |
| [Predictive Maintenance Decision Pipeline](./predictive_maintenance_pipeline) | Content-addressed machine-cycle pipeline with typed quarantine, target-leakage controls, calibrated rare-failure classification, drift monitoring and cost-sensitive alerting. | Python, Streamlit, UCI AI4I, gradient boosting, isotonic calibration |
| [NOAA Storm Impact Operations Pipeline](./storm_impact_pipeline) | Revision-aware storm-event pipeline with typed monetary labels, fail-closed quarantine, calibrated hurdle modeling, conditional uncertainty, drift and impact-based review ranking. | Python, Streamlit, NOAA/NCEI, gradient boosting, isotonic calibration |
| [Message Trust Gateway](./message_trust_gateway) | Replay-safe and privacy-aware SMS pipeline with duplicate-group isolation, calibrated word/character NLP, abstention, adversarial evaluation and drift monitoring. | Python, Streamlit, UCI SMS Spam, TF-IDF, logistic regression, Platt calibration |
| [Stack Overflow Tag Routing Pipeline](./stackoverflow_tag_router) | Replay-safe relational question pipeline with API backoff, question-tag reconciliation, chronological multi-label NLP evaluation, popularity baseline, abstention and drift monitoring. | Python, Streamlit, Stack Exchange API, TF-IDF, one-vs-rest logistic regression |
| [SEC Fundamentals Control](./sec_fundamentals_control) | Revision-aware XBRL pipeline with accession lineage, fail-closed quality gates, peer-local anomaly detection, controlled-stress evaluation and drift monitoring. | Python, Streamlit, SEC EDGAR XBRL, Local Outlier Factor, Plotly |
| [USGS River Flow Early-Warning Pipeline](./river_flow_early_warning) | Replay-safe multi-gauge pipeline with source qualifiers, content-addressed lineage, calibrated three-day high-flow classification, climatology baseline and drift monitoring. | Python, Streamlit, USGS Water Data, gradient boosting, isotonic calibration |
| [Flight Delay Operations Control](./flight_delay_control) | Bounded monthly aviation pipeline with safe ZIP ingestion, replay-proof contracts, calibrated pre-departure delay ranking, temporal evaluation and drift monitoring. | Python, Streamlit, BTS, histogram gradient boosting, isotonic calibration |
| [Kepler Candidate Reliability Control](./kepler_candidate_control) | Content-addressed NASA KOI pipeline with typed quarantine, star-group isolation, calibrated disposition modeling, permutation importance, drift and OOD withholding. | Python, Streamlit, NASA Exoplanet Archive TAP, gradient boosting, isotonic calibration |
| [Federal Procurement Entity Resolution Control](./federal_procurement_resolution) | Replay-safe USAspending contract pipeline with typed quarantine, UEI-keyed Gold products, evaluated character-level supplier matching, calibrated abstention and false-merge monitoring. | Python, Streamlit, USAspending, TF-IDF entity resolution, scikit-learn |
| [Wearable Activity Inference Gateway](./wearable_activity_gateway) | Safe nested-ZIP sensor pipeline with micro-batch replay, typed window contracts, subject-isolated multiclass inference, temperature scaling, drift and sensor/OOD withholding. | Python, Streamlit, UCI HAR, Extra Trees, calibrated selective inference |
| [Household Load Forecast Control](./household_load_forecast) | Chunked minute-power pipeline with event-time watermarks, replay-safe hourly contracts, leakage-safe day-ahead residual forecasting, calibrated intervals, drift and serving guardrails. | Python, Streamlit, UCI Power, gradient boosting, conformal forecasting |
| [Handwritten Digit Recognition Gateway](./digit_recognition_gateway) | Content-addressed image pipeline with replay suppression, typed pixel contracts, unseen-writer evaluation, temperature-scaled recognition, corruption testing and OOD withholding. | Python, Streamlit, UCI Optdigits, RBF SVM, calibrated selective inference |
| [MBTA Transit Graph Control](./mbta_transit_graph_control) | Bounded live-GTFS pipeline with relational contracts, parent-station graph normalization, replay suppression, masked-link recovery, ranking evaluation and confidence-based human review. | Python, Streamlit, MBTA GTFS, sparse graph embeddings, link prediction |

## Repository structure

```text
projects/
├── .streamlit/
│   └── config.toml
├── air_quality_intelligence/
├── biodiversity_observation_intelligence/
├── cyber_vulnerability_intelligence/
├── digit_recognition_gateway/
├── clinical_trial_ops_pipeline/
├── energy_transition_intelligence/
├── earthquake_intelligence/
├── eu_inflation_intelligence/
├── fda_recall_nlp_pipeline/
├── federal_procurement_resolution/
├── food_label_intelligence/
├── flight_delay_control/
├── fx_regime_intelligence/
├── household_load_forecast/
├── job-market-analytics-dashboard/
├── kepler_candidate_control/
├── london_bike_rebalancing_intelligence/
├── mbta_transit_graph_control/
├── message_trust_gateway/
├── movielens_recommendation_pipeline/
├── nyc_collision_intelligence/
├── nyc_311_resolution_pipeline/
├── open_source_health_intelligence/
├── orbital_launch_intelligence/
├── predictive_maintenance_pipeline/
├── portfolio_pages/
├── research_evidence_pipeline/
├── river_flow_early_warning/
├── sec_fundamentals_control/
├── storm_impact_pipeline/
├── stackoverflow_tag_router/
├── wikipedia_attention_pipeline/
├── wearable_activity_gateway/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Goal

The goal is to demonstrate practical Data Engineering, AI Engineering and data
science through useful, transparent mini-products: API ingestion, data contracts,
transformation, observability, retrieval and model evaluation, exploratory analysis,
resilient error handling, testing and cloud-ready delivery.
