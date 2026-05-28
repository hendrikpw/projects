# Job Market Analytics Dashboard

A portfolio-oriented data analytics application that ingests live labor-market data from a public jobs API and visualizes hiring trends, locations, companies, remote opportunities, and requested skills.

## Project Goals

This project demonstrates practical data science and analytics engineering skills through a real-world dashboard application.

Key focus areas:

- API ingestion
- Data cleaning
- Exploratory data analysis
- Interactive dashboards
- Basic NLP skill extraction
- Modular Python architecture
- Deployment readiness

## Features

- Live public API integration
- Interactive Streamlit dashboard
- Company analytics
- Geographic job analytics
- Remote job analysis
- Keyword filtering
- Skill extraction from job descriptions
- Dataset exploration

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas |
| Visualization | Plotly |
| API Integration | Requests |
| Deployment | Docker |

## Project Structure

```text
job-market-analytics-dashboard/
├── src/
│   ├── data_loader.py
│   └── skill_extractor.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## API

Current API source:

- Arbeitnow Job Board API

## Run Locally

### Standard setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker setup

```bash
docker build -t job-dashboard .
docker run -p 8501:8501 job-dashboard
```

## Dashboard Components

The dashboard currently contains:

- KPI overview
- Location analytics
- Hiring company analytics
- Skill demand analysis
- Interactive filtering
- Raw dataset exploration

## Future Roadmap

### Data Engineering

- PostgreSQL integration
- Scheduled ETL pipelines
- Historical snapshot storage
- Airflow orchestration

### Machine Learning

- Salary prediction
- Demand forecasting
- Skill clustering
- Job classification
- Recommendation systems

### Visualization

- Geographic heatmaps
- Time-series trends
- Company comparison dashboards
- Sankey diagrams for skill transitions

## Portfolio Value

This project is intentionally structured as a realistic analytics application rather than a simple tutorial project.

It demonstrates:

- API integration
- Data wrangling
- Dashboard development
- Analytical thinking
- Modular project organization
- Deployment preparation
