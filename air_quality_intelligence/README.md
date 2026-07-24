# European Air Quality Intelligence

A polished, live Streamlit data product for comparing seven-day air-quality
forecasts across major European cities.

## What the application does

The dashboard answers four practical questions:

1. Which selected city has the cleanest overall outlook?
2. When does the European Air Quality Index peak?
3. How many forecast hours exceed an AQI of 60 ("poor")?
4. Which pollutant concentration most often dominates the selected period?

Users can compare up to six cities, change the forecast window, inspect hourly
and daily patterns, review a city-level table and export the filtered data.

## Data source and provenance

| Item | Detail |
|---|---|
| API | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) |
| Upstream model | Copernicus Atmosphere Monitoring Service (CAMS) |
| European resolution | 0.1° / approximately 11 km |
| Temporal grain | Hourly |
| Forecast horizon | Up to 7 days through the API |
| CAMS Europe update | Every 24 hours according to the Open-Meteo source table |
| Authentication | No API key for non-commercial free-tier use |
| Licence | CC BY 4.0 for Open-Meteo API data; attribution required |

The application requests `european_aqi`, `pm2_5`, `pm10`,
`nitrogen_dioxide` and `ozone`. According to Open-Meteo, the consolidated
European AQI is the maximum of the pollutant-specific indices. PM indices use
rolling 24-hour values while the gas indices use hourly values.

Attribution: Open-Meteo and CAMS ENSEMBLE data providers.

Terms: https://open-meteo.com/en/terms

## How it works

```text
City selection
    ↓
Open-Meteo REST request
    ↓
Tidy hourly Pandas frame
    ↓
AQI categories + pollutant diagnostics
    ↓
KPI cards, Plotly charts, comparison table and CSV export
```

### Important modules

- `src/data.py` defines the city catalogue, fetches the API and supplies a
  deterministic, visibly labelled synthetic fallback if the endpoint fails.
- `src/analytics.py` validates the schema and creates AQI bands, daily profiles,
  city summaries and dominant-concentration diagnostics.
- `ui.py` contains the Streamlit presentation layer, filters, states, charts and export flow.
- `app.py` is the standalone Streamlit entrypoint.
- `tests/test_analytics.py` checks AQI boundaries and the core transformations.

The dominant-pollutant view normalizes concentrations against the upper boundary
of the European AQI moderate band (PM2.5 25, PM10 50, NO₂ 120 and O₃ 130 µg/m³).
It is an exploratory diagnostic—not a recalculation of the official AQI.

## Run locally

```bash
pip install -r air_quality_intelligence/requirements.txt
streamlit run air_quality_intelligence/app.py
```

The project is also integrated into the repository's central `streamlit_app.py`.

## Streamlit Community Cloud

Use the repository root entrypoint `streamlit_app.py`. Once the repository is
connected at https://share.streamlit.io, every commit to `main` automatically
updates the persistent portfolio app.

## Design and interaction

- responsive dark interface with accessible high-contrast accents
- meaningful default selection centred on Stuttgart
- loading, empty, API-error/fallback and filtered-empty states
- KPI-first information hierarchy and interactive Plotly charts
- bounded city comparison and downloadable filtered CSV

## Limitations

- This is modelled grid data, not a street-level monitoring station.
- CAMS Europe has an approximately 11 km grid, so neighbourhood variation is not represented.
- Forecast accuracy and availability depend on upstream providers.
- Synthetic fallback values only demonstrate the interface and are never presented as observations.
- The application is for exploratory analysis, not medical or regulatory advice.

## Portfolio skills demonstrated

API ingestion, resilient data pipelines, tidy transformations, metric design,
environmental analytics, interactive visualization, responsive Streamlit UX,
data provenance, testing and deployment-ready packaging.
