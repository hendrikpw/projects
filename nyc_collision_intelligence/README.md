# NYC Collision Risk Intelligence

A browser-based geospatial safety analytics product built from daily
police-reported collision records. It finds spatial concentrations, detects
unusual daily volumes, compares temporal patterns and keeps every result
inspectable.

The project is integrated into the repository's central Streamlit portfolio, so
the preferred usage path is the hosted application rather than a local install.

## Problem

Raw crash records are difficult to interpret because they mix location, time,
outcomes, road users and provisional contributing factors. Counting collisions
alone also treats a property-damage event the same as a crash involving injuries
or death.

This product addresses five questions:

1. How many crashes and casualties occurred in the selected period?
2. Where are severe outcomes spatially concentrated?
3. Which dates are unusual relative to the recent local baseline?
4. How do weekday and weekend hourly patterns differ?
5. Which reported factors combine high volume with high observed severity?

## Product features

- latest 30, 60, 90 or 120 days;
- filters for all five boroughs and four outcome groups;
- volume, injuries, deaths, vulnerable-road-user casualties and injury rate;
- approximately one-kilometre spatial risk grid;
- interactive dark map with outcome-aware hotspot ranking;
- robust rolling median/MAD anomaly detection;
- borough volume and injury-rate comparison;
- weekday versus weekend hourly heatmap;
- reported-factor volume/severity bubble chart;
- explicitly non-causal sensitivity envelope;
- data-quality indicators, audit table and CSV export;
- reproducible synthetic fallback during an upstream outage.

The interface inherits the portfolio's Audi-inspired design system with
anthracite surfaces, restrained red signals, editorial typography, responsive
layouts and scroll-driven reveal motion.

## Exact data source

**Provider:** New York City Police Department (NYPD) through NYC Open Data  
**Dataset:** Motor Vehicle Collisions - Crashes  
**Dataset ID:** `h9gi-nx95`  
**Granularity:** one row per police-reported crash event  
**Update frequency:** daily  
**API:** Socrata Open Data API, JSON  
**Authentication:** no token required for the bounded public requests used here  
**Application query window:** latest 120 available days  
**Cache cadence:** six hours  
**Usage:** NYC Open Data states that its public data has no use restrictions;
the City's terms and source attribution still apply

Official resources:

- [Motor Vehicle Collisions - Crashes dataset](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
- [Direct Socrata API endpoint](https://data.cityofnewyork.us/resource/h9gi-nx95.json)
- [NYC Vision Zero Open Data page](https://www.nyc.gov/content/visionzero/pages/open-data)
- [NYC Open Data FAQ and usage statement](https://opendata.cityofnewyork.us/faq/)
- [NYC Open Data Law](https://opendata.cityofnewyork.us/open-data-law/)

The dataset describes police-reported collisions. The underlying MV104-AN
report is required when somebody is injured or killed, or when damage is at
least the applicable reporting threshold. It is therefore not a complete record
of every minor road incident.

## Fields used

| Source field | Use |
|---|---|
| `crash_date`, `crash_time` | timestamp, hour, weekday and analysis window |
| `borough`, `zip_code` | geographic filtering and aggregation |
| `latitude`, `longitude` | coordinate validation and risk grid |
| `on_street_name`, `cross_street_name`, `off_street_name` | representative hotspot label |
| `number_of_persons_injured`, `number_of_persons_killed` | primary outcomes |
| pedestrian/cyclist/motorist injured and killed fields | road-user outcomes |
| `contributing_factor_vehicle_1`, `_2` | reported-factor analysis |
| `vehicle_type_code1` | retained context field |
| `collision_id` | unique event count and stable ordering |

## Retrieval and preprocessing

`src/data.py` first queries `max(crash_date)` so the application follows the
dataset's newest available observation instead of assuming today's date is
already published. It then:

1. subtracts 119 days from the latest available date;
2. requests only the documented fields needed by the product;
3. paginates in 20,000-row blocks up to a 60,000-row safety limit;
4. converts timestamps and numeric outcome columns;
5. standardizes borough and street labels;
6. validates coordinates against broad NYC bounds;
7. derives weekday, hour, day type and outcome flags;
8. labels every row as live or demo.

If the API times out, returns an HTTP error or changes structure, a deterministic
synthetic dataset is generated with `numpy.random.default_rng(20260726)`. The UI
shows a prominent warning and never presents this fallback as observed NYPD
data.

## Analytical methods

### Spatial risk grid

Coordinates are rounded to two decimals, producing cells of roughly one
kilometre. For every cell:

```text
risk points =
    collisions
  + 2 × people injured
  + 25 × people killed
  + 4 × pedestrian/cyclist casualties
```

Points are min-max normalized to a 0-100 index inside the current selection.
This is a transparent prioritization measure, not an official Vision Zero
metric. It is not adjusted for traffic, population, trips or street length.

### Robust daily anomaly detection

Daily crash volume is compared with a rolling 14-day median. Dispersion is
estimated using the rolling median absolute deviation (MAD):

```text
robust z = (daily count - rolling median) / (1.4826 × rolling MAD)
```

Days with an absolute score of at least 3 are flagged. Median/MAD is less
sensitive to single extreme days than mean/standard deviation.

### Factor severity

`Unspecified`, `Unknown` and `Other Vehicular` are excluded. Remaining primary
factors are compared by:

- number of associated crashes;
- injuries per 100 associated crashes;
- deaths;
- serious-outcome rate, defined as death or at least two injuries.

Factors are reported by officers and can be incomplete or provisional.
Associations do not establish causality.

### Sensitivity envelope

The user can apply a hypothetical proportional reduction to cases associated
with one reported factor. The result is deliberately described as an exposure
envelope, not an estimate of preventable crashes.

## Architecture

```text
nyc_collision_intelligence/
├── app.py
├── ui.py
├── src/
│   ├── data.py
│   └── analytics.py
├── tests/
│   └── test_collision_analytics.py
└── README.md
```

Important functions:

- `_latest_available_date()` anchors the query to the source.
- `fetch_collisions()` performs paginated, field-bounded retrieval.
- `_prepare_data()` enforces the analytical schema.
- `build_demo_data()` creates the fixed fallback.
- `filter_collisions()` applies product controls.
- `spatial_hotspots()` calculates the mapped risk grid.
- `daily_anomalies()` implements rolling median/MAD alerts.
- `factor_profile()` compares volume with observed severity.
- `render_dashboard()` owns the complete user experience.

## Running and testing

The central hosted application is the recommended route. For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the project tests:

```bash
pytest nyc_collision_intelligence/tests -q
```

## Limitations

- Police reporting thresholds exclude some minor incidents.
- Recent records and contributing factors can be revised.
- Missing coordinates reduce spatial coverage.
- Grid rounding can split one intersection across neighboring cells.
- A high count can reflect greater traffic exposure rather than greater
  per-trip risk.
- The risk weights are explicit analytical assumptions.
- No result identifies causal road-design or behavioral effects.
- The row cap can truncate unusually large source windows and is surfaced in
  the interface.
- Map tiles depend on a separate public tile service even when collision data
  loaded successfully.

## Possible extensions

- normalize hotspots by traffic and pedestrian counts;
- snap records to official intersection geometries;
- compare before/after street redesigns with causal methods;
- incorporate weather and daylight conditions;
- model injury severity with calibrated probabilities;
- generate recurring alerts for newly persistent hotspots;
- add council district and community board boundaries.
