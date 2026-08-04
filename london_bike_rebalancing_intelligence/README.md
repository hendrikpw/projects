# London Cycle Rebalancing Intelligence

A live, deployment-ready Streamlit operations product for Transport for London's
Santander Cycles network. It translates a volatile station snapshot into service
levels, spatial pressure clusters and a transparent bike-transfer scenario. The
page is integrated into the repository's central `streamlit_app.py`.

## Problem

A cycle-hire network can have thousands of available bikes overall while still
failing locally: an empty station cannot serve a rider and a full station cannot
accept a return. A single system-wide average hides both problems. Operators need
to know where pressure is concentrated, which stations can donate bikes, which
stations need them and how much distance a rebalancing scenario creates.

This project is an analytical prioritization tool. It is not connected to TfL's
dispatch operation and does not issue real-world vehicle instructions.

## What the application does

- loads the complete current public TfL BikePoint response;
- maps available standard bikes, e-bikes, docks and station state;
- lets the user define the critical fill threshold and target fill level;
- identifies empty-risk, full-risk, balanced and unavailable stations;
- detects same-type spatial pressure clusters with haversine DBSCAN;
- compares station fill distribution with physical dock capacity;
- creates a bounded donor-to-receiver transfer scenario;
- exposes van capacity, maximum moves and maximum distance as controls;
- estimates critical stations resolved before and after the scenario;
- audits capacity arithmetic, bike-type totals, timestamps and station flags;
- exports the station snapshot and proposed move plan as CSV;
- uses clearly labelled deterministic synthetic data if the live API fails.

## Data source

### Provider and endpoint

The application uses Transport for London's public BikePoint endpoint:

```text
GET https://api.tfl.gov.uk/BikePoint
```

- Provider: [Transport for London](https://tfl.gov.uk/info-for/open-data-users/our-open-data)
- API documentation: [BikePoint GetAll](https://api.tfl.gov.uk/swagger/ui/index.html#!/BikePoint/BikePoint_GetAll)
- Live endpoint: [api.tfl.gov.uk/BikePoint](https://api.tfl.gov.uk/BikePoint)
- TfL transport-data terms: [Transport Data Service terms](https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service)
- UK public-sector reuse framework: [Open Government Licence 3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

No paid credential or user-supplied secret is required for this public bounded
request. Reuse remains subject to TfL's current terms, attribution requirements
and any notices attached to the service.

### Fields used

TfL returns station identity and coordinates directly, with operating values in
`additionalProperties`.

| App field | TfL field | Use |
|---|---|---|
| `station_id` | `id` | stable station identifier in the snapshot |
| `station_name` | `commonName` | map, table and plan labels |
| `latitude`, `longitude` | `lat`, `lon` | maps and great-circle distance |
| `terminal_name` | `TerminalName` | operational station reference |
| `bikes` | `NbBikes` | total currently reported bikes |
| `standard_bikes` | `NbStandardBikes` | standard-bike availability |
| `ebikes` | `NbEBikes` | electric-bike availability |
| `empty_docks` | `NbEmptyDocks` | reported return capacity |
| `docks` | `NbDocks` | reported station dock count |
| `installed` | `Installed` | operational eligibility |
| `locked` | `Locked` | operational eligibility |
| `temporary` | `Temporary` | audit context |
| `station_updated_at` | `modified` | freshest property timestamp per station |

### Freshness and retrieval

The source is an operational snapshot, not a historical dataset. TfL controls
the upstream refresh frequency and does not guarantee that every station property
changes at the same instant. The application takes the maximum `modified` value
per station, displays the overall timestamp range and caches a successful response
for 60 seconds. User activity can change the network immediately after retrieval.

### Live validation snapshot

At **06:44 UTC on 4 August 2026**, the live endpoint returned 798 valid stations:

- 7,484 available bikes, including 939 e-bikes;
- 11,860 reported empty docks;
- 249 empty-risk and 38 full-risk stations at a 15% threshold;
- 64.0% of operational stations in the balanced band;
- 35.7% network-wide bike fill;
- six DBSCAN pressure clusters using a 0.65 km radius and three-station minimum;
- 1,633 docks not classified as available bikes or empty docks;
- zero capacity-arithmetic, bike-type-total or timestamp inconsistencies.

The default 30-move scenario transferred 93 bikes over 3.41 km of straight-line
station-pair distance and reduced the current critical-station count by six. All
values are a dated snapshot and will change.

## Preprocessing and quality rules

1. `additionalProperties` is flattened by its documented `key` field.
2. numeric availability fields are parsed defensively; incomplete stations are removed;
3. duplicate station IDs keep the last returned record;
4. coordinates are constrained to a broad London bounding box;
5. stations with zero or missing dock capacity are removed;
6. negative counts are rejected through lower-bound clipping;
7. `unavailable_docks = docks - bikes - empty_docks`, clipped at zero;
8. `bikes + empty_docks > docks` is flagged as capacity-inconsistent;
9. `standard_bikes + ebikes != bikes` is flagged separately;
10. locked or explicitly not-installed stations cannot donate or receive bikes;
11. no missing value is silently converted into evidence of zero availability.

The inferred `unavailable_docks` field is intentionally not called “broken
docks”. TfL's three visible totals can differ for several operational reasons;
the API response does not explain each residual dock.

## Analytical methods

### Service classification

For station `i`:

```text
fill_i = bikes_i / docks_i
```

With critical threshold `c`:

- empty risk: `fill_i <= c`;
- full risk: `fill_i >= 1 - c`;
- balanced: between the two limits;
- unavailable: locked or explicitly not installed.

The default `c = 0.15` is an operational scenario assumption, not an official TfL
service-level target.

### Target deficit and surplus

```text
effective capacity_i = docks_i - unavailable_docks_i
desired bikes_i = round(target fill × effective capacity_i)
deficit_i = max(desired bikes_i - bikes_i, 0)
surplus_i = max(bikes_i - desired bikes_i, 0)
```

The target fill defaults to 50% and is user-adjustable. This deliberately treats
station balance as a capacity-planning objective rather than a demand forecast.

### Spatial pressure clusters

DBSCAN is fitted separately to empty-risk and full-risk stations. Coordinates are
converted to radians and distances use the haversine metric:

```text
distance = 2R × asin(sqrt(a)), where R = 6,371.0088 km
```

Separating the two risk types prevents nearby but operationally opposite stations
from being labelled as one homogeneous pressure cluster.

### Rebalancing scenario

1. eligible donors have positive surplus;
2. eligible receivers have positive deficit;
3. all donor-receiver great-circle distances are calculated;
4. pairs beyond the maximum distance are removed;
5. remaining pairs are processed nearest-first;
6. each move is capped by donor surplus, receiver deficit and van capacity;
7. the algorithm stops at the visible maximum move count.

The algorithm conserves the total number of bikes and never makes a station
negative. It is a greedy spatial heuristic, not a globally optimal vehicle-routing
solution. Straight-line distance is not road distance.

## Architecture and code walkthrough

```text
london_bike_rebalancing_intelligence/
├── app.py                         # standalone Streamlit entrypoint
├── ui.py                          # responsive dashboard, states and exports
├── src/
│   ├── data.py                    # TfL request, parser, QA and fallback
│   └── analytics.py               # service, clusters, plan and scenario impact
└── tests/
    └── test_bike_rebalancing_analytics.py
```

Important functions:

- `parse_bike_points()` flattens and validates the TfL schema;
- `fetch_live_data()` retrieves one network snapshot without credentials;
- `build_demo_data()` creates a seeded London-like fallback network;
- `add_service_features()` calculates fill, status, deficit and surplus;
- `network_metrics()` reconciles operational totals;
- `pressure_clusters()` runs same-type haversine DBSCAN;
- `build_rebalancing_plan()` creates and applies the bounded greedy plan;
- `scenario_summary()` compares critical stations before and after;
- `quality_report()` keeps feed assumptions auditable;
- `render_dashboard()` contains the complete isolated Streamlit page.

The central app imports only the render function. The TfL request is executed only
when the user opens this page, and a live-data failure returns demo data instead
of breaking the rest of the portfolio.

## User interface and states

The page uses the portfolio's shared Audi-inspired visual system: black and deep
graphite surfaces, white typography, signal-red operational risk, square controls,
responsive metric cards and restrained scroll-reveal motion. It includes:

- loading state during the network request;
- green live-data confirmation with freshness;
- prominent demo-data warning;
- empty states for filters, clusters and infeasible plans;
- explicit methodological warnings beside potentially misleading outputs;
- responsive maps, charts, tables and downloads.

## Running locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone page:

```bash
streamlit run london_bike_rebalancing_intelligence/app.py
```

No `.env`, TfL application key or paid credential is required.

## Testing

```bash
python -m pytest -q
python -m compileall london_bike_rebalancing_intelligence streamlit_app.py
```

Tests cover parsing, deterministic fallback data, capacity reconciliation,
service classification, same-type clustering, bike conservation, move-capacity
constraints, scenario impact and quality-report bounds.

## Limitations

- one snapshot cannot estimate demand, arrivals, departures or causal patterns;
- current availability can change seconds after retrieval;
- property timestamps within one response need not be identical;
- the critical threshold and target fill are user-defined assumptions;
- inferred unavailable docks have no observed reason code;
- great-circle distance ignores roads, bridges, traffic and depot constraints;
- the greedy matching order is not a global routing optimum;
- no staff, vehicle, shift, charging or maintenance constraints are available;
- e-bike battery state and charging requirements are absent;
- no forecast is made for the next commute period;
- the fallback is synthetic and never presented as current evidence.

## Possible extensions

- checkpointed snapshots and station-level demand forecasting;
- weather and event enrichment with leakage-safe feature timing;
- road-network routing and vehicle-routing optimization;
- multi-vehicle capacity and depot constraints;
- e-bike charging-state optimization when legitimate data becomes available;
- probabilistic shortage forecasts with calibrated uncertainty;
- service-level alerts by user-selected destination area;
- historical intervention evaluation and uplift measurement.
