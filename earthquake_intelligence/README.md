# Global Seismic Activity Intelligence

A responsive, browser-based data science product for exploring the recent
global earthquake catalog. It combines near-real-time USGS data with physical
feature engineering, robust temporal diagnostics, magnitude-frequency analysis
and density-based spatial clustering.

The project is integrated into the repository's central Streamlit application,
so it can be used online without downloading the repository.

## Problem

Earthquake catalogs contain thousands of events with logarithmic magnitudes,
uncertain locations, changing review states and very different focal depths.
A simple map hides several important questions:

1. Where is recent activity spatially concentrated?
2. How much estimated seismic energy was released?
3. Are any days unusual relative to the local activity baseline?
4. Does the selected magnitude distribution follow the expected
   Gutenberg-Richter shape?
5. Which reported regions and spatial sequences dominate the current window?

The app answers these questions without claiming to forecast earthquakes or
identify geological faults.

## Product features

- one-, three-, seven-, 14- and 30-day catalog windows;
- magnitude, depth, review-status and tsunami-flag controls;
- event count, maximum magnitude, estimated energy and reviewed share;
- interactive global epicenter map;
- haversine DBSCAN spatial sequence discovery;
- magnitude-versus-depth exploration;
- daily event and energy timeline with robust anomaly flags;
- cumulative magnitude-frequency curve on a logarithmic scale;
- Gutenberg-Richter b-value estimation;
- ranked sequence, region and high-magnitude event tables;
- CSV export;
- deterministic, clearly labelled fallback data.

The visual layer inherits the shared Audi-inspired portfolio design: deep
anthracite, restrained red signals, editorial typography, responsive layouts
and scroll-driven entry motion.

## Exact data source

**Provider:** U.S. Geological Survey (USGS), Earthquake Hazards Program  
**Catalog:** ANSS Comprehensive Earthquake Catalog (ComCat)  
**Interface:** FDSN Event Web Service  
**Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query`  
**Format:** GeoJSON  
**Authentication:** none  
**Source request:** latest 30 days, earthquakes with magnitude 2.5 or greater  
**Maximum response:** 20,000 events  
**Application cache:** 15 minutes  
**Update cadence:** near real time; event solutions can be revised as networks
review them  
**Usage:** USGS-authored data and information are generally U.S. Public Domain;
USGS attribution is retained

Official resources:

- [USGS FDSN Event API documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
- [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/)
- [ANSS Comprehensive Earthquake Catalog](https://earthquake.usgs.gov/data/comcat/)
- [USGS real-time feeds and web services](https://earthquake.usgs.gov/earthquakes/feed/)
- [USGS copyright and credit policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)
- [USGS data licensing guidance](https://www.usgs.gov/data-management/data-licensing)

## Fields used

| GeoJSON field | Analytical use |
|---|---|
| feature `id` | stable event key |
| `properties.time`, `updated` | event time, daily aggregation and revision time |
| `properties.mag` | filtering, energy and magnitude-frequency analysis |
| `properties.place` | event label and reported-region summary |
| `properties.sig` | USGS significance context and display priority |
| `properties.felt`, `cdi`, `mmi` | felt-response and intensity context |
| `properties.alert` | PAGER alert context when present |
| `properties.status` | automatic versus reviewed filter |
| `properties.tsunami` | tsunami workflow flag |
| `properties.url` | direct official event page |
| geometry longitude, latitude | epicenter map and haversine DBSCAN |
| geometry depth | focal-depth filtering and classification |

## Retrieval and preprocessing

`src/data.py` creates one bounded query for the latest 30 days. The service is
asked for `eventtype=earthquake`, `minmagnitude=2.5`, GeoJSON output and newest
events first.

The parser then:

1. extracts feature properties and geometry;
2. converts epoch milliseconds to timezone-aware UTC timestamps;
3. validates latitude, longitude, depth and magnitude;
4. derives calendar day, UTC hour and reported region;
5. classifies focal depth as shallow, intermediate or deep;
6. creates review and tsunami flags;
7. calculates estimated seismic energy;
8. records whether observations are live or demo data.

If the API times out or changes structure, `build_demo_data()` uses
`numpy.random.default_rng(20260727)` to generate a fixed synthetic catalog
around eight tectonically plausible centers. The interface shows an explicit
warning and never calls those events observed USGS earthquakes.

## Analytical methods

### Seismic energy

Magnitude is logarithmic. Estimated radiated energy is calculated as:

```text
log10(E joules) = 1.5 × magnitude + 4.8
```

This approximation makes the dominance of large events visible. It is not a
damage estimate: impact also depends on depth, distance, ground conditions,
building vulnerability and population exposure.

### Spatial sequence discovery

Latitude and longitude are transformed to radians. DBSCAN uses haversine
distance on the sphere:

```text
epsilon radians = selected radius in km / 6,371.0088 km
```

The user chooses a radius between 50 and 800 km and a minimum of three to 12
events. Noise points remain visible as `Unclustered`. Clusters are deliberately
called sequences rather than faults or aftershock groups.

### Gutenberg-Richter b-value

For events at or above the user-selected completeness assumption `Mc`, the
Aki maximum-likelihood estimate is:

```text
b = log10(e) / (mean magnitude - (Mc - bin width / 2))
```

The bin width is 0.1. The metric is omitted when fewer than 20 events remain.
Because the user-selected threshold is only an assumption, the product exposes
it directly and does not present the value as a definitive regional parameter.

### Robust activity anomalies

Daily counts are compared with a rolling seven-day median. Dispersion uses
median absolute deviation:

```text
robust z = (events - rolling median) / (1.4826 × rolling MAD)
```

An absolute score of at least three is marked. This detects unusual catalog
activity but does not predict a future earthquake.

## Architecture

```text
earthquake_intelligence/
├── app.py
├── ui.py
├── src/
│   ├── data.py
│   └── analytics.py
├── tests/
│   └── test_seismic_analytics.py
└── README.md
```

Important functions:

- `fetch_earthquakes()` performs the bounded USGS request.
- `_prepare_frame()` validates and enriches the GeoJSON records.
- `build_demo_data()` provides the deterministic outage fallback.
- `filter_events()` applies all interactive controls.
- `cluster_events()` runs haversine DBSCAN.
- `gutenberg_richter_b_value()` estimates the catalog slope.
- `daily_activity()` calculates counts, energy and anomalies.
- `magnitude_frequency()` builds the cumulative frequency curve.
- `render_dashboard()` owns the complete product experience.

## Running and testing

The central hosted app is the preferred route. For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the project tests:

```bash
pytest earthquake_intelligence/tests -q
```

## Responsible interpretation and limitations

- The app is not an earthquake prediction or emergency-warning system.
- Event locations, depths and magnitudes can change during review.
- Global network sensitivity is spatially uneven.
- A selected magnitude threshold is not automatically the true completeness
  magnitude for every region.
- DBSCAN sequences depend on the chosen radius and minimum count.
- Spatial proximity does not prove that events share one fault or mechanism.
- Tsunami flags are workflow indicators, not local evacuation advice.
- Seismic energy is not equivalent to damage or risk.
- Map tiles are loaded from a separate public service.

Always use official national and local authorities for emergency information.

## Possible extensions

- estimate regional completeness magnitude automatically;
- add tectonic plate boundaries and named subduction zones;
- distinguish likely mainshock-aftershock sequences with temporal rules;
- incorporate moment tensors and focal mechanisms;
- compare catalog providers and location uncertainty;
- add exposure layers while keeping hazard and risk conceptually separate;
- create official-alert links based on the user's region.
