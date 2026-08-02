# Biodiversity Observation Intelligence

A deployment-ready Streamlit mini-product that compares European species
occurrence records while keeping taxonomy, sampling bias, quality, publisher and
licence provenance visible.

## Problem

Open biodiversity portals contain millions of records, but a dot on a map is not
a population estimate. Observation volume is shaped by people, platforms,
digitisation, reporting practices and dataset publication. This application
separates complete GBIF query aggregates from a bounded spatial audit sample and
uses both for their appropriate purpose.

## What the application does

- resolves scientific names against the GBIF taxonomic backbone;
- compares one to three European animal species from a documented preset list;
- retrieves complete indexed counts by year, month, country and evidence type;
- displays a bounded, provenance-rich georeferenced sample;
- discovers dense sample-record zones with haversine DBSCAN;
- measures pairwise occupied-grid overlap;
- audits coordinate uncertainty, event dates and GBIF interpretation issues;
- exposes dataset and occurrence-level licence composition;
- provides a filterable record explorer with direct GBIF links;
- exports publisher, dataset, licence and quality fields as CSV;
- falls back to clearly labelled deterministic synthetic records.

## Data sources

- **Provider:** Global Biodiversity Information Facility (GBIF)
- **Species API:** <https://techdocs.gbif.org/en/openapi/v1/species>
- **Occurrence API:** <https://techdocs.gbif.org/en/openapi/v1/occurrence>
- **API root:** <https://api.gbif.org/v1>
- **Terms:** <https://www.gbif.org/terms>
- **Citation guidance:** <https://www.gbif.org/citation-guidelines>
- **Authentication:** normal search requests require no account or API key
- **Update cadence:** depends on publishers and GBIF indexing; the application
  caches successful results for 12 hours

The query keeps records that:

- match the resolved GBIF taxon key;
- have `occurrenceStatus=PRESENT`;
- contain interpreted coordinates;
- are indexed as European;
- have a year from 2018 through 2026.

For each species, GBIF facet counts cover the complete matching query. The map
and quality audit retain at most 600 API records: up to 300 from the initial page
and up to 300 from a second bounded offset. This is a reproducible operational
sample, not a random sample.

### Fields used

| GBIF / Darwin Core field | App field | Use |
|---|---|---|
| `key`, `gbifID` | record identifiers | deduplication and direct record link |
| `taxonKey`, `scientificName`, `species` | taxonomy | resolution and record audit |
| `decimalLatitude`, `decimalLongitude` | coordinates | map, grid and DBSCAN |
| `eventDate`, `year`, `month` | time | coverage audit and temporal facets |
| `country`, `countryCode`, `locality` | geography | comparison and record context |
| `basisOfRecord` | evidence type | human observation, machine record, specimen, etc. |
| `coordinateUncertaintyInMeters` | uncertainty | spatial-quality assessment |
| `issues` | GBIF flags | interpreted record-quality audit |
| `datasetKey`, `datasetTitle` | source dataset | provenance and concentration |
| `publishingOrgKey` | publisher | export provenance |
| `license` | record licence | reuse conditions |

## Licences and citation

GBIF occurrence datasets use one of three standard machine-readable choices:
CC0, CC BY or CC BY-NC. Their conditions differ. The application therefore keeps
the licence URL and normalized licence category for each sampled record and does
not use occurrence photographs or media.

Search-API results do not receive one downloadable dataset DOI automatically.
GBIF recommends identifying and acknowledging the contributing publishers and,
for formal research outputs, creating a derived dataset or authenticated download
with a DOI. The in-app export preserves dataset keys to support that process.

## End-to-end pipeline

1. `resolve_species()` checks each supplied scientific name against GBIF's
   taxonomic backbone and records confidence, status and accepted name.
2. `fetch_species_occurrences()` runs the bounded occurrence request and requests
   complete facets for year, month, country and basis of record.
3. `parse_occurrences()` flattens selected Darwin Core and GBIF interpretation
   fields, validates coordinates and removes duplicate record IDs.
4. `add_quality_features()` creates visible, component-based record-quality fields.
5. Full-query facets drive reporting-pattern charts; the bounded sample drives
   maps, cluster, overlap, source and record-level audits.
6. `ui.py` renders responsive controls, states, visualizations and export.

## Analytical methods

### Record-quality score

The visible 0–100 audit score adds:

```text
30 points: interpretable event date
20 points: coordinate uncertainty supplied
25 points: supplied uncertainty ≤ 10 km
25 points: no GBIF interpretation issue flags
```

It is an application-specific completeness heuristic, not an official GBIF score
and not proof that the identification is correct.

### Spatial clusters

DBSCAN receives coordinates in radians and uses haversine great-circle distance.
Users control radius and minimum nearby records. `-1` represents records outside
dense sample zones. Clusters show collection concentration only.

### Grid overlap

Coordinates are assigned to one-degree cells. For each species pair:

```text
Jaccard overlap = shared occupied cells / union of occupied cells
```

Because it uses the bounded sample, it is not formal range overlap.

### Full-query reporting profiles

Year, month, country and basis charts use GBIF's facet counts over the complete
matching query, not just mapped rows. They describe indexed reporting volume.
Changes cannot be interpreted as population trends without designed sampling,
effort correction and ecological context.

## Architecture

```text
biodiversity_observation_intelligence/
├── app.py
├── ui.py
├── src/
│   ├── data.py
│   └── analytics.py
├── tests/
│   └── test_biodiversity_analytics.py
└── README.md
```

Important functions:

- `resolve_species()` — taxonomic resolution;
- `fetch_species_occurrences()` — sample and complete facets;
- `parse_occurrences()` — schema and coordinate validation;
- `build_demo_data()` — reproducible fallback;
- `add_quality_features()` — quality components;
- `spatial_clusters()` — haversine DBSCAN;
- `grid_overlap()` — sample-cell Jaccard comparison;
- `facet_table()` — full-query shares;
- `render_dashboard()` — complete interaction flow.

## Run and validate

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
pytest -q
```

Standalone entrypoint:

```bash
streamlit run biodiversity_observation_intelligence/app.py
```

No secret, GBIF account or local data download is required.

## Limitations

- Occurrence records are evidence of reporting, not population abundance.
- Search API pages are bounded and not random samples.
- Species detectability and observer effort differ dramatically.
- Coordinates may be generalized, uncertain, duplicated or sensitive.
- GBIF issue-free does not prove taxonomic identification.
- Monthly patterns mix phenology with observer behavior.
- Dataset updates and taxonomic interpretation can change results.
- CC BY-NC records require non-commercial use and attribution.
- Synthetic fallback records have no scientific meaning.

## Possible extensions

- authenticated DOI-bearing GBIF downloads for publishable analyses;
- effort-aware occupancy models from sampling-event datasets;
- protected-area and land-cover overlays;
- uncertainty-aware spatial smoothing;
- checklist completeness and taxonomic-change monitoring;
- user-uploaded monitoring routes compared locally with public observations.
