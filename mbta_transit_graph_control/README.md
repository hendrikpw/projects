# MBTA Transit Graph Control

> A relational GTFS data product and evaluated graph-AI gateway that can recover deliberately hidden stop–route relationships.

The project demonstrates that a public-transport network is not a flat CSV. Stops, routes, patterns and stop sequences form a versioned relational graph. Before an AI model can learn that graph, every identifier, foreign key, coordinate, delivery and transformation must be governed.

The Streamlit application combines:

- a secure, bounded GTFS ingestion pipeline;
- typed relational contracts and reasoned quarantine;
- parent-station normalization and replay-safe graph construction;
- content-addressed lineage and fail-closed publication;
- leakage-safe masked-link evaluation;
- hybrid graph embeddings, topology and geographic scoring;
- confidence-based review routing and controlled failure tests.

## Problem

Transit data products fail in ways that an attractive map does not reveal:

- a route pattern references a route that was not delivered;
- a stop sequence references an unknown stop;
- station platforms are counted as unrelated locations;
- the same stop–route edge arrives more than once;
- malformed coordinates enter downstream maps;
- model evaluation leaves the target relationship visible in training;
- a popular-route baseline is mistaken for learned network structure;
- a predicted relationship is presented as real service.

MBTA Transit Graph Control makes those risks visible and measurable.

## Product behavior

### Relational pipeline

The first workspace displays:

- current feed version and validity window;
- contracted nodes, routes, representative patterns and edges;
- stage volumes, rejections, runtimes and content hashes;
- ten publication gates;
- quarantine totals by source table and reason;
- spatial node coverage and graph-degree distributions;
- twenty intentionally replayed edge deliveries.

### Graph evaluation

For every station or stop served by at least two routes, one legitimate route relationship is deterministically hidden. Nodes are then assigned to disjoint calibration and test groups by stable hashes.

The model must rank the hidden route among all routes that are not still visible at that node. The application reports:

- Hit@1 and Hit@5;
- mean reciprocal rank;
- NDCG@5;
- performance against route popularity;
- per-route recovery behavior;
- degradation when additional visible graph edges disappear;
- a two-dimensional view of learned route embeddings.

### Link recovery lab

Select a real multi-route station, remove one authoritative relationship, and ask the model to recover it. The interface shows:

- the hidden route;
- recovered rank;
- top ten candidates and scores;
- confidence margin;
- `auto-candidate` or `human-review` route.

The hidden truth is used only to make evaluation observable. A candidate is never represented as authoritative service.

## Architecture

```text
Current MBTA GTFS ZIP
        │ retry · timeout · size · signature · traversal checks
        ▼
feed_info + stops + routes + route_patterns + chunked stop_times
        │
        ├── representative typical/canonical trips only
        ├── typed primary and foreign-key contracts
        └── reasoned relational quarantine
        ▼
Parent-station-normalized stop nodes
        │
        ├── stable stop–route edge IDs
        ├── intentional replay deliveries
        └── global idempotent suppression
        ▼
Gold bipartite transit graph + SHA-256 lineage
        │
        ├── hide one true edge per multi-route node
        ├── hash-disjoint calibration/test nodes
        └── masked graph fit
        ▼
Topology + Truncated SVD + geographic prior
        │
        ├── calibration-only factor/weight selection
        ├── untouched test ranking
        └── confidence and failure routing
```

## Data Engineering

### Safe extraction

`src/data.py` downloads the current feed with:

- an explicit portfolio User-Agent;
- a 45-second timeout;
- three attempts with exponential backoff;
- explicit transient HTTP handling;
- a compressed-size contract from 5 to 30 MB;
- ZIP-signature verification;
- rejection of absolute and parent-traversal paths;
- required-member contracts;
- a 10–150 MB expanded-size bound for the five consumed members.

Optional official GTFS files are not parsed and cannot influence the graph.

### Bounded parsing

`stop_times.txt` is the largest required relation. It is not loaded wholesale. The pipeline first selects typical or canonical route patterns and their `representative_trip_id` values, then scans stop times in 300,000-row chunks and retains only matching trips.

This keeps the hosted execution bounded while preserving a representative structural view of the published schedule.

### Source relations and fields

#### `feed_info.txt`

- `feed_id`
- `feed_version`
- `feed_start_date`
- `feed_end_date`

#### `stops.txt`

- `stop_id`, `stop_name`
- `stop_lat`, `stop_lon`
- `location_type`, `parent_station`
- `wheelchair_boarding`
- `municipality`

#### `routes.txt`

- `route_id`
- `route_short_name`, `route_long_name`
- `route_type`, `route_color`
- `network_id`

#### `route_patterns.txt`

- `route_pattern_id`, `route_id`
- `direction_id`, `route_pattern_name`
- `route_pattern_typicality`
- `canonical_route_pattern`
- `representative_trip_id`

#### `stop_times.txt`

- `trip_id`
- `stop_id`
- `stop_sequence`

### Data contracts

The pipeline enforces:

- unique, non-empty stop, route and pattern IDs;
- WGS84 coordinates inside a deliberately broad New England regional envelope;
- non-negative numeric GTFS route types;
- route-pattern foreign keys resolving to accepted routes;
- representative trips present on accepted patterns;
- stop-time trips resolving to retained representative trips;
- stop-time stops resolving to accepted stops;
- non-negative numeric stop sequences.

The first failing rule becomes a machine-readable quarantine reason. Rejected records never enter Gold.

### Station normalization

GTFS can describe a parent station and several boarding platforms. The graph maps a platform to `parent_station` when available; otherwise it keeps the original `stop_id`. Metadata comes from the parent row when present and from the child row otherwise.

This avoids treating the inbound and outbound platforms of the same station as unrelated network locations.

### Edge identity and idempotence

A graph edge is the unique combination of normalized node and route:

```text
edge_id = first_24_hex(SHA256(node_id + "|" + route_id))
```

Twenty edges are intentionally delivered twice. Global edge-ID deduplication removes the repeated deliveries even when delivery order differs. Input deliveries reconcile exactly to accepted Gold edges plus suppressed replays.

### Publication gates

Gold is published only when all ten controls pass:

1. sufficient source-relation volume;
2. unique node, route and pattern keys;
3. every Gold foreign key resolves;
4. stable edge IDs are unique;
5. all intentional replays are suppressed;
6. edge deliveries reconcile completely;
7. the graph meets minimum node, route and edge volume;
8. enough multi-route nodes exist for evaluation;
9. every published coordinate remains inside contract;
10. every published route has at least one representative node.

Any failure stops publication and produces a visible application error state.

### Lineage and observability

The run manifest records:

- source URL, byte size and archive SHA-256;
- feed version and validity dates;
- source and accepted relation counts;
- quarantine and replay totals;
- node, route, pattern and edge hashes;
- deterministic run ID;
- stage inputs, outputs, rejections and hashes;
- end-to-end pipeline duration.

Identical content and configuration reproduce the same graph hashes and run ID.

## AI Engineering

### Learning task

The Gold product is a binary matrix `A`:

```text
A[node, route] = 1 if the representative schedule links them
```

For each node with degree at least two, one true edge is selected by a deterministic SHA-256 ordering. The edge is removed before any similarity, embedding or calibration work.

### Leakage prevention

- Hidden relationships are absent from the fitted matrix.
- A node appears in either calibration or test, never both.
- Assignment uses stable node hashes rather than random runtime state.
- Factor count and score weights use calibration nodes only.
- The test set remains untouched until the final ranking.
- Route popularity is computed from the same masked matrix.

### Hybrid graph score

The candidate score blends three independently normalized signals:

1. **Topology:** cosine-normalized route co-occurrence from `AᵀA`.
2. **Embedding:** reconstruction from a deterministic `TruncatedSVD` route space.
3. **Geography:** distance between a node and the masked route centroid.

The calibration search compares 8, 16, 24 and 32 latent factors and five transparent weight combinations. It selects first by Hit@5 and then by MRR.

Already-visible routes are removed from every candidate ranking.

### Evaluation

For each hidden test edge:

- **Hit@1:** correct route ranked first;
- **Hit@5:** correct route among the first five;
- **MRR:** inverse position of the correct route;
- **NDCG@5:** rank-discounted recovery within the first five.

The baseline ranks routes only by their masked network popularity. Promotion requires Hit@5 above 25% and at least five percentage points above the baseline.

### Review routing

The score gap between the first and second candidate is the confidence margin. The 85th percentile of calibration margins becomes the automatic-candidate threshold. This is deliberately conservative: most cases remain available for human review instead of being silently linked.

The threshold is a workflow policy, not a calibrated probability.

### Failure evaluation

The robustness suite independently removes 0%, 10%, 20%, 30% and 40% of visible masked edges. It reports how Hit@1, Hit@5, MRR and NDCG change as graph context disappears.

This makes the failure direction measurable. It does not claim to represent every real feed outage.

## Data source and terms

| Item | Detail |
|---|---|
| Provider | Massachusetts Bay Transportation Authority (MBTA) |
| Current feed | [https://cdn.mbta.com/MBTA_GTFS.zip](https://cdn.mbta.com/MBTA_GTFS.zip) |
| MBTA field documentation | [GTFS implementation reference](https://github.com/mbta/gtfs-documentation/blob/master/reference/gtfs.md) |
| Standard | [Official GTFS Schedule Reference](https://gtfs.org/documentation/schedule/reference/) |
| Terms | [MBTA Developers License Agreement](https://github.com/mbta/gtfs-documentation/blob/master/developers-license-agreement.pdf) |
| Retrieval | Public HTTPS ZIP; no paid credential or user API key |
| Cadence | Current schedule feed; version and validity window change when MBTA publishes schedule revisions |

Use of the feed is governed by the MBTA developer agreement. This repository preserves attribution, links directly to the provider and does not imply MBTA endorsement.

The application reads scheduled GTFS, not GTFS-Realtime. It therefore does not know current vehicle positions, delays, cancellations, alerts or elevator state.

## Fallback behavior

If the entire live feed is unavailable or violates the archive contract, the app atomically switches to a deterministic demonstration network:

- 300 synthetic stops;
- 30 synthetic routes;
- overlapping representative corridors;
- fixed coordinates and random seed;
- the same relational, replay, model and failure pipeline.

The interface labels fallback mode and records the original failure. Live and synthetic records are never mixed.

## Important modules

| Module | Responsibility |
|---|---|
| `src/data.py` | Safe ZIP retrieval, bounded relational parsing and deterministic fallback |
| `src/pipeline.py` | Contracts, quarantine, normalization, graph construction, replay safety, quality and lineage |
| `src/model.py` | Masked-link splits, hybrid graph embeddings, evaluation, robustness and serving |
| `ui.py` | Pipeline control plane, spatial diagnostics, model evaluation and recovery lab |
| `tests/test_transit_graph.py` | Schema, security, idempotence, evaluation, failure and serving tests |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run only this project's tests:

```bash
pytest -q mbta_transit_graph_control/tests
```

No secret is required. `config.example.toml` documents non-sensitive defaults.

## Hosted use

The page is registered in the central portfolio entry point `streamlit_app.py`. An existing Streamlit Community Cloud deployment following `main` receives the project automatically.

If no deployment exists, the sole manual step is to select repository `hendrikpw/projects`, branch `main`, entry point `streamlit_app.py`, and click **Deploy**. A public live URL is not invented; once confirmed, it should be added here and to the root README.

## Operational considerations

- The current feed and fitted artifact are cached for six hours in the app process.
- Parsing and model selection are bounded for Community Cloud.
- Full production infrastructure would persist raw feed versions, tables and model artifacts in durable object storage.
- Historical feed comparisons should use the MBTA archive rather than overwriting the current snapshot.
- Production telemetry should track source age, quarantine rates, edge counts, route coverage, model margins and review outcomes.
- Human confirmation must precede any repair to an authoritative feed.

## Limitations

- Representative patterns approximate scheduled network structure; they do not include every trip variation.
- Parent-station normalization can hide platform-specific distinctions.
- Geographic centroids are a simple prior and can be misleading for long or branching routes.
- Masked-edge recovery measures internal graph consistency, not passenger usefulness.
- High ranking does not prove that a missing connection should exist.
- Schedule validity does not imply real-time service availability.
- Sparse or unique routes offer little collaborative graph evidence.
- Review precision can change with a new feed version and must be monitored.

## Extensions

- compare successive archived feeds and classify intentional versus suspicious edge changes;
- add GTFS-Realtime freshness and service-alert observability;
- replace route centroids with minimum distance to route shapes;
- evaluate inductive GraphSAGE or heterogeneous graph models;
- use conformal candidate sets for explicit empirical coverage;
- persist data and model lineage with OpenLineage and MLflow;
- expose a FastAPI review service with Pydantic contracts;
- capture reviewer decisions and evaluate precision without contaminating the benchmark.

## Skills demonstrated

Secure ZIP ingestion, bounded chunk processing, relational modeling, GTFS, data contracts, referential integrity, quarantine, parent-child normalization, idempotence, content hashes, lineage, observability, fail-closed publishing, sparse matrices, graph link prediction, embeddings, leakage prevention, calibration/test isolation, ranking evaluation, baselines, failure testing, human-review routing, geospatial visualization, Streamlit product design and automated testing.
