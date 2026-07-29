# Orbital Launch Reliability Intelligence

A modern Streamlit product for exploring recent global launch operations,
comparing providers with uncertainty-aware reliability estimates and inspecting
upcoming mission schedules.

It is integrated into the repository's central `streamlit_app.py`, so an
existing Streamlit Community Cloud deployment updates from `main` automatically.

## What it does

- loads up to 500 recent and 50 upcoming launch records;
- compares cadence, outcomes and active launch providers;
- ranks providers with observed success rates and 95% Wilson intervals;
- prevents tiny samples from appearing falsely certain;
- measures provider concentration with HHI and effective-provider count;
- maps launch activity by geocoded pad;
- explores orbit and mission-type composition;
- provides a transparent future-record scenario simulator;
- presents an upcoming launch board and CSV export;
- falls back to deterministic, explicitly synthetic data on API failure.

## Data source

| Item | Detail |
|---|---|
| Provider | The Space Devs |
| Product | Launch Library 2 |
| Documentation | <https://ll.thespacedevs.com/docs/> |
| Endpoint | <https://ll.thespacedevs.com/2.3.0/launches/> |
| Product page | <https://thespacedevs.com/llapi> |
| Retrieval | Public REST API, JSON, no API key |
| Update cadence | Community-maintained operational data; records can update whenever schedules or outcomes change |
| Free-tier limit | 15 calls per hour according to the provider FAQ |

The app uses `id`, `name`, `net`, `last_updated`, status, launch service
provider, rocket configuration/family, mission type/orbit, pad/location/country,
coordinates, launch probability, failure reason and source URL. Requests are
cached for 12 hours and require at most six calls per refresh.

The Space Devs states that the database is accessible for free. API software
documentation identifies Apache License 2.0. Individual images have separate
licence metadata; this project intentionally does not retrieve or display those
media assets and uses only structured launch metadata. Users should consult the
provider documentation before redistributing data at scale.

## Pipeline

```text
Launch Library 2
      │
      ├── latest historical pages
      └── upcoming schedule
      │
      ▼
Nested JSON flattening
      │
      ▼
Schema, time, coordinate and outcome validation
      │
      ├── live records
      └── labelled deterministic fallback
      │
      ▼
Cadence · Wilson reliability · HHI · pad and mission aggregation
      │
      ▼
Streamlit controls · Plotly charts · audit table · CSV
```

## Analytical method

### Decided outcomes

Only `Launch Successful` and failure-labelled records enter reliability
calculations. Scheduled, uncertain and in-flight launches do not affect the
denominator.

### Wilson score interval

For `s` successes in `n` attempts and `z = 1.96`, the app calculates the Wilson
95% interval. Unlike the basic normal approximation, Wilson intervals remain
useful near 0% or 100% and with smaller samples. Providers are ordered by their
lower bound, which rewards both performance and evidence volume.

### Concentration

For provider launch shares `pᵢ`:

```text
HHI = Σ pᵢ²
effective providers = 1 / HHI
```

The effective-provider count is easier to interpret: activity split equally
between four providers yields approximately four; domination by one provider
moves the value toward one.

### Scenario simulator

Hypothetical successes and failures are added to one provider's observed record.
The app recomputes its success rate and Wilson interval. This is arithmetic
sensitivity analysis—not a forecast of future launches.

## Code guide

- `src/data.py`: paginated API requests, nested JSON parsing, validation,
  source metadata and deterministic fallback.
- `src/analytics.py`: filtering, Wilson intervals, provider ranking, cadence,
  HHI, pad/orbit aggregation and scenario calculations.
- `ui.py`: responsive controls, status states, KPI cards, charts, map,
  simulator, schedule table and export.
- `tests/test_launch_analytics.py`: parser, interval, reconciliation,
  concentration, scenario and fallback tests.

## Run

From the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Standalone:

```bash
streamlit run orbital_launch_intelligence/app.py
```

No secret or paid credential is required.

## Limitations

- The bounded recent sample is not the complete history of spaceflight.
- Reliability does not control for rocket generation, payload, orbit, mission
  difficulty or provider reporting differences.
- Launch schedules can change quickly; `NET` means No Earlier Than.
- Community-maintained records can be incomplete or revised.
- Wilson intervals cover sampling uncertainty, not all operational risk.
- The fallback is synthetic and must not be interpreted as observed activity.

## Extensions

- rocket-family and configuration-level reliability;
- rolling learning curves after vehicle introduction;
- launch-delay histories from captured snapshots;
- mission-complexity adjustment;
- provider and country market-share trends;
- notification links for upcoming missions.
