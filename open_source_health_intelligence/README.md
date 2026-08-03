# Open Source Repository Health Intelligence

A deployment-ready Streamlit mini-product for exploring public GitHub repository
delivery flow, backlog aging, contributor concentration and release rhythm. The
application is integrated into the repository's central `streamlit_app.py` and
therefore becomes available automatically in the same hosted portfolio after a
commit to `main`.

## Problem

Stars and open-issue totals are easy to read but poor proxies for whether a
software project is delivering changes consistently, carrying an aging backlog,
depending heavily on a few contributors or publishing releases regularly. This
project turns bounded public activity into a transparent operational workbench.

It does **not** claim to measure maintainer wellbeing, code quality, security,
project governance or long-term sustainability. Those questions require richer
qualitative and full-history evidence.

## What the application does

- opens six curated repositories or a custom public `owner/repository`;
- shows repository metadata, license declaration, stars and forks;
- calculates a fully visible five-component Project Pulse heuristic;
- estimates censoring-aware issue and pull-request resolution curves;
- compares monthly opened, closed and merged activity in the bounded samples;
- audits sampled open-issue age bands;
- calculates contributor HHI, top-one/top-five shares and effective contributors;
- visualizes the GitHub language-byte mix;
- summarizes stable-release recency and median cadence;
- simulates pull-request backlog clearance under explicit weekly capacity;
- provides a sortable source-record table and CSV export;
- falls back to deterministic, clearly labelled synthetic data when GitHub is
  unavailable or the unauthenticated API quota has been exhausted.

## Data source

### Provider and access

Data comes from the [GitHub REST API](https://docs.github.com/en/rest). The app
uses public endpoints without authentication and never requests a token or
secret. The exact base request is:

```text
https://api.github.com/repos/{owner}/{repository}
```

It then requests:

| Endpoint | Fields used | Bound |
|---|---|---:|
| `/repos/{owner}/{repo}` | name, description, stars, forks, subscribers, license, default branch, created/pushed dates, archive status | one repository |
| [`/issues`](https://docs.github.com/en/rest/issues/issues#list-repository-issues) | number, title, state, dates, comments, labels, author, URL | 100 endpoint items |
| [`/pulls`](https://docs.github.com/en/rest/pulls/pulls#list-pull-requests) | number, title, state, created/updated/closed/merged dates, draft, author, URL | 100 PRs |
| `/contributors` | login/name, contribution count, profile URL | 100 contributors |
| `/releases` | tag, name, publication date, prerelease/draft flags, URL | 50 releases |
| `/commits` | SHA, author date/name, first message line, URL | 100 commits |
| `/languages` | language and byte total | all returned languages |

The issues endpoint also returns pull requests. The parser removes any item with
a `pull_request` field, so the final issue sample can contain fewer than 100
issues. Lists are intentionally sorted by recent update where supported. They
are operational windows, not full history.

### Retrieval and update cadence

GitHub serves the currently indexed public state when the API is requested. It
does not promise a fixed dataset release cadence. The Streamlit application
caches each repository for six hours to reduce quota consumption. A manual page
rerun after cache expiry obtains a new snapshot.

According to GitHub's
[REST rate-limit documentation](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api),
unauthenticated requests are normally limited to 60 per originating IP address
per hour. One new repository load makes seven calls. A hosted environment can
share an originating IP with other users; therefore the application includes a
reproducible fallback.

### Terms, licensing and responsible reuse

API use is governed by the [GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service).
Repository contents retain their own licenses. The repository metadata field
`license.spdx_id` is displayed but not treated as legal advice or proof that
every file has the same license. The app stores no token, does not scrape HTML,
does not collect email addresses and only processes fields returned by public
REST endpoints.

## Preprocessing and assumptions

1. A URL or `owner/repository` input is normalized and validated locally.
2. Private repositories are rejected.
3. Timestamps are parsed as timezone-aware UTC values.
4. Pull requests returned by the issues endpoint are removed from the issue table.
5. Open records end at retrieval time and are marked right-censored.
6. Closed issues use `closed_at`; PRs use `merged_at`, then `closed_at`, then
   retrieval time as their terminal/censoring timestamp.
7. Negative or invalid durations are excluded or clipped at zero.
8. Draft releases are removed; prereleases remain visible but stable-release
   cadence prefers non-prerelease rows.
9. Contributor metrics use GitHub's returned contribution count for at most the
   top 100 contributors. Identities are not deduplicated across renamed accounts.
10. Language shares use GitHub's byte classification, not file counts or runtime use.

## Analytical methods

### Kaplan–Meier unresolved-share curve

Computing a median only from closed work creates survivorship bias because open
work disappears. The Kaplan–Meier estimator keeps it as right-censored evidence:

```text
S(t) = product over event times ti ≤ t of (1 - di / ni)
```

`di` is the number resolved at event time `ti`; `ni` is the number still at risk
immediately before it. `S(t)` is shown as the estimated share unresolved after
`t` days. The median is the first day with `S(t) ≤ 0.5`. If the curve never
crosses 50%, the UI reports “Not reached”.

For PRs, the pulse's delivery component uses the median merge duration among
merged PRs, whereas the visible Kaplan–Meier curve treats merged or otherwise
closed PRs as terminal workflow events and preserves open PRs as censored.

### Contributor concentration

For contributor shares `si`:

```text
HHI = 10,000 × Σ(si²)
effective contributors = 1 / Σ(si²)
```

The effective count answers how many equally active contributors would create
the same concentration. It is a contribution-distribution proxy, not a “bus
factor”: it knows nothing about maintainership, review authority, employment,
knowledge ownership or future availability.

### Project Pulse heuristic

The score is deliberately transparent and is not an official GitHub metric:

| Component | Weight | Transformation |
|---|---:|---|
| Recent activity | 25% | exponential decay by days since latest push, 30-day half-life |
| PR delivery | 25% | exponential decay by sampled median merge time, 21-day half-life |
| Backlog freshness | 20% | share of sampled open issues no older than 90 days |
| Contributor spread | 15% | effective contributors divided by 15, capped at 100 |
| Release recency | 15% | exponential decay by days since stable release, 120-day half-life |

It is useful for inspecting trade-offs inside one repository snapshot. It is not
safe as an acquisition, investment, employment or security decision score.

### Capacity scenario

```text
net weekly reduction = weekly merge capacity - expected weekly arrivals
clearance weeks = sampled open PRs / net weekly reduction
```

If arrivals are at least capacity, the sampled backlog does not clear. This is a
deterministic sensitivity analysis, not a forecast of volunteer behavior.

## Architecture and code walkthrough

```text
open_source_health_intelligence/
├── app.py                         # standalone Streamlit entrypoint
├── ui.py                          # controls, states, charts, metrics and export
├── src/
│   ├── data.py                    # REST requests, parsers, validation, fallback
│   └── analytics.py               # survival, flow, HHI, pulse and scenario
└── tests/
    └── test_open_source_health_analytics.py
```

Important functions:

- `normalize_repository()` validates URL and shorthand input;
- `fetch_repository_data()` performs seven bounded public API calls;
- `_parse_issues()` and `_parse_pulls()` construct duration and censoring fields;
- `build_demo_data()` creates a seeded synthetic repository snapshot;
- `kaplan_meier()` implements the estimator without an extra heavy dependency;
- `contributor_concentration()` calculates HHI and effective contributors;
- `repository_pulse()` returns both the score and its audit table;
- `capacity_scenario()` keeps every planning assumption visible;
- `record_audit()` creates the downloadable operational evidence table;
- `render_dashboard()` isolates the entire Streamlit page from central navigation.

Heavy work happens after the user selects a page and repository. Calls are cached
for six hours. The central app imports a render function but does not fetch data
until that page is opened. A failure in the live API returns demo data instead of
breaking portfolio navigation.

## Live validation snapshot

On **3 August 2026 at 06:49 UTC**, a successful unauthenticated request for
`streamlit/streamlit` returned:

- 45,453 stars and 4,349 forks;
- Apache-2.0 as the declared repository license;
- 31 issues remaining after pull requests were removed from 100 issue-endpoint items;
- 100 pull requests, 100 contributors, 50 releases and 100 commits;
- a heuristic Project Pulse of 86.2/100;
- 14.4 effective contributors and a top-five contribution share of 53.1%;
- a censoring-aware sampled PR resolution median of about 4.1 days;
- a sampled stable-release median interval of about 13.3 days;
- 59 unauthenticated requests remaining after the first response.

These numbers are a dated API snapshot and will change. The pulse and samples
must not be interpreted as full-history or official project-health statistics.

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
streamlit run open_source_health_intelligence/app.py
```

No `.env`, GitHub token or paid credential is required.

## Testing

```bash
pytest -q
python -m compileall open_source_health_intelligence streamlit_app.py
```

The tests cover input validation, deterministic fallback schemas, censoring,
Kaplan–Meier behavior, HHI, age-band preservation, language shares, capacity
stability and score bounds.

## Limitations

- bounded recent-update samples can overrepresent active or recently closed work;
- the issues endpoint sample is shared with pull-request objects before filtering;
- pagination beyond the documented bounds is intentionally not performed;
- bots are not removed because naming conventions are incomplete;
- contribution counts do not equal review work, maintainership or knowledge;
- closed records can represent rejection, duplication or administrative cleanup;
- releases can follow irregular project-specific versioning conventions;
- repository license metadata can be missing or incomplete;
- GitHub API quota can be shared by users on a hosted IP;
- deleted, private and security-sensitive activity is unavailable.

## Possible extensions

- authenticated opt-in for higher quotas without storing tokens;
- GraphQL-based review and discussion latency;
- full-history extraction with checkpointed pagination;
- bot classification with a visible override;
- issue-label service-level objectives by project;
- contributor cohort retention and review-network analysis;
- release-note semantic clustering;
- security-policy, funding and governance document checks;
- multi-repository comparison with normalized observation windows.
