"""Streamlit interface for Open Source Repository Health Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from open_source_health_intelligence.src.analytics import (
    capacity_scenario,
    contributor_concentration,
    issue_age_bands,
    kaplan_meier,
    km_percentile,
    language_mix,
    monthly_flow,
    record_audit,
    release_cadence,
    repository_pulse,
)
from open_source_health_intelligence.src.data import (
    ISSUES_DOCS_URL,
    PRESETS,
    PULLS_DOCS_URL,
    RATE_LIMIT_URL,
    REST_DOCS_URL,
    TERMS_URL,
    load_data,
    normalize_repository,
)


PALETTE = ["#e5484d", "#fcfcfd", "#9da2aa", "#676c75", "#3d424a", "#24282e"]


def _style_figure(figure: go.Figure, height: int = 470) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=64, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="#171a20", font_color="#fcfcfd", bordercolor="#5c6068"),
        colorway=PALETTE,
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=21_600, show_spinner=False)
def _cached_repository(repository: str) -> tuple[dict[str, pd.DataFrame], dict]:
    return load_data(repository)


def _number(value: int | float) -> str:
    return f"{value:,.0f}"


def render_dashboard() -> None:
    """Render the complete repository operations and sustainability workbench."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">12 / Software delivery intelligence</div>
          <h1>Open Source<br>Repository Health.</h1>
          <p>
            Turn public GitHub activity into an auditable view of delivery flow,
            backlog aging, contributor concentration and release rhythm.
          </p>
          <div class="source-line">GitHub REST API · Kaplan–Meier · Flow analytics</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.7, 1.3])
    with c1:
        preset = st.selectbox("Repository preset", [*PRESETS, "Custom repository"], format_func=lambda value: PRESETS.get(value, value))
    with c2:
        custom = st.text_input(
            "Public repository",
            value="streamlit/streamlit" if preset != "Custom repository" else "",
            placeholder="owner/repository or https://github.com/owner/repository",
            disabled=preset != "Custom repository",
        )
    candidate = preset if preset != "Custom repository" else custom
    if not candidate:
        st.info("Enter a public GitHub repository to open the workbench.")
        return
    try:
        repository = normalize_repository(candidate)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.spinner("Loading bounded public GitHub activity…"):
        frames, metadata = _cached_repository(repository)
    issues, pulls = frames["issues"], frames["pulls"]
    contributors, releases = frames["contributors"], frames["releases"]
    commits, languages = frames["commits"], frames["languages"]
    if metadata["mode"] == "demo":
        st.warning(
            "GitHub live data is unavailable or rate-limited. All visible records are a "
            "deterministic synthetic fallback and must not be interpreted as repository evidence."
        )
    else:
        st.success(
            f"Live GitHub data · {metadata['repository']} · bounded samples of "
            f"{len(issues)} issues, {len(pulls)} pull requests and {len(commits)} commits"
        )

    pulse, components = repository_pulse(frames, metadata)
    concentration = contributor_concentration(contributors)
    issue_curve = kaplan_meier(issues)
    pull_curve = kaplan_meier(pulls)
    issue_median = km_percentile(issue_curve)
    pull_median = km_percentile(pull_curve)
    open_issue_count = int(issues["event_observed"].eq(0).sum()) if not issues.empty else 0

    st.markdown(f"### {metadata['repository']}")
    st.caption(metadata["description"])
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Heuristic pulse", f"{pulse:.0f} / 100")
    k2.metric("Stars", _number(metadata["stars"]))
    k3.metric("Sampled open issues", open_issue_count)
    k4.metric("PR resolution median", "Not reached" if pull_median is None else f"{pull_median:.1f} days")
    k5.metric("Effective contributors", f"{concentration['effective_contributors']:.1f}")
    st.caption(
        "The pulse is a transparent portfolio heuristic, not an official GitHub rating. "
        "GitHub list endpoints are bounded and sorted by recent update, so values describe "
        "the retrieved activity window—not complete repository history."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Repository pulse</div>
          <h2>Make the score<br>inspectable.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.7, 1.3])
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pulse,
                number={"suffix": "/100", "font": {"color": "#fcfcfd", "size": 42}},
                title={"text": "Transparent project pulse", "font": {"color": "rgba(252,252,253,.65)", "size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "rgba(252,252,253,.35)"},
                    "bar": {"color": "#e5484d"},
                    "bgcolor": "rgba(252,252,253,.04)",
                    "bordercolor": "rgba(252,252,253,.12)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(252,252,253,.03)"},
                        {"range": [40, 70], "color": "rgba(252,252,253,.06)"},
                        {"range": [70, 100], "color": "rgba(229,72,77,.08)"},
                    ],
                },
            )
        )
        st.plotly_chart(_style_figure(gauge, 430), width="stretch")
    with right:
        component_chart = px.bar(
            components.sort_values("score"),
            x="score",
            y="component",
            orientation="h",
            text=components.sort_values("score")["score"].map(lambda value: f"{value:.0f}"),
            hover_data={"weight": True, "evidence": True, "weighted_points": ":.1f"},
            title="Five visible score components",
            labels={"score": "Component score · 0–100", "component": ""},
            color="component",
            color_discrete_sequence=PALETTE,
        )
        component_chart.update_layout(showlegend=False)
        st.plotly_chart(_style_figure(component_chart, 430), width="stretch")
    with st.expander("Pulse formula and evidence"):
        audit = components[["component", "score", "weight", "weighted_points", "evidence"]].copy()
        audit.columns = ["Component", "Score", "Weight · %", "Weighted points", "Evidence"]
        st.dataframe(audit.style.format({"Score": "{:.1f}", "Weighted points": "{:.1f}"}), width="stretch", hide_index=True)
        st.markdown(
            "**Weights:** recent activity 25%, PR delivery 25%, backlog freshness 20%, "
            "contributor spread 15%, release recency 15%. Recency and cycle-time components "
            "use exponential decay; contributor spread uses the effective contributor count."
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Delivery flow</div>
          <h2>Measure resolution.<br>Keep open work visible.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    curves = []
    if not issue_curve.empty:
        frame = issue_curve.copy(); frame["work_type"] = "Issues"; curves.append(frame)
    if not pull_curve.empty:
        frame = pull_curve.copy(); frame["work_type"] = "Pull requests"; curves.append(frame)
    flow_col, curve_col = st.columns([0.95, 1.05])
    with flow_col:
        flow = monthly_flow(issues, pulls, 12)
        if flow.empty:
            st.info("No dated issue or pull-request activity exists in the bounded sample.")
        else:
            flow_chart = px.bar(
                flow,
                x="month",
                y="count",
                color="metric",
                barmode="group",
                title="Recent sampled delivery flow",
                labels={"month": "", "count": "Records", "metric": ""},
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(_style_figure(flow_chart, 520), width="stretch")
    with curve_col:
        if not curves:
            st.info("Not enough duration evidence for a resolution curve.")
        else:
            curve_data = pd.concat(curves, ignore_index=True)
            curve_chart = px.line(
                curve_data,
                x="day",
                y="unresolved_share",
                color="work_type",
                line_shape="hv",
                title="Kaplan–Meier estimated unresolved share",
                labels={"day": "Days since creation", "unresolved_share": "Estimated unresolved · %", "work_type": ""},
                color_discrete_sequence=PALETTE,
            )
            curve_chart.update_yaxes(range=[0, 102])
            st.plotly_chart(_style_figure(curve_chart, 520), width="stretch")
    st.caption(
        "Kaplan–Meier retains still-open records as right-censored observations instead of "
        "pretending that only completed work exists. Closing and merging are workflow events, "
        "not measures of code quality."
    )

    bands = issue_age_bands(issues)
    age_col, capacity_col = st.columns([1.05, 0.95])
    with age_col:
        if bands.empty:
            st.info("No open issues are present in the bounded sample.")
        else:
            age_chart = px.bar(
                bands,
                x="age_band",
                y="count",
                text="count",
                title="Sampled open-issue age",
                labels={"age_band": "", "count": "Open issues"},
                color="age_band",
                color_discrete_sequence=PALETTE,
            )
            age_chart.update_layout(showlegend=False)
            st.plotly_chart(_style_figure(age_chart, 440), width="stretch")
    with capacity_col:
        st.markdown("#### Pull-request capacity scenario")
        recent_merged = pulls[pulls["merged_at"].notna() & pulls["merged_at"].ge(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90))] if not pulls.empty else pulls
        observed_weekly = len(recent_merged) / 13
        weekly_capacity = st.slider("Weekly merge capacity", 0, 30, max(1, min(30, round(observed_weekly))))
        weekly_arrivals = st.slider("Expected new PRs per week", 0.0, 30.0, float(min(30, round(observed_weekly, 1))), 0.5)
        scenario = capacity_scenario(pulls, weekly_capacity, weekly_arrivals)
        s1, s2 = st.columns(2)
        s1.metric("Sampled open PRs", scenario["open_prs"])
        s2.metric("Net change / week", f"{scenario['net_weekly_reduction']:+.1f}")
        if scenario["clearance_weeks"] is None:
            st.warning(scenario["status"])
        else:
            st.success(f"{scenario['status']} · {scenario['clearance_weeks']:.1f} weeks")
        st.caption("A deterministic capacity scenario on the sampled backlog—not a forecast of contributor behavior.")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Community structure</div>
          <h2>See concentration.<br>Read release rhythm.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    community_col, language_col = st.columns([1.1, 0.9])
    with community_col:
        if contributors.empty:
            st.info("GitHub returned no contributor sample.")
        else:
            top = contributors.head(20).copy()
            top["share"] = top["contributions"].div(contributors["contributions"].sum()).mul(100)
            contributor_chart = px.bar(
                top.sort_values("share"),
                x="share",
                y="contributor",
                orientation="h",
                title="Top contributor share in API sample",
                labels={"share": "Share of sampled contributions · %", "contributor": ""},
                color="share",
                color_continuous_scale=["#31353c", "#e5484d"],
            )
            contributor_chart.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_style_figure(contributor_chart, 560), width="stretch")
    with language_col:
        mix = language_mix(languages)
        if mix.empty:
            st.info("No language-byte breakdown is available.")
        else:
            visible = mix.head(8).copy()
            if len(mix) > 8:
                visible = pd.concat([visible, pd.DataFrame([{"language": "Other", "bytes": mix.iloc[8:]["bytes"].sum(), "share": mix.iloc[8:]["share"].sum()}])])
            language_chart = px.pie(
                visible,
                names="language",
                values="bytes",
                hole=0.68,
                title="Repository language mix",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(_style_figure(language_chart, 390), width="stretch")
        cadence = release_cadence(releases)
        r1, r2 = st.columns(2)
        r1.metric("Stable releases sampled", cadence["releases"])
        r2.metric("Median release interval", "N/A" if cadence["median_interval_days"] is None else f"{cadence['median_interval_days']:.0f} days")
        st.metric("Top-five contribution share", f"{concentration['top5_share']:.1f}%")
        st.caption(
            f"HHI: {concentration['hhi']:.0f} / 10,000. Effective contributors: "
            f"{concentration['effective_contributors']:.1f}. These metrics use GitHub's "
            "bounded contributor ranking, not full identity or employment data."
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Operational audit</div>
          <h2>Trace every signal<br>to a record.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    audit = record_audit(issues, pulls)
    if audit.empty:
        st.info("The API sample contains no issues or pull requests.")
    else:
        state_filter = st.multiselect("Record type", ["Issue", "Pull request"], default=["Issue", "Pull request"])
        visible_audit = audit[audit["type"].isin(state_filter)] if state_filter else audit.iloc[0:0]
        st.dataframe(
            visible_audit[["type", "number", "title", "state", "created_at", "age_or_cycle_days", "author", "url"]],
            width="stretch",
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("GitHub record", display_text="Open ↗"),
                "created_at": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
                "age_or_cycle_days": st.column_config.NumberColumn("Age / cycle days", format="%.1f"),
            },
        )
        st.download_button(
            "Download sampled activity CSV",
            visible_audit.to_csv(index=False).encode("utf-8"),
            file_name=f"{repository.replace('/', '_')}_activity_audit.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Data scope, freshness, limits and source links"):
        st.markdown(
            f"""
            **Provider:** GitHub REST API  
            **Repository:** [{metadata['repository']}]({metadata['html_url']})  
            **Retrieved:** `{metadata['retrieved_at']}`  
            **Declared license:** `{metadata['license']}`  
            **Default branch:** `{metadata['default_branch']}`  
            **API mode:** `{metadata['mode']}`  
            **Rate limit remaining at first response:** `{metadata.get('rate_limit_remaining_at_first_call')}`  

            The app retrieves repository metadata plus at most 100 recently updated issues,
            100 recently updated pull requests, 100 contributors, 50 releases and 100 commits.
            Language totals come from GitHub's language endpoint. Public unauthenticated requests
            are normally limited to 60 per originating IP address per hour.

            [REST documentation]({REST_DOCS_URL}) · [Issue endpoint]({ISSUES_DOCS_URL}) ·
            [Pull-request endpoint]({PULLS_DOCS_URL}) · [Rate limits]({RATE_LIMIT_URL}) ·
            [GitHub Terms]({TERMS_URL})
            """
        )
        if metadata["mode"] == "demo":
            st.code(metadata.get("fallback_reason", "Unknown live-data error"), language="text")

    st.warning(
        "Repository activity is not maintainer wellbeing, security, code quality or project "
        "governance. Compare these indicators with documentation, roadmap, security policy, "
        "community norms and full-history analysis before making decisions."
    )
