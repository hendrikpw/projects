"""Streamlit interface for Orbital Launch Reliability Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from orbital_launch_intelligence.src.analytics import (
    filter_history,
    monthly_cadence,
    orbit_mix,
    pad_activity,
    provider_reliability,
    simulate_provider_record,
    summary_metrics,
)
from orbital_launch_intelligence.src.data import DOCS_URL, load_data


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#747982", "#4a4f57", "#292d33"]


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


@st.cache_data(ttl=43_200, show_spinner=False)
def _cached_data() -> tuple[pd.DataFrame, dict]:
    return load_data()


def render_dashboard() -> None:
    """Render the complete launch reliability mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">07 / Spaceflight operations analytics</div>
          <h1>Orbital Launch<br>Reliability Intelligence.</h1>
          <p>
            Explore global launch cadence, compare providers with uncertainty-aware
            reliability estimates and inspect where and what the world launches next.
          </p>
          <div class="source-line">The Space Devs · Launch Library 2</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.spinner("Loading recent and upcoming Launch Library 2 records…"):
        data, metadata = _cached_data()

    if metadata["mode"] == "demo":
        st.warning(
            "Launch Library 2 is currently unavailable or rate-limited. A deterministic "
            "synthetic catalog is shown and never presented as observed launch history."
        )
    else:
        retrieved = pd.Timestamp(metadata["retrieved_at"]).strftime("%d %b %Y · %H:%M UTC")
        st.success(
            f"Live catalog loaded at {retrieved}: {metadata['historical_rows']:,} recent "
            f"records and {metadata['upcoming_rows']:,} scheduled launches.",
            icon="✅",
        )

    providers = sorted(data.loc[~data["is_upcoming"], "provider"].dropna().unique())
    with st.expander("Mission window and comparison controls", expanded=True):
        a, b, c, d = st.columns([0.8, 1.6, 0.9, 0.9])
        with a:
            months = st.select_slider("History window", [3, 6, 12, 18, 24], value=18, format_func=lambda v: f"{v} months")
            include_suborbital = st.checkbox("Include suborbital", value=True)
        with b:
            selected_providers = st.multiselect(
                "Providers",
                providers,
                default=[],
                placeholder="All providers",
            )
        with c:
            minimum_attempts = st.slider("Minimum decided attempts", 3, 25, 5)
        with d:
            upcoming_days = st.select_slider("Upcoming horizon", [14, 30, 60, 90, 180], value=60, format_func=lambda v: f"{v} days")
            st.caption("Source requests are cached for 12 hours to respect the free API tier.")

    history = filter_history(data, months, selected_providers, include_suborbital)
    if history.empty:
        st.info("No historical launches match these controls. Broaden the window or provider selection.")
        return
    reliability = provider_reliability(history, minimum_attempts)
    metrics = summary_metrics(history)
    now = pd.Timestamp.now(tz="UTC")
    upcoming = data[
        data["is_upcoming"]
        & (data["net"] >= now)
        & (data["net"] <= now + pd.Timedelta(days=upcoming_days))
    ].sort_values("net")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Launch pulse</div>
          <h2>Cadence, outcomes<br>and market breadth.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Launch records", f"{metrics['launches']:,}", f"{metrics['decided']:,} decided")
    m2.metric("Observed success rate", f"{metrics['success_rate']:.1f}%")
    m3.metric("Monthly cadence", f"{metrics['monthly_cadence']:.1f}")
    m4.metric("Active providers", f"{metrics['providers']:,}")
    m5.metric("Effective providers", f"{metrics['effective_providers']:.1f}", "inverse HHI")
    st.caption(
        "Success rate includes only decided outcomes in the selected window. Effective "
        "providers equals 1 / Herfindahl concentration; a lower value means activity is "
        "more concentrated among a few providers."
    )

    cadence = monthly_cadence(history)
    cadence_figure = make_subplots(specs=[[{"secondary_y": True}]])
    cadence_figure.add_trace(
        go.Bar(x=cadence["month"], y=cadence["launches"], name="Launches", marker_color="#747982"),
        secondary_y=False,
    )
    cadence_figure.add_trace(
        go.Scatter(
            x=cadence["month"],
            y=cadence["success_rate"],
            name="Decided success rate",
            mode="lines+markers",
            line=dict(color="#e5484d", width=2),
        ),
        secondary_y=True,
    )
    cadence_figure.update_yaxes(title_text="Launches", secondary_y=False)
    cadence_figure.update_yaxes(title_text="Success rate · %", range=[0, 105], secondary_y=True)
    cadence_figure.update_layout(title="Monthly launch cadence and decided outcomes")
    st.plotly_chart(_style_figure(cadence_figure, 520), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Reliability benchmark</div>
          <h2>Reward evidence.<br>Show uncertainty.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if reliability.empty:
        st.info("No provider reaches the selected minimum number of decided attempts.")
    else:
        reliability_plot = reliability.sort_values("wilson_low")
        error_plus = reliability_plot["wilson_high"] - reliability_plot["success_rate"]
        error_minus = reliability_plot["success_rate"] - reliability_plot["wilson_low"]
        figure = go.Figure(
            go.Scatter(
                x=reliability_plot["success_rate"],
                y=reliability_plot["provider"],
                mode="markers",
                marker=dict(
                    size=(reliability_plot["attempts"].clip(upper=100) ** 0.5) * 3.2,
                    color=reliability_plot["wilson_low"],
                    colorscale=[[0, "#747982"], [0.7, "#fcfcfd"], [1, "#e5484d"]],
                    colorbar=dict(title="Wilson<br>lower %"),
                    line=dict(color="#fcfcfd", width=1),
                ),
                error_x=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus, color="rgba(252,252,253,.45)"),
                customdata=reliability_plot[["attempts", "successes", "failures", "wilson_low", "wilson_high"]],
                hovertemplate=(
                    "<b>%{y}</b><br>Observed: %{x:.1f}%<br>Attempts: %{customdata[0]}"
                    "<br>Successes: %{customdata[1]}<br>Failures: %{customdata[2]}"
                    "<br>95% Wilson: %{customdata[3]:.1f}–%{customdata[4]:.1f}%<extra></extra>"
                ),
            )
        )
        figure.update_layout(title="Provider success rates · 95% Wilson intervals")
        figure.update_xaxes(range=[max(0, reliability["wilson_low"].min() - 5), 101], title="Decided launch success · %")
        st.plotly_chart(_style_figure(figure, max(460, len(reliability) * 40 + 140)), width="stretch")
        st.caption(
            "Bubble size reflects sample size. Ranking by the Wilson lower bound prevents "
            "a provider with only a few successful launches from appearing artificially certain."
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Geography and purpose</div>
          <h2>Where launches happen.<br>What they are built to do.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    map_col, mix_col = st.columns([1.15, 0.85])
    with map_col:
        pads = pad_activity(history)
        if not pads.empty:
            launch_map = px.scatter_mapbox(
                pads,
                lat="latitude",
                lon="longitude",
                size="launches",
                color="launches",
                hover_name="location",
                hover_data={"pad": True, "country": True, "launches": True, "providers": True, "latitude": False, "longitude": False},
                color_continuous_scale=["#747982", "#fcfcfd", "#e5484d"],
                size_max=42,
                zoom=0.5,
                center={"lat": 15, "lon": 0},
                title="Launch activity by pad",
            )
            launch_map.update_layout(mapbox_style="carto-darkmatter")
            st.plotly_chart(_style_figure(launch_map, 610), width="stretch")
        else:
            st.info("No valid pad coordinates are available for this selection.")
    with mix_col:
        mix = orbit_mix(history).head(28)
        mix_chart = px.sunburst(
            mix,
            path=["orbit", "mission_type"],
            values="launches",
            color="launches",
            color_continuous_scale=["#292d33", "#fcfcfd", "#e5484d"],
            title="Orbit and mission mix",
        )
        st.plotly_chart(_style_figure(mix_chart, 610), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Scenario lab</div>
          <h2>Stress-test a record.<br>Keep the math visible.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if not reliability.empty:
        s1, s2 = st.columns([0.45, 0.55])
        with s1:
            simulated_provider = st.selectbox("Provider record", reliability["provider"].tolist())
            base = reliability[reliability["provider"].eq(simulated_provider)].iloc[0]
            x, y = st.columns(2)
            with x:
                added_successes = st.number_input("Future successes", 0, 50, 5)
            with y:
                added_failures = st.number_input("Future failures", 0, 20, 0)
            scenario = simulate_provider_record(
                int(base["successes"]), int(base["attempts"]), added_successes, added_failures
            )
        with s2:
            k1, k2, k3 = st.columns(3)
            k1.metric("Scenario success rate", f"{scenario['success_rate']:.1f}%", f"{scenario['success_rate'] - base['success_rate']:+.1f} pp")
            k2.metric("Wilson lower bound", f"{scenario['wilson_low']:.1f}%", f"{scenario['wilson_low'] - base['wilson_low']:+.1f} pp")
            k3.metric("Evidence base", f"{scenario['attempts']} launches", f"+{added_successes + added_failures}")
            st.caption(
                "This is a record arithmetic simulator, not a prediction. It shows how "
                "new observed outcomes would change both the headline rate and uncertainty."
            )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Next missions</div>
          <h2>The scheduled board.<br>Exact times can move.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if upcoming.empty:
        st.info("No upcoming launches fall inside the selected horizon.")
    else:
        board = upcoming[["net", "name", "provider", "rocket", "mission_type", "orbit", "location", "status", "probability"]].copy()
        board["net"] = board["net"].dt.strftime("%Y-%m-%d %H:%M UTC")
        board.columns = ["Scheduled NET", "Mission", "Provider", "Rocket", "Mission type", "Orbit", "Location", "Status", "Probability (%)"]
        st.dataframe(board, width="stretch", hide_index=True)
        st.caption("NET means No Earlier Than. Launch dates are operational schedules and may change.")

    export = history.to_csv(index=False).encode("utf-8")
    left, right = st.columns(2)
    with left:
        st.download_button("Download filtered launch data", export, "launch_reliability_selection.csv", "text/csv", width="stretch")
    with right:
        st.link_button("Open Launch Library 2 documentation", DOCS_URL, width="stretch")

    with st.expander("Method, assumptions and limitations"):
        st.markdown(
            """
            - Only outcomes labelled successful or failed are used for reliability.
              Scheduled, in-flight and uncertain records stay visible but are excluded.
            - The 95% Wilson interval quantifies binomial sampling uncertainty. It does
              not adjust for mission difficulty, rocket version, payload or era.
            - The dashboard loads up to 500 recent historical records plus 50 upcoming
              records and therefore describes a bounded recent window, not all spaceflight.
            - HHI is based on launch-count shares in the active filter.
            - Launch Library 2 is maintained by The Space Devs, not by the displayed
              launch providers. Schedule and status data can be revised.
            - No API images are displayed because individual media records carry their
              own licences. Only structured text, numeric fields and coordinates are used.
            """
        )
