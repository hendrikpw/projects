"""Streamlit product interface for NYC Collision Risk Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nyc_collision_intelligence.src.analytics import (
    borough_profile,
    daily_anomalies,
    factor_profile,
    filter_collisions,
    hourly_profile,
    spatial_hotspots,
    summary_metrics,
)
from nyc_collision_intelligence.src.data import DATASET_URL, load_data


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#6f747d", "#343840"]


def _style_figure(figure: go.Figure, height: int = 470) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=64, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title=None,
        ),
        hoverlabel=dict(bgcolor="#171a20", font_color="#fcfcfd", bordercolor="#5c6068"),
        colorway=PALETTE,
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=21_600, show_spinner=False)
def _cached_data() -> tuple[pd.DataFrame, dict]:
    return load_data()


def render_dashboard() -> None:
    """Render the complete hosted collision intelligence product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">04 / Geospatial safety analytics</div>
          <h1>NYC Collision<br>Risk Intelligence.</h1>
          <p>
            Turn daily police-reported crash records into spatial risk signals,
            robust anomaly alerts and inspectable evidence about time, place and
            reported contributing factors.
          </p>
          <div class="source-line">NYPD · NYC Open Data · Daily updates</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading the latest NYC collision records…"):
        data, metadata = _cached_data()

    if metadata["mode"] == "demo":
        st.warning(
            "NYC Open Data is currently unavailable. A deterministic synthetic "
            "dataset is shown and is never presented as observed collision data."
        )
    else:
        st.success(
            f"Live collision records loaded through {metadata['latest_date']}.",
            icon="✅",
        )
    if metadata.get("row_limit_reached"):
        st.warning("The safety row limit was reached; the displayed window may be truncated.")

    borough_options = sorted(
        value for value in data["borough"].dropna().unique() if value != "UNKNOWN"
    )
    with st.expander("Analysis controls", expanded=True):
        col_a, col_b, col_c, col_d = st.columns([1, 1.5, 1.2, 1])
        with col_a:
            days = st.select_slider("Latest window", options=[30, 60, 90, 120], value=90)
        with col_b:
            selected_boroughs = st.multiselect(
                "Boroughs",
                borough_options,
                default=borough_options,
            )
        with col_c:
            outcome = st.selectbox(
                "Outcome",
                [
                    "All collisions",
                    "Injury collisions",
                    "Fatal collisions",
                    "Pedestrian / cyclist casualties",
                ],
            )
        with col_d:
            minimum_hotspot = st.slider("Minimum crashes per grid", 2, 12, 4)

    filtered = filter_collisions(data, days, selected_boroughs, outcome)
    if filtered.empty:
        st.info("No collisions match the current filters. Broaden the time or outcome selection.")
        return

    metrics = summary_metrics(filtered)
    previous_start = filtered["crash_date"].min() - pd.Timedelta(days=days)
    previous_end = filtered["crash_date"].min() - pd.Timedelta(days=1)
    previous = data[
        data["crash_date"].between(previous_start, previous_end)
        & data["borough"].isin(selected_boroughs)
    ].copy()
    if outcome == "Injury collisions":
        previous = previous[previous["injury_collision"]]
    elif outcome == "Fatal collisions":
        previous = previous[previous["fatal_collision"]]
    elif outcome == "Pedestrian / cyclist casualties":
        previous = previous[
            (previous["vulnerable_injured"] + previous["vulnerable_killed"]) > 0
        ]
    previous_count = len(previous)
    volume_delta = (
        (metrics["crashes"] - previous_count) / previous_count * 100
        if previous_count
        else None
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Current safety pulse</div>
          <h2>Outcomes first.<br>Volume in context.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    kpi_1.metric(
        "Reported collisions",
        f"{metrics['crashes']:,}",
        None if volume_delta is None else f"{volume_delta:+.1f}% vs prior window",
        delta_color="inverse",
    )
    kpi_2.metric("People injured", f"{metrics['injuries']:,}")
    kpi_3.metric("People killed", f"{metrics['fatalities']:,}")
    kpi_4.metric("Pedestrian / cyclist casualties", f"{metrics['vulnerable_casualties']:,}")
    kpi_5.metric("Injuries per 100 crashes", f"{metrics['injuries_per_100_crashes']:.1f}")

    hotspots = spatial_hotspots(filtered, minimum_crashes=minimum_hotspot)
    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Spatial concentration</div>
          <h2>Find the grid cells<br>where outcomes accumulate.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if hotspots.empty:
        st.info("Not enough geocoded collisions meet the hotspot threshold.")
    else:
        map_figure = px.scatter_mapbox(
            hotspots.head(180),
            lat="latitude_grid",
            lon="longitude_grid",
            size="crashes",
            color="risk_index",
            color_continuous_scale=["#343840", "#a7abb2", "#e5484d"],
            size_max=30,
            zoom=9,
            center={
                "lat": float(hotspots["latitude_grid"].median()),
                "lon": float(hotspots["longitude_grid"].median()),
            },
            hover_name="street",
            hover_data={
                "borough": True,
                "crashes": True,
                "injuries": ":.0f",
                "fatalities": ":.0f",
                "vulnerable_casualties": ":.0f",
                "risk_index": ":.1f",
                "latitude_grid": False,
                "longitude_grid": False,
            },
            title="Collision risk grid · approximately 1 km cells",
        )
        map_figure.update_layout(
            mapbox_style="carto-darkmatter",
            coloraxis_colorbar=dict(title="Risk"),
        )
        st.plotly_chart(_style_figure(map_figure, height=650), width="stretch")
        st.caption(
            "Risk index = crashes + 2× injuries + 25× fatalities + 4× vulnerable-road-user "
            "casualties, min–max scaled within the current selection. It measures observed "
            "concentration, not exposure-adjusted danger."
        )

    temporal, borough_col = st.columns([1.35, 0.65])
    with temporal:
        daily = daily_anomalies(filtered)
        timeline = go.Figure()
        timeline.add_trace(
            go.Scatter(
                x=daily["crash_date"],
                y=daily["crashes"],
                mode="lines",
                name="Daily crashes",
                line=dict(color="#fcfcfd", width=1.6),
            )
        )
        timeline.add_trace(
            go.Scatter(
                x=daily["crash_date"],
                y=daily["baseline"],
                mode="lines",
                name="14-day median",
                line=dict(color="#81848a", width=2, dash="dot"),
            )
        )
        anomalies = daily[daily["is_anomaly"]]
        timeline.add_trace(
            go.Scatter(
                x=anomalies["crash_date"],
                y=anomalies["crashes"],
                mode="markers",
                name="Robust anomaly",
                marker=dict(color="#e5484d", size=11, symbol="diamond"),
                customdata=anomalies[["robust_z"]],
                hovertemplate="%{x|%d %b %Y}<br>%{y} crashes<br>MAD z: %{customdata[0]:.1f}<extra></extra>",
            )
        )
        timeline.update_layout(title="Daily volume and robust anomaly signals")
        timeline.update_xaxes(title=None)
        timeline.update_yaxes(title="Reported collisions")
        st.plotly_chart(_style_figure(timeline), width="stretch")

    with borough_col:
        boroughs = borough_profile(filtered)
        borough_chart = px.bar(
            boroughs.sort_values("crashes"),
            x="crashes",
            y="borough",
            orientation="h",
            text="crashes",
            color="injuries_per_100",
            color_continuous_scale=["#343840", "#81848a", "#e5484d"],
            title="Borough volume & injury rate",
            hover_data={"injuries_per_100": ":.1f", "injuries": True, "fatalities": True},
        )
        borough_chart.update_layout(coloraxis_colorbar=dict(title="Injuries<br>/ 100"))
        borough_chart.update_yaxes(title=None)
        borough_chart.update_xaxes(title="Collisions")
        st.plotly_chart(_style_figure(borough_chart), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Temporal fingerprint</div>
          <h2>Separate weekday rhythm<br>from weekend behavior.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    hours = hourly_profile(filtered)
    heatmap_table = hours.pivot(index="day_type", columns="hour", values="crashes").reindex(
        ["Weekday", "Weekend"]
    )
    heatmap = go.Figure(
        go.Heatmap(
            z=heatmap_table.values,
            x=[f"{hour:02d}:00" for hour in heatmap_table.columns],
            y=heatmap_table.index,
            colorscale=[[0, "#171a20"], [0.55, "#6f747d"], [1, "#e5484d"]],
            colorbar=dict(title="Crashes"),
            hovertemplate="%{y}<br>%{x}<br>%{z:.0f} crashes<extra></extra>",
        )
    )
    heatmap.update_layout(title="Collision volume by hour and day type")
    heatmap.update_xaxes(dtick=2)
    st.plotly_chart(_style_figure(heatmap, height=360), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Reported factors</div>
          <h2>Volume and severity<br>tell different stories.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    minimum_factor_cases = max(5, min(25, len(filtered) // 150))
    factors = factor_profile(filtered, minimum_cases=minimum_factor_cases)
    if factors.empty:
        st.info("Not enough specified contributing-factor observations are available.")
    else:
        top_factors = factors.nlargest(15, "crashes")
        factor_chart = px.scatter(
            top_factors,
            x="crashes",
            y="serious_rate",
            size="injuries",
            color="injuries_per_100",
            text="primary_factor",
            size_max=48,
            color_continuous_scale=["#81848a", "#e5484d"],
            title="Reported factor volume vs observed serious-outcome rate",
            labels={
                "crashes": "Factor-associated collisions",
                "serious_rate": "Serious collisions (%)",
                "injuries_per_100": "Injuries / 100",
            },
            hover_data={"fatalities": True, "injuries": True},
        )
        factor_chart.update_traces(textposition="top center")
        st.plotly_chart(_style_figure(factor_chart, height=560), width="stretch")
        st.caption(
            "Factors are officer-reported and can be missing or provisional. Associations "
            "must not be interpreted as causal effects."
        )

        scenario_left, scenario_right = st.columns([0.7, 1.3])
        with scenario_left:
            factor_name = st.selectbox(
                "Sensitivity factor",
                top_factors.sort_values("crashes", ascending=False)["primary_factor"],
            )
            reduction = st.slider("Hypothetical reduction in associated cases (%)", 0, 50, 15)
        selected = top_factors[top_factors["primary_factor"] == factor_name].iloc[0]
        associated_cases = float(selected["crashes"])
        associated_injuries = float(selected["injuries"])
        with scenario_right:
            scenario_1, scenario_2 = st.columns(2)
            scenario_1.metric(
                "Cases inside sensitivity envelope",
                f"{associated_cases * reduction / 100:,.0f}",
            )
            scenario_2.metric(
                "Injuries inside sensitivity envelope",
                f"{associated_injuries * reduction / 100:,.1f}",
            )
            st.caption(
                "A proportional exposure calculation only. It does not claim that an "
                "intervention would prevent these outcomes."
            )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Hotspot audit</div>
          <h2>Inspect and export<br>the ranked evidence.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if not hotspots.empty:
        audit = hotspots[
            [
                "borough",
                "street",
                "latitude_grid",
                "longitude_grid",
                "crashes",
                "injuries",
                "fatalities",
                "vulnerable_casualties",
                "risk_index",
            ]
        ].head(100)
        audit.columns = [
            "Borough",
            "Representative street",
            "Latitude",
            "Longitude",
            "Crashes",
            "Injuries",
            "Fatalities",
            "Pedestrian/cyclist casualties",
            "Risk index",
        ]
        st.dataframe(
            audit.style.format(
                {
                    "Latitude": "{:.2f}",
                    "Longitude": "{:.2f}",
                    "Risk index": "{:.1f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download hotspot CSV",
            audit.to_csv(index=False).encode("utf-8"),
            file_name="nyc_collision_hotspots.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Data quality, method and interpretation"):
        coordinate_share = data["valid_coordinate"].mean() * 100
        specified_factor_share = (
            ~data["primary_factor"].str.casefold().eq("unspecified")
        ).mean() * 100
        st.markdown(
            f"""
            **Source mode:** `{metadata['mode']}` · **source window:**
            `{metadata['start_date']}` to `{metadata['latest_date']}` ·
            **retrieved:** `{metadata['retrieved_at']}`.

            **Quality snapshot:** {coordinate_share:.1f}% of loaded rows contain coordinates
            inside the NYC validation bounds; {specified_factor_share:.1f}% report a primary
            factor other than “Unspecified”.

            A collision row represents a police-reported event. Reporting thresholds,
            missing coordinates, incomplete factors and exposure differences mean this
            product cannot estimate individual danger or causal intervention effects.
            """
        )
        st.link_button("Open the official NYC dataset", DATASET_URL)
