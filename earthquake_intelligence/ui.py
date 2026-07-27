"""Streamlit interface for Global Seismic Activity Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from earthquake_intelligence.src.analytics import (
    cluster_events,
    cluster_summary,
    daily_activity,
    filter_events,
    magnitude_frequency,
    summary_metrics,
)
from earthquake_intelligence.src.data import CATALOG_URL, load_data


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#747982", "#4a4f57", "#292d33"]


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


def _energy_label(joules: float) -> str:
    if joules >= 1e15:
        return f"{joules / 1e15:,.2f} PJ"
    if joules >= 1e12:
        return f"{joules / 1e12:,.1f} TJ"
    return f"{joules / 1e9:,.1f} GJ"


@st.cache_data(ttl=900, show_spinner=False)
def _cached_data() -> tuple[pd.DataFrame, dict]:
    return load_data()


def render_dashboard() -> None:
    """Render the full seismic intelligence mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">05 / Real-time geospatial science</div>
          <h1>Global Seismic<br>Activity Intelligence.</h1>
          <p>
            Explore recent earthquakes through spatial sequences, focal depth,
            physical energy and magnitude-frequency behavior—without turning
            descriptive catalog data into a prediction claim.
          </p>
          <div class="source-line">USGS · ANSS Comprehensive Earthquake Catalog</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading the current USGS earthquake catalog…"):
        data, metadata = _cached_data()

    if metadata["mode"] == "demo":
        st.warning(
            "The USGS service is currently unavailable. A deterministic synthetic "
            "catalog is displayed and is never presented as observed earthquake data."
        )
    else:
        latest = data["time"].max().strftime("%d %b %Y · %H:%M UTC")
        st.success(f"Live USGS catalog loaded through {latest}.", icon="✅")
    if metadata.get("row_limit_reached"):
        st.warning("The 20,000-event safety limit was reached; the source window may be truncated.")

    with st.expander("Analysis controls", expanded=True):
        control_a, control_b, control_c, control_d = st.columns([0.9, 1.2, 1, 1])
        with control_a:
            days = st.select_slider("Latest window", [1, 3, 7, 14, 30], value=14)
            minimum_magnitude = st.slider(
                "Minimum magnitude",
                min_value=2.5,
                max_value=6.5,
                value=3.0,
                step=0.1,
            )
        with control_b:
            maximum_depth = st.slider("Maximum focal depth (km)", 25, 700, 700, 25)
            reviewed_only = st.checkbox("Reviewed events only", value=False)
        with control_c:
            radius_km = st.slider("Sequence radius (km)", 50, 800, 250, 25)
            minimum_cluster_events = st.slider("Minimum events per sequence", 3, 12, 4)
        with control_d:
            tsunami_only = st.checkbox("Tsunami-flagged only", value=False)
            st.caption(
                "The source is loaded once and cached for 15 minutes. Controls filter "
                "and recluster that bounded catalog."
            )

    filtered = filter_events(
        data,
        days=days,
        minimum_magnitude=minimum_magnitude,
        maximum_depth=maximum_depth,
        reviewed_only=reviewed_only,
        tsunami_only=tsunami_only,
    )
    if filtered.empty:
        st.info("No earthquakes match these controls. Reduce the magnitude or broaden the filters.")
        return

    clustered = cluster_events(
        filtered,
        radius_km=radius_km,
        minimum_events=minimum_cluster_events,
    )
    sequences = cluster_summary(clustered)
    metrics = summary_metrics(filtered, completeness_magnitude=minimum_magnitude)
    b_value_text = "n/a" if metrics["b_value"] is None else f"{metrics['b_value']:.2f}"

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Activity pulse</div>
          <h2>Count the events.<br>Respect the logarithmic scale.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    kpi_1.metric("Earthquakes", f"{metrics['events']:,}")
    kpi_2.metric("Maximum magnitude", f"M {metrics['maximum_magnitude']:.1f}")
    kpi_3.metric("Estimated seismic energy", _energy_label(metrics["total_energy_joules"]))
    kpi_4.metric("Tsunami flags", f"{metrics['tsunami_flags']:,}")
    kpi_5.metric("Gutenberg–Richter b", b_value_text)

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Spatial sequences</div>
          <h2>Discover nearby activity<br>without inventing fault boundaries.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    mapped = clustered.sort_values(
        ["magnitude", "significance"], ascending=False
    ).head(2_500)
    map_figure = px.scatter_mapbox(
        mapped,
        lat="latitude",
        lon="longitude",
        color="cluster",
        size="magnitude",
        size_max=24,
        zoom=0.7,
        center={"lat": 12, "lon": 0},
        hover_name="place",
        hover_data={
            "magnitude": ":.1f",
            "depth_km": ":.1f",
            "time": True,
            "region": True,
            "cluster": True,
            "latitude": False,
            "longitude": False,
        },
        color_discrete_sequence=PALETTE,
        title="Recent epicenters · haversine DBSCAN sequences",
    )
    map_figure.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(_style_figure(map_figure, height=650), width="stretch")
    if len(clustered) > len(mapped):
        st.caption(
            f"The map prioritizes the {len(mapped):,} highest-magnitude/significance "
            f"events from {len(clustered):,} filtered rows for browser performance."
        )
    st.caption(
        "DBSCAN groups epicenters by great-circle distance and minimum event count. "
        "Clusters are exploratory spatial concentrations—not identified faults, "
        "aftershock declarations or forecasts."
    )

    timeline_col, depth_col = st.columns([1.15, 0.85])
    with timeline_col:
        daily = daily_activity(filtered)
        timeline = make_subplots(specs=[[{"secondary_y": True}]])
        timeline.add_trace(
            go.Bar(
                x=daily["time"],
                y=daily["events"],
                name="Daily events",
                marker_color="#81848a",
            ),
            secondary_y=False,
        )
        timeline.add_trace(
            go.Scatter(
                x=daily["time"],
                y=daily["energy_terajoules"],
                name="Energy · TJ",
                mode="lines+markers",
                line=dict(color="#e5484d", width=2),
            ),
            secondary_y=True,
        )
        anomalies = daily[daily["is_anomaly"]]
        timeline.add_trace(
            go.Scatter(
                x=anomalies["time"],
                y=anomalies["events"],
                name="Count anomaly",
                mode="markers",
                marker=dict(color="#fcfcfd", size=10, symbol="diamond"),
                customdata=anomalies[["robust_z"]],
                hovertemplate=(
                    "%{x|%d %b}<br>%{y} events"
                    "<br>Robust z %{customdata[0]:.1f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        timeline.update_layout(title="Daily activity and estimated seismic energy")
        timeline.update_yaxes(title_text="Events", secondary_y=False)
        timeline.update_yaxes(title_text="Energy · TJ", type="log", secondary_y=True)
        st.plotly_chart(_style_figure(timeline), width="stretch")

    with depth_col:
        depth_scatter = px.scatter(
            filtered,
            x="magnitude",
            y="depth_km",
            color="depth_class",
            size="significance",
            size_max=26,
            hover_name="place",
            hover_data={"time": True, "significance": True},
            color_discrete_sequence=PALETTE,
            title="Magnitude versus focal depth",
            labels={"magnitude": "Magnitude", "depth_km": "Depth · km"},
        )
        depth_scatter.update_yaxes(autorange="reversed")
        st.plotly_chart(_style_figure(depth_scatter), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Magnitude frequency</div>
          <h2>Test the catalog shape.<br>Show the completeness assumption.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    frequency = magnitude_frequency(filtered, minimum_magnitude=minimum_magnitude)
    frequency_figure = px.line(
        frequency,
        x="magnitude_threshold",
        y="events_at_or_above",
        markers=True,
        title="Cumulative magnitude-frequency distribution",
        labels={
            "magnitude_threshold": "Magnitude threshold",
            "events_at_or_above": "Events at or above threshold · log scale",
        },
    )
    frequency_figure.update_traces(line_color="#e5484d")
    frequency_figure.update_yaxes(type="log")
    st.plotly_chart(_style_figure(frequency_figure, height=430), width="stretch")
    st.caption(
        f"The b-value uses M{minimum_magnitude:.1f} as the assumed completeness "
        "threshold and a 0.1 magnitude bin. It is omitted when fewer than 20 events remain."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Sequence audit</div>
          <h2>Inspect every cluster<br>and every high-magnitude event.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    sequence_col, region_col = st.columns([1.15, 0.85])
    with sequence_col:
        if sequences.empty:
            st.info("No DBSCAN sequence meets the selected radius and event threshold.")
        else:
            sequence_table = sequences[
                [
                    "cluster",
                    "representative_region",
                    "events",
                    "maximum_magnitude",
                    "median_depth_km",
                    "energy_terajoules",
                    "latest_event",
                ]
            ].copy()
            sequence_table.columns = [
                "Sequence",
                "Representative region",
                "Events",
                "Maximum magnitude",
                "Median depth (km)",
                "Energy (TJ)",
                "Latest event",
            ]
            st.dataframe(
                sequence_table.style.format(
                    {
                        "Maximum magnitude": "{:.1f}",
                        "Median depth (km)": "{:.1f}",
                        "Energy (TJ)": "{:,.1f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    with region_col:
        regions = (
            filtered.groupby("region", as_index=False)
            .agg(events=("event_id", "nunique"), maximum_magnitude=("magnitude", "max"))
            .nlargest(12, "events")
            .sort_values("events")
        )
        region_chart = px.bar(
            regions,
            x="events",
            y="region",
            orientation="h",
            text="events",
            color="maximum_magnitude",
            color_continuous_scale=["#343840", "#81848a", "#e5484d"],
            title="Most active reported regions",
            labels={"events": "Events", "region": ""},
        )
        region_chart.update_layout(coloraxis_colorbar=dict(title="Max M"))
        st.plotly_chart(_style_figure(region_chart), width="stretch")

    top_events = filtered.nlargest(100, ["magnitude", "significance"])[
        [
            "time",
            "magnitude",
            "place",
            "depth_km",
            "significance",
            "felt_reports",
            "tsunami_flag",
            "status",
            "event_url",
        ]
    ].copy()
    top_events.columns = [
        "Time UTC",
        "Magnitude",
        "Place",
        "Depth (km)",
        "USGS significance",
        "Felt reports",
        "Tsunami flag",
        "Status",
        "USGS event URL",
    ]
    st.dataframe(
        top_events.style.format({"Magnitude": "{:.1f}", "Depth (km)": "{:.1f}"}),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download filtered earthquake CSV",
        filtered.drop(columns=["depth_class"]).to_csv(index=False).encode("utf-8"),
        file_name="usgs_recent_earthquakes.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Method, data quality and responsible interpretation"):
        st.markdown(
            f"""
            **Source mode:** `{metadata['mode']}` · **source request:**
            `{metadata['start_time']}` to `{metadata['end_time']}` ·
            **retrieved:** `{metadata['retrieved_at']}`.

            Energy is derived from magnitude with
            `log10(E joules) = 1.5 × magnitude + 4.8`. Because magnitude is
            logarithmic, one large event can dominate the total.

            Locations, depths, magnitudes and statuses can be revised as seismic
            networks review an event. A tsunami flag indicates inclusion in a
            USGS/NOAA tsunami workflow; it is not itself a local warning.

            This application describes a recent catalog. It cannot predict the
            time, place or magnitude of a future earthquake and must never replace
            official emergency information.
            """
        )
        st.link_button("Open the official USGS earthquake catalog", CATALOG_URL)
