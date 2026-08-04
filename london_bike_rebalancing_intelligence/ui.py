"""Streamlit interface for London Cycle Rebalancing Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from london_bike_rebalancing_intelligence.src.analytics import (
    add_service_features,
    build_rebalancing_plan,
    network_metrics,
    pressure_clusters,
    quality_report,
    scenario_summary,
)
from london_bike_rebalancing_intelligence.src.data import (
    API_DOCS_URL,
    API_URL,
    OGL_URL,
    OPEN_DATA_URL,
    TERMS_URL,
    load_data,
)


STATUS_COLORS = {
    "Empty risk": "#e5484d",
    "Full risk": "#fcfcfd",
    "Balanced": "#737982",
    "Unavailable": "#292d33",
}
PALETTE = ["#e5484d", "#fcfcfd", "#9ba0a8", "#656a73", "#3c4149", "#24282e"]


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


@st.cache_data(ttl=60, show_spinner=False)
def _cached_network() -> tuple[pd.DataFrame, dict]:
    return load_data()


def _freshness_minutes(metadata: dict) -> float | None:
    retrieved = pd.to_datetime(metadata.get("retrieved_at"), utc=True, errors="coerce")
    updated = pd.to_datetime(metadata.get("station_update_max"), utc=True, errors="coerce")
    if pd.isna(retrieved) or pd.isna(updated):
        return None
    return max(float((retrieved - updated).total_seconds() / 60), 0.0)


def render_dashboard() -> None:
    """Render the live network operations and rebalancing mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">13 / Urban mobility operations</div>
          <h1>London Cycle<br>Rebalancing Intelligence.</h1>
          <p>
            Turn the live Santander Cycles station network into service-level
            signals, spatial pressure clusters and an actionable bike-move plan.
          </p>
          <div class="source-line">Transport for London · Live BikePoint API · Haversine DBSCAN</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading the live TfL cycle-hire network…"):
        raw, metadata = _cached_network()
    if metadata["mode"] == "demo":
        st.warning(
            "TfL live data is unavailable. Every visible station is deterministic synthetic "
            "fallback data and must not be interpreted as the current London network."
        )
    else:
        freshness = _freshness_minutes(metadata)
        freshness_text = "unknown station timestamp" if freshness is None else f"latest station update {freshness:.1f} min before retrieval"
        st.success(f"Live TfL snapshot · {len(raw):,} stations · {freshness_text}")

    controls = st.columns([1, 1, 1, 1])
    with controls[0]:
        critical_percent = st.slider("Critical fill threshold", 5, 30, 15, 1, format="%d%%")
    with controls[1]:
        target_percent = st.slider("Target station fill", 30, 70, 50, 5, format="%d%%")
    with controls[2]:
        radius_km = st.slider("Pressure-cluster radius", 0.2, 2.0, 0.65, 0.05, format="%.2f km")
    with controls[3]:
        min_stations = st.slider("Stations per cluster", 2, 10, 3)

    featured = add_service_features(raw, critical_percent / 100, target_percent / 100)
    clustered, cluster_summary = pressure_clusters(featured, radius_km, min_stations)
    metrics = network_metrics(clustered)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Operational stations", f"{metrics['operational_stations']:,}")
    k2.metric("Available bikes", f"{metrics['bikes']:,}")
    k3.metric("E-bikes", f"{metrics['ebikes']:,}")
    k4.metric("Empty-risk stations", metrics["empty_risk"])
    k5.metric("Full-risk stations", metrics["full_risk"])
    k6.metric("Balanced service", f"{metrics['balanced_share']:.1f}%")
    st.caption(
        "A station is critical when its current bike fill is at or below the selected "
        "threshold, or at or above its mirror value. Availability changes continuously; "
        "this page is a snapshot, not a prediction of future demand."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Live network state</div>
          <h2>See where service<br>breaks first.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    status_options = ["Empty risk", "Full risk", "Balanced", "Unavailable"]
    selected_status = st.multiselect("Map service state", status_options, default=status_options)
    map_data = clustered[clustered["service_status"].isin(selected_status)].copy()
    if map_data.empty:
        st.info("No stations match the selected service states.")
    else:
        network_map = px.scatter_mapbox(
            map_data,
            lat="latitude",
            lon="longitude",
            color="service_status",
            size="docks",
            size_max=18,
            hover_name="station_name",
            hover_data={
                "bikes": True,
                "ebikes": True,
                "empty_docks": True,
                "docks": True,
                "fill_percent": ":.1f",
                "cluster_label": True,
                "latitude": False,
                "longitude": False,
            },
            color_discrete_map=STATUS_COLORS,
            center={"lat": 51.5074, "lon": -0.1278},
            zoom=10.25,
            opacity=0.82,
            title="Current station availability and pressure state",
            labels={"service_status": "Service state"},
        )
        network_map.update_layout(mapbox_style="carto-darkmatter")
        st.plotly_chart(_style_figure(network_map, 720), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pressure clusters", len(cluster_summary))
    c2.metric("Stations inside clusters", int(clustered["cluster_id"].ge(0).sum()))
    c3.metric("Network fill", f"{metrics['network_fill']:.1f}%")
    if cluster_summary.empty:
        st.info("No same-type pressure cluster meets the current radius and station threshold.")
    else:
        st.dataframe(
            cluster_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "center_latitude": None,
                "center_longitude": None,
                "mean_fill_percent": st.column_config.NumberColumn("Mean fill", format="%.1f%%"),
            },
        )
    st.caption(
        "DBSCAN groups empty-risk and full-risk stations separately using great-circle "
        "distance. A cluster describes simultaneous operational pressure, not historical demand."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Network balance</div>
          <h2>Read distribution.<br>Separate bikes from capacity.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        histogram = px.histogram(
            clustered[clustered["operational"]],
            x="fill_percent",
            color="service_status",
            nbins=20,
            title="Station fill distribution",
            labels={"fill_percent": "Bikes / docks · %", "count": "Stations", "service_status": ""},
            color_discrete_map=STATUS_COLORS,
        )
        histogram.add_vline(x=target_percent, line_dash="dash", line_color="#e5484d")
        st.plotly_chart(_style_figure(histogram, 500), width="stretch")
    with right:
        capacity_chart = px.scatter(
            clustered[clustered["operational"]],
            x="docks",
            y="bikes",
            color="service_status",
            size="pressure_score",
            hover_name="station_name",
            title="Available bikes relative to station capacity",
            labels={"docks": "Total docks", "bikes": "Available bikes", "service_status": ""},
            color_discrete_map=STATUS_COLORS,
            opacity=0.72,
        )
        max_capacity = int(clustered["docks"].max())
        capacity_chart.add_trace(
            go.Scatter(
                x=[0, max_capacity],
                y=[0, max_capacity],
                mode="lines",
                line={"color": "rgba(252,252,253,.25)", "dash": "dot"},
                name="100% full",
                hoverinfo="skip",
            )
        )
        st.plotly_chart(_style_figure(capacity_chart, 500), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Rebalancing scenario</div>
          <h2>Move fewer bikes.<br>Resolve more pressure.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    settings = st.columns(3)
    with settings[0]:
        van_capacity = st.slider("Bikes per move", 2, 20, 10)
    with settings[1]:
        max_moves = st.slider("Maximum moves", 5, 60, 30, 5)
    with settings[2]:
        max_distance = st.slider("Maximum donor distance", 1.0, 15.0, 8.0, 0.5, format="%.1f km")
    plan, simulated = build_rebalancing_plan(clustered, van_capacity, max_moves, max_distance)
    scenario = scenario_summary(clustered, simulated, critical_percent / 100)
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Planned moves", len(plan))
    p2.metric("Bikes moved", int(plan["bikes_to_move"].sum()) if not plan.empty else 0)
    p3.metric("Route distance", f"{plan['distance_km'].sum():.1f} km" if not plan.empty else "0.0 km")
    p4.metric("Critical before", scenario["before_empty"] + scenario["before_full"])
    p5.metric("Critical resolved", scenario["critical_resolved"])

    before_after = pd.DataFrame(
        [
            {"state": "Empty risk", "scenario": "Before", "stations": scenario["before_empty"]},
            {"state": "Full risk", "scenario": "Before", "stations": scenario["before_full"]},
            {"state": "Empty risk", "scenario": "After", "stations": scenario["after_empty"]},
            {"state": "Full risk", "scenario": "After", "stations": scenario["after_full"]},
        ]
    )
    plan_col, impact_col = st.columns([1.2, 0.8])
    with plan_col:
        if plan.empty:
            st.info("No feasible donor-to-receiver move meets the current constraints.")
        else:
            st.dataframe(
                plan[["move", "from_station", "to_station", "bikes_to_move", "distance_km", "bike_km"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "distance_km": st.column_config.NumberColumn("Distance · km", format="%.2f"),
                    "bike_km": st.column_config.NumberColumn("Bike-km", format="%.1f"),
                },
            )
            st.download_button(
                "Download rebalancing plan CSV",
                plan.to_csv(index=False).encode("utf-8"),
                file_name="london_cycle_rebalancing_plan.csv",
                mime="text/csv",
                width="stretch",
            )
    with impact_col:
        impact_chart = px.bar(
            before_after,
            x="state",
            y="stations",
            color="scenario",
            barmode="group",
            text="stations",
            title="Critical station impact",
            labels={"state": "", "stations": "Stations", "scenario": ""},
            color_discrete_map={"Before": "#fcfcfd", "After": "#e5484d"},
        )
        st.plotly_chart(_style_figure(impact_chart, 450), width="stretch")
    st.warning(
        "The plan is a nearest-feasible greedy scenario. It ignores road routing, traffic, "
        "depot locations, vehicle shifts, live user arrivals, safety rules and TfL operating "
        "constraints. It is an analytical prioritization aid, not dispatch instructions."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Data quality</div>
          <h2>Audit the feed.<br>Keep uncertainty visible.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    quality = quality_report(clustered)
    q1, q2 = st.columns([0.9, 1.1])
    with q1:
        quality_chart = px.bar(
            quality.sort_values("share"),
            x="share",
            y="check",
            orientation="h",
            text=quality.sort_values("share")["share"].map(lambda value: f"{value:.1f}%"),
            title="Feed and station-state checks",
            labels={"share": "Affected stations · %", "check": ""},
            color="share",
            color_continuous_scale=["#30343b", "#e5484d"],
        )
        quality_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_style_figure(quality_chart, 480), width="stretch")
    with q2:
        audit_columns = [
            "station_name", "station_id", "bikes", "standard_bikes", "ebikes",
            "empty_docks", "unavailable_docks", "docks", "fill_percent",
            "service_status", "cluster_label", "station_updated_at", "quality_score",
        ]
        audit_frame = clustered.sort_values(
            ["pressure_score", "station_name"], ascending=[False, True]
        )[audit_columns]
        st.dataframe(
            audit_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "fill_percent": st.column_config.ProgressColumn("Fill", min_value=0, max_value=100, format="%.1f%%"),
                "station_updated_at": st.column_config.DatetimeColumn("Updated · UTC", format="HH:mm:ss"),
                "quality_score": st.column_config.NumberColumn("Quality", format="%d/100"),
            },
        )
        st.download_button(
            "Download station snapshot CSV",
            audit_frame.to_csv(index=False).encode("utf-8"),
            file_name="tfl_bikepoint_station_snapshot.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Source, fields, freshness, license and assumptions"):
        st.markdown(
            f"""
            **Provider:** Transport for London (TfL)  
            **Endpoint:** [`{API_URL}`]({API_URL})  
            **Retrieved:** `{metadata['retrieved_at']}`  
            **Station update range:** `{metadata.get('station_update_min')}` to `{metadata.get('station_update_max')}`  
            **Mode:** `{metadata['mode']}`  

            Used fields are station ID/name, latitude, longitude, `NbBikes`,
            `NbStandardBikes`, `NbEBikes`, `NbEmptyDocks`, `NbDocks`, `Installed`,
            `Locked`, `Temporary` and each property's `modified` timestamp. The API
            requires no paid credential for this bounded public request. The app caches
            the response for 60 seconds; TfL controls the upstream refresh frequency.

            [BikePoint API documentation]({API_DOCS_URL}) · [TfL open data]({OPEN_DATA_URL}) ·
            [TfL transport data terms]({TERMS_URL}) · [Open Government Licence 3.0]({OGL_URL})
            """
        )
        if metadata["mode"] == "demo":
            st.code(metadata.get("fallback_reason", "Unknown TfL live-data error"), language="text")

    st.caption(
        "Contains Transport for London data. Availability is volatile and may differ from "
        "the official app by the time it is viewed. Always use official customer information "
        "for an actual journey."
    )
