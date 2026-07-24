"""Modern Streamlit user interface for the air-quality project."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from air_quality_intelligence.src.analytics import (
    aqi_band,
    city_summary,
    daily_profile,
    pollutant_mix,
    prepare_frame,
)
from air_quality_intelligence.src.data import CITIES, fetch_cities, generate_demo_data


CHART_COLORS = ["#fcfcfd", "#e5484d", "#a8adb4", "#6d727a", "#d5d7da", "#8e9299"]


@st.cache_data(ttl=1800, show_spinner=False)
def _load(city_names: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    try:
        return fetch_cities(city_names), "Live forecast"
    except (requests.RequestException, ValueError, KeyError):
        return generate_demo_data(city_names), "Synthetic demo"


def _transparent_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial, sans-serif", "color": "#c7c9cd"},
        title={"font": {"size": 20, "color": "#fcfcfd"}},
        margin={"l": 18, "r": 18, "t": 64, "b": 18},
        hoverlabel={"bgcolor": "#171a20", "bordercolor": "#4c5058"},
    )
    fig.update_xaxes(gridcolor="rgba(252,252,253,.08)", zerolinecolor="rgba(252,252,253,.12)")
    fig.update_yaxes(gridcolor="rgba(252,252,253,.08)", zerolinecolor="rgba(252,252,253,.12)")
    return fig


def render_dashboard() -> None:
    st.markdown(
        """
        <section class="page-hero">
          <div class="brand-line">Project 01 / Environment</div>
          <h1>European<br>Air Quality</h1>
          <p>
            Compare urban forecasts, isolate high-risk hours and understand
            which pollutants shape each city's seven-day outlook.
          </p>
          <div class="source-line">Open-Meteo × CAMS / Live model data</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Forecast explorer</div>
          <h2>One atmosphere.<br>Different city signals.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    selected = st.multiselect(
        "Cities",
        options=list(CITIES),
        default=["Stuttgart", "Berlin", "Paris", "Madrid"],
        max_selections=6,
        help="Choose up to six cities. Each selection requests a seven-day hourly forecast.",
    )
    if not selected:
        st.info("Select at least one city to start the comparison.")
        return
    with st.spinner("Loading the latest atmospheric forecast…"):
        raw, mode = _load(tuple(selected))
    if raw.empty:
        st.warning("No forecast rows are available for the selected cities.")
        return
    frame = prepare_frame(raw)
    if mode == "Synthetic demo":
        st.warning(
            "Open-Meteo is temporarily unavailable. The dashboard is showing clearly labelled "
            "synthetic demo data; these are not measurements or forecasts."
        )
    min_date, max_date = min(frame["date"]), max(frame["date"])
    date_range = st.slider("Forecast window", min_date, max_date, (min_date, max_date))
    frame = frame[(frame["date"] >= date_range[0]) & (frame["date"] <= date_range[1])]
    if frame.empty:
        st.info("No rows fall inside this forecast window.")
        return

    summary = city_summary(frame)
    best = summary.iloc[0]
    peak = frame.loc[frame["european_aqi"].idxmax()]
    poor_hours = int((frame["european_aqi"] > 60).sum())
    average = frame["european_aqi"].mean()
    st.markdown(
        '<div class="section-kicker" style="margin:2rem 0 1rem">Current selection / overview</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average AQI", f"{average:.0f}", aqi_band(average))
    c2.metric("Cleanest outlook", best["city"], f"AQI {best['average_aqi']:.0f}")
    c3.metric("Forecast peak", f"{peak['european_aqi']:.0f}", peak["city"])
    c4.metric("Poor-air hours", f"{poor_hours:,}", "AQI > 60")

    trend = px.line(
        frame, x="time", y="european_aqi", color="city",
        title="Hourly European AQI forecast",
        labels={"time": "", "european_aqi": "European AQI", "city": "City"},
        color_discrete_sequence=CHART_COLORS,
    )
    trend.add_hrect(
        y0=60, y1=max(105, frame["european_aqi"].max() + 5),
        fillcolor="#e5484d", opacity=0.07, line_width=0,
    )
    trend.add_hline(y=60, line_dash="dot", line_color="#e5484d", annotation_text="Poor threshold")
    trend.update_traces(line={"width": 2.4})
    st.plotly_chart(_transparent_layout(trend), width="stretch")

    left, right = st.columns([1.15, 0.85])
    with left:
        daily = daily_profile(frame)
        bars = px.bar(
            daily, x="date", y="mean_aqi", color="city", barmode="group",
            title="Daily average by city",
            labels={"date": "", "mean_aqi": "Average AQI", "city": "City"},
            color_discrete_sequence=CHART_COLORS,
        )
        st.plotly_chart(_transparent_layout(bars), width="stretch")
    with right:
        mix = pollutant_mix(frame)
        donut = px.pie(
            mix, values="hours", names="pollutant", hole=0.68,
            title="Dominant concentration signal",
            color_discrete_sequence=["#fcfcfd", "#e5484d", "#8e9299", "#4c5058"],
        )
        donut.update_traces(textinfo="percent+label")
        st.plotly_chart(_transparent_layout(donut), width="stretch")
        st.caption(
            "Diagnostic attribution based on normalized concentrations; "
            "not a replacement for the official consolidated AQI."
        )

    st.markdown(
        """
        <section class="section-intro" style="margin-top:4rem">
          <div class="section-kicker">Detail / cities</div>
          <h2>City comparison.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    table = summary.rename(columns={
        "city": "City", "average_aqi": "Average AQI", "peak_aqi": "Peak AQI",
        "poor_hours": "Hours > 60", "pm25_average": "Avg PM2.5 (µg/m³)",
    })
    for column in ("Average AQI", "Peak AQI", "Avg PM2.5 (µg/m³)"):
        table[column] = table[column].round(1)
    st.dataframe(table, hide_index=True, width="stretch")

    export = frame[
        ["time", "city", "european_aqi", "aqi_band", "pm2_5", "pm10",
         "nitrogen_dioxide", "ozone", "data_mode"]
    ].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered forecast (CSV)", data=export,
        file_name="european_air_quality_forecast.csv", mime="text/csv",
    )
    with st.expander("Methodology, source and limitations"):
        st.markdown(
            """
            - **Source:** Open-Meteo Air Quality API, based on CAMS forecasts.
            - **Resolution:** CAMS Europe is approximately 11 km and hourly; this is modelled
              grid data, not a street-level sensor reading.
            - **AQI:** The consolidated European AQI is supplied by the API and represents
              the maximum of its pollutant-specific indices.
            - **Fallback:** Synthetic data is used only when the live endpoint fails and is visibly labelled.
            - **Use:** Exploratory portfolio application—not medical or regulatory advice.
            """
        )
        st.link_button(
            "Open-Meteo Air Quality documentation",
            "https://open-meteo.com/en/docs/air-quality-api",
        )
