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


CHART_COLORS = ["#5cc8c1", "#f3a45b", "#7da7e8", "#c3a6e8", "#e88b9b", "#b6c1c9"]


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
          <div class="brand-line">Air quality · Europe</div>
          <h1>What will the air<br>be like this week?</h1>
          <p>
            Compare the forecast for a handful of European cities. See when air
            quality is expected to worsen and which pollutant is driving the change.
          </p>
          <div class="source-line">Forecasts from Open-Meteo, based on CAMS</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Choose a few places</div>
          <h2>Start with the cities<br>you care about.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    selected = st.multiselect(
        "Cities",
        options=list(CITIES),
        default=["Stuttgart", "Berlin", "Paris", "Madrid"],
        max_selections=6,
        help="Pick up to six cities to compare their hourly forecasts for the next seven days.",
    )
    if not selected:
        st.info("Choose at least one city and the forecast will appear here.")
        return
    with st.spinner("Getting the latest forecast…"):
        raw, mode = _load(tuple(selected))
    if raw.empty:
        st.warning("No forecast rows are available for the selected cities.")
        return
    frame = prepare_frame(raw)
    if mode == "Synthetic demo":
        st.warning(
            "The live forecast is unavailable right now, so this page is using sample data. "
            "It does not describe current conditions."
        )
    min_date, max_date = min(frame["date"]), max(frame["date"])
    date_range = st.slider("Dates to compare", min_date, max_date, (min_date, max_date))
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
        '<div class="section-kicker" style="margin:2rem 0 1rem">At a glance</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average AQI", f"{average:.0f}", aqi_band(average))
    c2.metric("Lowest average", best["city"], f"AQI {best['average_aqi']:.0f}")
    c3.metric("Highest hourly AQI", f"{peak['european_aqi']:.0f}", peak["city"])
    c4.metric("Hours above 60", f"{poor_hours:,}", "European AQI")

    st.caption(
        f"{best['city']} has the lowest average in this selection "
        f"(AQI {best['average_aqi']:.0f}). The highest hourly value is "
        f"{peak['european_aqi']:.0f} in {peak['city']}."
    )

    trend = px.line(
        frame, x="time", y="european_aqi", color="city",
        title="Hourly forecast by city",
        labels={"time": "", "european_aqi": "European AQI", "city": "City"},
        color_discrete_sequence=CHART_COLORS,
    )
    trend.add_hrect(
        y0=60, y1=max(105, frame["european_aqi"].max() + 5),
        fillcolor="#e5484d", opacity=0.07, line_width=0,
    )
    trend.add_hline(y=60, line_dash="dot", line_color="#e5484d", annotation_text="AQI 60")
    trend.update_traces(line={"width": 2.4})
    st.plotly_chart(_transparent_layout(trend), width="stretch")

    left, right = st.columns([1.15, 0.85])
    with left:
        daily = daily_profile(frame)
        bars = px.bar(
            daily, x="date", y="mean_aqi", color="city", barmode="group",
            title="Average AQI each day",
            labels={"date": "", "mean_aqi": "Average AQI", "city": "City"},
            color_discrete_sequence=CHART_COLORS,
        )
        st.plotly_chart(_transparent_layout(bars), width="stretch")
    with right:
        mix = pollutant_mix(frame)
        donut = px.pie(
            mix, values="hours", names="pollutant", hole=0.68,
            title="Pollutant most often driving the AQI",
            color_discrete_sequence=["#fcfcfd", "#e5484d", "#8e9299", "#4c5058"],
        )
        donut.update_traces(textinfo="percent+label")
        st.plotly_chart(_transparent_layout(donut), width="stretch")
        st.caption(
            "This shows which pollutant most often has the highest normalized index. "
            "The official AQI still combines all pollutants."
        )

    st.markdown(
        """
        <section class="section-intro" style="margin-top:4rem">
          <div class="section-kicker">The numbers behind the chart</div>
          <h2>Compare city by city.</h2>
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
        "Download these forecast data (CSV)", data=export,
        file_name="european_air_quality_forecast.csv", mime="text/csv",
    )
    with st.expander("Where the data comes from"):
        st.markdown(
            """
            - The forecast comes from the Open-Meteo Air Quality API and is based on CAMS model data.
            - CAMS Europe has an approximate grid size of 11 km. These are model forecasts, not readings from sensors on a particular street.
            - The European AQI shown here is supplied by the API. It reflects the highest of the pollutant-specific indices.
            - If the live service is unavailable, the page switches to sample data and labels it clearly.
            - This comparison is for exploration. It is not medical or regulatory advice.
            """
        )
        st.link_button(
            "Open-Meteo Air Quality documentation",
            "https://open-meteo.com/en/docs/air-quality-api",
        )
