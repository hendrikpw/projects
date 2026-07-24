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


PAGE_CSS = """
<style>
.aq-hero {
  padding: 1.65rem 1.8rem; border-radius: 22px;
  background: linear-gradient(135deg, rgba(8, 47, 73, .92), rgba(15, 23, 42, .92));
  border: 1px solid rgba(94, 234, 212, .18); margin-bottom: 1rem;
}
.aq-hero h1 { margin: .28rem 0 .5rem; font-size: clamp(2rem, 4vw, 3.4rem); }
.aq-hero p { color: #a9bdd2; margin: 0; max-width: 820px; }
.source-pill {
  display: inline-block; margin-top: .8rem; padding: .35rem .7rem; border-radius: 999px;
  color: #b7fff4; background: rgba(45, 212, 191, .10);
  border: 1px solid rgba(45, 212, 191, .23); font-size: .76rem; font-weight: 600;
}
[data-testid="stMetric"] {
  background: rgba(16, 32, 57, .78); border: 1px solid rgba(148, 163, 184, .16);
  border-radius: 16px; padding: 1rem 1.1rem;
}
</style>
"""


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
        font={"color": "#dbe7f3"},
        margin={"l": 12, "r": 12, "t": 56, "b": 12},
        hoverlabel={"bgcolor": "#102039"},
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)")
    return fig


def render_dashboard() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <section class="aq-hero">
          <div class="eyebrow">Environmental intelligence · 7-day outlook</div>
          <h1>European Air Quality Intelligence</h1>
          <p>Compare urban air-quality forecasts, isolate high-risk hours and understand
          which pollutants shape each city's outlook.</p>
          <span class="source-pill">Live model data · Open-Meteo × CAMS</span>
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average AQI", f"{average:.0f}", aqi_band(average))
    c2.metric("Cleanest outlook", best["city"], f"AQI {best['average_aqi']:.0f}")
    c3.metric("Forecast peak", f"{peak['european_aqi']:.0f}", peak["city"])
    c4.metric("Poor-air hours", f"{poor_hours:,}", "AQI > 60")

    trend = px.line(
        frame, x="time", y="european_aqi", color="city",
        title="Hourly European AQI forecast",
        labels={"time": "", "european_aqi": "European AQI", "city": "City"},
        color_discrete_sequence=["#5eead4", "#60a5fa", "#c084fc", "#fbbf24", "#fb7185", "#a3e635"],
    )
    trend.add_hrect(
        y0=60, y1=max(105, frame["european_aqi"].max() + 5),
        fillcolor="#fb7185", opacity=0.06, line_width=0,
    )
    trend.add_hline(y=60, line_dash="dot", line_color="#fb923c", annotation_text="Poor threshold")
    trend.update_traces(line={"width": 2.4})
    st.plotly_chart(_transparent_layout(trend), use_container_width=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        daily = daily_profile(frame)
        bars = px.bar(
            daily, x="date", y="mean_aqi", color="city", barmode="group",
            title="Daily average by city",
            labels={"date": "", "mean_aqi": "Average AQI", "city": "City"},
            color_discrete_sequence=["#5eead4", "#60a5fa", "#c084fc", "#fbbf24", "#fb7185", "#a3e635"],
        )
        st.plotly_chart(_transparent_layout(bars), use_container_width=True)
    with right:
        mix = pollutant_mix(frame)
        donut = px.pie(
            mix, values="hours", names="pollutant", hole=0.68,
            title="Dominant concentration signal",
            color_discrete_sequence=["#5eead4", "#60a5fa", "#fbbf24", "#fb7185"],
        )
        donut.update_traces(textinfo="percent+label")
        st.plotly_chart(_transparent_layout(donut), use_container_width=True)
        st.caption(
            "Diagnostic attribution based on normalized concentrations; "
            "not a replacement for the official consolidated AQI."
        )

    st.subheader("City comparison")
    table = summary.rename(columns={
        "city": "City", "average_aqi": "Average AQI", "peak_aqi": "Peak AQI",
        "poor_hours": "Hours > 60", "pm25_average": "Avg PM2.5 (µg/m³)",
    })
    for column in ("Average AQI", "Peak AQI", "Avg PM2.5 (µg/m³)"):
        table[column] = table[column].round(1)
    st.dataframe(table, hide_index=True, use_container_width=True)

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
