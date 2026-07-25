"""Streamlit interface for the Energy Transition Intelligence Explorer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_transition_intelligence.src.analytics import (
    SCORE_CODES,
    cluster_countries,
    latest_snapshot,
    scenario_score,
    score_countries,
)
from energy_transition_intelligence.src.data import INDICATORS, load_data


PALETTE = ["#e5484d", "#fcfcfd", "#81848a", "#b8bbc0", "#5c6068"]


def _layout(figure: go.Figure, height: int = 470) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=62, b=20),
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


@st.cache_data(ttl=86_400, show_spinner=False)
def _cached_data() -> tuple[pd.DataFrame, dict]:
    return load_data()


def render_dashboard() -> None:
    """Render the complete mini-product inside the shared portfolio."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">03 / Machine learning & policy scenarios</div>
          <h1>Energy Transition<br>Intelligence.</h1>
          <p>
            Compare European transition profiles, discover structural peer groups
            and test how renewable growth, lower emissions and efficiency gains
            could change a country's relative position.
          </p>
          <div class="source-line">World Bank · World Development Indicators</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading World Bank indicators…"):
        data, metadata = _cached_data()

    if metadata["mode"] == "demo":
        st.warning(
            "The World Bank API is currently unavailable. The app is showing a "
            "deterministic synthetic demo dataset; it is never presented as observed data."
        )
    else:
        st.success("Live World Bank data loaded successfully.", icon="✓")

    snapshot = latest_snapshot(data)
    available_countries = sorted(snapshot["country"].tolist())

    with st.expander("Analysis controls", expanded=True):
        control_a, control_b = st.columns([2.2, 1])
        with control_a:
            selected_countries = st.multiselect(
                "Countries",
                available_countries,
                default=[
                    country
                    for country in [
                        "Germany",
                        "France",
                        "Spain",
                        "Italy",
                        "Denmark",
                        "Sweden",
                        "Norway",
                        "Poland",
                    ]
                    if country in available_countries
                ],
                help="The score is relative to the countries included in this comparison.",
            )
        with control_b:
            max_clusters = min(5, max(2, len(selected_countries)))
            clusters = st.slider("Transition profiles", 2, max_clusters, min(3, max_clusters))

    if len(selected_countries) < 2:
        st.info("Select at least two countries to build a meaningful comparison.")
        return

    filtered_snapshot = snapshot[snapshot["country"].isin(selected_countries)].copy()
    try:
        scored = score_countries(filtered_snapshot)
        clustered = cluster_countries(scored, min(clusters, len(scored)))
    except ValueError as exc:
        st.error(f"The selected comparison cannot be calculated: {exc}")
        return

    leader = scored.iloc[0]
    renewable_median = scored["EG.ELC.RNEW.ZS"].median()
    complete_share = (scored["imputed_fields"] == 0).mean() * 100

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Relative benchmark</div>
          <h2>One transparent score.<br>Three measurable drivers.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Transition leader", leader["country"], f"{leader['transition_score']:.1f} / 100")
    kpi_2.metric("Median score", f"{scored['transition_score'].median():.1f}")
    kpi_3.metric("Median renewable output", f"{renewable_median:.1f}%")
    kpi_4.metric("Complete score coverage", f"{complete_share:.0f}%")

    left, right = st.columns([1.1, 0.9])
    with left:
        scatter = px.scatter(
            clustered,
            x="pca_x",
            y="pca_y",
            color="cluster",
            text="country_code",
            hover_name="country",
            hover_data={
                "transition_score": ":.1f",
                "EG.ELC.RNEW.ZS": ":.1f",
                "EN.ATM.CO2E.PC": ":.2f",
                "EG.EGY.PRIM.PP.KD": ":.2f",
                "pca_x": False,
                "pca_y": False,
                "cluster": False,
            },
            color_discrete_sequence=PALETTE,
            title="Structural peer groups · PCA projection",
        )
        scatter.update_traces(marker=dict(size=15, line=dict(width=1, color="#101319")))
        scatter.update_traces(textposition="top center")
        scatter.update_xaxes(title="Principal component 1", showticklabels=False)
        scatter.update_yaxes(title="Principal component 2", showticklabels=False)
        st.plotly_chart(_layout(scatter), width="stretch")
        st.caption(
            "K-means clusters standardized indicator values; PCA compresses the same "
            "three dimensions for display. Distance suggests similarity, not causation."
        )

    with right:
        ranking = scored.sort_values("transition_score")
        bars = px.bar(
            ranking,
            x="transition_score",
            y="country",
            orientation="h",
            text="transition_score",
            title="Relative transition score",
            color="transition_score",
            color_continuous_scale=["#343840", "#81848a", "#e5484d"],
        )
        bars.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        bars.update_layout(coloraxis_showscale=False)
        bars.update_xaxes(range=[0, 105], title="Score / 100")
        bars.update_yaxes(title=None)
        st.plotly_chart(_layout(bars), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Time series</div>
          <h2>See the trajectory,<br>not only the latest value.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    renewable = data[
        (data["country"].isin(selected_countries))
        & (data["indicator_code"] == "EG.ELC.RNEW.ZS")
    ].copy()
    if renewable.empty:
        st.info("No renewable-electricity time series are available for this selection.")
    else:
        trend = px.line(
            renewable,
            x="year",
            y="value",
            color="country",
            markers=True,
            title="Renewable electricity output over time",
            labels={"value": "% of total electricity output", "year": ""},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(_layout(trend, height=510), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Scenario lab</div>
          <h2>Change the drivers.<br>Measure the relative effect.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    scenario_a, scenario_b = st.columns([0.75, 1.25])
    with scenario_a:
        scenario_country = st.selectbox("Country", scored["country"].tolist())
        renewable_change = st.slider("Renewable output increase (percentage points)", 0, 30, 10)
        co2_reduction = st.slider("CO₂ reduction (%)", 0, 50, 15)
        intensity_reduction = st.slider("Energy-intensity reduction (%)", 0, 40, 10)

    row = scored.loc[scored["country"] == scenario_country].iloc[0]
    baseline, simulated, changed = scenario_score(
        row,
        scored,
        renewable_change,
        co2_reduction,
        intensity_reduction,
    )
    with scenario_b:
        scenario_chart = go.Figure(
            go.Bar(
                x=["Current", "Scenario"],
                y=[baseline, simulated],
                marker_color=["#81848a", "#e5484d"],
                text=[f"{baseline:.1f}", f"{simulated:.1f}"],
                textposition="outside",
                customdata=[
                    [
                        row["EG.ELC.RNEW.ZS"],
                        row["EN.ATM.CO2E.PC"],
                        row["EG.EGY.PRIM.PP.KD"],
                    ],
                    [
                        changed["EG.ELC.RNEW.ZS"],
                        changed["EN.ATM.CO2E.PC"],
                        changed["EG.EGY.PRIM.PP.KD"],
                    ],
                ],
                hovertemplate=(
                    "Score %{y:.1f}<br>Renewable %{customdata[0]:.1f}%"
                    "<br>CO₂ %{customdata[1]:.2f} t/capita"
                    "<br>Intensity %{customdata[2]:.2f} MJ/$<extra></extra>"
                ),
            )
        )
        scenario_chart.update_layout(title=f"{scenario_country} · score impact")
        scenario_chart.update_yaxes(range=[0, 105], title="Score / 100")
        st.plotly_chart(_layout(scenario_chart, height=420), width="stretch")
        st.metric("Simulated score change", f"{simulated - baseline:+.1f} points")
        st.caption(
            "This is a sensitivity analysis, not a forecast. Other countries and the "
            "normalization range remain fixed so the isolated scenario effect is visible."
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Audit table</div>
          <h2>Every input remains<br>inspectable and exportable.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    table = scored[
        [
            "rank",
            "country",
            "transition_score",
            "EG.ELC.RNEW.ZS",
            "EN.ATM.CO2E.PC",
            "EG.EGY.PRIM.PP.KD",
            "EG.USE.ELEC.KH.PC",
            "coverage",
            "imputed_fields",
        ]
    ].copy()
    table.columns = [
        "Rank",
        "Country",
        "Score",
        "Renewable output (%)",
        "CO₂ per capita (t)",
        "Energy intensity (MJ/$)",
        "Power use (kWh/capita)",
        "Coverage",
        "Imputed score fields",
    ]
    st.dataframe(
        table.style.format(
            {
                "Score": "{:.1f}",
                "Renewable output (%)": "{:.1f}",
                "CO₂ per capita (t)": "{:.2f}",
                "Energy intensity (MJ/$)": "{:.2f}",
                "Power use (kWh/capita)": "{:,.0f}",
                "Coverage": "{:.0%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download comparison CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="energy_transition_comparison.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Method, freshness and interpretation"):
        st.markdown(
            f"""
            **Score formula:** 50% renewable electricity output + 30% inverse CO₂
            emissions per capita + 20% inverse primary-energy intensity. Each component
            is min–max normalized inside the selected comparison. Missing score inputs
            are median-imputed and counted in the audit table.

            **Source mode:** `{metadata['mode']}` · **retrieved:** `{metadata['retrieved_at']}`.
            World Development Indicators are annual, but publication schedules and the
            latest available year differ by indicator and country. The app queries
            2010–2024 and selects the latest non-null observation per input.

            The score is an explanatory portfolio metric, not an official World Bank
            index and not a policy recommendation.
            """
        )
        st.link_button(
            "Open World Bank API documentation",
            "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392",
        )
