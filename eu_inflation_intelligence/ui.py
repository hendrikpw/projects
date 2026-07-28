"""Streamlit interface for EU Inflation & Household Basket Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eu_inflation_intelligence.src.analytics import (
    category_snapshot,
    country_summary,
    inflation_breadth,
    inflation_regime,
    latest_observations,
    personal_basket,
    spending_pressure,
)
from eu_inflation_intelligence.src.data import (
    CATEGORY_LABELS,
    DATASET_URL,
    GEO_LABELS,
    load_data,
)


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#747982", "#4a4f57", "#292d33"]
BASKET_DEFAULTS = {
    "CP01": 20,
    "CP03": 8,
    "CP04": 30,
    "CP07": 15,
    "CP08": 7,
    "CP09": 10,
    "CP11": 10,
}


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


def _weight_controls() -> dict[str, int]:
    st.markdown("##### Your household spending mix")
    st.caption(
        "Adjust the relative shares. The app normalises the available categories "
        "to 100%, so the sliders do not need to add up manually."
    )
    weights: dict[str, int] = {}
    columns = st.columns(2)
    for index, (code, default) in enumerate(BASKET_DEFAULTS.items()):
        with columns[index % 2]:
            weights[code] = st.slider(
                CATEGORY_LABELS[code],
                min_value=0,
                max_value=60,
                value=default,
                step=1,
                key=f"inflation_weight_{code}",
            )
    return weights


def render_dashboard() -> None:
    """Render the complete inflation comparison and household simulator."""
    st.markdown(
        """
        <section class="page-hero inflation-hero">
          <div class="eyebrow">06 / European price intelligence</div>
          <h1>EU Inflation &<br>Household Basket.</h1>
          <p>
            Compare official harmonised inflation across Europe, reveal category
            pressure and translate the latest rates into a transparent personal
            spending-basket estimate.
          </p>
          <div class="source-line">Eurostat · HICP · ECOICOP version 2</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading official Eurostat HICP observations…"):
        data, metadata = _cached_data()

    if metadata["mode"] == "demo":
        st.warning(
            "Eurostat is temporarily unavailable. Deterministic synthetic data is "
            "shown and is explicitly excluded from official-data claims."
        )
    else:
        latest_period = pd.Period(metadata["latest_period"], freq="M").strftime("%B %Y")
        st.success(
            f"Official Eurostat data loaded through {latest_period} "
            f"({metadata['observations']:,} observations).",
            icon="✅",
        )

    country_options = [
        code for code in GEO_LABELS if code in set(data["geo"].unique())
    ]
    with st.expander("Comparison and budget controls", expanded=True):
        top_a, top_b, top_c = st.columns([1, 1.6, 1])
        with top_a:
            focus_geo = st.selectbox(
                "Focus economy",
                country_options,
                index=country_options.index("DE") if "DE" in country_options else 0,
                format_func=GEO_LABELS.get,
            )
        with top_b:
            default_comparison = [
                code for code in ["EA", "DE", "FR", "IT", "ES"] if code in country_options
            ]
            comparison_geos = st.multiselect(
                "Comparison economies",
                country_options,
                default=default_comparison,
                format_func=GEO_LABELS.get,
                max_selections=8,
            )
        with top_c:
            monthly_spend = st.number_input(
                "Monthly basket spend · €",
                min_value=250,
                max_value=20_000,
                value=2_500,
                step=50,
            )
            start_year = st.select_slider(
                "History starts",
                options=sorted(data["date"].dt.year.unique().tolist()),
                value=max(int(data["date"].dt.year.min()), 2024),
            )
        weights = _weight_controls()

    focus_categories = category_snapshot(data, focus_geo)
    basket_detail, basket = personal_basket(focus_categories, weights)
    focus_latest = latest_observations(
        data[data["geo"].eq(focus_geo) & data["coicop18"].eq("TOTAL")]
    )
    if focus_latest.empty or basket_detail.empty:
        st.info(
            "The chosen economy does not have enough current category data. "
            "Select another economy or give at least one basket category a positive weight."
        )
        return

    official = focus_latest.iloc[0]
    breadth = inflation_breadth(focus_categories)
    pressure = spending_pressure(monthly_spend, basket["rate"])
    regime = inflation_regime(
        float(official["rate"]), float(official["monthly_acceleration"])
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Household pulse</div>
          <h2>Official signal.<br>Your spending mix.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric(
        f"{GEO_LABELS[focus_geo]} HICP",
        f"{official['rate']:.1f}%",
        f"{official['monthly_acceleration']:+.1f} pp vs prior month",
    )
    metric_2.metric("Personal basket estimate", f"{basket['rate']:.1f}%")
    metric_3.metric("Illustrative monthly pressure", f"€{pressure['monthly']:,.0f}")
    metric_4.metric("Illustrative annual pressure", f"€{pressure['annual']:,.0f}")
    metric_5.metric("Categories above 2%", f"{breadth:.0f}%", regime)
    st.caption(
        "HICP is the official annual rate. Personal basket and euro pressure are "
        "illustrations using the latest category rates and your normalised weights; "
        "they are neither an official personal index nor a forecast."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">European comparison</div>
          <h2>One methodology.<br>Different price paths.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    chosen_geos = comparison_geos or [focus_geo]
    history = data[
        data["geo"].isin(chosen_geos)
        & data["coicop18"].eq("TOTAL")
        & (data["date"].dt.year >= start_year)
    ]
    trend = px.line(
        history,
        x="date",
        y="rate",
        color="country",
        markers=True,
        title="All-items HICP · annual rate of change",
        labels={"date": "", "rate": "Annual change · %", "country": ""},
        color_discrete_sequence=PALETTE,
    )
    trend.add_hline(
        y=2,
        line_dash="dot",
        line_color="rgba(252,252,253,.38)",
        annotation_text="2% reference line",
    )
    st.plotly_chart(_style_figure(trend, 520), width="stretch")

    latest = latest_observations(data)
    heatmap_data = latest[
        latest["geo"].isin(chosen_geos)
        & ~latest["coicop18"].eq("TOTAL")
    ]
    heatmap_matrix = heatmap_data.pivot(
        index="category", columns="country", values="rate"
    )
    if not heatmap_matrix.empty:
        heatmap = px.imshow(
            heatmap_matrix,
            color_continuous_scale=[
                [0.0, "#254c3f"],
                [0.48, "#171a20"],
                [0.5, "#fcfcfd"],
                [0.52, "#352226"],
                [1.0, "#e5484d"],
            ],
            color_continuous_midpoint=2,
            text_auto=".1f",
            aspect="auto",
            title="Latest category pressure · annual %",
            labels={"color": "Annual %"},
        )
        st.plotly_chart(_style_figure(heatmap, 610), width="stretch")
        st.caption(
            "Each cell uses that country/category series' latest available month. "
            "The month may differ where Eurostat publication timing is uneven."
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Basket decomposition</div>
          <h2>See what drives<br>your estimate.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    contribution_col, history_col = st.columns([0.9, 1.1])
    with contribution_col:
        contribution = px.bar(
            basket_detail.sort_values("contribution"),
            x="contribution",
            y="category",
            orientation="h",
            color="rate",
            color_continuous_scale=["#747982", "#fcfcfd", "#e5484d"],
            color_continuous_midpoint=2,
            title="Contribution to personal basket rate",
            labels={
                "contribution": "Percentage-point contribution",
                "category": "",
                "rate": "Category %",
            },
            hover_data={"normalized_weight": ":.1f", "rate": ":.1f"},
        )
        st.plotly_chart(_style_figure(contribution, 520), width="stretch")
    with history_col:
        selected_codes = basket_detail["coicop18"].head(5).tolist()
        category_history = data[
            data["geo"].eq(focus_geo)
            & data["coicop18"].isin(selected_codes)
            & (data["date"].dt.year >= start_year)
        ]
        category_trend = px.line(
            category_history,
            x="date",
            y="rate",
            color="category",
            title=f"Highest-contribution categories · {GEO_LABELS[focus_geo]}",
            labels={"date": "", "rate": "Annual change · %", "category": ""},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(_style_figure(category_trend, 520), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Cross-country diagnostic</div>
          <h2>Level, momentum<br>and dispersion.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    summary = country_summary(data)
    diagnostic = px.scatter(
        summary,
        x="rate",
        y="acceleration",
        size="twelve_month_volatility",
        color="robust_distance",
        hover_name="country",
        text="geo",
        color_continuous_scale=["#747982", "#fcfcfd", "#e5484d"],
        color_continuous_midpoint=0,
        title="Latest all-items rate vs monthly acceleration",
        labels={
            "rate": "Latest annual rate · %",
            "acceleration": "Change vs prior monthly reading · pp",
            "robust_distance": "Robust distance",
            "twelve_month_volatility": "12m volatility",
        },
    )
    diagnostic.add_vline(x=2, line_dash="dot", line_color="rgba(252,252,253,.35)")
    diagnostic.add_hline(y=0, line_dash="dot", line_color="rgba(252,252,253,.35)")
    diagnostic.update_traces(textposition="top center")
    st.plotly_chart(_style_figure(diagnostic, 570), width="stretch")
    st.caption(
        "Bubble size is the standard deviation of the last 12 published annual "
        "rates. Colour is a median/MAD distance across economies; it describes "
        "cross-sectional separation and is not an anomaly alert or forecast."
    )

    st.markdown("### Audit and export")
    audit = basket_detail[
        [
            "category",
            "date",
            "rate",
            "raw_weight",
            "normalized_weight",
            "contribution",
        ]
    ].copy()
    audit.columns = [
        "Category",
        "Latest period",
        "Annual rate (%)",
        "Input weight",
        "Normalised weight (%)",
        "Contribution (pp)",
    ]
    st.dataframe(
        audit.style.format(
            {
                "Latest period": lambda value: value.strftime("%Y-%m"),
                "Annual rate (%)": "{:.2f}",
                "Normalised weight (%)": "{:.2f}",
                "Contribution (pp)": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    export = data[data["geo"].isin(chosen_geos)].to_csv(index=False).encode("utf-8")
    download_col, source_col = st.columns([0.45, 0.55])
    with download_col:
        st.download_button(
            "Download selected Eurostat observations",
            export,
            file_name="eurostat_hicp_selected.csv",
            mime="text/csv",
            width="stretch",
        )
    with source_col:
        st.link_button(
            "Open the official Eurostat dataset",
            DATASET_URL,
            width="stretch",
        )

    with st.expander("Method, assumptions and limitations"):
        st.markdown(
            """
            - **Official metric:** `RCH_A` is the percentage change from a month
              to the same month one year earlier. HICP enables harmonised
              cross-country comparison; it is not a cost-of-living index for one
              specific household.
            - **Personal estimate:** category rates are multiplied by your
              normalised input weights. It excludes within-category product
              choices, substitution, regional prices, taxes and personal
              contracts.
            - **Budget pressure:** monthly spend × estimated annual rate. This is
              a like-for-like illustration, not a cash-flow forecast.
            - **Publication timing:** latest observations can differ by country
              or category. Each displayed record retains its exact period.
            - **Source:** Eurostat dataset `prc_hicp_minr`, monthly HICP indices
              and rates, ECOICOP version 2.
            """
        )
