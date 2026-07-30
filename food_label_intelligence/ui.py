"""Streamlit interface for Food Label & Product Choice Intelligence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from food_label_intelligence.src.analytics import (
    brand_summary,
    choice_fit_score,
    filter_products,
    missingness_report,
    similar_products,
    summary_metrics,
)
from food_label_intelligence.src.data import CATEGORIES, COUNTRIES, DOCS_URL, load_data


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#747982", "#4a4f57", "#292d33"]
NUTRI_COLORS = {"a": "#287a55", "b": "#6c9340", "c": "#d0a632", "d": "#d77a32", "e": "#e5484d", "unknown": "#747982"}
DEFAULT_WEIGHTS = {
    "Lower sugar": 25,
    "Lower salt": 20,
    "Lower saturated fat": 15,
    "Higher fibre": 20,
    "Higher protein": 20,
}


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
def _cached_data(category: str, country: str) -> tuple[pd.DataFrame, dict]:
    return load_data(category, country)


def _product_label(row: pd.Series) -> str:
    return f"{row['product_name']} · {row['brands']} · {row['code']}"


def render_dashboard() -> None:
    """Render the complete food label comparison mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">08 / Consumer product intelligence</div>
          <h1>Food Label &<br>Product Choice Intelligence.</h1>
          <p>
            Compare packaged foods, inspect label completeness and find
            nutritionally similar products through a transparent preference model.
          </p>
          <div class="source-line">Open Food Facts · Community product database</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    intro_a, intro_b = st.columns(2)
    with intro_a:
        category = st.selectbox("Product category", CATEGORIES, index=0)
    with intro_b:
        country = st.selectbox("Market", COUNTRIES, index=0)

    with st.spinner("Loading the selected Open Food Facts product sample…"):
        data, metadata = _cached_data(category, country)
    if metadata["mode"] == "demo":
        st.warning(
            "Open Food Facts is temporarily unavailable or rate-limited. A deterministic "
            "synthetic catalog is shown and is never presented as observed product data."
        )
    else:
        st.success(
            f"Live Open Food Facts sample loaded: {metadata['returned_products']:,} "
            f"products from {metadata['matching_products']:,} matching records.",
            icon="✅",
        )

    with st.expander("Comparison model and data controls", expanded=True):
        a, b, c = st.columns([0.8, 1.2, 1.4])
        with a:
            minimum_coverage = st.slider("Minimum nutrition coverage", 25, 100, 60, 5, format="%d%%")
            grades = st.multiselect(
                "Official Nutri-Score",
                ["a", "b", "c", "d", "e", "unknown"],
                default=[],
                format_func=lambda value: value.upper(),
                placeholder="All grades",
            )
        with b:
            st.markdown("##### Preference weights")
            st.caption("These create a within-sample comparison score, not a health rating.")
            weights = {}
            for label, default in list(DEFAULT_WEIGHTS.items())[:3]:
                weights[label] = st.slider(label, 0, 50, default, 5)
        with c:
            st.markdown("##### Preference weights · continued")
            st.caption("Missing fields are excluded product-by-product and never treated as zero.")
            for label, default in list(DEFAULT_WEIGHTS.items())[3:]:
                weights[label] = st.slider(label, 0, 50, default, 5)
            minimum_brand_products = st.slider("Minimum products per brand", 2, 10, 3)

    filtered = filter_products(data, minimum_coverage, grades)
    scored = choice_fit_score(filtered, weights)
    scored = scored.dropna(subset=["choice_fit_score"]).sort_values(
        "choice_fit_score", ascending=False
    ).reset_index(drop=True)
    if scored.empty:
        st.info(
            "No products have enough information for these controls. Reduce the "
            "coverage threshold, remove grade filters or activate at least one weight."
        )
        return

    metrics = summary_metrics(scored)
    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Category pulse</div>
          <h2>Labels, nutrients<br>and evidence coverage.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Comparable products", f"{metrics['products']:,}", f"{metrics['brands']:,} brands")
    m2.metric("Median Nutri-Score", metrics["median_grade"])
    m3.metric("Median sugars", f"{metrics['median_sugar']:.1f} g", "per 100 g/ml")
    m4.metric("NOVA group 4", f"{metrics['nova_4_share']:.0f}%", "where recorded")
    m5.metric("Median field coverage", f"{metrics['median_coverage']:.0f}%")
    st.caption(
        "Nutri-Score and NOVA are fields supplied by Open Food Facts. The Choice Fit "
        "Score below is a separate, custom percentile comparison controlled by your weights."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Product landscape</div>
          <h2>See the trade-offs.<br>Keep the model inspectable.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    plot_data = scored.dropna(
        subset=["sugars_100g", "proteins_100g", "fiber_100g"]
    ).copy()
    landscape = px.scatter(
        plot_data,
        x="sugars_100g",
        y="proteins_100g",
        size="fiber_100g",
        color="choice_fit_score",
        hover_name="product_name",
        hover_data={
            "brands": True,
            "nutri_grade": True,
            "salt_100g": ":.2f",
            "nutrition_coverage": ":.0f",
            "sugars_100g": ":.1f",
            "proteins_100g": ":.1f",
            "fiber_100g": ":.1f",
        },
        color_continuous_scale=["#747982", "#fcfcfd", "#e5484d"],
        size_max=28,
        title="Sugar–protein trade-off · bubble size is fibre",
        labels={
            "sugars_100g": "Sugars · g per 100 g/ml",
            "proteins_100g": "Protein · g per 100 g/ml",
            "choice_fit_score": "Choice Fit",
        },
    )
    st.plotly_chart(_style_figure(landscape, 570), width="stretch")

    grade_col, brand_col = st.columns([0.42, 0.58])
    with grade_col:
        grades_count = (
            scored.groupby("nutri_grade", as_index=False)
            .agg(products=("code", "nunique"))
        )
        grade_chart = px.bar(
            grades_count,
            x="nutri_grade",
            y="products",
            color="nutri_grade",
            color_discrete_map=NUTRI_COLORS,
            category_orders={"nutri_grade": ["a", "b", "c", "d", "e", "unknown"]},
            title="Official Nutri-Score availability",
            labels={"nutri_grade": "Grade", "products": "Products"},
        )
        grade_chart.update_layout(showlegend=False)
        st.plotly_chart(_style_figure(grade_chart, 440), width="stretch")
    with brand_col:
        brands = brand_summary(scored, minimum_brand_products).head(15)
        if brands.empty:
            st.info("No brand reaches the selected minimum product count.")
        else:
            brand_chart = px.bar(
                brands.sort_values("median_fit"),
                x="median_fit",
                y="brands",
                orientation="h",
                color="products",
                color_continuous_scale=["#747982", "#fcfcfd", "#e5484d"],
                title="Brand median Choice Fit · sample-aware",
                labels={"median_fit": "Median Choice Fit", "brands": "", "products": "Products"},
                hover_data={"median_sugar": ":.1f", "median_protein": ":.1f", "nova_4_share": ":.0f"},
            )
            st.plotly_chart(_style_figure(brand_chart, 440), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Product workbench</div>
          <h2>Inspect one label.<br>Find nearby profiles.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    labels = {_product_label(row): row["code"] for _, row in scored.iterrows()}
    selected_label = st.selectbox("Product to inspect", list(labels), index=0)
    selected_code = labels[selected_label]
    selected = scored[scored["code"].eq(selected_code)].iloc[0]
    alternatives = similar_products(scored, selected_code, limit=8)

    detail_col, radar_col = st.columns([0.44, 0.56])
    with detail_col:
        st.markdown(f"### {selected['product_name']}")
        st.caption(f"{selected['brands']} · barcode {selected['code']} · {selected['quantity'] or 'quantity not recorded'}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Choice Fit", f"{selected['choice_fit_score']:.0f}")
        k2.metric("Nutri-Score", str(selected["nutri_grade"]).upper())
        k3.metric("Coverage", f"{selected['nutrition_coverage']:.0f}%")
        st.markdown("**Ingredients**")
        st.write(selected["ingredients_text"] or "Not recorded")
        st.markdown("**Recorded additives**")
        st.write(selected["additives"] or "None recorded")
        st.markdown("**Recorded allergens**")
        st.write(selected["allergens"] or "None recorded")
    with radar_col:
        radar_fields = [
            ("Sugars", "sugars_100g"),
            ("Salt", "salt_100g"),
            ("Saturated fat", "saturated_fat_100g"),
            ("Fibre", "fiber_100g"),
            ("Protein", "proteins_100g"),
        ]
        category_median = scored[[field for _, field in radar_fields]].median()
        values = [selected[field] for _, field in radar_fields]
        median_values = [category_median[field] for _, field in radar_fields]
        theta = [label for label, _ in radar_fields]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name="Selected product", line_color="#e5484d"))
        radar.add_trace(go.Scatterpolar(r=median_values, theta=theta, fill="toself", name="Sample median", line_color="#fcfcfd", opacity=0.65))
        radar.update_layout(title="Recorded nutrients · g per 100 g/ml", polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor="rgba(252,252,253,.12)")))
        st.plotly_chart(_style_figure(radar, 500), width="stretch")

    if alternatives.empty:
        st.info("Not enough complete products are available for similarity matching.")
    else:
        similar_table = alternatives[
            ["product_name", "brands", "nutrition_distance", "choice_fit_score", "nutri_grade", "sugars_100g", "proteins_100g", "fiber_100g", "salt_100g"]
        ].copy()
        similar_table.columns = ["Product", "Brand", "Nutrition distance", "Choice Fit", "Nutri-Score", "Sugars", "Protein", "Fibre", "Salt"]
        st.markdown("#### Nutritionally similar products")
        st.dataframe(
            similar_table.style.format({"Nutrition distance": "{:.2f}", "Choice Fit": "{:.1f}", "Sugars": "{:.1f}", "Protein": "{:.1f}", "Fibre": "{:.1f}", "Salt": "{:.2f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Similarity is Euclidean distance after median imputation and z-score "
            "standardisation across eight recorded nutrient fields. It is not a recommendation."
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Data quality</div>
          <h2>Measure what is missing.<br>Do not hide it.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    quality = missingness_report(data)
    quality_chart = px.bar(
        quality,
        x="coverage",
        y="field",
        orientation="h",
        color="coverage",
        color_continuous_scale=["#e5484d", "#747982", "#fcfcfd"],
        range_color=[0, 100],
        title="Field availability in the unfiltered source sample",
        labels={"coverage": "Available products · %", "field": ""},
        text_auto=".0f",
    )
    st.plotly_chart(_style_figure(quality_chart, 500), width="stretch")

    export = scored.to_csv(index=False).encode("utf-8")
    left, right = st.columns(2)
    with left:
        st.download_button("Download comparable product sample", export, "food_label_product_sample.csv", "text/csv", width="stretch")
    with right:
        st.link_button("Open Open Food Facts API documentation", DOCS_URL, width="stretch")

    with st.expander("Method, assumptions and limitations"):
        st.markdown(
            """
            - **Choice Fit** is calculated only within the currently loaded sample.
              Each nutrient becomes a percentile and the visible weights are normalised
              over fields available for that product.
            - Lower sugar, salt and saturated fat receive higher percentiles; higher
              fibre and protein receive higher percentiles. This is not an official
              nutrition score, medical advice or a statement that one food is healthy.
            - Nutri-Score and NOVA are reused fields from Open Food Facts and can be
              absent where contributors supplied insufficient information.
            - Values are normally expressed per 100 g or 100 ml. Comparing foods and
              beverages, or different serving patterns, requires care.
            - Open Food Facts is crowdsourced. Labels can be incomplete, outdated,
              incorrectly entered or market-specific.
            - The API response is capped at 100 products and sorted by scan activity;
              it is not a random or representative sample of an entire market.
            """
        )
