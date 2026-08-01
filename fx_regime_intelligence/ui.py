"""Streamlit interface for FX Market Regime Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fx_regime_intelligence.src.analytics import (
    correlation_and_clusters,
    detect_anomalies,
    inverse_volatility_allocation,
    market_regimes,
    normalized_rates,
    rate_matrix,
    risk_summary,
    rolling_volatility,
    shock_scenario,
)
from fx_regime_intelligence.src.data import (
    API_DOCS_URL,
    CURRENCIES,
    DATASET_URL,
    USAGE_POLICY_URL,
    load_data,
)


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#747982", "#4a4f57", "#292d33"]
REGIME_COLORS = {"Stress": "#e5484d", "Normal": "#a7abb2", "Calm": "#4a4f57"}


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
    """Render the complete exchange-rate regime and risk mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">10 / Financial market intelligence</div>
          <h1>FX Market<br>Regime Intelligence.</h1>
          <p>
            Monitor euro reference rates, detect unusual multi-currency sessions
            and translate volatility into transparent exposure decisions.
          </p>
          <div class="source-line">European Central Bank · Daily reference rates</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading daily ECB reference exchange rates…"):
        data, metadata = _cached_data()
    if metadata["mode"] == "demo":
        st.warning(
            "The ECB endpoint is temporarily unavailable. All components are running "
            "on deterministic synthetic FX paths and must not be used for decisions."
        )
    else:
        st.success(
            f"Live ECB data · {metadata['observations']:,} observations · "
            f"{metadata['currencies']} currencies · through {metadata['end_date']}"
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Market scope</div>
          <h2>One euro base.<br>Multiple risk perspectives.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.25, 0.65, 0.65])
    with c1:
        selected = st.multiselect(
            "Currencies",
            list(CURRENCIES),
            default=["USD", "GBP", "CHF", "JPY", "CAD"],
            format_func=lambda code: f"{code} · {CURRENCIES[code]}",
        )
    with c2:
        lookback = st.selectbox("History", [1, 3, 5], index=1, format_func=lambda value: f"{value} year" if value == 1 else f"{value} years")
    with c3:
        window = st.select_slider("Risk window", options=[10, 20, 30, 60, 90], value=20, format_func=lambda value: f"{value} days")

    if len(selected) < 2:
        st.info("Select at least two currencies to unlock regimes, anomalies and correlation analysis.")
        return
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=int(lookback))
    scoped = data[data["date"] >= start_date].copy()
    rates = rate_matrix(scoped, selected).dropna(how="all")
    if rates.empty or len(rates) < 80:
        st.info("Not enough aligned ECB observations are available for this selection.")
        return

    summary = risk_summary(rates, window)
    lead = summary[summary["currency"].eq(selected[0])].iloc[0]
    regimes = market_regimes(rates, window)
    latest_regime = regimes.iloc[-1]["regime"] if not regimes.empty else "n/a"
    latest_regime_days = int((regimes["regime"] == latest_regime).iloc[::-1].cumprod().sum()) if not regimes.empty else 0
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"EUR/{selected[0]}", f"{lead['latest_rate']:,.4f}")
    k2.metric("30-day change", f"{lead['change_30d']:+.2f}%")
    k3.metric(f"{window}d annualised vol", f"{lead['annualized_volatility']:.2f}%")
    k4.metric("Market regime", latest_regime, f"{latest_regime_days} sessions")
    k5.metric("Currencies", len(selected))
    st.caption(
        "Rates are foreign-currency units per euro. A rising EUR/USD rate means the euro "
        "buys more dollars. All returns and risk measures are calculated by this app and "
        "are not official ECB indicators or investment advice."
    )

    normalized = normalized_rates(rates).reset_index().melt(id_vars="date", var_name="currency", value_name="index")
    performance = px.line(
        normalized,
        x="date",
        y="index",
        color="currency",
        title="Euro purchasing-power paths · rebased to 100",
        labels={"date": "", "index": "Index", "currency": "Currency"},
    )
    performance.add_hline(y=100, line_dash="dot", line_color="rgba(252,252,253,.28)")
    st.plotly_chart(_style_figure(performance, 530), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Regime monitor</div>
          <h2>Separate normal movement<br>from unusual market stress.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    contamination = st.slider("Isolation Forest alert share", 1, 10, 3, 1, format="%d%%") / 100
    anomalies = detect_anomalies(rates, contamination, window)
    anomaly_count = int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0
    stress_share = float(regimes["regime"].eq("Stress").mean() * 100) if not regimes.empty else 0
    a1, a2, a3 = st.columns(3)
    a1.metric("Flagged sessions", anomaly_count)
    a2.metric("Stress-regime share", f"{stress_share:.1f}%")
    a3.metric("Worst lead-currency day", f"{lead['worst_day']:.2f}%")

    left, right = st.columns([1.06, 0.94])
    with left:
        regime_chart = px.scatter(
            regimes.reset_index(),
            x="date",
            y="market_volatility",
            color="regime",
            color_discrete_map=REGIME_COLORS,
            title="Rolling cross-currency volatility regime",
            labels={"date": "", "market_volatility": "Median annualised volatility · %", "regime": "Regime"},
            hover_data={"absolute_move": ":.3f", "dispersion": ":.3f"},
        )
        regime_chart.update_traces(marker_size=5)
        st.plotly_chart(_style_figure(regime_chart, 500), width="stretch")
    with right:
        anomaly_chart = px.scatter(
            anomalies,
            x="date",
            y="anomaly_score",
            color="is_anomaly",
            size="market_move",
            color_discrete_map={True: "#e5484d", False: "#4a4f57"},
            title="Isolation Forest session anomaly score",
            labels={"date": "", "anomaly_score": "Relative anomaly score", "is_anomaly": "Alert"},
            hover_data={"market_move": ":.3f", "dispersion": ":.3f"},
        )
        anomaly_chart.update_layout(showlegend=False)
        st.plotly_chart(_style_figure(anomaly_chart, 500), width="stretch")

    with st.expander("How the regime and anomaly models work"):
        st.markdown(
            f"""
            **Regimes** use the median annualised {window}-day volatility across the
            selected currencies. An expanding historical baseline labels observations
            below its 35th percentile *Calm*, above its 80th percentile *Stress* and
            everything else *Normal*. This avoids using future observations to classify
            earlier dates.

            **Isolation Forest** learns unusual combinations of all daily currency
            returns, the median absolute market move, cross-currency dispersion and
            rolling volatility. The visible {contamination:.0%} setting controls the
            approximate alert share. An alert means statistically unusual—not harmful,
            causal or forecast to continue.
            """
        )

    vol = rolling_volatility(rates, window).reset_index().melt(id_vars="date", var_name="currency", value_name="volatility").dropna()
    vol_chart = px.line(
        vol,
        x="date",
        y="volatility",
        color="currency",
        title=f"Rolling {window}-day annualised volatility",
        labels={"date": "", "volatility": "Volatility · %", "currency": "Currency"},
    )
    st.plotly_chart(_style_figure(vol_chart, 470), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Portfolio structure</div>
          <h2>See common behavior.<br>Avoid hidden concentration.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    cluster_count = st.slider("Behavior groups", 2, min(5, len(selected)), min(3, len(selected)))
    correlation, groups = correlation_and_clusters(rates, cluster_count)
    allocation = inverse_volatility_allocation(rates, max(60, window))
    cor_col, alloc_col = st.columns([1, 1])
    with cor_col:
        heatmap = px.imshow(
            correlation,
            color_continuous_scale=["#e5484d", "#171a20", "#fcfcfd"],
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Daily-return correlation",
            labels={"color": "Correlation"},
        )
        st.plotly_chart(_style_figure(heatmap, 500), width="stretch")
    with alloc_col:
        allocation_chart = px.bar(
            allocation.sort_values("weight"),
            x="weight",
            y="currency",
            orientation="h",
            color="annualized_volatility",
            color_continuous_scale=["#fcfcfd", "#a7abb2", "#e5484d"],
            text_auto=".1f",
            title="Inverse-volatility reference allocation",
            labels={"weight": "Reference weight · %", "currency": "", "annualized_volatility": "Volatility · %"},
        )
        st.plotly_chart(_style_figure(allocation_chart, 500), width="stretch")
    cluster_text = " · ".join(
        f"Group {cluster}: {', '.join(group['currency'])}"
        for cluster, group in groups.groupby("cluster")
    )
    st.caption(
        f"Hierarchical correlation groups: {cluster_text}. The allocation gives lower "
        "weights to recently volatile currencies; it ignores expected return, transaction "
        "costs, liabilities and correlation in the final weights."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Exposure lab</div>
          <h2>Translate a market move<br>into an amount you understand.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3)
    with s1:
        scenario_currency = st.selectbox("Scenario currency", selected)
    with s2:
        eur_amount = st.number_input("EUR exposure", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=500.0)
    with s3:
        shock = st.slider("Hypothetical rate shock", -20.0, 20.0, 5.0, 0.5, format="%+.1f%%")
    latest_rate = float(rates[scenario_currency].dropna().iloc[-1])
    scenario = shock_scenario(latest_rate, eur_amount, shock)
    x1, x2, x3 = st.columns(3)
    x1.metric("Current quoted rate", f"{scenario['current_rate']:,.4f}")
    x2.metric("Shocked quoted rate", f"{scenario['shocked_rate']:,.4f}", f"{shock:+.1f}%")
    x3.metric(
        f"Value change in {scenario_currency}",
        f"{scenario['foreign_value_change']:+,.2f}",
        f"from {scenario['current_foreign_value']:,.2f}",
    )
    st.caption(
        "A positive shock means more foreign-currency units per euro. This is a static "
        "sensitivity calculation, not a forecast, recommendation or hedging instruction."
    )

    audit = summary.merge(groups, on="currency", how="left").merge(
        allocation[["currency", "weight"]], on="currency", how="left"
    )
    audit.columns = [
        "Currency", "Latest rate", "30d change", "Annualised volatility",
        "Historical VaR 95", "Worst day", "Maximum drawdown", "Observations",
        "Behavior group", "Inverse-volatility weight",
    ]
    st.markdown("#### Currency risk audit")
    st.dataframe(
        audit.style.format(
            {
                "Latest rate": "{:,.4f}", "30d change": "{:+.2f}%",
                "Annualised volatility": "{:.2f}%", "Historical VaR 95": "{:.2f}%",
                "Worst day": "{:.2f}%", "Maximum drawdown": "{:.2f}%",
                "Inverse-volatility weight": "{:.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download FX risk audit",
        audit.to_csv(index=False).encode("utf-8"),
        "fx_market_regime_risk_audit.csv",
        "text/csv",
        width="stretch",
    )
    link_a, link_b, link_c = st.columns(3)
    link_a.link_button("ECB exchange-rate dataset", DATASET_URL, width="stretch")
    link_b.link_button("ECB Data API", API_DOCS_URL, width="stretch")
    link_c.link_button("ECB reuse policy", USAGE_POLICY_URL, width="stretch")
    st.caption(
        f"Source: ECB statistics · retrieved {metadata['retrieved_at']} · "
        f"source mode: {metadata['mode']} · reference period through {metadata['end_date']}."
    )
