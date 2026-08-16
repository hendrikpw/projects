"""Modern operational Streamlit interface for river-flow early warning."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from river_flow_early_warning.src.data import DOCS, RIGHTS, WDFN
from river_flow_early_warning.src.model import score_latest, train_and_evaluate
from river_flow_early_warning.src.pipeline import run_pipeline

NAVY = "#06151e"; CYAN = "#56d6ff"; AQUA = "#55efc4"; AMBER = "#ffc857"; RED = "#ff6178"


def _style(figure, height=410):
    figure.update_layout(height=height, margin=dict(l=18, r=18, t=62, b=25), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(255,255,255,.025)", font=dict(color="#edf8fb", family="Inter, sans-serif"),
                         legend=dict(orientation="h", y=1.13), colorway=[CYAN, AQUA, AMBER, RED])
    figure.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return figure


@st.cache_resource(ttl=21_600, show_spinner=False)
def _load():
    product = run_pipeline()
    return product, train_and_evaluate(product.gold)


def _manifest(product, model):
    return json.dumps({"pipeline": product.metadata, "model": model.metadata, "metrics": model.metrics,
                       "quality": product.quality.to_dict("records")}, indent=2, default=str).encode()


def render_dashboard():
    st.markdown("""<style>
    .river-hero{padding:3.5rem 3rem;border-radius:28px;background:radial-gradient(circle at 88% 15%,rgba(86,214,255,.22),transparent 30%),radial-gradient(circle at 85% 90%,rgba(85,239,196,.13),transparent 32%),linear-gradient(145deg,#0a2633,#041017);border:1px solid rgba(86,214,255,.22);margin-bottom:1.4rem}.river-kicker{color:#56d6ff;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.river-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#f4fcff;letter-spacing:-.06em;max-width:940px;margin:.8rem 0}.river-hero p{max-width:800px;color:#afc5cd;font-size:1.05rem}.river-boundary{border-left:3px solid #ffc857;background:rgba(255,200,87,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0;color:#dbe9ed}.river-stage{min-height:170px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #56d6ff;border-radius:0 0 14px 14px}.river-stage b{color:#56d6ff;font-size:.76rem;letter-spacing:.08em}.river-stage p{color:#adbec5;font-size:.9rem}.river-section{padding-top:2.8rem}.river-section small{color:#56d6ff;letter-spacing:.15em;text-transform:uppercase}.river-section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style>""", unsafe_allow_html=True)
    st.markdown("""<section class="river-hero"><div class="river-kicker">USGS water data / Data + AI Engineering</div><h1>Trace every reading.<br>Anticipate the rise.</h1><p>A replay-safe streamflow pipeline turns public gauge observations into a governed daily product, then tests a calibrated three-day high-flow warning model against seasonal climatology.</p></section>""", unsafe_allow_html=True)
    try:
        with st.spinner("Loading six USGS gauges, reconciling deliveries and evaluating the warning model …"):
            product, model = _load()
    except Exception as exc:
        st.error("No warning product was published because a data or model gate failed.")
        st.exception(exc)
        return
    if product.metadata["mode"] == "demo":
        st.warning(f"Deterministic demonstration readings are active: {product.metadata['fallback_reason']}")
    else:
        st.success(f"Live USGS daily values verified through {product.metadata['current_through']} · every publication gate passed")
    columns = st.columns(6)
    values = [("Gauge-days", f"{len(product.silver):,}"), ("Gauges", "6"),
              ("Replays suppressed", f"{product.metadata['duplicate_deliveries']:,}"),
              ("Model AP", f"{model.metrics['average_precision']:.3f}"),
              ("Climatology AP", f"{model.metrics['baseline_average_precision']:.3f}"),
              ("Recall @ 10%", f"{model.metrics['recall_at_10pct']:.1%}")]
    for column, (label, value) in zip(columns, values): column.metric(label, value)
    st.caption(f"Run {product.metadata['run_id']} · {model.metadata['train_period']} train · {model.metadata['calibration_period']} calibrate · {model.metadata['test_period']} test")
    st.markdown('<div class="river-boundary"><b>Decision boundary:</b> this is a portfolio early-warning model, not an official flood forecast. “High flow” means exceeding a gauge-specific training-period percentile—not flood stage, damage or personal danger. Use official USGS/NWS alerts for safety decisions.</div>', unsafe_allow_html=True)

    st.markdown('<section class="river-section"><small>Data engineering control plane</small><h2>Reconcile the stream.<br>Keep the evidence.</h2></section>', unsafe_allow_html=True)
    stages = st.columns(4)
    descriptions = [("01 · EXTRACT", "One bounded USGS RDB request with identified User-Agent, timeout, retry, response contract and atomic fallback."),
                    ("02 · BRONZE", "Station-day deliveries retain qualifier, event key, delivery key, monthly batch and source payload lineage."),
                    ("03 · SILVER", "Typed dates and discharge values, range validation, quarantine and replay-safe station-day deduplication."),
                    ("04 · GOLD", "Lag-only hydrology features, seasonal cycles and future three-day labels with source rows still traceable." )]
    for column, (title, body) in zip(stages, descriptions): column.markdown(f'<div class="river-stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95])
    with left:
        fig = px.bar(product.stages, x="stage", y="output", color="stage", text="output", hover_data=["input","rejected","duration_ms","content_hash"], title="Layer volumes and content-addressed lineage")
        fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig), width="stretch")
    with right:
        st.markdown("#### Stage ledger"); st.dataframe(product.stages, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        quality = product.quality.assign(result=product.quality.passed.map({True:"Passed",False:"Failed"}))
        st.markdown("#### Publication gates"); st.dataframe(quality[["check","result","detail"]], hide_index=True, width="stretch")
    with right:
        current = product.silver.sort_values("event_date").groupby(["site_no","site_name"], as_index=False).tail(1)
        current["freshness_days"] = (product.silver.event_date.max()-current.event_date).dt.days
        st.markdown("#### Latest governed readings"); st.dataframe(current[["site_name","event_date","discharge_cfs","qualifier","is_provisional","freshness_days"]], hide_index=True, width="stretch")
    map_data = current.copy(); map_data["size"] = np.log1p(map_data.discharge_cfs)
    fig = px.scatter_map(map_data, lat="latitude", lon="longitude", size="size", color="discharge_cfs", hover_name="site_name", hover_data={"discharge_cfs":":,.0f","event_date":True,"latitude":False,"longitude":False,"size":False}, color_continuous_scale="Tealgrn", zoom=2.6, title="Six governed streamflow stations · latest daily mean")
    st.plotly_chart(_style(fig, 480), width="stretch")

    st.markdown('<section class="river-section"><small>AI engineering evaluation</small><h2>Predict the threshold.<br>Audit the miss.</h2></section>', unsafe_allow_html=True)
    metrics = st.columns(7)
    values = [("Average precision", f"{model.metrics['average_precision']:.3f}"), ("Climatology", f"{model.metrics['baseline_average_precision']:.3f}"),
              ("ROC-AUC", f"{model.metrics['roc_auc']:.3f}"), ("Brier", f"{model.metrics['brier']:.3f}"),
              ("Recall @ 10%", f"{model.metrics['recall_at_10pct']:.1%}"), ("Baseline @ 10%", f"{model.metrics['baseline_recall_at_10pct']:.1%}"),
              ("Test event rate", f"{model.metrics['event_rate']:.1%}")]
    for column, (label, value) in zip(metrics, values): column.metric(label, value)
    left, right = st.columns(2)
    with left:
        ranked = model.evaluation.sort_values("ranking_score", ascending=False).reset_index(drop=True)
        ranked["review_share"] = (ranked.index+1)/len(ranked); ranked["event_capture"] = ranked.high_flow_next_3d.cumsum()/ranked.high_flow_next_3d.sum()
        baseline = model.evaluation.sort_values("baseline_score", ascending=False).reset_index(drop=True)
        baseline["review_share"] = (baseline.index+1)/len(baseline); baseline["event_capture"] = baseline.high_flow_next_3d.cumsum()/baseline.high_flow_next_3d.sum()
        fig = go.Figure(); fig.add_scatter(x=ranked.review_share, y=ranked.event_capture, name="Gradient boosting"); fig.add_scatter(x=baseline.review_share, y=baseline.event_capture, name="Seasonal climatology", line=dict(dash="dash")); fig.add_vline(x=.10, line_dash="dot")
        fig.update_layout(title="High-flow capture by review budget", xaxis_title="Share of station-days reviewed", yaxis_title="Future events captured"); st.plotly_chart(_style(fig), width="stretch")
    with right:
        calibration = model.evaluation.assign(bin=pd.cut(model.evaluation.probability, bins=np.linspace(0,1,11), include_lowest=True)).groupby("bin", observed=True).agg(predicted=("probability","mean"), observed=("high_flow_next_3d","mean"), rows=("probability","size")).reset_index()
        fig = go.Figure(); fig.add_scatter(x=[0,1], y=[0,1], name="Perfect", line=dict(dash="dash", color="#80939a")); fig.add_scatter(x=calibration.predicted, y=calibration.observed, mode="lines+markers", marker=dict(size=np.sqrt(calibration.rows)*2.2), name="Held-out bins")
        fig.update_layout(title="Calibrated probability reliability", xaxis_title="Predicted probability", yaxis_title="Observed event rate"); st.plotly_chart(_style(fig), width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Evaluation by gauge"); st.dataframe(model.station_metrics, hide_index=True, width="stretch")
    with right:
        fig = px.bar(model.drift.sort_values("psi"), x="psi", y="feature", orientation="h", color="status", color_discrete_map={"stable":AQUA,"watch":AMBER,"high":RED}, title="Train-to-test feature drift · PSI")
        fig.add_vline(x=.10, line_dash="dot"); fig.add_vline(x=.25, line_dash="dash"); st.plotly_chart(_style(fig, 480), width="stretch")
    queue = model.evaluation.sort_values("probability", ascending=False).head(50)
    st.markdown("#### Held-out warning audit"); st.dataframe(queue[["site_name","event_date","discharge_cfs","threshold_cfs","probability","status","high_flow_next_3d","future_max_3d"]], hide_index=True, width="stretch")
    st.caption("The threshold is the station's 90th percentile learned only from 2018–2023. Features end at the decision day. The label looks at the following three days and is used only for training/evaluation. The baseline knows station and month but no recent flow.")

    st.markdown('<section class="river-section"><small>Scenario workbench</small><h2>Stress the latest flow.<br>Inspect the route.</h2></section>', unsafe_allow_html=True)
    names = product.silver[["site_no","site_name"]].drop_duplicates().set_index("site_name").site_no.to_dict()
    selected_name = st.selectbox("Gauge", list(names)); multiplier = st.slider("Latest-flow multiplier", .25, 3.0, 1.0, .05)
    scenario = score_latest(model, product.gold, names[selected_name], multiplier)
    columns = st.columns(5); columns[0].metric("Decision date", scenario["event_date"].date().isoformat()); columns[1].metric("Scenario flow", f"{scenario['discharge_cfs']:,.0f} cfs"); columns[2].metric("High-flow reference", f"{scenario['threshold_cfs']:,.0f} cfs"); columns[3].metric("3-day probability", f"{scenario['probability']:.1%}"); columns[4].metric("Route", scenario["status"].title())
    history = product.silver[product.silver.site_no==names[selected_name]].sort_values("event_date").tail(365)
    fig = px.line(history, x="event_date", y="discharge_cfs", title=f"{selected_name} · latest 365 governed daily means"); fig.add_hline(y=scenario["threshold_cfs"], line_dash="dash", annotation_text="training P90")
    st.plotly_chart(_style(fig, 450), width="stretch")
    if scenario["status"] == "alert": st.warning("The scenario crosses the portfolio alert route. Verify current official observations, forecasts and local flood guidance.")
    elif scenario["status"] == "watch": st.info("The scenario enters watch. It remains a statistical signal rather than an official warning.")
    else: st.success("The scenario remains in the normal route. Low model probability is not a safety guarantee.")
    downloads = st.columns(3)
    downloads[0].download_button("Export governed observations", product.silver.to_csv(index=False).encode(), f"river_silver_{product.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[1].download_button("Export model audit", model.evaluation.to_csv(index=False).encode(), f"river_model_{product.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[2].download_button("Export manifest", _manifest(product, model), f"river_manifest_{product.metadata['run_id']}.json", "application/json", width="stretch")
    with st.expander("Source semantics, target, evaluation and safety limits"):
        st.markdown(f"""**Source.** Daily mean discharge comes from the official [USGS Daily Values service]({DOCS}) for six [Water Data for the Nation]({WDFN}) monitoring locations. Parameter `00060` is discharge in cubic feet per second and statistic `00003` is the daily mean. Historical and active-site data are public and keyless. USGS data are generally public domain under its [copyright and credits guidance]({RIGHTS}), while separately credited material can have other rights.

**Qualifiers.** `A` generally denotes approved data and `P` provisional data subject to revision; compound codes can carry further remarks. The pipeline preserves the complete qualifier rather than silently declaring every value final.

**Target.** A positive label means the maximum daily mean in the next three days exceeds that station's 2018–2023 90th percentile. That portfolio definition is not official flood stage. A station can have no positive event in a recent test segment, so per-station AP may be unavailable.

**Evaluation.** Training uses 2018–2023, 2024 calibrates probability with isotonic regression, and 2025 onward is untouched testing. The comparison baseline is station/month training climatology. Average precision measures rare-event ranking; Brier measures probability error; recall at 10% measures operational capture under a fixed review budget.

**Limits.** Daily means smooth intraday peaks. The model has no rainfall, reservoir operations, snowpack or upstream gauges. Provisional readings can change. Six gauges do not represent U.S. hydrology. PSI is a diagnostic, not an automatic retraining command. This educational product must never replace official flood forecasts, local emergency management or evacuation instructions.""")
