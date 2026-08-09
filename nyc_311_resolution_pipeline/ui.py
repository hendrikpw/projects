"""Responsive operational UI for the NYC 311 data and AI lifecycle."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nyc_311_resolution_pipeline.src.data import API_DOCS_URL, DATASET_URL, OPEN_DATA_URL
from nyc_311_resolution_pipeline.src.model import ModelBundle, score_case, train_and_evaluate
from nyc_311_resolution_pipeline.src.pipeline import PipelineBundle, run_pipeline


ORANGE, BLUE = "#fb923c", "#38bdf8"
COLORS = [ORANGE, BLUE, "#a78bfa", "#34d399", "#f87171", "#facc15"]


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=18, r=18, t=62, b=24), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"), colorway=COLORS, legend=dict(orientation="h", y=1.04, title=None),
        hoverlabel=dict(bgcolor="#161524", font_color="#fcfcfd", bordercolor="#58536b"),
    )
    fig.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return fig


@st.cache_data(ttl=21_600, show_spinner=False)
def _pipeline(history_days: int, sample_remainders: int) -> PipelineBundle:
    return run_pipeline(history_days, sample_remainders)


@st.cache_resource(show_spinner=False)
def _model(gold_hash: str, _gold: pd.DataFrame) -> ModelBundle:
    return train_and_evaluate(_gold)


def _manifest(data: PipelineBundle, model: ModelBundle) -> bytes:
    return json.dumps({
        "pipeline": data.metadata, "stages": data.stages.to_dict("records"),
        "quality": data.quality.to_dict("records"), "model": {**model.metadata, **model.metrics},
    }, indent=2, default=str).encode()


def render_dashboard() -> None:
    st.markdown("""
    <style>
    .ops-stage{padding:1rem;border-top:3px solid #fb923c;background:rgba(255,255,255,.035);min-height:132px}
    .ops-stage b{font-size:.82rem;color:#fcfcfd}.ops-stage p{font-size:.78rem;margin:.55rem 0 0}
    .ops-boundary{padding:1rem 1.15rem;border:1px solid rgba(251,146,60,.38);background:rgba(251,146,60,.08);border-radius:4px}
    </style>
    <section class="page-hero">
      <div class="eyebrow">18 / Data + AI engineering</div>
      <h1>NYC 311 Resolution<br>Operations Pipeline.</h1>
      <p>Contract mature civic-service records, quantify delivery uncertainty and estimate intake-time resolution ranges.</p>
      <div class="source-line">NYC Open Data · Content-addressed layers · Calibrated quantile regression</div>
    </section>
    """, unsafe_allow_html=True)
    controls = st.columns([1, 1, 1])
    history = controls[0].select_slider("History window", [270, 365, 450, 540], value=365, format_func=lambda x: f"{x} days")
    sample = controls[1].select_slider("Deterministic sample", [1, 2, 3], value=2, format_func=lambda x: f"{x}/1999 keys")
    agency_filter = controls[2].multiselect("Operational view", ["All agencies"], default=["All agencies"], disabled=True)
    try:
        with st.spinner("Extracting mature requests, validating contracts and backtesting resolution quantiles…"):
            data = _pipeline(history, sample)
            model = _model(data.metadata["gold_hash"], data.gold)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        st.error("The run stopped before publishing predictions because a data or model contract failed.")
        st.caption(f"Failure state · {type(exc).__name__}: {exc}")
        st.info("Use the default history and sample. Failed runs never expose an older cached prediction as current output.")
        return
    if data.metadata["mode"] == "demo":
        st.warning("NYC Open Data was unavailable or incomplete. The full product is running on deterministic source-shaped demo records.")
        st.caption("Fallback reason: " + data.metadata["fallback_reason"])
    else:
        st.success(f"Live NYC Open Data sample · {len(data.silver):,} valid mature requests · {data.metadata['start_date']} to {data.metadata['end_date']}")
    cards = st.columns(7)
    vals = [
        ("Silver requests", f"{len(data.silver):,}"), ("Agencies", f"{data.metadata['agency_count']:,}"),
        ("Complaint types", f"{data.metadata['complaint_count']:,}"), ("Quarantine", f"{len(data.quarantine):,}"),
        ("DQ pass", f"{data.metadata['quality_pass_rate']:.0%}"), ("Median-model MAE", f"{model.metrics['mae_hours']:.1f} h"),
        ("90% upper coverage", f"{model.metrics['upper_coverage']:.1%}"),
    ]
    for col, (label, value) in zip(cards, vals): col.metric(label, value)
    st.caption(f"Run {data.metadata['run_id']} · {model.metadata['training_rows']} train / {model.metadata['calibration_rows']} calibration / {model.metadata['test_rows']} test rows")
    st.markdown('<div class="ops-boundary"><b>Decision boundary:</b> This estimates historical resolution time for similar closed requests. It does not set an official SLA, prove service quality, prioritize emergencies or represent still-open requests.</div>', unsafe_allow_html=True)

    st.markdown('<section class="section-intro"><div class="section-kicker">Data engineering control plane</div><h2>Sample deterministically.<br>Publish contractually.</h2></section>', unsafe_allow_html=True)
    stage_cols = st.columns(4)
    stage_copy = [
        ("01 · EXTRACT", "Key-modulo sampling spreads a bounded pull across the mature label window; pagination, timeout and retry are explicit."),
        ("02 · BRONZE", "Canonical source records receive SHA-256 payload hashes and stable ingestion sequence metadata."),
        ("03 · SILVER", "Typed timestamps and categorical contracts deduplicate keys; invalid labels enter reason-coded quarantine."),
        ("04 · GOLD", "Only fields observable when a request is created become model features; closure fields remain labels."),
    ]
    for col, (title, body) in zip(stage_cols, stage_copy): col.markdown(f'<div class="ops-stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, .85])
    with left:
        fig = px.bar(data.stages, x="stage", y="output_rows", text="output_rows", color="stage", title="Layer volume, rejection and content lineage", hover_data=["input_rows", "rejected_rows", "duration_ms", "content_hash"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(_style(fig), width="stretch")
    with right:
        st.markdown("#### Immutable stage ledger")
        st.dataframe(data.stages, hide_index=True, width="stretch")
    ql, qr = st.columns([.85, 1.15])
    with ql:
        qa = data.quality.assign(result=data.quality["passed"].map({True: "Passed", False: "Failed"}))
        st.markdown("#### Contract and quality gates")
        st.dataframe(qa[["check", "result", "detail"]], hide_index=True, width="stretch")
    with qr:
        volume = data.gold.groupby(["created_date", "agency"], as_index=False).agg(requests=("unique_key", "size"), median_hours=("resolution_hours", "median"))
        top = volume.groupby("agency")["requests"].sum().nlargest(8).index
        fig = px.area(volume[volume["agency"].isin(top)], x="created_date", y="requests", color="agency", title="Deterministically sampled request flow", labels={"created_date":"", "requests":"Requests", "agency":""})
        st.plotly_chart(_style(fig), width="stretch")
    inspect = st.radio("Inspect data product", ["Silver contract", "Gold feature view", "Quarantine"], horizontal=True)
    if inspect == "Silver contract": preview = data.silver[["unique_key", "created_at", "closed_at", "agency", "complaint_type", "borough", "resolution_hours"]].tail(18)
    elif inspect == "Gold feature view": preview = data.gold[["unique_key", "agency", "complaint_type", "created_hour", "created_dow", "is_weekend", "resolution_hours"]].tail(18)
    else: preview = data.quarantine if not data.quarantine.empty else pd.DataFrame({"state":["No quarantined records"]})
    st.dataframe(preview, hide_index=True, width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">AI engineering lifecycle</div><h2>Estimate the middle.<br>Calibrate the tail.</h2></section>', unsafe_allow_html=True)
    metrics = st.columns(7)
    items = [
        ("MAE", f"{model.metrics['mae_hours']:.1f} h"), ("Median AE", f"{model.metrics['median_ae_hours']:.1f} h"),
        ("RMSE", f"{model.metrics['rmse_hours']:.1f} h"), ("Group baseline", f"{model.metrics['baseline_mae_hours']:.1f} h"),
        ("Skill", f"{model.metrics['skill_vs_group_median']:+.1%}"), ("Upper coverage", f"{model.metrics['upper_coverage']:.1%}"),
        ("≤24h accuracy", f"{model.metrics['within_24h_accuracy']:.1%}"),
    ]
    for col, (label, value) in zip(metrics, items): col.metric(label, value)
    if model.metrics["skill_vs_group_median"] <= 0:
        st.warning("Promotion gate failed: the candidate median model did not beat the train-only group-median baseline. The simulator remains an evaluation aid and should not be promoted as the production default.")
    else:
        st.success("Promotion gate passed: the candidate median model beat the train-only group-median baseline on the untouched holdout.")
    st.caption(f"Train ends {model.metadata['training_end']}; 45 days calibrate the upper quantile; {model.metadata['test_start']}–{model.metadata['test_end']} is untouched holdout. Calibration correction: {model.metadata['upper_correction_hours']:.1f} hours.")
    eval_left, eval_right = st.columns([1.2, .8])
    with eval_left:
        plot = model.predictions.sample(min(600, len(model.predictions)), random_state=42).sort_values("resolution_hours")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot["predicted_median_hours"], y=plot["resolution_hours"], mode="markers", marker=dict(color=plot["upper_breach"].map({False: BLUE, True: "#f87171"}), opacity=.66), name="Requests"))
        maximum = float(max(plot["resolution_hours"].quantile(.97), plot["predicted_median_hours"].quantile(.97)))
        fig.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], mode="lines", line=dict(color=ORANGE, dash="dash"), name="Perfect median"))
        fig.update_layout(title="Untouched holdout: expected vs observed resolution", xaxis_title="Predicted median hours", yaxis_title="Actual hours")
        st.plotly_chart(_style(fig, 460), width="stretch")
    with eval_right:
        st.markdown("#### Agency scorecard")
        st.dataframe(model.scorecard.round(3), hide_index=True, width="stretch")
    dist_left, dist_right = st.columns([1.1, .9])
    with dist_left:
        risk = model.predictions.groupby(["agency", "risk_band"], observed=True).size().reset_index(name="requests")
        fig = px.bar(risk, x="agency", y="requests", color="risk_band", title="Predicted upper-bound mix by agency", labels={"agency":"", "requests":"Holdout requests", "risk_band":"Risk band"})
        st.plotly_chart(_style(fig), width="stretch")
    with dist_right:
        st.markdown("#### Input drift monitor")
        st.dataframe(model.drift.round(4), hide_index=True, width="stretch")
        st.caption("PSI below 0.10 is stable, 0.10–0.25 deserves watching, and values above 0.25 require investigation—not automatic retraining.")

    st.markdown('<section class="section-intro"><div class="section-kicker">Intake simulator</div><h2>Describe a request.<br>Inspect uncertainty.</h2></section>', unsafe_allow_html=True)
    simulator = st.columns(4)
    agencies = sorted(data.gold["agency"].value_counts().head(12).index)
    agency = simulator[0].selectbox("Agency", agencies)
    agency_rows = data.gold[data.gold["agency"].eq(agency)]
    complaint = simulator[1].selectbox("Complaint", sorted(agency_rows["complaint_type"].value_counts().head(20).index))
    subset = agency_rows[agency_rows["complaint_type"].eq(complaint)]
    descriptor = simulator[2].selectbox("Descriptor", sorted(subset["descriptor"].value_counts().head(20).index))
    borough = simulator[3].selectbox("Borough", sorted(data.gold["borough"].unique()))
    context = st.columns(4)
    location = context[0].selectbox("Location type", sorted(subset["location_type"].value_counts().head(15).index))
    channel = context[1].selectbox("Channel", sorted(data.gold["open_data_channel_type"].unique()))
    hour = context[2].slider("Created hour", 0, 23, 10)
    dow = context[3].selectbox("Created weekday", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    result = score_case(model, data.gold, {"agency":agency,"complaint_type":complaint,"descriptor":descriptor,"location_type":location,"borough":borough,"open_data_channel_type":channel,"created_hour":hour,"created_dow":dow})
    outcome = st.columns(3)
    outcome[0].metric("Estimated median", f"{result['median_hours']:.1f} hours")
    outcome[1].metric("Calibrated upper estimate", f"{result['upper_hours']:.1f} hours")
    outcome[2].metric("Planning band", str(result["risk_band"]))
    st.caption("This is a historical similarity estimate, not a promise, official SLA or emergency-response recommendation. Rare combinations can be unreliable even when the encoder accepts them.")
    exports = st.columns(3)
    exports[0].download_button("Export Silver", data.silver.to_csv(index=False).encode(), f"nyc311_silver_{data.metadata['run_id']}.csv", "text/csv", width="stretch")
    exports[1].download_button("Export holdout audit", model.predictions.to_csv(index=False).encode(), f"nyc311_holdout_{data.metadata['run_id']}.csv", "text/csv", width="stretch")
    exports[2].download_button("Export run manifest", _manifest(data, model), f"nyc311_manifest_{data.metadata['run_id']}.json", "application/json", width="stretch")
    with st.expander("Method, source and production boundaries"):
        st.markdown(f"""
        **Source.** Mature closed requests are read from the [NYC 311 Service Requests dataset]({DATASET_URL}) through the [Socrata API]({API_DOCS_URL}). A deterministic numeric-key modulo rule samples the full historical window rather than only its newest rows.

        **Label maturity.** The window ends 35 days before retrieval and records over 30 days enter quarantine. This reduces—but cannot eliminate—right-censoring and closed-case selection bias.

        **AI lifecycle.** Two gradient-boosted quantile regressors estimate the conditional median and 90th percentile of `log1p(resolution_hours)`. Older data trains; a separate 45-day window calibrates the upper estimate; the newest 45 days are tested once. A train-only agency/complaint median is the explicit baseline.

        **Governance.** Inputs are limited to fields available when a request is created. Addresses, coordinates, resolution text, closure status and closure time are excluded. See the [NYC Open Data overview and policy context]({OPEN_DATA_URL}).
        """)
