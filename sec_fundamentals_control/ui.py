"""Modern Streamlit control plane for SEC XBRL fundamentals and anomaly review."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sec_fundamentals_control.src.data import API_DOCS, FAIR_ACCESS
from sec_fundamentals_control.src.model import FEATURES, LABELS, score_scenario, train_and_evaluate
from sec_fundamentals_control.src.pipeline import run_pipeline

INK = "#071015"; MINT = "#79f2c0"; BLUE = "#66a6ff"; AMBER = "#ffbf69"; RED = "#ff6b7a"


def _style(figure, height=410):
    figure.update_layout(height=height, margin=dict(l=18, r=18, t=62, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", font=dict(color="#edf7f4", family="Inter, sans-serif"), title_font=dict(size=18), legend=dict(orientation="h", y=1.13), colorway=[MINT, BLUE, AMBER, RED])
    figure.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False); figure.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return figure


@st.cache_resource(ttl=21_600, show_spinner=False)
def _load():
    data = run_pipeline(); model = train_and_evaluate(data.gold); return data, model


def _manifest(data, model):
    return json.dumps({"run": data.metadata, "model": model.metadata, "metrics": model.metrics, "quality": data.quality.to_dict("records")}, indent=2, default=str).encode()


def render_dashboard():
    st.markdown("""<style>.sec-hero{padding:3.4rem 3rem;border-radius:28px;background:radial-gradient(circle at 90% 10%,rgba(121,242,192,.2),transparent 31%),radial-gradient(circle at 92% 90%,rgba(102,166,255,.18),transparent 30%),linear-gradient(140deg,#0c1a20,#05090c);border:1px solid rgba(121,242,192,.2);margin-bottom:1.4rem}.sec-kicker{color:#79f2c0;font-size:.75rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.sec-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#f6fffc;letter-spacing:-.065em;max-width:920px;margin:.8rem 0}.sec-hero p{max-width:790px;color:#aebfba;font-size:1.05rem}.sec-boundary{border-left:3px solid #ffbf69;background:rgba(255,191,105,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0;color:#dce9e5}.sec-stage{min-height:170px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #79f2c0;border-radius:0 0 14px 14px}.sec-stage b{color:#79f2c0;font-size:.76rem;letter-spacing:.08em}.sec-stage p{color:#adbbb7;font-size:.9rem}.sec-section{padding-top:2.8rem}.sec-section small{color:#79f2c0;letter-spacing:.15em;text-transform:uppercase}.sec-section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style>""", unsafe_allow_html=True)
    st.markdown("""<section class="sec-hero"><div class="sec-kicker">SEC XBRL / Data + AI Engineering</div><h1>Trace the fact.<br>Review the break.</h1><p>A revision-aware fundamentals pipeline reconciles standardized filings before a peer-local anomaly model surfaces unusual financial combinations—with stress evaluation, drift and every accession visible.</p></section>""", unsafe_allow_html=True)
    try:
        with st.spinner("Loading eight SEC Company Facts feeds, resolving revisions and evaluating peer-local anomalies …"):
            data, model = _load()
    except Exception as exc:
        st.error("No analytical release was published because a data or model gate failed."); st.exception(exc); return
    if data.metadata["mode"] == "demo": st.warning(f"Deterministic demonstration filings are active: {data.metadata['fallback_reason']}")
    else: st.success("Live SEC Company Facts verified · latest revisions selected · all publication and model gates passed")
    columns = st.columns(6); values = [
        ("Company-quarters", f"{len(data.gold):,}"), ("Current XBRL facts", f"{len(data.silver):,}"),
        ("Superseded facts", f"{data.metadata['superseded_facts']:,}"), ("Stress AP", f"{model.metrics['stress_average_precision']:.3f}"),
        ("Baseline AP", f"{model.metrics['baseline_average_precision']:.3f}"), ("Natural review", f"{model.metrics['natural_review_count']:,}"),
    ]
    for column, (label, value) in zip(columns, values): column.metric(label, value)
    st.caption(f"Run {data.metadata['run_id']} · {model.metadata['train_periods']} train · {model.metadata['calibration_periods']} calibrate · {model.metadata['test_periods']} test · {model.metadata['n_neighbors']} neighbors")
    st.markdown('<div class="sec-boundary"><b>Interpretation boundary:</b> “Anomalous” means locally unusual within this small technology-company benchmark. It does not mean fraud, error, future underperformance or investment risk. XBRL concepts, fiscal calendars and company economics remain different.</div>', unsafe_allow_html=True)

    st.markdown('<section class="sec-section"><small>Data engineering control plane</small><h2>Resolve revisions.<br>Preserve the filing trail.</h2></section>', unsafe_allow_html=True)
    stages = st.columns(4); descriptions = [
        ("01 · EXTRACT", "Eight bounded Company Facts JSON feeds with identified User-Agent, retry, timeout, rate pacing and atomic fallback."),
        ("02 · BRONZE", "Standardized USD facts retain CIK, taxonomy concept, calendar frame, period, form, accession and filing date."),
        ("03 · SILVER", "Typed contracts quarantine invalid facts; latest filing wins per company, frame and metric while revision counts remain."),
        ("04 · GOLD", "Revenue, income, assets and equity reconcile into one feature-ready company-quarter with hashes and accession lineage."),
    ]
    for column, (title, body) in zip(stages, descriptions): column.markdown(f'<div class="sec-stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95])
    with left:
        fig = px.bar(data.stages, x="stage", y="output_rows", color="stage", text="output_rows", hover_data=["input_bytes_or_rows", "rejected_rows", "duration_ms", "content_hash"], title="Layer volumes and content-addressed lineage"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig), width="stretch")
    with right: st.markdown("#### Stage ledger"); st.dataframe(data.stages, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        quality = data.quality.assign(result=data.quality.passed.map({True:"Passed", False:"Failed"})); st.markdown("#### Publication gates"); st.dataframe(quality[["check", "result", "detail"]], hide_index=True, width="stretch")
    with right:
        st.markdown("#### Selected facts by metric"); counts = data.silver.groupby("metric", as_index=False).agg(facts=("fact_id", "size"), revisions=("revision_count", lambda x: int((x-1).clip(lower=0).sum())), companies=("ticker", "nunique")); st.dataframe(counts, hide_index=True, width="stretch")
    inspect = st.selectbox("Inspect a governed company-quarter", [f"{row.ticker} · {row.frame}" for row in data.gold.sort_values(["calendar_year", "calendar_quarter", "ticker"], ascending=False).itertuples()])
    ticker, frame = inspect.split(" · "); audit = data.silver[(data.silver.ticker == ticker) & (data.silver.frame == frame)][["metric", "concept", "value", "filed", "form", "accession", "revision_count", "source_record_hash"]]
    st.dataframe(audit, hide_index=True, width="stretch")

    st.markdown('<section class="sec-section"><small>AI engineering evaluation</small><h2>Learn local peers.<br>Attack the detector first.</h2></section>', unsafe_allow_html=True)
    metrics = st.columns(7); values = [
        ("Stress AP", f"{model.metrics['stress_average_precision']:.3f}"), ("Baseline AP", f"{model.metrics['baseline_average_precision']:.3f}"),
        ("Stress ROC-AUC", f"{model.metrics['stress_roc_auc']:.3f}"), ("Recall @ 10%", f"{model.metrics['recall_at_10pct']:.1%}"),
        ("Baseline @ 10%", f"{model.metrics['baseline_recall_at_10pct']:.1%}"), ("Calibration alerts", f"{model.metrics['calibration_false_alert_rate']:.1%}"),
        ("Observed alerts", f"{model.metrics['natural_review_rate']:.1%}"),
    ]
    for column, (label, value) in zip(metrics, values): column.metric(label, value)
    left, right = st.columns(2)
    with left:
        ranked = model.stress_evaluation.sort_values("candidate_score", ascending=False).reset_index(drop=True); ranked["review_share"] = (ranked.index+1)/len(ranked); ranked["captured_stress"] = ranked.label.cumsum()/ranked.label.sum()
        base = model.stress_evaluation.sort_values("baseline_score", ascending=False).reset_index(drop=True); base["review_share"] = (base.index+1)/len(base); base["captured_stress"] = base.label.cumsum()/base.label.sum()
        fig = go.Figure(); fig.add_scatter(x=ranked.review_share, y=ranked.captured_stress, name="Peer-local LOF"); fig.add_scatter(x=base.review_share, y=base.captured_stress, name="Max robust distance", line=dict(dash="dash")); fig.add_vline(x=.10, line_dash="dot"); fig.update_layout(title="Controlled-stress capture by review budget", xaxis_title="Share reviewed", yaxis_title="Injected stresses captured"); st.plotly_chart(_style(fig), width="stretch")
    with right:
        fig = px.bar(model.drift.sort_values("psi"), x="psi", y="label", orientation="h", color="status", color_discrete_map={"stable":MINT,"watch":AMBER,"high":RED}, title="Train-to-test feature drift · PSI"); fig.add_vline(x=.10, line_dash="dot"); fig.add_vline(x=.25, line_dash="dash"); st.plotly_chart(_style(fig), width="stretch")
    left, right = st.columns(2)
    with left:
        fig = px.scatter(model.evaluation, x="revenue_growth_yoy", y="net_margin", color="status", size="anomaly_score", hover_name="ticker", hover_data=["frame", "liability_ratio", "asset_turnover_quarterly"], color_discrete_map={"monitor":BLUE,"review":RED}, title="Untouched recent holdout · local review queue"); st.plotly_chart(_style(fig, 480), width="stretch")
    with right:
        st.markdown("#### Held-out review queue"); queue = model.evaluation.sort_values("anomaly_score", ascending=False)[["ticker", "frame", "status", "anomaly_score", "revenue_growth_yoy", "net_margin", "liability_ratio"]]; st.dataframe(queue, hide_index=True, width="stretch")
    st.caption("Promotion uses deterministic controlled perturbations only on the untouched test rows. Each stress moves two robust-scaled features by 2.5 and 2.0 units. This validates detector sensitivity and ranking—not the correctness of alerts in real filings.")

    st.markdown('<section class="sec-section"><small>Scenario workbench</small><h2>Move the fundamentals.<br>Inspect the local evidence.</h2></section>', unsafe_allow_html=True)
    company = st.selectbox("Company", sorted(data.gold.ticker.unique()), index=0); row = data.gold[(data.gold.ticker == company) & data.gold[FEATURES].notna().all(axis=1)].sort_values(["calendar_year", "calendar_quarter"]).iloc[-1]
    controls = st.columns(3)
    growth_delta = controls[0].slider("Revenue-growth change (decimal)", -0.50, 0.50, 0.0, 0.01, format="%.2f")
    margin_delta = controls[1].slider("Net-margin change (decimal)", -0.30, 0.30, 0.0, 0.01, format="%.2f")
    leverage_delta = controls[2].slider("Liability-ratio change (decimal)", -0.30, 0.50, 0.0, 0.01, format="%.2f")
    result = score_scenario(model, row, {"revenue_growth_yoy":growth_delta, "net_margin":margin_delta, "liability_ratio":leverage_delta})
    columns = st.columns(4); columns[0].metric("Source frame", row.frame); columns[1].metric("Scenario score", f"{result['anomaly_score']:.3f}"); columns[2].metric("Review threshold", f"{model.metrics['threshold']:.3f}"); columns[3].metric("Route", result["status"].title())
    if result["status"] == "review": st.warning("The scenario crosses the calibration threshold and enters analyst review. It is not an accusation or trading signal.")
    else: st.info("The scenario remains below the review threshold. This does not prove that the filing is correct or low-risk.")
    left, right = st.columns([1.1, .9]);
    with left:
        fig = px.bar(result["evidence"].sort_values("local_deviation"), x="local_deviation", y="feature", orientation="h", title="Local deviation from eight nearest historical peers", color="local_deviation", color_continuous_scale=[[0, BLUE], [1, RED]]); st.plotly_chart(_style(fig, 450), width="stretch")
    with right: st.markdown("#### Nearest reference quarters"); st.dataframe(result["peers"], hide_index=True, width="stretch")
    downloads = st.columns(3); downloads[0].download_button("Export Gold product", data.gold.to_csv(index=False).encode(), f"sec_gold_{data.metadata['run_id']}.csv", "text/csv", width="stretch"); downloads[1].download_button("Export model audit", model.evaluation.to_csv(index=False).encode(), f"sec_model_audit_{data.metadata['run_id']}.csv", "text/csv", width="stretch"); downloads[2].download_button("Export manifest", _manifest(data, model), f"sec_manifest_{data.metadata['run_id']}.json", "application/json", width="stretch")
    with st.expander("Source, XBRL semantics, evaluation and limitations"):
        st.markdown(f"""**Source.** The application uses the official [SEC EDGAR Company Facts API]({API_DOCS}) at `data.sec.gov/api/xbrl/companyfacts/CIK##########.json`. SEC states that these JSON APIs require no authentication or API key and update throughout the day as submissions are disseminated. Automated access follows the [SEC developer and fair-access guidance]({FAIR_ACCESS}) with an identifying User-Agent, sequential requests, cache and bounded retries.

**Fields.** CIK, entity name, US-GAAP taxonomy concept, USD unit, calendar frame, period end, filing date, accession number, form and value. The product selects revenue, net income, assets and stockholders' equity. Implied liabilities equal assets minus equity; ratios are analytical transformations, not SEC-published metrics.

**Revisions.** For each company, calendar frame and metric, the latest filed fact is selected. Prior facts remain counted in lineage. Calendar frames improve comparability but do not erase different fiscal calendars or accounting policies.

**Model.** Robust scaling is fitted on earlier quarters. Local Outlier Factor learns dense historical peer neighborhoods. The next eight reported frames set the 95th-percentile review threshold; the latest eight form an untouched test. Controlled stress injections and a max-robust-distance baseline quantify ranking behavior.

**Limits.** Eight technology companies are not the market. The model is unsupervised and has no fraud labels. High PSI indicates the recent period differs from training and should prompt investigation, not automatic retraining. SEC provides public filing data but does not endorse this application. This is educational software, not accounting, legal or investment advice.""")
