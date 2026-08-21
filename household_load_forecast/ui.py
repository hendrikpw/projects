"""Streamlit control plane for the household load forecast product."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from household_load_forecast.src.data import DATASET_PAGE, DOI
from household_load_forecast.src.model import make_features, score_scenario, train_and_evaluate
from household_load_forecast.src.pipeline import run_pipeline

INK = "#07111f"; ELECTRIC = "#4de4ff"; VIOLET = "#9b7bff"; PINK = "#ff6db0"; AMBER = "#ffc857"; MINT = "#63f5c5"


def _style(fig, height: int = 410):
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=58, b=24), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", font=dict(color="#eef7ff", family="Inter, sans-serif"), legend=dict(orientation="h", y=1.14), colorway=[ELECTRIC, VIOLET, PINK, MINT, AMBER])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


@st.cache_resource(ttl=21600, show_spinner=False)
def _load():
    product = run_pipeline()
    return product, train_and_evaluate(product.gold)


def _manifest(product, model) -> bytes:
    return json.dumps({"pipeline": product.metadata, "quality": product.quality.to_dict("records"), "model": model.metadata, "metrics": model.metrics}, indent=2, default=str).encode()


def render_dashboard() -> None:
    st.markdown("""<style>
    .load-hero{padding:3.7rem 3rem;border-radius:28px;background:radial-gradient(circle at 88% 16%,rgba(77,228,255,.24),transparent 30%),radial-gradient(circle at 72% 92%,rgba(255,109,176,.17),transparent 34%),linear-gradient(145deg,#15223d,#07111f);border:1px solid rgba(77,228,255,.27);margin-bottom:1.4rem}.load-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#f7fbff;letter-spacing:-.06em;max-width:1050px;margin:.8rem 0}.load-hero p{max-width:900px;color:#b7c5d8;font-size:1.05rem}.kicker{color:#4de4ff;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.boundary{border-left:3px solid #ffc857;background:rgba(255,200,87,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:174px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #4de4ff;border-radius:0 0 14px 14px}.stage b{color:#4de4ff;font-size:.76rem;letter-spacing:.08em}.stage p{color:#b7c5d8;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#4de4ff;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}.route{padding:1rem;border:1px solid rgba(77,228,255,.25);border-radius:14px;background:rgba(77,228,255,.06)}</style>
    <section class="load-hero"><div class="kicker">UCI Power / Data + AI Engineering</div><h1>Forecast the load.<br>Publish the uncertainty.</h1><p>A replay-safe event-time pipeline contracts more than two million minute readings into an auditable hourly data product. A leakage-safe day-ahead model then backtests point forecasts, calibrated intervals, peak review and safe serving behavior.</p></section>""", unsafe_allow_html=True)
    try:
        with st.spinner("Downloading minute readings, reconciling hourly events and backtesting the forecast …"):
            product, model = _load()
    except Exception as exc:
        st.error("No forecast product was published because a source, data-quality or model-promotion gate failed."); st.exception(exc); return
    if product.metadata["mode"] == "demo": st.warning(f"Deterministic demonstration data are active: {product.metadata['fallback_reason']}")
    else: st.success("Official UCI archive loaded · ten data gates and all forecast promotion gates passed")
    values = [("Minute readings", f"{product.metadata['source_rows']:,}"), ("Gold hours", f"{len(product.gold):,}"), ("Day-ahead MAE", f"{model.metrics['mae_kw']:.3f} kW"), ("Baseline MAE", f"{model.metrics['baseline_mae_kw']:.3f} kW"), ("Interval coverage", f"{model.metrics['interval_coverage']:.1%}"), ("Peak capture", f"{model.metrics['peak_capture_at_10pct']:.1%}")]
    for column, (label, value) in zip(st.columns(6), values): column.metric(label, value)
    st.caption(f"Run {product.metadata['run_id']} · forecast horizon {model.metadata['horizon_hours']} h · train {model.metadata['train_hours']:,} / calibration {model.metadata['calibration_hours']:,} / test {model.metadata['test_hours']:,} hours · {model.metrics['inference_ms_per_hour']:.3f} ms per forecast")
    st.markdown('<div class="boundary"><b>Operational boundary:</b> this is a retrospective forecast for one French household, not a grid-control signal, tariff recommendation or representation of other homes. Intervals express empirical model uncertainty, not every possible operational risk.</div>', unsafe_allow_html=True)

    pipeline_tab, forecast_tab, serving_tab = st.tabs(["◫ Pipeline control", "⌁ Forecast evaluation", "◎ Serving workbench"])
    with pipeline_tab:
        st.markdown('<section class="section"><small>Data engineering</small><h2>Event time first.<br>Every hour reconciled.</h2></section>', unsafe_allow_html=True)
        cards = [("01 · SAFE EXTRACT", "Versioned ZIP, retry, timeout, byte bounds, signature, path protection and one-file allowlist."), ("02 · STREAM AGGREGATE", "Chunked parsing turns minute events into hourly sums and weighted means without retaining the 127 MB text file."), ("03 · WATERMARK", "Stable event and payload hashes, weekly micro-batches, 48-hour watermark and 24 intentional replays."), ("04 · CONTRACT", "Typed ranges, quarantine, short-gap interpolation, full reconciliation and ten fail-closed publication gates.")]
        for column, (title, body) in zip(st.columns(4), cards): column.markdown(f'<div class="stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, .85])
        with left:
            batch_view = product.batches.tail(80); fig = px.line(batch_view, x="batch_id", y=["deliveries", "unique_events"], markers=True, title="Latest micro-batches and replay behavior"); st.plotly_chart(_style(fig), width="stretch")
        with right: st.markdown("#### Layer ledger"); st.dataframe(product.stages, hide_index=True, width="stretch")
        left, right = st.columns(2)
        with left:
            checks = product.quality.assign(result=product.quality.passed.map({True: "Passed", False: "Failed"})); st.markdown("#### Publication gates"); st.dataframe(checks[["check", "result", "detail"]], hide_index=True, width="stretch")
        with right:
            daily = product.gold.set_index("timestamp").resample("D").agg(load_kw=("load_kw", "mean"), completeness=("completeness", "mean")).reset_index(); fig = px.line(daily, x="timestamp", y="load_kw", title="Contracted daily load history"); st.plotly_chart(_style(fig), width="stretch")
        if len(product.quarantine): st.markdown("#### Quarantine audit"); st.dataframe(product.quarantine[["timestamp", "event_id", "quarantine_reason"]].head(250), hide_index=True, width="stretch")
        else: st.info("The current run produced no contract quarantine. Source-level missing minutes remain visible in the extract ledger.")

    with forecast_tab:
        st.markdown('<section class="section"><small>AI engineering</small><h2>Backtest tomorrow.<br>Calibrate the range.</h2></section>', unsafe_allow_html=True)
        st.markdown("The model predicts the residual over same-hour persistence. Training, interval calibration and the final test are strictly chronological; every feature is known at issue time. The latest 15% of history is never used for fitting or policy selection.")
        recent = model.evaluation.tail(24 * 21)
        fig = go.Figure(); fig.add_trace(go.Scatter(x=recent.target_time, y=recent.upper, line=dict(width=0), showlegend=False)); fig.add_trace(go.Scatter(x=recent.target_time, y=recent.lower, fill="tonexty", fillcolor="rgba(77,228,255,.15)", line=dict(width=0), name="80% interval")); fig.add_trace(go.Scatter(x=recent.target_time, y=recent.actual, name="Actual", line=dict(color="#f7fbff", width=2))); fig.add_trace(go.Scatter(x=recent.target_time, y=recent.forecast, name="Forecast", line=dict(color=ELECTRIC, width=2))); st.plotly_chart(_style(fig, 470), width="stretch")
        left, right = st.columns(2)
        with left:
            errors = model.evaluation.assign(month=model.evaluation.target_time.dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(model_mae=("absolute_error", "mean")); baseline_error = model.evaluation.assign(error=lambda x: abs(x.actual - x.baseline), month=model.evaluation.target_time.dt.strftime("%Y-%m")).groupby("month", as_index=False).error.mean(); errors["persistence_mae"] = baseline_error.error; fig = px.line(errors, x="month", y=["model_mae", "persistence_mae"], markers=True, title="Monthly MAE versus same-hour persistence"); st.plotly_chart(_style(fig), width="stretch")
        with right:
            coverage = model.evaluation.assign(month=model.evaluation.target_time.dt.strftime("%Y-%m")).groupby("month", as_index=False).covered.mean(); fig = px.bar(coverage, x="month", y="covered", title="Empirical interval coverage by month", range_y=[0, 1]); fig.add_hline(y=model.metadata["nominal_coverage"], line_dash="dot", line_color=AMBER); st.plotly_chart(_style(fig), width="stretch")
        left, right = st.columns(2)
        with left: fig = px.bar(model.importance.head(12).sort_values("importance"), x="importance", y="feature", orientation="h", title="Permutation importance on the untouched test"); st.plotly_chart(_style(fig), width="stretch")
        with right: fig = px.bar(model.drift.head(12).sort_values("psi"), x="psi", y="feature", color="psi", orientation="h", title="Train-to-test population stability", color_continuous_scale=[INK, ELECTRIC, PINK]); fig.add_vline(x=.25, line_dash="dot", line_color=AMBER); st.plotly_chart(_style(fig), width="stretch")
        st.markdown("#### Held-out forecast audit"); st.dataframe(model.evaluation.tail(300), hide_index=True, width="stretch")

    with serving_tab:
        st.markdown('<section class="section"><small>Serving safety</small><h2>Stress the input.<br>Withhold unsafe forecasts.</h2></section>', unsafe_allow_html=True)
        feature_frame, _ = make_features(product.gold); available = feature_frame.tail(24 * 30).copy()
        selected = st.select_slider("Forecast issue time", options=list(available.timestamp), value=available.timestamp.iloc[-1], format_func=lambda value: value.strftime("%d %b %Y · %H:%M UTC"))
        left, right = st.columns(2)
        load_scale = left.slider("Recent-load scale", .60, 1.60, 1.00, .05, help="Simulates a unit, sensor or behavior shift across recent-load features.")
        missing_lags = right.slider("Missing lag features", 0, 6, 0, 1)
        row = available.loc[available.timestamp == selected].iloc[0]; result = score_scenario(model, row, load_scale=load_scale, missing_lags=missing_lags)
        columns = st.columns(5); columns[0].metric("Forecast", f"{result['forecast_kw']:.2f} kW"); columns[1].metric("Lower", f"{result['lower_kw']:.2f} kW"); columns[2].metric("Upper", f"{result['upper_kw']:.2f} kW"); columns[3].metric("OOD features", result["ood_features"]); columns[4].metric("Missing", f"{result['missing_share']:.0%}")
        st.markdown(f'<div class="route"><b>Serving route · {result["route"].upper()}</b><br>Auto-forecast requires complete inputs and fewer than four features outside the training envelope. Missing or shifted sensor states are reviewed or withheld.</div>', unsafe_allow_html=True)
        probabilities = pd.DataFrame({"bound": ["Lower", "Forecast", "Upper"], "load_kw": [result["lower_kw"], result["forecast_kw"], result["upper_kw"]]}); fig = px.bar(probabilities, x="bound", y="load_kw", color="bound", title="Scenario forecast envelope", color_discrete_sequence=[VIOLET, ELECTRIC, PINK]); st.plotly_chart(_style(fig, 350), width="stretch")

    st.markdown('<section class="section"><small>Audit and provenance</small><h2>Export the evidence.</h2></section>', unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    left.download_button("Download run manifest", _manifest(product, model), "household_load_manifest.json", "application/json", width="stretch")
    middle.download_button("Download forecast audit", model.evaluation.to_csv(index=False).encode(), "household_load_backtest.csv", "text/csv", width="stretch")
    right.download_button("Download quality ledger", product.quality.to_csv(index=False).encode(), "household_load_quality.csv", "text/csv", width="stretch")
    st.caption(f"Source: UCI Individual Household Electric Power Consumption · {product.gold.timestamp.min():%d %b %Y}–{product.gold.timestamp.max():%d %b %Y} · CC BY 4.0 · [dataset]({DATASET_PAGE}) · [DOI]({DOI})")
