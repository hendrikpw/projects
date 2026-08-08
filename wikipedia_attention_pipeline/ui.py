"""Modern Streamlit control plane for attention data and forecasting operations."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from wikipedia_attention_pipeline.src.data import ARTICLES, DOCS_URL, TERMS_URL, USAGE_URL
from wikipedia_attention_pipeline.src.model import ModelBundle, train_and_evaluate
from wikipedia_attention_pipeline.src.pipeline import PipelineBundle, run_pipeline


ACCENT = "#a78bfa"
CYAN = "#22d3ee"
PALETTE = [ACCENT, CYAN, "#fb7185", "#fbbf24", "#fcfcfd"]


def _style(figure: go.Figure, height: int = 430) -> go.Figure:
    figure.update_layout(
        height=height, margin=dict(l=18, r=18, t=62, b=24), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"), colorway=PALETTE,
        legend=dict(orientation="h", y=1.03, title=None),
        hoverlabel=dict(bgcolor="#161524", font_color="#fcfcfd", bordercolor="#58536b"),
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=21_600, show_spinner=False)
def _pipeline(articles: tuple[str, ...], history_days: int, batch_size: int) -> PipelineBundle:
    return run_pipeline(list(articles), history_days, batch_size)


@st.cache_resource(show_spinner=False)
def _model(gold_hash: str, _gold: pd.DataFrame, _silver: pd.DataFrame, horizon: int) -> ModelBundle:
    return train_and_evaluate(_gold, _silver, horizon)


def _manifest(bundle: PipelineBundle, model: ModelBundle) -> str:
    return json.dumps({
        "pipeline": bundle.metadata, "stages": bundle.events.to_dict(orient="records"),
        "quality": bundle.quality.to_dict(orient="records"), "forecast": {**model.metadata, **model.metrics},
    }, indent=2, default=str)


def render_dashboard() -> None:
    st.markdown(
        """
        <style>
        .wiki-stage {padding:1rem;border-left:3px solid #a78bfa;background:rgba(255,255,255,.035);min-height:132px}
        .wiki-stage b {color:#fcfcfd;font-size:.86rem}.wiki-stage p {font-size:.78rem;margin:.6rem 0 0}
        .wiki-boundary {padding:1rem 1.15rem;border:1px solid rgba(167,139,250,.38);background:rgba(167,139,250,.08);border-radius:4px}
        </style>
        <section class="page-hero">
          <div class="eyebrow">17 / Data + AI engineering</div>
          <h1>Wikipedia Attention<br>Forecast Pipeline.</h1>
          <p>
            Replay public attention signals through event-time micro-batches,
            then forecast demand with calibrated uncertainty and anomaly review.
          </p>
          <div class="source-line">Wikimedia Analytics · Watermarks + dedupe · Conformal forecasting</div>
        </section>
        """, unsafe_allow_html=True,
    )

    labels = list(ARTICLES)
    controls = st.columns([1.65, 0.75, 0.75, 0.7])
    with controls[0]:
        chosen_labels = st.multiselect("Attention portfolio", labels, default=labels[:4])
    with controls[1]:
        history_days = st.select_slider("History", [120, 150, 180, 240, 300], value=180, format_func=lambda value: f"{value} days")
    with controls[2]:
        batch_size = st.select_slider("Micro-batch", [60, 90, 120, 180], value=120, format_func=lambda value: f"{value} events")
    with controls[3]:
        horizon = st.select_slider("Forecast", [7, 14, 21], value=14, format_func=lambda value: f"{value} days")
    articles = tuple(ARTICLES[label] for label in chosen_labels)
    if len(articles) < 2:
        st.info("Select at least two articles so the global forecast can be evaluated across multiple series.")
        return

    try:
        with st.spinner("Loading attention history, replaying event-time batches and running the rolling-origin forecast…"):
            bundle = _pipeline(articles, history_days, batch_size)
            model = _model(bundle.metadata["gold_hash"], bundle.gold, bundle.silver, horizon)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        st.error("The pipeline stopped before publishing a forecast because a data or model contract failed.")
        st.caption(f"Failure state · {type(exc).__name__}: {exc}")
        st.info("Select a longer history or the default four-article portfolio. Failed runs never expose stale predictions.")
        return

    if bundle.metadata["mode"] == "demo":
        st.warning("Wikimedia was unavailable or incomplete. Every control is running on a deterministic synthetic attention stream.")
        st.caption("Fallback reason: " + bundle.metadata["fallback_reason"])
    else:
        retrieved = pd.to_datetime(bundle.metadata["retrieved_at"], utc=True)
        st.success(
            f"Live Wikimedia snapshot · {len(bundle.silver):,} unique article-days · "
            f"{bundle.metadata['start_date']} to {bundle.metadata['end_date']} · retrieved {retrieved:%d %b %Y, %H:%M UTC}"
        )

    cards = st.columns(7)
    values = [
        ("Unique events", f"{len(bundle.silver):,}", None),
        ("Batches", f"{len(bundle.batches):,}", None),
        ("Replay duplicates", f"{bundle.metadata['duplicate_replays']:,}", None),
        ("Late events", f"{bundle.metadata['late_events']:,}", None),
        ("DQ pass", f"{bundle.metadata['quality_pass_rate']:.0%}", None),
        ("Forecast WAPE", f"{model.metrics['wape']:.1%}", "Lower is better"),
        ("90% interval coverage", f"{model.metrics['interval_coverage']:.1%}", "Observed holdout coverage"),
    ]
    for column, (label, value, help_text) in zip(cards, values):
        column.metric(label, value, help=help_text)
    st.caption(
        f"Run {bundle.metadata['run_id']} · model {model.metadata['model']} · "
        f"{model.metadata['training_rows']} train / {model.metadata['calibration_rows']} calibration / {model.metadata['test_rows']} test rows."
    )
    st.markdown(
        '<div class="wiki-boundary"><b>Interpretation boundary:</b> Pageviews measure observed attention, not sentiment, truth, importance or future events. '
        'Intervals quantify recent forecast error only; an anomaly is a review signal, never an explanation.</div>', unsafe_allow_html=True,
    )

    st.markdown(
        '<section class="section-intro"><div class="section-kicker">Data engineering control plane</div>'
        '<h2>Replay disorder.<br>Publish order.</h2></section>', unsafe_allow_html=True,
    )
    stage_columns = st.columns(4)
    stages = [
        ("01 · EXTRACT", "Bounded, sequential requests with a compliant user agent, timeout, retry budget and atomic fallback."),
        ("02 · BRONZE", "Source payloads receive stable hashes, event IDs, deterministic arrival delays and micro-batch IDs."),
        ("03 · SILVER", "Event-time watermarks expose lateness; duplicate replays and contract violations enter quarantine."),
        ("04 · GOLD", "Only shifted lags and rolling windows become forecast features, preventing future-value leakage."),
    ]
    for column, (title, body) in zip(stage_columns, stages):
        column.markdown(f'<div class="wiki-stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        flow = bundle.events.copy()
        figure = px.bar(
            flow, x="stage", y="output_rows", text="output_rows", color="status",
            color_discrete_map={"passed": ACCENT}, hover_data={"input_rows": True, "rejected_rows": True, "duration_ms": ":.2f", "content_hash": True},
            title="Stage volume, latency and lineage", labels={"stage": "", "output_rows": "Rows", "status": ""},
        )
        figure.update_traces(textposition="outside")
        st.plotly_chart(_style(figure, 440), width="stretch")
    with right:
        st.markdown("#### Immutable run ledger")
        st.dataframe(bundle.events, hide_index=True, width="stretch")
        st.caption("The run and layer hashes remain stable when identical source data is replayed, independent of retrieval time.")

    batch_left, batch_right = st.columns([1.15, 0.85])
    with batch_left:
        batch_metrics = bundle.batches.melt("batch_id", value_vars=["duplicate_events", "late_events"], var_name="signal", value_name="events")
        figure = px.bar(
            batch_metrics, x="batch_id", y="events", color="signal", barmode="stack",
            title="Operational exceptions by micro-batch", labels={"batch_id": "Batch", "events": "Events", "signal": ""},
        )
        st.plotly_chart(_style(figure, 400), width="stretch")
    with batch_right:
        qa = bundle.quality.assign(result=lambda frame: frame["passed"].map({True: "Passed", False: "Failed"}))
        st.markdown("#### Data quality gates")
        st.dataframe(qa[["check", "result", "detail"]], hide_index=True, width="stretch")

    inspect_left, inspect_right = st.columns([1.15, 0.85])
    with inspect_left:
        history = bundle.silver.copy()
        history["article"] = history["article"].str.replace("_", " ")
        figure = px.line(history, x="event_time", y="views", color="article", title="Validated attention history", labels={"event_time": "", "views": "Daily views", "article": ""})
        st.plotly_chart(_style(figure, 450), width="stretch")
    with inspect_right:
        layer = st.radio("Inspect layer", ["Silver contract", "Gold features", "Quarantine"], horizontal=True)
        if layer == "Silver contract":
            preview = bundle.silver[["article", "event_time", "views", "batch_id", "late_beyond_watermark"]].tail(16)
        elif layer == "Gold features":
            preview = bundle.gold[["article", "event_time", "views", "lag_7", "rolling_28", "rolling_std_28"]].tail(16)
        else:
            preview = bundle.quarantine if not bundle.quarantine.empty else pd.DataFrame({"state": ["No quarantined events"]})
        st.dataframe(preview, hide_index=True, width="stretch")
        st.download_button("Export Silver contract", bundle.silver.to_csv(index=False).encode(), f"wikipedia_silver_{bundle.metadata['run_id']}.csv", "text/csv", width="stretch")

    st.markdown(
        '<section class="section-intro"><div class="section-kicker">AI engineering lifecycle</div>'
        '<h2>Backtest first.<br>Forecast second.</h2></section>', unsafe_allow_html=True,
    )
    metric_columns = st.columns(7)
    metric_values = [
        ("MAE", f"{model.metrics['mae']:,.0f}"), ("RMSE", f"{model.metrics['rmse']:,.0f}"),
        ("WAPE", f"{model.metrics['wape']:.1%}"), ("SMAPE", f"{model.metrics['smape']:.1%}"),
        ("Weekly-naive WAPE", f"{model.metrics['baseline_wape']:.1%}"),
        ("Skill vs baseline", f"{model.metrics['skill_vs_weekly_naive']:+.1%}"),
        ("Interval coverage", f"{model.metrics['interval_coverage']:.1%}"),
    ]
    for column, (label, value) in zip(metric_columns, metric_values): column.metric(label, value)
    st.caption(
        f"Train ends {model.metadata['training_end']}; the next 28 days calibrate a 90% split-conformal interval; "
        f"the newest 28 days ({model.metadata['test_start']}–{model.metadata['test_end']}) remain untouched until evaluation."
    )

    eval_left, eval_right = st.columns([1.25, 0.75])
    with eval_left:
        selected_article = st.selectbox("Holdout series", sorted(model.predictions["article"].unique()), format_func=lambda value: value.replace("_", " "))
        sample = model.predictions[model.predictions["article"].eq(selected_article)]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=sample["event_time"], y=sample["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=sample["event_time"], y=sample["lower"], fill="tonexty", fillcolor="rgba(167,139,250,.16)", line=dict(width=0), name="90% interval"))
        figure.add_trace(go.Scatter(x=sample["event_time"], y=sample["views"], mode="lines+markers", name="Actual", line=dict(color="#fcfcfd")))
        figure.add_trace(go.Scatter(x=sample["event_time"], y=sample["prediction"], mode="lines", name="Forecast", line=dict(color=ACCENT, width=3)))
        figure.add_trace(go.Scatter(x=sample["event_time"], y=sample["lag_7"], mode="lines", name="Weekly naive", line=dict(color=CYAN, dash="dot")))
        figure.update_layout(title="Rolling-origin holdout and conformal interval", xaxis_title="", yaxis_title="Daily views")
        st.plotly_chart(_style(figure, 470), width="stretch")
    with eval_right:
        st.markdown("#### Per-article scorecard")
        scorecard = model.per_article.copy()
        for column in ["wape", "smape", "baseline_wape", "skill_vs_weekly_naive", "interval_coverage"]:
            scorecard[column] = scorecard[column].map(lambda value: f"{value:.1%}")
        st.dataframe(scorecard[["article", "wape", "baseline_wape", "skill_vs_weekly_naive", "interval_coverage", "test_days"]], hide_index=True, width="stretch")

    anomaly_left, anomaly_right = st.columns([0.8, 1.2])
    with anomaly_left:
        st.markdown("#### Forecast anomaly queue")
        if model.anomalies.empty:
            st.success("No holdout observations fell outside the calibrated interval.")
        else:
            st.dataframe(model.anomalies[["article", "event_time", "views", "prediction", "interval_direction", "severity"]], hide_index=True, width="stretch")
        st.caption("Outside-interval events deserve investigation; the model cannot infer why attention changed.")
    with anomaly_right:
        figure = px.bar(
            model.residuals, x="event_time", y="mean_residual", color="anomaly_count",
            color_continuous_scale=[[0, "#4b5563"], [1, "#fb7185"]], title="Portfolio residuals and anomaly concentration",
            labels={"event_time": "", "mean_residual": "Actual − forecast", "anomaly_count": "Anomalies"},
        )
        st.plotly_chart(_style(figure, 390), width="stretch")

    st.markdown(
        '<section class="section-intro"><div class="section-kicker">Forward planning</div>'
        f'<h2>{horizon}-day attention<br>capacity outlook.</h2></section>', unsafe_allow_html=True,
    )
    future = model.future.copy(); future["article_label"] = future["article"].str.replace("_", " ")
    figure = px.line(future, x="event_time", y="prediction", color="article_label", markers=True, title="Recursive production forecast", labels={"event_time": "", "prediction": "Expected daily views", "article_label": ""})
    st.plotly_chart(_style(figure, 470), width="stretch")
    latest = future.groupby("article", as_index=False).agg(
        forecast_views=("prediction", "sum"), average_daily=("prediction", "mean"), peak_daily=("upper", "max")
    ).sort_values("forecast_views", ascending=False)
    st.dataframe(latest, hide_index=True, width="stretch")
    st.caption("The production estimator is refit on all validated history only after the untouched backtest is scored. Recursive horizons accumulate uncertainty not fully represented by a fixed conformal radius.")

    downloads = st.columns(3)
    downloads[0].download_button("Export forecast", future.to_csv(index=False).encode(), f"wikipedia_forecast_{bundle.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[1].download_button("Export holdout audit", model.predictions.to_csv(index=False).encode(), f"wikipedia_holdout_{bundle.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[2].download_button("Export run manifest", _manifest(bundle, model).encode(), f"wikipedia_manifest_{bundle.metadata['run_id']}.json", "application/json", width="stretch")

    with st.expander("Method, source boundaries and production path"):
        st.markdown(
            f"""
            **Source.** Daily user pageviews come from the [Wikimedia Analytics API]({DOCS_URL}) for English Wikipedia,
            `all-access`, `user` traffic. The API serves pageview data from July 2015 onward. This app deliberately
            ends two days before retrieval because aggregates commonly need a full day to populate.

            **Streaming demonstration.** The source is a daily aggregate API, not a live event bus. The hosted app
            therefore performs a deterministic replay: content-addressed events receive simulated arrival delays,
            duplicate deliveries and 120-event micro-batches. A two-day watermark makes late data observable.

            **Forecast.** One global gradient-boosted model uses only shifted 1/7/14-day lags, rolling 7/28-day
            statistics, weekday cycles, article identity and an ordinal trend. The newest 28 days are the test set;
            the preceding 28 calibrate a 90% conformal residual interval. Weekly lag-7 is the explicit baseline.

            **Limitations.** Attention is not sentiment or causality. Traffic can be affected by current events,
            links, bots not fully excluded by source definitions and article renames. Split-conformal coverage is
            empirical and can fail under distribution shift; recursive future intervals are approximate.

            This independent app follows Wikimedia's [API Usage Guidelines]({USAGE_URL}) and [Terms of Use]({TERMS_URL});
            it is not developed, sponsored or endorsed by the Wikimedia Foundation.
            """
        )
