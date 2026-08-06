"""Modern Streamlit control plane for trial data and model operations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clinical_trial_ops_pipeline.src.data import (
    API_DOCS_URL, API_URL, CONDITION_PRESETS, STRUCTURE_URL, TERMS_URL, safe_condition,
)
from clinical_trial_ops_pipeline.src.model import ModelBundle, score_scenario, train_and_evaluate
from clinical_trial_ops_pipeline.src.pipeline import PipelineBundle, run_pipeline


ACCENT = "#f59e0b"
PALETTE = [ACCENT, "#fcfcfd", "#9ba0a8", "#656a73", "#343942"]


def _style(figure: go.Figure, height: int = 430) -> go.Figure:
    figure.update_layout(
        height=height, margin=dict(l=18, r=18, t=62, b=22),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"), colorway=PALETTE,
        legend=dict(orientation="h", y=1.03, title=None),
        hoverlabel=dict(bgcolor="#171a20", font_color="#fcfcfd", bordercolor="#5c6068"),
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=21_600, show_spinner=False)
def _pipeline(condition: str, batch_size: int) -> PipelineBundle:
    return run_pipeline(condition, batch_size)


@st.cache_resource(show_spinner=False)
def _model(feature_hash: str, _features: pd.DataFrame) -> ModelBundle:
    return train_and_evaluate(_features)


def _manifest(bundle: PipelineBundle, model: ModelBundle) -> str:
    payload = {
        "pipeline": bundle.metadata,
        "events": bundle.events.to_dict(orient="records"),
        "quality_checks": bundle.quality.to_dict(orient="records"),
        "model": {**model.metadata, **{key: value for key, value in model.metrics.items() if key != "confusion_matrix"}},
        "drift": model.drift.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, default=str)


def render_dashboard() -> None:
    st.markdown(
        """
        <style>
        .trial-alert {padding:1rem 1.15rem;border:1px solid rgba(245,158,11,.35);background:rgba(245,158,11,.08);border-radius:4px}
        .trial-step {padding:1rem;border-left:3px solid #f59e0b;background:rgba(255,255,255,.035);min-height:126px}
        .trial-step b {color:#fcfcfd;font-size:.88rem}.trial-step p {font-size:.78rem;margin:.6rem 0 0}
        </style>
        <section class="page-hero">
          <div class="eyebrow">15 / Data + AI engineering</div>
          <h1>Clinical Trial<br>Operations ML.</h1>
          <p>
            Turn public registry snapshots into a contracted feature product, then
            evaluate and monitor an explainable model for trial-discontinuation signals.
          </p>
          <div class="source-line">ClinicalTrials.gov API v2 · Idempotent snapshots · Time-aware model validation</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1.1, 1.45, 0.8])
    with controls[0]:
        preset = st.selectbox("Condition portfolio", list(CONDITION_PRESETS), index=0)
    with controls[1]:
        custom = st.text_input("Optional custom condition", placeholder="e.g. chronic kidney disease")
    with controls[2]:
        batch_size = st.select_slider("Snapshot size", options=[120, 180, 240, 320, 400], value=240)
    condition = safe_condition(custom) if custom.strip() else CONDITION_PRESETS[preset]
    if custom.strip() and not condition:
        st.warning("The custom condition contains no searchable characters. The preset is used instead.")
        condition = CONDITION_PRESETS[preset]

    try:
        with st.spinner("Ingesting registry records, enforcing the contract and training the time-aware model…"):
            bundle = _pipeline(condition, batch_size)
            model = _model(bundle.metadata["feature_hash"], bundle.features)
    except (ValueError, KeyError, TypeError) as exc:
        st.error("The pipeline could not produce a trustworthy model run.")
        st.caption(f"Failure state · {type(exc).__name__}: {exc}")
        st.info("Try a broader condition or the deterministic fallback corpus. No score is shown when validation fails.")
        return

    if bundle.metadata["mode"] == "demo":
        st.warning(
            "ClinicalTrials.gov is currently unavailable or returned too few usable records. "
            "Pipeline, evaluation and controls are running on a deterministic synthetic registry snapshot."
        )
        st.caption("Fallback reason: " + bundle.metadata["fallback_reason"])
    else:
        retrieved = pd.to_datetime(bundle.metadata["retrieved_at"], utc=True)
        st.success(
            f"Live ClinicalTrials.gov snapshot · {bundle.metadata['source_matches']:,} matching terminal studies · "
            f"retrieved {retrieved.strftime('%d %b %Y, %H:%M UTC')}"
        )

    retention = len(bundle.validated) / max(len(bundle.snapshot), 1)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Source matches", f"{bundle.metadata['source_matches']:,}")
    k2.metric("Validated trials", f"{len(bundle.validated):,}")
    k3.metric("Contract retention", f"{retention:.1%}")
    k4.metric("Quality pass", f"{bundle.metadata['quality_pass_rate']:.0%}")
    k5.metric("Holdout ROC AUC", f"{model.metrics['roc_auc']:.3f}")
    k6.metric("Brier score", f"{model.metrics['brier_score']:.3f}", help="Lower is better; 0 is perfect probabilistic accuracy.")
    st.caption(
        f"Run {bundle.metadata['run_id']} · model: {model.metadata['model']} · "
        f"{model.metadata['training_rows']} training / {model.metadata['holdout_rows']} holdout rows · "
        f"split: {model.metadata['split_strategy']}."
    )
    st.markdown(
        '<div class="trial-alert"><b>Decision boundary:</b> This is an operations-research demonstration, not a medical, ethical or investment decision tool. '
        'Scores rank patterns in this bounded sample; they are not real-world failure probabilities and never replace protocol, safety or domain review.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Data engineering control plane</div>
          <h2>Every model begins<br>with a trustworthy snapshot.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    flow_cols = st.columns(4)
    descriptions = [
        ("01 · EXTRACT", "Bounded API v2 request with timeout, status handling and deterministic fallback."),
        ("02 · SNAPSHOT", "Canonical payload hashes provide idempotency and content-level lineage."),
        ("03 · CONTRACT", "Nested JSON becomes typed, deduplicated trial records with explicit rejection rules."),
        ("04 · FEATURE VIEW", "Only registration/design-time features cross the leakage boundary into ML."),
    ]
    for column, (title, body) in zip(flow_cols, descriptions):
        column.markdown(f'<div class="trial-step"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        flow = bundle.events.copy()
        flow["rows"] = flow["output_rows"].map(lambda value: f"{value:,}")
        fig = px.bar(
            flow, x="stage", y="output_rows", text="rows", color="status",
            color_discrete_map={"passed": ACCENT, "failed": "#656a73"},
            hover_data={"input_rows": True, "rejected_rows": True, "duration_ms": ":.2f", "content_hash": True},
            title="Rows, latency and lineage by stage", labels={"stage": "", "output_rows": "Output rows", "status": ""},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(_style(fig, 450), width="stretch")
    with right:
        st.markdown("#### Run ledger")
        st.dataframe(bundle.events, hide_index=True, width="stretch")
        st.caption(
            "The hosted product executes in memory. The same run ID and content hashes can key immutable object-store snapshots "
            "and make retries idempotent in a production scheduler."
        )

    quality_left, quality_right = st.columns([0.9, 1.1])
    with quality_left:
        qa = bundle.quality.copy()
        qa["result"] = qa["passed"].map({True: "Passed", False: "Failed"})
        fig = px.bar(
            qa, y="check", x=[1] * len(qa), color="result", orientation="h",
            color_discrete_map={"Passed": ACCENT, "Failed": "#656a73"},
            hover_data={"detail": True}, title="Contract, reconciliation and leakage checks",
            labels={"value": "", "check": "", "result": ""},
        )
        fig.update_xaxes(visible=False)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_style(fig, 470), width="stretch")
    with quality_right:
        layer = st.radio("Inspect data product", ["Validated contract", "ML feature view"], horizontal=True)
        if layer == "Validated contract":
            preview = bundle.validated[["nct_id", "title", "overall_status", "phase", "enrollment", "sponsor_class", "first_post_date"]].tail(15)
        else:
            preview = bundle.features[["nct_id", "phase", "sponsor_class", "enrollment_log", "country_count", "discontinued"]].tail(15)
        st.dataframe(preview, hide_index=True, width="stretch")
        st.download_button(
            "Export validated feature snapshot",
            bundle.features.to_csv(index=False).encode("utf-8"),
            file_name=f"clinical_trial_features_{bundle.metadata['run_id']}.csv",
            mime="text/csv", width="stretch",
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">AI engineering lifecycle</div>
          <h2>Evaluate discrimination,<br>calibration and drift.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(6)
    for column, (label, key) in zip(metric_cols, [
        ("ROC AUC", "roc_auc"), ("Avg precision", "average_precision"), ("Accuracy", "accuracy"),
        ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
    ]):
        column.metric(label, f"{model.metrics[key]:.3f}")
    st.caption(
        "The newest registry records are held out whenever both labels remain available. If that becomes impossible, "
        "the run ledger explicitly reports the deterministic stratified fallback. Balanced class weights improve learning but alter score calibration."
    )

    eval_left, eval_right = st.columns(2)
    with eval_left:
        calibration = model.calibration.copy()
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#656a73", dash="dash")))
        chart.add_trace(go.Scatter(
            x=calibration["mean_score"], y=calibration["observed_rate"], mode="lines+markers+text",
            text=calibration["records"].map(lambda value: f"n={value}"), textposition="top center", name="Holdout",
            line=dict(color=ACCENT, width=3), marker=dict(size=10),
        ))
        chart.update_layout(title="Calibration: score versus observed rate", xaxis_title="Mean model score", yaxis_title="Observed discontinuation rate")
        st.plotly_chart(_style(chart, 440), width="stretch")
    with eval_right:
        matrix = pd.DataFrame(model.metrics["confusion_matrix"], index=["Actual completed", "Actual discontinued"], columns=["Predicted completed", "Predicted discontinued"])
        chart = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, "#20242c"], [1, ACCENT]], title="Holdout confusion matrix")
        chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_style(chart, 440), width="stretch")

    explain_left, explain_right = st.columns([1.15, 0.85])
    with explain_left:
        coefficients = model.coefficients.head(14).sort_values("coefficient")
        chart = px.bar(
            coefficients, x="coefficient", y="feature", orientation="h", color="direction",
            color_discrete_map={"Higher discontinuation signal": ACCENT, "Lower discontinuation signal": "#fcfcfd"},
            title="Largest model coefficients", labels={"coefficient": "Log-odds coefficient", "feature": "", "direction": ""},
        )
        st.plotly_chart(_style(chart, 500), width="stretch")
        st.caption("Coefficients are associations after preprocessing, not causal effects. Correlated features can redistribute weight.")
    with explain_right:
        st.markdown("#### Drift monitor")
        st.dataframe(
            model.drift, hide_index=True, width="stretch",
            column_config={"drift_score": st.column_config.ProgressColumn("PSI / shift", min_value=0, max_value=max(0.5, model.drift["drift_score"].max()), format="%.3f")},
        )
        high = int((model.drift["level"] == "High").sum())
        watch = int((model.drift["level"] == "Watch").sum())
        if high:
            st.error(f"{high} monitored fields show high shift between training and holdout. Treat scores as unstable.")
        elif watch:
            st.warning(f"{watch} monitored fields need review before promotion.")
        else:
            st.success("No monitored numeric feature exceeds the configured drift thresholds.")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Model workbench</div>
          <h2>Stress-test a design<br>without pretending causality.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    inputs = st.columns(4)
    phase = inputs[0].selectbox("Phase", ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"], index=2)
    enrollment = inputs[1].number_input("Planned enrollment", min_value=1, max_value=100_000, value=220, step=10)
    countries = inputs[2].slider("Countries", 1, 30, 3)
    interventions = inputs[3].slider("Interventions", 0, 12, 2)
    inputs2 = st.columns(4)
    sponsor = inputs2[0].selectbox("Sponsor class", ["INDUSTRY", "NIH", "FED", "OTHER", "UNKNOWN"])
    allocation = inputs2[1].selectbox("Allocation", ["RANDOMIZED", "NON_RANDOMIZED", "NA"])
    masking = inputs2[2].selectbox("Masking", ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", "NA"], index=2)
    purpose = inputs2[3].selectbox("Primary purpose", ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", "NA"])
    scenario = {
        "phase": phase, "study_type": "INTERVENTIONAL", "sponsor_class": sponsor,
        "allocation": allocation, "masking": masking, "primary_purpose": purpose,
        "enrollment_log": float(np.log1p(enrollment)), "condition_count": 2,
        "intervention_count": interventions, "country_count": countries, "minimum_age": 18,
        "age_span": 57, "healthy_volunteers": 0,
    }
    scenario_score = score_scenario(model, scenario)
    left_score, right_score = st.columns([0.35, 0.65])
    left_score.metric("Model discontinuation signal", f"{scenario_score:.1%}", help="A sample-relative model score, not a real-world probability.")
    right_score.progress(scenario_score, text="Higher means the design resembles discontinued records in this model snapshot")
    right_score.caption(
        "Use the workbench to test model sensitivity and identify fields worth reviewing. Changing a control does not prove that the change would alter a real trial outcome."
    )

    st.markdown("#### Time-holdout audit records")
    audit = model.holdout.sort_values("risk_score", ascending=False).copy()
    audit["risk_score"] = audit["risk_score"].round(4)
    st.dataframe(
        audit[["nct_id", "title", "first_post_date", "overall_status", "risk_score", "predicted_class", "record_url"]],
        hide_index=True, width="stretch",
        column_config={"record_url": st.column_config.LinkColumn("Registry record", display_text="Open")},
    )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Export holdout predictions", audit.to_csv(index=False).encode("utf-8"),
        file_name=f"clinical_trial_holdout_{bundle.metadata['run_id']}.csv", mime="text/csv", width="stretch",
    )
    downloads[1].download_button(
        "Export operational manifest", _manifest(bundle, model).encode("utf-8"),
        file_name=f"clinical_trial_manifest_{bundle.metadata['run_id']}.json", mime="application/json", width="stretch",
    )

    with st.expander("Method, source and limitations"):
        st.markdown(
            f"""
            **Source.** [ClinicalTrials.gov API v2]({API_DOCS_URL}) from the U.S. National Library of Medicine.
            The app requests terminal study records from `{API_URL}` and parses identification, status, design,
            sponsor, condition, intervention, location and eligibility modules. See the official
            [study-data structure]({STRUCTURE_URL}) and [terms and conditions]({TERMS_URL}).

            **Label.** `COMPLETED` is class 0; `TERMINATED`, `WITHDRAWN` and `SUSPENDED` are class 1.
            Registry status is sponsor-reported and the classes combine different operational situations.

            **Model.** L2-regularized logistic regression with median imputation, standardization,
            one-hot encoding and balanced class weights. The newest 25% form the preferred holdout.
            ROC AUC measures ranking, average precision emphasizes the positive class, Brier score measures
            probabilistic error, and calibration bins compare score with observed frequency.

            **Limitations.** The bounded snapshot is not prevalence-representative; class weighting changes
            calibration; fields can be missing or updated after registration; status is not a causal outcome;
            protocol quality, safety, funding and site-level context are not modeled. Never use this app to
            decide whether a study, intervention or participant is medically appropriate.
            """
        )
