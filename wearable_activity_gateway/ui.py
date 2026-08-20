"""Modern Streamlit operations and model control plane."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from wearable_activity_gateway.src.data import DATASET_PAGE, DOI
from wearable_activity_gateway.src.model import score_window, train_and_evaluate
from wearable_activity_gateway.src.pipeline import run_pipeline

VOID = "#050907"; LIME = "#b9ff66"; CYAN = "#5ce1e6"; AMBER = "#ffd166"; RED = "#ff6b7a"; MINT = "#6ef0b2"


def _style(fig, height=410):
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=58, b=24), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", font=dict(color="#eef8ef", family="Inter, sans-serif"), legend=dict(orientation="h", y=1.14), colorway=[LIME, CYAN, MINT, AMBER, RED])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


@st.cache_resource(ttl=21600, show_spinner=False)
def _load():
    product = run_pipeline()
    return product, train_and_evaluate(product.gold, product.features)


def _manifest(product, model):
    return json.dumps({"pipeline": product.metadata, "quality": product.quality.to_dict("records"), "model": model.metadata, "metrics": model.metrics}, indent=2, default=str).encode()


def render_dashboard():
    st.markdown("""<style>
    .wear-hero{padding:3.7rem 3rem;border-radius:28px;background:radial-gradient(circle at 88% 18%,rgba(185,255,102,.25),transparent 29%),radial-gradient(circle at 78% 94%,rgba(92,225,230,.13),transparent 31%),linear-gradient(145deg,#13251c,#050907);border:1px solid rgba(185,255,102,.28);margin-bottom:1.4rem}.wear-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#f9fff7;letter-spacing:-.06em;max-width:1050px;margin:.8rem 0}.wear-hero p{max-width:900px;color:#b9cbbd;font-size:1.05rem}.kicker{color:#b9ff66;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.boundary{border-left:3px solid #ffd166;background:rgba(255,209,102,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:174px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #b9ff66;border-radius:0 0 14px 14px}.stage b{color:#b9ff66;font-size:.76rem;letter-spacing:.08em}.stage p{color:#b9cbbd;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#b9ff66;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style>
    <section class="wear-hero"><div class="kicker">UCI HAR / Data + AI Engineering</div><h1>Window the signal.<br>Guard the inference.</h1><p>A replay-safe micro-batch pipeline contracts smartphone sensor windows. A subject-isolated activity model then calibrates confidence, measures class behavior and withholds predictions when sensors drift, disappear or leave the training domain.</p></section>""", unsafe_allow_html=True)
    try:
        with st.spinner("Validating sensor windows, replaying micro-batches and evaluating held-out subjects …"):
            product, model = _load()
    except Exception as exc:
        st.error("No wearable inference product was published because a pipeline or model gate failed."); st.exception(exc); return
    if product.metadata["mode"] == "demo": st.warning(f"Deterministic demonstration windows are active: {product.metadata['fallback_reason']}")
    else: st.success("Official UCI HAR archive loaded · all ten publication gates and model promotion gates passed")
    values = [("Sensor windows", f"{len(product.gold):,}"), ("Features", f"{len(product.features):,}"), ("Subjects", f"{product.gold.subject_id.nunique():,}"), ("Macro F1", f"{model.metrics['macro_f1']:.3f}"), ("Top-2 accuracy", f"{model.metrics['top2_accuracy']:.1%}"), ("Selective accuracy", f"{model.metrics['selective_accuracy']:.1%}")]
    for column, (label, value) in zip(st.columns(6), values): column.metric(label, value)
    st.caption(f"Run {product.metadata['run_id']} · {model.metadata['train_subjects']} train / {model.metadata['calibration_subjects']} calibration / {model.metadata['test_subjects']} test subjects · {model.metrics['inference_ms_per_window']:.3f} ms per window")
    st.markdown('<div class="boundary"><b>Operational boundary:</b> activity inference from a waist-mounted 2012 smartphone study is not medical monitoring, fall detection or proof of a person’s behavior. Low confidence, missing features and out-of-distribution windows are withheld for review.</div>', unsafe_allow_html=True)

    st.markdown('<section class="section"><small>Data engineering control plane</small><h2>Replay every delivery.<br>Reconcile every window.</h2></section>', unsafe_allow_html=True)
    cards = [("01 · SAFE EXTRACT", "Versioned ZIP, retry, timeout, byte limits, signature check, traversal protection and exact file allowlist."), ("02 · MICRO-BATCH", "Stable window IDs, payload hashes, event timestamps, 256-window batches and 20 intentional replays."), ("03 · CONTRACT", "Finite normalized features, subject/activity consistency, range checks, quarantine reasons and replay suppression."), ("04 · GOLD", "Model-ready windows, subject keys, content hashes, batch ledger and ten fail-closed publication gates.")]
    for column, (title, body) in zip(st.columns(4), cards): column.markdown(f'<div class="stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.08, .92])
    with left:
        fig = px.line(product.batches, x="batch_id", y=["deliveries", "unique_windows"], markers=True, title="Micro-batch delivery and replay observability"); st.plotly_chart(_style(fig), width="stretch")
    with right: st.markdown("#### Layer ledger"); st.dataframe(product.stages, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        checks = product.quality.assign(result=product.quality.passed.map({True: "Passed", False: "Failed"})); st.markdown("#### Publication gates"); st.dataframe(checks[["check", "result", "detail"]], hide_index=True, width="stretch")
    with right:
        counts = product.gold.groupby(["source_split", "activity"], as_index=False).size(); fig = px.bar(counts, x="activity", y="size", color="source_split", barmode="group", title="Contracted windows by source partition"); st.plotly_chart(_style(fig), width="stretch")
    if len(product.quarantine):
        st.markdown("#### Quarantine audit"); st.dataframe(product.quarantine.quarantine_reason.value_counts().rename_axis("reason").reset_index(name="deliveries"), hide_index=True, width="stretch")

    st.markdown('<section class="section"><small>AI engineering evaluation</small><h2>Separate the people.<br>Calibrate the certainty.</h2></section>', unsafe_allow_html=True)
    metrics = [("Accuracy", f"{model.metrics['accuracy']:.1%}"), ("Balanced accuracy", f"{model.metrics['balanced_accuracy']:.1%}"), ("Macro F1", f"{model.metrics['macro_f1']:.3f}"), ("Majority baseline", f"{model.metrics['baseline_macro_f1']:.3f}"), ("ECE", f"{model.metrics['ece']:.3f}"), ("Review route", f"{model.metrics['review_rate']:.1%}")]
    for column, (label, value) in zip(st.columns(6), metrics): column.metric(label, value)
    left, right = st.columns(2)
    with left:
        fig = px.density_heatmap(model.confusion, x="predicted", y="actual", z="rate", color_continuous_scale=[[0, "#07110a"], [1, LIME]], text_auto=".0%", title="Subject-held-out normalized confusion matrix"); st.plotly_chart(_style(fig, 480), width="stretch")
    with right:
        long = model.class_metrics.melt(id_vars=["activity", "windows"], value_vars=["recall", "review_rate"], var_name="metric", value_name="rate"); fig = px.bar(long, x="activity", y="rate", color="metric", barmode="group", title="Recall and withholding by activity"); fig.update_yaxes(tickformat=".0%", range=[0, 1.05]); st.plotly_chart(_style(fig, 480), width="stretch")
    left, right = st.columns(2)
    with left:
        top = model.importance.head(15).sort_values("importance"); fig = px.bar(top, x="importance", y="feature", orientation="h", title="Global Extra Trees feature importance"); st.plotly_chart(_style(fig, 500), width="stretch")
    with right:
        ordered = model.drift.sort_values("psi"); fig = px.bar(ordered, x="psi", y="feature", orientation="h", color="status", color_discrete_map={"stable": MINT, "watch": AMBER, "high": RED}, title="Train-to-test feature drift · PSI"); fig.add_vline(x=.1, line_dash="dot"); fig.add_vline(x=.25, line_dash="dash"); st.plotly_chart(_style(fig, 500), width="stretch")
    st.markdown("#### Held-out inference audit"); st.dataframe(model.evaluation.sort_values(["route", "confidence"]).head(100), hide_index=True, width="stretch")

    st.markdown('<section class="section"><small>Inference workbench</small><h2>Stress the device.<br>Observe the fail-safe route.</h2></section>', unsafe_allow_html=True)
    reference_id = st.selectbox("Reference test window", model.evaluation.window_id.tolist())
    reference = product.gold[product.gold.window_id == reference_id].iloc[0]
    controls = st.columns(3)
    scale = controls[0].slider("Sensor scale", .5, 1.8, 1.0, .05); noise = controls[1].slider("Deterministic noise", 0.0, .50, 0.0, .01); missing = controls[2].slider("Missing feature share", 0.0, .30, 0.0, .01)
    result = score_window(model, reference, scale=scale, noise=noise, missing_share=missing)
    columns = st.columns(5)
    for column, (label, value) in zip(columns, [("Prediction", result["prediction"].replace("_", " ").title()), ("Confidence", f"{result['confidence']:.1%}"), ("Route", result["route"].replace("-", " ").title()), ("OOD features", f"{result['ood_share']:.1%}"), ("Max |z|", f"{result['max_z']:.1f}")]): column.metric(label, value)
    if result["route"] == "sensor-fault-review": st.error("Inference withheld: missingness or feature-domain checks indicate a sensor/OOD failure state.")
    elif result["route"] == "human-review": st.warning("Inference withheld because calibrated confidence is below the automatic decision threshold.")
    else: st.success("Window clears the confidence and sensor-domain gates. This remains a model inference, not an observation of fact.")
    fig = px.bar(result["ranking"], x="probability", y="activity", orientation="h", title="Calibrated class probabilities"); st.plotly_chart(_style(fig, 360), width="stretch")
    downloads = st.columns(3)
    downloads[0].download_button("Export governed windows", product.gold[["window_id", "subject_id", "activity", "source_split", "batch_id"]].to_csv(index=False).encode(), f"wearable_windows_{product.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[1].download_button("Export model audit", model.evaluation.to_csv(index=False).encode(), f"wearable_model_{product.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[2].download_button("Export run manifest", _manifest(product, model), f"wearable_manifest_{product.metadata['run_id']}.json", "application/json", width="stretch")
    with st.expander("Source semantics, evaluation design and limits"):
        st.markdown(f"""**Source.** [UCI Human Activity Recognition Using Smartphones]({DATASET_PAGE}), DOI [10.24432/C54S4K]({DOI}), CC BY 4.0. Thirty volunteers aged 19–48 carried a Samsung Galaxy S II on the waist while performing walking, upstairs, downstairs, sitting, standing and laying. Accelerometer and gyroscope signals were recorded at 50 Hz, filtered and sampled into 2.56-second windows with 50% overlap. The published 561 time/frequency features are normalized to `[-1, 1]`.

**Isolation.** The original test subjects remain untouched. Subjects from the original training partition whose ID is divisible by five form a separate calibration group; every other original training subject is used for fitting. No subject appears in two model stages. This measures transfer to people not used for training rather than memorization of one wearer.

**Model.** A balanced Extra Trees ensemble learns six classes. Calibration selects a scalar temperature by multiclass log loss, then chooses an automatic-inference confidence threshold targeting 95% calibration accuracy. Test metrics include accuracy, balanced accuracy, macro F1, Top-2 accuracy, log loss, expected calibration error, coverage and selective accuracy. Majority-class macro F1 is the explicit baseline.

**Guardrails.** Missing features are median-filled only for computation and force sensor review above 5%. Per-feature training z-scores detect unusual windows; the prediction is withheld when more than 2% exceed four standard deviations or any feature exceeds eight. PSI is descriptive train/test drift. Neither guardrail proves that a real device is calibrated or worn in the documented waist position.

**Limits.** The study is static, small and device/placement-specific. Labels cover only six controlled activities and no transitions, falls, vehicles or clinical states. Overlapping windows are correlated. The model must be revalidated on the intended population, hardware, placement and raw-to-feature implementation before real deployment.""")
