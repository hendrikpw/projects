"""Streamlit control plane for the handwritten-digit recognition gateway."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from digit_recognition_gateway.src.data import DATASET_PAGE, DOI
from digit_recognition_gateway.src.model import score_image, train_and_evaluate
from digit_recognition_gateway.src.pipeline import run_pipeline

INK = "#090b10"; AMBER = "#ffb000"; ORANGE = "#ff6b2c"; CREAM = "#fff7e6"; MINT = "#5ee6b8"


def _style(fig, height: int = 410):
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=58, b=24), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", font=dict(color=CREAM, family="Inter, sans-serif"), legend=dict(orientation="h", y=1.14), colorway=[AMBER, ORANGE, MINT, "#9d7cff"])
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
    .digit-hero{padding:3.8rem 3rem;border-radius:28px;background:radial-gradient(circle at 86% 15%,rgba(255,176,0,.3),transparent 28%),radial-gradient(circle at 74% 95%,rgba(255,107,44,.2),transparent 33%),linear-gradient(145deg,#21180b,#090b10);border:1px solid rgba(255,176,0,.3);margin-bottom:1.4rem}.digit-hero h1{font:850 clamp(2.8rem,6vw,5.8rem)/.9 Inter;color:#fff7e6;letter-spacing:-.065em;max-width:1050px;margin:.8rem 0}.digit-hero p{max-width:900px;color:#d7c9ae;font-size:1.05rem}.kicker{color:#ffb000;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.boundary{border-left:3px solid #ff6b2c;background:rgba(255,107,44,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:174px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #ffb000;border-radius:0 0 14px 14px}.stage b{color:#ffb000;font-size:.76rem;letter-spacing:.08em}.stage p{color:#d7c9ae;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#ffb000;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}.route{padding:1rem;border:1px solid rgba(255,176,0,.28);border-radius:14px;background:rgba(255,176,0,.07)}</style>
    <section class="digit-hero"><div class="kicker">UCI Optdigits / Data + AI Engineering</div><h1>Read the mark.<br>Guard the decision.</h1><p>A content-addressed image pipeline turns official writer-separated digit matrices into a contracted data product. A calibrated vision model then exposes clean accuracy, corruption behavior, confidence routing and pixel-level evidence.</p></section>""", unsafe_allow_html=True)
    try:
        with st.spinner("Downloading digit matrices, reconciling micro-batches and evaluating unseen writers …"):
            product, model = _load()
    except Exception as exc:
        st.error("No recognition product was published because a source, quality or model-promotion gate failed."); st.exception(exc); return
    if product.metadata["mode"] == "demo": st.warning(f"Deterministic demonstration data are active: {product.metadata['fallback_reason']}")
    else: st.success("Official UCI archive loaded · ten data gates and all model promotion gates passed")
    values = [("Images", f"{len(product.gold):,}"), ("Unseen-writer accuracy", f"{model.metrics['accuracy']:.2%}"), ("Macro F1", f"{model.metrics['macro_f1']:.2%}"), ("Top-3 accuracy", f"{model.metrics['top3_accuracy']:.2%}"), ("Auto-read precision", f"{model.metrics['selective_accuracy']:.2%}"), ("Corrupted accuracy", f"{model.metrics['corrupted_accuracy']:.2%}")]
    for column, (label, value) in zip(st.columns(6), values): column.metric(label, value)
    st.caption(f"Run {product.metadata['run_id']} · train {model.metadata['train_images']:,} / calibration {model.metadata['calibration_images']:,} / untouched test {model.metadata['test_images']:,} · threshold {model.threshold:.2f} · {model.metrics['inference_ms_per_image']:.3f} ms/image")
    st.markdown('<div class="boundary"><b>Decision boundary:</b> this gateway reads normalized 8×8 single-digit images resembling the UCI training domain. It is not handwriting authentication, document OCR, identity evidence or a safety-critical decision system. Low-confidence and out-of-domain inputs are reviewed or withheld.</div>', unsafe_allow_html=True)

    pipeline_tab, model_tab, lab_tab = st.tabs(["◫ Pipeline control", "◎ Model evaluation", "⌁ Robustness lab"])
    with pipeline_tab:
        st.markdown('<section class="section"><small>Data engineering</small><h2>Every image hashed.<br>Every delivery reconciled.</h2></section>', unsafe_allow_html=True)
        cards = [("01 · SAFE EXTRACT", "Retry, timeout, byte bounds, ZIP signature, traversal protection and an exact two-file allowlist."), ("02 · MICRO-BATCH", "Stable sample and image hashes, 128-row batches and twenty intentional replay deliveries."), ("03 · CONTRACT", "Typed 8×8 pixel ranges, integer labels, split preservation, quarantine and row reconciliation."), ("04 · PUBLISH", "Normalized Gold tensors, writer-boundary audit, SHA-256 lineage and ten fail-closed gates.")]
        for column, (title, body) in zip(st.columns(4), cards): column.markdown(f'<div class="stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
        left, right = st.columns([1.1, .9])
        with left:
            distribution = product.gold.groupby(["source_split", "label"], as_index=False).size(); fig = px.bar(distribution, x="label", y="size", color="source_split", barmode="group", title="Class balance and official writer split"); st.plotly_chart(_style(fig), width="stretch")
        with right: st.markdown("#### Layer ledger"); st.dataframe(product.stages, hide_index=True, width="stretch")
        left, right = st.columns(2)
        with left:
            checks = product.quality.assign(result=product.quality.passed.map({True: "Passed", False: "Failed"})); st.markdown("#### Publication gates"); st.dataframe(checks[["check", "result", "detail"]], hide_index=True, width="stretch")
        with right:
            batches = product.batches.tail(50); fig = px.line(batches, x="batch_id", y=["deliveries", "unique_images"], markers=True, title="Micro-batch delivery and deduplication"); st.plotly_chart(_style(fig), width="stretch")
        if len(product.quarantine): st.dataframe(product.quarantine.head(250), hide_index=True, width="stretch")
        else: st.info("No official sample violated the image contract in this run; twenty replay deliveries were still detected and suppressed.")

    with model_tab:
        st.markdown('<section class="section"><small>AI engineering</small><h2>Test on new writers.<br>Measure every failure.</h2></section>', unsafe_allow_html=True)
        st.markdown("An RBF support-vector classifier is fitted only on the official 30-writer training partition. A disjoint calibration slice selects temperature and the confidence route; all 1,797 samples from 13 different writers remain untouched until final evaluation.")
        left, right = st.columns([1.05, .95])
        with left:
            fig = px.imshow(model.confusion.pivot(index="actual", columns="predicted", values="rate"), text_auto=".0%", color_continuous_scale=[INK, ORANGE, AMBER], zmin=0, zmax=1, title="Normalized confusion matrix · untouched writers"); st.plotly_chart(_style(fig, 500), width="stretch")
        with right:
            fig = px.bar(model.class_metrics, x="digit", y="recall", color="review_rate", text_auto=".1%", title="Recall and review rate by digit", color_continuous_scale=[MINT, AMBER, ORANGE], range_y=[.85,1]); st.plotly_chart(_style(fig, 500), width="stretch")
        left, right = st.columns(2)
        with left:
            fig = px.line(model.robustness, x="severity", y=["accuracy", "macro_f1", "coverage"], markers=True, title="Controlled noise + pixel-dropout stress curve"); st.plotly_chart(_style(fig), width="stretch")
        with right:
            grid = model.importance.pivot(index="row", columns="column", values="importance"); fig = px.imshow(grid, color_continuous_scale=[INK, ORANGE, AMBER], title="Global permutation importance by pixel"); st.plotly_chart(_style(fig), width="stretch")
        st.markdown("#### Held-out prediction audit"); st.dataframe(model.evaluation.head(300), hide_index=True, width="stretch")

    with lab_tab:
        st.markdown('<section class="section"><small>Serving safety</small><h2>Damage the input.<br>Watch the route change.</h2></section>', unsafe_allow_html=True)
        labels = sorted(product.gold.loc[product.gold.source_split == "test", "label"].unique()); left, right = st.columns(2); label = left.selectbox("Actual digit", labels); choices = product.gold[(product.gold.source_split == "test") & (product.gold.label == label)].reset_index(drop=True); sample = right.slider("Held-out sample", 1, len(choices), 1) - 1
        controls = st.columns(2); noise = controls[0].slider("Gaussian sensor noise", 0.0, .35, 0.0, .01); dropout = controls[1].slider("Missing-pixel probability", 0.0, .40, 0.0, .01)
        result = score_image(model, choices.iloc[sample], noise=noise, dropout=dropout)
        metrics = st.columns(5); metrics[0].metric("Actual", label); metrics[1].metric("Prediction", result["prediction"]); metrics[2].metric("Confidence", f"{result['confidence']:.1%}"); metrics[3].metric("Dropped pixels", f"{result['dropout_share']:.1%}"); metrics[4].metric("Route", result["route"])
        left, middle, right = st.columns([.8, .8, 1.1])
        with left: st.plotly_chart(_style(px.imshow(result["image"], color_continuous_scale=[INK, AMBER], zmin=0, zmax=1, title="Served 8×8 image"), 380), width="stretch")
        with middle: st.plotly_chart(_style(px.imshow(result["sensitivity"], color_continuous_scale=[INK, ORANGE, AMBER], title="Local occlusion sensitivity"), 380), width="stretch")
        with right: st.plotly_chart(_style(px.bar(result["ranking"].head(5), x="probability", y="digit", orientation="h", title="Top candidate probabilities"), 380), width="stretch")
        st.markdown(f'<div class="route"><b>Serving route · {result["route"].upper()}</b><br>Auto-read requires calibrated confidence, plausible ink density and no severe missing-pixel condition. The sensitivity view removes one pixel at a time and measures the drop in support for the winning class.</div>', unsafe_allow_html=True)

    st.markdown('<section class="section"><small>Audit and provenance</small><h2>Export the evidence.</h2></section>', unsafe_allow_html=True)
    left, middle, right = st.columns(3); left.download_button("Download run manifest", _manifest(product, model), "digit_gateway_manifest.json", "application/json", width="stretch"); middle.download_button("Download prediction audit", model.evaluation.to_csv(index=False).encode(), "digit_gateway_predictions.csv", "text/csv", width="stretch"); right.download_button("Download quality ledger", product.quality.to_csv(index=False).encode(), "digit_gateway_quality.csv", "text/csv", width="stretch")
    st.caption(f"Source: UCI Optical Recognition of Handwritten Digits · static · CC BY 4.0 · [dataset]({DATASET_PAGE}) · [DOI]({DOI})")
