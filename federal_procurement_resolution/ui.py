"""Streamlit control plane for procurement entity resolution."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from federal_procurement_resolution.src.data import ABOUT, API, CONTRACT, DOCS
from federal_procurement_resolution.src.pipeline import run_pipeline
from federal_procurement_resolution.src.resolution import resolve_name, train_and_evaluate

INK = "#07111f"; COPPER = "#ff9d5c"; SKY = "#66d6ff"; MINT = "#77e6bb"; GOLD = "#f3ca52"; RED = "#ff6e7f"


def _style(fig, height=410):
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=58, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)", font=dict(color="#edf6ff", family="Inter, sans-serif"), legend=dict(orientation="h", y=1.14), colorway=[COPPER, SKY, MINT, GOLD, RED])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


@st.cache_resource(ttl=21600, show_spinner=False)
def _load():
    product = run_pipeline()
    return product, train_and_evaluate(product.recipients)


def _manifest(product, model):
    return json.dumps({"pipeline": product.metadata, "quality": product.quality.to_dict("records"), "matching_metrics": model.metrics, "thresholds": model.thresholds}, indent=2, default=str).encode()


def render_dashboard():
    st.markdown("""<style>
    .proc-hero{padding:3.7rem 3rem;border-radius:28px;background:radial-gradient(circle at 88% 18%,rgba(255,157,92,.30),transparent 28%),radial-gradient(circle at 78% 92%,rgba(102,214,255,.15),transparent 31%),linear-gradient(145deg,#15263b,#07111f);border:1px solid rgba(255,157,92,.34);margin-bottom:1.4rem}
    .proc-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#fbfdff;letter-spacing:-.06em;max-width:1050px;margin:.8rem 0}.proc-hero p{max-width:900px;color:#b9cadb;font-size:1.05rem}.kicker{color:#ff9d5c;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}
    .boundary{border-left:3px solid #f3ca52;background:rgba(243,202,82,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:174px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #ff9d5c;border-radius:0 0 14px 14px}.stage b{color:#ff9d5c;font-size:.76rem;letter-spacing:.08em}.stage p{color:#b9cadb;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#ff9d5c;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}
    </style><section class="proc-hero"><div class="kicker">USAspending / Data + AI Engineering</div><h1>Trust the award.<br>Resolve the supplier.</h1><p>A replay-safe federal contract pipeline publishes governed award and recipient tables. An evaluated character-level matching service then repairs noisy vendor names, measures false merges and defers ambiguous identities.</p></section>""", unsafe_allow_html=True)
    try:
        with st.spinner("Reading USAspending, enforcing contracts and evaluating supplier matching …"):
            product, model = _load()
    except Exception as exc:
        st.error("No procurement data product was published because a pipeline or matching gate failed.")
        st.exception(exc)
        return
    if product.metadata["mode"] == "demo":
        st.warning(f"Deterministic demonstration awards are active: {product.metadata['fallback_reason']}")
    else:
        st.success(f"Official USAspending contract snapshot loaded through {product.metadata['as_of']} · all eight publication gates passed")
    values = [("Governed awards", f"{len(product.awards):,}"), ("Resolved recipients", f"{len(product.recipients):,}"), ("Replays suppressed", f"{product.metadata['duplicates']:,}"), ("Top-1 accuracy", f"{model.metrics['top1_accuracy']:.1%}"), ("Selective accuracy", f"{model.metrics['selective_accuracy']:.1%}"), ("Auto-link coverage", f"{model.metrics['coverage']:.1%}")]
    for column, (label, value) in zip(st.columns(6), values): column.metric(label, value)
    st.caption(f"Run {product.metadata['run_id']} · {product.metadata['pages']} API pages · {model.metrics['test_queries']:,} held-out corruption queries · thresholds {model.thresholds['score']:.2f} similarity / {model.thresholds['margin']:.3f} margin")
    st.markdown('<div class="boundary"><b>Decision boundary:</b> a proposed link is search assistance, not legal identity verification. UEI remains the authoritative key. Ambiguous or unseen names are explicitly routed to human review; large awards are not evidence of waste or fraud.</div>', unsafe_allow_html=True)

    st.markdown('<section class="section"><small>Data engineering control plane</small><h2>Replay the delivery.<br>Publish only reconciled records.</h2></section>', unsafe_allow_html=True)
    cards = [("01 · EXTRACT", "Bounded FY contract pages, explicit requested fields, timeouts and three-attempt exponential retry."), ("02 · BRONZE", "Unchanged deliveries, payload hashes and 15 intentional replays expose duplicate behavior."), ("03 · SILVER", "Typed dates and amounts, UEI/name/key contracts, reasoned quarantine and idempotent delivery IDs."), ("04 · GOLD", "Referentially complete award and canonical-recipient tables with deterministic hashes and a run manifest.")]
    for column, (title, body) in zip(st.columns(4), cards): column.markdown(f'<div class="stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95])
    with left:
        fig = px.bar(product.stages, x="stage", y="output", color="stage", text="output", hover_data=["input", "rejected", "hash"], title="Pipeline layers and content-addressed lineage")
        fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig), width="stretch")
    with right:
        st.markdown("#### Run ledger"); st.dataframe(product.stages, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Publication gates")
        checks = product.quality.assign(result=product.quality.passed.map({True: "Passed", False: "Failed"}))
        st.dataframe(checks[["check", "result", "detail"]], hide_index=True, width="stretch")
    with right:
        agencies = product.awards.groupby("awarding_agency", as_index=False).agg(awards=("award_id", "nunique"), value=("award_amount", "sum")).sort_values("value", ascending=False).head(10)
        fig = px.bar(agencies.sort_values("value"), x="value", y="awarding_agency", orientation="h", color="awards", title="Governed contract value by awarding agency")
        st.plotly_chart(_style(fig), width="stretch")
    if len(product.quarantine):
        st.markdown("#### Quarantine audit")
        st.dataframe(product.quarantine.quarantine_reason.value_counts().rename_axis("reason").reset_index(name="deliveries"), hide_index=True, width="stretch")

    st.markdown('<section class="section"><small>AI engineering evaluation</small><h2>Break the name.<br>Measure the recovery.</h2></section>', unsafe_allow_html=True)
    metrics = [("Top-1", f"{model.metrics['top1_accuracy']:.1%}"), ("Hit@5", f"{model.metrics['hit_at_5']:.1%}"), ("MRR@5", f"{model.metrics['mrr_at_5']:.3f}"), ("Exact baseline", f"{model.metrics['exact_baseline_accuracy']:.1%}"), ("False merges", f"{model.metrics['false_merge_rate']:.1%}"), ("Unknown rejected", f"{model.metrics['unknown_rejection_rate']:.1%}")]
    for column, (label, value) in zip(st.columns(6), metrics): column.metric(label, value)
    left, right = st.columns(2)
    with left:
        long = model.drift.melt(id_vars=["corruption", "queries", "mean_score"], value_vars=["top1_accuracy", "coverage"], var_name="metric", value_name="rate")
        fig = px.bar(long, x="corruption", y="rate", color="metric", barmode="group", title="Held-out recovery by controlled failure mode")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05]); st.plotly_chart(_style(fig), width="stretch")
    with right:
        test = model.evaluation[model.evaluation.split == "test"]
        fig = px.scatter(test, x="score", y="margin", color="route", symbol="correct", hover_data=["query_name", "predicted_name", "corruption"], color_discrete_map={"auto-link": MINT, "human-review": GOLD}, title="Similarity, ambiguity and routing")
        fig.add_vline(x=model.thresholds["score"], line_dash="dash"); fig.add_hline(y=model.thresholds["margin"], line_dash="dot")
        st.plotly_chart(_style(fig), width="stretch")
    st.markdown("#### Evaluation audit")
    audit = model.evaluation[model.evaluation.split == "test"][["query_name", "canonical_name", "predicted_name", "corruption", "score", "margin", "rank", "route", "correct"]].sort_values(["route", "score"]).head(100)
    st.dataframe(audit, hide_index=True, width="stretch")

    st.markdown('<section class="section"><small>Resolution workbench</small><h2>Paste the supplier.<br>Inspect every candidate.</h2></section>', unsafe_allow_html=True)
    default_name = model.evaluation[model.evaluation.split == "test"].iloc[0].query_name
    query = st.text_input("Incoming supplier name", default_name, help="Try typos, missing legal suffixes, punctuation changes or an unknown supplier.")
    candidates = resolve_name(model, query)
    if candidates.empty:
        st.info("Enter a non-empty supplier name to request candidates.")
    else:
        top = candidates.iloc[0]
        columns = st.columns(4)
        columns[0].metric("Routing", str(top.route).replace("-", " ").title()); columns[1].metric("Best similarity", f"{top.score:.3f}"); columns[2].metric("Decision margin", f"{top.margin:.3f}"); columns[3].metric("Candidate UEI", top.recipient_uei)
        if top.route == "auto-link": st.success("The best candidate clears both calibrated gates. Keep the UEI and evidence in the downstream audit log.")
        else: st.warning("The service abstains: similarity or separation from the runner-up is insufficient. Review the candidates and authoritative UEI manually.")
        st.dataframe(candidates[["canonical_name", "recipient_uei", "score", "award_count", "total_award_value", "route"]], hide_index=True, width="stretch")
    downloads = st.columns(3)
    downloads[0].download_button("Export governed awards", product.awards.to_csv(index=False).encode(), f"procurement_awards_{product.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[1].download_button("Export match audit", model.evaluation.to_csv(index=False).encode(), f"entity_resolution_{product.metadata['run_id']}.csv", "text/csv", width="stretch")
    downloads[2].download_button("Export run manifest", _manifest(product, model), f"procurement_manifest_{product.metadata['run_id']}.json", "application/json", width="stretch")
    with st.expander("Source semantics, evaluation design and operational limits"):
        st.markdown(f"""**Source.** The product calls the official, keyless [USAspending Advanced Award Search API]({API}) documented in the [endpoint index]({DOCS}) and exact [request contract]({CONTRACT}). USAspending is the U.S. government's official open source for federal spending. This bounded snapshot requests procurement types A–D and fields including award ID, recipient name, UEI, amount, dates, awarding agency, NAICS, PSC, description and last-modified timestamp.

**Pipeline semantics.** The API filter selects awards overlapping the current federal fiscal year; `Last Modified Date` controls ordering, not event time. Bronze stores delivery rows and hashes. Silver parses types, quarantines failed keys/names/UEIs/amounts/dates and suppresses duplicate delivery IDs. Gold publishes one row per award plus a UEI-keyed recipient reference. Signed amounts are retained because contract modifications can reduce an award.

**Matching evaluation.** The official UEI is the ground-truth identity. Five deterministic defects are generated from each canonical name: spacing/case loss, legal-suffix removal, token swap, one-character loss and abbreviation. UEI groups are hashed into calibration or test sets. A character 3–5-gram TF-IDF index returns cosine candidates. Calibration selects similarity and runner-up-margin gates subject to 95% selective accuracy; held-out Top-1, Hit@5, MRR@5, coverage, false merges and unknown rejection are then reported. Exact normalized equality is the baseline.

**Operational limits.** Synthetic corruption is a controlled reliability test, not a substitute for labeled historical linkage decisions. The reference snapshot may omit inactive or newly registered suppliers, aliases, parents and subsidiaries. Matching names must never merge legal entities without authoritative UEI evidence. Amounts are current award totals, not necessarily current-year obligations. USAspending notes reporting lags and source-specific caveats in its [About the Data disclosure]({ABOUT}); procurement data are generally available within five days, while certain Defense/USACE records can be delayed.""")
