"""Polished Streamlit control plane for message trust operations."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve

from message_trust_gateway.src.data import DATASET_URL,DOI_URL
from message_trust_gateway.src.model import score_message,train_and_evaluate
from message_trust_gateway.src.pipeline import run_pipeline

LIME="#b6f36b"; VIOLET="#9674ff"; RED="#ff647c"; CYAN="#50d9e7"


def _style(fig,height=410):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=58,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color="#f1f5fb",family="Inter, sans-serif"),title_font=dict(size=18),legend=dict(orientation="h",y=1.12),colorway=[LIME,VIOLET,RED,CYAN]); fig.update_xaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); fig.update_yaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); return fig


@st.cache_resource(ttl=86_400,show_spinner=False)
def _load():
    data=run_pipeline(); model=train_and_evaluate(data.gold); return data,model


def _manifest(data,model): return json.dumps({"run":data.metadata,"model":model.metadata,"metrics":model.metrics,"quality":data.quality.to_dict("records")},indent=2,default=str).encode()


def render_dashboard():
    st.markdown("""<style>.trust-hero{padding:3.5rem 3rem;border-radius:26px;border:1px solid rgba(182,243,107,.25);background:radial-gradient(circle at 86% 15%,rgba(182,243,107,.18),transparent 30%),radial-gradient(circle at 95% 90%,rgba(150,116,255,.2),transparent 30%),linear-gradient(135deg,#11151c,#07090d);margin-bottom:1.4rem}.trust-kicker{color:#b6f36b;font-size:.75rem;font-weight:750;letter-spacing:.16em;text-transform:uppercase}.trust-hero h1{font:800 clamp(2.8rem,6vw,5.8rem)/.9 Inter;color:#f8fbff;letter-spacing:-.065em;max-width:900px;margin:.8rem 0}.trust-hero p{max-width:740px;color:#b4bdc9;font-size:1.05rem}.trust-stage{min-height:165px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #b6f36b;border-radius:0 0 14px 14px}.trust-stage b{font-size:.77rem;color:#b6f36b;letter-spacing:.08em}.trust-stage p{font-size:.9rem;color:#b4bdc9}.trust-boundary{border-left:3px solid #9674ff;background:rgba(150,116,255,.09);padding:1rem 1.2rem;border-radius:0 10px 10px 0;color:#d3d8e1}.section-intro{padding-top:2.8rem}.section-kicker{color:#b6f36b;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase}.section-intro h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style>""",unsafe_allow_html=True)
    st.markdown("""<section class="trust-hero"><div class="trust-kicker">Privacy-aware NLP / Data + AI Engineering</div><h1>Route uncertain messages safely.</h1><p>An idempotent ingestion pipeline groups duplicates and tokenizes contact patterns before a word-and-character model chooses allow, block or human review—with calibration, adversarial evaluation and drift visible.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Validating the corpus, building privacy-safe features and running group-isolated evaluation …"): data,model=_load()
    except Exception as exc: st.error("No gateway release was published because a data or model gate failed."); st.exception(exc); return
    if data.metadata["mode"]=="demo": st.warning(f"Deterministic demo messages are active because UCI could not be loaded: {data.metadata['fallback_reason']}")
    else: st.success("Live UCI archive verified · duplicate groups isolated · privacy and model gates passed")
    c=st.columns(6); vals=[("Messages",f"{len(data.gold):,}"),("Spam",f"{data.gold.target.sum():,}"),("Replay suppressed",f"{data.metadata['replayed_deliveries']:,}"),("AUCPR",f"{model.metrics['average_precision']:.3f}"),("Auto coverage",f"{model.metrics['auto_coverage']:.1%}"),("Review queue",f"{model.metrics['review_rate']:.1%}")]
    for col,(label,value) in zip(c,vals): col.metric(label,value)
    st.caption(f"Run {data.metadata['run_id']} · duplicate-group hash split 70/15/15 · {model.metadata['train_groups']:,}/{model.metadata['calibration_groups']:,}/{model.metadata['test_groups']:,} groups · seed 42")
    st.markdown('<div class="trust-boundary"><b>Safety boundary:</b> This is an offline English-SMS benchmark from 2012, not a production moderation system. “Allow” never proves a message is safe and “block” can be wrong. Real deployment requires current multilingual data, abuse monitoring, appeals and policy/legal review.</div>',unsafe_allow_html=True)

    st.markdown('<section class="section-intro"><div class="section-kicker">Data engineering control plane</div><h2>Make replay harmless.<br>Publish less personal text.</h2></section>',unsafe_allow_html=True)
    stages=st.columns(4); copy=[("01 · EXTRACT","Download one bounded ZIP with retry, timeout, exact member allowlist, path and expanded-size guards."),("02 · REPLAY","Process deterministic 500-row batches with payload hashes and event identity for idempotency audit."),("03 · SILVER","Normalize Unicode and whitespace, validate labels, assign message IDs and exact-text duplicate groups."),("04 · GOLD","Replace URL, email, phone and money patterns with typed tokens; publish model text and drift features.")]
    for col,(title,body) in zip(stages,copy): col.markdown(f'<div class="trust-stage"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
    a,b=st.columns([1.05,.95])
    with a:
        fig=px.bar(data.stages,x="stage",y="output_rows",color="stage",text="output_rows",hover_data=["input_rows","rejected_rows","duration_ms","content_hash"],title="Layer volume and content lineage"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with b: st.markdown("#### Stage ledger"); st.dataframe(data.stages,hide_index=True,width="stretch")
    a,b=st.columns(2)
    with a:
        qa=data.quality.assign(result=data.quality.passed.map({True:"Passed",False:"Failed"})); st.markdown("#### Data, privacy and split gates"); st.dataframe(qa[["check","result","detail"]],hide_index=True,width="stretch")
    with b:
        st.markdown("#### Batch replay observability"); st.dataframe(data.batches,hide_index=True,width="stretch")
    layer=st.radio("Inspect governed output",["Gold privacy-safe messages","Silver lineage","Quarantine"],horizontal=True); frame=data.gold.head(18) if layer.startswith("Gold") else data.silver[["message_id","group_hash","label","message_normalized"]].head(18) if layer.startswith("Silver") else data.quarantine.head(18); st.dataframe(frame if len(frame) else pd.DataFrame({"state":["No quarantined events"]}),hide_index=True,width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">AI engineering evaluation</div><h2>Calibrate confidence.<br>Attack the model before release.</h2></section>',unsafe_allow_html=True)
    m=st.columns(7); values=[("Robust AUCPR",f"{model.metrics['average_precision']:.3f}"),("Word baseline",f"{model.metrics['baseline_average_precision']:.3f}"),("ROC-AUC",f"{model.metrics['roc_auc']:.3f}"),("Brier",f"{model.metrics['brier']:.3f}"),("Precision",f"{model.metrics['precision']:.1%}"),("Recall",f"{model.metrics['recall']:.1%}"),("Adversarial AP",f"{model.metrics['adversarial_ap']:.3f}")]
    for col,(label,value) in zip(m,values): col.metric(label,value)
    a,b=st.columns(2)
    with a:
        true,pred=calibration_curve(model.evaluation.target,model.evaluation.spam_probability,n_bins=8,strategy="quantile"); fig=go.Figure(); fig.add_scatter(x=pred,y=true,mode="lines+markers",name="Calibrated gateway"); fig.add_scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash"),name="Perfect"); fig.update_layout(title="Reliability on held-out duplicate groups",xaxis_title="Predicted spam probability",yaxis_title="Observed spam rate"); st.plotly_chart(_style(fig),width="stretch")
    with b:
        comparison=pd.DataFrame({"condition":["Clean block recall","Obfuscated block recall","Clean AP","Obfuscated AP"],"score":[model.metrics["blocked_spam_recall"],model.metrics["adversarial_blocked_recall"],model.metrics["average_precision"],model.metrics["adversarial_ap"]]}); fig=px.bar(comparison,x="condition",y="score",color="condition",title="Adversarial replacement test",range_y=[0,1]); st.plotly_chart(_style(fig),width="stretch")
    a,b=st.columns(2)
    with a:
        fig=px.bar(model.features.sort_values("weight"),x="weight",y="feature",orientation="h",color="direction",title="Largest global n-gram weights",color_discrete_map={"spam":RED,"ham":CYAN}); st.plotly_chart(_style(fig,520),width="stretch")
    with b:
        fig=px.bar(model.drift.sort_values("psi"),x="psi",y="feature",orientation="h",color="status",title="Train-to-test input drift · PSI",color_discrete_map={"stable":CYAN,"watch":"#ffbf55","high":RED}); fig.add_vline(x=.1,line_dash="dot"); fig.add_vline(x=.25,line_dash="dash"); st.plotly_chart(_style(fig,520),width="stretch")
    st.caption(f"Allow ≤ {model.metrics['low_threshold']:.2f} · review between thresholds · block ≥ {model.metrics['high_threshold']:.2f}. Thresholds are selected only on calibration groups for 99% ham precision and 95% spam precision where feasible. Adversarial testing substitutes common spam words such as free → fr33 and claim → cla1m.")

    st.markdown('<section class="section-intro"><div class="section-kicker">Inference workbench</div><h2>Submit one message.<br>Inspect the route and evidence.</h2></section>',unsafe_allow_html=True)
    text=st.text_area("Message",value="URGENT! You have won a free £1000 prize. Call +44 7700 900123 now!",height=120,max_chars=2_000); result=score_message(model,text); a,b,c=st.columns(3); a.metric("Spam probability",f"{result['spam_probability']:.1%}"); b.metric("Route",result["decision"].title()); c.metric("Privacy-safe serving text",f"{len(result['redacted_text'])} characters")
    if result["decision"]=="block": st.warning("Block candidate. A production system must preserve an appeal/review path and record the policy version.")
    elif result["decision"]=="review": st.warning("Uncertain message routed to human review. No automated final judgment is made.")
    else: st.info("Auto-allow candidate. This is not a guarantee that the message or linked content is safe.")
    left,right=st.columns([1.1,.9]); left.text_area("Tokenized text sent to the model",result["redacted_text"],height=120,disabled=True); right.dataframe(result["evidence"],hide_index=True,width="stretch")
    d=st.columns(3); d[0].download_button("Export Gold contract",data.gold.to_csv(index=False).encode(),f"message_gold_{data.metadata['run_id']}.csv","text/csv",width="stretch"); d[1].download_button("Export test audit",model.evaluation.to_csv(index=False).encode(),f"message_eval_{data.metadata['run_id']}.csv","text/csv",width="stretch"); d[2].download_button("Export manifest",_manifest(data,model),f"message_manifest_{data.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Source, model lifecycle, rights and limitations"):
        st.markdown(f"""**Source.** [UCI SMS Spam Collection]({DATASET_URL}), DOI [10.24432/C5CC84]({DOI_URL}), 5,574 labeled messages donated in 2012. The corpus combines several earlier public/research collections and is licensed CC BY 4.0.

**Split.** UCI states messages are not chronologically sorted. Exact normalized text hashes therefore define groups; a stable hash assigns whole groups to 70% train, 15% calibration or 15% test. This prevents identical messages crossing the evaluation boundary without pretending the split is temporal.

**Model.** Logistic regression combines word 1–2 grams and character 3–5 grams. A word-only model is the baseline. Platt calibration and allow/review/block thresholds use only calibration groups. Test groups measure AUCPR, ROC-AUC, Brier, precision, recall, F1, policy coverage and adversarial degradation.

**Privacy.** Regex tokenization reduces direct exposure of URLs, emails, phone numbers and monetary strings; it is not complete anonymization. Raw Bronze/Silver text exists in this demonstration's in-memory pipeline and must be access-controlled and retention-limited in production.

**Limits.** English SMS from multiple older sources does not represent modern messaging, phishing, other languages or real deployment prevalence. Labels and regexes can be wrong; character robustness is narrow; coefficients are associations, not causal explanations. Do not use this demo for consequential moderation.""")
