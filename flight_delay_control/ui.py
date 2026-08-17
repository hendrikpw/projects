"""Operational UI for the governed flight-delay product."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from flight_delay_control.src.data import DEFINITION,DOCS,FIELDS,URL
from flight_delay_control.src.model import score_scenario,train_and_evaluate
from flight_delay_control.src.pipeline import run_pipeline

INK="#07131f"; SKY="#43c7ff"; MINT="#5ce1b7"; AMBER="#ffbd59"; RED="#ff6577"
def _style(fig,height=410):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=60,b=25),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color="#eef8ff",family="Inter, sans-serif"),legend=dict(orientation="h",y=1.13),colorway=[SKY,MINT,AMBER,RED])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.07)"); return fig
@st.cache_resource(ttl=21600,show_spinner=False)
def _load():
    product=run_pipeline(); return product,train_and_evaluate(product.gold)
def _manifest(p,m): return json.dumps({"pipeline":p.metadata,"model":m.metadata,"metrics":m.metrics,"quality":p.quality.to_dict("records")},indent=2,default=str).encode()

def render_dashboard():
    st.markdown("""<style>.flight-hero{padding:3.5rem 3rem;border-radius:28px;background:radial-gradient(circle at 88% 15%,rgba(67,199,255,.24),transparent 30%),radial-gradient(circle at 84% 90%,rgba(92,225,183,.12),transparent 30%),linear-gradient(145deg,#10283b,#06101a);border:1px solid rgba(67,199,255,.24);margin-bottom:1.4rem}.flight-kicker{color:#43c7ff;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.flight-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#f6fbff;letter-spacing:-.06em;max-width:960px;margin:.8rem 0}.flight-hero p{max-width:820px;color:#afc4d3;font-size:1.05rem}.boundary{border-left:3px solid #ffbd59;background:rgba(255,189,89,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:165px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #43c7ff;border-radius:0 0 14px 14px}.stage b{color:#43c7ff;font-size:.76rem;letter-spacing:.08em}.stage p{color:#afc2cf;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#43c7ff;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style><section class="flight-hero"><div class="flight-kicker">BTS aviation / Data + AI Engineering</div><h1>Govern every flight.<br>Focus the review.</h1><p>A bounded monthly archive becomes a replay-safe flight data product, then a calibrated, time-split delay-risk model turns schedule-known facts into an auditable operations queue.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Validating the BTS archive, reconciling deliveries and evaluating the held-out model …"): p,m=_load()
    except Exception as exc: st.error("No flight product was published because a data or model gate failed."); st.exception(exc); return
    if p.metadata["mode"]=="demo": st.warning(f"Deterministic demonstration flights are active: {p.metadata['fallback_reason']}")
    else: st.success("Official BTS June 2026 archive loaded · all ten publication gates passed")
    vals=[("Unique flights",f"{len(p.silver):,}"),("Operated labels",f"{len(p.gold):,}"),("Replays suppressed",f"{p.metadata['duplicates']:,}"),("Model AP",f"{m.metrics['average_precision']:.3f}"),("Baseline AP",f"{m.metrics['baseline_average_precision']:.3f}"),("Recall @ 10%",f"{m.metrics['recall_at_10pct']:.1%}")]
    for c,(a,b) in zip(st.columns(6),vals): c.metric(a,b)
    st.caption(f"Run {p.metadata['run_id']} · {m.metadata['train_days']} train · {m.metadata['calibration_days']} calibrate · {m.metadata['test_days']} test")
    st.markdown('<div class="boundary"><b>Decision boundary:</b> this model prioritizes operational review before departure. It does not predict an official flight status, passenger outcome or cause. Probability is conditional on this sample and schedule-known fields only.</div>',unsafe_allow_html=True)
    st.markdown('<section class="section"><small>Data engineering control plane</small><h2>Bound the archive.<br>Publish with proof.</h2></section>',unsafe_allow_html=True)
    desc=[("01 · EXTRACT","Safety-bounded ZIP download, explicit User-Agent, timeout, member validation and atomic deterministic fallback."),("02 · BRONZE","Payload hashes, stable event IDs and injected replay deliveries preserve source-level lineage."),("03 · SILVER","Typed contracts, HHMM/range checks, quarantine and idempotent natural-key reconciliation."),("04 · GOLD","Only operated flights with valid 15-minute labels; all post-arrival outcomes excluded from model features.")]
    for c,(a,b) in zip(st.columns(4),desc): c.markdown(f'<div class="stage"><b>{a}</b><p>{b}</p></div>',unsafe_allow_html=True)
    l,r=st.columns([1.05,.95])
    with l:
        fig=px.bar(p.stages,x="stage",y="output",color="stage",text="output",hover_data=["input","rejected","duration_ms","hash"],title="Layer volumes and content hashes"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with r: st.markdown("#### Run ledger"); st.dataframe(p.stages,hide_index=True,width="stretch")
    l,r=st.columns(2)
    with l: st.markdown("#### Publication gates"); st.dataframe(p.quality.assign(result=p.quality.passed.map({True:"Passed",False:"Failed"}))[["check","result","detail"]],hide_index=True,width="stretch")
    with r:
        ops=pd.DataFrame({"state":["Operated","Cancelled","Diverted"],"flights":[int(p.silver.is_operated.sum()),p.metadata["cancelled"],p.metadata["diverted"]]}); fig=px.bar(ops,x="state",y="flights",color="state",title="Governed operational states"); st.plotly_chart(_style(fig),width="stretch")
    st.markdown('<section class="section"><small>AI engineering evaluation</small><h2>Rank tomorrow’s work.<br>Audit today’s errors.</h2></section>',unsafe_allow_html=True)
    vals=[("Average precision",f"{m.metrics['average_precision']:.3f}"),("Prevalence baseline",f"{m.metrics['baseline_average_precision']:.3f}"),("ROC-AUC",f"{m.metrics['roc_auc']:.3f}"),("Brier score",f"{m.metrics['brier']:.3f}"),("Test event rate",f"{m.metrics['event_rate']:.1%}"),("Review routed",f"{m.metrics['review_rows']:,}")]
    for c,(a,b) in zip(st.columns(6),vals): c.metric(a,b)
    l,r=st.columns(2)
    with l:
        q=m.evaluation.sort_values("ranking_score",ascending=False).reset_index(drop=True); q["review_share"]=(q.index+1)/len(q); q["capture"]=q.is_delayed_15.cumsum()/q.is_delayed_15.sum(); fig=go.Figure(); fig.add_scatter(x=q.review_share,y=q.capture,name="Candidate"); fig.add_scatter(x=[0,1],y=[0,1],name="Random",line=dict(dash="dash")); fig.add_vline(x=.1,line_dash="dot"); fig.update_layout(title="Delay capture by review budget",xaxis_title="Share reviewed",yaxis_title="Delayed flights captured"); st.plotly_chart(_style(fig),width="stretch")
    with r:
        cal=m.evaluation.assign(bin=pd.cut(m.evaluation.probability,np.linspace(0,1,11),include_lowest=True)).groupby("bin",observed=True).agg(predicted=("probability","mean"),observed=("is_delayed_15","mean"),rows=("probability","size")).reset_index(); fig=go.Figure(); fig.add_scatter(x=[0,1],y=[0,1],name="Perfect",line=dict(dash="dash")); fig.add_scatter(x=cal.predicted,y=cal.observed,mode="lines+markers",marker=dict(size=np.sqrt(cal.rows)*1.8),name="Held-out bins"); fig.update_layout(title="Held-out calibration",xaxis_title="Predicted",yaxis_title="Observed"); st.plotly_chart(_style(fig),width="stretch")
    l,r=st.columns(2)
    with l: st.markdown("#### Carrier evaluation"); st.dataframe(m.carrier_metrics.sort_values("flights",ascending=False),hide_index=True,width="stretch")
    with r:
        fig=px.bar(m.drift,x="psi",y="feature",orientation="h",color="status",color_discrete_map={"stable":MINT,"watch":AMBER,"high":RED},title="Train-to-test feature drift · PSI"); fig.add_vline(x=.1,line_dash="dot"); fig.add_vline(x=.25,line_dash="dash"); st.plotly_chart(_style(fig,470),width="stretch")
    st.markdown("#### Held-out operations queue"); st.dataframe(m.evaluation.sort_values("probability",ascending=False).head(60),hide_index=True,width="stretch")
    st.markdown('<section class="section"><small>Scenario workbench</small><h2>Change the schedule.<br>Inspect the route.</h2></section>',unsafe_allow_html=True)
    base=m.evaluation.sort_values("probability",ascending=False).iloc[len(m.evaluation)//3]; carrier=st.selectbox("Carrier",sorted(p.gold.Reporting_Airline.unique()),index=0); origin=st.selectbox("Origin",sorted(p.gold.Origin.unique()),index=0); hour=st.slider("Scheduled departure hour",0,23,int(base.scheduled_hour)); distance=st.slider("Distance (miles)",50,5000,int(base.Distance),50)
    result=score_scenario(m,p.gold.loc[base.name],{"Reporting_Airline":carrier,"Origin":origin,"scheduled_hour":hour,"Distance":distance})
    cs=st.columns(4); cs[0].metric("Risk probability",f"{result['probability']:.1%}"); cs[1].metric("Operations route",result["status"].title()); cs[2].metric("Carrier",carrier); cs[3].metric("Origin",origin)
    if result["status"]=="review": st.warning("Scenario enters review. Investigate current weather, network conditions and official carrier status before acting.")
    elif result["status"]=="watch": st.info("Scenario enters watch. The score is a prioritization signal, not a prediction guarantee.")
    else: st.success("Scenario remains in monitor. A low score does not guarantee an on-time arrival.")
    dl=st.columns(3); dl[0].download_button("Export governed flights",p.silver.to_csv(index=False).encode(),f"flight_silver_{p.metadata['run_id']}.csv","text/csv",width="stretch"); dl[1].download_button("Export model audit",m.evaluation.to_csv(index=False).encode(),f"flight_model_{p.metadata['run_id']}.csv","text/csv",width="stretch"); dl[2].download_button("Export manifest",_manifest(p,m),f"flight_manifest_{p.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Source, label, evaluation and limitations"):
        st.markdown(f"""**Source.** The app downloads the official [BTS June 2026 on-time archive]({URL}). See the [database description]({DOCS}) and [field reference]({FIELDS}). BTS defines an arrival as delayed when it is [15 minutes or more after schedule]({DEFINITION}).

**Features.** Carrier, origin, destination, day of week, scheduled departure hour, scheduled elapsed time and distance are available before departure. Actual arrival delay, cancellation and diversion outcomes never enter the feature matrix.

**Evaluation.** Days 1–18 train the gradient-boosted classifier, days 19–24 fit isotonic probability calibration, and days 25–30 remain untouched testing. Average precision measures rare-event ranking; Brier score measures probability error; recall at 10% measures capture under a fixed review budget. Unknown/rare airports collapse into a training-derived `__OTHER__` category.

**Limits.** The bounded hash sample is reproducible but not a full-population estimate. The model has no live weather, aircraft rotation, crew or air-traffic constraints; June is one seasonal window. Archive data can be revised. Review status is operational triage, never an official flight-status claim.""")
