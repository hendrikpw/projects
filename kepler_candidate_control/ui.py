"""Modern Streamlit control plane for Kepler candidate reliability."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kepler_candidate_control.src.data import DOCS,FIELDS,KOI_DOCS,TAP
from kepler_candidate_control.src.model import score_candidate,train_and_evaluate
from kepler_candidate_control.src.pipeline import FEATURES,run_pipeline

VOID="#070916"; VIOLET="#a78bfa"; CYAN="#4de4ff"; MINT="#63efbf"; AMBER="#ffc857"; RED="#ff6577"
def _style(fig,height=410):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=60,b=25),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color="#f4f1ff",family="Inter, sans-serif"),legend=dict(orientation="h",y=1.13),colorway=[VIOLET,CYAN,MINT,AMBER,RED]); fig.update_xaxes(gridcolor="rgba(255,255,255,.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.07)"); return fig
@st.cache_resource(ttl=21600,show_spinner=False)
def _load():
    product=run_pipeline(); return product,train_and_evaluate(product.gold)
def _manifest(p,m): return json.dumps({"pipeline":p.metadata,"model":m.metadata,"metrics":m.metrics,"quality":p.quality.to_dict("records")},indent=2,default=str).encode()

def render_dashboard():
    st.markdown("""<style>.koi-hero{padding:3.6rem 3rem;border-radius:28px;background:radial-gradient(circle at 87% 15%,rgba(167,139,250,.28),transparent 30%),radial-gradient(circle at 82% 88%,rgba(77,228,255,.13),transparent 31%),linear-gradient(145deg,#17112f,#070916);border:1px solid rgba(167,139,250,.3);margin-bottom:1.4rem}.kicker{color:#a78bfa;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.koi-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#fbfaff;letter-spacing:-.06em;max-width:980px;margin:.8rem 0}.koi-hero p{max-width:850px;color:#c2bbd9;font-size:1.05rem}.boundary{border-left:3px solid #ffc857;background:rgba(255,200,87,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:170px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #a78bfa;border-radius:0 0 14px 14px}.stage b{color:#a78bfa;font-size:.76rem;letter-spacing:.08em}.stage p{color:#bcb6ce;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#a78bfa;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style><section class="koi-hero"><div class="kicker">NASA Exoplanet Archive / Data + AI Engineering</div><h1>Trace the signal.<br>Challenge the candidate.</h1><p>A content-addressed TAP pipeline governs every Kepler Object of Interest, while a calibrated, star-isolated vetting model measures planet-like reliability, uncertainty, drift and out-of-distribution risk.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Querying NASA TAP, validating KOIs and evaluating the star-isolated model …"): p,m=_load()
    except Exception as exc: st.error("No candidate product was published because a pipeline or model gate failed."); st.exception(exc); return
    if p.metadata["mode"]=="demo": st.warning(f"Deterministic demonstration KOIs are active: {p.metadata['fallback_reason']}")
    else: st.success("NASA KOI cumulative delivery loaded · all ten publication gates passed")
    vals=[("Unique KOIs",f"{len(p.silver):,}"),("Quarantined",f"{p.metadata['quarantine']:,}"),("Replays suppressed",f"{p.metadata['duplicates']:,}"),("Model AP",f"{m.metrics['average_precision']:.3f}"),("Baseline AP",f"{m.metrics['baseline_average_precision']:.3f}"),("ROC-AUC",f"{m.metrics['roc_auc']:.3f}")]
    for c,(a,b) in zip(st.columns(6),vals): c.metric(a,b)
    st.caption(f"Run {p.metadata['run_id']} · {m.metadata['train_rows']:,} train · {m.metadata['calibration_rows']:,} calibrate · {m.metadata['test_rows']:,} test")
    st.markdown('<div class="boundary"><b>Scientific boundary:</b> the score reproduces current archive dispositions from selected catalog measurements. It does not confirm a planet, replace light-curve analysis or estimate occurrence rates. Human scientific vetting remains authoritative.</div>',unsafe_allow_html=True)
    st.markdown('<section class="section"><small>Data engineering control plane</small><h2>Contract the catalog.<br>Preserve its provenance.</h2></section>',unsafe_allow_html=True)
    cards=[("01 · TAP EXTRACT","Current SQL-style TAP query, explicit schema, timeout, response-size and row-count bounds, plus atomic fallback."),("02 · BRONZE","Raw catalog fields, deterministic event IDs, row payload hashes and 20 replay deliveries prove traceability."),("03 · SILVER","Typed quantities, coordinate and physical contracts, quarantine reasons and KOI-level idempotency."),("04 · GOLD","Vetting-ready physical measurements with disposition labels; archive score and pipeline disposition are blocked from features.")]
    for c,(a,b) in zip(st.columns(4),cards): c.markdown(f'<div class="stage"><b>{a}</b><p>{b}</p></div>',unsafe_allow_html=True)
    l,r=st.columns([1.05,.95])
    with l:
        fig=px.bar(p.stages,x="stage",y="output",color="stage",text="output",hover_data=["input","rejected","duration_ms","hash"],title="Layer volumes and content-addressed lineage"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with r: st.markdown("#### Run ledger"); st.dataframe(p.stages,hide_index=True,width="stretch")
    l,r=st.columns(2)
    with l: st.markdown("#### Publication gates"); st.dataframe(p.quality.assign(result=p.quality.passed.map({True:"Passed",False:"Failed"}))[["check","result","detail"]],hide_index=True,width="stretch")
    with r:
        states=p.silver.koi_disposition.value_counts().rename_axis("disposition").reset_index(name="KOIs"); fig=px.bar(states,x="disposition",y="KOIs",color="disposition",title="Governed archive dispositions"); st.plotly_chart(_style(fig),width="stretch")
    if len(p.quarantine):
        reasons=p.quarantine.reason.value_counts().rename_axis("reason").reset_index(name="rows"); st.markdown("#### Quarantine audit"); st.dataframe(reasons,hide_index=True,width="stretch")
    st.markdown('<section class="section"><small>AI engineering evaluation</small><h2>Separate the stars.<br>Calibrate the doubt.</h2></section>',unsafe_allow_html=True)
    vals=[("Average precision",f"{m.metrics['average_precision']:.3f}"),("Prevalence baseline",f"{m.metrics['baseline_average_precision']:.3f}"),("ROC-AUC",f"{m.metrics['roc_auc']:.3f}"),("Brier score",f"{m.metrics['brier']:.3f}"),("Recall @ 10%",f"{m.metrics['recall_at_10pct']:.1%}"),("Uncertain route",f"{m.metrics['uncertain_rows']:,}")]
    for c,(a,b) in zip(st.columns(6),vals): c.metric(a,b)
    l,r=st.columns(2)
    with l:
        q=m.evaluation.sort_values("ranking_score",ascending=False).reset_index(drop=True); q["review_share"]=(q.index+1)/len(q); q["capture"]=q.planet_like.cumsum()/q.planet_like.sum(); fig=go.Figure(); fig.add_scatter(x=q.review_share,y=q.capture,name="Candidate model"); fig.add_scatter(x=[0,1],y=[0,1],name="Random",line=dict(dash="dash")); fig.add_vline(x=.1,line_dash="dot"); fig.update_layout(title="Planet-like capture by review budget",xaxis_title="Share reviewed",yaxis_title="Planet-like KOIs captured"); st.plotly_chart(_style(fig),width="stretch")
    with r:
        cal=m.evaluation.assign(bin=pd.cut(m.evaluation.probability,np.linspace(0,1,11),include_lowest=True)).groupby("bin",observed=True).agg(predicted=("probability","mean"),observed=("planet_like","mean"),rows=("probability","size")).reset_index(); fig=go.Figure(); fig.add_scatter(x=[0,1],y=[0,1],name="Perfect",line=dict(dash="dash")); fig.add_scatter(x=cal.predicted,y=cal.observed,mode="lines+markers",marker=dict(size=np.sqrt(cal.rows)*2),name="Held-out bins"); fig.update_layout(title="Held-out probability reliability",xaxis_title="Predicted",yaxis_title="Observed"); st.plotly_chart(_style(fig),width="stretch")
    l,r=st.columns(2)
    with l:
        fig=px.bar(m.importance.head(10).sort_values("ap_drop"),x="ap_drop",y="feature",orientation="h",title="Permutation importance · AP decrease"); st.plotly_chart(_style(fig,470),width="stretch")
    with r:
        fig=px.bar(m.drift.sort_values("psi"),x="psi",y="feature",orientation="h",color="status",color_discrete_map={"stable":MINT,"watch":AMBER,"high":RED},title="Star-group train-to-test drift · PSI"); fig.add_vline(x=.1,line_dash="dot"); fig.add_vline(x=.25,line_dash="dash"); st.plotly_chart(_style(fig,470),width="stretch")
    st.markdown("#### Held-out vetting audit"); st.dataframe(m.evaluation.sort_values("probability").head(60),hide_index=True,width="stretch")
    st.markdown('<section class="section"><small>Candidate workbench</small><h2>Stress the transit.<br>Inspect the guardrails.</h2></section>',unsafe_allow_html=True)
    idx=st.selectbox("Reference KOI",m.evaluation.sort_values("probability").kepoi_name.tolist()); row=p.gold[p.gold.kepoi_name==idx].iloc[0]; period=st.number_input("Orbital period (days)",.01,1000.0,float(row.koi_period)); radius=st.number_input("Planet radius (Earth radii)",.01,500.0,float(row.koi_prad)); depth=st.number_input("Transit depth (ppm)",.1,1_000_000.0,float(row.koi_depth)); snr=st.number_input("Transit model SNR",.1,10_000.0,float(row.koi_model_snr)); result=score_candidate(m,row,{"koi_period":period,"koi_prad":radius,"koi_depth":depth,"koi_model_snr":snr})
    cs=st.columns(4); cs[0].metric("Planet-like probability",f"{result['probability']:.1%}"); cs[1].metric("Routing",result["route"].replace("-"," ").title()); cs[2].metric("Reference disposition",row.koi_disposition.title()); cs[3].metric("OOD features",len(result["outside_features"]))
    if result["route"]=="ood-review": st.error("Two or more values fall outside the central training ranges. The prediction is withheld for explicit out-of-distribution review.")
    elif "review" in result["route"]: st.warning("The case is routed to human review because the model is uncertain or strongly rejects the planet-like profile.")
    else: st.success("The profile is routed planet-like. This remains model triage—not confirmation.")
    if result["outside_features"]: st.caption("Outside training guardrails: "+", ".join(result["outside_features"]))
    dl=st.columns(3); dl[0].download_button("Export governed KOIs",p.silver.to_csv(index=False).encode(),f"kepler_silver_{p.metadata['run_id']}.csv","text/csv",width="stretch"); dl[1].download_button("Export model audit",m.evaluation.to_csv(index=False).encode(),f"kepler_model_{p.metadata['run_id']}.csv","text/csv",width="stretch"); dl[2].download_button("Export manifest",_manifest(p,m),f"kepler_manifest_{p.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Source semantics, evaluation and scientific limits"):
        st.markdown(f"""**Source.** The application queries the official [NASA Exoplanet Archive TAP service]({DOCS}) and its `cumulative` Kepler Objects of Interest table. See the [KOI documentation]({KOI_DOCS}) and exact [column definitions]({FIELDS}). The service is public and keyless; NASA/IPAC-Caltech attribution is retained.

**Target.** `CONFIRMED` and `CANDIDATE` are grouped as planet-like; `FALSE POSITIVE` is the alternative. This is a catalog-disposition reproduction task, not a new scientific confirmation. The Robovetter score, pipeline disposition, false-positive flags and names are excluded from features.

**Isolation.** All KOIs belonging to one `kepid` star stay in exactly one deterministic SHA-256 split: 60% training, 20% calibration and 20% test. That prevents sibling signals from the same star crossing evaluation boundaries.

**Metrics.** Average precision measures ranking against the planet-like prevalence baseline. ROC-AUC measures pairwise discrimination; Brier measures calibrated probability error; recall at 10% describes capture under a fixed review budget. Permutation importance is test-set sensitivity, not causality.

**Limits.** The cumulative table combines current results from several Kepler activities and can change. Selected catalog measurements omit raw light curves, centroid diagnostics, imaging, spectroscopy and domain review reports. Missing values are median-imputed. OOD guardrails detect extreme catalog values, but cannot certify scientific validity or dataset suitability for occurrence-rate studies.""")
