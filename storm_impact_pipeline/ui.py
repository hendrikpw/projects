"""Streamlit operations and model-control interface."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve

from storm_impact_pipeline.src.data import DATABASE_URL,FORMAT_URL
from storm_impact_pipeline.src.model import score_case,train_and_evaluate
from storm_impact_pipeline.src.pipeline import run_pipeline

INK="#07111f"; CYAN="#31d7c5"; AMBER="#ffb84d"; RED="#ff5e6c"


def _style(fig,height=410):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=58,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color="#eef7fb",family="Inter, sans-serif"),title_font=dict(size=18),legend=dict(orientation="h",y=1.12),colorway=[CYAN,AMBER,RED,"#7797ff"]); fig.update_xaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); fig.update_yaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); return fig


@st.cache_resource(ttl=86_400,show_spinner=False)
def _load():
    data=run_pipeline(); model=train_and_evaluate(data.gold); return data,model


def _manifest(data,model): return json.dumps({"run":data.metadata,"model":model.metadata,"metrics":model.metrics,"quality":data.quality.to_dict("records")},indent=2,default=str).encode()


def render_dashboard():
    st.markdown("""<style>.storm-hero{padding:3.5rem 3rem;border-radius:26px;border:1px solid rgba(49,215,197,.25);background:radial-gradient(circle at 84% 20%,rgba(49,215,197,.2),transparent 30%),radial-gradient(circle at 95% 80%,rgba(255,184,77,.15),transparent 30%),linear-gradient(135deg,#0d1b2a,#070c14);margin-bottom:1.5rem}.storm-kicker{color:#31d7c5;font-size:.75rem;font-weight:750;letter-spacing:.16em;text-transform:uppercase}.storm-hero h1{font:800 clamp(2.8rem,6vw,5.8rem)/.91 Inter;color:#f5fbff;letter-spacing:-.065em;max-width:900px;margin:.8rem 0}.storm-hero p{max-width:730px;color:#adbdc9;font-size:1.05rem}.storm-stage{min-height:165px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #31d7c5;border-radius:0 0 14px 14px}.storm-stage b{font-size:.77rem;color:#31d7c5;letter-spacing:.08em}.storm-stage p{font-size:.9rem;color:#adbdc9}.storm-boundary{border-left:3px solid #ffb84d;background:rgba(255,184,77,.08);padding:1rem 1.2rem;border-radius:0 10px 10px 0;color:#ccd8df}.section-intro{padding-top:2.8rem}.section-kicker{color:#31d7c5;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase}.section-intro h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style>""",unsafe_allow_html=True)
    st.markdown("""<section class="storm-hero"><div class="storm-kicker">Climate operations / Data + AI Engineering</div><h1>Turn storm reports into review priority.</h1><p>A revision-aware NOAA batch pipeline contracts monetary labels, then a two-stage hurdle model separates damage probability from damage magnitude—so scarce review capacity follows expected impact rather than raw event counts.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Discovering the latest NOAA revision, validating labels and evaluating the hurdle model …"): data,model=_load()
    except Exception as exc: st.error("Publication stopped because a data or model gate failed."); st.exception(exc); return
    if data.metadata["mode"]=="demo": st.warning(f"Deterministic demo data is active because NOAA could not be loaded: {data.metadata['fallback_reason']}")
    else: st.success(f"Live NOAA revision verified: {data.metadata['filename']} · all contracts passed")
    c=st.columns(6); vals=[("Reported events",f"{len(data.bronze):,}"),("Labeled events",f"{len(data.gold):,}"),("Damaged",f"{data.gold.has_damage.sum():,}"),("AUCPR",f"{model.metrics['average_precision']:.3f}"),("Brier",f"{model.metrics['brier']:.3f}"),("Top-10% capture",f"{model.metrics['damage_capture_at_capacity']:.1%}")]
    for col,(label,value) in zip(c,vals): col.metric(label,value)
    st.caption(f"Run {data.metadata['run_id']} · {model.metadata['train_months']} train / {model.metadata['calibration_months']} calibration / {model.metadata['test_months']} test · model seed 42")
    st.markdown('<div class="storm-boundary"><b>Decision boundary:</b> NOAA damage values are broad, unadjusted estimates assembled after events. This model prioritizes completed reports for analytical review; it does not forecast weather, dispatch responders, estimate insured loss or replace official warnings.</div>',unsafe_allow_html=True)

    st.markdown('<section class="section-intro"><div class="section-kicker">Revision-aware data product</div><h2>Verify the batch.<br>Refuse ambiguous labels.</h2></section>',unsafe_allow_html=True)
    stages=st.columns(4); copy=[("01 · DISCOVER","Parse NOAA's bulk directory and select the newest revision date for the contracted 2025 details file."),("02 · EXTRACT","Bound HTTP retries, compressed and expanded size, filename grammar and gzip signature."),("03 · SILVER","Parse identifiers and time; normalize K/M/B damage units; quarantine duplicates, coordinates and incomplete labels."),("04 · GOLD","Publish eight pre-outcome features plus monetary and hurdle targets with content-addressed lineage.")]
    for col,(title,body) in zip(stages,copy): col.markdown(f'<div class="storm-stage"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
    a,b=st.columns([1.05,.95])
    with a:
        fig=px.bar(data.stages,x="stage",y="output_rows",color="stage",text="output_rows",hover_data=["input_rows","rejected_rows","duration_ms","content_hash"],title="Layer volume, quarantine and lineage"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with b: st.markdown("#### Immutable stage ledger"); st.dataframe(data.stages,hide_index=True,width="stretch")
    a,b=st.columns(2)
    with a:
        qa=data.quality.assign(result=data.quality.passed.map({True:"Passed",False:"Failed"})); st.markdown("#### Quality and leakage gates"); st.dataframe(qa[["check","result","detail"]],hide_index=True,width="stretch")
    with b:
        reasons=data.quarantine.invalid_reason.value_counts().rename_axis("reason").reset_index(name="rows"); fig=px.bar(reasons.head(10),x="rows",y="reason",orientation="h",title="Reason-coded quarantine"); st.plotly_chart(_style(fig),width="stretch")
    layer=st.radio("Inspect governed layer",["Gold serving contract","Silver labeled events","Quarantine"],horizontal=True); frame=data.gold.head(18) if layer.startswith("Gold") else data.silver.head(18) if layer.startswith("Silver") else data.quarantine.head(18); st.dataframe(frame if len(frame) else pd.DataFrame({"state":["No rejected records"]}),hide_index=True,width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">Hurdle-model evaluation</div><h2>Probability first.<br>Magnitude only when damage occurs.</h2></section>',unsafe_allow_html=True)
    m=st.columns(7); values=[("Damage AP",f"{model.metrics['average_precision']:.3f}"),("Extreme AP",f"{model.metrics['extreme_average_precision']:.3f}"),("ROC-AUC",f"{model.metrics['roc_auc']:.3f}"),("Expected-loss MAE",f"${model.metrics['damage_mae']:,.0f}"),("Baseline MAE",f"${model.metrics['baseline_damage_mae']:,.0f}"),("Damage capture",f"{model.metrics['damage_capture_at_capacity']:.1%}"),("Baseline capture",f"{model.metrics['baseline_damage_capture']:.1%}")]
    for col,(label,value) in zip(m,values): col.metric(label,value)
    a,b=st.columns(2)
    with a:
        true,pred=calibration_curve(model.evaluation.has_damage,model.evaluation.damage_probability,n_bins=8,strategy="quantile"); fig=go.Figure(); fig.add_scatter(x=pred,y=true,mode="lines+markers",name="Hurdle probability"); fig.add_scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash"),name="Perfect"); fig.update_layout(title="Probability reliability · untouched test months",xaxis_title="Predicted",yaxis_title="Observed"); st.plotly_chart(_style(fig),width="stretch")
    with b:
        ranked=model.evaluation.sort_values("extreme_probability",ascending=False).reset_index(drop=True); ranked["review_share"]=(ranked.index+1)/len(ranked); ranked["captured_damage"]=ranked.total_damage_usd.cumsum()/max(ranked.total_damage_usd.sum(),1); fig=px.line(ranked,x="review_share",y="captured_damage",title="Damage captured by extreme-impact review score"); fig.add_vline(x=.1,line_dash="dash",line_color=AMBER); fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dot",color="#789")); st.plotly_chart(_style(fig),width="stretch")
    a,b=st.columns(2)
    with a:
        fig=px.bar(model.importance.sort_values("importance"),x="importance",y="feature",orientation="h",error_x="std",title="Permutation importance · extreme-impact AUCPR loss"); st.plotly_chart(_style(fig),width="stretch")
    with b:
        fig=px.bar(model.drift.sort_values("psi"),x="psi",y="feature",orientation="h",color="status",color_discrete_map={"stable":CYAN,"watch":AMBER,"high":RED},title="Train-to-test population stability · PSI"); fig.add_vline(x=.1,line_dash="dot"); fig.add_vline(x=.25,line_dash="dash"); st.plotly_chart(_style(fig),width="stretch")
    st.caption("The hurdle estimates any-damage probability and conditional amount. A separately calibrated tail classifier identifies reports at or above $250,000 for review ranking. This tail threshold and 10% capacity are fixed before the untouched test period. Event-type historical rate × median positive loss is the explicit baseline.")

    st.markdown('<section class="section-intro"><div class="section-kicker">Impact scoring workbench</div><h2>Explore one reported event.<br>Keep uncertainty explicit.</h2></section>',unsafe_allow_html=True)
    states=sorted(data.gold.state.unique()); types=sorted(data.gold.event_type.unique()); c1,c2,c3=st.columns(3); state=c1.selectbox("State",states,index=states.index("TEXAS") if "TEXAS" in states else 0); event=c1.selectbox("Event type",types,index=types.index("Tornado") if "Tornado" in types else 0); month=c2.slider("Month",1,12,6); hour=c2.slider("Begin hour",0,23,16); magnitude=c3.number_input("Reported magnitude · event-specific",value=1.0,step=.25); cz=c3.selectbox("Geography type",sorted(data.gold.cz_type.unique())); lat=st.slider("Begin latitude",float(data.gold.begin_lat.min()),float(data.gold.begin_lat.max()),float(data.gold.begin_lat.median())); lon=st.slider("Begin longitude",float(data.gold.begin_lon.min()),float(data.gold.begin_lon.max()),float(data.gold.begin_lon.median())); result=score_case(model,{"state":state,"event_type":event,"cz_type":cz,"month":month,"begin_hour":hour,"magnitude":magnitude,"begin_lat":lat,"begin_lon":lon})
    x,y,z,w=st.columns(4); x.metric("Damage probability",f"{result['damage_probability']:.1%}"); y.metric("Extreme-impact probability",f"{result['extreme_probability']:.1%}"); z.metric("Conditional amount",f"${result['conditional_damage_usd']:,.0f}"); w.metric("Expected impact",f"${result['expected_damage_usd']:,.0f}")
    st.info("This is an analytical what-if score for a completed-report feature profile. Magnitude units vary by event type, so cross-type scenarios require meteorological interpretation.")
    d=st.columns(3); d[0].download_button("Export Gold contract",data.gold.to_csv(index=False).encode(),f"storm_gold_{data.metadata['run_id']}.csv","text/csv",width="stretch"); d[1].download_button("Export test ranking",model.evaluation.to_csv(index=False).encode(),f"storm_evaluation_{data.metadata['run_id']}.csv","text/csv",width="stretch"); d[2].download_button("Export run manifest",_manifest(data,model),f"storm_manifest_{data.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Source, method, rights and limitations"):
        st.markdown(f"""**Source.** [NOAA/NCEI Storm Events Database]({DATABASE_URL}), bulk details revision `{data.metadata['filename']}`. The [official bulk-format specification]({FORMAT_URL}) defines identifiers, event fields and K/M/B damage notation. The database is updated as NWS submissions arrive, commonly around 75–90 days after a data month; historical annual files can be revised.

**Labels.** Property and crop damage must both be present. Blank values remain unknown and are quarantined—not silently converted to zero. Monetary estimates are nominal and not inflation-adjusted.

**Leakage controls.** Narratives, injuries, deaths, final damage values, event duration, and post-assessment tornado scale never enter model features. The app uses state, event type, geography class, month/hour, magnitude and starting coordinates.

**Rights.** U.S. federal-government works are generally public domain in the United States; NOAA attribution is retained. NCEI also notes archived holdings can contain externally submitted material with separate rights. This app uses structured database fields, no external media, seals or logos.

**Limits.** NOAA warns that some reports originate outside NWS and may be unverified. Damage is a broad best estimate. Selection, delayed reporting, missing labels, changing collection practice, inflation and extreme outliers limit model interpretation. This independent portfolio project is not endorsed by NOAA.""")
