"""Streamlit control plane for the maintenance data and AI lifecycle."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve

from predictive_maintenance_pipeline.src.data import DATASET_PAGE
from predictive_maintenance_pipeline.src.model import decision_table, score_case, train_and_evaluate
from predictive_maintenance_pipeline.src.pipeline import FEATURES, run_pipeline

ACCENT="#ffb000"; RED="#ff5c57"; BLUE="#4ecdc4"; BG="#0a0d14"


def _style(fig, height=410):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=58,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color="#e9eef8",family="Inter, sans-serif"),title_font=dict(size=18),legend=dict(orientation="h",y=1.12),colorway=[ACCENT,BLUE,RED,"#8b7cff"])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); fig.update_yaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False)
    return fig


@st.cache_resource(ttl=43_200,show_spinner=False)
def _load():
    data=run_pipeline(); model=train_and_evaluate(data.gold); return data,model


def _manifest(data,model):
    return json.dumps({"run":data.metadata,"model":model.metadata,"metrics":model.metrics,"quality":data.quality.to_dict("records")},indent=2,default=str).encode()


def render_dashboard():
    st.markdown("""<style>
    .maint-hero{padding:3.4rem 3rem;border:1px solid rgba(255,176,0,.28);border-radius:24px;background:radial-gradient(circle at 90% 0%,rgba(255,176,0,.2),transparent 35%),linear-gradient(135deg,#131923,#090c12);margin-bottom:1.4rem}.maint-kicker{color:#ffb000;font:700 .76rem Inter;letter-spacing:.16em;text-transform:uppercase}.maint-hero h1{font:800 clamp(2.7rem,6vw,5.8rem)/.9 Inter;margin:.8rem 0;color:#f7f9fc;letter-spacing:-.07em;max-width:850px}.maint-hero p{font-size:1.05rem;max-width:700px;color:#aeb8c8}.maint-stage{min-height:160px;padding:1.2rem;border-top:2px solid #ffb000;background:rgba(255,255,255,.035);border-radius:0 0 14px 14px}.maint-stage b{font-size:.78rem;color:#ffb000;letter-spacing:.08em}.maint-stage p{color:#aeb8c8;font-size:.9rem}.maint-boundary{border-left:3px solid #ff5c57;background:rgba(255,92,87,.08);padding:1rem 1.2rem;border-radius:0 10px 10px 0;color:#c9d2df}.section-intro{padding-top:2.8rem}.section-kicker{color:#ffb000;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase}.section-intro h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}
    </style>""",unsafe_allow_html=True)
    st.markdown("""<section class="maint-hero"><div class="maint-kicker">Industrial Data + AI Engineering / Decision system</div><h1>Know when to intervene.</h1><p>A content-addressed machine-cycle pipeline feeds a calibrated failure classifier. Every alert exposes data quality, leakage controls, probability quality and the operating cost behind its threshold.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Validating source, rebuilding data layers and evaluating the model …"): data,model=_load()
    except Exception as exc:
        st.error("No decision product was published because a data or model gate failed."); st.exception(exc); return
    if data.metadata["mode"]=="demo": st.warning(f"Reproducible demo data is active because UCI could not be reached: {data.metadata['fallback_reason']}")
    else: st.success("Live UCI archive verified · all data contracts passed · model evaluated on an untouched ordered test block")
    cards=st.columns(6); values=[("Cycles",f"{len(data.gold):,}"),("Failures",f"{data.gold.machine_failure.sum():,}"),("Prevalence",f"{model.metrics['prevalence']:.2%}"),("AUCPR",f"{model.metrics['average_precision']:.3f}"),("Recall",f"{model.metrics['recall']:.1%}"),("Brier",f"{model.metrics['brier']:.4f}")]
    for col,(label,value) in zip(cards,values): col.metric(label,value)
    st.caption(f"Run {data.metadata['run_id']} · source mode {data.metadata['mode']} · seed 42 · ordered 60/20/20 train/calibration/test split")
    st.markdown('<div class="maint-boundary"><b>Decision boundary:</b> This system estimates failure risk in a synthetic benchmark. It is not a safety controller, remaining-useful-life model or substitute for sensor history, maintenance records and a plant-specific validation.</div>',unsafe_allow_html=True)

    st.markdown('<section class="section-intro"><div class="section-kicker">Data engineering control plane</div><h2>Reject bad cycles.<br>Publish reproducible features.</h2></section>',unsafe_allow_html=True)
    stages=st.columns(4); copy=[("01 · EXTRACT","Download one bounded ZIP with timeout, retry, exponential backoff, member allowlist and size guard."),("02 · BRONZE","Preserve the source-shaped table and fingerprint both archive and normalized payload."),("03 · SILVER","Cast the contract, enforce domains and ranges, deduplicate UDI and quarantine failures."),("04 · GOLD","Derive temperature gap and power proxy; remove all five target-derived failure modes before training.")]
    for col,(title,body) in zip(stages,copy): col.markdown(f'<div class="maint-stage"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
    left,right=st.columns([1.05,.95])
    with left:
        fig=px.bar(data.stages,x="stage",y="output_rows",color="stage",text="output_rows",hover_data=["input_rows","rejected_rows","duration_ms","content_hash"],title="Layer volume and content lineage"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with right:
        st.markdown("#### Immutable stage ledger"); st.dataframe(data.stages,hide_index=True,width="stretch")
    q1,q2=st.columns([1,1])
    with q1:
        qa=data.quality.assign(result=data.quality.passed.map({True:"Passed",False:"Failed"})); st.markdown("#### Contract gates"); st.dataframe(qa[["check","result","detail"]],hide_index=True,width="stretch")
    with q2:
        view=st.radio("Inspect layer",["Gold serving contract","Quarantine","Silver audit"],horizontal=True)
        frame=data.gold.head(20) if view.startswith("Gold") else data.quarantine if view=="Quarantine" else data.silver.head(20)
        if frame.empty: frame=pd.DataFrame({"state":["No records quarantined"]})
        st.dataframe(frame,hide_index=True,width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">AI engineering evaluation</div><h2>Calibrate probability.<br>Price every miss.</h2></section>',unsafe_allow_html=True)
    fn_cost=st.slider("Relative cost of one missed failure",1,100,25); fp_cost=st.slider("Relative cost of one unnecessary inspection",1,20,1)
    choices=decision_table(model.calibration.label.to_numpy(),model.calibration.probability.to_numpy(),fn_cost,fp_cost); threshold=float(choices.loc[choices.cost.idxmin(),"threshold"])
    eval_=model.evaluation; pred=eval_.failure_probability>=threshold; y=eval_.machine_failure; tn=((~pred)&(y==0)).sum(); fp=(pred&(y==0)).sum(); fn=((~pred)&(y==1)).sum(); tp=(pred&(y==1)).sum()
    cols=st.columns(6); items=[("Operating threshold",f"{threshold:.2f}"),("True positives",f"{tp:,}"),("False negatives",f"{fn:,}"),("False positives",f"{fp:,}"),("Test cost",f"{fn*fn_cost+fp*fp_cost:,.0f}"),("AP lift",f"{model.metrics['average_precision']/model.metrics['baseline_ap']:.1f}×")]
    for col,(label,value) in zip(cols,items): col.metric(label,value)
    a,b=st.columns(2)
    with a:
        fig=px.line(choices,x="threshold",y="cost",title="Calibration-set operating cost by threshold"); fig.add_vline(x=threshold,line_dash="dash",line_color=ACCENT); st.plotly_chart(_style(fig),width="stretch")
    with b:
        true,predicted=calibration_curve(model.evaluation.machine_failure,model.evaluation.failure_probability,n_bins=8,strategy="quantile")
        fig=go.Figure(); fig.add_scatter(x=predicted,y=true,mode="lines+markers",name="Calibrated model"); fig.add_scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect",line=dict(dash="dash")); fig.update_layout(title="Reliability on untouched test block",xaxis_title="Mean predicted probability",yaxis_title="Observed failure rate"); st.plotly_chart(_style(fig),width="stretch")
    a,b=st.columns(2)
    with a:
        fig=px.bar(model.importance.sort_values("importance"),x="importance",y="feature",orientation="h",error_x="std",title="Permutation importance · AUCPR loss"); st.plotly_chart(_style(fig),width="stretch")
    with b:
        fig=px.bar(model.drift.sort_values("psi"),x="psi",y="feature",orientation="h",color="status",title="Train-to-test feature drift · PSI",color_discrete_map={"stable":BLUE,"watch":ACCENT,"high":RED}); fig.add_vline(x=.1,line_dash="dot"); fig.add_vline(x=.25,line_dash="dash"); st.plotly_chart(_style(fig),width="stretch")
    st.caption("AUCPR is the primary ranking metric because failures are rare. Isotonic calibration is fitted only on the middle block. The alert threshold is also selected there; the final block remains untouched until evaluation.")

    st.markdown('<section class="section-intro"><div class="section-kicker">What-if serving workbench</div><h2>Inspect one machine cycle.<br>See the decision, not just a score.</h2></section>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    kind=c1.selectbox("Product quality type",["L","M","H"],index=1); air=c1.slider("Air temperature · K",295.0,305.0,300.0,.1); process=c2.slider("Process temperature · K",305.0,315.0,310.0,.1); rpm=c2.slider("Rotational speed · rpm",1100,2900,1500,10); torque=c3.slider("Torque · Nm",3.0,77.0,40.0,.5); wear=c3.slider("Tool wear · min",0,253,100)
    values={"type":kind,"air_temperature_k":air,"process_temperature_k":process,"rotational_speed_rpm":rpm,"torque_nm":torque,"tool_wear_min":wear,"temperature_gap_k":process-air,"power_proxy":rpm*torque}; probability=score_case(model,values); alert=probability>=threshold
    x,y,z=st.columns(3); x.metric("Calibrated failure probability",f"{probability:.1%}"); y.metric("Current threshold",f"{threshold:.0%}"); z.metric("Decision","Inspect" if alert else "Monitor")
    if alert: st.error("Inspection is triggered under the selected cost policy. Confirm with plant telemetry and engineering procedures.")
    else: st.info("No inspection alert at this threshold. Low modeled probability is not proof of safe operation.")
    downloads=st.columns(3); downloads[0].download_button("Export Gold contract",data.gold.to_csv(index=False).encode(),f"maintenance_gold_{data.metadata['run_id']}.csv","text/csv",width="stretch"); downloads[1].download_button("Export test decisions",model.evaluation.to_csv(index=False).encode(),f"maintenance_eval_{data.metadata['run_id']}.csv","text/csv",width="stretch"); downloads[2].download_button("Export run manifest",_manifest(data,model),f"maintenance_manifest_{data.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Method, source rights and limitations"):
        st.markdown(f"""**Source.** [UCI AI4I 2020 Predictive Maintenance Dataset]({DATASET_PAGE}), a static synthetic benchmark with 10,000 rows and a CC BY 4.0 license. Fields used are product type, air/process temperature, rotational speed, torque, tool wear and machine failure. UDI is retained only for lineage and ordered splitting.

**Leakage boundary.** TWF, HDF, PWF, OSF and RNF describe failure modes and directly determine the target. They may be audited in Silver but are contractually absent from Gold and model inputs.

**Model lifecycle.** An ordered 60/20/20 split approximates future deployment without claiming UDI is time. Class weights fit gradient boosting on the first block; isotonic calibration and cost thresholding use the second; AUCPR, ROC-AUC, Brier score, recall, precision, drift and permutation importance use the untouched third block.

**Limits.** The benchmark is synthetic and lacks timestamps, multiple observations per asset, maintenance actions and censoring. Correlation is not causation; probabilities require site-specific validation. Do not connect this demonstration to equipment or safety logic.""")
