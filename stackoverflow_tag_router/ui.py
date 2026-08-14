"""Streamlit control plane for the Stack Overflow tag routing pipeline."""

from __future__ import annotations
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from stackoverflow_tag_router.src.data import DOCS_URL,LICENSE_URL
from stackoverflow_tag_router.src.model import suggest_tags,train_and_evaluate
from stackoverflow_tag_router.src.pipeline import run_pipeline

ORANGE="#ff8a4c"; BLUE="#64b5ff"; MINT="#5ee6b8"; RED="#ff657a"


def _style(fig,height=410):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=58,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color="#eef3f8",family="Inter, sans-serif"),title_font=dict(size=18),legend=dict(orientation="h",y=1.12),colorway=[ORANGE,BLUE,MINT,RED]); fig.update_xaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); fig.update_yaxes(gridcolor="rgba(255,255,255,.08)",zeroline=False); return fig


@st.cache_resource(ttl=21_600,show_spinner=False)
def _load():
    data=run_pipeline(); return data,train_and_evaluate(data.gold)


def _manifest(data,model): return json.dumps({"run":data.metadata,"model":model.metadata,"metrics":model.metrics,"quality":data.quality.to_dict("records")},indent=2,default=str).encode()


def render_dashboard():
    st.markdown("""<style>.so-hero{padding:3.4rem 3rem;border-radius:26px;border:1px solid rgba(255,138,76,.28);background:radial-gradient(circle at 88% 18%,rgba(255,138,76,.2),transparent 31%),radial-gradient(circle at 98% 95%,rgba(100,181,255,.18),transparent 30%),linear-gradient(135deg,#161411,#08090b);margin-bottom:1.4rem}.so-kicker{color:#ff8a4c;font-size:.75rem;font-weight:750;letter-spacing:.16em;text-transform:uppercase}.so-hero h1{font:800 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#f8fbff;letter-spacing:-.06em;max-width:950px;margin:.8rem 0}.so-hero p{max-width:760px;color:#b9c0c9;font-size:1.05rem}.so-stage{min-height:170px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #ff8a4c;border-radius:0 0 14px 14px}.so-stage b{font-size:.77rem;color:#ff8a4c;letter-spacing:.08em}.so-stage p{font-size:.9rem;color:#b9c0c9}.so-boundary{border-left:3px solid #64b5ff;background:rgba(100,181,255,.09);padding:1rem 1.2rem;border-radius:0 10px 10px 0;color:#d5dae1}.section-intro{padding-top:2.8rem}.section-kicker{color:#ff8a4c;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase}.section-intro h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}</style>""",unsafe_allow_html=True)
    st.markdown("""<section class="so-hero"><div class="so-kicker">Multi-label NLP / Data + AI Engineering</div><h1>Route technical questions to the right experts.</h1><p>A replay-safe pipeline converts current Stack Overflow questions into a governed relational data product, then evaluates time-separated tag recommendations against a popularity baseline—with abstention, drift and evidence visible.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Ingesting questions, validating tag edges and evaluating the newest holdout …"): data,model=_load()
    except Exception as exc: st.error("No routing release was published because a data or model gate failed."); st.exception(exc); return
    if data.metadata["mode"]=="demo": st.warning(f"Deterministic demo questions are active because the Stack Exchange API could not be loaded: {data.metadata['fallback_reason']}")
    else: st.success("Live Stack Exchange API snapshot verified · temporal holdout · all release gates passed")
    c=st.columns(6); values=[("Questions",f"{len(data.gold):,}"),("Tag edges",f"{len(data.tag_bridge):,}"),("Replay suppressed",f"{data.metadata['replayed_deliveries']:,}"),("Precision@3",f"{model.metrics['precision_at_3']:.1%}"),("Baseline",f"{model.metrics['baseline_precision_at_3']:.1%}"),("Review route",f"{model.metrics['review_rate']:.1%}")]
    for col,(label,value) in zip(c,values): col.metric(label,value)
    st.caption(f"Run {data.metadata['run_id']} · {model.metadata['train_rows']:,}/{model.metadata['calibration_rows']:,}/{model.metadata['test_rows']:,} chronological train/policy/test rows · seed 42")
    st.markdown('<div class="so-boundary"><b>Decision boundary:</b> Suggested tags assist routing; they do not judge question quality, correctness or admissibility. Low-confidence questions are deferred. Stack Overflow remains the source of truth and every exported row retains its source link.</div>',unsafe_allow_html=True)

    st.markdown('<section class="section-intro"><div class="section-kicker">Data engineering control plane</div><h2>Respect the API.<br>Reconcile every relationship.</h2></section>',unsafe_allow_html=True)
    stages=st.columns(4); descriptions=[("01 · INGEST","Bounded pages, timeout, retry, API backoff, quota metadata and a six-hour application cache."),("02 · REPLAY","Deterministic 200-row micro-batches, event hashes and intentionally repeated deliveries."),("03 · CONTRACT","UTC timestamps, sanitized HTML, one-to-five tag rules, stable event IDs and reason-coded quarantine."),("04 · PUBLISH","Privacy-tokenized model text plus a reconciled question↔tag bridge and content-addressed lineage.")]
    for col,(title,body) in zip(stages,descriptions): col.markdown(f'<div class="so-stage"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
    a,b=st.columns([1.05,.95])
    with a:
        fig=px.bar(data.stages,x="stage",y="output_rows",color="stage",text="output_rows",hover_data=["input_rows","rejected_rows","duration_ms","content_hash"],title="Layer volume and content lineage"); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with b: st.markdown("#### Stage ledger"); st.dataframe(data.stages,hide_index=True,width="stretch")
    a,b=st.columns(2)
    with a: st.markdown("#### Contract and reconciliation gates"); st.dataframe(data.quality.assign(result=data.quality.passed.map({True:"Passed",False:"Failed"}))[["check","result","detail"]],hide_index=True,width="stretch")
    with b: st.markdown("#### Replay observability"); st.dataframe(data.batches,hide_index=True,width="stretch")
    view=st.radio("Inspect governed output",["Gold questions","Question-tag bridge","Quarantine"],horizontal=True); frame=data.gold.head(18) if view=="Gold questions" else data.tag_bridge.head(24) if view=="Question-tag bridge" else data.quarantine.head(18); st.dataframe(frame if len(frame) else pd.DataFrame({"state":["No quarantined records"]}),hide_index=True,width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">AI engineering evaluation</div><h2>Test on the future.<br>Defer weak recommendations.</h2></section>',unsafe_allow_html=True)
    m=st.columns(7); metrics=[("Precision@3",f"{model.metrics['precision_at_3']:.1%}"),("Popularity",f"{model.metrics['baseline_precision_at_3']:.1%}"),("Recall@3",f"{model.metrics['recall_at_3']:.1%}"),("Micro F1",f"{model.metrics['micro_f1']:.3f}"),("Macro F1",f"{model.metrics['macro_f1']:.3f}"),("Brier",f"{model.metrics['brier']:.3f}"),("Auto coverage",f"{model.metrics['auto_coverage']:.1%}")]
    for col,(label,value) in zip(m,metrics): col.metric(label,value)
    a,b=st.columns(2)
    with a:
        ranked=model.tag_metrics.sort_values("f1"); fig=px.bar(ranked,x="f1",y="tag",orientation="h",color="support",title="Newest-holdout F1 by learned tag",color_continuous_scale=["#23364c",BLUE]); st.plotly_chart(_style(fig,520),width="stretch")
    with b:
        fig=go.Figure(); fig.add_bar(name="Train",x=model.tag_metrics.tag,y=model.tag_metrics.train_share); fig.add_bar(name="Newest test",x=model.tag_metrics.tag,y=model.tag_metrics.test_share); fig.update_layout(barmode="group",title="Tag prevalence drift across time",yaxis_tickformat=".0%"); st.plotly_chart(_style(fig,520),width="stretch")
    a,b=st.columns(2)
    with a:
        fig=px.histogram(model.evaluation,x="max_confidence",color="route",nbins=24,title="Confidence and abstention policy",color_discrete_map={"auto-suggest":MINT,"review":ORANGE}); fig.add_vline(x=model.metrics["abstain_threshold"],line_dash="dash"); st.plotly_chart(_style(fig),width="stretch")
    with b:
        fig=px.bar(model.drift.sort_values("psi"),x="psi",y="feature",orientation="h",color="status",title="Input drift · PSI",color_discrete_map={"stable":BLUE,"watch":ORANGE,"high":RED}); fig.add_vline(x=.1,line_dash="dot"); fig.add_vline(x=.25,line_dash="dash"); st.plotly_chart(_style(fig),width="stretch")
    st.caption("The oldest 70% train the model, the next 15% choose the abstention threshold, and only the newest 15% report final metrics. Precision@3 counts correct suggestions among three slots; popularity always proposes the three most frequent training tags.")

    st.markdown('<section class="section-intro"><div class="section-kicker">Routing workbench</div><h2>Describe a problem.<br>Inspect tags and evidence.</h2></section>',unsafe_allow_html=True)
    title=st.text_input("Question title",value="Pandas merge creates duplicate rows after joining by customer id"); body=st.text_area("Question body",value="I am merging two Python dataframes and need only the newest record for each customer. My current df.merge call returns multiple rows.",height=130,max_chars=8_000); result=suggest_tags(model,title,body); a,b,c=st.columns(3); a.metric("Route",result["route"].replace("-"," ").title()); b.metric("Top suggestion",result["suggestions"].iloc[0].tag); c.metric("Top confidence",f"{result['suggestions'].iloc[0].confidence:.1%}")
    if result["route"]=="review": st.warning("Confidence is below the policy threshold; preserve human routing rather than forcing a tag.")
    else: st.info("Automatic suggestion candidate. The user or reviewer should still confirm the tags.")
    a,b=st.columns(2); a.dataframe(result["suggestions"],hide_index=True,width="stretch"); b.dataframe(result["evidence"],hide_index=True,width="stretch")
    d=st.columns(3); d[0].download_button("Export Gold questions",data.gold.to_csv(index=False).encode(),f"so_gold_{data.metadata['run_id']}.csv","text/csv",width="stretch"); d[1].download_button("Export evaluation",model.evaluation.to_csv(index=False).encode(),f"so_eval_{data.metadata['run_id']}.csv","text/csv",width="stretch"); d[2].download_button("Export manifest",_manifest(data,model),f"so_manifest_{data.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Source, license, lifecycle and limits"):
        st.markdown(f"""**Source.** [Stack Exchange API `/questions`]({DOCS_URL}), API v2.3, `site=stackoverflow`, creation order, maximum 100 rows per page, and the built-in `withbody` filter. The source can change continuously; this app caches a bounded snapshot for six hours and honors response `backoff`.

**License.** Public user contributions created after 2 May 2018 are [CC BY-SA 4.0]({LICENSE_URL}). Every record retains the original Stack Overflow question link. Exports are therefore not relicensed as proprietary data and must preserve attribution/share-alike obligations.

**Model lifecycle.** The candidate is one-vs-rest logistic regression over word 1–2 grams and character 3–5 grams. Labels are the twelve most frequent tags in training only. A chronological split prevents newer questions entering training. The popularity baseline, per-tag F1, multilabel Brier score, abstention policy and drift are all visible.

**Limits.** A recent bounded snapshot does not represent all Stack Overflow history. Rare tags outside the learned vocabulary cannot be suggested. Tags may be edited after collection, HTML cleaning loses structure, probabilities are not causal, and current terminology can drift. This tool does not post, edit, moderate or answer questions.""")
