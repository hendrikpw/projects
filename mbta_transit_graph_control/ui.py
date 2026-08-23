"""Modern Streamlit control plane for the MBTA transit graph product."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mbta_transit_graph_control.src.data import DOCS_URL, FEED_URL, LICENSE_URL
from mbta_transit_graph_control.src.model import score_node, train_and_evaluate
from mbta_transit_graph_control.src.pipeline import run_pipeline

NAVY="#061724"; CYAN="#32d5ff"; LIME="#b7f34a"; CORAL="#ff6b5e"; ICE="#edfaff"
TYPE_NAMES={0:"Light rail",1:"Subway",2:"Commuter rail",3:"Bus",4:"Ferry"}


def _style(fig, height: int = 420):
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=58,b=24),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.025)",font=dict(color=ICE,family="Inter, sans-serif"),legend=dict(orientation="h",y=1.14),colorway=[CYAN,LIME,CORAL,"#a98cff"])
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.07)"); return fig


@st.cache_resource(ttl=21600,show_spinner=False)
def _load():
    product=run_pipeline(); return product,train_and_evaluate(product)


def _manifest(product,model)->bytes:
    return json.dumps({"pipeline":product.metadata,"quality":product.quality.to_dict("records"),"model":model.metadata,"metrics":model.metrics},indent=2,default=str).encode()


def render_dashboard()->None:
    st.markdown("""<style>
    .gtfs-hero{padding:3.8rem 3rem;border-radius:28px;background:radial-gradient(circle at 88% 12%,rgba(50,213,255,.25),transparent 28%),radial-gradient(circle at 74% 94%,rgba(183,243,74,.17),transparent 34%),linear-gradient(145deg,#102f43,#061724);border:1px solid rgba(50,213,255,.28);margin-bottom:1.4rem}.gtfs-hero h1{font:850 clamp(2.8rem,6vw,5.7rem)/.9 Inter;color:#edfaff;letter-spacing:-.065em;max-width:1080px;margin:.8rem 0}.gtfs-hero p{max-width:920px;color:#b9d2df;font-size:1.05rem}.kicker{color:#b7f34a;font-size:.74rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.boundary{border-left:3px solid #ff6b5e;background:rgba(255,107,94,.08);padding:1rem 1.2rem;border-radius:0 12px 12px 0}.stage{min-height:174px;padding:1.2rem;background:rgba(255,255,255,.035);border-top:2px solid #32d5ff;border-radius:0 0 14px 14px}.stage b{color:#b7f34a;font-size:.76rem;letter-spacing:.08em}.stage p{color:#b9d2df;font-size:.9rem}.section{padding-top:2.8rem}.section small{color:#b7f34a;letter-spacing:.15em;text-transform:uppercase}.section h2{font-size:clamp(2rem,4vw,3.4rem);line-height:.98;letter-spacing:-.04em;margin:.5rem 0 1.2rem}.route-box{padding:1rem;border:1px solid rgba(50,213,255,.27);border-radius:14px;background:rgba(50,213,255,.06)}</style>
    <section class="gtfs-hero"><div class="kicker">MBTA GTFS / Graph Data + AI Engineering</div><h1>Validate every link.<br>Recover the missing one.</h1><p>A relational GTFS pipeline contracts stops, routes, representative patterns and stop sequences into a lineage-aware transit graph. A hybrid graph-embedding model then learns network structure and is tested by hiding real stop–route connections.</p></section>""",unsafe_allow_html=True)
    try:
        with st.spinner("Downloading the current MBTA schedule, reconciling relations and masking graph links …"): product,model=_load()
    except Exception as exc:
        st.error("No graph product was published because a source, relational-quality or model-promotion gate failed."); st.exception(exc); return
    if product.metadata["mode"]=="demo": st.warning(f"Deterministic demonstration network active: {product.metadata['fallback_reason']}")
    else: st.success("Current MBTA GTFS loaded · ten publication gates and graph-model promotion passed")
    values=[("Graph nodes",f"{product.metadata['nodes']:,}"),("Routes",f"{product.metadata['routes']:,}"),("Stop–route edges",f"{product.metadata['edges']:,}"),("Multi-route nodes",f"{product.metadata['multi_route_nodes']:,}"),("Hidden-edge Hit@5",f"{model.metrics['hit_at_5']:.1%}"),("Popularity baseline",f"{model.metrics['baseline_hit_at_5']:.1%}")]
    for column,(label,value) in zip(st.columns(6),values): column.metric(label,value)
    st.caption(f"Run {product.metadata['run_id']} · {product.metadata['feed_version']} · valid {product.metadata['feed_start_date']}–{product.metadata['feed_end_date']} · {model.metadata['factors']} latent factors · {model.metrics['fit_seconds']:.2f}s fit")
    st.markdown('<div class="boundary"><b>Operational boundary:</b> candidates are data-quality review signals inferred from scheduled network structure. They are not journey plans, live arrivals, accessibility guarantees or instructions to add service. Confirm every connection against the authoritative MBTA feed.</div>',unsafe_allow_html=True)

    pipeline_tab,model_tab,lab_tab=st.tabs(["◫ Relational pipeline","⌁ Graph evaluation","◎ Link recovery lab"])
    with pipeline_tab:
        st.markdown('<section class="section"><small>Data engineering</small><h2>Five tables in.<br>One governed graph out.</h2></section>',unsafe_allow_html=True)
        cards=[("01 · SAFE EXTRACT","Retry, timeout, byte bounds, ZIP signature, traversal protection and five required-member contracts."),("02 · BOUNDED PARSE","Only typical or canonical representative trips are retained; the 104 MB stop-time table is filtered in chunks."),("03 · RELATIONAL CONTRACT","Typed keys, regional coordinates, route types, foreign-key checks, quarantine and parent-station normalization."),("04 · GRAPH PUBLISH","Stable edge IDs, intentional replays, reconciliation, SHA-256 lineage and ten fail-closed gates.")]
        for column,(title,body) in zip(st.columns(4),cards): column.markdown(f'<div class="stage"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
        degrees=product.edges.groupby("node_id").size().rename("route_degree"); map_data=product.nodes.join(degrees,on="node_id").fillna({"route_degree":0}); map_data=map_data.sort_values("route_degree",ascending=False).head(3500)
        left,right=st.columns([1.25,.75])
        with left:
            fig=px.scatter_map(map_data,lat="latitude",lon="longitude",color="route_degree",size="route_degree",hover_name="stop_name",hover_data=["municipality","platforms"],zoom=8.2,map_style="carto-darkmatter",color_continuous_scale=[NAVY,CYAN,LIME],title="Contracted network nodes · size = route degree"); fig.update_layout(height=520,margin=dict(l=0,r=0,t=45,b=0)); st.plotly_chart(fig,width="stretch")
        with right: st.markdown("#### Stage ledger"); st.dataframe(product.stages,hide_index=True,width="stretch"); st.markdown("#### Publication gates"); st.dataframe(product.quality.assign(result=product.quality.passed.map({True:"Passed",False:"Failed"}))[["check","result","detail"]],hide_index=True,width="stretch")
        left,right=st.columns(2)
        with left:
            route_view=product.routes.assign(mode=product.routes.route_type.map(TYPE_NAMES).fillna("Other")).groupby("mode",as_index=False).size(); st.plotly_chart(_style(px.bar(route_view,x="mode",y="size",color="mode",title="Published routes by GTFS mode")),width="stretch")
        with right:
            degree_view=degrees.clip(upper=10).value_counts().sort_index().rename_axis("routes_per_node").reset_index(name="nodes"); st.plotly_chart(_style(px.bar(degree_view,x="routes_per_node",y="nodes",title="Graph-degree distribution · 10 = 10+")),width="stretch")
        if len(product.quarantine): st.markdown("#### Quarantine audit"); st.dataframe(product.quarantine.groupby(["table","reason"],as_index=False).size(),hide_index=True,width="stretch")

    with model_tab:
        st.markdown('<section class="section"><small>AI engineering</small><h2>Hide a true edge.<br>Rank it from structure.</h2></section>',unsafe_allow_html=True)
        st.markdown("One real route is deterministically removed from every multi-route node. Calibration and test nodes are separated by stable hashes. The model sees neither hidden edge while learning route co-occurrence, Truncated-SVD embeddings and a geographic prior; candidate weights and factor count are selected only on calibration edges.")
        metrics=st.columns(5); metrics[0].metric("Hit@1",f"{model.metrics['hit_at_1']:.1%}"); metrics[1].metric("Hit@5",f"{model.metrics['hit_at_5']:.1%}"); metrics[2].metric("MRR",f"{model.metrics['mrr']:.3f}"); metrics[3].metric("NDCG@5",f"{model.metrics['ndcg_at_5']:.3f}"); metrics[4].metric("Test edges",f"{model.metrics['test_edges']:,}")
        left,right=st.columns([1.05,.95])
        with left:
            embed=model.embeddings.copy(); embed["mode"]=embed.route_type.map(TYPE_NAMES).fillna("Other"); fig=px.scatter(embed,x="x",y="y",color="mode",size="graph_degree",hover_name="route_label",hover_data=["route_id"],title="Learned route embedding space"); st.plotly_chart(_style(fig,500),width="stretch")
        with right:
            fig=px.line(model.robustness,x="edge_dropout",y=["hit_at_1","hit_at_5","mrr"],markers=True,title="Failure curve under additional visible-edge loss"); st.plotly_chart(_style(fig,500),width="stretch")
        route_eval=model.route_metrics.dropna(subset=["hit_at_5"]).sort_values(["held_edges","hit_at_5"],ascending=False).head(30)
        left,right=st.columns(2)
        with left: st.plotly_chart(_style(px.bar(route_eval.sort_values("hit_at_5"),x="hit_at_5",y="route_label",orientation="h",color="held_edges",title="Held-edge recovery by route"),520),width="stretch")
        with right:
            comparison=pd.DataFrame({"method":["Hybrid graph model","Route-popularity baseline"],"Hit@5":[model.metrics["hit_at_5"],model.metrics["baseline_hit_at_5"]],"MRR":[model.metrics["mrr"],model.metrics["baseline_mrr"]]}); st.plotly_chart(_style(px.bar(comparison,x="method",y=["Hit@5","MRR"],barmode="group",title="Promotion against a transparent baseline"),520),width="stretch")
        st.markdown("#### Masked-edge audit"); st.dataframe(model.evaluation[model.evaluation.split=="test"][["node_id","actual_route","predicted_route","rank","hit_at_5","margin","route"]].head(300),hide_index=True,width="stretch")

    with lab_tab:
        st.markdown('<section class="section"><small>Serving and failure behavior</small><h2>Remove a connection.<br>Ask the graph to recover it.</h2></section>',unsafe_allow_html=True)
        degrees=product.edges.groupby("node_id").size(); eligible=product.nodes[product.nodes.node_id.isin(degrees[degrees>=2].index)].copy(); eligible["display"]=eligible.stop_name.fillna(eligible.node_id)+" · "+eligible.node_id
        selected=st.selectbox("Multi-route station or stop",eligible.node_id,format_func=dict(zip(eligible.node_id,eligible.display)).get); attached=product.edges[product.edges.node_id==selected].route_id.tolist(); route_names=product.routes.set_index("route_id").route_label.to_dict(); hidden=st.selectbox("Simulate a missing authoritative link",attached,format_func=lambda value:f"{route_names.get(value,value)} · {value}")
        result=score_node(model,selected,hide_route=hidden); columns=st.columns(4); columns[0].metric("Hidden route",route_names.get(hidden,hidden)); columns[1].metric("Recovered rank",result["recovered_rank"]); columns[2].metric("Hit@5","Yes" if result["hit_at_5"] else "No"); columns[3].metric("Serving route",result["route"])
        ranking=result["ranking"].merge(product.routes[["route_id","route_label","route_type"]],on="route_id",how="left"); ranking["candidate"]=ranking.route_label.fillna(ranking.route_id)+" · "+ranking.route_id; ranking["is_hidden_truth"]=ranking.route_id.eq(hidden)
        st.plotly_chart(_style(px.bar(ranking.sort_values("score"),x="score",y="candidate",orientation="h",color="is_hidden_truth",title="Ranked recovery candidates",color_discrete_map={True:LIME,False:CYAN}),470),width="stretch")
        st.markdown(f'<div class="route-box"><b>{result["route"].upper()}</b><br>The margin between the first two candidates is {result["margin"]:.3f}. Even a high-ranked recovery remains a review suggestion; the hidden relationship is shown only to make evaluation observable.</div>',unsafe_allow_html=True)

    st.markdown('<section class="section"><small>Audit and provenance</small><h2>Export the evidence.</h2></section>',unsafe_allow_html=True)
    left,middle,right=st.columns(3); left.download_button("Download run manifest",_manifest(product,model),"mbta_graph_manifest.json","application/json",width="stretch"); middle.download_button("Download masked-edge audit",model.evaluation.to_csv(index=False).encode(),"mbta_graph_evaluation.csv","text/csv",width="stretch"); right.download_button("Download quality ledger",product.quality.to_csv(index=False).encode(),"mbta_graph_quality.csv","text/csv",width="stretch")
    st.caption(f"Source: MBTA GTFS schedule feed · [current ZIP]({FEED_URL}) · [MBTA field documentation]({DOCS_URL}) · [developer license agreement]({LICENSE_URL})")
