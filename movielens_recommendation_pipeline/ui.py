"""Modern Streamlit control plane for recommendation data and model operations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from movielens_recommendation_pipeline.src.data import CITATION_URL, DATASETS_URL, README_URL
from movielens_recommendation_pipeline.src.model import RecommenderBundle, cold_start_recommendations, recommend_user, train_and_evaluate
from movielens_recommendation_pipeline.src.pipeline import PipelineBundle, run_pipeline


PINK, CYAN = "#f472b6", "#22d3ee"
COLORS=[PINK,CYAN,"#a78bfa","#fbbf24","#34d399","#fb7185"]


def _style(fig:go.Figure,height:int=420)->go.Figure:
    fig.update_layout(height=height,margin=dict(l=18,r=18,t=62,b=24),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Inter, Arial",color="rgba(252,252,253,.72)"),title_font=dict(size=17,color="#fcfcfd"),colorway=COLORS,legend=dict(orientation="h",y=1.04,title=None),hoverlabel=dict(bgcolor="#161524",font_color="#fcfcfd",bordercolor="#58536b"))
    fig.update_xaxes(gridcolor="rgba(252,252,253,.08)",zeroline=False); fig.update_yaxes(gridcolor="rgba(252,252,253,.08)",zeroline=False)
    return fig


@st.cache_data(ttl=86_400,show_spinner=False)
def _pipeline()->PipelineBundle: return run_pipeline()


@st.cache_resource(show_spinner=False)
def _model(gold_hash:str,_ratings:pd.DataFrame,_movies:pd.DataFrame,factors:int)->RecommenderBundle:
    return train_and_evaluate(_ratings,_movies,factors,20)


def _manifest(data:PipelineBundle,model:RecommenderBundle)->bytes:
    return json.dumps({"pipeline":data.metadata,"stages":data.stages.to_dict("records"),"quality":data.quality.to_dict("records"),"model":{**model.metadata,**model.metrics}},indent=2,default=str).encode()


def render_dashboard()->None:
    st.markdown("""
    <style>
    .rec-stage{padding:1rem;border-left:3px solid #f472b6;background:rgba(255,255,255,.035);min-height:132px}.rec-stage b{font-size:.82rem;color:#fcfcfd}.rec-stage p{font-size:.78rem;margin:.55rem 0 0}.rec-boundary{padding:1rem 1.15rem;border:1px solid rgba(244,114,182,.38);background:rgba(244,114,182,.08);border-radius:4px}
    </style>
    <section class="page-hero"><div class="eyebrow">19 / Data + AI engineering</div><h1>MovieLens Recommendation<br>Serving Pipeline.</h1><p>Contract interaction data, hide future preference signals and serve evaluated, diversity-aware recommendations.</p><div class="source-line">GroupLens · Temporal holdout · Latent factors + cold start</div></section>
    """,unsafe_allow_html=True)
    factors=st.select_slider("Latent model capacity",[24,32,48,64],value=48,format_func=lambda value:f"{value} factors")
    try:
        with st.spinner("Downloading the contracted archive, validating interactions and evaluating the recommender…"):
            data=_pipeline(); model=_model(data.metadata["gold_hash"],data.ratings,data.movies,factors)
    except (ValueError,KeyError,TypeError,RuntimeError) as exc:
        st.error("The run stopped before recommendations were published because a data or model contract failed.")
        st.caption(f"Failure state · {type(exc).__name__}: {exc}")
        st.info("Use the default model capacity. Failed runs never expose stale recommendations as current output.")
        return
    if data.metadata["mode"]=="demo":
        st.warning("The GroupLens archive was unavailable or invalid. Every component is running on a deterministic source-shaped interaction dataset.")
        st.caption("Fallback reason: "+data.metadata["fallback_reason"])
    else: st.success(f"Verified GroupLens archive · {data.metadata['ratings']:,} ratings · {data.metadata['movies']:,} movies · static dataset generated 26 Sep 2018")
    cards=st.columns(7); values=[("Users",f"{data.metadata['users']:,}"),("Movies",f"{data.metadata['movies']:,}"),("Ratings",f"{data.metadata['ratings']:,}"),("DQ pass",f"{data.metadata['quality_pass_rate']:.0%}"),("HitRate@20",f"{model.metrics['hit_rate_at_k']:.1%}"),("MRR@20",f"{model.metrics['mrr_at_k']:.3f}"),("Catalog coverage",f"{model.metrics['catalog_coverage']:.1%}")]
    for col,(label,value) in zip(cards,values): col.metric(label,value)
    st.caption(f"Run {data.metadata['run_id']} · model seed 42 · {model.metadata['training_ratings']:,} training interactions / {model.metadata['holdout_users']:,} chronological holdouts")
    st.markdown('<div class="rec-boundary"><b>Interpretation boundary:</b> Recommendations estimate preference from historical ratings. They do not measure artistic quality, guarantee enjoyment or infer demographics. The small development dataset is not a production catalog.</div>',unsafe_allow_html=True)

    st.markdown('<section class="section-intro"><div class="section-kicker">Data engineering control plane</div><h2>Verify the archive.<br>Contract every interaction.</h2></section>',unsafe_allow_html=True)
    stage_cols=st.columns(4); stage_copy=[("01 · EXTRACT","Download one versioned ZIP with timeout, retry, size guard and strict member allowlist."),("02 · BRONZE","Preserve four source tables and compute deterministic file, archive and layer hashes."),("03 · SILVER","Parse UTC events, enforce half-star ratings, deduplicate and quarantine broken references."),("04 · GOLD","Publish movie popularity, activity, tag, genre, year and novelty features for serving and audit.")]
    for col,(title,body) in zip(stage_cols,stage_copy): col.markdown(f'<div class="rec-stage"><b>{title}</b><p>{body}</p></div>',unsafe_allow_html=True)
    left,right=st.columns([1.1,.9])
    with left:
        fig=px.bar(data.stages,x="stage",y="output_rows",text="output_rows",color="stage",title="Layer volume, rejection and content lineage",hover_data=["input_rows","rejected_rows","duration_ms","content_hash"]); fig.update_traces(textposition="outside"); st.plotly_chart(_style(fig),width="stretch")
    with right:
        st.markdown("#### Immutable stage ledger"); st.dataframe(data.stages,hide_index=True,width="stretch")
    ql,qr=st.columns([.9,1.1])
    with ql:
        qa=data.quality.assign(result=data.quality["passed"].map({True:"Passed",False:"Failed"})); st.markdown("#### Contract and quality gates"); st.dataframe(qa[["check","result","detail"]],hide_index=True,width="stretch")
    with qr:
        dist=data.ratings.groupby("rating",as_index=False).size(); fig=px.bar(dist,x="rating",y="size",title="Validated explicit-feedback distribution",labels={"rating":"Rating","size":"Interactions"}); st.plotly_chart(_style(fig),width="stretch")
    layer=st.radio("Inspect data product",["Ratings contract","Movie feature view","Quarantine"],horizontal=True)
    if layer=="Ratings contract": preview=data.ratings.tail(18)
    elif layer=="Movie feature view": preview=data.movie_features.sort_values("rating_count",ascending=False).head(18)
    else: preview=data.quarantine if not data.quarantine.empty else pd.DataFrame({"state":["No quarantined interactions"]})
    st.dataframe(preview,hide_index=True,width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">AI engineering evaluation</div><h2>Hide the future.<br>Earn the recommendation.</h2></section>',unsafe_allow_html=True)
    metrics=st.columns(7); items=[("HitRate@20",f"{model.metrics['hit_rate_at_k']:.1%}"),("Popularity HR@20",f"{model.metrics['baseline_hit_rate_at_k']:.1%}"),("Hit lift",f"{model.metrics['hit_lift_vs_popularity']:+.1%}"),("MRR@20",f"{model.metrics['mrr_at_k']:.3f}"),("NDCG@20",f"{model.metrics['ndcg_at_k']:.3f}"),("Coverage",f"{model.metrics['catalog_coverage']:.1%}"),("Novelty",f"{model.metrics['mean_novelty_bits']:.1f} bits")]
    for col,(label,value) in zip(metrics,items): col.metric(label,value)
    if model.metrics["hit_rate_at_k"]<=model.metrics["baseline_hit_rate_at_k"]: st.warning("Promotion gate failed: latent factors did not beat popularity on full-catalog HitRate@20. Treat the model as an evaluated candidate, not a production default.")
    else: st.success("Promotion gate passed: latent factors beat the popularity baseline on full-catalog HitRate@20.")
    st.caption("For every eligible user, the chronologically last rating of at least four stars is removed. Ranking is performed against the complete unseen catalog—not a sampled set of easy negatives.")
    el,er=st.columns([1.1,.9])
    with el:
        comparison=pd.DataFrame({"metric":["HitRate@20","MRR@20","Catalog coverage"],"Latent factors":[model.metrics["hit_rate_at_k"],model.metrics["mrr_at_k"],model.metrics["catalog_coverage"]],"Popularity":[model.metrics["baseline_hit_rate_at_k"],model.metrics["baseline_mrr_at_k"],model.metrics["baseline_catalog_coverage"]]}).melt("metric",var_name="system",value_name="score")
        fig=px.bar(comparison,x="metric",y="score",color="system",barmode="group",title="Candidate versus non-personalized baseline",labels={"metric":"","score":"Score","system":""}); st.plotly_chart(_style(fig,440),width="stretch")
    with er:
        ranks=model.evaluation.assign(rank_display=model.evaluation["rank"].replace(0,21)); fig=px.histogram(ranks,x="rank_display",nbins=21,title="Rank of the hidden positive item",labels={"rank_display":"Rank (21 = not retrieved)","count":"Users"}); st.plotly_chart(_style(fig,440),width="stretch")
    latent=model.svd.components_.T; embedding=pd.DataFrame({"movieId":model.movie_ids,"factor_1":latent[:,0],"factor_2":latent[:,1]}).merge(data.movie_features[["movieId","title","genres","rating_count"]],on="movieId"); embedding=embedding.nlargest(1200,"rating_count")
    fig=px.scatter(embedding,x="factor_1",y="factor_2",size="rating_count",hover_name="title",hover_data=["genres"],title="Learned item space · two latent dimensions",labels={"factor_1":"Latent factor 1","factor_2":"Latent factor 2","rating_count":"Ratings"}); st.plotly_chart(_style(fig,500),width="stretch")

    st.markdown('<section class="section-intro"><div class="section-kicker">Recommendation workbench</div><h2>Known taste or cold start.<br>Both paths stay explicit.</h2></section>',unsafe_allow_html=True)
    mode=st.radio("Serving path",["Known user","Cold start"],horizontal=True)
    if mode=="Known user":
        eligible=set(model.holdout["userId"].astype(int)); active=[int(user) for user in model.train.groupby("userId").size().sort_values(ascending=False).index if int(user) in eligible][:120]; controls=st.columns([1,.8])
        user=controls[0].selectbox("Anonymized MovieLens user",active); novelty_weight=controls[1].slider("Novelty weight",0.0,.7,.25,.05)
        history=model.train[model.train["userId"].eq(user)].merge(data.movies,on="movieId").sort_values(["rating","rated_at"],ascending=False)
        recommendations=recommend_user(model,data.movies,user,12,novelty_weight)
        profile=st.columns(4); profile[0].metric("Training ratings",f"{len(history):,}"); profile[1].metric("Liked films",f"{history['rating'].ge(4).sum():,}"); profile[2].metric("Mean rating",f"{history['rating'].mean():.2f}"); profile[3].metric("Hidden test title",data.movies.set_index("movieId").loc[int(model.holdout.loc[model.holdout['userId'].eq(user),'movieId'].iloc[0]),"title"])
        rl,rr=st.columns([1.15,.85]); rl.dataframe(recommendations,hide_index=True,width="stretch"); rr.dataframe(history[["title","genres","rating"]].head(12),hide_index=True,width="stretch")
    else:
        genres=sorted({genre for value in data.movies["genres"] for genre in str(value).split("|") if genre!="(no genres listed)"}); selected=st.multiselect("Preferred genres",genres,default=["Drama","Sci-Fi"])
        if not selected: st.info("Select at least one genre to activate the explicit cold-start fallback.")
        else: st.dataframe(cold_start_recommendations(model,data.movies,selected,12),hide_index=True,width="stretch")
        st.caption("Cold start deliberately uses declared genres plus training popularity. It does not pretend latent collaborative factors exist before interaction history is available.")
    exports=st.columns(3); exports[0].download_button("Export Gold movies",data.movie_features.to_csv(index=False).encode(),f"movielens_gold_{data.metadata['run_id']}.csv","text/csv",width="stretch"); exports[1].download_button("Export evaluation",model.evaluation.to_csv(index=False).encode(),f"movielens_evaluation_{data.metadata['run_id']}.csv","text/csv",width="stretch"); exports[2].download_button("Export run manifest",_manifest(data,model),f"movielens_manifest_{data.metadata['run_id']}.json","application/json",width="stretch")
    with st.expander("Method, data rights and production boundaries"):
        st.markdown(f"""
        **Dataset.** [MovieLens latest-small]({README_URL}) contains 100,836 ratings, 3,683 tags and 9,742 movies from 610 anonymized users. It was generated on 26 September 2018 and is a static development dataset—not current market behavior.

        **Model.** Explicit ratings become non-negative implicit preference weights. Truncated SVD learns user and item factors. The newest positive interaction per eligible user is hidden, all earlier interactions train, and evaluation ranks against the full unseen catalog. Popularity is the explicit baseline.

        **Rights.** GroupLens permits research use with attribution and same-condition redistribution, forbids implied endorsement and requires permission for commercial or revenue-bearing use. See the [official README/license]({README_URL}), [dataset page]({DATASETS_URL}) and [dataset paper]({CITATION_URL}).

        **Limitations.** IDs are anonymized, genres and titles can contain errors, exposure is unobserved, missing ratings are not dislikes, and offline ranking metrics do not prove user satisfaction. This independent portfolio app is not endorsed by GroupLens or the University of Minnesota.
        """)
