"""Streamlit product for an observable research pipeline and evidence retriever."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from research_evidence_pipeline.src.data import (
    ABOUT_URL,
    API_DOCS_URL,
    API_URL,
    QUERY_PRESETS,
    SEARCH_HELP_URL,
    safe_custom_query,
)
from research_evidence_pipeline.src.pipeline import PipelineBundle, run_pipeline
from research_evidence_pipeline.src.retrieval import (
    build_index,
    evaluate_retrieval,
    evidence_brief,
    search,
)


PALETTE = ["#e5484d", "#fcfcfd", "#9ba0a8", "#656a73", "#3c4149", "#24282e"]
DEFAULT_QUESTIONS = {
    "Clinical AI": "How are machine-learning systems externally validated in clinical care?",
    "Antimicrobial resistance": "Which machine-learning methods predict antimicrobial resistance?",
    "Climate & health": "How is climate exposure linked to measurable health outcomes?",
    "Rare-disease diagnostics": "How can phenotype matching improve rare-disease diagnosis?",
    "Digital mental health": "How are digital mental-health interventions evaluated?",
}


def _style_figure(figure: go.Figure, height: int = 450) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=62, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="#171a20", font_color="#fcfcfd", bordercolor="#5c6068"),
        colorway=PALETTE,
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=21_600, show_spinner=False)
def _cached_pipeline(query: str, page_size: int) -> PipelineBundle:
    return run_pipeline(query, page_size)


def _manifest_json(bundle: PipelineBundle, evaluation: dict) -> str:
    payload = {
        "pipeline": bundle.metadata,
        "stages": bundle.events.to_dict(orient="records"),
        "quality_checks": bundle.quality.to_dict(orient="records"),
        "retrieval_evaluation": evaluation,
    }
    return json.dumps(payload, indent=2, default=str)


def _render_evidence_cards(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("No document matched the question. Try broader terminology or a larger corpus.")
        return
    for _, row in results.head(6).iterrows():
        with st.container(border=True):
            top = st.columns([0.12, 0.68, 0.20])
            top[0].markdown(f"### {int(row['rank']):02d}")
            top[1].markdown(f"**{row['title']}**")
            top[1].caption(f"{row['authors']} · {int(row['publication_year'])} · {row['journal']}")
            top[2].metric("Relevance", f"{row['relevance_percent']:.1f}%")
            st.write(row["abstract"][:680] + ("…" if len(row["abstract"]) > 680 else ""))
            meta = st.columns(4)
            meta[0].caption(f"Semantic {row['semantic_score']:.3f}")
            meta[1].caption(f"Lexical {row['lexical_score']:.3f}")
            meta[2].caption(f"Citations {int(row['cited_by_count'])}")
            meta[3].link_button("Open evidence", row["epmc_url"], width="stretch")


def render_dashboard() -> None:
    """Render the complete Data Engineering + AI Engineering mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">14 / Data + AI engineering</div>
          <h1>Research Evidence<br>Pipeline.</h1>
          <p>
            Ingest scientific metadata, enforce a versioned data contract and
            turn the validated corpus into an evaluated, citation-bound evidence engine.
          </p>
          <div class="source-line">Europe PMC · Bronze / Silver / Gold · Hybrid semantic retrieval</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1.05, 1.6, 0.75])
    with controls[0]:
        preset = st.selectbox("Research corpus", list(QUERY_PRESETS), index=0)
    with controls[1]:
        custom_terms = st.text_input("Optional custom topic", placeholder="e.g. federated learning radiology")
    with controls[2]:
        page_size = st.select_slider("Batch size", options=[50, 75, 100, 150, 200], value=100)
    source_query = safe_custom_query(custom_terms) if custom_terms.strip() else QUERY_PRESETS[preset]

    with st.spinner("Running extract, contract validation and AI-ready feature build…"):
        bundle = _cached_pipeline(source_query, page_size)
    if bundle.metadata["mode"] == "demo":
        st.warning(
            "Europe PMC is currently unavailable. The entire pipeline and AI interface are running "
            "on deterministic synthetic publication metadata; no result represents current literature."
        )
    else:
        retrieved = pd.to_datetime(bundle.metadata["retrieved_at"], utc=True)
        st.success(
            f"Live Europe PMC batch · {bundle.metadata['hit_count']:,} matching records in the source · "
            f"retrieved {retrieved.strftime('%d %b %Y, %H:%M UTC')}"
        )

    index = build_index(bundle.gold)
    evaluation = evaluate_retrieval(bundle.gold, sample_size=min(40, len(bundle.gold)))
    retention = len(bundle.silver) / max(len(bundle.bronze), 1)
    oa_share = bundle.gold["is_open_access"].mean()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Source matches", f"{bundle.metadata['hit_count']:,}")
    k2.metric("AI-ready documents", f"{len(bundle.gold):,}")
    k3.metric("Pipeline retention", f"{retention:.1%}")
    k4.metric("Contract pass rate", f"{bundle.metadata['quality_pass_rate']:.0%}")
    k5.metric("Open-access flag", f"{oa_share:.1%}")
    k6.metric("Retrieval Hit@5", f"{evaluation['hit_rate_at_5']:.1%}")
    st.caption(
        f"Run {bundle.metadata['run_id']} · source batch → {len(bundle.bronze):,} bronze → "
        f"{len(bundle.silver):,} silver → {len(bundle.gold):,} gold. Open-access is a source flag; "
        "rights to individual abstracts and articles remain publication-specific."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Data engineering control plane</div>
          <h2>Trust the corpus<br>before the model.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 0.85])
    with left:
        flow = bundle.events.copy()
        flow["row_label"] = flow["output_rows"].map(lambda value: f"{value:,} rows")
        chart = px.bar(
            flow,
            x="stage",
            y="output_rows",
            color="status",
            text="row_label",
            hover_data={"input_rows": True, "dropped_rows": True, "duration_ms": ":.2f", "content_hash": True},
            color_discrete_map={"passed": "#e5484d", "failed": "#737982"},
            title="Rows and execution metadata by pipeline stage",
            labels={"stage": "", "output_rows": "Output rows", "status": ""},
        )
        chart.update_traces(textposition="outside")
        st.plotly_chart(_style_figure(chart, 470), width="stretch")
    with right:
        st.markdown("#### Pipeline run ledger")
        st.dataframe(
            bundle.events[["stage", "status", "input_rows", "output_rows", "dropped_rows", "duration_ms", "content_hash"]],
            width="stretch",
            hide_index=True,
            column_config={
                "duration_ms": st.column_config.NumberColumn("Duration · ms", format="%.2f"),
                "content_hash": "Content hash",
            },
        )
        st.caption(
            "Content-addressed run and document hashes make repeated payloads comparable. "
            "The hosted demo executes in memory; a production extension can persist the same layers to object storage."
        )

    quality_left, quality_right = st.columns([1, 1])
    with quality_left:
        check_plot = bundle.quality.copy()
        check_plot["result"] = check_plot["passed"].map({True: "Passed", False: "Failed"})
        qa = px.bar(
            check_plot,
            y="check",
            x=[1] * len(check_plot),
            color="result",
            orientation="h",
            title="Data-contract and reconciliation checks",
            color_discrete_map={"Passed": "#e5484d", "Failed": "#656a73"},
            labels={"value": "", "check": "", "result": ""},
            hover_data={"detail": True},
        )
        qa.update_xaxes(visible=False)
        qa.update_layout(showlegend=False)
        st.plotly_chart(_style_figure(qa, 460), width="stretch")
    with quality_right:
        layer = st.radio("Inspect layer", ["Silver contract", "Gold AI-ready"], horizontal=True)
        if layer == "Silver contract":
            preview = bundle.silver[["record_id", "title", "publication_date", "cited_by_count", "is_open_access"]].head(15)
        else:
            preview = bundle.gold[["record_id", "abstract_words", "citation_band", "document_hash"]].head(15)
        st.dataframe(preview, width="stretch", hide_index=True)
        st.download_button(
            "Export validated Gold CSV",
            bundle.gold.drop(columns=["document_text"]).to_csv(index=False).encode("utf-8"),
            file_name=f"research_evidence_gold_{bundle.metadata['run_id']}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Corpus diagnostics</div>
          <h2>Measure coverage.<br>Expose selection bias.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    corpus_left, corpus_right = st.columns(2)
    with corpus_left:
        years = bundle.gold.groupby("publication_year", as_index=False).agg(
            documents=("record_id", "size"), citations=("cited_by_count", "sum")
        )
        year_chart = px.bar(
            years,
            x="publication_year",
            y="documents",
            color="citations",
            color_continuous_scale=[[0, "#3c4149"], [1, "#e5484d"]],
            title="Publication coverage by year",
            labels={"publication_year": "Publication year", "documents": "Documents", "citations": "Citations"},
        )
        st.plotly_chart(_style_figure(year_chart, 440), width="stretch")
    with corpus_right:
        venue = (
            bundle.gold.groupby("journal", as_index=False)
            .agg(documents=("record_id", "size"), open_access=("is_open_access", "mean"))
            .sort_values(["documents", "journal"], ascending=[False, True])
            .head(12)
        )
        venue_chart = px.bar(
            venue.sort_values("documents"),
            x="documents",
            y="journal",
            orientation="h",
            color="open_access",
            color_continuous_scale=[[0, "#3c4149"], [1, "#e5484d"]],
            title="Most represented publication venues",
            labels={"documents": "Documents", "journal": "", "open_access": "OA share"},
        )
        st.plotly_chart(_style_figure(venue_chart, 440), width="stretch")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">AI evidence workbench</div>
          <h2>Retrieve meaning.<br>Keep every claim grounded.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    search_controls = st.columns([2.1, 0.9, 0.7])
    with search_controls[0]:
        question = st.text_input("Evidence question", value=DEFAULT_QUESTIONS[preset])
    with search_controls[1]:
        semantic_weight = st.slider("Semantic weight", 0.0, 1.0, 0.70, 0.05)
    with search_controls[2]:
        top_k = st.slider("Results", 3, 15, 8)
    results, diagnostics = search(index, question, top_k=top_k, semantic_weight=semantic_weight)
    brief = evidence_brief(results, question)

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Confidence gate", diagnostics["confidence"])
    a2.metric("Top relevance", f"{diagnostics['top_score']:.3f}")
    a3.metric("MRR@10", f"{evaluation['mrr_at_10']:.3f}")
    a4.metric("Median eval rank", f"{evaluation['median_rank']:.1f}")
    a5.metric("Embedding vocabulary", f"{evaluation.get('vocabulary_size', 0):,}")
    if diagnostics["zero_vector"]:
        st.error("The question contains no terms represented in this corpus. The system withheld an evidence brief.")
    elif diagnostics["confidence"] == "Weak match":
        st.warning("Retrieval confidence is weak. Treat the evidence brief as a navigation aid and broaden the corpus or query.")

    brief_left, brief_right = st.columns([1.25, 0.75])
    with brief_left:
        st.markdown("#### Citation-bound evidence brief")
        st.markdown(f"**{brief['headline']}**")
        for finding in brief["findings"]:
            st.write(finding)
        st.caption(brief["caveat"])
    with brief_right:
        st.markdown("#### Retrieval evaluation")
        st.write(
            f"**{evaluation['evaluated_queries']} title queries** were matched against abstracts only. "
            f"The correct publication appeared in the top five for **{evaluation['hit_rate_at_5']:.1%}** "
            f"of evaluated queries; zero-vector rate was **{evaluation['zero_query_rate']:.1%}**."
        )
        st.caption(
            "This deterministic title-to-abstract test monitors index health. It does not prove clinical relevance, "
            "answer correctness or systematic-review completeness."
        )
        st.download_button(
            "Export run manifest",
            _manifest_json(bundle, evaluation),
            file_name=f"research_evidence_manifest_{bundle.metadata['run_id']}.json",
            mime="application/json",
            width="stretch",
        )

    _render_evidence_cards(results)
    if not results.empty:
        export_columns = [
            "rank", "record_id", "title", "authors", "publication_year", "journal",
            "relevance_score", "semantic_score", "lexical_score", "epmc_url",
        ]
        st.download_button(
            "Export ranked evidence CSV",
            results[export_columns].to_csv(index=False).encode("utf-8"),
            file_name="ranked_research_evidence.csv",
            mime="text/csv",
        )

    with st.expander("Architecture, source scope and responsible-use notes"):
        st.markdown(
            f"""
            **Architecture:** Europe PMC REST → content-addressed Bronze → validated Silver → AI-ready Gold →
            TF-IDF lexical index + Truncated-SVD latent-semantic index → hybrid ranking → extractive evidence brief.

            **Source:** [Europe PMC REST API]({API_URL}) · [API documentation]({API_DOCS_URL}) ·
            [search syntax]({SEARCH_HELP_URL}) · [provider overview]({ABOUT_URL}). The app retrieves a bounded
            relevance/date-sorted batch, not the complete literature universe.

            **Responsible use:** This is a literature-discovery tool, not medical advice, a systematic review or
            a generative clinical assistant. The brief only reuses sentences from retrieved abstracts and keeps
            direct source links visible. Article and abstract reuse rights can differ by publication.
            """
        )
