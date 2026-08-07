"""Streamlit data control plane and selective NLP triage workbench."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fda_recall_nlp_pipeline.src.data import API_DOCS_URL, AUTH_URL, DOMAINS, TERMS_URL
from fda_recall_nlp_pipeline.src.model import ModelBundle, score_text, train_and_evaluate
from fda_recall_nlp_pipeline.src.pipeline import PipelineBundle, run_pipeline


ACCENT = "#22c55e"
PALETTE = [ACCENT, "#38bdf8", "#f59e0b", "#fcfcfd", "#8b92a1"]


def _style(figure: go.Figure, height: int = 430) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=62, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"),
        colorway=PALETTE,
        legend=dict(orientation="h", y=1.03, title=None),
        hoverlabel=dict(bgcolor="#151a1f", font_color="#fcfcfd", bordercolor="#4b5563"),
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=21_600, show_spinner=False)
def _pipeline(domains: tuple[str, ...], snapshot_size: int) -> PipelineBundle:
    return run_pipeline(list(domains), snapshot_size)


@st.cache_resource(show_spinner=False)
def _model(feature_hash: str, _features: pd.DataFrame) -> ModelBundle:
    return train_and_evaluate(_features)


def _manifest(bundle: PipelineBundle, model: ModelBundle) -> str:
    payload = {
        "pipeline": bundle.metadata,
        "events": bundle.events.to_dict(orient="records"),
        "quality": bundle.quality.to_dict(orient="records"),
        "model": {**model.metadata, **model.metrics},
        "drift": model.drift.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, default=str)


def render_dashboard() -> None:
    st.markdown(
        """
        <style>
        .recall-alert {padding:1rem 1.15rem;border:1px solid rgba(34,197,94,.35);background:rgba(34,197,94,.07);border-radius:4px}
        .recall-stage {padding:1rem;border-top:3px solid #22c55e;background:rgba(255,255,255,.035);min-height:132px}
        .recall-stage b {color:#fcfcfd;font-size:.86rem}.recall-stage p {font-size:.78rem;margin:.6rem 0 0}
        .recall-result {padding:1.25rem;border:1px solid rgba(56,189,248,.35);background:rgba(56,189,248,.07);border-radius:4px}
        </style>
        <section class="page-hero">
          <div class="eyebrow">16 / Data + AI engineering</div>
          <h1>FDA Recall<br>Triage Pipeline.</h1>
          <p>
            Contract three public enforcement streams into an observable data product,
            then evaluate a confidence-aware NLP model that can defer uncertain cases.
          </p>
          <div class="source-line">openFDA / RES · Idempotent lineage · Selective multi-class NLP</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1.5, 0.8])
    with controls[0]:
        domains = st.multiselect(
            "Enforcement streams",
            list(DOMAINS),
            default=list(DOMAINS),
            format_func=lambda value: DOMAINS[value]["label"],
        )
    with controls[1]:
        snapshot_size = st.select_slider("Target snapshot", options=[180, 270, 360, 450, 540], value=360)
    if not domains:
        st.info("Select at least one enforcement stream to start the contracted pipeline.")
        return

    try:
        with st.spinner("Ingesting class strata, enforcing contracts and evaluating the NLP model…"):
            bundle = _pipeline(tuple(domains), snapshot_size)
            model = _model(bundle.metadata["feature_hash"], bundle.features)
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        st.error("The run was stopped because a trustworthy data product or model could not be produced.")
        st.caption(f"Failure state · {type(exc).__name__}: {exc}")
        st.info("No classification is shown after a failed contract or evaluation run. Re-select all three streams and retry.")
        return

    if bundle.metadata["mode"] == "demo":
        st.warning(
            "openFDA was unavailable or returned an incomplete class stratum. The full pipeline and model lifecycle are running "
            "on a deterministic, source-shaped demo snapshot. Demo outputs are clearly separated from FDA records."
        )
        st.caption("Fallback reason: " + bundle.metadata["fallback_reason"])
    else:
        retrieved = pd.to_datetime(bundle.metadata["retrieved_at"], utc=True)
        st.success(
            f"Live openFDA snapshot · {len(bundle.snapshot):,} records across {len(domains)} streams · "
            f"retrieved {retrieved.strftime('%d %b %Y, %H:%M UTC')}"
        )

    accepted_at_55 = model.selective.loc[model.selective["threshold"].eq(0.55)].iloc[0]
    columns = st.columns(6)
    cards = [
        ("Ingested", f"{len(bundle.snapshot):,}", None),
        ("Validated", f"{len(bundle.validated):,}", None),
        ("Quarantined", f"{len(bundle.quarantine):,}", None),
        ("DQ pass", f"{bundle.metadata['quality_pass_rate']:.0%}", None),
        ("Macro F1", f"{model.metrics['macro_f1']:.3f}", None),
        ("Coverage @ .55", f"{accepted_at_55['coverage']:.1%}", "Share not deferred by the model"),
    ]
    for column, (label, value, help_text) in zip(columns, cards):
        column.metric(label, value, help=help_text)
    st.caption(
        f"Run {bundle.metadata['run_id']} · {bundle.metadata['request_count']} bounded requests · "
        f"{model.metadata['training_rows']} train / {model.metadata['holdout_rows']} holdout · {model.metadata['split_strategy']}."
    )
    st.markdown(
        '<div class="recall-alert"><b>Decision boundary:</b> This portfolio model reconstructs FDA recall classes from historical report language. '
        'It cannot classify a real recall, establish a hazard, or replace FDA and qualified safety review. Low-confidence inputs are deliberately deferred.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Data engineering control plane</div>
          <h2>Three endpoints.<br>One auditable contract.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    stage_columns = st.columns(4)
    stages = [
        ("01 · INGEST", "Nine bounded class strata, HTTPS timeouts, two retry attempts and a deterministic fallback."),
        ("02 · SNAPSHOT", "Canonical JSON and SHA-256 content hashes make identical payloads and replays observable."),
        ("03 · CONTRACT", "Food, drug and device fields become typed records; invalid and duplicate rows enter quarantine."),
        ("04 · NLP VIEW", "Text is assembled without the target field, reconciled row-for-row and versioned by content hash."),
    ]
    for column, (title, body) in zip(stage_columns, stages):
        column.markdown(f'<div class="recall-stage"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        flow = bundle.events.copy()
        figure = px.bar(
            flow, x="stage", y="output_rows", text="output_rows", color="status",
            color_discrete_map={"passed": ACCENT, "failed": "#ef4444"},
            hover_data={"input_rows": True, "rejected_rows": True, "duration_ms": ":.2f", "content_hash": True},
            title="Rows, latency and lineage by stage", labels={"stage": "", "output_rows": "Rows", "status": ""},
        )
        figure.update_traces(textposition="outside")
        st.plotly_chart(_style(figure, 450), width="stretch")
    with right:
        st.markdown("#### Run ledger")
        st.dataframe(bundle.events, hide_index=True, width="stretch")
        st.caption("Content hashes are stable audit keys. In production they can address immutable object-store partitions and suppress duplicate loads.")

    qa_left, qa_right = st.columns([0.9, 1.1])
    with qa_left:
        qa = bundle.quality.assign(result=lambda frame: frame["passed"].map({True: "Passed", False: "Failed"}))
        figure = px.bar(
            qa, y="check", x=[1] * len(qa), color="result", orientation="h",
            color_discrete_map={"Passed": ACCENT, "Failed": "#ef4444"}, hover_data={"detail": True},
            title="Contract, reconciliation and leakage checks", labels={"value": "", "check": "", "result": ""},
        )
        figure.update_xaxes(visible=False)
        figure.update_layout(showlegend=False)
        st.plotly_chart(_style(figure, 465), width="stretch")
    with qa_right:
        product = st.radio("Inspect product layer", ["Validated contract", "NLP view", "Quarantine"], horizontal=True)
        if product == "Validated contract":
            preview = bundle.validated[["domain", "recall_number", "report_date", "classification", "status", "recalling_firm"]].tail(18)
        elif product == "NLP view":
            preview = bundle.features[["record_id", "domain", "report_date", "classification", "document_text"]].tail(18)
        else:
            preview = bundle.quarantine if not bundle.quarantine.empty else pd.DataFrame({"state": ["No records quarantined in this run"]})
        st.dataframe(preview, hide_index=True, width="stretch")
        st.download_button(
            "Export contracted snapshot", bundle.validated.to_csv(index=False).encode("utf-8"),
            file_name=f"fda_recall_contract_{bundle.metadata['run_id']}.csv", mime="text/csv", width="stretch",
        )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">AI engineering lifecycle</div>
          <h2>Measure the model.<br>Defer uncertainty.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metrics = st.columns(5)
    for column, (label, key, inverse) in zip(metrics, [
        ("Accuracy", "accuracy", False), ("Balanced accuracy", "balanced_accuracy", False),
        ("Macro F1", "macro_f1", False), ("Log loss", "log_loss", True),
        ("Calibration error", "expected_calibration_error", True),
    ]):
        column.metric(label, f"{model.metrics[key]:.3f}", help="Lower is better" if inverse else "Higher is better")
    st.caption(
        "Evaluation uses an untouched holdout. Word and character TF-IDF feed balanced multinomial logistic regression. "
        "The API snapshot is intentionally class-stratified, so its predicted probabilities are not population prevalence estimates."
    )

    eval_left, eval_right = st.columns(2)
    with eval_left:
        confusion = model.confusion.rename_axis("Actual").reset_index().melt("Actual", var_name="Predicted", value_name="records")
        figure = px.density_heatmap(
            confusion, x="Predicted", y="Actual", z="records", text_auto=True,
            color_continuous_scale=[[0, "#111827"], [1, ACCENT]], title="Holdout confusion matrix",
        )
        st.plotly_chart(_style(figure, 420), width="stretch")
    with eval_right:
        figure = px.bar(
            model.per_class.melt("classification", value_vars=["precision", "recall", "f1"]),
            x="classification", y="value", color="variable", barmode="group", range_y=[0, 1],
            title="Performance by recall class", labels={"classification": "", "value": "Score", "variable": ""},
        )
        st.plotly_chart(_style(figure, 420), width="stretch")

    selective_left, selective_right = st.columns(2)
    with selective_left:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=model.selective["threshold"], y=model.selective["coverage"], name="Coverage", mode="lines+markers"))
        figure.add_trace(go.Scatter(x=model.selective["threshold"], y=model.selective["selective_accuracy"], name="Accepted accuracy", mode="lines+markers"))
        figure.update_layout(title="Abstention trade-off", xaxis_title="Confidence threshold", yaxis_title="Share / accuracy", yaxis_range=[0, 1.03])
        st.plotly_chart(_style(figure, 410), width="stretch")
    with selective_right:
        drift = model.drift.copy()
        figure = px.bar(
            drift, x="signal", y="value", color="status", text="value",
            color_discrete_map={"Healthy": ACCENT, "Watch": "#f59e0b"},
            hover_data={"warning_threshold": ":.2f"}, title="Input and population drift sentinels",
            labels={"signal": "", "value": "Observed shift", "status": ""},
        )
        figure.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(_style(figure, 410), width="stretch")

    terms = model.top_terms.copy()
    figure = px.bar(
        terms, x="weight", y="term", color="classification", facet_col="classification",
        facet_col_spacing=0.08, orientation="h", title="Most influential learned text features",
        labels={"weight": "Positive coefficient", "term": "", "classification": ""},
    )
    figure.update_yaxes(matches=None, showticklabels=True)
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    st.plotly_chart(_style(figure, 520), width="stretch")
    st.caption("Coefficients describe associations in this bounded corpus—not causes, medical severity rules or FDA policy.")

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Selective prediction workbench</div>
          <h2>Score a case.<br>Know when not to answer.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    form_left, form_right = st.columns([1.1, 0.9])
    with form_left:
        domain = st.selectbox("Product domain", list(DOMAINS), format_func=lambda value: DOMAINS[value]["label"])
        product_text = st.text_input("Product description", value="Sealed injectable solution, selected lots")
        reason_text = st.text_area(
            "Recall reason", value="Product may have reduced potency and could cause temporary or medically reversible effects.", height=115,
        )
        firm_text = st.text_input("Recalling firm", value="Example Manufacturing")
        threshold = st.slider("Abstention threshold", 0.35, 0.85, 0.55, 0.05)
    with form_right:
        try:
            result = score_text(model, domain, product_text, reason_text, firm_text, threshold)
            if result["abstained"]:
                st.warning(f"Deferred · confidence {result['confidence']:.1%} is below the {threshold:.0%} review threshold.")
            else:
                st.markdown(
                    f'<div class="recall-result"><b>Historical-pattern output</b><h3>{result["prediction"]}</h3>'
                    f'<p>Model confidence {result["confidence"]:.1%} · accepted by current threshold</p></div>',
                    unsafe_allow_html=True,
                )
            probability = pd.DataFrame({"classification": list(result["probabilities"]), "probability": list(result["probabilities"].values())})
            figure = px.bar(probability, x="probability", y="classification", orientation="h", range_x=[0, 1], title="Class probability vector")
            st.plotly_chart(_style(figure, 285), width="stretch")
        except ValueError as exc:
            st.info(str(exc))

    st.markdown("#### Holdout audit")
    audit = model.holdout[["record_id", "domain", "report_date", "classification", "prediction", "confidence", "correct"]].sort_values("confidence", ascending=False)
    st.dataframe(audit, hide_index=True, width="stretch")
    export_left, export_right = st.columns(2)
    export_left.download_button(
        "Export holdout predictions", audit.to_csv(index=False).encode("utf-8"),
        file_name=f"fda_recall_holdout_{bundle.metadata['run_id']}.csv", mime="text/csv", width="stretch",
    )
    export_right.download_button(
        "Export run manifest", _manifest(bundle, model).encode("utf-8"),
        file_name=f"fda_recall_manifest_{bundle.metadata['run_id']}.json", mime="application/json", width="stretch",
    )

    with st.expander("Method, source boundaries and production path"):
        st.markdown(
            f"""
            **Source.** FDA Recall Enterprise System records are retrieved through the
            [food]({DOMAINS['food']['docs']}), [drug]({DOMAINS['drug']['docs']}) and
            [device]({DOMAINS['device']['docs']}) enforcement APIs. openFDA documents weekly updates,
            publicly releasable records from 2004 onward and its [access limits]({AUTH_URL}).

            **Sampling.** Each selected domain contributes an equal bounded sample for Class I, II and III.
            This makes all classes testable but does not reproduce real-world class prevalence. `classification`
            is the supervised target and is excluded from model text.

            **Failure states.** HTTP 429/5xx and transient transport failures are retried once. An incomplete
            response switches the whole run to a deterministic demo snapshot. Contract violations enter a
            reason-coded quarantine; failed validation suppresses every model output.

            **Production path.** Persist content-addressed Bronze payloads, typed Silver records and the Gold NLP
            view in object storage; schedule weekly after FDA refresh; register the vectorizer/model; gate promotion
            on macro F1, calibration error, coverage, OOV and population-shift thresholds; send abstentions to human review.

            [openFDA API documentation]({API_DOCS_URL}) · [Terms of Service]({TERMS_URL})
            """
        )
