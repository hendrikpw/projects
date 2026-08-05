"""Tests for hybrid retrieval, evaluation and grounded evidence output."""

from __future__ import annotations

from research_evidence_pipeline.src.data import build_demo_records
from research_evidence_pipeline.src.pipeline import bronze_table, gold_table, silver_table
from research_evidence_pipeline.src.retrieval import (
    build_index,
    evaluate_retrieval,
    evidence_brief,
    search,
)


def _documents():
    return gold_table(silver_table(bronze_table(build_demo_records())))


def test_index_and_search_are_reproducible() -> None:
    index = build_index(_documents())
    first, first_meta = search(index, "phenotype matching rare disease diagnosis", top_k=6)
    second, second_meta = search(index, "phenotype matching rare disease diagnosis", top_k=6)
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert first_meta == second_meta
    assert first["relevance_score"].is_monotonic_decreasing
    assert len(first) == 6


def test_semantic_weight_changes_visible_score_components() -> None:
    index = build_index(_documents())
    lexical, _ = search(index, "heat risk forecasting public health", semantic_weight=0.0)
    semantic, _ = search(index, "heat risk forecasting public health", semantic_weight=1.0)
    assert lexical.iloc[0]["relevance_score"] == lexical.iloc[0]["lexical_score"]
    assert semantic.iloc[0]["relevance_score"] == semantic.iloc[0]["semantic_score"]


def test_empty_query_is_withheld() -> None:
    results, diagnostics = search(build_index(_documents()), "   ")
    assert results.empty
    assert diagnostics["zero_vector"] is True
    assert diagnostics["confidence"] == "No query"


def test_unknown_query_triggers_zero_vector_guardrail() -> None:
    _, diagnostics = search(build_index(_documents()), "zzzxxyyqqq")
    assert diagnostics["zero_vector"] is True


def test_evidence_brief_is_source_bound() -> None:
    results, _ = search(build_index(_documents()), "antibiotic resistance prediction", top_k=5)
    brief = evidence_brief(results, "antibiotic resistance prediction", max_sources=3)
    assert len(brief["findings"]) == 3
    assert len(brief["sources"]) == 3
    assert brief["findings"][0].startswith("[1]")
    assert all(source["url"].startswith("https://europepmc.org/article/") for source in brief["sources"])


def test_retrieval_evaluation_is_bounded_and_valid() -> None:
    metrics = evaluate_retrieval(_documents(), sample_size=24)
    assert metrics["evaluated_queries"] == 24
    assert 0 <= metrics["hit_rate_at_5"] <= 1
    assert 0 <= metrics["mrr_at_10"] <= 1
    assert metrics["median_rank"] >= 1
    assert metrics["vocabulary_size"] > 20
