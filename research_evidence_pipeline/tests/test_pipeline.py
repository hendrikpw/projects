"""Tests for ingestion contracts, deterministic lineage and quality reporting."""

from __future__ import annotations

import pandas as pd

from research_evidence_pipeline.src.data import build_demo_records, safe_custom_query
from research_evidence_pipeline.src.pipeline import (
    REQUIRED_SILVER_COLUMNS,
    bronze_table,
    gold_table,
    quality_report,
    silver_table,
    stable_hash,
)


def test_safe_custom_query_is_bounded_and_requires_abstract() -> None:
    query = safe_custom_query("federated learning; DROP TABLE works! radiology extra terms here please now")
    assert "HAS_ABSTRACT:Y" in query
    assert ";" not in query
    assert "DROP" in query
    assert query.count('"') <= 16


def test_stable_hash_ignores_dictionary_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_pipeline_layers_reconcile_and_are_deterministic() -> None:
    records = build_demo_records()
    bronze_first = bronze_table(records)
    bronze_second = bronze_table(records)
    assert bronze_first["payload_hash"].tolist() == bronze_second["payload_hash"].tolist()

    silver = silver_table(bronze_first)
    gold = gold_table(silver)
    assert REQUIRED_SILVER_COLUMNS.issubset(silver.columns)
    assert len(bronze_first) == len(silver) == len(gold) == 48
    assert gold["document_hash"].is_unique
    assert gold["document_text"].str.len().gt(gold["abstract"].str.len()).all()


def test_silver_contract_drops_invalid_and_duplicate_rows() -> None:
    records = build_demo_records()[:5]
    records.append(records[0].copy())
    records.append({"id": "BAD", "source": "DEMO", "title": "short", "abstractText": "tiny"})
    silver = silver_table(bronze_table(records))
    assert len(silver) == 5
    assert silver["record_id"].is_unique


def test_quality_report_has_explicit_passes_and_denominators() -> None:
    bronze = bronze_table(build_demo_records())
    silver = silver_table(bronze)
    gold = gold_table(silver)
    report = quality_report(bronze, silver, gold)
    assert report["passed"].all()
    assert {"check", "passed", "detail"} == set(report.columns)
    assert len(report) >= 8


def test_gold_dates_and_counts_have_operational_types() -> None:
    gold = gold_table(silver_table(bronze_table(build_demo_records())))
    assert pd.api.types.is_datetime64_any_dtype(gold["publication_date"])
    assert pd.api.types.is_integer_dtype(gold["cited_by_count"])
    assert gold["abstract_words"].min() > 10

