from __future__ import annotations

import pandas as pd
import pytest

from wikipedia_attention_pipeline.src import data
from wikipedia_attention_pipeline.src.data import _demo_records, fetch_pageviews, safe_articles
from wikipedia_attention_pipeline.src.pipeline import (
    bronze_events, gold_features, normalize_and_contract, run_pipeline, stable_hash,
)


def _records(days: int = 180):
    return _demo_records(["Artificial_intelligence", "Machine_learning"], days, pd.Timestamp("2026-08-06", tz="UTC").to_pydatetime())


def test_article_allowlist_preserves_order_and_removes_duplicates():
    assert safe_articles(["Machine_learning", "bad", "Artificial_intelligence", "Machine_learning"]) == ["Machine_learning", "Artificial_intelligence"]


def test_fallback_is_deterministic_and_complete():
    first = _records(120); second = _records(120)
    assert stable_hash(first) == stable_hash(second)
    assert len(first) == 240
    assert {row["granularity"] for row in first} == {"daily"}


def test_micro_batches_expose_duplicates_lateness_and_monotonic_watermarks():
    bronze, batches = bronze_events(_records(), micro_batch_size=60)
    assert len(bronze) > len(_records())
    assert bronze["replayed_duplicate"].any()
    assert batches["watermark"].is_monotonic_increasing
    assert batches["late_events"].sum() == bronze["late_beyond_watermark"].sum()


def test_contract_quarantines_duplicate_replays_and_reconciles():
    bronze, _ = bronze_events(_records(), 90)
    silver, quarantine = normalize_and_contract(bronze)
    assert silver["event_id"].is_unique
    assert len(silver) + len(quarantine) == len(bronze)
    assert quarantine["contract_error"].str.contains("duplicate").all()


def test_gold_features_are_shifted_and_complete():
    bronze, _ = bronze_events(_records(), 90)
    silver, _ = normalize_and_contract(bronze)
    gold = gold_features(silver)
    first = gold[gold["article"].eq("Artificial_intelligence")].iloc[0]
    history = silver[(silver["article"].eq(first["article"])) & (silver["event_time"] < first["event_time"])].sort_values("event_time")
    assert first["lag_1"] == history.iloc[-1]["views"]
    assert first["lag_7"] == history.iloc[-7]["views"]
    assert gold[["lag_1", "lag_7", "lag_14", "rolling_28"]].notna().all().all()


def test_pipeline_replay_is_idempotent(monkeypatch):
    records = _records()
    metadata = {"mode": "demo", "fallback_reason": "test", "retrieved_at": "2026-08-08T00:00:00+00:00",
                "articles": ["Artificial_intelligence", "Machine_learning"], "history_days": 180,
                "start_date": "2026-02-08", "end_date": "2026-08-06", "requests": 2,
                "source_counts": [], "source_url": "demo"}
    monkeypatch.setattr("wikipedia_attention_pipeline.src.pipeline.fetch_pageviews", lambda *_: (records, metadata))
    first = run_pipeline(metadata["articles"], 180, 90); second = run_pipeline(metadata["articles"], 180, 90)
    assert first.metadata["run_id"] == second.metadata["run_id"]
    assert first.metadata["gold_hash"] == second.metadata["gold_hash"]
    assert first.quality["passed"].all()


def test_api_failure_uses_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data, "_request_json", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    rows, metadata = fetch_pageviews(["Artificial_intelligence", "Machine_learning"], 120)
    assert metadata["mode"] == "demo"
    assert len(rows) == 240
    assert "offline" in metadata["fallback_reason"]


def test_retry_recovers_from_transient_failure(monkeypatch):
    calls = {"count": 0}
    class Response:
        status_code = 200
        def raise_for_status(self): return None
        def json(self): return {"items": [{"views": 1}]}
    def get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1: raise data.requests.Timeout("temporary")
        return Response()
    monkeypatch.setattr(data.requests, "get", get); monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    assert data._request_json("https://example.test", 2)["items"]
    assert calls["count"] == 2


def test_invalid_source_configuration_fails_closed():
    with pytest.raises(ValueError, match="at least one"):
        fetch_pageviews([], 120)
