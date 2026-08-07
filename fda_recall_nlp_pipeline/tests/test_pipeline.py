from __future__ import annotations

import pandas as pd
import pytest

from fda_recall_nlp_pipeline.src import data
from fda_recall_nlp_pipeline.src.data import _demo_records, fetch_recalls, safe_domains
from fda_recall_nlp_pipeline.src.pipeline import (
    enforce_contract,
    feature_view,
    normalize_snapshot,
    run_pipeline,
    snapshot_table,
    stable_hash,
)


def test_safe_domains_is_allowlisted_and_canonical():
    assert safe_domains(["device", "food", "unknown", "device"]) == ["food", "device"]


def test_demo_records_are_deterministic_balanced_and_source_shaped():
    first = _demo_records(["food", "drug"], 12)
    second = _demo_records(["food", "drug"], 12)
    assert stable_hash(first) == stable_hash(second)
    counts = pd.DataFrame(first).groupby(["_domain", "classification"]).size()
    assert len(first) == 72
    assert counts.eq(12).all()


def test_stable_hash_ignores_dictionary_key_order():
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash({"b": [2, 3], "a": 1})


def test_contract_quarantines_duplicates_and_bad_text():
    records = _demo_records(["food"], 30)
    records.append(dict(records[0]))
    records[1]["reason_for_recall"] = "short"
    normalized = normalize_snapshot(snapshot_table(records))
    valid, quarantine = enforce_contract(normalized)
    assert valid["record_id"].is_unique
    assert len(quarantine) == 2
    assert quarantine["contract_error"].str.contains("duplicate_record_id|reason_text_too_short").all()


def test_feature_view_reconciles_and_excludes_target_phrase():
    normalized = normalize_snapshot(snapshot_table(_demo_records(["food"], 30)))
    valid, _ = enforce_contract(normalized)
    features = feature_view(valid)
    assert len(features) == len(valid)
    assert "classification" in features
    assert not features["document_text"].str.contains("classification Class", case=False).any()


def test_pipeline_is_idempotent_for_identical_source(monkeypatch):
    rows = _demo_records(["food", "drug", "device"], 30)
    metadata = {
        "mode": "demo", "fallback_reason": "test", "retrieved_at": "2026-08-07T00:00:00+00:00",
        "domains": ["food", "drug", "device"], "requested_snapshot_size": 270,
        "stratum_size": 30, "request_count": 9, "availability": [], "source_urls": [],
    }
    monkeypatch.setattr("fda_recall_nlp_pipeline.src.pipeline.fetch_recalls", lambda *_: (rows, metadata))
    first = run_pipeline(["food", "drug", "device"], 270)
    second = run_pipeline(["food", "drug", "device"], 270)
    assert first.metadata["run_id"] == second.metadata["run_id"]
    assert first.metadata["feature_hash"] == second.metadata["feature_hash"]
    assert first.quality["passed"].all()


def test_api_failure_switches_atomically_to_fallback(monkeypatch):
    monkeypatch.setattr(data, "_request_json", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    records, metadata = fetch_recalls(["food", "device"], 180)
    assert metadata["mode"] == "demo"
    assert "offline" in metadata["fallback_reason"]
    assert len(records) == 180


def test_request_retries_transient_error(monkeypatch):
    calls = {"count": 0}

    class Response:
        status_code = 200
        def raise_for_status(self): return None
        def json(self): return {"results": [{"ok": True}]}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise data.requests.Timeout("transient")
        return Response()

    monkeypatch.setattr(data.requests, "get", fake_get)
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    assert data._request_json("https://example.test", {}, attempts=2)["results"]
    assert calls["count"] == 2


def test_pipeline_rejects_empty_domain_selection():
    with pytest.raises(ValueError, match="at least one"):
        fetch_recalls([], 180)
