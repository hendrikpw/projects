from __future__ import annotations

import pandas as pd
import pytest

from nyc_311_resolution_pipeline.src import data
from nyc_311_resolution_pipeline.src.pipeline import _bronze, _gold, _silver, run_pipeline


def records():
    return data._fallback("2025-01-01", "2025-12-31", rows=1300)


def test_contract_and_reconciliation():
    bronze = _bronze(records()); silver, quarantine = _silver(bronze); gold = _gold(silver)
    assert len(bronze) == len(silver) + len(quarantine)
    assert silver["unique_key"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(silver["created_at"])
    assert set(["created_hour", "created_dow", "target_log_hours"]).issubset(gold)


def test_invalid_and_duplicate_records_are_quarantined():
    raw = records()[:5]
    raw.append(dict(raw[0]))
    raw[1]["closed_date"] = "2020-01-01"
    silver, quarantine = _silver(_bronze(raw))
    assert len(silver) + len(quarantine) == len(raw)
    assert {"duplicate_unique_key", "negative_resolution_time"}.issubset(set(quarantine["invalid_reason"]))


def test_idempotent_layer_hashes(monkeypatch):
    monkeypatch.setattr("nyc_311_resolution_pipeline.src.pipeline.fetch_requests", lambda **_: (records(), {"mode":"demo","fallback_reason":"test","retrieved_at":"ignored","start_date":"2025-01-01","end_date":"2025-12-31","history_days":365,"maturity_days":35,"source_rows":1300,"sample_rule":"test","source_hash":"a"*64,"source_url":"test"}))
    first, second = run_pipeline(), run_pipeline()
    assert first.metadata["run_id"] == second.metadata["run_id"]
    assert first.metadata["silver_hash"] == second.metadata["silver_hash"]
    assert first.quality["passed"].all()


def test_fallback_is_deterministic():
    assert data._fallback("2025-01-01", "2025-12-31", 20) == data._fallback("2025-01-01", "2025-12-31", 20)


def test_api_failure_uses_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data, "_request_page", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    rows, meta = data.fetch_requests(max_rows=1000)
    assert meta["mode"] == "demo" and len(rows) == 4200
    assert "offline" in meta["fallback_reason"]


def test_retry_recovers(monkeypatch):
    calls = {"n":0}
    class Response:
        status_code=200
        def raise_for_status(self): return None
        def json(self): return [{"unique_key":"1"}]
    def get(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1: raise data.requests.Timeout("temporary")
        return Response()
    monkeypatch.setattr(data.requests, "get", get); monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    assert data._request_page({}) == [{"unique_key":"1"}]
    assert calls["n"] == 2


def test_invalid_config_fails_closed():
    with pytest.raises(ValueError): data.fetch_requests(history_days=90)
