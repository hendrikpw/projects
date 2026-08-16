from __future__ import annotations

import pandas as pd
import pytest

from river_flow_early_warning.src import data, pipeline
from river_flow_early_warning.src.model import FEATURES, score_latest, train_and_evaluate


def _source():
    raw = data._fallback()
    return raw, {"mode":"demo", "fallback_reason":"test", "source_hash":"a"*64,
                 "source_bytes":len(raw), "site_count":6, "endpoint":data.API, "docs":data.DOCS}


def _product(monkeypatch):
    monkeypatch.setattr(pipeline, "load_source", _source)
    return pipeline.run_pipeline()


def test_fallback_is_deterministic_and_complete():
    first, second = data._fallback(), data._fallback()
    assert first == second and all(site.encode() in first for site in data.SITES)


def test_rdb_parser_handles_repeated_dynamic_headers():
    frame = data.parse_rdb(data._fallback())
    assert frame.site_no.nunique() == 6
    assert {"event_date", "discharge_cfs", "qualifier"}.issubset(frame)


def test_replayed_deliveries_are_idempotently_suppressed(monkeypatch):
    product = _product(monkeypatch)
    assert product.metadata["duplicate_deliveries"] == 12
    assert not product.silver.event_id.duplicated().any()
    assert len(product.bronze) == len(product.silver) + len(product.quarantine) + 12


def test_invalid_observation_is_quarantined():
    bronze = pipeline._bronze(data._fallback())
    bronze.loc[0, "discharge_cfs"] = "broken"
    silver, quarantine = pipeline._silver(bronze)
    assert "invalid_value" in set(quarantine.reason)
    assert len(silver) < bronze.event_id.nunique()


def test_pipeline_hashes_are_reproducible(monkeypatch):
    first = _product(monkeypatch); second = pipeline.run_pipeline()
    assert first.metadata["run_id"] == second.metadata["run_id"]
    assert first.metadata["gold_hash"] == second.metadata["gold_hash"]
    assert first.quality.passed.all()


def test_source_failure_uses_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data, "_request", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    raw, metadata = data.load_source()
    assert metadata["mode"] == "demo" and "offline" in metadata["fallback_reason"]
    assert data.parse_rdb(raw).site_no.nunique() == 6


def test_features_use_only_lagged_values(monkeypatch):
    product = _product(monkeypatch)
    row = product.gold.dropna(subset=FEATURES+["future_max_3d"]).iloc[100]
    source = product.silver[(product.silver.site_no==row.site_no) & (product.silver.event_date < row.event_date)].sort_values("event_date")
    assert row.log_lag_1 == pytest.approx(float(__import__("numpy").log1p(source.iloc[-1].discharge_cfs)))


def test_temporal_model_evaluation_and_promotion(monkeypatch):
    product = _product(monkeypatch); first = train_and_evaluate(product.gold); second = train_and_evaluate(product.gold)
    assert first.metrics == second.metrics
    assert first.metrics["average_precision"] > first.metrics["baseline_average_precision"]
    assert first.evaluation.event_date.min() >= pd.Timestamp("2025-01-01")
    assert first.evaluation.probability.between(0, 1).all()


def test_model_shapes_and_edge_scenario(monkeypatch):
    product = _product(monkeypatch); model = train_and_evaluate(product.gold)
    result = score_latest(model, product.gold, next(iter(data.SITES)), 3.0)
    assert result["status"] in {"normal", "watch", "alert"}
    assert 0 <= result["probability"] <= 1 and result["discharge_cfs"] >= 0
    assert set(model.drift.status) <= {"stable", "watch", "high"}


def test_malformed_source_fails_closed():
    with pytest.raises((ValueError, UnicodeDecodeError)):
        data.parse_rdb(b"not an rdb response")
