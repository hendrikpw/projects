from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sec_fundamentals_control.src import data
from sec_fundamentals_control.src.model import FEATURES, _split, score_scenario, train_and_evaluate
from sec_fundamentals_control.src.pipeline import _gold, _parse, _silver, run_pipeline


def _payloads():
    return data._fallback()


def test_fallback_is_deterministic_and_complete():
    first, second = _payloads(), _payloads()
    assert first.keys() == second.keys() == data.COMPANIES.keys()
    assert all(first[key] == second[key] for key in first)


def test_parser_contract_and_frames():
    frame = _parse(_payloads())
    assert {"ticker", "metric", "frame", "value", "accession"}.issubset(frame)
    assert frame.frame.str.match(r"CY20\d{2}Q[1-4]").all()


def test_revision_resolution_prefers_latest_filing():
    bronze = _parse(_payloads()); duplicate = bronze.iloc[[0]].copy()
    duplicate["filed"] = "2029-01-01"; duplicate["value"] *= 1.1; duplicate["accession"] = "amendment"
    silver, quarantine = _silver(pd.concat([bronze, duplicate], ignore_index=True))
    chosen = silver[(silver.ticker == duplicate.iloc[0].ticker) & (silver.frame == duplicate.iloc[0].frame) & (silver.metric == duplicate.iloc[0].metric)].iloc[0]
    assert chosen.accession == "amendment" and chosen.revision_count == 2 and quarantine.empty


def test_invalid_fact_goes_to_quarantine():
    bronze = _parse(_payloads()); bronze.loc[0, "value"] = None
    silver, quarantine = _silver(bronze)
    assert "invalid_value" in set(quarantine.invalid_reason) and len(silver) < len(bronze)


def test_pipeline_hashes_are_idempotent(monkeypatch):
    payloads = _payloads(); monkeypatch.setattr("sec_fundamentals_control.src.pipeline.load_payloads", lambda: (payloads, {"mode":"demo","fallback_reason":"test","source_hash":"a"*64,"retrieved_at":"ignored","company_count":8,"api_template":"x","api_docs":"x"}))
    first, second = run_pipeline(), run_pipeline()
    assert first.metadata["run_id"] == second.metadata["run_id"]
    assert first.metadata["gold_hash"] == second.metadata["gold_hash"] and first.quality.passed.all()


def test_source_failure_activates_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data, "_get", lambda cik: (_ for _ in ()).throw(RuntimeError("offline")))
    payloads, metadata = data.load_payloads()
    assert metadata["mode"] == "demo" and len(payloads) == 8 and "offline" in metadata["fallback_reason"]


def test_temporal_splits_do_not_overlap():
    silver, _ = _silver(_parse(_payloads())); train, calibration, test = _split(_gold(silver))
    assert not set(train.frame) & set(calibration.frame)
    assert not set(train.frame) & set(test.frame) and not set(calibration.frame) & set(test.frame)


def test_model_reproducibility_and_promotion():
    silver, _ = _silver(_parse(_payloads())); gold = _gold(silver)
    first, second = train_and_evaluate(gold), train_and_evaluate(gold)
    assert first.metrics == second.metrics
    assert first.metrics["stress_average_precision"] >= first.metrics["baseline_average_precision"]
    assert first.stress_evaluation.candidate_score.notna().all()


def test_model_outputs_and_edge_scenario():
    silver, _ = _silver(_parse(_payloads())); gold = _gold(silver); model = train_and_evaluate(gold)
    latest = gold.dropna(subset=FEATURES).iloc[-1]
    result = score_scenario(model, latest, {"revenue_growth_yoy": -0.5})
    assert result["status"] in {"monitor", "review"} and np.isfinite(result["anomaly_score"])
    assert set(result["evidence"].columns) == {"feature", "local_deviation", "scenario_value"}


def test_malformed_json_fails_closed():
    with pytest.raises(json.JSONDecodeError):
        _parse({"AAPL": b"not-json"})
