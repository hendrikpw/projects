from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from household_load_forecast.src.data import NUMERIC, SOURCE_FILE, _hourly_from_member, _safe_member, fallback_data
from household_load_forecast.src.model import make_features, score_scenario, train_and_evaluate
from household_load_forecast.src.pipeline import MEASURES, build_product, contract_silver, frame_hash, make_bronze


@pytest.fixture(scope="module")
def product_model():
    raw, metadata = fallback_data()
    product = build_product(raw, metadata)
    return product, train_and_evaluate(product.gold)


def _zip(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items(): archive.writestr(name, content)
    return stream.getvalue()


def test_fallback_is_deterministic():
    first, meta_a = fallback_data(); second, meta_b = fallback_data()
    pd.testing.assert_frame_equal(first, second); assert meta_a["source_hash"] == meta_b["source_hash"]


def test_archive_rejects_missing_allowlisted_file():
    with pytest.raises(ValueError, match="allowlist"):
        _safe_member(_zip({"other.txt": b"x"}))


def test_archive_rejects_path_traversal():
    with pytest.raises(ValueError, match="unsafe"):
        _safe_member(_zip({"../evil.txt": b"x", SOURCE_FILE: b"x"}))


def test_chunk_parser_aggregates_weighted_hour():
    header = ";".join(["Date", "Time", *NUMERIC])
    rows = ["16/12/2006;17:24:00;1;0.1;240;4;1;2;3", "16/12/2006;17:25:00;3;0.3;242;12;2;3;4"]
    content = _zip({SOURCE_FILE: (header + "\n" + "\n".join(rows)).encode()})
    archive = zipfile.ZipFile(io.BytesIO(content)); hourly, audit = _hourly_from_member(archive, archive.getinfo(SOURCE_FILE), chunk_rows=1)
    assert audit == {"source_rows": 2, "missing_source_rows": 0}; assert hourly.load_kw.iloc[0] == 2; assert hourly.kitchen_wh.iloc[0] == 3; assert hourly.readings.iloc[0] == 2


def test_bronze_replays_are_exact_and_late():
    raw, _ = fallback_data(periods=1000); bronze = make_bronze(raw, replay_rows=12)
    assert len(bronze) == 1012; assert bronze.duplicated("event_id").sum() == 12; assert bronze.after_watermark.sum() >= 11


def test_contract_quarantines_invalid_measure():
    raw, _ = fallback_data(periods=1000); bronze = make_bronze(raw, replay_rows=0); bronze.loc[0, "voltage_v"] = 999
    silver, quarantine, duplicates = contract_silver(bronze)
    assert duplicates == 0; assert len(silver) == 999; assert quarantine.quarantine_reason.iloc[0] == "voltage_out_of_range"


def test_pipeline_reconciles_and_passes(product_model):
    product, _ = product_model
    assert product.quality.passed.all(); assert len(product.bronze) == len(product.silver) + len(product.quarantine) + product.metadata["duplicates"]


def test_pipeline_is_idempotent(product_model):
    product, _ = product_model; raw, metadata = fallback_data(); rerun = build_product(raw, metadata)
    assert product.metadata["run_id"] == rerun.metadata["run_id"]; assert frame_hash(product.gold) == frame_hash(rerun.gold)


def test_feature_horizon_and_shape(product_model):
    product, _ = product_model; features, names = make_features(product.gold)
    assert len(names) == 20; assert (features.target_time - features.timestamp).eq(pd.to_timedelta(24, unit="h")).all(); assert np.isfinite(features[names].to_numpy()).all()


def test_temporal_partitions_are_ordered(product_model):
    _, model = product_model
    assert model.metadata["train_until"] < model.metadata["calibration_until"] < model.metadata["test_until"]


def test_forecast_beats_persistence(product_model):
    _, model = product_model
    assert model.metrics["mae_kw"] < model.metrics["baseline_mae_kw"]; assert model.metrics["peak_capture_at_10pct"] > model.metrics["baseline_peak_capture_at_10pct"]


def test_interval_and_output_contract(product_model):
    _, model = product_model; result = model.evaluation
    assert .70 <= model.metrics["interval_coverage"] <= .95; assert (result.lower <= result.forecast).all(); assert (result.forecast <= result.upper).all()


def test_training_is_reproducible(product_model):
    product, model = product_model; repeated = train_and_evaluate(product.gold)
    np.testing.assert_allclose(model.evaluation.forecast, repeated.evaluation.forecast); assert model.metrics["mae_kw"] == repeated.metrics["mae_kw"]


def test_normal_scenario_auto_forecasts(product_model):
    product, model = product_model; rows, _ = make_features(product.gold); result = score_scenario(model, rows.iloc[-1])
    assert result["route"] == "auto-forecast"; assert result["lower_kw"] <= result["forecast_kw"] <= result["upper_kw"]


def test_missing_and_ood_scenarios_fail_safe(product_model):
    product, model = product_model; rows, _ = make_features(product.gold); row = rows.iloc[-1]
    assert score_scenario(model, row, missing_lags=6)["route"] == "forecast-withheld"
    assert score_scenario(model, row, load_scale=1.6)["route"] in {"review", "forecast-withheld"}
