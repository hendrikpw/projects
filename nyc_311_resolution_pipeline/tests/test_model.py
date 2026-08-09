from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyc_311_resolution_pipeline.src.data import _fallback
from nyc_311_resolution_pipeline.src.model import score_case, train_and_evaluate
from nyc_311_resolution_pipeline.src.pipeline import _bronze, _gold, _silver


def gold():
    silver, _ = _silver(_bronze(_fallback("2024-01-01", "2025-12-31", 3600)))
    return _gold(silver)


def test_model_outputs_and_metric_domains():
    frame = gold(); bundle = train_and_evaluate(frame)
    assert len(bundle.predictions) >= 120
    assert (bundle.predictions["predicted_median_hours"] >= 0).all()
    assert (bundle.predictions["predicted_upper_hours"] >= bundle.predictions["predicted_median_hours"]).all()
    assert 0 <= bundle.metrics["upper_coverage"] <= 1
    assert np.isfinite(list(bundle.metrics.values())).all()


def test_model_is_reproducible():
    frame = gold(); first, second = train_and_evaluate(frame), train_and_evaluate(frame)
    np.testing.assert_allclose(first.predictions["predicted_median_hours"], second.predictions["predicted_median_hours"])
    assert first.metrics == second.metrics


def test_score_case_shape():
    frame = gold(); bundle = train_and_evaluate(frame); row = frame.iloc[0]
    result = score_case(bundle, frame, {key: row[key] for key in ["agency","complaint_type","descriptor","location_type","borough","open_data_channel_type","created_hour","created_dow"]})
    assert set(result) == {"median_hours", "upper_hours", "risk_band"}
    assert result["upper_hours"] >= result["median_hours"] >= 0


def test_small_data_guard():
    with pytest.raises(ValueError, match="1,000"):
        train_and_evaluate(gold().head(500))


def test_unknown_categories_do_not_crash():
    frame = gold(); bundle = train_and_evaluate(frame)
    result = score_case(bundle, frame, {"agency":"NEW","complaint_type":"NEW","descriptor":"NEW","location_type":"NEW","borough":"NEW","open_data_channel_type":"NEW","created_hour":12,"created_dow":2})
    assert np.isfinite(result["median_hours"])
