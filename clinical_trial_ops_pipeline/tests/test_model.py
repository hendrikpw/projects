"""AI model output, evaluation, drift and guardrail tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinical_trial_ops_pipeline.src.data import build_demo_studies
from clinical_trial_ops_pipeline.src.model import drift_report, score_scenario, train_and_evaluate
from clinical_trial_ops_pipeline.src.pipeline import FEATURE_COLUMNS, feature_view, normalize_snapshot, snapshot_table


def _features(size: int = 240) -> pd.DataFrame:
    return feature_view(normalize_snapshot(snapshot_table(build_demo_studies(size=size))))


def test_model_training_is_reproducible_and_holdout_is_shaped() -> None:
    first = train_and_evaluate(_features())
    second = train_and_evaluate(_features())
    assert np.allclose(first.holdout["risk_score"], second.holdout["risk_score"])
    assert list(first.holdout.columns) == [
        "nct_id", "title", "first_post_date", "overall_status", "record_url",
        "discontinued", "risk_score", "predicted_class",
    ]
    assert first.metadata["training_rows"] + first.metadata["holdout_rows"] == 240


def test_evaluation_metrics_are_bounded() -> None:
    model = train_and_evaluate(_features())
    for metric in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"):
        assert 0 <= model.metrics[metric] <= 1
    assert 0 <= model.metrics["brier_score"] <= 1
    assert np.asarray(model.metrics["confusion_matrix"]).shape == (2, 2)
    assert model.calibration["records"].sum() == model.metadata["holdout_rows"]


def test_coefficients_and_drift_are_explainable_outputs() -> None:
    model = train_and_evaluate(_features())
    assert {"feature", "coefficient", "direction", "magnitude"}.issubset(model.coefficients)
    assert model.coefficients["magnitude"].is_monotonic_decreasing
    assert {"feature", "drift_score", "level"} == set(model.drift.columns)
    assert model.drift["drift_score"].ge(0).all()


def test_drift_detects_material_distribution_change() -> None:
    features = _features()
    train = features.iloc[:160].copy()
    holdout = features.iloc[160:].copy()
    holdout["enrollment_log"] = holdout["enrollment_log"] + 8
    report = drift_report(train, holdout)
    enrollment = report.loc[report["feature"] == "enrollment_log"].iloc[0]
    assert enrollment["level"] == "High"
    assert enrollment["drift_score"] >= 0.25


def test_scenario_score_is_valid_and_reproducible() -> None:
    model = train_and_evaluate(_features())
    row = _features().iloc[0][FEATURE_COLUMNS].to_dict()
    first = score_scenario(model, row)
    second = score_scenario(model, row)
    assert 0 <= first <= 1
    assert first == second


def test_model_refuses_small_or_single_class_input() -> None:
    frame = _features(80).head(40).copy()
    frame["discontinued"] = 0
    with pytest.raises(ValueError, match="both outcome classes"):
        train_and_evaluate(frame)
