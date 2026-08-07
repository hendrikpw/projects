from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fda_recall_nlp_pipeline.src.data import _demo_records
from fda_recall_nlp_pipeline.src.model import score_text, selective_report, train_and_evaluate
from fda_recall_nlp_pipeline.src.pipeline import enforce_contract, feature_view, normalize_snapshot, snapshot_table


def _features(per_class: int = 40) -> pd.DataFrame:
    records = _demo_records(["food", "drug", "device"], per_class)
    valid, _ = enforce_contract(normalize_snapshot(snapshot_table(records)))
    return feature_view(valid)


def test_model_output_shape_metrics_and_reproducibility():
    features = _features()
    first = train_and_evaluate(features)
    second = train_and_evaluate(features)
    assert first.confusion.shape == (3, 3)
    assert len(first.per_class) == 3
    assert 0 <= first.metrics["macro_f1"] <= 1
    assert 0 <= first.metrics["balanced_accuracy"] <= 1
    assert first.metrics == second.metrics
    assert first.holdout["prediction"].tolist() == second.holdout["prediction"].tolist()


def test_probability_vector_and_abstention_are_valid():
    model = train_and_evaluate(_features())
    result = score_text(
        model, "device", "implantable infusion pump",
        "Device may stop therapy without an alarm and could cause serious injury or death.",
        "Example Medical", threshold=0.99,
    )
    assert set(result["probabilities"]) == {"Class I", "Class II", "Class III"}
    assert np.isclose(sum(result["probabilities"].values()), 1)
    assert result["abstained"] is True


def test_selective_coverage_is_monotonic():
    actual = np.array(["a", "a", "b", "b"])
    predicted = np.array(["a", "b", "b", "a"])
    confidence = np.array([0.9, 0.7, 0.5, 0.4])
    report = selective_report(actual, predicted, confidence)
    assert report["coverage"].is_monotonic_decreasing
    assert report["coverage"].between(0, 1).all()


def test_model_exposes_explanations_and_drift_signals():
    model = train_and_evaluate(_features())
    assert model.top_terms.groupby("classification").size().eq(10).all()
    assert set(model.drift["status"]).issubset({"Healthy", "Watch"})
    assert {"Holdout word OOV", "Domain mix TVD", "Label mix TVD"}.issubset(set(model.drift["signal"]))


def test_invalid_workbench_input_fails_closed():
    model = train_and_evaluate(_features())
    with pytest.raises(ValueError, match="detailed"):
        score_text(model, "food", "x", "short", "firm")


def test_small_or_single_class_training_data_is_rejected():
    features = _features(10).head(20)
    features["classification"] = "Class I"
    with pytest.raises(ValueError, match="90 records"):
        train_and_evaluate(features)
