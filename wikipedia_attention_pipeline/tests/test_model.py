from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wikipedia_attention_pipeline.src.data import _demo_records
from wikipedia_attention_pipeline.src.model import FEATURES, forecast_future, train_and_evaluate
from wikipedia_attention_pipeline.src.pipeline import bronze_events, gold_features, normalize_and_contract


def _products(days: int = 180, articles: int = 3):
    names = ["Artificial_intelligence", "Machine_learning", "Data_engineering"][:articles]
    rows = _demo_records(names, days, pd.Timestamp("2026-08-06", tz="UTC").to_pydatetime())
    bronze, _ = bronze_events(rows, 90); silver, _ = normalize_and_contract(bronze)
    return gold_features(silver), silver


def test_forecast_metrics_shapes_and_reproducibility():
    gold, silver = _products()
    first = train_and_evaluate(gold, silver, 14); second = train_and_evaluate(gold, silver, 14)
    assert first.metrics == second.metrics
    assert len(first.predictions) == 3 * 28
    assert len(first.future) == 3 * 14
    assert first.future["prediction"].ge(0).all()
    assert first.predictions["lower"].le(first.predictions["upper"]).all()


def test_metrics_and_interval_coverage_are_valid():
    model = train_and_evaluate(*_products(), horizon=7)
    assert model.metrics["wape"] >= 0
    assert model.metrics["smape"] >= 0
    assert 0 <= model.metrics["interval_coverage"] <= 1
    assert set(model.per_article["article"]) == set(model.article_map)


def test_design_contains_only_expected_past_features():
    gold, _ = _products()
    assert set(FEATURES).issubset(set(gold.columns) | {"article_index"})
    assert "views" not in FEATURES
    assert all(not name.startswith("future") for name in FEATURES)


def test_anomalies_match_interval_breaches():
    model = train_and_evaluate(*_products(), horizon=7)
    expected = model.predictions[model.predictions["outside_interval"]]
    assert len(model.anomalies) == len(expected)
    if not model.anomalies.empty:
        assert model.anomalies["severity"].gt(0).all()


def test_future_forecast_has_one_row_per_article_and_horizon():
    model = train_and_evaluate(*_products(), horizon=21)
    counts = model.future.groupby("article").size()
    assert counts.eq(21).all()
    assert model.future.groupby("article")["event_time"].apply(lambda series: series.is_monotonic_increasing).all()


def test_small_dataset_is_rejected():
    gold, silver = _products(days=90, articles=2)
    with pytest.raises(ValueError, match="240"):
        train_and_evaluate(gold.head(100), silver, 7)
