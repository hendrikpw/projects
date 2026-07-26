"""Tests for collision preprocessing and analytical invariants."""

import pandas as pd

from nyc_collision_intelligence.src.analytics import (
    daily_anomalies,
    filter_collisions,
    spatial_hotspots,
    summary_metrics,
)
from nyc_collision_intelligence.src.data import build_demo_data


def test_demo_data_is_deterministic_and_clearly_labelled():
    first = build_demo_data(250)
    second = build_demo_data(250)
    assert first["collision_id"].tolist() == second["collision_id"].tolist()
    assert first["is_demo"].all()
    assert first["timestamp"].equals(second["timestamp"])


def test_filters_and_summary_metrics_are_consistent():
    data = build_demo_data(800)
    filtered = filter_collisions(data, days=30, boroughs=["MANHATTAN"], outcome="All collisions")
    metrics = summary_metrics(filtered)
    assert metrics["crashes"] == len(filtered)
    assert metrics["injuries"] == int(filtered["number_of_persons_injured"].sum())
    assert filtered["borough"].eq("MANHATTAN").all()
    assert (
        filtered["crash_date"].max() - filtered["crash_date"].min()
        <= pd.to_timedelta(29, unit="D")
    )


def test_spatial_risk_is_bounded_and_sorted():
    hotspots = spatial_hotspots(build_demo_data(1_500), minimum_crashes=2)
    assert not hotspots.empty
    assert hotspots["risk_index"].between(0, 100).all()
    assert hotspots["risk_index"].is_monotonic_decreasing


def test_daily_anomaly_output_preserves_complete_calendar():
    data = build_demo_data(1_000)
    daily = daily_anomalies(data)
    expected_days = (data["crash_date"].max() - data["crash_date"].min()).days + 1
    assert len(daily) == expected_days
    assert {"baseline", "robust_z", "is_anomaly"}.issubset(daily.columns)
