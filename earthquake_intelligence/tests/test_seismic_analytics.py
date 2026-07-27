"""Analytical invariants for the seismic intelligence project."""

import numpy as np

from earthquake_intelligence.src.analytics import (
    cluster_events,
    daily_activity,
    filter_events,
    gutenberg_richter_b_value,
    magnitude_frequency,
    summary_metrics,
)
from earthquake_intelligence.src.data import build_demo_data


def test_demo_data_is_deterministic_and_energy_is_positive():
    first = build_demo_data(300)
    second = build_demo_data(300)
    assert first["event_id"].tolist() == second["event_id"].tolist()
    assert np.allclose(first["magnitude"], second["magnitude"])
    assert (first["energy_joules"] > 0).all()
    assert first["is_demo"].all()


def test_filter_and_summary_are_consistent():
    data = build_demo_data(800)
    filtered = filter_events(
        data,
        days=7,
        minimum_magnitude=3.0,
        maximum_depth=300,
        reviewed_only=False,
        tsunami_only=False,
    )
    metrics = summary_metrics(filtered, completeness_magnitude=3.0)
    assert metrics["events"] == len(filtered)
    assert filtered["magnitude"].ge(3.0).all()
    assert filtered["depth_km"].le(300).all()


def test_dbscan_labels_every_event():
    data = build_demo_data(500)
    clustered = cluster_events(data, radius_km=350, minimum_events=3)
    assert len(clustered) == len(data)
    assert clustered["cluster"].notna().all()
    assert clustered["cluster"].str.match(r"Sequence \d+|Unclustered").all()


def test_b_value_and_frequency_are_well_formed():
    data = build_demo_data(1_200)
    b_value = gutenberg_richter_b_value(data["magnitude"], 2.5)
    frequency = magnitude_frequency(data, 2.5)
    assert b_value is not None and b_value > 0
    assert frequency["events_at_or_above"].is_monotonic_decreasing
    assert frequency.iloc[0]["events_at_or_above"] == len(data)


def test_daily_activity_preserves_energy_and_calendar():
    data = build_demo_data(1_000)
    daily = daily_activity(data)
    expected_days = (data["date"].max() - data["date"].min()).days + 1
    assert len(daily) == expected_days
    assert np.isclose(daily["energy_joules"].sum(), data["energy_joules"].sum())
