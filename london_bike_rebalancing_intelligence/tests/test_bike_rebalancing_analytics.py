"""Tests for TfL parsing and cycle-rebalancing analytical invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from london_bike_rebalancing_intelligence.src.analytics import (
    add_service_features,
    build_rebalancing_plan,
    network_metrics,
    pressure_clusters,
    quality_report,
    scenario_summary,
)
from london_bike_rebalancing_intelligence.src.data import build_demo_data, parse_bike_points


def _payload() -> list[dict]:
    return [
        {
            "id": "BikePoints_1",
            "commonName": "Test Station",
            "lat": 51.51,
            "lon": -0.12,
            "additionalProperties": [
                {"key": "TerminalName", "value": "001", "modified": "2026-08-04T06:00:00Z"},
                {"key": "NbBikes", "value": "4", "modified": "2026-08-04T06:00:00Z"},
                {"key": "NbStandardBikes", "value": "3", "modified": "2026-08-04T06:00:00Z"},
                {"key": "NbEBikes", "value": "1", "modified": "2026-08-04T06:00:00Z"},
                {"key": "NbEmptyDocks", "value": "6", "modified": "2026-08-04T06:00:00Z"},
                {"key": "NbDocks", "value": "11", "modified": "2026-08-04T06:00:00Z"},
                {"key": "Installed", "value": "true", "modified": "2026-08-04T06:00:00Z"},
                {"key": "Locked", "value": "false", "modified": "2026-08-04T06:00:00Z"},
            ],
        }
    ]


def test_tfl_parser_flattens_properties_and_preserves_capacity_gap() -> None:
    result = parse_bike_points(_payload())
    assert len(result) == 1
    assert result.iloc[0]["bikes"] == 4
    assert result.iloc[0]["ebikes"] == 1
    assert result.iloc[0]["unavailable_docks"] == 1
    assert not bool(result.iloc[0]["capacity_inconsistent"])


def test_demo_data_is_deterministic_and_valid() -> None:
    first = build_demo_data()
    second = build_demo_data()
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 420
    assert (first["bikes"] + first["empty_docks"] + first["unavailable_docks"] == first["docks"]).all()


def test_service_features_classify_empty_full_and_balanced() -> None:
    frame = build_demo_data().head(3).copy()
    frame["docks"] = 20
    frame["unavailable_docks"] = 0
    frame["bikes"] = [1, 10, 19]
    frame["empty_docks"] = 20 - frame["bikes"]
    result = add_service_features(frame, critical_threshold=0.1, target_fill=0.5)
    assert result["service_status"].tolist() == ["Empty risk", "Balanced", "Full risk"]
    assert result["bike_deficit"].tolist() == [9, 0, 0]
    assert result["bike_surplus"].tolist() == [0, 0, 9]


def test_network_metrics_reconcile_bike_totals() -> None:
    result = add_service_features(build_demo_data())
    metrics = network_metrics(result)
    operational = result[result["operational"]]
    assert metrics["bikes"] == operational["bikes"].sum()
    assert 0 <= metrics["balanced_share"] <= 100


def test_pressure_clusters_only_label_dense_same_type_groups() -> None:
    frame = build_demo_data().head(6).copy()
    frame["latitude"] = [51.5000, 51.5002, 51.5004, 51.5300, 51.5302, 51.5304]
    frame["longitude"] = [-0.1000, -0.1002, -0.1004, -0.1500, -0.1502, -0.1504]
    frame["docks"] = 20
    frame["unavailable_docks"] = 0
    frame["bikes"] = [1, 1, 1, 19, 19, 19]
    frame["empty_docks"] = 20 - frame["bikes"]
    featured = add_service_features(frame, critical_threshold=0.1)
    clustered, summary = pressure_clusters(featured, radius_km=0.2, min_stations=3)
    assert clustered["cluster_id"].ge(0).sum() == 6
    assert len(summary) == 2
    assert set(summary["pressure_type"]) == {"Empty risk", "Full risk"}


def test_rebalancing_plan_conserves_bikes_and_respects_van_capacity() -> None:
    frame = build_demo_data().head(4).copy()
    frame["latitude"] = [51.5000, 51.5001, 51.5002, 51.5003]
    frame["longitude"] = [-0.1000, -0.1001, -0.1002, -0.1003]
    frame["docks"] = 20
    frame["unavailable_docks"] = 0
    frame["bikes"] = [20, 16, 0, 4]
    frame["empty_docks"] = 20 - frame["bikes"]
    featured = add_service_features(frame, target_fill=0.5)
    plan, simulated = build_rebalancing_plan(featured, van_capacity=4, max_moves=10, max_distance_km=3)
    assert not plan.empty
    assert plan["bikes_to_move"].le(4).all()
    assert simulated["simulated_bikes"].sum() == featured["bikes"].sum()
    assert simulated["simulated_bikes"].ge(0).all()


def test_scenario_never_creates_more_critical_stations_in_balanced_transfer_case() -> None:
    frame = build_demo_data().head(4).copy()
    frame["latitude"] = [51.5, 51.5001, 51.5002, 51.5003]
    frame["longitude"] = [-0.1, -0.1001, -0.1002, -0.1003]
    frame["docks"] = 20
    frame["unavailable_docks"] = 0
    frame["bikes"] = [20, 16, 0, 4]
    frame["empty_docks"] = 20 - frame["bikes"]
    featured = add_service_features(frame, critical_threshold=0.15, target_fill=0.5)
    _, simulated = build_rebalancing_plan(featured, van_capacity=10, max_moves=10, max_distance_km=3)
    summary = scenario_summary(featured, simulated, 0.15)
    assert summary["after_empty"] + summary["after_full"] <= summary["before_empty"] + summary["before_full"]


def test_quality_report_contains_all_documented_checks() -> None:
    report = quality_report(add_service_features(build_demo_data()))
    assert len(report) == 5
    assert report["share"].between(0, 100).all()
