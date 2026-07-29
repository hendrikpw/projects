import pandas as pd
import pytest

from orbital_launch_intelligence.src.analytics import (
    provider_reliability,
    simulate_provider_record,
    summary_metrics,
    wilson_interval,
)
from orbital_launch_intelligence.src.data import build_demo_data, parse_launches


def test_parser_flattens_nested_launch_fields():
    data = parse_launches(
        [
            {
                "id": "x",
                "name": "Test",
                "net": "2026-01-01T00:00:00Z",
                "status": {"name": "Launch Successful"},
                "launch_service_provider": {"name": "Agency", "type": {"name": "Commercial"}},
                "rocket": {"configuration": {"full_name": "Rocket X", "families": [{"name": "Family X"}]}},
                "mission": {"type": "Science", "orbit": {"name": "Low Earth Orbit", "abbrev": "LEO"}},
                "pad": {"name": "Pad", "location": {"name": "Site"}, "country": {"name": "Country"}, "latitude": 1, "longitude": 2},
            }
        ],
        False,
        False,
    )
    assert data.iloc[0]["provider"] == "Agency"
    assert data.iloc[0]["rocket_family"] == "Family X"
    assert data.iloc[0]["outcome"] == "Success"


def test_wilson_interval_is_bounded_and_penalises_small_samples():
    low_small, high_small = wilson_interval(5, 5)
    low_large, high_large = wilson_interval(50, 50)
    assert 0 <= low_small < high_small <= 1
    assert low_large > low_small


def test_provider_reliability_reconciles_attempts():
    frame = build_demo_data()
    result = provider_reliability(frame[~frame["is_upcoming"]], minimum_attempts=5)
    assert (result["successes"] + result["failures"] == result["attempts"]).all()
    assert result["wilson_low"].between(0, 100).all()


def test_summary_effective_provider_count_is_valid():
    frame = build_demo_data()
    history = frame[~frame["is_upcoming"]]
    metrics = summary_metrics(history)
    assert 1 <= metrics["effective_providers"] <= metrics["providers"]
    assert 0 <= metrics["success_rate"] <= 100


def test_scenario_recalculates_record():
    result = simulate_provider_record(9, 10, 2, 1)
    assert result["attempts"] == 13
    assert result["successes"] == 11
    assert result["success_rate"] == pytest.approx(11 / 13 * 100)


def test_demo_data_is_deterministic_and_has_both_modes():
    first = build_demo_data()
    second = build_demo_data()
    pd.testing.assert_frame_equal(first, second)
    assert set(first["is_upcoming"]) == {False, True}
    assert first["is_demo"].all()
