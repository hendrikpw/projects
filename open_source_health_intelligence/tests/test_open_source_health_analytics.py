"""Tests for repository ingestion guards and analytical invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from open_source_health_intelligence.src.analytics import (
    capacity_scenario,
    contributor_concentration,
    issue_age_bands,
    kaplan_meier,
    km_percentile,
    language_mix,
    repository_pulse,
)
from open_source_health_intelligence.src.data import build_demo_data, normalize_repository


def test_repository_normalization_accepts_url_and_git_suffix() -> None:
    assert normalize_repository("https://github.com/pandas-dev/pandas.git") == "pandas-dev/pandas"
    assert normalize_repository(" streamlit/streamlit/ ") == "streamlit/streamlit"


@pytest.mark.parametrize("value", ["pandas", "a/b/c", "https://example.com/a/b", "a b/repo"])
def test_repository_normalization_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_repository(value)


def test_demo_data_has_complete_bounded_schema() -> None:
    frames, metadata = build_demo_data()
    assert metadata["mode"] == "demo"
    assert set(frames) == {"issues", "pulls", "contributors", "releases", "commits", "languages"}
    assert len(frames["issues"]) == 100
    assert len(frames["pulls"]) == 100
    assert frames["issues"]["duration_days"].ge(0).all()


def test_kaplan_meier_accounts_for_censoring() -> None:
    records = pd.DataFrame({"duration_days": [2, 4, 4, 10], "event_observed": [1, 1, 0, 0]})
    curve = kaplan_meier(records)
    assert curve.iloc[0]["unresolved_share"] == 100
    assert curve["unresolved_share"].is_monotonic_decreasing
    assert curve.iloc[-1]["unresolved_share"] == pytest.approx(50)
    assert km_percentile(curve) == pytest.approx(4)


def test_contributor_hhi_and_effective_count() -> None:
    contributors = pd.DataFrame({"contributions": [50, 30, 20]})
    result = contributor_concentration(contributors)
    assert result["top1_share"] == pytest.approx(50)
    assert result["hhi"] == pytest.approx(3800)
    assert result["effective_contributors"] == pytest.approx(1 / 0.38)


def test_open_issue_age_bands_preserve_count() -> None:
    issues = pd.DataFrame({"duration_days": [10, 45, 120, 220, 600, 3], "event_observed": [0, 0, 0, 0, 0, 1]})
    result = issue_age_bands(issues)
    assert result["count"].sum() == 5
    assert result["share"].sum() == pytest.approx(100)


def test_language_mix_sums_to_one_hundred() -> None:
    result = language_mix(pd.DataFrame({"language": ["Python", "CSS"], "bytes": [900, 100]}))
    assert result["share"].sum() == pytest.approx(100)
    assert result.iloc[0]["share"] == pytest.approx(90)


def test_capacity_scenario_detects_unstable_flow() -> None:
    pulls = pd.DataFrame({"event_observed": [0, 0, 1]})
    unstable = capacity_scenario(pulls, weekly_capacity=2, weekly_arrivals=2.5)
    stable = capacity_scenario(pulls, weekly_capacity=4, weekly_arrivals=2)
    assert unstable["clearance_weeks"] is None
    assert stable["clearance_weeks"] == pytest.approx(1)


def test_pulse_is_bounded_and_components_sum_to_score() -> None:
    frames, metadata = build_demo_data()
    score, components = repository_pulse(frames, metadata)
    assert 0 <= score <= 100
    assert len(components) == 5
    assert components["weight"].sum() == 100
    assert components["weighted_points"].sum() == pytest.approx(score)
