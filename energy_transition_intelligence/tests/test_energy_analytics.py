"""Unit tests for the transparent analytical core."""

import pandas as pd

from energy_transition_intelligence.src.analytics import (
    latest_snapshot,
    scenario_score,
    score_countries,
)
from energy_transition_intelligence.src.data import build_demo_data


def test_latest_snapshot_uses_most_recent_non_null_value():
    data = pd.DataFrame(
        [
            {
                "country": "A",
                "country_code": "AAA",
                "year": 2020,
                "indicator_code": "EG.ELC.RNEW.ZS",
                "value": 10.0,
            },
            {
                "country": "A",
                "country_code": "AAA",
                "year": 2022,
                "indicator_code": "EG.ELC.RNEW.ZS",
                "value": 20.0,
            },
        ]
    )
    result = latest_snapshot(data)
    assert result.loc[0, "EG.ELC.RNEW.ZS"] == 20.0
    assert result.loc[0, "EG.ELC.RNEW.ZS_year"] == 2022


def test_scores_are_bounded_and_reward_cleaner_profile():
    snapshot = latest_snapshot(build_demo_data())
    result = score_countries(snapshot)
    assert result["transition_score"].between(0, 100).all()
    assert result.iloc[0]["transition_score"] > result.iloc[-1]["transition_score"]
    assert result["rank"].min() == 1


def test_positive_scenario_cannot_reduce_score():
    scored = score_countries(latest_snapshot(build_demo_data()))
    row = scored.iloc[len(scored) // 2]
    baseline, simulated, changed = scenario_score(
        row,
        scored,
        renewable_change_pp=10,
        co2_reduction_pct=15,
        intensity_reduction_pct=10,
    )
    assert simulated >= baseline
    assert changed["EG.ELC.RNEW.ZS"] > row["EG.ELC.RNEW.ZS"]
    assert changed["EN.ATM.CO2E.PC"] < row["EN.ATM.CO2E.PC"]
