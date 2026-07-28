import pandas as pd
import pytest

from eu_inflation_intelligence.src.analytics import (
    country_summary,
    inflation_breadth,
    latest_observations,
    personal_basket,
    spending_pressure,
)
from eu_inflation_intelligence.src.data import build_demo_data, parse_jsonstat


def test_jsonstat_parser_respects_dimension_order():
    payload = {
        "id": ["geo", "time"],
        "size": [2, 2],
        "dimension": {
            "geo": {"category": {"index": {"DE": 0, "FR": 1}}},
            "time": {"category": {"index": {"2026-01": 0, "2026-02": 1}}},
        },
        "value": {"0": 2.0, "1": 2.2, "2": 1.1, "3": 1.3},
    }
    result = parse_jsonstat(payload)
    assert result.loc[3, ["geo", "time", "rate"]].tolist() == ["FR", "2026-02", 1.3]


def test_latest_observations_calculates_acceleration():
    data = pd.DataFrame(
        {
            "geo": ["DE", "DE"],
            "coicop18": ["TOTAL", "TOTAL"],
            "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "rate": [2.0, 2.4],
        }
    )
    latest = latest_observations(data)
    assert latest.iloc[0]["rate"] == 2.4
    assert latest.iloc[0]["monthly_acceleration"] == pytest.approx(0.4)


def test_personal_basket_normalises_and_reconciles_contributions():
    categories = pd.DataFrame(
        {
            "coicop18": ["CP01", "CP04"],
            "category": ["Food", "Housing"],
            "rate": [3.0, 1.0],
        }
    )
    detail, summary = personal_basket(categories, {"CP01": 25, "CP04": 75})
    assert summary["rate"] == pytest.approx(1.5)
    assert detail["normalized_weight"].sum() == pytest.approx(100)
    assert detail["contribution"].sum() == pytest.approx(summary["rate"])


def test_breadth_and_spending_pressure():
    categories = pd.DataFrame({"rate": [1.0, 2.5, 3.0, -0.5]})
    assert inflation_breadth(categories) == 50
    assert spending_pressure(2_000, 2.5) == {"monthly": 50.0, "annual": 600.0}


def test_country_summary_and_demo_are_complete_and_deterministic():
    first = build_demo_data()
    second = build_demo_data()
    pd.testing.assert_frame_equal(first, second)
    summary = country_summary(first)
    assert len(summary) == first["geo"].nunique()
    assert summary["rate"].notna().all()
