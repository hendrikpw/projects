import pandas as pd
import pytest

from food_label_intelligence.src.analytics import (
    brand_summary,
    choice_fit_score,
    filter_products,
    missingness_report,
    similar_products,
)
from food_label_intelligence.src.data import build_demo_data, parse_products


def test_parser_reads_nested_nutriments():
    frame = parse_products(
        [{"code": "1", "product_name": "Test", "brands": "Brand", "nutriments": {"sugars_100g": 8, "proteins_100g": 4}}],
        "Yogurts",
        "Germany",
        False,
    )
    assert frame.iloc[0]["sugars_100g"] == 8
    assert frame.iloc[0]["proteins_100g"] == 4


def test_choice_fit_rewards_lower_sugar_when_it_is_only_priority():
    data = pd.DataFrame({"sugars_100g": [2, 10], "salt_100g": [1, 1], "saturated_fat_100g": [1, 1], "fiber_100g": [1, 1], "proteins_100g": [1, 1]})
    scored = choice_fit_score(data, {"Lower sugar": 100})
    assert scored.loc[0, "choice_fit_score"] > scored.loc[1, "choice_fit_score"]


def test_filter_and_similarity_exclude_selected_product():
    data = choice_fit_score(build_demo_data(), {"Lower sugar": 50, "Higher fibre": 50})
    filtered = filter_products(data, minimum_coverage=50)
    code = filtered.iloc[0]["code"]
    similar = similar_products(filtered, code, limit=5)
    assert len(similar) == 5
    assert code not in set(similar["code"])
    assert similar["nutrition_distance"].is_monotonic_increasing


def test_brand_summary_obeys_minimum_sample():
    data = choice_fit_score(build_demo_data(), {"Lower salt": 100})
    summary = brand_summary(data, minimum_products=2)
    assert (summary["products"] >= 2).all()
    assert summary["median_fit"].notna().all()


def test_missingness_report_is_bounded():
    report = missingness_report(build_demo_data())
    assert report["coverage"].between(0, 100).all()
    assert "Sugars" in set(report["field"])


def test_demo_data_is_deterministic_and_labelled():
    first = build_demo_data("Yogurts", "Germany")
    second = build_demo_data("Yogurts", "Germany")
    pd.testing.assert_frame_equal(first, second)
    assert first["is_demo"].all()
    assert first["product_name"].str.startswith("Synthetic").all()
