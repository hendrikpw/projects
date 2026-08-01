import numpy as np
import pandas as pd
import pytest

from fx_regime_intelligence.src.analytics import (
    correlation_and_clusters,
    detect_anomalies,
    inverse_volatility_allocation,
    market_regimes,
    normalized_rates,
    rate_matrix,
    risk_summary,
    shock_scenario,
)
from fx_regime_intelligence.src.data import build_demo_data, parse_ecb_csv


def test_parser_keeps_documented_daily_reference_rate():
    content = (
        "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-07-31,1.17,A\n"
        "EXR.M.USD.EUR.SP00.A,M,USD,EUR,SP00,A,2026-07-31,1.16,A\n"
    )
    frame = parse_ecb_csv(content)
    assert len(frame) == 1
    assert frame.iloc[0]["currency"] == "USD"
    assert frame.iloc[0]["rate_per_eur"] == 1.17


def test_normalization_starts_at_100():
    rates = rate_matrix(build_demo_data(), ["USD", "GBP"])
    normalized = normalized_rates(rates)
    assert np.allclose(normalized.iloc[0], 100)


def test_risk_summary_has_bounded_and_reconcilable_metrics():
    rates = rate_matrix(build_demo_data(), ["USD", "GBP", "JPY"])
    summary = risk_summary(rates)
    assert set(summary["currency"]) == {"USD", "GBP", "JPY"}
    assert summary["annualized_volatility"].gt(0).all()
    assert summary["max_drawdown"].le(0).all()


def test_regimes_use_expected_labels():
    rates = rate_matrix(build_demo_data(), ["USD", "GBP", "CHF", "JPY"])
    regimes = market_regimes(rates)
    assert set(regimes["regime"]).issubset({"Calm", "Normal", "Stress"})
    assert {"Calm", "Stress"}.issubset(set(regimes["regime"]))


def test_isolation_forest_flags_requested_small_share():
    rates = rate_matrix(build_demo_data(), ["USD", "GBP", "CHF", "JPY"])
    anomalies = detect_anomalies(rates, contamination=0.03)
    assert not anomalies.empty
    assert 0.02 <= anomalies["is_anomaly"].mean() <= 0.04
    assert anomalies["anomaly_score"].between(0, 100).all()


def test_clusters_cover_each_currency_once():
    rates = rate_matrix(build_demo_data(), ["USD", "GBP", "CHF", "JPY"])
    correlation, groups = correlation_and_clusters(rates, clusters=3)
    assert correlation.shape == (4, 4)
    assert groups["currency"].nunique() == 4
    assert groups["cluster"].nunique() == 3


def test_inverse_volatility_weights_sum_to_100():
    rates = rate_matrix(build_demo_data(), ["USD", "GBP", "CHF", "JPY"])
    allocation = inverse_volatility_allocation(rates)
    assert allocation["weight"].sum() == pytest.approx(100)
    assert allocation["weight"].gt(0).all()


def test_shock_scenario_reconciles_values():
    scenario = shock_scenario(1.2, 10_000, 5)
    assert scenario["current_foreign_value"] == 12_000
    assert scenario["shocked_foreign_value"] == 12_600
    assert scenario["foreign_value_change"] == 600


def test_demo_data_is_deterministic():
    pd.testing.assert_frame_equal(build_demo_data(), build_demo_data())
