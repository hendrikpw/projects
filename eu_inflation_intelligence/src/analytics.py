"""Transparent household-basket and comparative inflation analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def latest_observations(data: pd.DataFrame) -> pd.DataFrame:
    """Return each series' latest available value and previous-month change."""
    if data.empty:
        return data.copy()
    ordered = data.sort_values("date").copy()
    ordered["previous_rate"] = ordered.groupby(["geo", "coicop18"])["rate"].shift(1)
    ordered["monthly_acceleration"] = ordered["rate"] - ordered["previous_rate"]
    latest = (
        ordered.groupby(["geo", "coicop18"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def country_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise latest all-items rates, momentum and 12-month volatility."""
    all_items = data[data["coicop18"].eq("TOTAL")].sort_values("date").copy()
    if all_items.empty:
        return pd.DataFrame()
    all_items["previous_rate"] = all_items.groupby("geo")["rate"].shift(1)
    all_items["acceleration"] = all_items["rate"] - all_items["previous_rate"]
    all_items["twelve_month_average"] = all_items.groupby("geo")["rate"].transform(
        lambda values: values.rolling(12, min_periods=3).mean()
    )
    all_items["twelve_month_volatility"] = all_items.groupby("geo")["rate"].transform(
        lambda values: values.rolling(12, min_periods=3).std()
    )
    latest = all_items.groupby("geo", as_index=False).tail(1).copy()
    median = float(latest["rate"].median())
    mad = float((latest["rate"] - median).abs().median())
    latest["robust_distance"] = (
        (latest["rate"] - median) / (1.4826 * mad)
        if mad > 0
        else 0.0
    )
    return latest.sort_values("rate", ascending=False).reset_index(drop=True)


def category_snapshot(data: pd.DataFrame, geo: str) -> pd.DataFrame:
    """Return the selected economy's latest non-total category rates."""
    latest = latest_observations(data[data["geo"].eq(geo)])
    return (
        latest[~latest["coicop18"].eq("TOTAL")]
        .sort_values("rate", ascending=False)
        .reset_index(drop=True)
    )


def personal_basket(
    categories: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, dict]:
    """Estimate a user-weighted annual rate and percentage-point contributions."""
    selected = categories[categories["coicop18"].isin(weights)].copy()
    selected["raw_weight"] = selected["coicop18"].map(weights).astype(float)
    selected = selected[selected["raw_weight"] > 0]
    available_weight = float(selected["raw_weight"].sum())
    requested_weight = float(sum(max(float(value), 0.0) for value in weights.values()))
    if selected.empty or available_weight <= 0:
        return pd.DataFrame(), {
            "rate": np.nan,
            "available_weight": 0.0,
            "requested_weight": requested_weight,
            "coverage": 0.0,
        }

    selected["normalized_weight"] = selected["raw_weight"] / available_weight * 100
    selected["contribution"] = (
        selected["rate"] * selected["normalized_weight"] / 100
    )
    selected = selected.sort_values("contribution", ascending=False)
    return selected, {
        "rate": float(selected["contribution"].sum()),
        "available_weight": available_weight,
        "requested_weight": requested_weight,
        "coverage": available_weight / requested_weight * 100
        if requested_weight > 0
        else 0.0,
    }


def inflation_breadth(categories: pd.DataFrame, threshold: float = 2.0) -> float:
    """Measure the share of available categories above a chosen annual rate."""
    rates = pd.to_numeric(categories.get("rate"), errors="coerce").dropna()
    return float((rates > threshold).mean() * 100) if len(rates) else np.nan


def spending_pressure(monthly_spend: float, annual_rate: float) -> dict:
    """Translate an annual rate into a simple like-for-like budget illustration."""
    if not np.isfinite(annual_rate):
        return {"monthly": np.nan, "annual": np.nan}
    monthly = float(monthly_spend) * annual_rate / 100
    return {"monthly": monthly, "annual": monthly * 12}


def inflation_regime(current_rate: float, acceleration: float) -> str:
    """Create a descriptive—not predictive—price-pressure label."""
    if not np.isfinite(current_rate) or not np.isfinite(acceleration):
        return "Insufficient data"
    direction = "accelerating" if acceleration > 0.3 else "cooling" if acceleration < -0.3 else "steady"
    level = "elevated" if current_rate > 3 else "moderate" if current_rate >= 1 else "low"
    return f"{level.capitalize()} · {direction}"
