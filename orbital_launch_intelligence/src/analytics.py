"""Reliability, cadence and market-structure analytics for launch data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def filter_history(
    data: pd.DataFrame,
    months: int,
    providers: list[str] | None = None,
    include_suborbital: bool = True,
) -> pd.DataFrame:
    """Filter recent non-upcoming launches without mutating the source frame."""
    history = data[~data["is_upcoming"]].copy()
    if history.empty:
        return history
    latest = history["net"].max()
    history = history[history["net"] >= latest - pd.DateOffset(months=int(months))]
    if providers:
        history = history[history["provider"].isin(providers)]
    if not include_suborbital:
        history = history[~history["orbit"].str.contains("Suborbital", case=False, na=False)]
    return history


def wilson_interval(successes: int, attempts: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial success proportion."""
    if attempts <= 0:
        return (np.nan, np.nan)
    proportion = successes / attempts
    denominator = 1 + z**2 / attempts
    center = (proportion + z**2 / (2 * attempts)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / attempts + z**2 / (4 * attempts**2)
        )
        / denominator
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def provider_reliability(
    history: pd.DataFrame,
    minimum_attempts: int = 5,
) -> pd.DataFrame:
    """Rank providers using an uncertainty-aware Wilson lower bound."""
    decided = history[history["is_decided"]]
    if decided.empty:
        return pd.DataFrame()
    grouped = (
        decided.groupby(["provider", "provider_type"], as_index=False)
        .agg(
            attempts=("launch_id", "nunique"),
            successes=("is_success", "sum"),
            latest_launch=("net", "max"),
            rocket_families=("rocket_family", "nunique"),
        )
    )
    grouped = grouped[grouped["attempts"] >= int(minimum_attempts)].copy()
    if grouped.empty:
        return grouped
    intervals = [
        wilson_interval(int(row.successes), int(row.attempts))
        for row in grouped.itertuples()
    ]
    grouped["success_rate"] = grouped["successes"] / grouped["attempts"] * 100
    grouped["wilson_low"] = [value[0] * 100 for value in intervals]
    grouped["wilson_high"] = [value[1] * 100 for value in intervals]
    grouped["failures"] = grouped["attempts"] - grouped["successes"]
    return grouped.sort_values(
        ["wilson_low", "attempts"], ascending=False
    ).reset_index(drop=True)


def summary_metrics(history: pd.DataFrame) -> dict:
    """Calculate auditable portfolio-level launch metrics."""
    decided = history[history["is_decided"]]
    if history.empty:
        return {
            "launches": 0,
            "decided": 0,
            "success_rate": np.nan,
            "providers": 0,
            "monthly_cadence": 0.0,
            "effective_providers": np.nan,
        }
    span_days = max((history["net"].max() - history["net"].min()).days, 1)
    counts = history.groupby("provider")["launch_id"].nunique()
    shares = counts / counts.sum()
    hhi = float((shares**2).sum())
    return {
        "launches": int(history["launch_id"].nunique()),
        "decided": int(decided["launch_id"].nunique()),
        "success_rate": float(decided["is_success"].mean() * 100)
        if not decided.empty
        else np.nan,
        "providers": int(history["provider"].nunique()),
        "monthly_cadence": float(len(history) / span_days * 30.437),
        "effective_providers": float(1 / hhi) if hhi > 0 else np.nan,
        "hhi": hhi,
    }


def monthly_cadence(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate launches and decided success rate by calendar month."""
    if history.empty:
        return pd.DataFrame()
    launches = history.groupby("month", as_index=False).agg(
        launches=("launch_id", "nunique"),
        providers=("provider", "nunique"),
    )
    decided = (
        history[history["is_decided"]]
        .groupby("month", as_index=False)
        .agg(decided=("launch_id", "nunique"), success_rate=("is_success", "mean"))
    )
    result = launches.merge(decided, on="month", how="left")
    result["success_rate"] = result["success_rate"] * 100
    return result.sort_values("month")


def pad_activity(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate geocoded launch activity for a browser-safe map."""
    located = history.dropna(subset=["latitude", "longitude"]).copy()
    if located.empty:
        return pd.DataFrame()
    return (
        located.groupby(
            ["pad", "location", "country", "latitude", "longitude"], as_index=False
        )
        .agg(
            launches=("launch_id", "nunique"),
            providers=("provider", "nunique"),
            latest_launch=("net", "max"),
        )
        .sort_values("launches", ascending=False)
    )


def orbit_mix(history: pd.DataFrame) -> pd.DataFrame:
    """Build orbit and mission-type counts for composition analysis."""
    return (
        history.groupby(["orbit", "mission_type"], as_index=False)
        .agg(launches=("launch_id", "nunique"))
        .sort_values("launches", ascending=False)
    )


def simulate_provider_record(
    successes: int,
    attempts: int,
    additional_successes: int,
    additional_failures: int,
) -> dict:
    """Recalculate reliability after a transparent hypothetical record."""
    new_successes = int(successes) + int(additional_successes)
    new_attempts = int(attempts) + int(additional_successes) + int(additional_failures)
    low, high = wilson_interval(new_successes, new_attempts)
    return {
        "successes": new_successes,
        "attempts": new_attempts,
        "success_rate": new_successes / new_attempts * 100 if new_attempts else np.nan,
        "wilson_low": low * 100,
        "wilson_high": high * 100,
    }
