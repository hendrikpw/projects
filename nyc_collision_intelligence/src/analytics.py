"""Spatial, temporal and severity analytics for collision records."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def filter_collisions(
    data: pd.DataFrame,
    days: int,
    boroughs: list[str],
    outcome: str,
) -> pd.DataFrame:
    """Apply the user-selected time, geography and outcome filters."""
    if data.empty:
        return data.copy()
    latest = data["crash_date"].max()
    filtered = data[
        data["crash_date"] >= latest - pd.to_timedelta(int(days) - 1, unit="D")
    ].copy()
    if boroughs:
        filtered = filtered[filtered["borough"].isin(boroughs)]
    if outcome == "Injury collisions":
        filtered = filtered[filtered["injury_collision"]]
    elif outcome == "Fatal collisions":
        filtered = filtered[filtered["fatal_collision"]]
    elif outcome == "Pedestrian / cyclist casualties":
        filtered = filtered[(filtered["vulnerable_injured"] + filtered["vulnerable_killed"]) > 0]
    return filtered


def summary_metrics(data: pd.DataFrame) -> dict:
    """Calculate portfolio-ready headline safety metrics."""
    crashes = len(data)
    injuries = int(data["number_of_persons_injured"].sum()) if crashes else 0
    fatalities = int(data["number_of_persons_killed"].sum()) if crashes else 0
    vulnerable = int((data["vulnerable_injured"] + data["vulnerable_killed"]).sum()) if crashes else 0
    injury_rate = injuries / crashes * 100 if crashes else 0.0
    return {
        "crashes": crashes,
        "injuries": injuries,
        "fatalities": fatalities,
        "vulnerable_casualties": vulnerable,
        "injuries_per_100_crashes": injury_rate,
    }


def daily_anomalies(data: pd.DataFrame, baseline_days: int = 14) -> pd.DataFrame:
    """Detect unusual daily crash counts with a robust rolling MAD score."""
    if data.empty:
        return pd.DataFrame(
            columns=["crash_date", "crashes", "baseline", "robust_z", "is_anomaly"]
        )
    daily = (
        data.set_index("crash_date")
        .resample("D")
        .size()
        .rename("crashes")
        .to_frame()
    )
    daily["baseline"] = daily["crashes"].rolling(
        baseline_days, min_periods=max(4, baseline_days // 3)
    ).median()
    absolute_deviation = (daily["crashes"] - daily["baseline"]).abs()
    mad = absolute_deviation.rolling(
        baseline_days, min_periods=max(4, baseline_days // 3)
    ).median()
    denominator = (1.4826 * mad).replace(0, np.nan)
    daily["robust_z"] = ((daily["crashes"] - daily["baseline"]) / denominator).fillna(0)
    daily["is_anomaly"] = daily["robust_z"].abs() >= 3.0
    return daily.reset_index()


def spatial_hotspots(
    data: pd.DataFrame,
    minimum_crashes: int = 3,
    precision: int = 2,
) -> pd.DataFrame:
    """Aggregate geocoded crashes into explainable coordinate grid cells."""
    geocoded = data[data["valid_coordinate"]].copy()
    if geocoded.empty:
        return pd.DataFrame()
    geocoded["latitude_grid"] = geocoded["latitude"].round(precision)
    geocoded["longitude_grid"] = geocoded["longitude"].round(precision)
    grouped = (
        geocoded.groupby(["latitude_grid", "longitude_grid"], as_index=False)
        .agg(
            crashes=("collision_id", "nunique"),
            injuries=("number_of_persons_injured", "sum"),
            fatalities=("number_of_persons_killed", "sum"),
            vulnerable_casualties=("vulnerable_injured", "sum"),
            borough=("borough", lambda values: values.mode().iloc[0]),
            street=("street", lambda values: values.mode().iloc[0]),
        )
    )
    grouped = grouped[grouped["crashes"] >= minimum_crashes].copy()
    if grouped.empty:
        return grouped
    grouped["risk_points"] = (
        grouped["crashes"]
        + 2 * grouped["injuries"]
        + 25 * grouped["fatalities"]
        + 4 * grouped["vulnerable_casualties"]
    )
    spread = grouped["risk_points"].max() - grouped["risk_points"].min()
    grouped["risk_index"] = (
        50.0
        if spread == 0
        else 100
        * (grouped["risk_points"] - grouped["risk_points"].min())
        / spread
    )
    return grouped.sort_values(["risk_index", "crashes"], ascending=False).reset_index(drop=True)


def hourly_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Return a complete day-type by hour matrix."""
    if data.empty:
        return pd.DataFrame()
    profile = data.groupby(["day_type", "hour"]).size().rename("crashes").reset_index()
    complete = pd.MultiIndex.from_product(
        [["Weekday", "Weekend"], range(24)], names=["day_type", "hour"]
    ).to_frame(index=False)
    return complete.merge(profile, on=["day_type", "hour"], how="left").fillna({"crashes": 0})


def factor_profile(data: pd.DataFrame, minimum_cases: int = 8) -> pd.DataFrame:
    """Compare reported primary factors by volume and observed severity."""
    valid = data[
        ~data["primary_factor"].str.casefold().isin(
            {"unspecified", "unknown", "other vehicular"}
        )
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = (
        valid.groupby("primary_factor", as_index=False)
        .agg(
            crashes=("collision_id", "nunique"),
            injuries=("number_of_persons_injured", "sum"),
            fatalities=("number_of_persons_killed", "sum"),
            serious_collisions=("serious_outcome", "sum"),
        )
    )
    grouped = grouped[grouped["crashes"] >= minimum_cases].copy()
    grouped["serious_rate"] = grouped["serious_collisions"] / grouped["crashes"] * 100
    grouped["injuries_per_100"] = grouped["injuries"] / grouped["crashes"] * 100
    return grouped.sort_values(["serious_rate", "crashes"], ascending=False).reset_index(drop=True)


def borough_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize volume and outcome severity for each reported borough."""
    if data.empty:
        return pd.DataFrame()
    grouped = (
        data[data["borough"] != "UNKNOWN"]
        .groupby("borough", as_index=False)
        .agg(
            crashes=("collision_id", "nunique"),
            injuries=("number_of_persons_injured", "sum"),
            fatalities=("number_of_persons_killed", "sum"),
        )
    )
    grouped["injuries_per_100"] = grouped["injuries"] / grouped["crashes"] * 100
    return grouped.sort_values("crashes", ascending=False)
