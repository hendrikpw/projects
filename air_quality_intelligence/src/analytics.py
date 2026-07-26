"""Reusable analytical transformations for air-quality forecasts."""

from __future__ import annotations

import pandas as pd


POLLUTANTS = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO₂", "ozone": "O₃"}


def aqi_band(value: float) -> str:
    """Map European AQI to the official Open-Meteo/EEA display bands."""
    if pd.isna(value):
        return "Unknown"
    if value <= 20:
        return "Good"
    if value <= 40:
        return "Fair"
    if value <= 60:
        return "Moderate"
    if value <= 80:
        return "Poor"
    if value <= 100:
        return "Very poor"
    return "Extremely poor"


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add AQI category, day, hour and dominant-pollutant diagnostics."""
    required = {"time", "city", "european_aqi", *POLLUTANTS.keys()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"])
    result["date"] = result["time"].dt.date
    result["hour"] = result["time"].dt.hour
    result["aqi_band"] = result["european_aqi"].map(aqi_band)
    normalized = pd.DataFrame(index=result.index)
    normalized["PM2.5"] = result["pm2_5"] / 25.0
    normalized["PM10"] = result["pm10"] / 50.0
    normalized["NO₂"] = result["nitrogen_dioxide"] / 120.0
    normalized["O₃"] = result["ozone"] / 130.0
    result["dominant_pollutant"] = "Unknown"
    has_pollutant_value = normalized.notna().any(axis=1)
    result.loc[has_pollutant_value, "dominant_pollutant"] = normalized.loc[
        has_pollutant_value
    ].idxmax(axis=1)
    return result


def city_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a city-level comparison for the current filtered horizon."""
    return (
        frame.groupby("city", as_index=False)
        .agg(average_aqi=("european_aqi", "mean"), peak_aqi=("european_aqi", "max"),
             poor_hours=("european_aqi", lambda values: int((values > 60).sum())),
             pm25_average=("pm2_5", "mean"))
        .sort_values("average_aqi")
    )


def daily_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly forecasts to city-day mean and maximum AQI."""
    return frame.groupby(["city", "date"], as_index=False).agg(
        mean_aqi=("european_aqi", "mean"), peak_aqi=("european_aqi", "max"))


def pollutant_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """Count which normalized pollutant concentration dominates each hour."""
    return frame["dominant_pollutant"].value_counts().rename_axis("pollutant").reset_index(name="hours")
