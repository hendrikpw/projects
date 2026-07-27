"""Seismological feature engineering and spatial-temporal analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6_371.0088


def filter_events(
    data: pd.DataFrame,
    days: int,
    minimum_magnitude: float,
    maximum_depth: int,
    reviewed_only: bool,
    tsunami_only: bool,
) -> pd.DataFrame:
    """Apply user controls without mutating the cached source frame."""
    if data.empty:
        return data.copy()
    latest = data["time"].max()
    result = data[
        (data["time"] >= latest - pd.to_timedelta(int(days), unit="D"))
        & (data["magnitude"] >= minimum_magnitude)
        & (data["depth_km"] <= maximum_depth)
    ].copy()
    if reviewed_only:
        result = result[result["reviewed"]]
    if tsunami_only:
        result = result[result["tsunami_flag"]]
    return result


def gutenberg_richter_b_value(
    magnitudes: pd.Series,
    completeness_magnitude: float,
    bin_width: float = 0.1,
) -> float | None:
    """Estimate the Gutenberg-Richter b-value with the Aki MLE."""
    values = pd.to_numeric(magnitudes, errors="coerce").dropna()
    values = values[values >= completeness_magnitude]
    if len(values) < 20:
        return None
    denominator = values.mean() - (completeness_magnitude - bin_width / 2)
    if denominator <= 0:
        return None
    return float(np.log10(np.e) / denominator)


def summary_metrics(data: pd.DataFrame, completeness_magnitude: float) -> dict:
    """Calculate headline activity metrics with transparent units."""
    if data.empty:
        return {
            "events": 0,
            "maximum_magnitude": np.nan,
            "total_energy_joules": 0.0,
            "tsunami_flags": 0,
            "reviewed_share": 0.0,
            "b_value": None,
        }
    return {
        "events": len(data),
        "maximum_magnitude": float(data["magnitude"].max()),
        "total_energy_joules": float(data["energy_joules"].sum()),
        "tsunami_flags": int(data["tsunami_flag"].sum()),
        "reviewed_share": float(data["reviewed"].mean() * 100),
        "b_value": gutenberg_richter_b_value(
            data["magnitude"], completeness_magnitude
        ),
    }


def cluster_events(
    data: pd.DataFrame,
    radius_km: int = 250,
    minimum_events: int = 4,
) -> pd.DataFrame:
    """Group nearby epicenters with haversine DBSCAN and label noise."""
    if data.empty:
        result = data.copy()
        result["cluster"] = pd.Series(dtype=str)
        return result

    from sklearn.cluster import DBSCAN

    coordinates = np.radians(data[["latitude", "longitude"]].to_numpy())
    raw_labels = DBSCAN(
        eps=float(radius_km) / EARTH_RADIUS_KM,
        min_samples=int(minimum_events),
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coordinates)

    counts = pd.Series(raw_labels[raw_labels >= 0]).value_counts()
    ordered_labels = {label: rank + 1 for rank, label in enumerate(counts.index)}
    names = [
        "Unclustered" if label < 0 else f"Sequence {ordered_labels[label]:02d}"
        for label in raw_labels
    ]
    result = data.copy()
    result["cluster"] = names
    return result


def cluster_summary(clustered: pd.DataFrame) -> pd.DataFrame:
    """Create an auditable table for non-noise DBSCAN sequences."""
    selected = clustered[clustered["cluster"] != "Unclustered"]
    if selected.empty:
        return pd.DataFrame()
    result = (
        selected.groupby("cluster", as_index=False)
        .agg(
            events=("event_id", "nunique"),
            maximum_magnitude=("magnitude", "max"),
            median_depth_km=("depth_km", "median"),
            energy_joules=("energy_joules", "sum"),
            center_latitude=("latitude", "mean"),
            center_longitude=("longitude", "mean"),
            latest_event=("time", "max"),
            representative_region=("region", lambda values: values.mode().iloc[0]),
        )
        .sort_values(["events", "maximum_magnitude"], ascending=False)
        .reset_index(drop=True)
    )
    result["energy_terajoules"] = result["energy_joules"] / 1e12
    return result


def daily_activity(data: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Aggregate counts and energy and mark robust activity anomalies."""
    if data.empty:
        return pd.DataFrame()
    daily = (
        data.set_index("time")
        .resample("D")
        .agg(events=("event_id", "nunique"), energy_joules=("energy_joules", "sum"))
    )
    daily["baseline"] = daily["events"].rolling(
        window, min_periods=max(3, window // 2)
    ).median()
    absolute_deviation = (daily["events"] - daily["baseline"]).abs()
    mad = absolute_deviation.rolling(
        window, min_periods=max(3, window // 2)
    ).median()
    daily["robust_z"] = (
        (daily["events"] - daily["baseline"]) / (1.4826 * mad).replace(0, np.nan)
    ).fillna(0)
    daily["is_anomaly"] = daily["robust_z"].abs() >= 3
    daily["energy_terajoules"] = daily["energy_joules"] / 1e12
    return daily.reset_index()


def magnitude_frequency(
    data: pd.DataFrame,
    minimum_magnitude: float,
    step: float = 0.1,
) -> pd.DataFrame:
    """Return cumulative event counts above each magnitude threshold."""
    if data.empty:
        return pd.DataFrame(columns=["magnitude_threshold", "events_at_or_above"])
    upper = np.ceil(data["magnitude"].max() / step) * step
    thresholds = np.arange(minimum_magnitude, upper + step / 2, step)
    return pd.DataFrame(
        {
            "magnitude_threshold": thresholds,
            "events_at_or_above": [
                int((data["magnitude"] >= threshold).sum()) for threshold in thresholds
            ],
        }
    )
