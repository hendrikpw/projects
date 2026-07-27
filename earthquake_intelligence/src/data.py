"""USGS earthquake ingestion with a deterministic synthetic fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
CATALOG_URL = "https://earthquake.usgs.gov/earthquakes/search/"

TECTONIC_CENTERS = {
    "Alaska Peninsula": (55.2, -160.4, 38),
    "Japan Trench": (38.2, 142.2, 45),
    "Indonesia": (-3.2, 128.2, 65),
    "Central Chile": (-31.5, -71.8, 42),
    "California": (36.2, -118.1, 11),
    "Mediterranean": (37.8, 24.2, 28),
    "South Pacific": (-20.3, -175.1, 120),
    "Himalaya": (29.8, 81.0, 24),
}


def _region_from_place(place: object) -> str:
    text = str(place or "Location not reported").strip()
    if "," in text:
        return text.rsplit(",", 1)[-1].strip()
    return text[:48]


def _prepare_frame(rows: list[dict], is_demo: bool) -> pd.DataFrame:
    """Validate the analytical schema and derive physical features."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No earthquake events were returned")

    numeric = [
        "magnitude",
        "latitude",
        "longitude",
        "depth_km",
        "significance",
        "felt_reports",
        "cdi",
        "mmi",
        "tsunami",
    ]
    for field in numeric:
        if field not in frame:
            frame[field] = np.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame["updated"] = pd.to_datetime(frame["updated"], utc=True, errors="coerce")
    frame = frame.dropna(
        subset=["event_id", "time", "magnitude", "latitude", "longitude", "depth_km"]
    ).copy()
    frame = frame[
        frame["latitude"].between(-90, 90)
        & frame["longitude"].between(-180, 180)
        & frame["depth_km"].between(-5, 800)
    ].copy()
    if frame.empty:
        raise ValueError("No valid earthquake coordinates remained after validation")

    frame["magnitude"] = frame["magnitude"].clip(lower=-2, upper=10)
    frame["depth_km"] = frame["depth_km"].clip(lower=0)
    frame["energy_joules"] = np.power(10.0, 1.5 * frame["magnitude"] + 4.8)
    frame["date"] = frame["time"].dt.floor("D").dt.tz_localize(None)
    frame["hour_utc"] = frame["time"].dt.hour
    frame["region"] = frame["place"].map(_region_from_place)
    frame["depth_class"] = pd.cut(
        frame["depth_km"],
        bins=[-0.01, 70, 300, np.inf],
        labels=["Shallow · 0–70 km", "Intermediate · 70–300 km", "Deep · 300+ km"],
    )
    frame["reviewed"] = frame["status"].fillna("").str.casefold().eq("reviewed")
    frame["tsunami_flag"] = frame["tsunami"].fillna(0).astype(int).gt(0)
    frame["is_demo"] = is_demo
    return frame.sort_values("time", ascending=False).reset_index(drop=True)


def fetch_earthquakes(
    lookback_days: int = 30,
    minimum_magnitude: float = 2.5,
    timeout: int = 25,
    limit: int = 20_000,
) -> tuple[pd.DataFrame, dict]:
    """Fetch a bounded GeoJSON query from the USGS FDSN event service."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    response = requests.get(
        API_URL,
        params={
            "format": "geojson",
            "starttime": start.isoformat(timespec="seconds"),
            "endtime": end.isoformat(timespec="seconds"),
            "minmagnitude": minimum_magnitude,
            "eventtype": "earthquake",
            "orderby": "time",
            "limit": limit,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError("Unexpected USGS GeoJSON response")

    rows = []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 3:
            continue
        rows.append(
            {
                "event_id": feature.get("id"),
                "time": pd.to_datetime(properties.get("time"), unit="ms", utc=True),
                "updated": pd.to_datetime(properties.get("updated"), unit="ms", utc=True),
                "magnitude": properties.get("mag"),
                "place": properties.get("place"),
                "longitude": coordinates[0],
                "latitude": coordinates[1],
                "depth_km": coordinates[2],
                "significance": properties.get("sig"),
                "felt_reports": properties.get("felt"),
                "cdi": properties.get("cdi"),
                "mmi": properties.get("mmi"),
                "alert": properties.get("alert"),
                "status": properties.get("status"),
                "tsunami": properties.get("tsunami"),
                "event_url": properties.get("url"),
                "network": properties.get("net"),
            }
        )

    data = _prepare_frame(rows, is_demo=False)
    metadata = payload.get("metadata") or {}
    return data, {
        "mode": "live",
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "catalog_generated_at": pd.to_datetime(
            metadata.get("generated"), unit="ms", utc=True
        ).isoformat()
        if metadata.get("generated")
        else None,
        "row_limit_reached": len(features) >= limit,
    }


def build_demo_data(rows: int = 2_400) -> pd.DataFrame:
    """Generate stable, realistic-looking events for resilient UI demonstration."""
    rng = np.random.default_rng(20260727)
    names = np.array(list(TECTONIC_CENTERS))
    probabilities = np.array([0.18, 0.14, 0.15, 0.12, 0.15, 0.08, 0.11, 0.07])
    centers = rng.choice(names, size=rows, p=probabilities)
    end = pd.Timestamp("2026-07-27T07:00:00Z")
    age_hours = np.minimum(rng.exponential(scale=190, size=rows), 719)
    times = end - pd.to_timedelta(age_hours, unit="h")
    magnitudes = np.clip(2.5 + rng.exponential(scale=0.58, size=rows), 2.5, 7.4)
    reviewed = rng.random(rows) < np.clip((magnitudes - 2.1) / 3.8, 0.25, 0.96)

    generated = []
    for index, center_name in enumerate(centers):
        center_lat, center_lon, base_depth = TECTONIC_CENTERS[center_name]
        latitude = float(np.clip(center_lat + rng.normal(0, 2.1), -89.5, 89.5))
        longitude = float(((center_lon + rng.normal(0, 2.8) + 180) % 360) - 180)
        depth = float(np.clip(rng.lognormal(np.log(max(base_depth, 4)), 0.62), 2, 690))
        magnitude = float(magnitudes[index])
        tsunami = int(magnitude >= 6.4 and depth < 80 and rng.random() < 0.38)
        generated.append(
            {
                "event_id": f"demo-{index:05d}",
                "time": times[index],
                "updated": times[index] + pd.to_timedelta(35, unit="m"),
                "magnitude": magnitude,
                "place": f"{center_name} synthetic sequence",
                "longitude": longitude,
                "latitude": latitude,
                "depth_km": depth,
                "significance": int(np.clip(80 + 70 * (magnitude - 2.5) ** 1.7, 0, 1000)),
                "felt_reports": int(max(0, rng.lognormal(magnitude - 3.0, 1.1) - 1)),
                "cdi": np.nan,
                "mmi": np.nan,
                "alert": None,
                "status": "reviewed" if reviewed[index] else "automatic",
                "tsunami": tsunami,
                "event_url": "",
                "network": "demo",
            }
        )
    return _prepare_frame(generated, is_demo=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    """Return live USGS data or a clearly labelled deterministic fallback."""
    try:
        return fetch_earthquakes()
    except (requests.RequestException, ValueError, TypeError, OverflowError) as exc:
        data = build_demo_data()
        return data, {
            "mode": "demo",
            "start_time": data["time"].min().isoformat(),
            "end_time": data["time"].max().isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_generated_at": None,
            "row_limit_reached": False,
            "fallback_reason": type(exc).__name__,
        }
