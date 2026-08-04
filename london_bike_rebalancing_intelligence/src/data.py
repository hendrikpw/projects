"""TfL BikePoint ingestion, validation and deterministic fallback data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


API_URL = "https://api.tfl.gov.uk/BikePoint"
API_DOCS_URL = "https://api.tfl.gov.uk/swagger/ui/index.html#!/BikePoint/BikePoint_GetAll"
OPEN_DATA_URL = "https://tfl.gov.uk/info-for/open-data-users/our-open-data"
TERMS_URL = "https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service"
OGL_URL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
USER_AGENT = "HendrikDataPortfolio/1.0 (https://github.com/hendrikpw/projects)"


def _as_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def parse_bike_points(payload: list[dict], is_demo: bool = False) -> pd.DataFrame:
    """Convert TfL's additionalProperties records into one validated station table."""
    rows: list[dict] = []
    for item in payload:
        properties = {
            str(prop.get("key")): prop.get("value")
            for prop in item.get("additionalProperties", [])
            if prop.get("key")
        }
        modified_values = [
            pd.to_datetime(prop.get("modified"), utc=True, errors="coerce")
            for prop in item.get("additionalProperties", [])
        ]
        modified_values = [value for value in modified_values if pd.notna(value)]
        lat = pd.to_numeric(item.get("lat"), errors="coerce")
        lon = pd.to_numeric(item.get("lon"), errors="coerce")
        docks = _as_int(properties.get("NbDocks"))
        bikes = _as_int(properties.get("NbBikes"))
        empty = _as_int(properties.get("NbEmptyDocks"))
        standard = _as_int(properties.get("NbStandardBikes"))
        ebikes = _as_int(properties.get("NbEBikes"))
        if pd.isna(lat) or pd.isna(lon) or docks is None or bikes is None or empty is None:
            continue
        rows.append(
            {
                "station_id": str(item.get("id") or ""),
                "station_name": str(item.get("commonName") or "Unnamed station"),
                "terminal_name": str(properties.get("TerminalName") or ""),
                "latitude": float(lat),
                "longitude": float(lon),
                "bikes": max(bikes, 0),
                "standard_bikes": max(standard if standard is not None else bikes - (ebikes or 0), 0),
                "ebikes": max(ebikes or 0, 0),
                "empty_docks": max(empty, 0),
                "docks": max(docks, 0),
                "installed": _as_bool(properties.get("Installed")),
                "locked": _as_bool(properties.get("Locked")),
                "temporary": _as_bool(properties.get("Temporary")),
                "station_updated_at": max(modified_values) if modified_values else pd.NaT,
                "is_demo": bool(is_demo),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("TfL returned no valid BikePoint station records")
    frame = frame.drop_duplicates("station_id", keep="last")
    frame = frame[
        frame["latitude"].between(51.25, 51.75)
        & frame["longitude"].between(-0.55, 0.35)
        & frame["docks"].gt(0)
    ].copy()
    if frame.empty:
        raise ValueError("No stations remained after geographic and capacity validation")
    frame["unavailable_docks"] = (frame["docks"] - frame["bikes"] - frame["empty_docks"]).clip(lower=0)
    frame["capacity_inconsistent"] = (frame["bikes"] + frame["empty_docks"]).gt(frame["docks"])
    frame["bike_type_inconsistent"] = (frame["standard_bikes"] + frame["ebikes"]).ne(frame["bikes"])
    return frame.sort_values("station_name").reset_index(drop=True)


def fetch_live_data(timeout: int = 35) -> tuple[pd.DataFrame, dict]:
    """Fetch the current public TfL BikePoint network snapshot without credentials."""
    response = requests.get(
        API_URL,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected TfL BikePoint response schema")
    data = parse_bike_points(payload)
    updated = data["station_updated_at"].dropna()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return data, {
        "mode": "live",
        "source_url": response.url,
        "retrieved_at": retrieved_at,
        "station_count": len(data),
        "station_update_min": updated.min().isoformat() if not updated.empty else None,
        "station_update_max": updated.max().isoformat() if not updated.empty else None,
    }


def build_demo_data() -> pd.DataFrame:
    """Generate a seeded London-like station network for resilient UI behavior."""
    rng = np.random.default_rng(20260804)
    centers = np.array(
        [
            [51.5074, -0.1278],
            [51.5202, -0.1050],
            [51.5007, -0.0754],
            [51.5150, -0.1700],
            [51.4900, -0.1350],
            [51.5350, -0.0550],
        ]
    )
    rows = []
    now = pd.Timestamp("2026-08-04T06:15:00Z")
    for index in range(420):
        center = centers[index % len(centers)]
        lat = center[0] + rng.normal(0, 0.018)
        lon = center[1] + rng.normal(0, 0.026)
        docks = int(rng.integers(14, 48))
        commuting_wave = 0.50 + 0.32 * np.sin((lon + 0.12) * 35) - 0.18 * np.cos((lat - 51.51) * 55)
        fill = float(np.clip(commuting_wave + rng.normal(0, 0.18), 0, 1))
        bikes = int(round(docks * fill))
        unavailable = int(rng.choice([0, 0, 0, 1, 2]))
        bikes = min(bikes, docks - unavailable)
        ebikes = int(round(bikes * rng.uniform(0.04, 0.25)))
        rows.append(
            {
                "station_id": f"Demo_{index + 1}",
                "station_name": f"Demo Station {index + 1:03d}",
                "terminal_name": f"D{index + 1:05d}",
                "latitude": float(lat),
                "longitude": float(lon),
                "bikes": bikes,
                "standard_bikes": bikes - ebikes,
                "ebikes": ebikes,
                "empty_docks": docks - bikes - unavailable,
                "docks": docks,
                "installed": True,
                "locked": bool(rng.random() < 0.015),
                "temporary": bool(rng.random() < 0.03),
                "station_updated_at": now - timedelta(seconds=int(rng.integers(0, 360))),
                "is_demo": True,
                "unavailable_docks": unavailable,
                "capacity_inconsistent": False,
                "bike_type_inconsistent": False,
            }
        )
    return pd.DataFrame(rows).sort_values("station_name").reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    """Return a live TfL snapshot or a clearly labelled deterministic fallback."""
    try:
        return fetch_live_data()
    except (requests.RequestException, ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        data = build_demo_data()
        return data, {
            "mode": "demo",
            "source_url": API_URL,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "station_count": len(data),
            "station_update_min": data["station_updated_at"].min().isoformat(),
            "station_update_max": data["station_updated_at"].max().isoformat(),
            "fallback_reason": f"{type(exc).__name__}: {str(exc)[:180]}",
        }
