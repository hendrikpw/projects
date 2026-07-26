"""NYC Open Data ingestion and a deterministic synthetic fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


DATASET_ID = "h9gi-nx95"
API_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
DATASET_URL = (
    "https://data.cityofnewyork.us/Public-Safety/"
    "Motor-Vehicle-Collisions-Crashes/h9gi-nx95"
)

SELECT_FIELDS = [
    "crash_date",
    "crash_time",
    "borough",
    "zip_code",
    "latitude",
    "longitude",
    "on_street_name",
    "cross_street_name",
    "off_street_name",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_pedestrians_killed",
    "number_of_cyclist_injured",
    "number_of_cyclist_killed",
    "number_of_motorist_injured",
    "number_of_motorist_killed",
    "contributing_factor_vehicle_1",
    "contributing_factor_vehicle_2",
    "collision_id",
    "vehicle_type_code1",
]

NUMERIC_FIELDS = [
    "latitude",
    "longitude",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_pedestrians_killed",
    "number_of_cyclist_injured",
    "number_of_cyclist_killed",
    "number_of_motorist_injured",
    "number_of_motorist_killed",
]

BOROUGH_CENTERS = {
    "BRONX": (40.8448, -73.8648),
    "BROOKLYN": (40.6501, -73.9496),
    "MANHATTAN": (40.7831, -73.9712),
    "QUEENS": (40.7282, -73.7949),
    "STATEN ISLAND": (40.5795, -74.1502),
}


def _latest_available_date(session: requests.Session, timeout: int) -> pd.Timestamp:
    response = session.get(
        API_URL,
        params={"$select": "max(crash_date) as latest_crash_date"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload or not payload[0].get("latest_crash_date"):
        raise ValueError("NYC Open Data did not return a latest crash date")
    return pd.Timestamp(payload[0]["latest_crash_date"]).normalize()


def _prepare_data(records: list[dict], is_demo: bool) -> pd.DataFrame:
    """Apply a stable schema and quality rules to raw collision records."""
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No collision records were returned")

    for field in SELECT_FIELDS:
        if field not in data:
            data[field] = np.nan
    for field in NUMERIC_FIELDS:
        data[field] = pd.to_numeric(data[field], errors="coerce")

    data["crash_date"] = pd.to_datetime(data["crash_date"], errors="coerce").dt.normalize()
    time_text = data["crash_time"].fillna("00:00").astype(str)
    data["timestamp"] = pd.to_datetime(
        data["crash_date"].dt.strftime("%Y-%m-%d") + " " + time_text,
        errors="coerce",
    )
    data["hour"] = data["timestamp"].dt.hour
    data["weekday"] = data["timestamp"].dt.day_name()
    data["day_type"] = np.where(data["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")
    data["borough"] = data["borough"].fillna("UNKNOWN").str.upper().str.strip()

    street_fields = ["on_street_name", "off_street_name", "cross_street_name"]
    data["street"] = (
        data[street_fields]
        .bfill(axis=1)
        .iloc[:, 0]
        .fillna("Location not reported")
        .astype(str)
        .str.strip()
        .str.title()
    )
    data["primary_factor"] = (
        data["contributing_factor_vehicle_1"]
        .fillna("Unspecified")
        .astype(str)
        .str.strip()
    )

    for field in NUMERIC_FIELDS[2:]:
        data[field] = data[field].fillna(0).clip(lower=0)
    data["vulnerable_injured"] = (
        data["number_of_pedestrians_injured"] + data["number_of_cyclist_injured"]
    )
    data["vulnerable_killed"] = (
        data["number_of_pedestrians_killed"] + data["number_of_cyclist_killed"]
    )
    data["injury_collision"] = data["number_of_persons_injured"] > 0
    data["fatal_collision"] = data["number_of_persons_killed"] > 0
    data["serious_outcome"] = data["fatal_collision"] | (
        data["number_of_persons_injured"] >= 2
    )
    data["valid_coordinate"] = (
        data["latitude"].between(40.45, 40.95)
        & data["longitude"].between(-74.30, -73.65)
    )
    data["is_demo"] = is_demo

    data = data.dropna(subset=["crash_date", "timestamp"])
    data["collision_id"] = data["collision_id"].fillna(
        pd.Series(data.index, index=data.index).map(lambda value: f"row-{value}")
    )
    return data.sort_values("timestamp", ascending=False).reset_index(drop=True)


def fetch_collisions(
    lookback_days: int = 120,
    timeout: int = 20,
    page_size: int = 20_000,
    max_rows: int = 60_000,
) -> tuple[pd.DataFrame, dict]:
    """Fetch the latest available window through the keyless Socrata API."""
    records: list[dict] = []
    with requests.Session() as session:
        latest_date = _latest_available_date(session, timeout)
        start_date = latest_date - timedelta(days=lookback_days - 1)

        for offset in range(0, max_rows, page_size):
            response = session.get(
                API_URL,
                params={
                    "$select": ",".join(SELECT_FIELDS),
                    "$where": (
                        f"crash_date >= '{start_date:%Y-%m-%dT00:00:00.000}' "
                        f"AND crash_date <= '{latest_date:%Y-%m-%dT23:59:59.999}'"
                    ),
                    "$order": "crash_date DESC, collision_id DESC",
                    "$limit": page_size,
                    "$offset": offset,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError("Unexpected NYC Open Data response")
            records.extend(page)
            if len(page) < page_size:
                break

    data = _prepare_data(records, is_demo=False)
    return data, {
        "mode": "live",
        "latest_date": latest_date.date().isoformat(),
        "start_date": start_date.date().isoformat(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_limit_reached": len(records) >= max_rows,
    }


def build_demo_data(rows: int = 4_500) -> pd.DataFrame:
    """Generate a fixed, realistic-looking dataset that is never called observed."""
    rng = np.random.default_rng(20260726)
    boroughs = np.array(list(BOROUGH_CENTERS))
    borough_probabilities = np.array([0.16, 0.31, 0.19, 0.28, 0.06])
    chosen = rng.choice(boroughs, size=rows, p=borough_probabilities)
    end_date = pd.Timestamp("2026-06-11")
    day_offsets = np.minimum(rng.exponential(scale=38, size=rows).astype(int), 119)
    dates = end_date - pd.to_timedelta(day_offsets, unit="D")
    hour_weights = np.array(
        [
                0.025,
                0.020,
                0.017,
                0.014,
                0.013,
                0.018,
                0.035,
                0.055,
                0.063,
                0.049,
                0.043,
                0.044,
                0.047,
                0.050,
                0.054,
                0.061,
                0.069,
                0.076,
                0.071,
                0.058,
                0.047,
                0.040,
                0.034,
                0.037,
        ]
    )
    hours = rng.choice(
        np.arange(24),
        size=rows,
        p=hour_weights / hour_weights.sum(),
    )
    minutes = rng.integers(0, 60, size=rows)
    factors = rng.choice(
        [
            "Driver Inattention/Distraction",
            "Failure to Yield Right-of-Way",
            "Following Too Closely",
            "Unsafe Speed",
            "Passing or Lane Usage Improper",
            "Traffic Control Disregarded",
            "Backing Unsafely",
            "Unspecified",
        ],
        size=rows,
        p=[0.24, 0.14, 0.13, 0.08, 0.10, 0.06, 0.07, 0.18],
    )
    injury_probability = np.where(np.isin(factors, ["Unsafe Speed", "Traffic Control Disregarded"]), 0.34, 0.19)
    injured = rng.binomial(1, injury_probability) + rng.binomial(1, 0.025, size=rows)
    killed = rng.binomial(1, np.where(factors == "Unsafe Speed", 0.008, 0.002))
    vulnerable = rng.binomial(1, np.where(chosen == "MANHATTAN", 0.09, 0.055))

    records = []
    for index in range(rows):
        center_lat, center_lon = BOROUGH_CENTERS[chosen[index]]
        crash_date = dates[index]
        records.append(
            {
                "crash_date": crash_date.isoformat(),
                "crash_time": f"{hours[index]}:{minutes[index]:02d}",
                "borough": chosen[index],
                "zip_code": "",
                "latitude": center_lat + rng.normal(0, 0.035),
                "longitude": center_lon + rng.normal(0, 0.045),
                "on_street_name": f"Demo corridor {index % 37 + 1}",
                "cross_street_name": "",
                "off_street_name": "",
                "number_of_persons_injured": int(injured[index] + vulnerable[index]),
                "number_of_persons_killed": int(killed[index]),
                "number_of_pedestrians_injured": int(vulnerable[index]),
                "number_of_pedestrians_killed": 0,
                "number_of_cyclist_injured": 0,
                "number_of_cyclist_killed": 0,
                "number_of_motorist_injured": int(injured[index]),
                "number_of_motorist_killed": int(killed[index]),
                "contributing_factor_vehicle_1": factors[index],
                "contributing_factor_vehicle_2": "Unspecified",
                "collision_id": f"demo-{index}",
                "vehicle_type_code1": "Demo vehicle",
            }
        )
    return _prepare_data(records, is_demo=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    """Use live public data when possible and degrade gracefully otherwise."""
    try:
        return fetch_collisions()
    except (requests.RequestException, ValueError, TypeError) as exc:
        data = build_demo_data()
        return data, {
            "mode": "demo",
            "latest_date": data["crash_date"].max().date().isoformat(),
            "start_date": data["crash_date"].min().date().isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_limit_reached": False,
            "fallback_reason": type(exc).__name__,
        }
