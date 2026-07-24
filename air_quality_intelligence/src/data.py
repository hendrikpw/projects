"""Open-Meteo air-quality ingestion with a clearly labelled demo fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import requests


API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_VARIABLES = ("european_aqi", "pm2_5", "pm10", "nitrogen_dioxide", "ozone")


@dataclass(frozen=True)
class City:
    name: str
    country: str
    latitude: float
    longitude: float


CITIES = {
    city.name: city
    for city in (
        City("Stuttgart", "Germany", 48.7758, 9.1829),
        City("Berlin", "Germany", 52.5200, 13.4050),
        City("Paris", "France", 48.8566, 2.3522),
        City("Madrid", "Spain", 40.4168, -3.7038),
        City("Milan", "Italy", 45.4642, 9.1900),
        City("Amsterdam", "Netherlands", 52.3676, 4.9041),
        City("Vienna", "Austria", 48.2082, 16.3738),
        City("Prague", "Czechia", 50.0755, 14.4378),
    )
}


def fetch_city(city: City, timeout: int = 15) -> pd.DataFrame:
    """Fetch seven forecast days for one city and return tidy hourly rows."""
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": 7,
        "timezone": "auto",
    }
    response = requests.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    if not hourly.get("time"):
        raise ValueError(f"No hourly observations returned for {city.name}")
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["city"] = city.name
    frame["country"] = city.country
    frame["latitude"] = city.latitude
    frame["longitude"] = city.longitude
    frame["data_mode"] = "Live forecast"
    return frame


def fetch_cities(city_names: Iterable[str]) -> pd.DataFrame:
    """Fetch and combine selected cities; fail atomically to avoid mixed provenance."""
    frames = [fetch_city(CITIES[name]) for name in city_names]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_demo_data(city_names: Iterable[str], hours: int = 168) -> pd.DataFrame:
    """Generate deterministic, explicitly synthetic rows when the API is down."""
    start = pd.Timestamp(datetime.now()).floor("h")
    times = pd.date_range(start, periods=hours, freq="h")
    frames: list[pd.DataFrame] = []
    for index, name in enumerate(city_names):
        city = CITIES[name]
        rng = np.random.default_rng(2407 + index)
        hour = np.arange(hours)
        cycle = np.sin((hour - 7) * 2 * np.pi / 24)
        pm25 = np.clip(11 + index * 1.7 + cycle * 4 + rng.normal(0, 1.8, hours), 2, None)
        pm10 = np.clip(pm25 * 1.55 + rng.normal(1.5, 2.2, hours), 4, None)
        no2 = np.clip(18 + index * 2 + np.maximum(cycle, 0) * 18 + rng.normal(0, 3, hours), 2, None)
        ozone = np.clip(55 - cycle * 18 + rng.normal(0, 4, hours), 5, None)
        aqi = np.maximum.reduce([pm25 * 2.0, pm10, no2 * 0.72, ozone * 0.45])
        frames.append(pd.DataFrame({
            "time": times, "european_aqi": aqi.round(1), "pm2_5": pm25.round(1),
            "pm10": pm10.round(1), "nitrogen_dioxide": no2.round(1), "ozone": ozone.round(1),
            "city": city.name, "country": city.country, "latitude": city.latitude,
            "longitude": city.longitude, "data_mode": "Synthetic demo",
        }))
    return pd.concat(frames, ignore_index=True)
