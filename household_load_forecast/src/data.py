"""Bounded UCI household-power ingestion and deterministic fallback."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"
DOI = "https://doi.org/10.24432/C58K54"
ARCHIVE = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
SOURCE_FILE = "household_power_consumption.txt"
NUMERIC = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def _download(timeout: int = 45, attempts: int = 3, max_bytes: int = 30_000_000) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(ARCHIVE, timeout=timeout, headers={"User-Agent": "hendrik-data-portfolio/1.0"})
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"transient HTTP {response.status_code}", response=response)
            response.raise_for_status()
            content = response.content
            if not (10_000_000 <= len(content) <= max_bytes) or not content.startswith(b"PK"):
                raise ValueError(f"archive contract failed for {len(content)} bytes")
            return content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(.4 * (2**attempt))
    raise RuntimeError(f"UCI source unavailable after {attempts} attempts: {last}")


def _safe_member(content: bytes) -> tuple[zipfile.ZipFile, zipfile.ZipInfo]:
    archive = zipfile.ZipFile(io.BytesIO(content))
    for info in archive.infolist():
        parts = info.filename.replace("\\", "/").split("/")
        if info.filename.startswith(("/", "\\")) or ".." in parts:
            raise ValueError("unsafe archive path")
    matches = [item for item in archive.infolist() if item.filename == SOURCE_FILE]
    if len(matches) != 1 or not (100_000_000 <= matches[0].file_size <= 150_000_000):
        raise ValueError("archive allowlist or expanded-size contract failed")
    return archive, matches[0]


def _hourly_from_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, chunk_rows: int = 250_000) -> tuple[pd.DataFrame, dict]:
    pieces: list[pd.DataFrame] = []
    received = missing = 0
    with archive.open(member) as handle:
        chunks = pd.read_csv(handle, sep=";", dtype=str, na_values=["?", ""], chunksize=chunk_rows)
        for chunk in chunks:
            received += len(chunk)
            if list(chunk.columns) != ["Date", "Time", *NUMERIC]:
                raise ValueError("source schema changed")
            timestamp = pd.to_datetime(chunk.Date + " " + chunk.Time, format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True)
            values = chunk[NUMERIC].apply(pd.to_numeric, errors="coerce")
            valid = timestamp.notna() & values.Global_active_power.notna()
            missing += int((~valid).sum())
            x = values.loc[valid].copy(); x["timestamp"] = timestamp.loc[valid].dt.floor("h")
            sums = x.groupby("timestamp", as_index=False)[NUMERIC].sum(min_count=1)
            counts = x.groupby("timestamp", as_index=False).size().rename(columns={"size": "readings"})
            pieces.append(sums.merge(counts, on="timestamp"))
    combined = pd.concat(pieces, ignore_index=True)
    aggregation = {column: "sum" for column in NUMERIC}; aggregation["readings"] = "sum"
    hourly = combined.groupby("timestamp", as_index=False).agg(aggregation).sort_values("timestamp")
    for column in ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]:
        hourly[column] = hourly[column] / hourly.readings
    hourly = hourly.rename(columns={
        "Global_active_power": "load_kw", "Global_reactive_power": "reactive_kw", "Voltage": "voltage_v",
        "Global_intensity": "intensity_a", "Sub_metering_1": "kitchen_wh", "Sub_metering_2": "laundry_wh", "Sub_metering_3": "climate_wh",
    }).reset_index(drop=True)
    return hourly, {"source_rows": received, "missing_source_rows": missing}


def fetch_live() -> tuple[pd.DataFrame, dict]:
    content = _download(); archive, member = _safe_member(content)
    hourly, audit = _hourly_from_member(archive, member)
    digest = hashlib.sha256(content).hexdigest()
    return hourly, {"mode": "live", "source_url": ARCHIVE, "source_hash": digest, "source_bytes": len(content), "license": "CC BY 4.0", "fallback_reason": "", **audit}


def fallback_data(seed: int = 42, periods: int = 34_344) -> tuple[pd.DataFrame, dict]:
    """Create realistic hourly load with daily, weekly, annual and autoregressive structure."""
    rng = np.random.default_rng(seed); ts = pd.date_range("2006-12-16 18:00", periods=periods, freq="h", tz="UTC")
    hour = ts.hour.to_numpy(); dow = ts.dayofweek.to_numpy(); doy = ts.dayofyear.to_numpy()
    base = 1.15 + .48 * np.cos(2 * np.pi * (hour - 20) / 24) + .24 * np.cos(4 * np.pi * (hour - 19) / 24)
    base += .20 * (dow >= 5) + .24 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    noise = rng.normal(0, .16, periods); ar = np.zeros(periods)
    for index in range(1, periods): ar[index] = .76 * ar[index - 1] + noise[index]
    load = np.clip(base + ar, .12, 6.0); voltage = 240 + rng.normal(0, 2.4, periods) - .5 * load
    frame = pd.DataFrame({"timestamp": ts, "load_kw": load, "reactive_kw": np.clip(.12 + .07 * load + rng.normal(0, .025, periods), 0, None), "voltage_v": voltage, "intensity_a": load * 1000 / voltage, "kitchen_wh": np.clip((hour == 19) * 12 + rng.gamma(1.2, 1.4, periods), 0, None), "laundry_wh": rng.gamma(1.1, 1.8, periods), "climate_wh": np.clip(8 + 9 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2, periods), 0, None), "readings": 60})
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return frame, {"mode": "demo", "source_url": ARCHIVE, "source_hash": digest, "source_bytes": int(frame.memory_usage(deep=True).sum()), "source_rows": periods * 60, "missing_source_rows": 0, "license": "CC BY 4.0", "fallback_reason": "Live UCI archive was unavailable or failed its contract."}


def load_source() -> tuple[pd.DataFrame, dict]:
    try:
        return fetch_live()
    except Exception as exc:
        frame, metadata = fallback_data(); metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return frame, metadata
