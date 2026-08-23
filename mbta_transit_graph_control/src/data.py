"""Bounded MBTA GTFS ingestion and an atomic deterministic fallback."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

FEED_URL = "https://cdn.mbta.com/MBTA_GTFS.zip"
DOCS_URL = "https://github.com/mbta/gtfs-documentation/blob/master/reference/gtfs.md"
LICENSE_URL = "https://github.com/mbta/gtfs-documentation/blob/master/developers-license-agreement.pdf"
GTFS_SPEC = "https://gtfs.org/documentation/schedule/reference/"
REQUIRED = {"feed_info.txt", "stops.txt", "routes.txt", "route_patterns.txt", "stop_times.txt"}


@dataclass(frozen=True)
class GTFSExtract:
    stops: pd.DataFrame
    routes: pd.DataFrame
    patterns: pd.DataFrame
    stop_times: pd.DataFrame
    feed_info: pd.DataFrame
    metadata: dict


def _download(timeout: int = 45, attempts: int = 3, max_bytes: int = 30_000_000) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(FEED_URL, timeout=timeout, headers={"User-Agent": "hendrik-data-portfolio/1.0"})
            if response.status_code in {429, 500, 502, 503, 504}: raise requests.HTTPError(f"transient HTTP {response.status_code}", response=response)
            response.raise_for_status(); content = response.content
            if not (5_000_000 <= len(content) <= max_bytes) or not content.startswith(b"PK"): raise ValueError(f"archive contract failed for {len(content)} bytes")
            return content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts: time.sleep(.4 * 2**attempt)
    raise RuntimeError(f"MBTA source unavailable after {attempts} attempts: {last}")


def _safe_archive(content: bytes) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(io.BytesIO(content)); names = set(archive.namelist())
    for info in archive.infolist():
        parts = info.filename.replace("\\", "/").split("/")
        if info.filename.startswith(("/", "\\")) or ".." in parts: raise ValueError("unsafe archive path")
    if not REQUIRED <= names: raise ValueError(f"missing GTFS members: {sorted(REQUIRED - names)}")
    required_size = sum(info.file_size for info in archive.infolist() if info.filename in REQUIRED)
    if not (10_000_000 <= required_size <= 150_000_000): raise ValueError(f"expanded required members outside bounds: {required_size}")
    return archive


def _read(archive: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(archive.open(name), usecols=columns, dtype="string", low_memory=False)
    if set(frame.columns) != set(columns): raise ValueError(f"{name} schema mismatch")
    return frame


def fetch_live(content: bytes | None = None) -> GTFSExtract:
    content = content if content is not None else _download(); archive = _safe_archive(content)
    feed = _read(archive, "feed_info.txt", ["feed_start_date", "feed_end_date", "feed_version", "feed_id"])
    stops = _read(archive, "stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station", "wheelchair_boarding", "municipality"])
    routes = _read(archive, "routes.txt", ["route_id", "route_short_name", "route_long_name", "route_type", "route_color", "network_id"])
    patterns = _read(archive, "route_patterns.txt", ["route_pattern_id", "route_id", "direction_id", "route_pattern_name", "route_pattern_typicality", "representative_trip_id", "canonical_route_pattern"])
    typical = patterns[(pd.to_numeric(patterns.route_pattern_typicality, errors="coerce") == 1) | (pd.to_numeric(patterns.canonical_route_pattern, errors="coerce") == 1)].copy()
    representative = set(typical.representative_trip_id.dropna().astype(str)); chunks = []
    for chunk in pd.read_csv(archive.open("stop_times.txt"), usecols=["trip_id", "stop_id", "stop_sequence"], dtype="string", chunksize=300_000):
        chunks.append(chunk[chunk.trip_id.isin(representative)])
    stop_times = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["trip_id", "stop_id", "stop_sequence"])
    digest = hashlib.sha256(content).hexdigest(); info = feed.iloc[0]
    metadata = {"mode":"live","source_url":FEED_URL,"source_hash":digest,"source_bytes":len(content),"feed_version":str(info.feed_version),"feed_start_date":str(info.feed_start_date),"feed_end_date":str(info.feed_end_date),"fallback_reason":""}
    return GTFSExtract(stops, routes, typical.reset_index(drop=True), stop_times, feed, metadata)


def fallback_extract(seed: int = 42) -> GTFSExtract:
    rng = np.random.default_rng(seed); route_rows = []; stop_rows = []; pattern_rows = []; time_rows = []
    for route in range(30):
        route_id = f"R{route:02d}"; route_rows.append([route_id, route_id, f"Demonstration corridor {route}", str(route % 4), f"{(route*7919)%0xFFFFFF:06X}", "demo"])
        trip = f"T{route:02d}"; pattern_rows.append([f"P{route:02d}", route_id, str(route % 2), f"Pattern {route}", "1", trip, "1"])
        nodes = [(route * 9 + offset) % 300 for offset in range(24)]
        for sequence, node in enumerate(nodes): time_rows.append([trip, f"S{node:03d}", str(sequence)])
    for node in range(300):
        angle = 2 * np.pi * node / 300; stop_rows.append([f"S{node:03d}", f"Demo stop {node}", f"{42.36 + .16*np.sin(angle) + rng.normal(0,.002):.6f}", f"{-71.06 + .22*np.cos(angle) + rng.normal(0,.002):.6f}", "0", "", "1", "Demonstration"])
    stops = pd.DataFrame(stop_rows, columns=["stop_id","stop_name","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","municipality"])
    routes = pd.DataFrame(route_rows, columns=["route_id","route_short_name","route_long_name","route_type","route_color","network_id"])
    patterns = pd.DataFrame(pattern_rows, columns=["route_pattern_id","route_id","direction_id","route_pattern_name","route_pattern_typicality","representative_trip_id","canonical_route_pattern"])
    times = pd.DataFrame(time_rows, columns=["trip_id","stop_id","stop_sequence"]); feed = pd.DataFrame([["20260801","20260930","deterministic-demo-v1","demo"]], columns=["feed_start_date","feed_end_date","feed_version","feed_id"])
    digest = hashlib.sha256(pd.concat([stops.astype(str), routes.reindex(columns=stops.columns).astype(str)], ignore_index=True).to_csv(index=False).encode()).hexdigest()
    metadata = {"mode":"demo","source_url":FEED_URL,"source_hash":digest,"source_bytes":int(sum(x.memory_usage(deep=True).sum() for x in [stops,routes,patterns,times,feed])),"feed_version":"deterministic-demo-v1","feed_start_date":"20260801","feed_end_date":"20260930","fallback_reason":"Live MBTA feed was unavailable or failed its contract."}
    return GTFSExtract(stops, routes, patterns, times, feed, metadata)


def load_source(force_fallback: bool = False) -> GTFSExtract:
    if force_fallback: return fallback_extract()
    try: return fetch_live()
    except Exception as exc:
        result = fallback_extract(); result.metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"; return result
