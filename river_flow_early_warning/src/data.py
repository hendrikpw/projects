"""Bounded USGS daily streamflow ingestion with retry and atomic fallback."""

from __future__ import annotations

import hashlib
import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

SITES = {
    "01463500": ("Delaware River at Trenton", 40.22, -74.78),
    "01646500": ("Potomac River near Washington", 38.95, -77.13),
    "02177000": ("Chattooga River near Clayton", 34.81, -83.31),
    "07010000": ("Mississippi River at St. Louis", 38.63, -90.18),
    "09402500": ("Colorado River near Grand Canyon", 36.10, -112.09),
    "12149000": ("Snoqualmie River near Carnation", 47.67, -121.93),
}
API = "https://waterservices.usgs.gov/nwis/dv/"
DOCS = "https://waterservices.usgs.gov/docs/dv-service/daily-values-service-details/"
WDFN = "https://waterdata.usgs.gov/"
RIGHTS = "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
USER_AGENT = "hendrikpw-projects river-flow-control/1.0 https://github.com/hendrikpw/projects"


def _request(start: str = "2018-01-01", end: str | None = None, retries: int = 3) -> bytes:
    end = end or date.today().isoformat()
    params = {"format": "rdb", "sites": ",".join(SITES), "startDT": start, "endDT": end,
              "parameterCd": "00060", "statCd": "00003", "siteStatus": "all"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=(5, 50))
            response.raise_for_status()
            if not 50_000 < len(response.content) < 3_000_000:
                raise ValueError("USGS response outside safety bounds")
            if b"agency_cd\tsite_no\tdatetime" not in response.content:
                raise ValueError("USGS RDB header missing")
            return response.content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"USGS source unavailable after {retries} attempts: {last}")


def parse_rdb(raw: bytes) -> pd.DataFrame:
    """Parse repeated per-site RDB blocks without assuming internal series IDs."""
    records: list[dict] = []
    header: list[str] | None = None
    for line in io.StringIO(raw.decode("utf-8", errors="strict")):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "agency_cd":
            header = parts
            continue
        if parts[0].endswith("s") and header and len(parts) == len(header):
            continue
        if header and parts[0] == "USGS" and len(parts) == len(header):
            row = dict(zip(header, parts))
            value_column = next((name for name in header if name.endswith("_00060_00003")), None)
            qualifier_column = f"{value_column}_cd" if value_column else None
            records.append({"agency": row["agency_cd"], "site_no": row["site_no"],
                            "event_date": row["datetime"], "discharge_cfs": row.get(value_column, ""),
                            "qualifier": row.get(qualifier_column, "")})
    if not records:
        raise ValueError("USGS response contained no daily discharge rows")
    return pd.DataFrame(records)


def _fallback() -> bytes:
    """Reproducible eight-year hydrology-like snapshot with seasonal peaks."""
    rng = np.random.default_rng(20260816)
    rows = ["# deterministic demonstration data"]
    for index, site in enumerate(SITES):
        rows.extend(["agency_cd\tsite_no\tdatetime\tdemo_00060_00003\tdemo_00060_00003_cd", "5s\t15s\t20d\t14n\t10s"])
        value = 900.0 * (1 + index * 0.75)
        for day in pd.date_range("2018-01-01", "2026-08-15", freq="D"):
            seasonal = 1 + 0.48 * np.sin(2*np.pi*(day.dayofyear + index*24)/365.25)
            shock = rng.lognormal(0, .18)
            if rng.random() < .018:
                shock *= rng.uniform(2.5, 6)
            value = max(5, .72*value + .28*(900*(1+index*.75)*seasonal*shock))
            rows.append(f"USGS\t{site}\t{day.date()}\t{value:.2f}\tA")
    return ("\n".join(rows) + "\n").encode()


def load_source() -> tuple[bytes, dict]:
    try:
        raw, mode, reason = _request(), "live", ""
    except Exception as exc:
        raw, mode, reason = _fallback(), "demo", str(exc)
    return raw, {"mode": mode, "fallback_reason": reason, "source_hash": hashlib.sha256(raw).hexdigest(),
                 "source_bytes": len(raw), "site_count": len(SITES), "endpoint": API, "docs": DOCS}
