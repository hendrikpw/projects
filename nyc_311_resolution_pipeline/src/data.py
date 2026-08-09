"""Bounded, deterministic extraction from NYC Open Data's 311 dataset."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests


API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
DATASET_URL = "https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data"
API_DOCS_URL = "https://dev.socrata.com/foundry/data.cityofnewyork.us/erm2-nwe9"
OPEN_DATA_URL = "https://opendata.cityofnewyork.us/overview/"
FIELDS = [
    "unique_key", "created_date", "closed_date", "agency", "agency_name",
    "complaint_type", "descriptor", "location_type", "borough",
    "open_data_channel_type", "status", "resolution_action_updated_date",
]
USER_AGENT = "hendrikpw-portfolio/1.0 (public educational data product; github.com/hendrikpw/projects)"


def _request_page(params: dict[str, Any], attempts: int = 2) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=(8, 35)
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Socrata response was not a record list")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.35 * (2**attempt))
    raise RuntimeError(f"NYC Open Data request failed after {attempts} attempts: {last_error}")


def _bounds(history_days: int, maturity_days: int, anchor: datetime | None = None) -> tuple[str, str]:
    if history_days < 180 or history_days > 730:
        raise ValueError("history_days must be between 180 and 730")
    if maturity_days < 30:
        raise ValueError("maturity_days must be at least 30")
    now = anchor or datetime.now(timezone.utc)
    end = (now - timedelta(days=maturity_days)).date()
    start = end - timedelta(days=history_days - 1)
    return start.isoformat(), end.isoformat()


def _fallback(start_date: str, end_date: str, rows: int = 4200) -> list[dict[str, Any]]:
    """Return deterministic source-shaped records with known operational structure."""
    rng = np.random.default_rng(31142)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    agencies = {
        "NYPD": [("Noise - Residential", "Loud Music/Party"), ("Illegal Parking", "Blocked Hydrant")],
        "DSNY": [("Missed Collection", "Trash"), ("Dirty Condition", "Litter")],
        "DEP": [("Water System", "Hydrant Running Full"), ("Sewer", "Catch Basin Clogged/Flooding")],
        "DOT": [("Street Light Condition", "Street Light Out"), ("Sidewalk Condition", "Broken Sidewalk")],
        "HPD": [("HEAT/HOT WATER", "ENTIRE BUILDING"), ("UNSANITARY CONDITION", "PESTS")],
    }
    boroughs = ["BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "STATEN ISLAND"]
    channels = ["ONLINE", "MOBILE", "PHONE"]
    agency_names = {"NYPD": "New York City Police Department", "DSNY": "Department of Sanitation", "DEP": "Department of Environmental Protection", "DOT": "Department of Transportation", "HPD": "Department of Housing Preservation and Development"}
    agency_choices = list(agencies)
    base_hours = {"NYPD": 3.0, "DSNY": 36.0, "DEP": 18.0, "DOT": 120.0, "HPD": 84.0}
    output: list[dict[str, Any]] = []
    total_seconds = max(int((end - start).total_seconds()), 1)
    for index in range(rows):
        created = start + pd.to_timedelta(int(rng.integers(0, total_seconds)), unit="s")
        agency = rng.choice(agency_choices, p=[.36, .18, .15, .16, .15])
        complaint, descriptor = agencies[agency][int(rng.integers(0, 2))]
        weekend = created.dayofweek >= 5
        multiplier = 1.18 if weekend and agency in {"DSNY", "DOT"} else 1.0
        duration = max(.08, float(rng.lognormal(np.log(base_hours[agency] * multiplier), .75)))
        duration = min(duration, 700.0)
        closed = created + pd.to_timedelta(float(duration), unit="h")
        key = str(91000000 + index)
        output.append({
            "unique_key": key, "created_date": created.isoformat(), "closed_date": closed.isoformat(),
            "agency": agency, "agency_name": agency_names[agency], "complaint_type": complaint,
            "descriptor": descriptor, "location_type": "Street/Sidewalk" if agency != "HPD" else "Residential Building",
            "borough": rng.choice(boroughs), "open_data_channel_type": rng.choice(channels, p=[.48, .22, .30]),
            "status": "Closed", "resolution_action_updated_date": closed.isoformat(),
        })
    return output


def fetch_requests(
    history_days: int = 365,
    maturity_days: int = 35,
    page_size: int = 2000,
    max_rows: int = 6000,
    sample_modulus: int = 1999,
    sample_remainders: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a stable key-based sample of mature, closed requests with atomic fallback."""
    if not 200 <= page_size <= 2000:
        raise ValueError("page_size must be between 200 and 2000")
    if max_rows < 1000 or max_rows > 10000:
        raise ValueError("max_rows must be between 1000 and 10000")
    start_date, end_date = _bounds(history_days, maturity_days)
    where = (
        f"created_date between '{start_date}T00:00:00' and '{end_date}T23:59:59' "
        "AND closed_date IS NOT NULL AND agency IS NOT NULL AND complaint_type IS NOT NULL "
        f"AND (unique_key::number % {int(sample_modulus)}) < {int(sample_remainders)}"
    )
    records: list[dict[str, Any]] = []
    try:
        for offset in range(0, max_rows, page_size):
            page = _request_page({
                "$select": ",".join(FIELDS), "$where": where,
                "$order": "created_date ASC,unique_key ASC", "$limit": page_size, "$offset": offset,
            })
            records.extend(page)
            if len(page) < page_size:
                break
        if len(records) < 1200:
            raise ValueError(f"only {len(records)} mature sampled records returned")
        mode, reason = "live", ""
    except (RuntimeError, ValueError) as exc:
        records = _fallback(start_date, end_date)
        mode, reason = "demo", str(exc)
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str).encode()
    return records, {
        "mode": mode, "fallback_reason": reason, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date, "end_date": end_date, "history_days": history_days,
        "maturity_days": maturity_days, "source_rows": len(records),
        "sample_rule": f"unique_key modulo {sample_modulus} < {sample_remainders}",
        "source_hash": hashlib.sha256(canonical).hexdigest(), "source_url": API_URL,
    }
