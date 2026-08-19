"""Bounded, retrying USAspending ingestion with deterministic fallback."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

API = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
LAST_UPDATED = "https://api.usaspending.gov/api/v2/awards/last_updated/"
DOCS = "https://api.usaspending.gov/docs/endpoints"
CONTRACT = "https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/contracts/v2/search/spending_by_award.md"
ABOUT = "https://www.usaspending.gov/data/about-the-data-download.pdf"
FIELDS = [
    "Award ID", "Recipient Name", "Recipient UEI", "recipient_id", "Award Amount",
    "Start Date", "End Date", "Awarding Agency", "Awarding Sub Agency",
    "Contract Award Type", "NAICS", "PSC", "Description", "Last Modified Date",
]


def _request_json(method: str, url: str, *, payload: dict | None = None, timeout: int = 25, attempts: int = 3) -> dict:
    """Request JSON with bounded exponential retry for transient failures."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, json=payload, timeout=timeout, headers={"User-Agent": "hendrik-data-portfolio/1.0"})
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"transient HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.35 * (2**attempt))
    raise RuntimeError(f"USAspending request failed after {attempts} attempts: {last}")


def _flatten(record: dict) -> dict:
    naics, psc = record.get("NAICS") or {}, record.get("PSC") or {}
    return {
        "award_id": record.get("generated_internal_id") or record.get("Award ID"),
        "display_award_id": record.get("Award ID"),
        "recipient_name": record.get("Recipient Name"),
        "recipient_uei": record.get("Recipient UEI"),
        "recipient_id": record.get("recipient_id"),
        "award_amount": record.get("Award Amount"),
        "start_date": record.get("Start Date"),
        "end_date": record.get("End Date"),
        "awarding_agency": record.get("Awarding Agency"),
        "awarding_subagency": record.get("Awarding Sub Agency"),
        "award_type": record.get("Contract Award Type"),
        "naics_code": naics.get("code"),
        "naics_description": naics.get("description"),
        "psc_code": psc.get("code"),
        "psc_description": psc.get("description"),
        "description": record.get("Description"),
        "last_modified": record.get("Last Modified Date"),
    }


def fetch_live(page_limit: int = 6, page_size: int = 100) -> tuple[pd.DataFrame, dict]:
    """Fetch a bounded FY contract snapshot; stop safely at the final API page."""
    updated = _request_json("GET", LAST_UPDATED).get("last_updated")
    end = pd.to_datetime(updated, errors="coerce")
    if pd.isna(end):
        end = pd.Timestamp(date.today() - timedelta(days=1))
    start = pd.Timestamp(year=end.year if end.month >= 10 else end.year - 1, month=10, day=1)
    base = {
        "subawards": False,
        "limit": page_size,
        "filters": {"award_type_codes": ["A", "B", "C", "D"], "time_period": [{"start_date": start.date().isoformat(), "end_date": end.date().isoformat()}]},
        "fields": FIELDS,
        "sort": "Last Modified Date",
        "order": "desc",
    }
    rows, pages = [], 0
    for page in range(1, page_limit + 1):
        payload = {**base, "page": page}
        body = _request_json("POST", API, payload=payload)
        batch = body.get("results") or []
        rows.extend(_flatten(item) for item in batch)
        pages += 1
        if not body.get("page_metadata", {}).get("hasNext") or not batch:
            break
    if len(rows) < 120:
        raise RuntimeError(f"live extract returned only {len(rows)} contract awards")
    raw = pd.DataFrame(rows)
    source_hash = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    return raw, {"mode": "live", "source_url": API, "source_hash": source_hash, "source_bytes": int(raw.memory_usage(deep=True).sum()), "pages": pages, "as_of": end.date().isoformat(), "query": base["filters"]}


def fallback_data(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Build an auditable demonstration snapshot with realistic identities."""
    rng = np.random.default_rng(seed)
    stems = ["NORTHSTAR", "BLUE RIDGE", "CIVIC", "QUANTUM", "REDWOOD", "HORIZON", "SUMMIT", "ATLAS", "ORBITAL", "LIBERTY", "PIONEER", "CEDAR"]
    nouns = ["ANALYTICS", "SYSTEMS", "LOGISTICS", "TECHNOLOGIES", "RESEARCH"]
    agencies = ["Department of Energy", "Department of Transportation", "Department of Health and Human Services", "National Aeronautics and Space Administration"]
    rows = []
    for entity in range(60):
        canonical = f"{stems[entity % len(stems)]} {nouns[(entity // len(stems)) % len(nouns)]} {['LLC','INC','CORP'][entity % 3]}"
        uei = hashlib.sha256(f"uei-{entity}".encode()).hexdigest()[:12].upper()
        for award in range(3):
            rows.append({
                "award_id": f"CONT_AWD_DEMO{entity:03d}{award:02d}", "display_award_id": f"DEMO-{entity:03d}-{award:02d}",
                "recipient_name": canonical, "recipient_uei": uei, "recipient_id": f"recipient-{entity:03d}-C",
                "award_amount": round(float(rng.lognormal(13.2, 1.1)), 2), "start_date": f"2025-{1 + (entity + award) % 9:02d}-01",
                "end_date": f"2027-{1 + (entity + award) % 9:02d}-28", "awarding_agency": agencies[entity % len(agencies)],
                "awarding_subagency": agencies[entity % len(agencies)], "award_type": "DEFINITIVE CONTRACT",
                "naics_code": "541512", "naics_description": "COMPUTER SYSTEMS DESIGN SERVICES", "psc_code": "DA01",
                "psc_description": "IT AND TELECOM", "description": f"DEMONSTRATION SERVICE AWARD {award}", "last_modified": f"2026-08-{1 + entity % 18:02d} 12:00:00",
            })
    raw = pd.DataFrame(rows)
    source_hash = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    return raw, {"mode": "demo", "source_url": API, "source_hash": source_hash, "source_bytes": int(raw.memory_usage(deep=True).sum()), "pages": 0, "as_of": "2026-08-18", "query": "deterministic fallback", "fallback_reason": "Live USAspending delivery was unavailable or failed its minimum-volume check."}


def load_source(page_limit: int = 6) -> tuple[pd.DataFrame, dict]:
    try:
        return fetch_live(page_limit=page_limit)
    except Exception as exc:
        raw, metadata = fallback_data()
        metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return raw, metadata


def query_manifest(metadata: dict) -> bytes:
    return json.dumps(metadata, indent=2, default=str).encode()
