"""Bounded, retry-aware ingestion for openFDA enforcement reports."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import numpy as np
import requests


DOMAINS = {
    "food": {
        "label": "Food",
        "endpoint": "https://api.fda.gov/food/enforcement.json",
        "docs": "https://open.fda.gov/apis/food/enforcement/",
    },
    "drug": {
        "label": "Drug",
        "endpoint": "https://api.fda.gov/drug/enforcement.json",
        "docs": "https://open.fda.gov/apis/drug/enforcement/",
    },
    "device": {
        "label": "Device",
        "endpoint": "https://api.fda.gov/device/enforcement.json",
        "docs": "https://open.fda.gov/apis/device/enforcement/",
    },
}
CLASSES = ("Class I", "Class II", "Class III")
API_DOCS_URL = "https://open.fda.gov/apis/"
AUTH_URL = "https://open.fda.gov/apis/authentication/"
TERMS_URL = "https://open.fda.gov/terms/"


def safe_domains(domains: list[str] | tuple[str, ...]) -> list[str]:
    """Return unique supported domains in canonical order."""
    requested = set(domains)
    return [domain for domain in DOMAINS if domain in requested]


def _request_json(url: str, params: dict[str, Any], attempts: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=20,
                headers={"User-Agent": "hendrikpw-data-portfolio/1.0"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError("openFDA returned an unexpected response contract")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.25 * (2**attempt))
    raise RuntimeError(f"openFDA request failed after {attempts} attempts: {last_error}")


def _demo_records(domains: list[str], per_class: int) -> list[dict[str, Any]]:
    """Create deterministic, source-shaped records for hosted failure states."""
    rng = np.random.default_rng(20260807)
    patterns = {
        "Class I": [
            ("possible Salmonella contamination", "may cause serious or life-threatening illness"),
            ("undeclared peanut allergen", "risk of severe allergic reaction or anaphylaxis"),
            ("sterility failure", "potential for bloodstream infection in vulnerable patients"),
            ("device may stop therapy without alarm", "could result in serious injury or death"),
        ],
        "Class II": [
            ("potency outside specification", "may cause temporary or medically reversible effects"),
            ("incorrect package insert", "limited risk when used as directed under supervision"),
            ("component may deliver an inaccurate reading", "could delay treatment in some cases"),
            ("quality defect in selected lots", "adverse consequences are expected to be reversible"),
        ],
        "Class III": [
            ("minor labeling omission", "not likely to cause adverse health consequences"),
            ("package count discrepancy", "product quality issue without a health hazard"),
            ("noncritical seal cosmetic defect", "does not affect safety or intended performance"),
            ("formatting error on outer carton", "regulatory labeling correction with low health risk"),
        ],
    }
    products = {
        "food": ["frozen vegetable mix", "protein snack bar", "seasoning blend", "ready-to-eat salad"],
        "drug": ["injectable solution", "oral tablets", "topical cream", "diagnostic reagent"],
        "device": ["infusion pump", "patient monitor", "surgical kit", "laboratory analyzer"],
    }
    rows: list[dict[str, Any]] = []
    counter = 0
    for domain in domains:
        for label in CLASSES:
            for index in range(per_class):
                counter += 1
                reason, consequence = patterns[label][index % len(patterns[label])]
                product = products[domain][(index + counter) % len(products[domain])]
                year = 2022 + (index % 5)
                month = 1 + ((index * 3 + counter) % 12)
                day = 1 + ((index * 7 + counter) % 27)
                rows.append({
                    "_domain": domain,
                    "recall_number": f"D-{domain[:1].upper()}-{counter:06d}",
                    "event_id": str(900000 + counter),
                    "report_date": f"{year:04d}{month:02d}{day:02d}",
                    "classification": label,
                    "status": "Ongoing" if index % 5 == 0 else "Completed",
                    "recalling_firm": f"Demo {domain.title()} Works {1 + index % 18}",
                    "city": "Chicago",
                    "state": "IL",
                    "country": "United States",
                    "product_description": f"{product.title()}, lot {1000 + index}, distributed in sealed packages",
                    "reason_for_recall": f"Recall initiated because of {reason}; {consequence}.",
                    "product_quantity": f"{int(rng.integers(40, 8000))} units",
                    "distribution_pattern": "Distributed to retail and institutional customers in multiple states.",
                    "recall_initiation_date": f"{year:04d}{month:02d}{max(1, day - 2):02d}",
                    "termination_date": "" if index % 5 == 0 else f"{year:04d}{month:02d}{min(28, day + 1):02d}",
                    "voluntary_mandated": "Voluntary: Firm initiated",
                    "initial_firm_notification": "E-Mail",
                })
    return rows


def fetch_recalls(domains: list[str], snapshot_size: int = 270) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a bounded class-stratified snapshot with deterministic fallback."""
    selected = safe_domains(domains)
    if not selected:
        raise ValueError("Select at least one supported enforcement domain")
    if snapshot_size < 90 or snapshot_size > 600:
        raise ValueError("snapshot_size must be between 90 and 600")
    per_class = max(10, snapshot_size // (len(selected) * len(CLASSES)))
    records: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    try:
        for domain in selected:
            for label in CLASSES:
                payload = _request_json(
                    DOMAINS[domain]["endpoint"],
                    {
                        "search": f'classification:"{label}"',
                        "sort": "report_date:desc",
                        "limit": per_class,
                    },
                )
                results = payload["results"]
                for result in results:
                    result = dict(result)
                    result["_domain"] = domain
                    records.append(result)
                total = payload.get("meta", {}).get("results", {}).get("total", len(results))
                counts.append({"domain": domain, "classification": label, "available_records": int(total)})
        if len(records) < len(selected) * len(CLASSES) * 8:
            raise ValueError("too few class-stratified records were returned")
        mode, fallback_reason = "live", ""
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        records = _demo_records(selected, per_class)
        counts = [
            {"domain": domain, "classification": label, "available_records": per_class}
            for domain in selected for label in CLASSES
        ]
        mode, fallback_reason = "demo", f"{type(exc).__name__}: {exc}"
    return records, {
        "mode": mode,
        "fallback_reason": fallback_reason,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "domains": selected,
        "requested_snapshot_size": snapshot_size,
        "stratum_size": per_class,
        "request_count": len(selected) * len(CLASSES),
        "availability": counts,
        "source_urls": [DOMAINS[domain]["endpoint"] for domain in selected],
    }
