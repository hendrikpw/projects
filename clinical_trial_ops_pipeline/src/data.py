"""Bounded ClinicalTrials.gov ingestion with deterministic demo fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import time
from typing import Any

import numpy as np
import requests


API_URL = "https://clinicaltrials.gov/api/v2/studies"
API_DOCS_URL = "https://clinicaltrials.gov/data-api/api"
STRUCTURE_URL = "https://clinicaltrials.gov/data-api/about-api/study-data-structure"
TERMS_URL = "https://clinicaltrials.gov/about-site/terms-conditions"
STATUS_FILTER = "COMPLETED|TERMINATED|WITHDRAWN|SUSPENDED"
RETRYABLE_STATUS = {429, 502, 503, 504}
CONDITION_PRESETS = {
    "Type 2 diabetes": "type 2 diabetes",
    "Breast cancer": "breast cancer",
    "Cardiovascular disease": "cardiovascular disease",
    "Depression": "depression",
    "Alzheimer disease": "Alzheimer disease",
    "Asthma": "asthma",
}


def safe_condition(value: str) -> str:
    """Bound custom search input to a plain, auditable condition query."""
    clean = re.sub(r"[^A-Za-z0-9\- ,()]", " ", str(value))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:80]


def _demo_study(index: int, condition: str, rng: np.random.Generator) -> dict[str, Any]:
    """Create one realistic, deterministic v2-shaped registry record."""
    phases = ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
    phase = phases[index % len(phases)]
    enrollment = int(np.clip(rng.lognormal(4.4 + 0.15 * (phase == "PHASE3"), 0.85), 12, 4500))
    industry = index % 3 == 0
    randomized = index % 4 != 0
    masked = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"][index % 5]
    multicountry = index % 5 == 0
    risk_logit = (
        -1.55
        + 1.15 * (phase in {"PHASE1", "NA"})
        + 0.85 * (enrollment < 55)
        + 0.70 * (not randomized)
        + 0.55 * (masked == "NONE")
        - 0.50 * industry
        + rng.normal(0, 0.35)
    )
    discontinued = rng.random() < 1 / (1 + np.exp(-risk_logit))
    if discontinued:
        status = ["TERMINATED", "WITHDRAWN", "SUSPENDED"][index % 3]
    else:
        status = "COMPLETED"
    year = 2012 + index % 13
    start_month = index % 12 + 1
    countries = ["United States", "Germany", "France"] if multicountry else ["United States"]
    return {
        "protocolSection": {
            "identificationModule": {"nctId": f"NCTDEMO{index:05d}", "briefTitle": f"Demo {condition.title()} Study {index:03d}"},
            "statusModule": {
                "overallStatus": status,
                "startDateStruct": {"date": f"{year}-{start_month:02d}-01"},
                "completionDateStruct": {"date": f"{year + 2}-{start_month:02d}-01"},
                "studyFirstPostDateStruct": {"date": f"{year}-{max(start_month - 1, 1):02d}-15"},
                "lastUpdatePostDateStruct": {"date": f"{year + 2}-{start_month:02d}-20"},
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": [phase],
                "enrollmentInfo": {"count": enrollment, "type": "ACTUAL"},
                "designInfo": {
                    "allocation": "RANDOMIZED" if randomized else "NON_RANDOMIZED",
                    "interventionModel": "PARALLEL",
                    "primaryPurpose": ["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE"][index % 3],
                    "maskingInfo": {"masking": masked},
                },
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": f"Demo Sponsor {index % 18:02d}", "class": "INDUSTRY" if industry else "OTHER"}
            },
            "conditionsModule": {"conditions": [condition, f"Comorbidity {index % 7}"]},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": f"Intervention {index % 22}"}]},
            "contactsLocationsModule": {"locations": [{"country": country} for country in countries]},
            "eligibilityModule": {
                "sex": "ALL", "minimumAge": "18 Years", "maximumAge": f"{65 + index % 20} Years",
                "healthyVolunteers": False,
            },
        },
        "hasResults": status == "COMPLETED" and index % 3 != 0,
    }


def build_demo_studies(condition: str = "type 2 diabetes", size: int = 240) -> list[dict[str, Any]]:
    """Return a reproducible source-shaped fallback corpus."""
    rng = np.random.default_rng(20260806)
    return [_demo_study(index, condition, rng) for index in range(max(int(size), 80))]


def _request_payload(params: dict[str, Any], attempts: int = 2) -> tuple[requests.Response, dict[str, Any]]:
    """Execute a small retry budget for transport and explicitly transient HTTP failures."""
    last_error: Exception | None = None
    for attempt in range(max(int(attempts), 1)):
        try:
            response = requests.get(API_URL, params=params, timeout=(4, 25), headers={"Accept": "application/json"})
            if response.status_code in RETRYABLE_STATUS and attempt + 1 < attempts:
                wait = min(float(response.headers.get("Retry-After", 0.20 * (2 ** attempt))), 2.0)
                time.sleep(max(wait, 0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("ClinicalTrials.gov returned a non-object JSON payload")
            return response, payload
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.20 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("Request retry budget exhausted") from last_error


def fetch_studies(condition: str, batch_size: int = 240) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a bounded terminal-study snapshot, falling back safely on any source failure."""
    clean_condition = safe_condition(condition) or CONDITION_PRESETS["Type 2 diabetes"]
    size = int(np.clip(batch_size, 80, 500))
    params = {
        "query.cond": clean_condition,
        "filter.overallStatus": STATUS_FILTER,
        "pageSize": size,
        "countTotal": "true",
        "format": "json",
        "sort": "StudyFirstPostDate:desc",
    }
    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        response, payload = _request_payload(params)
        studies = payload.get("studies", [])
        if not isinstance(studies, list) or len(studies) < 50:
            raise ValueError("The source returned too few terminal studies for a stable model demonstration")
        return studies, {
            "mode": "live",
            "retrieved_at": retrieved_at,
            "condition": clean_condition,
            "source_matches": int(payload.get("totalCount", len(studies))),
            "requested_rows": size,
            "returned_rows": len(studies),
            "source_url": response.url,
            "fallback_reason": "",
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        studies = build_demo_studies(clean_condition, size)
        return studies, {
            "mode": "demo",
            "retrieved_at": retrieved_at,
            "condition": clean_condition,
            "source_matches": len(studies),
            "requested_rows": size,
            "returned_rows": len(studies),
            "source_url": API_URL,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }
