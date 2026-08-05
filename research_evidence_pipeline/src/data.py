"""Europe PMC ingestion and deterministic sample-data fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd
import requests


API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
API_DOCS_URL = "https://europepmc.org/RestfulWebService"
SEARCH_HELP_URL = "https://europepmc.org/searchsyntax"
ABOUT_URL = "https://europepmc.org/About"
USER_AGENT = "HendrikDataPortfolio/1.0 (https://github.com/hendrikpw/projects)"

QUERY_PRESETS = {
    "Clinical AI": '("artificial intelligence" OR "machine learning") AND HAS_ABSTRACT:Y',
    "Antimicrobial resistance": '"antimicrobial resistance" AND HAS_ABSTRACT:Y',
    "Climate & health": '("climate change" AND health) AND HAS_ABSTRACT:Y',
    "Rare-disease diagnostics": '("rare disease" AND diagnos*) AND HAS_ABSTRACT:Y',
    "Digital mental health": '("digital mental health" OR "mental health app") AND HAS_ABSTRACT:Y',
}


def safe_custom_query(value: str) -> str:
    """Reduce a free-text query to a bounded Europe PMC keyword expression."""
    clean = re.sub(r"[^\w\s\-]", " ", value, flags=re.UNICODE)
    terms = [term for term in clean.split() if len(term) > 1][:8]
    if not terms:
        return QUERY_PRESETS["Clinical AI"]
    return "(" + " AND ".join(f'"{term}"' for term in terms) + ") AND HAS_ABSTRACT:Y"


def fetch_publications(query: str, page_size: int = 100, timeout: int = 35) -> tuple[list[dict], dict]:
    """Fetch a bounded, recent publication batch from Europe PMC without credentials."""
    bounded_size = max(25, min(int(page_size), 250))
    today = datetime.now(timezone.utc).date().isoformat()
    params = {
        "query": f"({query}) AND FIRST_PDATE:[2022-01-01 TO {today}] sort_date:y",
        "format": "json",
        "resultType": "core",
        "pageSize": bounded_size,
    }
    response = requests.get(
        API_URL,
        params=params,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("resultList", {}).get("result", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Europe PMC returned no publication records")
    return rows, {
        "mode": "live",
        "source_url": response.url,
        "query": query,
        "hit_count": int(payload.get("hitCount") or 0),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api_version": payload.get("version"),
    }


def build_demo_records() -> list[dict]:
    """Build stable synthetic metadata that exercises the complete pipeline and retriever."""
    themes = [
        ("sepsis", "early warning", "gradient boosting", "intensive care"),
        ("retinal disease", "image screening", "convolutional model", "ophthalmology"),
        ("antibiotic resistance", "susceptibility prediction", "random forest", "microbiology"),
        ("depression", "symptom monitoring", "language model", "digital health"),
        ("rare disease", "phenotype matching", "knowledge graph", "genomics"),
        ("heart failure", "readmission risk", "calibrated classifier", "cardiology"),
        ("climate exposure", "heat-risk forecasting", "time-series model", "public health"),
        ("drug discovery", "molecular ranking", "graph neural network", "pharmacology"),
    ]
    journals = ["Open Clinical Systems", "Computational Medicine Reports", "Evidence AI Review"]
    rows: list[dict] = []
    for index in range(48):
        disease, task, method, field = themes[index % len(themes)]
        year = 2022 + index % 5
        rows.append(
            {
                "id": f"DEMO{index + 1:04d}",
                "source": "DEMO",
                "doi": f"10.0000/demo.{index + 1}",
                "title": f"{method.title()} for {task} in {disease}",
                "authorString": f"Researcher {index % 11 + 1} et al.",
                "journalTitle": journals[index % len(journals)],
                "pubYear": str(year),
                "firstPublicationDate": f"{year}-{index % 12 + 1:02d}-{index % 27 + 1:02d}",
                "citedByCount": str((index * 7) % 83),
                "isOpenAccess": "Y" if index % 3 else "N",
                "inEPMC": "Y" if index % 2 else "N",
                "abstractText": (
                    f"This reproducible study evaluates a {method} for {task} in {disease}. "
                    f"The analysis uses a multi-centre {field} cohort with temporal validation. "
                    f"Results report discrimination, calibration and subgroup performance, while the authors "
                    f"highlight dataset shift and external validation as important limitations."
                ),
                "publicationTypes": "research article; journal article",
                "meshHeadingList": {
                    "meshHeading": [
                        {"descriptorName": disease.title()},
                        {"descriptorName": method.title()},
                        {"descriptorName": field.title()},
                    ]
                },
                "fullTextUrlList": {"fullTextUrl": []},
            }
        )
    return rows


def load_records(query: str, page_size: int = 100) -> tuple[list[dict], dict]:
    """Return live records or a clearly labelled synthetic fallback."""
    try:
        return fetch_publications(query, page_size=page_size)
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        records = build_demo_records()
        return records, {
            "mode": "demo",
            "source_url": API_URL,
            "query": query,
            "hit_count": len(records),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "api_version": None,
            "fallback_reason": f"{type(exc).__name__}: {str(exc)[:180]}",
        }
