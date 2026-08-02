"""GBIF taxonomy and occurrence ingestion with provenance-aware fallback data."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


API_ROOT = "https://api.gbif.org/v1"
OCCURRENCE_DOCS_URL = "https://techdocs.gbif.org/en/openapi/v1/occurrence"
SPECIES_DOCS_URL = "https://techdocs.gbif.org/en/openapi/v1/species"
TERMS_URL = "https://www.gbif.org/terms"
CITATION_URL = "https://www.gbif.org/citation-guidelines"
USER_AGENT = "HendrikDataPortfolio/1.0 (https://github.com/hendrikpw/projects)"

SPECIES = {
    "Erinaceus europaeus": "European hedgehog",
    "Lutra lutra": "Eurasian otter",
    "Lynx lynx": "Eurasian lynx",
    "Canis lupus": "Grey wolf",
    "Vulpes vulpes": "Red fox",
    "Ursus arctos": "Brown bear",
    "Alcedo atthis": "Common kingfisher",
    "Ciconia ciconia": "White stork",
    "Bombus terrestris": "Buff-tailed bumblebee",
    "Vanessa atalanta": "Red admiral",
    "Salamandra salamandra": "Fire salamander",
    "Rana temporaria": "Common frog",
}


def resolve_species(name: str, timeout: int = 25) -> dict:
    """Resolve one scientific name against the GBIF taxonomic backbone."""
    response = requests.get(
        f"{API_ROOT}/species/match",
        params={"name": name, "strict": "true"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    key = payload.get("usageKey") or payload.get("speciesKey")
    if not key or payload.get("matchType") == "NONE":
        raise ValueError(f"GBIF could not resolve {name}")
    return {
        "query_name": name,
        "taxon_key": int(key),
        "accepted_name": str(payload.get("scientificName") or name),
        "canonical_name": str(payload.get("canonicalName") or name),
        "rank": str(payload.get("rank") or "unknown"),
        "status": str(payload.get("status") or "unknown"),
        "confidence": int(payload.get("confidence") or 0),
        "match_type": str(payload.get("matchType") or "unknown"),
        "family": str(payload.get("family") or "unknown"),
        "class": str(payload.get("class") or "unknown"),
    }


def _license_label(value: object) -> str:
    text = str(value or "").lower()
    if "by-nc" in text:
        return "CC BY-NC"
    if "/by/" in text:
        return "CC BY"
    if "zero" in text or "cc0" in text:
        return "CC0"
    return "Other / not recorded"


def parse_occurrences(records: list[dict], query_name: str, is_demo: bool = False) -> pd.DataFrame:
    """Flatten selected Darwin Core and GBIF interpretation fields."""
    rows = []
    for item in records:
        issues = item.get("issues") or []
        rows.append(
            {
                "record_id": str(item.get("gbifID") or item.get("key") or ""),
                "occurrence_key": str(item.get("key") or ""),
                "query_name": query_name,
                "taxon_key": pd.to_numeric(item.get("taxonKey"), errors="coerce"),
                "scientific_name": str(item.get("scientificName") or query_name),
                "species": str(item.get("species") or query_name),
                "latitude": pd.to_numeric(item.get("decimalLatitude"), errors="coerce"),
                "longitude": pd.to_numeric(item.get("decimalLongitude"), errors="coerce"),
                "event_date": pd.to_datetime(item.get("eventDate"), errors="coerce", utc=True),
                "year": pd.to_numeric(item.get("year"), errors="coerce"),
                "month": pd.to_numeric(item.get("month"), errors="coerce"),
                "country": str(item.get("country") or "Unknown country"),
                "country_code": str(item.get("countryCode") or ""),
                "locality": str(item.get("locality") or item.get("stateProvince") or ""),
                "basis_of_record": str(item.get("basisOfRecord") or "UNKNOWN"),
                "dataset_title": str(item.get("datasetTitle") or item.get("datasetName") or "Unknown dataset"),
                "institution_code": str(item.get("institutionCode") or ""),
                "coordinate_uncertainty_m": pd.to_numeric(item.get("coordinateUncertaintyInMeters"), errors="coerce"),
                "issues_count": len(issues),
                "issues": ", ".join(str(issue) for issue in issues),
                "license": _license_label(item.get("license")),
                "license_url": str(item.get("license") or ""),
                "dataset_key": str(item.get("datasetKey") or ""),
                "publisher_key": str(item.get("publishingOrgKey") or ""),
                "recorded_by": str(item.get("recordedBy") or ""),
                "media_count": len(item.get("media") or []),
                "is_demo": bool(is_demo),
            }
        )
    frame = pd.DataFrame(rows)
    required = {"record_id", "query_name", "latitude", "longitude", "basis_of_record", "license"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing occurrence fields: {sorted(missing)}")
    frame = frame[
        frame["record_id"].ne("")
        & frame["latitude"].between(-90, 90)
        & frame["longitude"].between(-180, 180)
    ].copy()
    frame = frame.sort_values("event_date", ascending=False).drop_duplicates("record_id")
    if frame.empty:
        raise ValueError("No valid georeferenced occurrence records remained")
    return frame.reset_index(drop=True)


def _facet_rows(payload: dict, query_name: str) -> list[dict]:
    rows = []
    for facet in payload.get("facets") or []:
        field = str(facet.get("field") or "")
        for item in facet.get("counts") or []:
            rows.append(
                {
                    "query_name": query_name,
                    "field": field,
                    "value": str(item.get("name") or "Unknown"),
                    "count": int(item.get("count") or 0),
                }
            )
    return rows


def fetch_species_occurrences(
    resolution: dict,
    start_year: int = 2018,
    end_year: int = 2026,
    sample_size: int = 600,
    timeout: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fetch a bounded European coordinate sample plus full-query facet counts."""
    params: list[tuple[str, object]] = [
        ("taxon_key", resolution["taxon_key"]),
        ("has_coordinate", "true"),
        ("occurrence_status", "present"),
        ("continent", "EUROPE"),
        ("year", f"{start_year},{end_year}"),
        ("limit", min(300, sample_size)),
        ("facet", "year"),
        ("facet", "month"),
        ("facet", "country"),
        ("facet", "basis_of_record"),
        ("facetLimit", 60),
    ]
    response = requests.get(
        f"{API_ROOT}/occurrence/search",
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    total = int(payload.get("count") or 0)
    records = list(payload.get("results") or [])
    remaining = min(max(int(sample_size) - len(records), 0), 300)
    if remaining and total > len(records):
        offset = min(max(total // 2, len(records)), 99_700)
        extra_params = dict(params)
        extra_params.update({"offset": offset, "limit": remaining})
        extra_params.pop("facet", None)
        extra = requests.get(
            f"{API_ROOT}/occurrence/search",
            params=extra_params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
        )
        extra.raise_for_status()
        records.extend(extra.json().get("results") or [])
    occurrences = parse_occurrences(records, resolution["query_name"])
    facets = pd.DataFrame(_facet_rows(payload, resolution["query_name"]))
    return occurrences, facets, {
        **resolution,
        "indexed_records": total,
        "sample_records": len(occurrences),
        "source_url": response.url,
    }


def fetch_live_data(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Resolve and retrieve two or three species with shared provenance metadata."""
    occurrence_frames = []
    facet_frames = []
    species_metadata = []
    for name in names:
        resolution = resolve_species(name)
        occurrences, facets, metadata = fetch_species_occurrences(resolution)
        occurrence_frames.append(occurrences)
        facet_frames.append(facets)
        species_metadata.append(metadata)
    data = pd.concat(occurrence_frames, ignore_index=True)
    facets = pd.concat(facet_frames, ignore_index=True)
    return data, facets, {
        "mode": "live",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "species": species_metadata,
        "sample_records": len(data),
        "indexed_records": sum(item["indexed_records"] for item in species_metadata),
        "start_year": 2018,
        "end_year": 2026,
    }


def build_demo_data(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create deterministic Europe-shaped occurrence samples and facet counts."""
    centers = [(52.5, 9.0), (48.7, 14.0), (46.2, 6.0), (55.5, 24.0), (43.5, -1.0)]
    all_frames = []
    facet_rows = []
    metadata = []
    for position, name in enumerate(names):
        rng = np.random.default_rng(20260802 + sum(ord(char) for char in name))
        records = []
        for index in range(520):
            center = centers[(index + position) % len(centers)]
            year = int(rng.integers(2018, 2027))
            month = int(np.clip(round(rng.normal(7 if position % 2 == 0 else 5, 2.2)), 1, 12))
            day = int(rng.integers(1, 28))
            license_url = rng.choice(
                [
                    "http://creativecommons.org/publicdomain/zero/1.0/legalcode",
                    "http://creativecommons.org/licenses/by/4.0/legalcode",
                    "http://creativecommons.org/licenses/by-nc/4.0/legalcode",
                ],
                p=[0.3, 0.55, 0.15],
            )
            records.append(
                {
                    "gbifID": f"demo-{position}-{index}",
                    "key": f"demo-{position}-{index}",
                    "taxonKey": 9_000_000 + position,
                    "scientificName": name,
                    "species": name,
                    "decimalLatitude": center[0] + rng.normal(0, 2.1),
                    "decimalLongitude": center[1] + rng.normal(0, 3.0),
                    "eventDate": f"{year}-{month:02d}-{day:02d}",
                    "year": year,
                    "month": month,
                    "country": rng.choice(["Germany", "France", "Poland", "Spain", "Sweden"]),
                    "countryCode": rng.choice(["DE", "FR", "PL", "ES", "SE"]),
                    "basisOfRecord": rng.choice(["HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "PRESERVED_SPECIMEN"], p=[0.86, 0.1, 0.04]),
                    "datasetName": f"Synthetic biodiversity dataset {position + 1}",
                    "coordinateUncertaintyInMeters": float(rng.lognormal(4.5, 1.0)),
                    "issues": [] if rng.random() > 0.35 else ["SYNTHETIC_QUALITY_FLAG"],
                    "license": license_url,
                    "datasetKey": f"demo-dataset-{position}",
                    "publishingOrgKey": "demo-publisher",
                }
            )
        frame = parse_occurrences(records, name, is_demo=True)
        all_frames.append(frame)
        for field, column in [("YEAR", "year"), ("MONTH", "month"), ("COUNTRY", "country"), ("BASIS_OF_RECORD", "basis_of_record")]:
            for value, count in frame[column].value_counts().items():
                facet_rows.append({"query_name": name, "field": field, "value": str(value), "count": int(count * 18)})
        metadata.append(
            {
                "query_name": name,
                "taxon_key": 9_000_000 + position,
                "accepted_name": name,
                "canonical_name": name,
                "rank": "SPECIES",
                "status": "SYNTHETIC",
                "confidence": 100,
                "match_type": "DEMO",
                "family": "Synthetic family",
                "class": "Synthetic class",
                "indexed_records": len(frame) * 18,
                "sample_records": len(frame),
            }
        )
    data = pd.concat(all_frames, ignore_index=True)
    return data, pd.DataFrame(facet_rows), {
        "mode": "demo",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "species": metadata,
        "sample_records": len(data),
        "indexed_records": sum(item["indexed_records"] for item in metadata),
        "start_year": 2018,
        "end_year": 2026,
    }


def load_data(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return live GBIF data or a clearly labelled synthetic fallback."""
    try:
        return fetch_live_data(names)
    except (requests.RequestException, ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        data, facets, metadata = build_demo_data(names)
        metadata["fallback_reason"] = type(exc).__name__
        return data, facets, metadata
