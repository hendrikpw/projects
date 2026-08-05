"""Deterministic Bronze/Silver/Gold publication pipeline with observability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any

import pandas as pd

from research_evidence_pipeline.src.data import load_records


REQUIRED_SILVER_COLUMNS = {
    "record_id",
    "source",
    "title",
    "abstract",
    "publication_date",
    "publication_year",
    "cited_by_count",
    "is_open_access",
}


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible content with stable ordering for idempotency checks."""
    canonical = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _list_values(container: object, key: str, value_key: str) -> list[str]:
    if not isinstance(container, dict):
        return []
    values = container.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item.get(value_key)).strip() for item in values if isinstance(item, dict) and item.get(value_key)]


def bronze_table(records: list[dict]) -> pd.DataFrame:
    """Create one immutable-looking raw row per API record with payload hashes."""
    rows = []
    for position, record in enumerate(records):
        rows.append(
            {
                "ingest_position": position,
                "source_record_id": str(record.get("id") or ""),
                "source": str(record.get("source") or ""),
                "payload_hash": stable_hash(record),
                "raw_payload": record,
            }
        )
    return pd.DataFrame(rows)


def silver_table(bronze: pd.DataFrame) -> pd.DataFrame:
    """Normalize the raw payloads and enforce the publication data contract."""
    rows: list[dict] = []
    for payload in bronze.get("raw_payload", pd.Series(dtype=object)):
        if not isinstance(payload, dict):
            continue
        mesh = _list_values(payload.get("meshHeadingList"), "meshHeading", "descriptorName")
        full_text = _list_values(payload.get("fullTextUrlList"), "fullTextUrl", "url")
        source = str(payload.get("source") or "").strip()
        external_id = str(payload.get("id") or "").strip()
        rows.append(
            {
                "record_id": f"{source}:{external_id}",
                "source": source,
                "external_id": external_id,
                "doi": str(payload.get("doi") or "").strip().lower(),
                "title": str(payload.get("title") or "").strip(),
                "abstract": str(payload.get("abstractText") or "").strip(),
                "authors": str(payload.get("authorString") or "Unknown authors").strip(),
                "journal": str(payload.get("journalTitle") or "Unknown venue").strip(),
                "publication_date": pd.to_datetime(payload.get("firstPublicationDate"), errors="coerce"),
                "publication_year": pd.to_numeric(payload.get("pubYear"), errors="coerce"),
                "cited_by_count": pd.to_numeric(payload.get("citedByCount"), errors="coerce"),
                "is_open_access": str(payload.get("isOpenAccess") or "N").upper() == "Y",
                "in_epmc": str(payload.get("inEPMC") or "N").upper() == "Y",
                "publication_types": str(payload.get("publicationTypes") or "").strip(),
                "mesh_terms": "; ".join(dict.fromkeys(mesh)),
                "full_text_url": full_text[0] if full_text else "",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No records could be normalized into the silver contract")
    frame["publication_year"] = frame["publication_year"].astype("Int64")
    frame["cited_by_count"] = frame["cited_by_count"].fillna(0).clip(lower=0).astype(int)
    frame = frame.drop_duplicates("record_id", keep="first")
    current_year = pd.Timestamp.now(tz="UTC").year
    frame = frame[
        frame["record_id"].str.len().gt(2)
        & frame["title"].str.len().ge(8)
        & frame["abstract"].str.len().ge(80)
        & frame["publication_year"].between(1900, current_year)
    ].copy()
    if frame.empty:
        raise ValueError("No publications passed the silver quality contract")
    missing = REQUIRED_SILVER_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Silver contract is missing columns: {sorted(missing)}")
    return frame.sort_values(["publication_date", "record_id"], ascending=[False, True]).reset_index(drop=True)


def gold_table(silver: pd.DataFrame) -> pd.DataFrame:
    """Create AI-ready features without modifying the validated silver layer."""
    frame = silver.copy()
    frame["abstract_words"] = frame["abstract"].str.split().str.len().astype(int)
    frame["document_text"] = (
        frame["title"].fillna("")
        + ". "
        + frame["abstract"].fillna("")
        + " "
        + frame["mesh_terms"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["citation_band"] = pd.cut(
        frame["cited_by_count"],
        bins=[-1, 0, 4, 19, float("inf")],
        labels=["Uncited", "1–4", "5–19", "20+"],
    ).astype(str)
    frame["epmc_url"] = "https://europepmc.org/article/" + frame["source"] + "/" + frame["external_id"]
    frame["document_hash"] = frame.apply(
        lambda row: stable_hash(
            {
                "record_id": row["record_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "mesh_terms": row["mesh_terms"],
            }
        ),
        axis=1,
    )
    return frame


def quality_report(bronze: pd.DataFrame, silver: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Return explicit contract and quality checks with denominators."""
    current_year = pd.Timestamp.now(tz="UTC").year
    checks = [
        ("Unique record IDs", silver["record_id"].nunique() == len(silver), f"{silver['record_id'].nunique():,}/{len(silver):,}"),
        ("Required fields complete", not silver[["record_id", "title", "abstract"]].isna().any().any(), "ID, title, abstract"),
        ("Abstract minimum", silver["abstract"].str.len().ge(80).all(), f"min {silver['abstract'].str.len().min():,} chars"),
        ("Valid publication year", silver["publication_year"].between(1900, current_year).all(), f"{silver['publication_year'].min()}–{silver['publication_year'].max()}"),
        ("Non-negative citations", silver["cited_by_count"].ge(0).all(), f"min {silver['cited_by_count'].min()}"),
        ("Gold row reconciliation", len(gold) == len(silver), f"{len(gold):,}/{len(silver):,}"),
        ("Deterministic document hashes", gold["document_hash"].nunique() == len(gold), f"{gold['document_hash'].nunique():,}/{len(gold):,}"),
        ("Bronze-to-silver retention", len(silver) / max(len(bronze), 1) >= 0.50, f"{len(silver) / max(len(bronze), 1):.1%}"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _event(stage: str, started: float, input_rows: int, output_rows: int, content_hash: str, status: str = "passed") -> dict:
    return {
        "stage": stage,
        "status": status,
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "dropped_rows": int(max(input_rows - output_rows, 0)),
        "duration_ms": round((perf_counter() - started) * 1000, 2),
        "content_hash": content_hash[:12],
    }


def run_pipeline(query: str, page_size: int = 100) -> PipelineBundle:
    """Run all pipeline stages and return data products plus an observable manifest."""
    events = []
    started = perf_counter()
    records, metadata = load_records(query, page_size=page_size)
    raw_hash = stable_hash(records)
    events.append(_event("Extract / API", started, len(records), len(records), raw_hash))

    started = perf_counter()
    bronze = bronze_table(records)
    events.append(_event("Bronze / raw", started, len(records), len(bronze), stable_hash(bronze["payload_hash"].tolist())))

    started = perf_counter()
    silver = silver_table(bronze)
    events.append(_event("Silver / contract", started, len(bronze), len(silver), stable_hash(silver["record_id"].tolist())))

    started = perf_counter()
    gold = gold_table(silver)
    events.append(_event("Gold / AI-ready", started, len(silver), len(gold), stable_hash(gold["document_hash"].tolist())))

    quality = quality_report(bronze, silver, gold)
    run_id = stable_hash({"query": query, "raw_hash": raw_hash, "gold_hashes": gold["document_hash"].tolist()})[:16]
    metadata = {
        **metadata,
        "run_id": run_id,
        "raw_hash": raw_hash,
        "bronze_rows": len(bronze),
        "silver_rows": len(silver),
        "gold_rows": len(gold),
        "quality_pass_rate": float(quality["passed"].mean()),
    }
    return PipelineBundle(bronze, silver, gold, pd.DataFrame(events), quality, metadata)
