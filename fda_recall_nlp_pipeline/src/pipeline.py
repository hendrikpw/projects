"""Idempotent recall pipeline with contracts, quarantine and run lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from fda_recall_nlp_pipeline.src.data import CLASSES, fetch_recalls


REQUIRED_COLUMNS = {
    "record_id", "domain", "recall_number", "report_date", "classification",
    "recalling_firm", "product_description", "reason_for_recall", "status",
}
MODEL_COLUMNS = ["record_id", "domain", "report_date", "classification", "document_text", "record_url"]


@dataclass(frozen=True)
class PipelineBundle:
    snapshot: pd.DataFrame
    validated: pd.DataFrame
    quarantine: pd.DataFrame
    features: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]


def stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _date(value: object) -> pd.Timestamp:
    text = _text(value)
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def snapshot_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for position, record in enumerate(records):
        rows.append({
            "ingest_position": position,
            "source_domain": _text(record.get("_domain")).lower(),
            "recall_number_hint": _text(record.get("recall_number")),
            "payload_hash": stable_hash(record),
            "raw_payload": record,
        })
    return pd.DataFrame(rows)


def normalize_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Map three source endpoints into one typed record contract."""
    rows: list[dict[str, Any]] = []
    for payload in snapshot.get("raw_payload", pd.Series(dtype=object)):
        if not isinstance(payload, dict):
            rows.append({"contract_error": "payload_not_object"})
            continue
        domain = _text(payload.get("_domain")).lower()
        recall_number = _text(payload.get("recall_number"))
        classification = {
            "CLASS I": "Class I",
            "CLASS II": "Class II",
            "CLASS III": "Class III",
        }.get(_text(payload.get("classification")).upper(), _text(payload.get("classification")))
        rows.append({
            "record_id": f"{domain}:{recall_number}",
            "domain": domain,
            "recall_number": recall_number,
            "event_id": _text(payload.get("event_id")),
            "report_date": _date(payload.get("report_date")),
            "recall_initiation_date": _date(payload.get("recall_initiation_date")),
            "termination_date": _date(payload.get("termination_date")),
            "classification": classification,
            "status": _text(payload.get("status")) or "Unknown",
            "recalling_firm": _text(payload.get("recalling_firm")),
            "product_description": _text(payload.get("product_description")),
            "reason_for_recall": _text(payload.get("reason_for_recall")),
            "product_quantity": _text(payload.get("product_quantity")),
            "distribution_pattern": _text(payload.get("distribution_pattern")),
            "country": _text(payload.get("country")),
            "state": _text(payload.get("state")),
            "voluntary_mandated": _text(payload.get("voluntary_mandated")),
            "initial_firm_notification": _text(payload.get("initial_firm_notification")),
            "contract_error": "",
        })
    return pd.DataFrame(rows)


def enforce_contract(normalized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate usable records from explicit, reason-coded quarantine rows."""
    frame = normalized.copy()
    if frame.empty:
        raise ValueError("No source records could be normalized")
    current_year = datetime.now(timezone.utc).year
    errors: list[str] = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        reasons: list[str] = []
        record_id = _text(row.get("record_id"))
        if not record_id or record_id.endswith(":"):
            reasons.append("missing_record_id")
        elif record_id in seen:
            reasons.append("duplicate_record_id")
        seen.add(record_id)
        if row.get("domain") not in {"food", "drug", "device"}:
            reasons.append("unsupported_domain")
        if row.get("classification") not in CLASSES:
            reasons.append("invalid_classification")
        date = row.get("report_date")
        if pd.isna(date) or not 2004 <= pd.Timestamp(date).year <= current_year + 1:
            reasons.append("invalid_report_date")
        if len(_text(row.get("product_description"))) < 8:
            reasons.append("product_text_too_short")
        if len(_text(row.get("reason_for_recall"))) < 15:
            reasons.append("reason_text_too_short")
        errors.append("|".join(reasons))
    frame["contract_error"] = errors
    quarantine = frame[frame["contract_error"].ne("")].copy().reset_index(drop=True)
    valid = frame[frame["contract_error"].eq("")].drop(columns="contract_error").copy()
    missing = REQUIRED_COLUMNS.difference(valid.columns)
    if missing:
        raise ValueError(f"Recall contract missing columns: {sorted(missing)}")
    if len(valid) < 60 or valid["classification"].nunique() != 3:
        raise ValueError("Too few valid records or recall classes for model evaluation")
    return valid.sort_values(["report_date", "record_id"]).reset_index(drop=True), quarantine


def feature_view(validated: pd.DataFrame) -> pd.DataFrame:
    """Build a label-separated NLP document view without outcome leakage."""
    frame = validated.copy()
    frame["document_text"] = (
        "domain " + frame["domain"] + ". product " + frame["product_description"]
        + ". reason " + frame["reason_for_recall"] + ". firm " + frame["recalling_firm"]
    ).map(_text)
    frame["record_url"] = "https://www.accessdata.fda.gov/scripts/ires/?Product=" + frame["recall_number"]
    return frame[MODEL_COLUMNS].copy()


def quality_report(
    snapshot: pd.DataFrame,
    validated: pd.DataFrame,
    quarantine: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    retention = len(validated) / max(len(snapshot), 1)
    checks = [
        ("Payload hashes unique", snapshot["payload_hash"].is_unique, f"{snapshot['payload_hash'].nunique():,}/{len(snapshot):,}"),
        ("Record IDs unique", validated["record_id"].is_unique, f"{validated['record_id'].nunique():,}/{len(validated):,}"),
        ("Three-domain contract", validated["domain"].isin(["food", "drug", "device"]).all(), ", ".join(sorted(validated["domain"].unique()))),
        ("Three recall classes", set(validated["classification"]) == set(CLASSES), ", ".join(sorted(validated["classification"].unique()))),
        ("Report dates valid", validated["report_date"].notna().all(), f"{validated['report_date'].min().date()}–{validated['report_date'].max().date()}"),
        ("Reason text complete", validated["reason_for_recall"].str.len().ge(15).all(), f"median {validated['reason_for_recall'].str.len().median():.0f} chars"),
        ("Feature view reconciles", len(features) == len(validated), f"{len(features):,}/{len(validated):,}"),
        ("No label text leakage", not features["document_text"].str.contains(r"classification\s+class\s+(?:i|ii|iii)", case=False, regex=True).any(), "classification excluded from document assembly"),
        ("Contract retention", retention >= 0.80, f"{retention:.1%}; {len(quarantine)} quarantined"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _event(stage: str, started: float, inputs: int, outputs: int, digest: str, status: str = "passed") -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "input_rows": int(inputs),
        "output_rows": int(outputs),
        "rejected_rows": int(max(inputs - outputs, 0)),
        "duration_ms": round((perf_counter() - started) * 1000, 2),
        "content_hash": digest[:12],
    }


def run_pipeline(domains: list[str], snapshot_size: int = 270) -> PipelineBundle:
    events: list[dict[str, Any]] = []
    started = perf_counter()
    records, metadata = fetch_recalls(domains, snapshot_size)
    raw_hash = stable_hash(records)
    events.append(_event("Extract / stratified API", started, len(records), len(records), raw_hash))

    started = perf_counter()
    snapshot = snapshot_table(records)
    events.append(_event("Content-addressed snapshot", started, len(records), len(snapshot), stable_hash(snapshot["payload_hash"].tolist())))

    started = perf_counter()
    normalized = normalize_snapshot(snapshot)
    validated, quarantine = enforce_contract(normalized)
    contract_hash = stable_hash(validated["record_id"].tolist())
    events.append(_event("Typed contract + quarantine", started, len(snapshot), len(validated), contract_hash))

    started = perf_counter()
    features = feature_view(validated)
    feature_hash = stable_hash(features.to_dict(orient="records"))
    events.append(_event("Leakage-safe NLP view", started, len(validated), len(features), feature_hash))

    quality = quality_report(snapshot, validated, quarantine, features)
    run_id = stable_hash({"domains": metadata["domains"], "raw_hash": raw_hash, "feature_hash": feature_hash})[:16]
    metadata = {
        **metadata,
        "run_id": run_id,
        "raw_hash": raw_hash,
        "contract_hash": contract_hash,
        "feature_hash": feature_hash,
        "snapshot_rows": len(snapshot),
        "validated_rows": len(validated),
        "quarantined_rows": len(quarantine),
        "quality_pass_rate": float(quality["passed"].mean()),
    }
    return PipelineBundle(snapshot, validated, quarantine, features, pd.DataFrame(events), quality, metadata)
