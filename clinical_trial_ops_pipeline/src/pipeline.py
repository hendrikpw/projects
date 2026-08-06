"""Idempotent clinical-trial snapshot pipeline and operational observability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from clinical_trial_ops_pipeline.src.data import fetch_studies


TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}
DISCONTINUED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
REQUIRED_COLUMNS = {
    "nct_id", "title", "overall_status", "study_type", "phase", "enrollment",
    "first_post_date", "sponsor_class", "condition_count", "country_count",
}
FEATURE_COLUMNS = [
    "phase", "study_type", "sponsor_class", "allocation", "masking", "primary_purpose",
    "enrollment_log", "condition_count", "intervention_count", "country_count",
    "minimum_age", "age_span", "healthy_volunteers",
]
CATEGORICAL_FEATURES = ["phase", "study_type", "sponsor_class", "allocation", "masking", "primary_purpose"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]


@dataclass(frozen=True)
class PipelineBundle:
    snapshot: pd.DataFrame
    validated: pd.DataFrame
    features: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]


def stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nested(value: object, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _age_years(value: object) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(Year|Month|Week|Day)", str(value), flags=re.I)
    if not match:
        return np.nan
    number = float(match.group(1))
    unit = match.group(2).lower()
    return number if unit == "year" else number / 12 if unit == "month" else number / 52 if unit == "week" else number / 365


def snapshot_table(studies: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a content-addressed immutable snapshot view of source records."""
    rows = []
    for position, study in enumerate(studies):
        nct_id = str(_nested(study, "protocolSection", "identificationModule", "nctId", default="")).strip()
        rows.append({"ingest_position": position, "nct_id_hint": nct_id, "payload_hash": stable_hash(study), "raw_payload": study})
    return pd.DataFrame(rows)


def normalize_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Normalize nested records, enforce types and reject contract violations."""
    rows: list[dict[str, Any]] = []
    for payload in snapshot.get("raw_payload", pd.Series(dtype=object)):
        if not isinstance(payload, dict):
            continue
        protocol = payload.get("protocolSection", {})
        design = _nested(protocol, "designModule", default={})
        design_info = _nested(design, "designInfo", default={})
        eligibility = _nested(protocol, "eligibilityModule", default={})
        locations = _nested(protocol, "contactsLocationsModule", "locations", default=[])
        interventions = _nested(protocol, "armsInterventionsModule", "interventions", default=[])
        conditions = _nested(protocol, "conditionsModule", "conditions", default=[])
        phases = _nested(design, "phases", default=[])
        min_age = _age_years(_nested(eligibility, "minimumAge", default=""))
        max_age = _age_years(_nested(eligibility, "maximumAge", default=""))
        countries = {str(item.get("country")) for item in locations if isinstance(item, dict) and item.get("country")}
        rows.append({
            "nct_id": str(_nested(protocol, "identificationModule", "nctId", default="")).strip(),
            "title": str(_nested(protocol, "identificationModule", "briefTitle", default="")).strip(),
            "overall_status": str(_nested(protocol, "statusModule", "overallStatus", default="")).upper(),
            "study_type": str(_nested(design, "studyType", default="UNKNOWN")).upper(),
            "phase": str(phases[0] if isinstance(phases, list) and phases else "NA").upper(),
            "enrollment": pd.to_numeric(_nested(design, "enrollmentInfo", "count"), errors="coerce"),
            "enrollment_type": str(_nested(design, "enrollmentInfo", "type", default="UNKNOWN")).upper(),
            "allocation": str(_nested(design_info, "allocation", default="NA")).upper(),
            "masking": str(_nested(design_info, "maskingInfo", "masking", default="NA")).upper(),
            "primary_purpose": str(_nested(design_info, "primaryPurpose", default="NA")).upper(),
            "sponsor_name": str(_nested(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name", default="Unknown")).strip(),
            "sponsor_class": str(_nested(protocol, "sponsorCollaboratorsModule", "leadSponsor", "class", default="UNKNOWN")).upper(),
            "condition_count": len(conditions) if isinstance(conditions, list) else 0,
            "intervention_count": len(interventions) if isinstance(interventions, list) else 0,
            "country_count": len(countries),
            "minimum_age": min_age,
            "maximum_age": max_age,
            "healthy_volunteers": bool(_nested(eligibility, "healthyVolunteers", default=False)),
            "first_post_date": pd.to_datetime(_nested(protocol, "statusModule", "studyFirstPostDateStruct", "date"), errors="coerce"),
            "start_date": pd.to_datetime(_nested(protocol, "statusModule", "startDateStruct", "date"), errors="coerce"),
            "completion_date": pd.to_datetime(_nested(protocol, "statusModule", "completionDateStruct", "date"), errors="coerce"),
            "last_update_date": pd.to_datetime(_nested(protocol, "statusModule", "lastUpdatePostDateStruct", "date"), errors="coerce"),
            "has_results": bool(payload.get("hasResults", False)),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No source records could be normalized")
    frame = frame.drop_duplicates("nct_id", keep="first")
    frame["enrollment"] = pd.to_numeric(frame["enrollment"], errors="coerce")
    frame = frame[
        frame["nct_id"].str.match(r"^NCT(?:\d{8}|DEMO\d{5})$")
        & frame["title"].str.len().ge(8)
        & frame["overall_status"].isin(TERMINAL_STATUSES)
        & frame["enrollment"].between(1, 1_000_000)
        & frame["first_post_date"].notna()
    ].copy()
    if len(frame) < 40:
        raise ValueError("Too few records passed the trial contract")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Trial contract missing columns: {sorted(missing)}")
    frame["discontinued"] = frame["overall_status"].isin(DISCONTINUED_STATUSES).astype(int)
    frame["record_url"] = "https://clinicaltrials.gov/study/" + frame["nct_id"]
    return frame.sort_values(["first_post_date", "nct_id"]).reset_index(drop=True)


def feature_view(validated: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-aware feature view using registration/design fields only."""
    frame = validated.copy()
    frame["enrollment_log"] = np.log1p(frame["enrollment"].clip(lower=0))
    frame["minimum_age"] = frame["minimum_age"].fillna(frame["minimum_age"].median()).fillna(18).clip(0, 120)
    frame["maximum_age"] = frame["maximum_age"].fillna(frame["maximum_age"].median()).fillna(80).clip(0, 125)
    frame["age_span"] = (frame["maximum_age"] - frame["minimum_age"]).clip(lower=0, upper=125)
    for column in ("condition_count", "intervention_count", "country_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).clip(lower=0)
    frame["healthy_volunteers"] = frame["healthy_volunteers"].astype(int)
    return frame[["nct_id", "title", "first_post_date", "overall_status", "discontinued", "record_url", *FEATURE_COLUMNS]].copy()


def quality_report(snapshot: pd.DataFrame, validated: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("Unique NCT IDs", validated["nct_id"].is_unique, f"{validated['nct_id'].nunique():,}/{len(validated):,}"),
        ("Terminal outcome contract", validated["overall_status"].isin(TERMINAL_STATUSES).all(), ", ".join(sorted(validated["overall_status"].unique()))),
        ("Enrollment domain", validated["enrollment"].between(1, 1_000_000).all(), f"{validated['enrollment'].min():,.0f}–{validated['enrollment'].max():,.0f}"),
        ("Posting date complete", validated["first_post_date"].notna().all(), f"{validated['first_post_date'].notna().mean():.1%}"),
        ("Feature view reconciles", len(features) == len(validated), f"{len(features):,}/{len(validated):,}"),
        ("Feature matrix finite", np.isfinite(features[NUMERIC_FEATURES].to_numpy(dtype=float)).all(), f"{len(NUMERIC_FEATURES)} numeric features"),
        ("Both model classes present", features["discontinued"].nunique() == 2, f"discontinued share {features['discontinued'].mean():.1%}"),
        ("Snapshot retention", len(validated) / max(len(snapshot), 1) >= 0.60, f"{len(validated) / max(len(snapshot), 1):.1%}"),
        ("No outcome leakage", not {"overall_status", "has_results", "completion_date"}.intersection(FEATURE_COLUMNS), "design-time fields only"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _event(stage: str, started: float, inputs: int, outputs: int, digest: str, status: str = "passed") -> dict[str, Any]:
    return {
        "stage": stage, "status": status, "input_rows": int(inputs), "output_rows": int(outputs),
        "rejected_rows": int(max(inputs - outputs, 0)), "duration_ms": round((perf_counter() - started) * 1000, 2),
        "content_hash": digest[:12],
    }


def run_pipeline(condition: str, batch_size: int = 240) -> PipelineBundle:
    events: list[dict[str, Any]] = []
    started = perf_counter()
    studies, metadata = fetch_studies(condition, batch_size)
    raw_hash = stable_hash(studies)
    events.append(_event("Extract / API", started, len(studies), len(studies), raw_hash))
    started = perf_counter()
    snapshot = snapshot_table(studies)
    events.append(_event("Content-addressed snapshot", started, len(studies), len(snapshot), stable_hash(snapshot["payload_hash"].tolist())))
    started = perf_counter()
    validated = normalize_snapshot(snapshot)
    events.append(_event("Typed contract", started, len(snapshot), len(validated), stable_hash(validated["nct_id"].tolist())))
    started = perf_counter()
    features = feature_view(validated)
    feature_hash = stable_hash(features[["nct_id", *FEATURE_COLUMNS]].to_dict(orient="records"))
    events.append(_event("Leakage-aware feature view", started, len(validated), len(features), feature_hash))
    quality = quality_report(snapshot, validated, features)
    run_id = stable_hash({"condition": metadata["condition"], "raw_hash": raw_hash, "feature_hash": feature_hash})[:16]
    metadata = {
        **metadata, "run_id": run_id, "raw_hash": raw_hash, "feature_hash": feature_hash,
        "snapshot_rows": len(snapshot), "validated_rows": len(validated), "feature_rows": len(features),
        "quality_pass_rate": float(quality["passed"].mean()),
    }
    return PipelineBundle(snapshot, validated, features, pd.DataFrame(events), quality, metadata)
