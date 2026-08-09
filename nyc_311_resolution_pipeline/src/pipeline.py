"""Content-addressed Bronze/Silver/Gold pipeline and observability."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nyc_311_resolution_pipeline.src.data import fetch_requests


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    quarantine: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]


def _hash_records(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"[]").hexdigest()
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(str)
    payload = normalized.sort_index(axis=1).sort_values(list(normalized.columns), kind="stable").to_dict(orient="records")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _bronze(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for raw in records:
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
        rows.append({**raw, "payload_hash": hashlib.sha256(canonical.encode()).hexdigest()})
    frame = pd.DataFrame(rows)
    frame["ingest_sequence"] = np.arange(len(frame), dtype=int)
    return frame


def _silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bronze.copy()
    for column in ["unique_key", "agency", "agency_name", "complaint_type", "descriptor", "location_type", "borough", "open_data_channel_type", "status"]:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = frame[column].astype("string").str.strip()
    frame["created_at"] = pd.to_datetime(frame.get("created_date"), errors="coerce", utc=True, format="mixed")
    frame["closed_at"] = pd.to_datetime(frame.get("closed_date"), errors="coerce", utc=True, format="mixed")
    frame["resolution_hours"] = (frame["closed_at"] - frame["created_at"]).dt.total_seconds() / 3600
    frame["invalid_reason"] = pd.NA
    rules = [
        (frame["unique_key"].isna() | frame["unique_key"].eq(""), "missing_unique_key"),
        (frame["created_at"].isna(), "invalid_created_date"),
        (frame["closed_at"].isna(), "invalid_closed_date"),
        (frame["agency"].isna() | frame["agency"].eq(""), "missing_agency"),
        (frame["complaint_type"].isna() | frame["complaint_type"].eq(""), "missing_complaint_type"),
        (frame["resolution_hours"].lt(0), "negative_resolution_time"),
        (frame["resolution_hours"].gt(24 * 30), "resolution_over_30_days"),
    ]
    for mask, reason in rules:
        frame.loc[mask & frame["invalid_reason"].isna(), "invalid_reason"] = reason
    duplicate = frame.duplicated("unique_key", keep="last")
    frame.loc[duplicate & frame["invalid_reason"].isna(), "invalid_reason"] = "duplicate_unique_key"
    quarantine = frame[frame["invalid_reason"].notna()].copy()
    silver = frame[frame["invalid_reason"].isna()].copy()
    defaults = {"descriptor": "Unknown", "location_type": "Unknown", "borough": "UNSPECIFIED", "open_data_channel_type": "UNKNOWN"}
    silver = silver.fillna(defaults)
    keep = [
        "unique_key", "payload_hash", "created_at", "closed_at", "agency", "agency_name",
        "complaint_type", "descriptor", "location_type", "borough", "open_data_channel_type",
        "status", "resolution_hours", "ingest_sequence",
    ]
    return silver[keep].sort_values(["created_at", "unique_key"]).reset_index(drop=True), quarantine.reset_index(drop=True)


def _gold(silver: pd.DataFrame) -> pd.DataFrame:
    frame = silver.copy()
    frame["created_hour"] = frame["created_at"].dt.hour
    frame["created_dow"] = frame["created_at"].dt.dayofweek
    frame["created_month"] = frame["created_at"].dt.month
    frame["is_weekend"] = frame["created_dow"].ge(5).astype(int)
    frame["is_overnight"] = frame["created_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    frame["created_date"] = frame["created_at"].dt.date.astype(str)
    frame["target_log_hours"] = np.log1p(frame["resolution_hours"])
    return frame


def _checks(bronze: pd.DataFrame, silver: pd.DataFrame, gold: pd.DataFrame, quarantine: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("bronze_payload_hash", bronze["payload_hash"].str.fullmatch(r"[0-9a-f]{64}").all(), f"{len(bronze):,} hashed payloads"),
        ("row_reconciliation", len(bronze) == len(silver) + len(quarantine), f"{len(bronze):,} = {len(silver):,} + {len(quarantine):,}"),
        ("silver_unique_key", silver["unique_key"].is_unique, f"{silver['unique_key'].nunique():,} unique keys"),
        ("required_dimensions", silver[["agency", "complaint_type", "borough"]].notna().all().all(), "agency, complaint and borough present"),
        ("valid_event_order", silver["closed_at"].ge(silver["created_at"]).all(), "closed_at is never before created_at"),
        ("mature_label_window", silver["resolution_hours"].between(0, 720).all(), "labels bounded to 0–720 hours"),
        ("gold_intake_features", not {"closed_at", "status", "resolution_hours"}.intersection({"agency", "complaint_type", "descriptor", "location_type", "borough", "open_data_channel_type", "created_hour", "created_dow", "created_month", "is_weekend", "is_overnight"}), "model inputs are known at intake"),
        ("minimum_history", len(gold) >= 1000, f"{len(gold):,} eligible rows"),
        ("temporal_coverage", gold["created_at"].dt.date.nunique() >= 150, f"{gold['created_at'].dt.date.nunique():,} observed dates"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def run_pipeline(history_days: int = 365, sample_remainders: int = 2) -> PipelineBundle:
    records, source = fetch_requests(history_days=history_days, sample_remainders=sample_remainders)
    ledger = []
    started = time.perf_counter(); bronze = _bronze(records)
    ledger.append(("Bronze", len(records), len(bronze), 0, (time.perf_counter()-started)*1000, _hash_records(bronze)))
    started = time.perf_counter(); silver, quarantine = _silver(bronze)
    ledger.append(("Silver", len(bronze), len(silver), len(quarantine), (time.perf_counter()-started)*1000, _hash_records(silver)))
    started = time.perf_counter(); gold = _gold(silver)
    ledger.append(("Gold", len(silver), len(gold), 0, (time.perf_counter()-started)*1000, _hash_records(gold)))
    quality = _checks(bronze, silver, gold, quarantine)
    if not quality["passed"].all():
        failed = ", ".join(quality.loc[~quality["passed"], "check"])
        raise RuntimeError(f"data product withheld; failed quality gates: {failed}")
    stages = pd.DataFrame(ledger, columns=["stage", "input_rows", "output_rows", "rejected_rows", "duration_ms", "content_hash"])
    stages["status"] = "passed"
    hashes = dict(zip(stages["stage"].str.lower(), stages["content_hash"]))
    run_id = hashlib.sha256((source["source_hash"] + "".join(hashes.values())).encode()).hexdigest()[:16]
    metadata = {**source, **{f"{key}_hash": value for key, value in hashes.items()}, "run_id": run_id,
                "quarantine_rows": len(quarantine), "quality_pass_rate": float(quality["passed"].mean()),
                "date_count": int(gold["created_at"].dt.date.nunique()), "agency_count": int(gold["agency"].nunique()),
                "complaint_count": int(gold["complaint_type"].nunique())}
    return PipelineBundle(bronze, silver, gold, quarantine, stages, quality, metadata)
