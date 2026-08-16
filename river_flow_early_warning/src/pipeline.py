"""Content-addressed Bronze/Silver/Gold streamflow pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from river_flow_early_warning.src.data import SITES, load_source, parse_rdb


@dataclass(frozen=True)
class DataProduct:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    quarantine: pd.DataFrame
    quality: pd.DataFrame
    stages: pd.DataFrame
    metadata: dict


def _hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["datetime", "datetimetz"]):
        normalized[column] = normalized[column].astype(str)
    payload = normalized.sort_index(axis=1).sort_values(list(normalized.columns), kind="stable").to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _bronze(raw: bytes) -> pd.DataFrame:
    frame = parse_rdb(raw).copy()
    frame["delivery_id"] = [hashlib.sha256(f"{r.site_no}|{r.event_date}|{r.discharge_cfs}|{r.qualifier}".encode()).hexdigest()[:20] for r in frame.itertuples()]
    frame["event_id"] = [hashlib.sha256(f"{r.site_no}|{r.event_date}".encode()).hexdigest()[:20] for r in frame.itertuples()]
    frame["batch_id"] = pd.to_datetime(frame.event_date, errors="coerce").dt.to_period("M").astype(str)
    # Replay a bounded delivery slice so deduplication and reconciliation stay demonstrable.
    repeats = frame.sort_values(["site_no", "event_date"]).groupby("site_no").tail(2).copy()
    repeats["batch_id"] = repeats.batch_id + "-replay"
    return pd.concat([frame, repeats], ignore_index=True)


def _silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bronze.copy()
    frame["event_date"] = pd.to_datetime(frame.event_date, errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame.discharge_cfs, errors="coerce")
    known = frame.site_no.isin(SITES)
    valid = known & frame.event_date.notna() & frame.discharge_cfs.notna() & frame.discharge_cfs.ge(0) & frame.event_id.notna()
    frame["reason"] = np.select([~known, frame.event_date.isna(), frame.discharge_cfs.isna(), frame.discharge_cfs.lt(0)],
                                ["unknown_site", "invalid_date", "invalid_value", "negative_discharge"], default="contract_failure")
    quarantine = frame.loc[~valid].copy()
    silver = frame.loc[valid].drop(columns="reason").sort_values(["event_id", "batch_id"]).drop_duplicates("event_id", keep="first").copy()
    silver["site_name"] = silver.site_no.map(lambda x: SITES[x][0])
    silver["latitude"] = silver.site_no.map(lambda x: SITES[x][1])
    silver["longitude"] = silver.site_no.map(lambda x: SITES[x][2])
    silver["is_provisional"] = silver.qualifier.str.contains("P", na=False)
    silver["record_hash"] = [hashlib.sha256(f"{r.event_id}|{r.discharge_cfs:.6f}|{r.qualifier}".encode()).hexdigest() for r in silver.itertuples()]
    return silver.sort_values(["site_no", "event_date"]).reset_index(drop=True), quarantine.reset_index(drop=True)


def _gold(silver: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, group in silver.groupby("site_no"):
        group = group.sort_values("event_date").copy()
        discharge = group.discharge_cfs.clip(lower=.01)
        group["log_discharge"] = np.log1p(discharge)
        for lag in (1, 2, 3, 7, 14, 30):
            group[f"log_lag_{lag}"] = group.log_discharge.shift(lag)
        group["log_roll_mean_7"] = group.log_discharge.shift(1).rolling(7).mean()
        group["log_roll_std_7"] = group.log_discharge.shift(1).rolling(7).std()
        group["log_roll_mean_30"] = group.log_discharge.shift(1).rolling(30).mean()
        group["change_1d"] = group.log_discharge.shift(1) - group.log_discharge.shift(2)
        group["change_7d"] = group.log_discharge.shift(1) - group.log_discharge.shift(8)
        group["future_max_3d"] = pd.concat([group.discharge_cfs.shift(-i) for i in (1, 2, 3)], axis=1).max(axis=1)
        groups.append(group)
    gold = pd.concat(groups, ignore_index=True)
    day = gold.event_date.dt.dayofyear
    gold["doy_sin"] = np.sin(2*np.pi*day/365.25); gold["doy_cos"] = np.cos(2*np.pi*day/365.25)
    return gold


def run_pipeline() -> DataProduct:
    started = time.perf_counter(); raw, source = load_source(); extract_ms = (time.perf_counter()-started)*1000
    bronze = _bronze(raw); silver, quarantine = _silver(bronze); gold = _gold(silver)
    duplicate_deliveries = len(bronze) - bronze.event_id.nunique()
    checks = [
        ("six_sources", silver.site_no.nunique() == 6, f"{silver.site_no.nunique()} stations"),
        ("unique_event_key", not silver.event_id.duplicated().any(), "one row per station-day"),
        ("finite_nonnegative_flow", np.isfinite(silver.discharge_cfs).all() and silver.discharge_cfs.ge(0).all(), "cubic feet per second"),
        ("date_range", silver.event_date.between("2018-01-01", pd.Timestamp.today().normalize()).all(), f"{silver.event_date.min().date()}–{silver.event_date.max().date()}"),
        ("daily_coverage", silver.groupby("site_no").event_date.nunique().min() >= 2500, f"minimum {silver.groupby('site_no').event_date.nunique().min()} days"),
        ("replay_suppression", duplicate_deliveries == 12, f"{duplicate_deliveries} replayed deliveries suppressed"),
        ("row_reconciliation", len(bronze) == len(silver)+len(quarantine)+duplicate_deliveries, f"{len(bronze)} input deliveries reconciled"),
        ("feature_readiness", gold[["log_lag_30", "future_max_3d"]].dropna().shape[0] >= 15000, f"{gold[["log_lag_30", "future_max_3d"]].dropna().shape[0]} model-ready rows"),
        ("lineage_complete", silver.record_hash.str.len().eq(64).all(), "SHA-256 record hashes"),
        ("source_bounds", source["source_bytes"] < 3_000_000, f"{source['source_bytes']:,} bytes"),
    ]
    quality = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    if not quality.passed.all():
        raise RuntimeError("river data product failed publication gates")
    hashes = {"bronze_hash": _hash(bronze), "silver_hash": _hash(silver), "gold_hash": _hash(gold.dropna(subset=["future_max_3d"]))}
    run_id = hashlib.sha256((source["source_hash"]+"".join(hashes.values())).encode()).hexdigest()[:12]
    stages = pd.DataFrame([
        {"stage":"Extract", "input":source["source_bytes"], "output":len(parse_rdb(raw)), "rejected":0, "duration_ms":round(extract_ms,1), "content_hash":source["source_hash"][:12]},
        {"stage":"Bronze", "input":len(parse_rdb(raw)), "output":len(bronze), "rejected":0, "duration_ms":0.0, "content_hash":hashes["bronze_hash"][:12]},
        {"stage":"Silver", "input":len(bronze), "output":len(silver), "rejected":len(quarantine)+duplicate_deliveries, "duration_ms":0.0, "content_hash":hashes["silver_hash"][:12]},
        {"stage":"Gold", "input":len(silver), "output":len(gold), "rejected":0, "duration_ms":0.0, "content_hash":hashes["gold_hash"][:12]},
    ])
    metadata = {**source, **hashes, "run_id":run_id, "duplicate_deliveries":duplicate_deliveries,
                "quarantine_rows":len(quarantine), "current_through":silver.event_date.max().date().isoformat()}
    return DataProduct(bronze, silver, gold, quarantine, quality, stages, metadata)
