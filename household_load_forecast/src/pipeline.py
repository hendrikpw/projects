"""Replay-safe hourly power data product with contracts, watermarks and lineage."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from household_load_forecast.src.data import load_source

MEASURES = ["load_kw", "reactive_kw", "voltage_v", "intensity_a", "kitchen_wh", "laundry_wh", "climate_wh"]


@dataclass(frozen=True)
class LoadProduct:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    quarantine: pd.DataFrame
    quality: pd.DataFrame
    stages: pd.DataFrame
    batches: pd.DataFrame
    metadata: dict


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1)
    if len(ordered):
        ordered = ordered.sort_values(list(ordered.columns), kind="stable", na_position="first")
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def make_bronze(raw: pd.DataFrame, replay_rows: int = 24, batch_hours: int = 168, watermark_hours: int = 48) -> pd.DataFrame:
    x = raw.sort_values("timestamp").reset_index(drop=True).copy()
    x["event_id"] = x.timestamp.map(lambda value: hashlib.sha256(value.isoformat().encode()).hexdigest()[:24])
    x["payload_hash"] = [hashlib.sha256("|".join(map(str, row)).encode()).hexdigest() for row in x[["timestamp", *MEASURES, "readings"]].itertuples(index=False, name=None)]
    x["delivery_sequence"] = np.arange(len(x), dtype=int)
    replay = x.iloc[np.linspace(0, len(x) - 1, min(replay_rows, len(x)), dtype=int)].copy()
    replay["delivery_sequence"] = np.arange(len(x), len(x) + len(replay), dtype=int)
    bronze = pd.concat([x, replay], ignore_index=True).sort_values("delivery_sequence").reset_index(drop=True)
    bronze["batch_id"] = (bronze.delivery_sequence // batch_hours).map(lambda value: f"batch-{value:04d}")
    prior_max = bronze.timestamp.cummax().shift(1)
    bronze["lateness_hours"] = ((prior_max - bronze.timestamp).dt.total_seconds() / 3600).clip(lower=0).fillna(0)
    bronze["after_watermark"] = bronze.lateness_hours > watermark_hours
    return bronze


def contract_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    x = bronze.copy()
    for column in [*MEASURES, "readings"]:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    rules = {
        "invalid_event_id": ~x.event_id.astype(str).str.fullmatch(r"[0-9a-f]{24}"),
        "invalid_payload_hash": ~x.payload_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}"),
        "invalid_timestamp": x.timestamp.isna(),
        "missing_measure": ~np.isfinite(x[MEASURES].to_numpy(float)).all(axis=1),
        "load_out_of_range": ~x.load_kw.between(0, 20),
        "voltage_out_of_range": ~x.voltage_v.between(150, 300),
        "intensity_out_of_range": ~x.intensity_a.between(0, 100),
        "negative_submeter": (x[["kitchen_wh", "laundry_wh", "climate_wh"]] < 0).any(axis=1),
        "invalid_reading_count": ~x.readings.between(1, 60),
    }
    invalid = pd.Series(False, index=x.index); reason = pd.Series("", index=x.index, dtype="string")
    for label, mask in rules.items():
        reason = reason.mask((reason == "") & mask, label); invalid |= mask
    x["quarantine_reason"] = reason.mask(reason == "", "contract_failure")
    quarantine = x[invalid].copy(); valid = x[~invalid].copy()
    replay = valid.duplicated("event_id", keep="first"); duplicates = int(replay.sum())
    silver = valid[~replay].drop(columns="quarantine_reason").sort_values("timestamp").reset_index(drop=True)
    return silver, quarantine.reset_index(drop=True), duplicates


def _gold_hours(silver: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    indexed = silver.set_index("timestamp").sort_index()
    full_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq="h", tz="UTC")
    gold = indexed.reindex(full_index); gold.index.name = "timestamp"
    gold["was_imputed"] = gold.event_id.isna()
    missing_hours = int(gold.was_imputed.sum())
    gold[MEASURES] = gold[MEASURES].interpolate(method="time", limit=6, limit_direction="both")
    gold["readings"] = gold.readings.fillna(0)
    gold["event_id"] = [hashlib.sha256(value.isoformat().encode()).hexdigest()[:24] for value in gold.index]
    gold["payload_hash"] = gold.payload_hash.fillna("imputed")
    unresolved = ~np.isfinite(gold[MEASURES].to_numpy(float)).all(axis=1)
    unresolved_count = int(unresolved.sum())
    gold = gold.loc[~unresolved, ["event_id", "payload_hash", "readings", "was_imputed", *MEASURES]].reset_index()
    gold["completeness"] = gold.readings / 60
    return gold, missing_hours, unresolved_count


def build_product(raw: pd.DataFrame, source_meta: dict, replay_rows: int = 24, batch_hours: int = 168) -> LoadProduct:
    started = time.perf_counter(); bronze = make_bronze(raw, replay_rows, batch_hours)
    silver, quarantine, duplicates = contract_silver(bronze); gold, missing_hours, unresolved = _gold_hours(silver)
    batch_source = bronze.assign(is_replay=bronze.duplicated("event_id", keep="first").astype(int))
    batches = batch_source.groupby("batch_id", as_index=False).agg(deliveries=("event_id", "size"), unique_events=("event_id", "nunique"), replays=("is_replay", "sum"), late_events=("after_watermark", "sum"), max_lateness_hours=("lateness_hours", "max"), first_event=("timestamp", "min"), last_event=("timestamp", "max"))
    acceptance = len(silver) / max(1, len(bronze) - duplicates)
    reconciliation = len(bronze) == len(silver) + len(quarantine) + duplicates
    gap_share = missing_hours / max(1, missing_hours + len(silver))
    coverage_days = (gold.timestamp.max() - gold.timestamp.min()).days
    gates = [
        ("source_volume", source_meta["source_rows"] >= 1_000_000, f"{source_meta['source_rows']:,} minute deliveries"),
        ("hourly_volume", len(gold) >= 30_000, f"{len(gold):,} publishable hours"),
        ("typed_acceptance", acceptance >= .95, f"{acceptance:.2%} observed hours accepted"),
        ("event_id_unique", gold.event_id.is_unique, "stable unique hourly key"),
        ("replay_suppression", duplicates == min(replay_rows, len(raw)), f"{duplicates} replays suppressed"),
        ("row_reconciliation", reconciliation, f"{len(bronze):,} deliveries reconciled"),
        ("finite_gold", np.isfinite(gold[MEASURES].to_numpy()).all(), "no unresolved model values"),
        ("gap_budget", gap_share < .02 and unresolved / max(1, len(gold) + unresolved) < .015, f"{missing_hours - unresolved} short gaps imputed; {unresolved} long-gap hours withheld"),
        ("temporal_order", gold.timestamp.is_monotonic_increasing and not gold.timestamp.duplicated().any(), "strict hourly event order"),
        ("history_coverage", coverage_days >= 1_300, f"{coverage_days:,} days of history"),
    ]
    quality = pd.DataFrame(gates, columns=["check", "passed", "detail"])
    if not quality.passed.all():
        raise RuntimeError("load data product failed publication gates: " + ", ".join(quality.loc[~quality.passed, "check"]))
    hashes = {name: frame_hash(frame) for name, frame in {"bronze": bronze, "silver": silver, "gold": gold, "batches": batches}.items()}
    run_id = hashlib.sha256((source_meta["source_hash"] + "".join(hashes.values())).encode()).hexdigest()[:12]
    stages = pd.DataFrame([
        {"stage": "Extract", "input": source_meta["source_bytes"], "output": source_meta["source_rows"], "rejected": source_meta["missing_source_rows"], "hash": source_meta["source_hash"][:12]},
        {"stage": "Bronze", "input": len(raw), "output": len(bronze), "rejected": 0, "hash": hashes["bronze"][:12]},
        {"stage": "Silver", "input": len(bronze), "output": len(silver), "rejected": len(quarantine) + duplicates, "hash": hashes["silver"][:12]},
        {"stage": "Gold", "input": len(silver), "output": len(gold), "rejected": unresolved, "hash": hashes["gold"][:12]},
    ])
    metadata = {**source_meta, **{f"{key}_hash": value for key, value in hashes.items()}, "run_id": run_id, "duplicates": duplicates, "quarantined": len(quarantine), "missing_hours": missing_hours, "unresolved_hours": unresolved, "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
    return LoadProduct(bronze, silver, gold, quarantine, quality, stages, batches, metadata)


def run_pipeline() -> LoadProduct:
    raw, metadata = load_source()
    return build_product(raw, metadata)
