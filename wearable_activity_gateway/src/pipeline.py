"""Replay-safe micro-batch pipeline for labeled sensor windows."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wearable_activity_gateway.src.data import ACTIVITIES, load_source


@dataclass(frozen=True)
class SensorProduct:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    quarantine: pd.DataFrame
    quality: pd.DataFrame
    stages: pd.DataFrame
    batches: pd.DataFrame
    features: list[str]
    metadata: dict


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1)
    if len(ordered): ordered = ordered.sort_values(list(ordered.columns), kind="stable", na_position="first")
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def make_bronze(raw: pd.DataFrame, replay_rows: int = 20, batch_size: int = 256) -> pd.DataFrame:
    x = raw.copy()
    x["window_id"] = [hashlib.sha256(f"{split}|{row}|{subject}|{activity}".encode()).hexdigest()[:24] for split, row, subject, activity in x[["source_split", "source_row", "subject_id", "activity_id"]].itertuples(index=False, name=None)]
    payload_columns = [column for column in x.columns if column not in {"window_id"}]
    x["payload_hash"] = [hashlib.sha256("|".join(map(str, row)).encode()).hexdigest() for row in x[payload_columns].itertuples(index=False, name=None)]
    x["event_time"] = pd.Timestamp("2012-01-01", tz="UTC") + pd.to_timedelta(x.subject_id * 100_000 + x.source_row * 1.28, unit="s")
    x["delivery_sequence"] = np.arange(len(x))
    replay = x.sort_values("window_id").head(min(replay_rows, len(x))).copy()
    replay["delivery_sequence"] = np.arange(len(x), len(x) + len(replay))
    bronze = pd.concat([x, replay], ignore_index=True).sort_values("delivery_sequence").reset_index(drop=True)
    bronze["batch_id"] = (bronze.delivery_sequence // batch_size).map(lambda value: f"batch-{value:04d}")
    return bronze


def contract_silver(bronze: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    x = bronze.copy()
    x[features] = x[features].apply(pd.to_numeric, errors="coerce")
    feature_values = x[features].to_numpy(float)
    rules = {
        "invalid_window_key": ~x.window_id.astype(str).str.fullmatch(r"[0-9a-f]{24}"),
        "invalid_subject": ~pd.to_numeric(x.subject_id, errors="coerce").between(1, 30),
        "invalid_activity": ~x.activity_id.isin(ACTIVITIES),
        "activity_label_mismatch": x.activity != x.activity_id.map(ACTIVITIES),
        "missing_sensor_value": pd.Series(~np.isfinite(feature_values).all(axis=1), index=x.index),
        "sensor_range_violation": pd.Series((np.abs(np.nan_to_num(feature_values, nan=99)) > 1.05).any(axis=1), index=x.index),
        "invalid_event_time": x.event_time.isna(),
    }
    invalid = pd.Series(False, index=x.index); reason = pd.Series("", index=x.index, dtype="string")
    for label, mask in rules.items():
        reason = reason.mask((reason == "") & mask, label); invalid |= mask
    x = x.copy()
    x["quarantine_reason"] = reason.mask(reason == "", "contract_failure")
    quarantine = x[invalid].copy()
    valid = x[~invalid].copy()
    replay = valid.duplicated("window_id", keep="first")
    duplicates = int(replay.sum())
    silver = valid[~replay].drop(columns="quarantine_reason").sort_values("window_id").reset_index(drop=True)
    return silver, quarantine.reset_index(drop=True), duplicates


def build_product(raw: pd.DataFrame, source_meta: dict, replay_rows: int = 20, batch_size: int = 256) -> SensorProduct:
    started = time.perf_counter()
    features = [column for column in raw.columns if column.startswith("f")]
    bronze = make_bronze(raw, replay_rows, batch_size)
    silver, quarantine, duplicates = contract_silver(bronze, features)
    gold_columns = ["window_id", "event_time", "subject_id", "activity_id", "activity", "source_split", "batch_id", "payload_hash", *features]
    gold = silver[gold_columns].copy()
    batch_source = bronze.assign(is_replay=bronze.duplicated("window_id", keep="first").astype(int))
    batches = batch_source.groupby("batch_id", as_index=False).agg(deliveries=("window_id", "size"), unique_windows=("window_id", "nunique"), replays=("is_replay", "sum"), first_event=("event_time", "min"), last_event=("event_time", "max"))
    valid_ratio = len(silver) / max(1, len(bronze) - duplicates)
    reconciliation = len(bronze) == len(silver) + len(quarantine) + duplicates
    shares = gold.activity.value_counts(normalize=True)
    gates = [
        ("source_volume", len(raw) >= 1000, f"{len(raw):,} source windows"),
        ("feature_contract", len(features) >= 40, f"{len(features):,} sensor features"),
        ("typed_acceptance", valid_ratio >= .95, f"{valid_ratio:.1%} accepted"),
        ("window_id_unique", gold.window_id.is_unique, "stable unique window key"),
        ("replay_suppression", duplicates == min(replay_rows, len(raw)), f"{duplicates} replays suppressed"),
        ("row_reconciliation", reconciliation, f"{len(bronze):,} deliveries reconciled"),
        ("finite_features", np.isfinite(gold[features].to_numpy()).all(), "no missing or infinite model values"),
        ("bounded_features", (gold[features].abs() <= 1.05).all().all(), "normalized feature domain"),
        ("activity_coverage", set(gold.activity_id) == set(ACTIVITIES), "all six activities"),
        ("subject_coverage", gold.subject_id.nunique() == 30 and shares.min() > .02, f"{gold.subject_id.nunique()} subjects; min class {shares.min():.1%}"),
    ]
    quality = pd.DataFrame(gates, columns=["check", "passed", "detail"])
    if not quality.passed.all():
        raise RuntimeError("sensor product failed publication gates: " + ", ".join(quality.loc[~quality.passed, "check"]))
    hashes = {name: frame_hash(frame) for name, frame in {"bronze": bronze, "silver": silver, "gold": gold, "batches": batches}.items()}
    run_id = hashlib.sha256((source_meta["source_hash"] + "".join(hashes.values())).encode()).hexdigest()[:12]
    stages = pd.DataFrame([
        {"stage": "Extract", "input": source_meta["source_bytes"], "output": len(raw), "rejected": 0, "hash": source_meta["source_hash"][:12]},
        {"stage": "Bronze", "input": len(raw), "output": len(bronze), "rejected": 0, "hash": hashes["bronze"][:12]},
        {"stage": "Silver", "input": len(bronze), "output": len(silver), "rejected": len(quarantine) + duplicates, "hash": hashes["silver"][:12]},
        {"stage": "Gold", "input": len(silver), "output": len(gold), "rejected": 0, "hash": hashes["gold"][:12]},
    ])
    metadata = {**source_meta, **{f"{key}_hash": value for key, value in hashes.items()}, "run_id": run_id, "duplicates": duplicates, "quarantined": len(quarantine), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
    return SensorProduct(bronze, silver, gold, quarantine, quality, stages, batches, features, metadata)


def run_pipeline() -> SensorProduct:
    raw, metadata = load_source()
    return build_product(raw, metadata)
