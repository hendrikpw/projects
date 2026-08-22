"""Replay-safe image micro-batches, pixel contracts and content lineage."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from digit_recognition_gateway.src.data import PIXELS, load_source


@dataclass(frozen=True)
class DigitProduct:
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
    if len(ordered): ordered = ordered.sort_values(list(ordered.columns), kind="stable", na_position="first")
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def make_bronze(raw: pd.DataFrame, replay_rows: int = 20, batch_size: int = 128) -> pd.DataFrame:
    x = raw.copy()
    x["sample_id"] = [hashlib.sha256(f"{split}|{row}|{label}".encode()).hexdigest()[:24] for split, row, label in x[["source_split", "source_row", "label"]].itertuples(index=False, name=None)]
    x["image_hash"] = [hashlib.sha256(bytes(np.asarray(row, dtype=np.uint8))).hexdigest() for row in x[PIXELS].itertuples(index=False, name=None)]
    x["delivery_sequence"] = np.arange(len(x)); replay = x.sort_values("sample_id").head(min(replay_rows, len(x))).copy(); replay["delivery_sequence"] = np.arange(len(x), len(x) + len(replay))
    bronze = pd.concat([x, replay], ignore_index=True).sort_values("delivery_sequence").reset_index(drop=True)
    bronze["batch_id"] = (bronze.delivery_sequence // batch_size).map(lambda value: f"batch-{value:04d}")
    return bronze


def contract_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    x = bronze.copy(); x[PIXELS] = x[PIXELS].apply(pd.to_numeric, errors="coerce"); values = x[PIXELS].to_numpy(float)
    rules = {
        "invalid_sample_id": ~x.sample_id.astype(str).str.fullmatch(r"[0-9a-f]{24}"),
        "invalid_image_hash": ~x.image_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}"),
        "invalid_split": ~x.source_split.isin(["train", "test"]),
        "invalid_label": ~pd.to_numeric(x.label, errors="coerce").between(0, 9),
        "missing_pixel": pd.Series(~np.isfinite(values).all(axis=1), index=x.index),
        "pixel_out_of_range": pd.Series(((np.nan_to_num(values, nan=99) < 0) | (np.nan_to_num(values, nan=99) > 16)).any(axis=1), index=x.index),
        "non_integer_pixel": pd.Series((np.nan_to_num(values) % 1 != 0).any(axis=1), index=x.index),
    }
    invalid = pd.Series(False, index=x.index); reason = pd.Series("", index=x.index, dtype="string")
    for label, mask in rules.items(): reason = reason.mask((reason == "") & mask, label); invalid |= mask
    x["quarantine_reason"] = reason.mask(reason == "", "contract_failure"); quarantine = x[invalid].copy(); valid = x[~invalid].copy()
    replay = valid.duplicated("sample_id", keep="first"); duplicates = int(replay.sum()); silver = valid[~replay].drop(columns="quarantine_reason").sort_values("sample_id").reset_index(drop=True)
    return silver, quarantine.reset_index(drop=True), duplicates


def build_product(raw: pd.DataFrame, source_meta: dict, replay_rows: int = 20, batch_size: int = 128) -> DigitProduct:
    started = time.perf_counter(); bronze = make_bronze(raw, replay_rows, batch_size); silver, quarantine, duplicates = contract_silver(bronze)
    gold = silver[["sample_id", "image_hash", "source_split", "source_row", "label", "batch_id", *PIXELS]].copy(); gold[PIXELS] = gold[PIXELS].astype(np.float32) / 16
    batch_source = bronze.assign(is_replay=bronze.duplicated("sample_id", keep="first").astype(int)); batches = batch_source.groupby("batch_id", as_index=False).agg(deliveries=("sample_id", "size"), unique_images=("sample_id", "nunique"), replays=("is_replay", "sum"), digits=("label", "nunique"))
    reconciliation = len(bronze) == len(silver) + len(quarantine) + duplicates; shares = gold.label.value_counts(normalize=True)
    gates = [
        ("source_volume", len(raw) >= 5_000, f"{len(raw):,} source images"), ("pixel_schema", len(PIXELS) == 64, "exact 8×8 pixel contract"),
        ("typed_acceptance", len(silver) / max(1, len(bronze) - duplicates) >= .98, f"{len(silver):,} images accepted"), ("sample_id_unique", gold.sample_id.is_unique, "stable unique sample key"),
        ("replay_suppression", duplicates == min(replay_rows, len(raw)), f"{duplicates} replays suppressed"), ("row_reconciliation", reconciliation, f"{len(bronze):,} deliveries reconciled"),
        ("finite_pixels", np.isfinite(gold[PIXELS].to_numpy()).all(), "no missing or infinite pixels"), ("normalized_pixels", gold[PIXELS].min().min() >= 0 and gold[PIXELS].max().max() <= 1, "Gold pixels normalized to [0,1]"),
        ("class_coverage", set(gold.label) == set(range(10)) and shares.min() > .08, f"all digits; min share {shares.min():.1%}"), ("official_partitions", set(gold.source_split) == {"train", "test"} and (gold.source_split == "test").sum() >= 1_700, "train/test writer boundary preserved"),
    ]
    quality = pd.DataFrame(gates, columns=["check", "passed", "detail"])
    if not quality.passed.all(): raise RuntimeError("digit product failed publication gates: " + ", ".join(quality.loc[~quality.passed, "check"]))
    hashes = {name: frame_hash(frame) for name, frame in {"bronze": bronze, "silver": silver, "gold": gold, "batches": batches}.items()}; run_id = hashlib.sha256((source_meta["source_hash"] + "".join(hashes.values())).encode()).hexdigest()[:12]
    stages = pd.DataFrame([{"stage":"Extract","input":source_meta["source_bytes"],"output":len(raw),"rejected":0,"hash":source_meta["source_hash"][:12]},{"stage":"Bronze","input":len(raw),"output":len(bronze),"rejected":0,"hash":hashes["bronze"][:12]},{"stage":"Silver","input":len(bronze),"output":len(silver),"rejected":len(quarantine)+duplicates,"hash":hashes["silver"][:12]},{"stage":"Gold","input":len(silver),"output":len(gold),"rejected":0,"hash":hashes["gold"][:12]}])
    metadata = {**source_meta, **{f"{key}_hash":value for key,value in hashes.items()}, "run_id":run_id, "source_rows":len(raw), "deliveries":len(bronze), "accepted":len(silver), "replays":duplicates, "duplicates":duplicates, "quarantined":len(quarantine), "duration_ms":round((time.perf_counter()-started)*1000,1)}
    return DigitProduct(bronze, silver, gold, quarantine, quality, stages, batches, metadata)


def run_pipeline(force_fallback: bool = False) -> DigitProduct:
    raw, metadata = load_source(force_fallback=force_fallback); return build_product(raw, metadata)
