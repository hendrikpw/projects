"""Content-addressed Bronze/Silver/Gold pipeline and operational quality gates."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predictive_maintenance_pipeline.src.data import load_dataset

TARGET = "machine_failure"
FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
FEATURES = ["type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min", "temperature_gap_k", "power_proxy"]
RENAME = {"UDI":"udi", "Product ID":"product_id", "Type":"type", "Air temperature [K]":"air_temperature_k",
    "Process temperature [K]":"process_temperature_k", "Rotational speed [rpm]":"rotational_speed_rpm",
    "Torque [Nm]":"torque_nm", "Tool wear [min]":"tool_wear_min", "Machine failure":TARGET,
    "TWF":"twf", "HDF":"hdf", "PWF":"pwf", "OSF":"osf", "RNF":"rnf"}


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    quarantine: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict


def _hash_frame(frame: pd.DataFrame) -> str:
    normalized = frame.sort_index(axis=1).sort_values(list(frame.columns), kind="stable") if len(frame) else frame
    return hashlib.sha256(normalized.to_csv(index=False, float_format="%.8g").encode()).hexdigest()


def _silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bronze.rename(columns=RENAME).copy()
    for column in ["udi", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min", TARGET, *FAILURE_MODES]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["product_id"] = frame["product_id"].astype("string").str.strip()
    frame["type"] = frame["type"].astype("string").str.strip().str.upper()
    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    rules = [
        (frame["udi"].isna(), "invalid_udi"),
        (frame["udi"].duplicated(keep="first"), "duplicate_udi"),
        (~frame["type"].isin(["L","M","H"]), "invalid_product_type"),
        (~frame["air_temperature_k"].between(250, 350), "air_temperature_out_of_range"),
        (~frame["process_temperature_k"].between(250, 400), "process_temperature_out_of_range"),
        (~frame["rotational_speed_rpm"].between(500, 5000), "speed_out_of_range"),
        (~frame["torque_nm"].between(0, 150), "torque_out_of_range"),
        (~frame["tool_wear_min"].between(0, 500), "tool_wear_out_of_range"),
        (~frame[TARGET].isin([0,1]), "invalid_target"),
    ]
    for mask, label in rules:
        reason.loc[mask.fillna(True) & reason.isna()] = label
    quarantine = frame.loc[reason.notna()].copy(); quarantine["invalid_reason"] = reason.dropna()
    valid = frame.loc[reason.isna()].drop_duplicates("udi").sort_values("udi").reset_index(drop=True)
    valid["udi"] = valid["udi"].astype(int)
    for column in [TARGET, *FAILURE_MODES]: valid[column] = valid[column].astype(int)
    return valid, quarantine.reset_index(drop=True)


def _gold(silver: pd.DataFrame) -> pd.DataFrame:
    gold = silver[["udi", TARGET, *FEATURES[:6]]].copy()
    gold["temperature_gap_k"] = gold["process_temperature_k"] - gold["air_temperature_k"]
    gold["power_proxy"] = gold["rotational_speed_rpm"] * gold["torque_nm"]
    return gold


def _checks(bronze: pd.DataFrame, silver: pd.DataFrame, quarantine: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    required = set(RENAME)
    checks = [
        ("source_schema", required.issubset(bronze.columns), f"{len(required)} required fields"),
        ("row_reconciliation", len(bronze)==len(silver)+len(quarantine), f"{len(bronze):,} = {len(silver):,} + {len(quarantine):,}"),
        ("entity_identity", silver["udi"].is_unique and silver["udi"].notna().all(), f"{len(silver):,} unique machine cycles"),
        ("product_domain", silver["type"].isin(["L","M","H"]).all(), "L/M/H only"),
        ("target_domain", silver[TARGET].isin([0,1]).all(), "binary target"),
        ("numeric_completeness", gold[FEATURES].notna().all().all(), "all model inputs complete"),
        ("minimum_scale", len(gold)>=2_000 and gold[TARGET].sum()>=20, f"{len(gold):,} rows / {gold[TARGET].sum():,} failures"),
        ("leakage_contract", not set(FAILURE_MODES).intersection(gold.columns), "failure-mode labels excluded from Gold"),
        ("feature_contract", FEATURES==[c for c in gold.columns if c not in ["udi",TARGET]], f"{len(FEATURES)} ordered serving features"),
        ("gold_reconciliation", len(gold)==len(silver), f"{len(gold):,} publishable feature rows"),
    ]
    return pd.DataFrame(checks, columns=["check","passed","detail"])


def run_pipeline() -> PipelineBundle:
    bronze, metadata = load_dataset(); ledger=[]
    started=time.perf_counter(); bronze_hash=_hash_frame(bronze)
    ledger.append(("Bronze",len(bronze),len(bronze),0,(time.perf_counter()-started)*1000,bronze_hash))
    started=time.perf_counter(); silver,quarantine=_silver(bronze); silver_hash=_hash_frame(silver)
    ledger.append(("Silver",len(bronze),len(silver),len(quarantine),(time.perf_counter()-started)*1000,silver_hash))
    started=time.perf_counter(); gold=_gold(silver); gold_hash=_hash_frame(gold)
    ledger.append(("Gold",len(silver),len(gold),0,(time.perf_counter()-started)*1000,gold_hash))
    quality=_checks(bronze,silver,quarantine,gold)
    if not quality["passed"].all():
        raise RuntimeError("data product withheld: "+", ".join(quality.loc[~quality.passed,"check"]))
    run_id=hashlib.sha256(f"{metadata['source_hash']}:{gold_hash}".encode()).hexdigest()[:12]
    stages=pd.DataFrame(ledger,columns=["stage","input_rows","output_rows","rejected_rows","duration_ms","content_hash"]); stages["status"]="passed"
    metadata={**metadata,"run_id":run_id,"bronze_hash":bronze_hash,"silver_hash":silver_hash,"gold_hash":gold_hash,
        "manifest_hash":hashlib.sha256(json.dumps({"run_id":run_id,"features":FEATURES},sort_keys=True).encode()).hexdigest()}
    return PipelineBundle(bronze,silver,gold,quarantine,stages,quality,metadata)
