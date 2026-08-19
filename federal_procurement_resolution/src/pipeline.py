"""Contracted Bronze/Silver/Gold procurement data product."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from federal_procurement_resolution.src.data import load_source


@dataclass(frozen=True)
class ProcurementProduct:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    awards: pd.DataFrame
    recipients: pd.DataFrame
    quarantine: pd.DataFrame
    quality: pd.DataFrame
    stages: pd.DataFrame
    metadata: dict


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1)
    if len(ordered):
        ordered = ordered.sort_values(list(ordered.columns), kind="stable", na_position="first")
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def make_bronze(raw: pd.DataFrame, replay_rows: int = 15) -> pd.DataFrame:
    bronze = raw.copy()
    bronze["delivery_id"] = [hashlib.sha256("|".join(map(str, row)).encode()).hexdigest()[:24] for row in bronze.itertuples(index=False, name=None)]
    bronze["payload_hash"] = [hashlib.sha256("|".join(map(str, row)).encode()).hexdigest() for row in raw.itertuples(index=False, name=None)]
    replay = bronze.sort_values("delivery_id").head(min(replay_rows, len(bronze)))
    return pd.concat([bronze, replay], ignore_index=True)


def contract_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    x = bronze.copy()
    x["recipient_name"] = x["recipient_name"].astype("string").str.strip().str.upper()
    x["recipient_uei"] = x["recipient_uei"].astype("string").str.strip().str.upper()
    x["award_amount"] = pd.to_numeric(x["award_amount"], errors="coerce")
    for column in ["start_date", "end_date", "last_modified"]:
        x[column] = pd.to_datetime(x[column], errors="coerce")
    rules = {
        "missing_award_key": x["award_id"].isna() | x["award_id"].astype(str).str.len().lt(4),
        "invalid_recipient_name": x["recipient_name"].isna() | x["recipient_name"].str.len().lt(3),
        "invalid_uei": ~x["recipient_uei"].str.fullmatch(r"[A-Z0-9]{12}", na=False),
        "invalid_amount": x["award_amount"].isna() | ~np.isfinite(x["award_amount"]) | x["award_amount"].abs().gt(1e14),
        "missing_agency": x["awarding_agency"].isna(),
        "invalid_dates": x["start_date"].isna() | x["last_modified"].isna(),
    }
    invalid = pd.Series(False, index=x.index)
    reason = pd.Series("", index=x.index, dtype="string")
    for label, mask in rules.items():
        reason = reason.mask((reason == "") & mask, label)
        invalid |= mask
    x["quarantine_reason"] = reason.mask(reason == "", "contract_failure")
    quarantine = x[invalid].copy()
    valid = x[~invalid].copy()
    duplicate_mask = valid.duplicated("delivery_id", keep="first")
    duplicates = int(duplicate_mask.sum())
    silver = valid[~duplicate_mask].drop(columns="quarantine_reason").sort_values("award_id").reset_index(drop=True)
    silver["amount_log"] = np.sign(silver["award_amount"]) * np.log1p(silver["award_amount"].abs())
    return silver, quarantine.reset_index(drop=True), duplicates


def build_product(raw: pd.DataFrame, source_meta: dict, replay_rows: int = 15) -> ProcurementProduct:
    started = time.perf_counter()
    bronze = make_bronze(raw, replay_rows)
    silver, quarantine, duplicates = contract_silver(bronze)
    awards = silver[["award_id", "display_award_id", "recipient_uei", "recipient_name", "award_amount", "amount_log", "start_date", "end_date", "awarding_agency", "awarding_subagency", "award_type", "naics_code", "naics_description", "psc_code", "psc_description", "description", "last_modified", "payload_hash"]].copy()
    name_counts = silver.groupby(["recipient_uei", "recipient_name"], as_index=False).size().sort_values(["recipient_uei", "size", "recipient_name"], ascending=[True, False, True])
    canonical = name_counts.drop_duplicates("recipient_uei")[["recipient_uei", "recipient_name"]].rename(columns={"recipient_name": "canonical_name"})
    aggregates = silver.groupby("recipient_uei", as_index=False).agg(award_count=("award_id", "nunique"), total_award_value=("award_amount", "sum"), agency_count=("awarding_agency", "nunique"), last_modified=("last_modified", "max"))
    recipients = canonical.merge(aggregates, on="recipient_uei", validate="one_to_one").sort_values("recipient_uei").reset_index(drop=True)
    reconciliation = len(bronze) == len(silver) + len(quarantine) + duplicates
    valid_ratio = len(silver) / max(1, len(bronze) - duplicates)
    gates = [
        ("source_volume", len(raw) >= 120, f"{len(raw):,} source awards"),
        ("typed_contract", valid_ratio >= .70, f"{valid_ratio:.1%} accepted"),
        ("award_key_unique", not silver["award_id"].duplicated().any(), "one current award row per stable key"),
        ("uei_contract", silver["recipient_uei"].str.fullmatch(r"[A-Z0-9]{12}").all(), "12-character UEI"),
        ("replay_suppression", duplicates == min(replay_rows, len(raw)), f"{duplicates} replay deliveries suppressed"),
        ("row_reconciliation", reconciliation, f"{len(bronze):,} deliveries reconciled"),
        ("recipient_reference", set(awards["recipient_uei"]) == set(recipients["recipient_uei"]), "award-to-recipient foreign keys complete"),
        ("finite_amount", np.isfinite(awards["award_amount"]).all(), "bounded signed award values"),
    ]
    quality = pd.DataFrame(gates, columns=["check", "passed", "detail"])
    if not quality["passed"].all():
        failed = ", ".join(quality.loc[~quality.passed, "check"])
        raise RuntimeError(f"procurement product failed publication gates: {failed}")
    hashes = {name: frame_hash(frame) for name, frame in {"bronze": bronze, "silver": silver, "awards": awards, "recipients": recipients}.items()}
    run_id = hashlib.sha256((source_meta["source_hash"] + "".join(hashes.values())).encode()).hexdigest()[:12]
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    stages = pd.DataFrame([
        {"stage": "Extract", "input": source_meta.get("source_bytes", 0), "output": len(raw), "rejected": 0, "hash": source_meta["source_hash"][:12]},
        {"stage": "Bronze", "input": len(raw), "output": len(bronze), "rejected": 0, "hash": hashes["bronze"][:12]},
        {"stage": "Silver", "input": len(bronze), "output": len(silver), "rejected": len(quarantine) + duplicates, "hash": hashes["silver"][:12]},
        {"stage": "Gold awards", "input": len(silver), "output": len(awards), "rejected": 0, "hash": hashes["awards"][:12]},
        {"stage": "Gold recipients", "input": len(silver), "output": len(recipients), "rejected": 0, "hash": hashes["recipients"][:12]},
    ])
    metadata = {**source_meta, **{f"{key}_hash": value for key, value in hashes.items()}, "run_id": run_id, "duplicates": duplicates, "quarantined": len(quarantine), "duration_ms": elapsed}
    return ProcurementProduct(bronze, silver, awards, recipients, quarantine, quality, stages, metadata)


def run_pipeline() -> ProcurementProduct:
    raw, metadata = load_source()
    return build_product(raw, metadata)
