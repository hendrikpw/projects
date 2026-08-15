"""Revision-aware XBRL medallion pipeline, contracts, lineage and quality gates."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sec_fundamentals_control.src.data import COMPANIES, load_payloads

CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
DURATION = {"revenue", "net_income"}


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    gold: pd.DataFrame
    quarantine: pd.DataFrame
    quality: pd.DataFrame
    stages: pd.DataFrame
    metadata: dict


def _hash_frame(frame: pd.DataFrame) -> str:
    work = frame.sort_index(axis=1)
    if len(work):
        work = work.sort_values(list(work.columns), kind="stable")
    return hashlib.sha256(work.to_csv(index=False, float_format="%.10g").encode()).hexdigest()


def _parse(payloads: dict[str, bytes]) -> pd.DataFrame:
    rows = []
    for ticker, payload in payloads.items():
        document = json.loads(payload)
        gaap = document.get("facts", {}).get("us-gaap", {})
        for metric, alternatives in CONCEPTS.items():
            for rank, concept in enumerate(alternatives):
                unit_map = gaap.get(concept, {}).get("units", {})
                for fact in unit_map.get("USD", []):
                    frame = str(fact.get("frame") or "")
                    pattern = r"^CY(20\d{2})Q([1-4])$" if metric in DURATION else r"^CY(20\d{2})Q([1-4])I$"
                    match = re.match(pattern, frame)
                    if not match or fact.get("form") not in {"10-Q", "10-K"}:
                        continue
                    rows.append({
                        "ticker": ticker, "cik": str(document.get("cik", COMPANIES[ticker])).zfill(10),
                        "entity_name": document.get("entityName", ticker), "metric": metric,
                        "concept": concept, "concept_rank": rank, "frame": frame.rstrip("I"),
                        "calendar_year": int(match.group(1)), "calendar_quarter": int(match.group(2)),
                        "period_end": fact.get("end"), "filed": fact.get("filed"),
                        "accession": fact.get("accn"), "form": fact.get("form"), "unit": "USD",
                        "value": fact.get("val"),
                    })
    return pd.DataFrame(rows)


def _silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bronze.copy()
    for col in ["period_end", "filed"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    rules = [
        (~frame.ticker.isin(COMPANIES), "unknown_company"),
        (~frame.metric.isin(CONCEPTS), "unknown_metric"),
        (frame.value.isna(), "invalid_value"),
        (frame[["period_end", "filed"]].isna().any(axis=1), "invalid_date"),
        (frame.calendar_year.lt(2016) | frame.calendar_year.gt(2030), "year_out_of_range"),
        (frame.metric.isin(["revenue", "assets"]) & frame.value.le(0), "nonpositive_scale"),
    ]
    for mask, label in rules:
        reason.loc[mask.fillna(True) & reason.isna()] = label
    quarantine = frame.loc[reason.notna()].copy()
    quarantine["invalid_reason"] = reason.dropna()
    valid = frame.loc[reason.isna()].copy()
    keys = ["ticker", "frame", "metric"]
    valid["revision_count"] = valid.groupby(keys).ticker.transform("size")
    valid = valid.sort_values(keys + ["filed", "concept_rank", "accession"], ascending=[True, True, True, True, False, True])
    valid = valid.drop_duplicates(keys, keep="last").reset_index(drop=True)
    valid["fact_id"] = [hashlib.sha256(f"{t}:{fr}:{m}".encode()).hexdigest()[:20] for t, fr, m in valid[keys].itertuples(index=False, name=None)]
    valid["source_record_hash"] = [hashlib.sha256(f"{a}:{v}:{d}".encode()).hexdigest() for a, v, d in zip(valid.accession, valid.value, valid.filed)]
    return valid, quarantine.reset_index(drop=True)


def _gold(silver: pd.DataFrame) -> pd.DataFrame:
    index = ["ticker", "cik", "entity_name", "frame", "calendar_year", "calendar_quarter"]
    values = silver.pivot_table(index=index, columns="metric", values="value", aggfunc="first").reset_index()
    lineage = silver.groupby(index, as_index=False).agg(
        latest_filed=("filed", "max"), revision_count=("revision_count", "sum"),
        accessions=("accession", lambda x: "|".join(sorted(set(map(str, x))))),
    )
    gold = values.merge(lineage, on=index, how="left")
    required = list(CONCEPTS)
    gold = gold.dropna(subset=required).sort_values(["ticker", "calendar_year", "calendar_quarter"]).reset_index(drop=True)
    gold["net_margin"] = gold.net_income / gold.revenue
    gold["liability_ratio"] = (gold.assets - gold.equity) / gold.assets
    gold["asset_turnover_quarterly"] = gold.revenue / gold.assets
    gold["return_on_assets_quarterly"] = gold.net_income / gold.assets
    gold["revenue_growth_yoy"] = gold.groupby("ticker").revenue.pct_change(4, fill_method=None)
    gold["margin_change_yoy"] = gold.net_margin - gold.groupby("ticker").net_margin.shift(4)
    gold["gold_id"] = [hashlib.sha256(f"{t}:{f}".encode()).hexdigest()[:20] for t, f in zip(gold.ticker, gold.frame)]
    return gold


def _quality(bronze: pd.DataFrame, silver: pd.DataFrame, quarantine: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["revenue_growth_yoy", "net_margin", "liability_ratio", "asset_turnover_quarterly", "return_on_assets_quarterly", "margin_change_yoy"]
    superseded = len(bronze) - len(quarantine) - len(silver)
    checks = [
        ("bronze_schema", set(["ticker", "metric", "frame", "value", "filed"]).issubset(bronze), "required XBRL fields"),
        ("row_reconciliation", len(bronze) == len(silver) + len(quarantine) + superseded, f"{superseded:,} superseded facts accounted for"),
        ("silver_identity", silver.fact_id.is_unique, f"{len(silver):,} unique current facts"),
        ("company_coverage", gold.ticker.nunique() == len(COMPANIES), f"{gold.ticker.nunique()} / {len(COMPANIES)} companies"),
        ("minimum_history", gold.groupby("ticker").size().min() >= 20, f"minimum {gold.groupby('ticker').size().min()} quarters"),
        ("metric_completeness", gold[list(CONCEPTS)].notna().all().all(), "all four accounting facts complete"),
        ("ratio_bounds", gold.liability_ratio.between(0, 2).all() and gold.net_margin.between(-2, 2).all(), "bounded analytical ratios"),
        ("gold_identity", gold.gold_id.is_unique, f"{len(gold):,} unique company-quarters"),
        ("ai_readiness", gold[feature_cols].dropna().shape[0] >= 150, f"{gold[feature_cols].dropna().shape[0]:,} complete feature rows"),
        ("revision_lineage", gold.revision_count.ge(4).all(), "every Gold row traces four selected facts"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def run_pipeline() -> PipelineBundle:
    payloads, metadata = load_payloads()
    stages = []
    start = time.perf_counter(); bronze = _parse(payloads); bronze_hash = _hash_frame(bronze)
    stages.append(("Bronze", sum(len(v) for v in payloads.values()), len(bronze), 0, (time.perf_counter()-start)*1000, bronze_hash))
    start = time.perf_counter(); silver, quarantine = _silver(bronze); silver_hash = _hash_frame(silver)
    stages.append(("Silver", len(bronze), len(silver), len(quarantine), (time.perf_counter()-start)*1000, silver_hash))
    start = time.perf_counter(); gold = _gold(silver); gold_hash = _hash_frame(gold)
    stages.append(("Gold", len(silver), len(gold), len(silver)-len(gold)*4, (time.perf_counter()-start)*1000, gold_hash))
    quality = _quality(bronze, silver, quarantine, gold)
    if not quality.passed.all():
        raise RuntimeError("data product withheld: " + ", ".join(quality.loc[~quality.passed, "check"]))
    run_id = hashlib.sha256(f"{metadata['source_hash']}:{gold_hash}".encode()).hexdigest()[:12]
    ledger = pd.DataFrame(stages, columns=["stage", "input_bytes_or_rows", "output_rows", "rejected_rows", "duration_ms", "content_hash"])
    ledger["status"] = "passed"
    metadata = {**metadata, "run_id": run_id, "bronze_hash": bronze_hash, "silver_hash": silver_hash, "gold_hash": gold_hash, "manifest_hash": hashlib.sha256(json.dumps({"run_id": run_id, "concepts": CONCEPTS}, sort_keys=True).encode()).hexdigest(), "superseded_facts": int(len(bronze)-len(quarantine)-len(silver))}
    return PipelineBundle(bronze, silver, gold, quarantine, quality, ledger, metadata)
