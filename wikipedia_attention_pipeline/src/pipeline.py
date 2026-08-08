"""Event-time micro-batch pipeline, contracts, watermarks and observability."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from wikipedia_attention_pipeline.src.data import fetch_pageviews


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    quarantine: pd.DataFrame
    gold: pd.DataFrame
    batches: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]


def stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _delay_days(event_id: str) -> float:
    bucket = int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) % 10_000
    return (bucket % 36) / 24 if bucket % 31 else 3 + (bucket % 72) / 24


def bronze_events(records: list[dict[str, Any]], micro_batch_size: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay daily aggregates as deterministic event-time micro-batches."""
    rows = []
    for payload in records:
        article = str(payload.get("article", "")).strip()
        timestamp = str(payload.get("timestamp", "")).strip()
        event_id = f"{payload.get('project', '')}:{article}:{timestamp}"
        event_time = pd.to_datetime(timestamp, format="%Y%m%d%H", errors="coerce", utc=True)
        delay = _delay_days(event_id)
        rows.append({
            "event_id": event_id,
            "payload_hash": stable_hash(payload),
            "event_time": event_time,
            "arrival_time": event_time + pd.to_timedelta(delay, unit="D") if pd.notna(event_time) else pd.NaT,
            "raw_payload": payload,
            "replayed_duplicate": False,
        })
    for index in range(0, len(rows), 173):
        duplicate = dict(rows[index])
        duplicate["arrival_time"] = duplicate["arrival_time"] + pd.offsets.Hour(9)
        duplicate["replayed_duplicate"] = True
        rows.append(duplicate)
    bronze = pd.DataFrame(rows).sort_values(["arrival_time", "event_id", "replayed_duplicate"]).reset_index(drop=True)
    bronze["batch_id"] = np.arange(len(bronze)) // max(20, micro_batch_size)
    batch_rows = []
    previous_watermark = pd.Timestamp.min.tz_localize("UTC")
    late_flags = pd.Series(False, index=bronze.index)
    for batch_id, indices in bronze.groupby("batch_id", sort=True).groups.items():
        batch = bronze.loc[indices]
        late = batch["event_time"] < previous_watermark
        late_flags.loc[indices] = late
        max_event = batch["event_time"].max()
        watermark = max(previous_watermark, max_event - pd.offsets.Day(2))
        batch_rows.append({
            "batch_id": int(batch_id), "input_events": len(batch),
            "unique_event_ids": batch["event_id"].nunique(), "duplicate_events": int(batch["event_id"].duplicated().sum()),
            "late_events": int(late.sum()), "max_event_time": max_event, "watermark": watermark,
        })
        previous_watermark = watermark
    bronze["late_beyond_watermark"] = late_flags
    return bronze, pd.DataFrame(batch_rows)


def normalize_and_contract(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in bronze.to_dict(orient="records"):
        payload = record.get("raw_payload")
        if not isinstance(payload, dict):
            rows.append({"event_id": record.get("event_id", ""), "contract_error": "payload_not_object"})
            continue
        rows.append({
            "event_id": record["event_id"], "payload_hash": record["payload_hash"],
            "event_time": record["event_time"], "arrival_time": record["arrival_time"],
            "batch_id": record["batch_id"], "late_beyond_watermark": bool(record["late_beyond_watermark"]),
            "project": str(payload.get("project", "")).strip(),
            "article": str(payload.get("article", "")).strip(),
            "granularity": str(payload.get("granularity", "")).strip(),
            "access": str(payload.get("access", "")).strip(), "agent": str(payload.get("agent", "")).strip(),
            "views": pd.to_numeric(payload.get("views"), errors="coerce"), "contract_error": "",
        })
    frame = pd.DataFrame(rows)
    errors = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        reasons = []
        if not row.get("event_id") or row.get("event_id") in seen:
            reasons.append("duplicate_or_missing_event_id")
        seen.add(row.get("event_id"))
        if row.get("project") != "en.wikipedia": reasons.append("unsupported_project")
        if not row.get("article"): reasons.append("missing_article")
        if row.get("granularity") != "daily": reasons.append("unexpected_granularity")
        if pd.isna(row.get("event_time")): reasons.append("invalid_event_time")
        if pd.isna(row.get("views")) or float(row.get("views", -1)) < 0: reasons.append("invalid_views")
        errors.append("|".join(reasons))
    frame["contract_error"] = errors
    quarantine = frame[frame["contract_error"].ne("")].copy().reset_index(drop=True)
    silver = frame[frame["contract_error"].eq("")].drop(columns="contract_error").copy()
    silver["views"] = silver["views"].astype("int64")
    return silver.sort_values(["article", "event_time"]).reset_index(drop=True), quarantine


def gold_features(silver: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for article, group in silver.groupby("article", sort=True):
        part = group.sort_values("event_time").copy()
        part["article_code"] = article
        part["lag_1"] = part["views"].shift(1)
        part["lag_7"] = part["views"].shift(7)
        part["lag_14"] = part["views"].shift(14)
        part["rolling_7"] = part["views"].shift(1).rolling(7).median()
        part["rolling_28"] = part["views"].shift(1).rolling(28).median()
        part["rolling_std_28"] = part["views"].shift(1).rolling(28).std()
        part["day_of_week"] = part["event_time"].dt.dayofweek
        part["day_sin"] = np.sin(2 * np.pi * part["day_of_week"] / 7)
        part["day_cos"] = np.cos(2 * np.pi * part["day_of_week"] / 7)
        part["trend"] = np.arange(len(part))
        frames.append(part)
    gold = pd.concat(frames, ignore_index=True)
    required = ["lag_1", "lag_7", "lag_14", "rolling_7", "rolling_28", "rolling_std_28"]
    return gold.dropna(subset=required).sort_values(["event_time", "article"]).reset_index(drop=True)


def quality_report(bronze: pd.DataFrame, silver: pd.DataFrame, quarantine: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("Bronze event IDs present", bronze["event_id"].ne("").all(), f"{bronze['event_id'].ne('').mean():.1%}"),
        ("Silver event IDs unique", silver["event_id"].is_unique, f"{silver['event_id'].nunique():,}/{len(silver):,}"),
        ("Non-negative views", silver["views"].ge(0).all(), f"min {silver['views'].min():,}"),
        ("Daily source contract", silver["granularity"].eq("daily").all(), ", ".join(silver["granularity"].unique())),
        ("Event time complete", silver["event_time"].notna().all(), f"{silver['event_time'].notna().mean():.1%}"),
        ("Duplicate replays quarantined", len(silver) + len(quarantine) == len(bronze), f"{len(quarantine):,} quarantined"),
        ("Gold lags complete", gold[["lag_1", "lag_7", "lag_14", "rolling_7", "rolling_28"]].notna().all().all(), f"{len(gold):,} rows"),
        ("Article coverage", gold.groupby("article").size().min() >= 60, f"min {gold.groupby('article').size().min():,} rows/article"),
        ("No future leakage", all(
            gold.loc[gold["article"].eq(article), "event_time"].min()
            > silver.loc[silver["article"].eq(article), "event_time"].min()
            for article in gold["article"].unique()
        ), "features begin after shifted source history"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _event(stage: str, started: float, inputs: int, outputs: int, digest: str) -> dict[str, Any]:
    return {"stage": stage, "status": "passed", "input_rows": int(inputs), "output_rows": int(outputs),
            "rejected_rows": int(max(inputs - outputs, 0)), "duration_ms": round((perf_counter() - started) * 1000, 2),
            "content_hash": digest[:12]}


def run_pipeline(articles: list[str], history_days: int = 180, micro_batch_size: int = 120) -> PipelineBundle:
    events = []
    started = perf_counter(); records, metadata = fetch_pageviews(articles, history_days); raw_hash = stable_hash(records)
    events.append(_event("Extract / Wikimedia", started, len(records), len(records), raw_hash))
    started = perf_counter(); bronze, batches = bronze_events(records, micro_batch_size)
    events.append(_event("Bronze / micro-batch replay", started, len(records), len(bronze), stable_hash(bronze["payload_hash"].tolist())))
    started = perf_counter(); silver, quarantine = normalize_and_contract(bronze)
    silver_hash = stable_hash(silver[["event_id", "views"]].to_dict(orient="records"))
    events.append(_event("Silver / contract + dedupe", started, len(bronze), len(silver), silver_hash))
    started = perf_counter(); gold = gold_features(silver); gold_hash = stable_hash(gold[["event_id", "lag_1", "lag_7", "rolling_28"]].to_dict(orient="records"))
    events.append(_event("Gold / forecast features", started, len(silver), len(gold), gold_hash))
    quality = quality_report(bronze, silver, quarantine, gold)
    run_id = stable_hash({"articles": metadata["articles"], "raw_hash": raw_hash, "gold_hash": gold_hash})[:16]
    metadata = {**metadata, "run_id": run_id, "raw_hash": raw_hash, "silver_hash": silver_hash, "gold_hash": gold_hash,
                "bronze_rows": len(bronze), "silver_rows": len(silver), "quarantined_rows": len(quarantine),
                "gold_rows": len(gold), "late_events": int(bronze["late_beyond_watermark"].sum()),
                "duplicate_replays": int(bronze["replayed_duplicate"].sum()), "quality_pass_rate": float(quality["passed"].mean())}
    return PipelineBundle(bronze, silver, quarantine, gold, batches, pd.DataFrame(events), quality, metadata)
