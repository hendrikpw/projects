"""Repository health, delivery-flow and contributor-concentration analytics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kaplan_meier(records: pd.DataFrame) -> pd.DataFrame:
    """Estimate a censoring-aware unresolved share from durations and outcomes."""
    required = {"duration_days", "event_observed"}
    if records.empty or not required.issubset(records.columns):
        return pd.DataFrame(columns=["day", "unresolved_share", "at_risk", "events"])
    frame = records[list(required)].copy()
    frame["duration_days"] = pd.to_numeric(frame["duration_days"], errors="coerce")
    frame["event_observed"] = pd.to_numeric(frame["event_observed"], errors="coerce").fillna(0).astype(int)
    frame = frame[frame["duration_days"].notna() & frame["duration_days"].ge(0)]
    if frame.empty:
        return pd.DataFrame(columns=["day", "unresolved_share", "at_risk", "events"])
    survival = 1.0
    rows = [{"day": 0.0, "unresolved_share": 100.0, "at_risk": len(frame), "events": 0}]
    for time in sorted(frame.loc[frame["event_observed"].eq(1), "duration_days"].unique()):
        at_risk = int(frame["duration_days"].ge(time).sum())
        events = int((frame["duration_days"].eq(time) & frame["event_observed"].eq(1)).sum())
        if at_risk:
            survival *= 1 - events / at_risk
        rows.append({"day": float(time), "unresolved_share": survival * 100, "at_risk": at_risk, "events": events})
    return pd.DataFrame(rows)


def km_percentile(curve: pd.DataFrame, percentile: float = 0.5) -> float | None:
    """Return the first day on which estimated unresolved share reaches a threshold."""
    if curve.empty:
        return None
    threshold = (1 - float(percentile)) * 100
    reached = curve[curve["unresolved_share"].le(threshold)]
    return None if reached.empty else float(reached.iloc[0]["day"])


def contributor_concentration(contributors: pd.DataFrame) -> dict:
    """Calculate top shares, HHI and effective contributor count on the API sample."""
    if contributors.empty or "contributions" not in contributors:
        return {"contributors": 0, "top1_share": 0.0, "top5_share": 0.0, "hhi": 0.0, "effective_contributors": 0.0}
    values = pd.to_numeric(contributors["contributions"], errors="coerce").fillna(0).clip(lower=0)
    total = float(values.sum())
    if total <= 0:
        return {"contributors": len(values), "top1_share": 0.0, "top5_share": 0.0, "hhi": 0.0, "effective_contributors": 0.0}
    shares = values / total
    hhi = float(np.square(shares).sum())
    return {
        "contributors": int(len(values)),
        "top1_share": float(shares.nlargest(1).sum() * 100),
        "top5_share": float(shares.nlargest(5).sum() * 100),
        "hhi": hhi * 10_000,
        "effective_contributors": float(1 / hhi) if hhi > 0 else 0.0,
    }


def monthly_flow(issues: pd.DataFrame, pulls: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Create created/closed/merged monthly flow counts from bounded API records."""
    parts = []
    for kind, frame in [("Issues opened", issues), ("Pull requests opened", pulls)]:
        if not frame.empty:
            part = frame.dropna(subset=["created_at"]).copy()
            part["month"] = part["created_at"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
            parts.append(part.groupby("month").size().rename(kind))
    if not issues.empty:
        part = issues.dropna(subset=["closed_at"]).copy()
        part["month"] = part["closed_at"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        parts.append(part.groupby("month").size().rename("Issues closed"))
    if not pulls.empty:
        part = pulls.dropna(subset=["merged_at"]).copy()
        part["month"] = part["merged_at"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        parts.append(part.groupby("month").size().rename("Pull requests merged"))
    if not parts:
        return pd.DataFrame(columns=["month", "metric", "count"])
    wide = pd.concat(parts, axis=1).fillna(0).sort_index().tail(max(int(months), 1))
    return wide.reset_index().melt("month", var_name="metric", value_name="count")


def issue_age_bands(issues: pd.DataFrame) -> pd.DataFrame:
    """Bucket open issue age for operational backlog triage."""
    if issues.empty:
        return pd.DataFrame(columns=["age_band", "count", "share"])
    opened = issues[issues["event_observed"].eq(0)].copy()
    if opened.empty:
        return pd.DataFrame(columns=["age_band", "count", "share"])
    labels = ["0–30 days", "31–90 days", "91–180 days", "181–365 days", ">365 days"]
    opened["age_band"] = pd.cut(opened["duration_days"], [-1, 30, 90, 180, 365, np.inf], labels=labels)
    result = opened.groupby("age_band", observed=False).size().reindex(labels, fill_value=0).rename("count").reset_index()
    result["share"] = result["count"].div(max(result["count"].sum(), 1)).mul(100)
    return result


def release_cadence(releases: pd.DataFrame) -> dict:
    """Summarize stable-release recency and interval dispersion."""
    if releases.empty:
        return {"releases": 0, "days_since_latest": None, "median_interval_days": None, "cadence_cv": None}
    stable = releases[~releases["prerelease"].fillna(False)].sort_values("published_at")
    if stable.empty:
        stable = releases.sort_values("published_at")
    now = pd.Timestamp.now(tz="UTC")
    latest = stable["published_at"].max()
    intervals = stable["published_at"].diff().dt.total_seconds().div(86400).dropna()
    median = float(intervals.median()) if not intervals.empty else None
    cv = float(intervals.std(ddof=1) / intervals.mean()) if len(intervals) > 1 and intervals.mean() else None
    return {
        "releases": int(len(stable)),
        "days_since_latest": max(float((now - latest).total_seconds() / 86400), 0.0),
        "median_interval_days": median,
        "cadence_cv": cv,
    }


def language_mix(languages: pd.DataFrame) -> pd.DataFrame:
    """Convert GitHub language-byte totals into repository shares."""
    if languages.empty:
        return pd.DataFrame(columns=["language", "bytes", "share"])
    result = languages.copy()
    result["bytes"] = pd.to_numeric(result["bytes"], errors="coerce").fillna(0).clip(lower=0)
    result["share"] = result["bytes"].div(max(float(result["bytes"].sum()), 1)).mul(100)
    return result.sort_values("bytes", ascending=False).reset_index(drop=True)


def _decay_score(days: float | None, half_life: float) -> float:
    if days is None or not np.isfinite(days):
        return 0.0
    return float(100 * math.exp(-math.log(2) * max(days, 0) / half_life))


def repository_pulse(frames: dict[str, pd.DataFrame], metadata: dict) -> tuple[float, pd.DataFrame]:
    """Build a transparent heuristic pulse from five bounded, auditable components."""
    issues, pulls = frames["issues"], frames["pulls"]
    contributors, releases = frames["contributors"], frames["releases"]
    retrieved = pd.to_datetime(metadata.get("retrieved_at"), utc=True, errors="coerce")
    pushed = pd.to_datetime(metadata.get("pushed_at"), utc=True, errors="coerce")
    activity_days = float((retrieved - pushed).total_seconds() / 86400) if pd.notna(retrieved) and pd.notna(pushed) else None
    merged = pulls[pulls["is_merged"]] if not pulls.empty else pulls
    merge_days = float(merged["duration_days"].median()) if not merged.empty else None
    open_issues = issues[issues["event_observed"].eq(0)] if not issues.empty else issues
    recent_share = float(open_issues["duration_days"].le(90).mean() * 100) if not open_issues.empty else 100.0
    concentration = contributor_concentration(contributors)
    cadence = release_cadence(releases)
    components = pd.DataFrame(
        [
            {"component": "Recent activity", "score": _decay_score(activity_days, 30), "weight": 25, "evidence": f"{activity_days:.1f} days since push" if activity_days is not None else "No push date"},
            {"component": "PR delivery", "score": _decay_score(merge_days, 21), "weight": 25, "evidence": f"{merge_days:.1f} median days to merge" if merge_days is not None else "No merged PR in sample"},
            {"component": "Backlog freshness", "score": recent_share, "weight": 20, "evidence": f"{recent_share:.1f}% of sampled open issues ≤90 days"},
            {"component": "Contributor spread", "score": min(concentration["effective_contributors"] / 15 * 100, 100), "weight": 15, "evidence": f"{concentration['effective_contributors']:.1f} effective contributors"},
            {"component": "Release recency", "score": _decay_score(cadence["days_since_latest"], 120), "weight": 15, "evidence": f"{cadence['days_since_latest']:.1f} days since stable release" if cadence["days_since_latest"] is not None else "No release in sample"},
        ]
    )
    components["weighted_points"] = components["score"] * components["weight"] / 100
    return float(components["weighted_points"].sum()), components


def capacity_scenario(pulls: pd.DataFrame, weekly_capacity: int, weekly_arrivals: float) -> dict:
    """Estimate bounded open-PR sample clearance under a transparent flow scenario."""
    open_count = int((pulls["event_observed"].eq(0)).sum()) if not pulls.empty else 0
    capacity = max(int(weekly_capacity), 0)
    arrivals = max(float(weekly_arrivals), 0.0)
    net = capacity - arrivals
    if open_count == 0:
        weeks = 0.0
        status = "No sampled backlog"
    elif net <= 0:
        weeks = None
        status = "Backlog grows or remains flat"
    else:
        weeks = open_count / net
        status = "Backlog clears in scenario"
    return {"open_prs": open_count, "net_weekly_reduction": net, "clearance_weeks": weeks, "status": status}


def record_audit(issues: pd.DataFrame, pulls: pd.DataFrame) -> pd.DataFrame:
    """Return one sortable operational table for the sampled issues and PRs."""
    rows = []
    for kind, frame in [("Issue", issues), ("Pull request", pulls)]:
        for _, item in frame.iterrows():
            rows.append(
                {
                    "type": kind,
                    "number": int(item["number"]),
                    "title": item["title"],
                    "state": item["state"],
                    "created_at": item["created_at"],
                    "age_or_cycle_days": float(item["duration_days"]),
                    "author": item["author"],
                    "url": item["url"],
                }
            )
    return pd.DataFrame(rows).sort_values("created_at", ascending=False).reset_index(drop=True)
