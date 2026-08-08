"""Respectful Wikimedia pageview ingestion with deterministic fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
import time
from typing import Any
from urllib.parse import quote

import numpy as np
import requests


BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
DOCS_URL = "https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html"
USAGE_URL = "https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines"
TERMS_URL = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
ARTICLES = {
    "Artificial intelligence": "Artificial_intelligence",
    "Machine learning": "Machine_learning",
    "Data engineering": "Data_engineering",
    "Large language model": "Large_language_model",
    "Apache Airflow": "Apache_Airflow",
    "Retrieval-augmented generation": "Retrieval-augmented_generation",
    "Python": "Python_(programming_language)",
    "Data science": "Data_science",
}


def safe_articles(values: list[str] | tuple[str, ...]) -> list[str]:
    """Allow only documented preset titles and preserve caller order."""
    allowed = set(ARTICLES.values())
    return list(dict.fromkeys(value for value in values if value in allowed))


def _request_json(url: str, attempts: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            response = requests.get(
                url,
                timeout=(4, 25),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "hendrikpw-data-portfolio/1.0 (https://github.com/hendrikpw/projects)",
                },
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise ValueError("Wikimedia returned an unexpected response contract")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.25 * 2**attempt)
    raise RuntimeError(f"Wikimedia request failed after {attempts} attempts: {last_error}")


def _demo_records(articles: list[str], history_days: int, end_date: datetime) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260808)
    rows: list[dict[str, Any]] = []
    for article_index, article in enumerate(articles):
        base = 2_300 + article_index * 1_450
        growth = 0.001 + article_index * 0.00025
        for day in range(history_days):
            date = end_date - timedelta(days=history_days - 1 - day)
            weekly = 1 + 0.16 * math.sin(2 * math.pi * date.weekday() / 7 + article_index)
            monthly = 1 + 0.07 * math.sin(2 * math.pi * day / 30)
            noise = rng.lognormal(0, 0.11)
            spike = 2.4 if (day + article_index * 11) % 73 == 0 else 1
            views = int(max(0, base * (1 + growth * day) * weekly * monthly * noise * spike))
            rows.append({
                "project": "en.wikipedia",
                "article": article,
                "granularity": "daily",
                "timestamp": date.strftime("%Y%m%d00"),
                "access": "all-access",
                "agent": "user",
                "views": views,
            })
    return rows


def fetch_pageviews(articles: list[str], history_days: int = 180) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = safe_articles(articles)
    if not selected:
        raise ValueError("Select at least one supported Wikipedia article")
    if history_days < 90 or history_days > 365:
        raise ValueError("history_days must be between 90 and 365")
    now = datetime.now(timezone.utc)
    end_date = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=history_days - 1)
    records: list[dict[str, Any]] = []
    source_counts: list[dict[str, Any]] = []
    try:
        for article in selected:
            url = (
                f"{BASE_URL}/en.wikipedia.org/all-access/user/{quote(article, safe='()_')}/daily/"
                f"{start_date:%Y%m%d}00/{end_date:%Y%m%d}00"
            )
            payload = _request_json(url)
            items = payload["items"]
            if len(items) < history_days * 0.90:
                raise ValueError(f"insufficient history for {article}: {len(items)} rows")
            records.extend(items)
            source_counts.append({"article": article, "rows": len(items), "endpoint": url})
        mode, fallback_reason = "live", ""
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        records = _demo_records(selected, history_days, end_date)
        source_counts = [{"article": article, "rows": history_days, "endpoint": "deterministic-demo"} for article in selected]
        mode, fallback_reason = "demo", f"{type(exc).__name__}: {exc}"
    return records, {
        "mode": mode,
        "fallback_reason": fallback_reason,
        "retrieved_at": now.isoformat(),
        "articles": selected,
        "history_days": history_days,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "requests": len(selected),
        "source_counts": source_counts,
        "source_url": BASE_URL,
    }
