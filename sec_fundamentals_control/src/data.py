"""Bounded SEC Company Facts ingestion with retries and deterministic fallback."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

API_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
API_DOCS = "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"
FAIR_ACCESS = "https://www.sec.gov/about/developer-resources"
USER_AGENT = "hendrikpw-projects sec-fundamentals-control/1.0 https://github.com/hendrikpw/projects"
COMPANIES = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
    "AMZN": "0001018724", "META": "0001326801", "NVDA": "0001045810",
    "INTC": "0000050863", "IBM": "0000051143",
}
MAX_BYTES = 8_000_000


def _get(cik: str, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            response = requests.get(
                API_TEMPLATE.format(cik=cik),
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
                timeout=(5, 35),
            )
            response.raise_for_status()
            if not 10_000 < len(response.content) <= MAX_BYTES:
                raise ValueError("Company Facts payload outside safety bounds")
            return response.content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(0.4 * (2**attempt))
    raise RuntimeError(f"SEC source unavailable after {retries} attempts: {last}")


def _fallback() -> dict[str, bytes]:
    rng = np.random.default_rng(20260815)
    payloads: dict[str, bytes] = {}
    for company_index, (ticker, cik) in enumerate(COMPANIES.items()):
        facts: dict[str, dict] = {}
        concepts = {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [],
            "NetIncomeLoss": [], "Assets": [], "StockholdersEquity": [],
        }
        revenue = 18e9 * (1 + company_index * 0.32)
        assets = revenue * 4.2
        for year in range(2017, 2026):
            for quarter in range(1, 5):
                growth = 1 + 0.018 + rng.normal(0, 0.025)
                revenue *= growth
                assets *= 1 + rng.normal(0.012, 0.01)
                margin = 0.13 + company_index * 0.012 + rng.normal(0, 0.018)
                end = f"{year}-{quarter * 3:02d}-28"
                common = {"fy": year, "fp": f"Q{quarter}", "form": "10-Q", "filed": f"{year}-{min(12, quarter*3+1):02d}-25", "accn": f"demo-{ticker}-{year}-{quarter}"}
                concepts["RevenueFromContractWithCustomerExcludingAssessedTax"].append({**common, "frame": f"CY{year}Q{quarter}", "end": end, "val": round(revenue)})
                concepts["NetIncomeLoss"].append({**common, "frame": f"CY{year}Q{quarter}", "end": end, "val": round(revenue * margin)})
                concepts["Assets"].append({**common, "frame": f"CY{year}Q{quarter}I", "end": end, "val": round(assets)})
                concepts["StockholdersEquity"].append({**common, "frame": f"CY{year}Q{quarter}I", "end": end, "val": round(assets * (0.58 - company_index * 0.025))})
        for concept, values in concepts.items():
            facts[concept] = {"label": concept, "units": {"USD": values}}
        body = {"cik": int(cik), "entityName": f"{ticker} Demonstration Corp", "facts": {"us-gaap": facts}}
        payloads[ticker] = json.dumps(body, separators=(",", ":")).encode()
    return payloads


def load_payloads() -> tuple[dict[str, bytes], dict]:
    payloads: dict[str, bytes] = {}
    errors: list[str] = []
    try:
        for ticker, cik in COMPANIES.items():
            payloads[ticker] = _get(cik)
            time.sleep(0.12)
        mode = "live"
    except Exception as exc:
        payloads = _fallback()
        errors.append(str(exc))
        mode = "demo"
    digest = hashlib.sha256(b"".join(t.encode() + payloads[t] for t in sorted(payloads))).hexdigest()
    return payloads, {
        "mode": mode, "fallback_reason": "; ".join(errors), "source_hash": digest,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "company_count": len(payloads),
        "api_template": API_TEMPLATE, "api_docs": API_DOCS,
    }
