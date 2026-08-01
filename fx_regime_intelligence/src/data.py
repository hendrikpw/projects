"""ECB exchange-rate ingestion, validation and deterministic fallback data."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests


API_ROOT = "https://data-api.ecb.europa.eu/service/data/EXR"
DATASET_URL = "https://data.ecb.europa.eu/data/datasets/EXR"
API_DOCS_URL = "https://data.ecb.europa.eu/help/api/data"
USAGE_POLICY_URL = (
    "https://www.ecb.europa.eu/stats/ecb_statistics/"
    "governance_and_quality_framework/html/usage_policy.en.html"
)
USER_AGENT = "HendrikDataPortfolio/1.0 (https://github.com/hendrikpw/projects)"

CURRENCIES = {
    "USD": "US dollar",
    "GBP": "Pound sterling",
    "CHF": "Swiss franc",
    "JPY": "Japanese yen",
    "CAD": "Canadian dollar",
    "AUD": "Australian dollar",
    "CNY": "Chinese yuan renminbi",
    "SEK": "Swedish krona",
    "NOK": "Norwegian krone",
    "PLN": "Polish zloty",
}


def parse_ecb_csv(content: str, is_demo: bool = False) -> pd.DataFrame:
    """Parse documented ECB CSV fields into validated daily EUR reference rates."""
    frame = pd.read_csv(StringIO(content))
    required = {
        "FREQ",
        "CURRENCY",
        "CURRENCY_DENOM",
        "EXR_TYPE",
        "EXR_SUFFIX",
        "TIME_PERIOD",
        "OBS_VALUE",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing ECB fields: {sorted(missing)}")
    frame = frame[
        frame["FREQ"].eq("D")
        & frame["CURRENCY_DENOM"].eq("EUR")
        & frame["EXR_TYPE"].eq("SP00")
        & frame["EXR_SUFFIX"].eq("A")
    ].copy()
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["TIME_PERIOD"], errors="coerce"),
            "currency": frame["CURRENCY"].astype(str).str.upper(),
            "rate_per_eur": pd.to_numeric(frame["OBS_VALUE"], errors="coerce"),
            "observation_status": frame.get(
                "OBS_STATUS", pd.Series("", index=frame.index)
            ).fillna(""),
            "series_key": frame.get(
                "KEY", pd.Series("", index=frame.index)
            ).fillna(""),
            "is_demo": bool(is_demo),
        }
    )
    result = result[
        result["date"].notna()
        & result["currency"].isin(CURRENCIES)
        & result["rate_per_eur"].gt(0)
    ]
    result = result.sort_values("date").drop_duplicates(
        ["date", "currency"], keep="last"
    )
    if result.empty:
        raise ValueError("No valid ECB observations remained after validation")
    return result.reset_index(drop=True)


def fetch_rates(start_date: str = "2021-01-01", timeout: int = 45) -> tuple[pd.DataFrame, dict]:
    """Fetch a bounded set of daily ECB reference-rate series in one request."""
    key = f"D.{'+'.join(CURRENCIES)}.EUR.SP00.A"
    response = requests.get(
        f"{API_ROOT}/{key}",
        params={"startPeriod": start_date, "format": "csvdata"},
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = parse_ecb_csv(response.text)
    return data, {
        "mode": "live",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": response.url,
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "observations": len(data),
        "currencies": int(data["currency"].nunique()),
    }


def build_demo_data() -> pd.DataFrame:
    """Generate deterministic correlated business-day FX paths for UI resilience."""
    rng = np.random.default_rng(20260801)
    dates = pd.bdate_range("2021-01-04", "2026-07-31")
    codes = list(CURRENCIES)
    initial = np.array([1.22, 0.90, 1.08, 126.0, 1.56, 1.59, 7.90, 10.05, 10.45, 4.55])
    daily_vol = np.array([0.0048, 0.0045, 0.0038, 0.0062, 0.0049, 0.0057, 0.0030, 0.0044, 0.0050, 0.0043])
    common = rng.normal(0, 1, (len(dates), 2))
    idiosyncratic = rng.normal(0, 1, (len(dates), len(codes)))
    loadings = np.array(
        [
            [0.55, 0.15], [0.50, 0.20], [0.42, -0.18], [0.60, -0.22],
            [0.58, 0.12], [0.62, 0.16], [0.28, 0.08], [0.44, -0.05],
            [0.48, -0.02], [0.38, 0.14],
        ]
    )
    shocks = common @ loadings.T + idiosyncratic * 0.68
    returns = shocks * daily_vol
    for start, length, magnitude in [(310, 8, 2.4), (820, 12, 2.8), (1220, 7, 3.1)]:
        returns[start : start + length] *= magnitude
    paths = initial * np.exp(np.cumsum(returns, axis=0))
    rows = []
    for column, currency in enumerate(codes):
        rows.extend(
            {
                "date": date,
                "currency": currency,
                "rate_per_eur": float(paths[index, column]),
                "observation_status": "A",
                "series_key": f"DEMO.D.{currency}.EUR.SP00.A",
                "is_demo": True,
            }
            for index, date in enumerate(dates)
        )
    return pd.DataFrame(rows).sort_values(["date", "currency"]).reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    """Return live ECB data or a clearly labelled reproducible fallback."""
    try:
        return fetch_rates()
    except (requests.RequestException, ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        data = build_demo_data()
        return data, {
            "mode": "demo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "start_date": data["date"].min().date().isoformat(),
            "end_date": data["date"].max().date().isoformat(),
            "observations": len(data),
            "currencies": int(data["currency"].nunique()),
            "fallback_reason": type(exc).__name__,
        }
