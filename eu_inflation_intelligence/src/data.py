"""Eurostat HICP ingestion with validation and a deterministic fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd
import requests


API_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/"
    "data/prc_hicp_minr"
)
DATASET_URL = "https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_minr/default/table"
DATASET_DOI = "https://doi.org/10.2908/PRC_HICP_MINR"

GEO_CODES = [
    "EA",
    "AT",
    "BE",
    "DE",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "HR",
    "IE",
    "IT",
    "LT",
    "NL",
    "PL",
    "PT",
    "SE",
]

CATEGORY_CODES = [
    "TOTAL",
    "CP01",
    "CP02",
    "CP03",
    "CP04",
    "CP05",
    "CP06",
    "CP07",
    "CP08",
    "CP09",
    "CP10",
    "CP11",
    "CP12",
    "CP13",
]

CATEGORY_LABELS = {
    "TOTAL": "All items",
    "CP01": "Food & non-alcoholic beverages",
    "CP02": "Alcohol, tobacco & narcotics",
    "CP03": "Clothing & footwear",
    "CP04": "Housing, water, electricity & fuels",
    "CP05": "Furnishings & household equipment",
    "CP06": "Health",
    "CP07": "Transport",
    "CP08": "Information & communication",
    "CP09": "Recreation, sport & culture",
    "CP10": "Education services",
    "CP11": "Restaurants & accommodation",
    "CP12": "Insurance & financial services",
    "CP13": "Personal care & miscellaneous services",
}

GEO_LABELS = {
    "EA": "Euro area",
    "AT": "Austria",
    "BE": "Belgium",
    "DE": "Germany",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GR": "Greece",
    "HR": "Croatia",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "SE": "Sweden",
}


def _ordered_codes(dimension: dict) -> list[str]:
    index = dimension.get("category", {}).get("index", {})
    if isinstance(index, list):
        return [str(code) for code in index]
    return [
        str(code)
        for code, _ in sorted(index.items(), key=lambda item: int(item[1]))
    ]


def parse_jsonstat(payload: dict) -> pd.DataFrame:
    """Decode a JSON-stat response without assuming a fixed dimension order."""
    dimension_ids = payload.get("id")
    sizes = payload.get("size")
    dimensions = payload.get("dimension")
    values = payload.get("value")
    if (
        not isinstance(dimension_ids, list)
        or not isinstance(sizes, list)
        or not isinstance(dimensions, dict)
        or not isinstance(values, (dict, list))
        or len(dimension_ids) != len(sizes)
    ):
        raise ValueError("Unexpected Eurostat JSON-stat response")

    codes_by_dimension = [_ordered_codes(dimensions[name]) for name in dimension_ids]
    if any(len(codes) != int(size) for codes, size in zip(codes_by_dimension, sizes)):
        raise ValueError("Eurostat dimension sizes do not match their categories")

    rows: list[dict] = []
    for position, coordinates in enumerate(product(*codes_by_dimension)):
        value = (
            values[position]
            if isinstance(values, list) and position < len(values)
            else values.get(str(position))
            if isinstance(values, dict)
            else None
        )
        if value is None:
            continue
        row = dict(zip(dimension_ids, coordinates))
        row["rate"] = value
        for dimension_name, code in zip(dimension_ids, coordinates):
            label = (
                dimensions[dimension_name]
                .get("category", {})
                .get("label", {})
                .get(code)
            )
            if label:
                row[f"{dimension_name}_label"] = label
        rows.append(row)
    if not rows:
        raise ValueError("Eurostat returned no observations")
    return pd.DataFrame(rows)


def _prepare_frame(frame: pd.DataFrame, is_demo: bool) -> pd.DataFrame:
    required = {"unit", "coicop18", "geo", "time", "rate"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Eurostat fields: {sorted(missing)}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["time"], format="%Y-%m", errors="coerce")
    result["rate"] = pd.to_numeric(result["rate"], errors="coerce")
    result = result[
        result["unit"].eq("RCH_A")
        & result["coicop18"].isin(CATEGORY_CODES)
        & result["geo"].isin(GEO_CODES)
    ].dropna(subset=["date", "rate"])
    if result.empty:
        raise ValueError("No valid HICP annual-rate observations remained")

    result["category"] = result["coicop18"].map(CATEGORY_LABELS)
    result["country"] = result["geo"].map(GEO_LABELS)
    result["rate"] = result["rate"].clip(-30, 100)
    result["is_demo"] = bool(is_demo)
    return (
        result[
            [
                "date",
                "geo",
                "country",
                "coicop18",
                "category",
                "rate",
                "is_demo",
            ]
        ]
        .drop_duplicates(["date", "geo", "coicop18"], keep="last")
        .sort_values(["date", "geo", "coicop18"])
        .reset_index(drop=True)
    )


def fetch_hicp(
    since_period: str = "2023-01",
    timeout: int = 45,
) -> tuple[pd.DataFrame, dict]:
    """Fetch a bounded selection of monthly HICP annual rates from Eurostat."""
    params: list[tuple[str, str]] = [
        ("lang", "en"),
        ("freq", "M"),
        ("unit", "RCH_A"),
        ("sinceTimePeriod", since_period),
    ]
    params.extend(("coicop18", code) for code in CATEGORY_CODES)
    params.extend(("geo", code) for code in GEO_CODES)
    response = requests.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError("Eurostat reported an API query error")
    data = _prepare_frame(parse_jsonstat(payload), is_demo=False)
    return data, {
        "mode": "live",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_period": data["date"].max().strftime("%Y-%m"),
        "observations": len(data),
        "source_url": response.url,
    }


def build_demo_data() -> pd.DataFrame:
    """Create stable synthetic rates that exercise every interface state."""
    rng = np.random.default_rng(20260728)
    dates = pd.date_range("2023-01-01", "2026-06-01", freq="MS")
    country_offset = {
        code: offset
        for code, offset in zip(GEO_CODES, np.linspace(-0.75, 1.1, len(GEO_CODES)))
    }
    category_offset = {
        "TOTAL": 0.0,
        "CP01": 1.15,
        "CP02": 1.7,
        "CP03": -0.45,
        "CP04": 0.5,
        "CP05": -0.35,
        "CP06": 0.8,
        "CP07": 0.25,
        "CP08": -1.1,
        "CP09": 0.4,
        "CP10": 1.0,
        "CP11": 1.65,
        "CP12": 1.25,
        "CP13": 1.45,
    }
    rows = []
    for geo in GEO_CODES:
        for category in CATEGORY_CODES:
            phase = rng.uniform(0, np.pi * 2)
            for index, date in enumerate(dates):
                disinflation = 4.2 * np.exp(-index / 16)
                seasonal = 0.55 * np.sin(index / 3.7 + phase)
                noise = rng.normal(0, 0.12)
                rate = (
                    1.35
                    + disinflation
                    + country_offset[geo]
                    + category_offset[category]
                    + seasonal
                    + noise
                )
                rows.append(
                    {
                        "unit": "RCH_A",
                        "coicop18": category,
                        "geo": geo,
                        "time": date.strftime("%Y-%m"),
                        "rate": round(rate, 2),
                    }
                )
    return _prepare_frame(pd.DataFrame(rows), is_demo=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    """Return official observations or a clearly labelled synthetic fallback."""
    try:
        return fetch_hicp()
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        data = build_demo_data()
        return data, {
            "mode": "demo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "latest_period": data["date"].max().strftime("%Y-%m"),
            "observations": len(data),
            "fallback_reason": type(exc).__name__,
        }
