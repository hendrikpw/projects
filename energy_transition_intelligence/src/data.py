"""World Bank ingestion with a deterministic, clearly labelled fallback."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


API_ROOT = "https://api.worldbank.org/v2"

INDICATORS = {
    "EG.ELC.RNEW.ZS": {
        "label": "Renewable electricity output",
        "short": "Renewable output",
        "unit": "% of total electricity output",
        "weight": 0.50,
        "direction": "higher",
    },
    "EN.ATM.CO2E.PC": {
        "label": "CO₂ emissions per capita",
        "short": "CO₂ per capita",
        "unit": "metric tons per capita",
        "weight": 0.30,
        "direction": "lower",
    },
    "EG.EGY.PRIM.PP.KD": {
        "label": "Energy intensity of primary energy",
        "short": "Energy intensity",
        "unit": "MJ per 2021 PPP dollar of GDP",
        "weight": 0.20,
        "direction": "lower",
    },
    "EG.USE.ELEC.KH.PC": {
        "label": "Electric power consumption",
        "short": "Power consumption",
        "unit": "kWh per capita",
        "weight": 0.00,
        "direction": "context",
    },
}

COUNTRIES = {
    "AUT": "Austria",
    "BEL": "Belgium",
    "CHE": "Switzerland",
    "CZE": "Czechia",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "HRV": "Croatia",
    "IRL": "Ireland",
    "ITA": "Italy",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "SWE": "Sweden",
}


def _parse_indicator_payload(payload: object, indicator_code: str) -> list[dict]:
    """Convert one World Bank V2 response into tidy observations."""
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError(f"Unexpected World Bank response for {indicator_code}")

    meta = INDICATORS[indicator_code]
    rows = []
    for item in payload[1]:
        value = item.get("value")
        code = item.get("countryiso3code")
        if value is None or code not in COUNTRIES:
            continue
        rows.append(
            {
                "country_code": code,
                "country": COUNTRIES[code],
                "year": int(item["date"]),
                "indicator_code": indicator_code,
                "indicator": meta["label"],
                "value": float(value),
                "unit": meta["unit"],
                "is_demo": False,
            }
        )
    return rows


def fetch_world_bank_data(
    country_codes: list[str] | None = None,
    start_year: int = 2010,
    end_year: int = 2024,
    timeout: int = 12,
) -> pd.DataFrame:
    """Fetch annual World Development Indicators without credentials."""
    codes = country_codes or list(COUNTRIES)
    invalid = sorted(set(codes) - set(COUNTRIES))
    if invalid:
        raise ValueError(f"Unsupported country codes: {', '.join(invalid)}")

    country_path = ";".join(codes)
    rows: list[dict] = []
    with requests.Session() as session:
        for indicator_code in INDICATORS:
            response = session.get(
                f"{API_ROOT}/country/{country_path}/indicator/{indicator_code}",
                params={
                    "format": "json",
                    "date": f"{start_year}:{end_year}",
                    "per_page": 20_000,
                    "source": 2,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            rows.extend(_parse_indicator_payload(response.json(), indicator_code))

    data = pd.DataFrame(rows)
    if data.empty or data["indicator_code"].nunique() < 3:
        raise ValueError("World Bank returned insufficient indicator coverage")
    return data.sort_values(["country", "indicator_code", "year"]).reset_index(drop=True)


def build_demo_data() -> pd.DataFrame:
    """Create stable synthetic observations for graceful offline demonstration."""
    profiles = {
        "AUT": (78, 6.9, 3.3, 8350),
        "BEL": (25, 7.8, 3.8, 7650),
        "CHE": (65, 4.2, 2.4, 7350),
        "CZE": (14, 8.6, 4.4, 6450),
        "DEU": (44, 7.2, 3.6, 6950),
        "DNK": (68, 5.4, 2.7, 5850),
        "ESP": (47, 4.8, 3.2, 5150),
        "EST": (33, 9.8, 4.8, 6850),
        "FIN": (52, 7.1, 4.2, 14_100),
        "FRA": (24, 4.5, 3.1, 6850),
        "GBR": (41, 5.1, 3.0, 5150),
        "ITA": (39, 5.3, 3.4, 4950),
        "NLD": (31, 8.0, 3.5, 6750),
        "NOR": (98, 7.5, 2.5, 22_400),
        "POL": (21, 8.9, 4.6, 4450),
        "PRT": (61, 4.0, 3.0, 4950),
        "SWE": (69, 3.5, 2.2, 12_800),
    }
    rows = []
    for country_index, (code, profile) in enumerate(profiles.items()):
        renewable, co2, intensity, power = profile
        for year in range(2015, 2024):
            step = year - 2023
            values = {
                "EG.ELC.RNEW.ZS": renewable + step * (1.2 + country_index % 3 * 0.12),
                "EN.ATM.CO2E.PC": co2 - step * 0.12,
                "EG.EGY.PRIM.PP.KD": intensity - step * 0.045,
                "EG.USE.ELEC.KH.PC": power * (1 + step * 0.004),
            }
            for indicator_code, value in values.items():
                meta = INDICATORS[indicator_code]
                rows.append(
                    {
                        "country_code": code,
                        "country": COUNTRIES[code],
                        "year": year,
                        "indicator_code": indicator_code,
                        "indicator": meta["label"],
                        "value": round(max(float(value), 0.0), 3),
                        "unit": meta["unit"],
                        "is_demo": True,
                    }
                )
    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame, dict]:
    """Return live data when possible and deterministic demo data otherwise."""
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        data = fetch_world_bank_data()
        metadata = {
            "mode": "live",
            "retrieved_at": retrieved_at,
            "message": "Live World Bank World Development Indicators",
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        data = build_demo_data()
        metadata = {
            "mode": "demo",
            "retrieved_at": retrieved_at,
            "message": f"Reproducible demo fallback ({type(exc).__name__})",
        }
    return data, metadata
