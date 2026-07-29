"""Launch Library 2 ingestion with validation and deterministic fallback data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


API_URL = "https://ll.thespacedevs.com/2.3.0/launches/"
DOCS_URL = "https://ll.thespacedevs.com/docs/"
PROVIDER_URL = "https://thespacedevs.com/llapi"
USER_AGENT = "HendrikDataPortfolio/1.0 (public educational analytics)"


def _nested(value: object, *keys: str, default: object = None) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def parse_launches(results: list[dict], is_upcoming: bool, is_demo: bool) -> pd.DataFrame:
    """Flatten the documented launch response into an analytical table."""
    rows = []
    for launch in results:
        status_name = str(_nested(launch, "status", "name", default="Unknown"))
        status_lower = status_name.casefold()
        if "successful" in status_lower:
            outcome = "Success"
        elif "failure" in status_lower or "failed" in status_lower:
            outcome = "Failure"
        else:
            outcome = "Undecided"

        family_items = _nested(
            launch, "rocket", "configuration", "families", default=[]
        )
        family = (
            family_items[0].get("name", "Unknown")
            if isinstance(family_items, list) and family_items
            else "Unknown"
        )
        mission = launch.get("mission") or {}
        pad = launch.get("pad") or {}
        rows.append(
            {
                "launch_id": launch.get("id"),
                "name": launch.get("name"),
                "net": launch.get("net"),
                "last_updated": launch.get("last_updated"),
                "status": status_name,
                "outcome": outcome,
                "provider": _nested(
                    launch, "launch_service_provider", "name", default="Unknown"
                ),
                "provider_type": _nested(
                    launch,
                    "launch_service_provider",
                    "type",
                    "name",
                    default="Unknown",
                ),
                "rocket": _nested(
                    launch,
                    "rocket",
                    "configuration",
                    "full_name",
                    default="Unknown",
                ),
                "rocket_family": family,
                "mission_type": mission.get("type") or "Unspecified",
                "orbit": _nested(mission, "orbit", "name", default="Unspecified"),
                "orbit_abbrev": _nested(
                    mission, "orbit", "abbrev", default="Other"
                ),
                "pad": pad.get("name") or "Unknown",
                "location": _nested(pad, "location", "name", default="Unknown"),
                "country": _nested(pad, "country", "name", default="Unknown"),
                "latitude": pad.get("latitude"),
                "longitude": pad.get("longitude"),
                "probability": launch.get("probability"),
                "fail_reason": launch.get("failreason") or "",
                "source_url": launch.get("url") or "",
                "is_upcoming": bool(is_upcoming),
                "is_demo": bool(is_demo),
            }
        )
    return pd.DataFrame(rows)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "launch_id",
        "name",
        "net",
        "status",
        "outcome",
        "provider",
        "rocket",
        "is_upcoming",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing launch fields: {sorted(missing)}")

    result = frame.copy()
    result["net"] = pd.to_datetime(result["net"], utc=True, errors="coerce")
    result["last_updated"] = pd.to_datetime(
        result["last_updated"], utc=True, errors="coerce"
    )
    for column in ["latitude", "longitude", "probability"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["launch_id", "name", "net"]).copy()
    result = result[
        result["latitude"].isna()
        | (
            result["latitude"].between(-90, 90)
            & result["longitude"].between(-180, 180)
        )
    ]
    result["is_decided"] = result["outcome"].isin(["Success", "Failure"])
    result["is_success"] = result["outcome"].eq("Success")
    result["month"] = result["net"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    result["date"] = result["net"].dt.date
    result = result.drop_duplicates("launch_id", keep="last")
    if result.empty:
        raise ValueError("No valid launches remained after validation")
    return result.sort_values("net", ascending=False).reset_index(drop=True)


def _request_page(params: dict, timeout: int) -> dict:
    response = requests.get(
        API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Unexpected Launch Library 2 response")
    return payload


def fetch_launches(
    history_pages: int = 5,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch recent history and upcoming launches within the public rate limit."""
    now = datetime.now(timezone.utc)
    historical_results: list[dict] = []
    for page in range(max(1, min(history_pages, 5))):
        payload = _request_page(
            {
                "format": "json",
                "mode": "normal",
                "limit": 100,
                "offset": page * 100,
                "ordering": "-net",
                "net__lte": now.isoformat(timespec="seconds"),
            },
            timeout,
        )
        historical_results.extend(payload["results"])
        if not payload.get("next"):
            break

    upcoming_payload = _request_page(
        {
            "format": "json",
            "mode": "normal",
            "limit": 50,
            "ordering": "net",
            "net__gte": now.isoformat(timespec="seconds"),
        },
        timeout,
    )
    historical = parse_launches(historical_results, False, False)
    upcoming = parse_launches(upcoming_payload["results"], True, False)
    data = _prepare_frame(
        pd.DataFrame.from_records(
            historical.to_dict("records") + upcoming.to_dict("records")
        )
    )
    return data, {
        "mode": "live",
        "retrieved_at": now.isoformat(timespec="seconds"),
        "historical_rows": len(historical_results),
        "upcoming_rows": len(upcoming_payload["results"]),
        "api_calls": min(history_pages, 5) + 1,
        "source_url": API_URL,
    }


def build_demo_data() -> pd.DataFrame:
    """Generate a stable, realistic-looking catalog for resilient demonstration."""
    rng = np.random.default_rng(20260729)
    providers = {
        "SpaceX": ("Commercial", "Falcon 9 Block 5", "Falcon", "USA"),
        "CASC": ("Government", "Long March 2C", "Long March", "China"),
        "Roscosmos": ("Government", "Soyuz 2.1b", "Soyuz", "Kazakhstan"),
        "Rocket Lab": ("Commercial", "Electron", "Electron", "New Zealand"),
        "Arianespace": ("Commercial", "Ariane 6", "Ariane", "French Guiana"),
        "ISRO": ("Government", "PSLV", "PSLV", "India"),
        "ULA": ("Commercial", "Vulcan VC2S", "Vulcan", "USA"),
        "JAXA": ("Government", "H3-22S", "H3", "Japan"),
    }
    probabilities = np.array([0.45, 0.15, 0.10, 0.10, 0.06, 0.06, 0.05, 0.03])
    names = np.array(list(providers))
    end = pd.Timestamp("2026-07-29T06:00:00Z")
    rows: list[dict] = []
    for index in range(500):
        provider = str(rng.choice(names, p=probabilities))
        provider_type, rocket, family, country = providers[provider]
        net = end - pd.to_timedelta(rng.uniform(0, 720), unit="D")
        success_probability = 0.975 if provider == "SpaceX" else rng.uniform(0.88, 0.97)
        successful = rng.random() < success_probability
        orbit = str(rng.choice(["Low Earth Orbit", "Geostationary Transfer Orbit", "Sun-Synchronous Orbit", "Suborbital"], p=[0.62, 0.16, 0.17, 0.05]))
        rows.append(
            {
                "id": f"demo-history-{index:04d}",
                "name": f"{rocket} | Synthetic mission {index + 1}",
                "net": net.isoformat(),
                "last_updated": net.isoformat(),
                "status": {"name": "Launch Successful" if successful else "Launch Failure"},
                "launch_service_provider": {
                    "name": provider,
                    "type": {"name": provider_type},
                },
                "rocket": {
                    "configuration": {
                        "full_name": rocket,
                        "families": [{"name": family}],
                    }
                },
                "mission": {
                    "type": str(rng.choice(["Communications", "Earth Science", "Technology", "Government/Top Secret"])),
                    "orbit": {"name": orbit, "abbrev": orbit[:3].upper()},
                },
                "pad": {
                    "name": f"{country} synthetic pad",
                    "location": {"name": f"{country} synthetic site"},
                    "country": {"name": country},
                    "latitude": rng.uniform(-45, 45),
                    "longitude": rng.uniform(-160, 160),
                },
                "probability": None,
                "failreason": "Synthetic failure" if not successful else "",
                "url": "",
            }
        )
    history = parse_launches(rows, False, True)

    upcoming_rows = []
    for index in range(30):
        provider = str(rng.choice(names, p=probabilities))
        provider_type, rocket, family, country = providers[provider]
        net = end + pd.to_timedelta(rng.uniform(1, 90), unit="D")
        upcoming_rows.append(
            {
                "id": f"demo-upcoming-{index:03d}",
                "name": f"{rocket} | Planned synthetic mission {index + 1}",
                "net": net.isoformat(),
                "last_updated": end.isoformat(),
                "status": {"name": "Go for Launch"},
                "launch_service_provider": {"name": provider, "type": {"name": provider_type}},
                "rocket": {"configuration": {"full_name": rocket, "families": [{"name": family}]}},
                "mission": {"type": "Demonstration", "orbit": {"name": "Low Earth Orbit", "abbrev": "LEO"}},
                "pad": {"name": f"{country} synthetic pad", "location": {"name": f"{country} synthetic site"}, "country": {"name": country}, "latitude": rng.uniform(-45, 45), "longitude": rng.uniform(-160, 160)},
                "probability": int(rng.integers(60, 96)),
                "failreason": "",
                "url": "",
            }
        )
    upcoming = parse_launches(upcoming_rows, True, True)
    return _prepare_frame(
        pd.DataFrame.from_records(
            history.to_dict("records") + upcoming.to_dict("records")
        )
    )


def load_data() -> tuple[pd.DataFrame, dict]:
    """Return live launches or a clearly labelled deterministic fallback."""
    try:
        return fetch_launches()
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        data = build_demo_data()
        return data, {
            "mode": "demo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "historical_rows": int((~data["is_upcoming"]).sum()),
            "upcoming_rows": int(data["is_upcoming"].sum()),
            "api_calls": 0,
            "fallback_reason": type(exc).__name__,
        }
