"""Bounded UCI archive ingestion with retry, validation and deterministic fallback."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

SOURCE_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
MEMBER = "ai4i2020.csv"
MAX_BYTES = 2_000_000


def _download(retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(SOURCE_URL, timeout=(5, 20), headers={"User-Agent": "portfolio-maintenance-pipeline/1.0"})
            response.raise_for_status()
            if not 1_000 < len(response.content) <= MAX_BYTES:
                raise ValueError("archive size outside safety bounds")
            return response.content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(0.25 * 2**attempt)
    raise RuntimeError(f"source unavailable after {retries} attempts: {last}")


def _parse_archive(payload: bytes) -> pd.DataFrame:
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError("payload is not a ZIP archive")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if names != [MEMBER] or archive.getinfo(MEMBER).file_size > MAX_BYTES:
            raise ValueError("archive member allowlist or size check failed")
        return pd.read_csv(archive.open(MEMBER), encoding="utf-8-sig")


def _fallback(n: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(601)
    uid = np.arange(1, n + 1)
    kind = rng.choice(["L", "M", "H"], n, p=[0.60, 0.30, 0.10])
    air = rng.normal(300.0, 2.0, n).round(1)
    process = (air + rng.normal(10.0, 1.0, n)).round(1)
    rpm = np.clip(rng.normal(1535, 175, n), 1100, 2900).round().astype(int)
    torque = np.clip(rng.normal(40, 10, n), 3, 77).round(1)
    wear = (uid * 3 % 254).astype(int)
    twf = (wear > 235) & (rng.random(n) < 0.11)
    hdf = ((process - air) < 8.8) & (rpm < 1450) & (rng.random(n) < 0.22)
    power = rpm * torque
    pwf = ((power < 35_000) | (power > 72_000)) & (rng.random(n) < 0.16)
    osf = ((wear * torque) > 10_500) & (rng.random(n) < 0.20)
    rnf = rng.random(n) < 0.002
    failure = twf | hdf | pwf | osf | rnf
    return pd.DataFrame({"UDI": uid, "Product ID": [f"{t}{i:05d}" for i, t in zip(uid, kind)], "Type": kind,
        "Air temperature [K]": air, "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear,
        "Machine failure": failure.astype(int), "TWF": twf.astype(int), "HDF": hdf.astype(int),
        "PWF": pwf.astype(int), "OSF": osf.astype(int), "RNF": rnf.astype(int)})


def load_dataset() -> tuple[pd.DataFrame, dict]:
    try:
        payload = _download()
        frame = _parse_archive(payload)
        mode, reason, digest = "live", "", hashlib.sha256(payload).hexdigest()
    except Exception as exc:
        frame, mode, reason = _fallback(), "demo", str(exc)
        digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return frame, {"mode": mode, "fallback_reason": reason, "source_url": SOURCE_URL,
        "dataset_page": DATASET_PAGE, "source_hash": digest,
        "retrieved_at": datetime.now(timezone.utc).isoformat()}
