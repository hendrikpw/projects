"""Safe UCI optical-digits ingestion with a deterministic image fallback."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"
DOI = "https://doi.org/10.24432/C50P49"
ARCHIVE = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
REQUIRED = {"optdigits.tra", "optdigits.tes"}
ALLOWED = REQUIRED | {"optdigits-orig.cv.Z", "optdigits-orig.names", "optdigits-orig.tra.Z", "optdigits-orig.wdep.Z", "optdigits-orig.windep.Z", "optdigits.names", "readme.txt"}
PIXELS = [f"px_{row}_{column}" for row in range(8) for column in range(8)]


def _download(timeout: int = 30, attempts: int = 3, max_bytes: int = 2_000_000) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(ARCHIVE, timeout=timeout, headers={"User-Agent": "hendrik-data-portfolio/1.0"})
            if response.status_code in {429, 500, 502, 503, 504}: raise requests.HTTPError(f"transient HTTP {response.status_code}", response=response)
            response.raise_for_status(); content = response.content
            if not (400_000 <= len(content) <= max_bytes) or not content.startswith(b"PK"): raise ValueError(f"archive contract failed for {len(content)} bytes")
            return content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts: time.sleep(.35 * (2**attempt))
    raise RuntimeError(f"UCI source unavailable after {attempts} attempts: {last}")


def _safe_archive(content: bytes) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(io.BytesIO(content))
    for info in archive.infolist():
        parts = info.filename.replace("\\", "/").split("/")
        if info.filename.startswith(("/", "\\")) or ".." in parts: raise ValueError("unsafe archive path")
    names = set(archive.namelist())
    if not REQUIRED <= names or not names <= ALLOWED: raise ValueError(f"unexpected archive members: {sorted((names - ALLOWED) | (REQUIRED - names))}")
    if sum(item.file_size for item in archive.infolist()) > 5_000_000: raise ValueError("expanded archive exceeds safe limit")
    return archive


def _parse(archive: zipfile.ZipFile, name: str, split: str) -> pd.DataFrame:
    matrix = np.loadtxt(io.BytesIO(archive.read(name)), delimiter=",", dtype=np.int16)
    if matrix.ndim != 2 or matrix.shape[1] != 65: raise ValueError(f"{name} must contain 64 pixels and one label")
    frame = pd.DataFrame(matrix[:, :64], columns=PIXELS)
    frame.insert(0, "label", matrix[:, 64].astype(int)); frame.insert(0, "source_row", np.arange(len(frame))); frame.insert(0, "source_split", split)
    return frame


def fetch_live() -> tuple[pd.DataFrame, dict]:
    content = _download(); archive = _safe_archive(content)
    raw = pd.concat([_parse(archive, "optdigits.tra", "train"), _parse(archive, "optdigits.tes", "test")], ignore_index=True)
    digest = hashlib.sha256(content).hexdigest()
    return raw, {"mode": "live", "source_url": ARCHIVE, "source_hash": digest, "source_bytes": len(content), "train_rows": int((raw.source_split == "train").sum()), "test_rows": int((raw.source_split == "test").sum()), "license": "CC BY 4.0", "fallback_reason": ""}


def _seven_segment(label: int) -> np.ndarray:
    segments = {0:"abcedf",1:"bc",2:"abged",3:"abgcd",4:"fgbc",5:"afgcd",6:"afgecd",7:"abc",8:"abcdefg",9:"abfgcd"}[label]
    image = np.zeros((8, 8), float)
    lines = {"a":(slice(1,2),slice(2,6)),"b":(slice(2,4),slice(6,7)),"c":(slice(4,7),slice(6,7)),"d":(slice(6,7),slice(2,6)),"e":(slice(4,7),slice(1,2)),"f":(slice(2,4),slice(1,2)),"g":(slice(3,4),slice(2,6))}
    for segment in segments: image[lines[segment]] = 14
    return image


def fallback_data(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed); rows = []
    for split, count in [("train", 380), ("test", 180)]:
        for label in range(10):
            prototype = _seven_segment(label)
            for index in range(count):
                image = np.clip(np.rint(prototype + rng.normal(0, 1.3 if split == "train" else 1.6, (8, 8))), 0, 16).astype(int)
                rows.append([split, index, label, *image.ravel()])
    raw = pd.DataFrame(rows, columns=["source_split", "source_row", "label", *PIXELS])
    digest = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    return raw, {"mode": "demo", "source_url": ARCHIVE, "source_hash": digest, "source_bytes": int(raw.memory_usage(deep=True).sum()), "train_rows": 3800, "test_rows": 1800, "license": "CC BY 4.0", "fallback_reason": "Live UCI archive was unavailable or failed its contract."}


def load_source(force_fallback: bool = False) -> tuple[pd.DataFrame, dict]:
    if force_fallback: return fallback_data()
    try: return fetch_live()
    except Exception as exc:
        raw, metadata = fallback_data(); metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"; return raw, metadata
