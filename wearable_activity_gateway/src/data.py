"""Safe UCI HAR ingestion with an atomic deterministic fallback."""
from __future__ import annotations

import hashlib
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones"
DOI = "https://doi.org/10.24432/C54S4K"
ARCHIVE = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ACTIVITIES = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS", 4: "SITTING", 5: "STANDING", 6: "LAYING"}
PREFIX = "UCI HAR Dataset/"
REQUIRED = {
    PREFIX + "features.txt", PREFIX + "activity_labels.txt",
    PREFIX + "train/X_train.txt", PREFIX + "train/y_train.txt", PREFIX + "train/subject_train.txt",
    PREFIX + "test/X_test.txt", PREFIX + "test/y_test.txt", PREFIX + "test/subject_test.txt",
}


def _download(timeout: int = 45, attempts: int = 3, max_bytes: int = 80_000_000) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(ARCHIVE, timeout=timeout, headers={"User-Agent": "hendrik-data-portfolio/1.0"})
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"transient HTTP {response.status_code}", response=response)
            response.raise_for_status()
            content = response.content
            if not (10_000_000 <= len(content) <= max_bytes):
                raise ValueError(f"archive size outside safe bounds: {len(content)}")
            if not content.startswith(b"PK"):
                raise ValueError("payload is not a ZIP archive")
            return content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(.4 * (2**attempt))
    raise RuntimeError(f"UCI archive unavailable after {attempts} attempts: {last}")


def _safe_archive(content: bytes) -> zipfile.ZipFile:
    archive = zipfile.ZipFile(io.BytesIO(content))
    def validate(candidate: zipfile.ZipFile) -> None:
        for info in candidate.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if info.filename.startswith(("/", "\\")) or ".." in parts:
                raise ValueError("unsafe archive path")
        if sum(item.file_size for item in candidate.infolist()) > 300_000_000:
            raise ValueError("expanded archive exceeds safe limit")
    validate(archive)
    # UCI currently wraps the original dataset ZIP in a repository ZIP.
    if not REQUIRED <= set(archive.namelist()) and "UCI HAR Dataset.zip" in archive.namelist():
        archive = zipfile.ZipFile(io.BytesIO(archive.read("UCI HAR Dataset.zip")))
        validate(archive)
    names = set(archive.namelist())
    if not REQUIRED <= names:
        raise ValueError(f"archive contract missing {sorted(REQUIRED - names)}")
    return archive


def _read_matrix(archive: zipfile.ZipFile, path: str, dtype=np.float32) -> np.ndarray:
    return np.loadtxt(io.BytesIO(archive.read(path)), dtype=dtype)


def _feature_names(archive: zipfile.ZipFile) -> list[str]:
    names = []
    for line in archive.read(PREFIX + "features.txt").decode().splitlines():
        index, raw = line.split(maxsplit=1)
        clean = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_").lower()
        names.append(f"f{int(index):03d}_{clean}")
    return names


def _partition(archive: zipfile.ZipFile, split: str, names: list[str]) -> pd.DataFrame:
    x = _read_matrix(archive, f"{PREFIX}{split}/X_{split}.txt")
    y = _read_matrix(archive, f"{PREFIX}{split}/y_{split}.txt", np.int16).astype(int)
    subjects = _read_matrix(archive, f"{PREFIX}{split}/subject_{split}.txt", np.int16).astype(int)
    if x.ndim != 2 or x.shape[1] != len(names) or len(x) != len(y) or len(y) != len(subjects):
        raise ValueError("UCI partition dimensions do not reconcile")
    frame = pd.DataFrame(x, columns=names)
    frame.insert(0, "activity", pd.Series(y).map(ACTIVITIES))
    frame.insert(0, "activity_id", y)
    frame.insert(0, "subject_id", subjects)
    frame.insert(0, "source_split", split)
    frame.insert(0, "source_row", np.arange(len(frame), dtype=int))
    return frame


def fetch_live() -> tuple[pd.DataFrame, dict]:
    content = _download()
    archive = _safe_archive(content)
    names = _feature_names(archive)
    if len(names) != 561:
        raise ValueError(f"expected 561 features, found {len(names)}")
    raw = pd.concat([_partition(archive, "train", names), _partition(archive, "test", names)], ignore_index=True)
    digest = hashlib.sha256(content).hexdigest()
    return raw, {"mode": "live", "source_url": ARCHIVE, "source_hash": digest, "source_bytes": len(content), "features": len(names), "subjects": int(raw.subject_id.nunique()), "license": "CC BY 4.0", "fallback_reason": ""}


def fallback_data(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Generate separable six-activity sensor windows with subject variation."""
    rng = np.random.default_rng(seed)
    names = [f"f{i:03d}_demo_sensor_feature" for i in range(1, 49)]
    rows = []
    class_profiles = rng.normal(0, .35, size=(6, len(names)))
    for subject in range(1, 31):
        subject_bias = rng.normal(0, .035, len(names))
        for activity_id, activity in ACTIVITIES.items():
            for window in range(10):
                values = np.clip(class_profiles[activity_id - 1] + subject_bias + rng.normal(0, .08, len(names)), -1, 1)
                rows.append([window, "train" if subject <= 21 else "test", subject, activity_id, activity, *values])
    raw = pd.DataFrame(rows, columns=["source_row", "source_split", "subject_id", "activity_id", "activity", *names])
    digest = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    return raw, {"mode": "demo", "source_url": ARCHIVE, "source_hash": digest, "source_bytes": int(raw.memory_usage(deep=True).sum()), "features": len(names), "subjects": 30, "license": "CC BY 4.0", "fallback_reason": "Live UCI archive was unavailable or failed its contract."}


def load_source() -> tuple[pd.DataFrame, dict]:
    try:
        return fetch_live()
    except Exception as exc:
        raw, metadata = fallback_data()
        metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return raw, metadata
