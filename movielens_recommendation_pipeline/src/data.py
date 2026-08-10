"""Safe download, extraction and deterministic fallback for MovieLens."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests


ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
README_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html"
DATASETS_URL = "https://grouplens.org/datasets/movielens/"
CITATION_URL = "https://doi.org/10.1145/2827872"
REQUIRED = {
    "ml-latest-small/movies.csv": {"movieId", "title", "genres"},
    "ml-latest-small/ratings.csv": {"userId", "movieId", "rating", "timestamp"},
    "ml-latest-small/tags.csv": {"userId", "movieId", "tag", "timestamp"},
    "ml-latest-small/links.csv": {"movieId", "imdbId", "tmdbId"},
}
USER_AGENT = "hendrikpw-portfolio/1.0 (research portfolio; github.com/hendrikpw/projects)"


def _download(attempts: int = 2) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(ZIP_URL, headers={"User-Agent": USER_AGENT}, timeout=(8, 30))
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            if len(response.content) < 100_000:
                raise ValueError("archive was unexpectedly small")
            return response.content
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(.4 * (2**attempt))
    raise RuntimeError(f"MovieLens download failed after {attempts} attempts: {last}")


def _parse_archive(content: bytes) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        missing = set(REQUIRED) - names
        if missing:
            raise ValueError(f"archive missing required files: {sorted(missing)}")
        for path, columns in REQUIRED.items():
            info = archive.getinfo(path)
            if info.file_size > 50_000_000 or path.startswith("/") or ".." in path.split("/"):
                raise ValueError(f"unsafe archive member: {path}")
            with archive.open(path) as handle:
                frame = pd.read_csv(handle)
            absent = columns - set(frame.columns)
            if absent:
                raise ValueError(f"{path} missing columns: {sorted(absent)}")
            tables[path.rsplit("/", 1)[-1].replace(".csv", "")] = frame
    return tables


def _fallback() -> dict[str, pd.DataFrame]:
    """Create deterministic source-shaped user/movie interactions."""
    rng = np.random.default_rng(4217)
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Romance", "Animation", "Documentary", "Fantasy", "Crime"]
    movie_rows = []
    for movie_id in range(1, 801):
        chosen = rng.choice(genres, size=int(rng.integers(1, 4)), replace=False)
        movie_rows.append({"movieId": movie_id, "title": f"Demo Film {movie_id:04d} ({1980 + movie_id % 40})", "genres": "|".join(chosen)})
    movies = pd.DataFrame(movie_rows)
    ratings = []
    tags = []
    base = 946684800
    for user_id in range(1, 241):
        preferences = set(rng.choice(genres, size=3, replace=False))
        candidates = movies[movies["genres"].map(lambda value: bool(preferences & set(value.split("|"))))]["movieId"].to_numpy()
        selected = rng.choice(candidates, size=int(rng.integers(35, 75)), replace=False)
        for order, movie_id in enumerate(selected):
            overlap = len(preferences & set(movies.loc[movies["movieId"].eq(movie_id), "genres"].iloc[0].split("|")))
            rating = float(np.clip(np.round((2.4 + .75 * overlap + rng.normal(0, .65)) * 2) / 2, .5, 5))
            timestamp = int(base + user_id * 86400 + order * 604800 + rng.integers(0, 86400))
            ratings.append({"userId": user_id, "movieId": int(movie_id), "rating": rating, "timestamp": timestamp})
        for tag in list(preferences)[:2]:
            tags.append({"userId": user_id, "movieId": int(selected[0]), "tag": tag.lower(), "timestamp": base + user_id * 86400})
    links = pd.DataFrame({"movieId": movies["movieId"], "imdbId": movies["movieId"] + 100000, "tmdbId": movies["movieId"] + 200000})
    return {"movies": movies, "ratings": pd.DataFrame(ratings), "tags": pd.DataFrame(tags), "links": links}


def load_dataset() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    try:
        content = _download()
        tables = _parse_archive(content)
        mode, reason, archive_hash = "live", "", hashlib.sha256(content).hexdigest()
    except (RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        tables = _fallback()
        mode, reason = "demo", str(exc)
        joined = b"".join(table.to_csv(index=False).encode() for table in tables.values())
        archive_hash = hashlib.sha256(joined).hexdigest()
    return tables, {
        "mode": mode, "fallback_reason": reason, "archive_hash": archive_hash,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "source_url": ZIP_URL,
        "dataset_version": "ml-latest-small, generated 2018-09-26",
    }
