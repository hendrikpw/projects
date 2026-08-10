"""Typed Bronze/Silver/Gold contracts, quarantine and lineage."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommendation_pipeline.src.data import load_dataset


@dataclass(frozen=True)
class PipelineBundle:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame
    movie_features: pd.DataFrame
    quarantine: pd.DataFrame
    stages: pd.DataFrame
    quality: pd.DataFrame
    metadata: dict[str, Any]


def _hash_frame(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]): normalized[column] = normalized[column].astype(str)
    payload = normalized.sort_index(axis=1).sort_values(list(normalized.columns), kind="stable").to_dict("records") if len(normalized) else []
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _silver(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    movies = tables["movies"].copy(); ratings = tables["ratings"].copy(); tags = tables["tags"].copy(); links = tables["links"].copy()
    for frame in [movies, ratings, tags, links]:
        frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce").astype("Int64")
    movies["title"] = movies["title"].astype("string").str.strip(); movies["genres"] = movies["genres"].astype("string").str.strip()
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce", utc=True)
    tags["userId"] = pd.to_numeric(tags["userId"], errors="coerce").astype("Int64")
    tags["tag"] = tags["tag"].astype("string").str.strip()
    tags["tagged_at"] = pd.to_datetime(tags["timestamp"], unit="s", errors="coerce", utc=True)
    links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")

    reasons = pd.Series(pd.NA, index=ratings.index, dtype="string")
    rules = [
        (ratings["userId"].isna(), "invalid_user_id"), (ratings["movieId"].isna(), "invalid_movie_id"),
        (ratings["rating"].isna() | ~ratings["rating"].between(.5, 5), "rating_out_of_range"),
        (ratings["rated_at"].isna(), "invalid_timestamp"),
        (~ratings["movieId"].isin(movies["movieId"]), "unknown_movie_reference"),
        (ratings.duplicated(["userId", "movieId", "timestamp"], keep="first"), "duplicate_rating_event"),
    ]
    for mask, reason in rules: reasons.loc[mask & reasons.isna()] = reason
    quarantine = ratings[reasons.notna()].copy(); quarantine["invalid_reason"] = reasons[reasons.notna()]
    ratings = ratings[reasons.isna()].copy()
    movies = movies.dropna(subset=["movieId", "title"]).drop_duplicates("movieId", keep="first")
    tags = tags.dropna(subset=["userId", "movieId", "tagged_at"]).drop_duplicates(["userId", "movieId", "tag", "timestamp"])
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId")
    return {
        "movies": movies[["movieId", "title", "genres"]].sort_values("movieId").reset_index(drop=True),
        "ratings": ratings[["userId", "movieId", "rating", "rated_at", "timestamp"]].sort_values(["userId", "rated_at", "movieId"]).reset_index(drop=True),
        "tags": tags[["userId", "movieId", "tag", "tagged_at", "timestamp"]].reset_index(drop=True),
        "links": links[["movieId", "imdbId", "tmdbId"]].sort_values("movieId").reset_index(drop=True),
    }, quarantine.reset_index(drop=True)


def _gold(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movieId", as_index=False).agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"), unique_users=("userId", "nunique"), first_rating=("rated_at", "min"), last_rating=("rated_at", "max"))
    positive = ratings.assign(positive=ratings["rating"].ge(4)).groupby("movieId", as_index=False)["positive"].sum().rename(columns={"positive":"positive_ratings"})
    tag_stats = tags.groupby("movieId", as_index=False).agg(tag_count=("tag", "size"), unique_tags=("tag", "nunique"))
    gold = movies.merge(stats, on="movieId", how="left").merge(positive, on="movieId", how="left").merge(tag_stats, on="movieId", how="left")
    gold[["rating_count", "unique_users", "positive_ratings", "tag_count", "unique_tags"]] = gold[["rating_count", "unique_users", "positive_ratings", "tag_count", "unique_tags"]].fillna(0).astype(int)
    gold["mean_rating"] = gold["mean_rating"].fillna(0.0)
    gold["year"] = pd.to_numeric(gold["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    gold["genre_count"] = gold["genres"].str.split("|").str.len()
    gold["popularity_share"] = gold["positive_ratings"] / max(gold["positive_ratings"].sum(), 1)
    gold["novelty_bits"] = -np.log2(gold["popularity_share"].clip(lower=1 / max(gold["positive_ratings"].sum(), 1)))
    return gold


def _checks(tables: dict[str, pd.DataFrame], silver: dict[str, pd.DataFrame], quarantine: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    ratings, movies, tags = silver["ratings"], silver["movies"], silver["tags"]
    checks = [
        ("required_tables", set(tables) == {"movies", "ratings", "tags", "links"}, "four contracted source files"),
        ("rating_reconciliation", len(tables["ratings"]) == len(ratings) + len(quarantine), f"{len(tables['ratings']):,} = {len(ratings):,} + {len(quarantine):,}"),
        ("movie_identity", movies["movieId"].is_unique and movies["movieId"].notna().all(), f"{len(movies):,} unique movies"),
        ("rating_identity", not ratings.duplicated(["userId", "movieId", "timestamp"]).any(), f"{len(ratings):,} unique rating events"),
        ("rating_domain", ratings["rating"].between(.5, 5).all() and ((ratings["rating"]*2)%1==0).all(), "0.5–5.0 in half-star increments"),
        ("rating_references", ratings["movieId"].isin(movies["movieId"]).all(), "all rating movie IDs resolve"),
        ("tag_references", tags["movieId"].isin(movies["movieId"]).all(), "all tag movie IDs resolve"),
        ("timestamps_utc", str(ratings["rated_at"].dtype) == "datetime64[ns, UTC]", "rating timestamps parsed as UTC"),
        ("minimum_scale", ratings["userId"].nunique() >= 100 and len(movies) >= 500 and len(ratings) >= 5000, f"{ratings['userId'].nunique():,} users / {len(movies):,} movies / {len(ratings):,} ratings"),
        ("gold_reconciliation", len(gold) == len(movies), f"{len(gold):,} movie feature rows"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def run_pipeline() -> PipelineBundle:
    source, metadata = load_dataset(); ledger=[]
    started=time.perf_counter()
    manifest=pd.DataFrame([{"table":name,"rows":len(frame),"content_hash":_hash_frame(frame)} for name,frame in sorted(source.items())])
    ledger.append(("Bronze", sum(len(x) for x in source.values()), sum(len(x) for x in source.values()), 0, (time.perf_counter()-started)*1000, _hash_frame(manifest)))
    started=time.perf_counter(); silver, quarantine=_silver(source)
    silver_hash=hashlib.sha256("".join(_hash_frame(silver[name]) for name in sorted(silver)).encode()).hexdigest()
    ledger.append(("Silver", sum(len(x) for x in source.values()), sum(len(x) for x in silver.values()), len(quarantine), (time.perf_counter()-started)*1000, silver_hash))
    started=time.perf_counter(); gold=_gold(silver["movies"],silver["ratings"],silver["tags"])
    ledger.append(("Gold",len(silver["movies"]),len(gold),0,(time.perf_counter()-started)*1000,_hash_frame(gold)))
    stages=pd.DataFrame(ledger,columns=["stage","input_rows","output_rows","rejected_rows","duration_ms","content_hash"]); stages["status"]="passed"
    quality=_checks(source,silver,quarantine,gold)
    if not quality["passed"].all(): raise RuntimeError("data product withheld; failed gates: "+", ".join(quality.loc[~quality["passed"],"check"]))
    hashes=dict(zip(stages["stage"].str.lower(),stages["content_hash"])); run_id=hashlib.sha256((metadata["archive_hash"]+"".join(hashes.values())).encode()).hexdigest()[:16]
    metadata={**metadata,**{f"{k}_hash":v for k,v in hashes.items()},"run_id":run_id,"quality_pass_rate":float(quality["passed"].mean()),"quarantine_rows":len(quarantine),"users":int(silver["ratings"]["userId"].nunique()),"movies":len(silver["movies"]),"ratings":len(silver["ratings"]),"tags":len(silver["tags"])}
    return PipelineBundle(silver["movies"],silver["ratings"],silver["tags"],silver["links"],gold,quarantine,stages,quality,metadata)
