"""Observation intensity, spatial pattern and data-quality analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6_371.0088


def add_quality_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create explicit completeness and coordinate-quality indicators."""
    result = data.copy()
    uncertainty = pd.to_numeric(result["coordinate_uncertainty_m"], errors="coerce")
    result["has_event_date"] = result["event_date"].notna()
    result["has_uncertainty"] = uncertainty.notna()
    result["uncertainty_acceptable"] = uncertainty.le(10_000).fillna(False)
    result["issue_free"] = result["issues_count"].eq(0)
    result["quality_score"] = (
        result["has_event_date"].astype(int) * 30
        + result["has_uncertainty"].astype(int) * 20
        + result["uncertainty_acceptable"].astype(int) * 25
        + result["issue_free"].astype(int) * 25
    )
    result["record_url"] = np.where(
        result["is_demo"], "", "https://www.gbif.org/occurrence/" + result["occurrence_key"].astype(str)
    )
    return result


def summary_metrics(data: pd.DataFrame, metadata: dict) -> dict:
    """Return bounded-sample and full-index headline measures without conflation."""
    uncertainty = pd.to_numeric(data["coordinate_uncertainty_m"], errors="coerce")
    return {
        "indexed_records": int(metadata.get("indexed_records", 0)),
        "sample_records": len(data),
        "countries": int(data["country"].nunique()),
        "datasets": int(data["dataset_key"].replace("", np.nan).nunique()),
        "issue_free_share": float(data["issue_free"].mean() * 100),
        "median_uncertainty_m": float(uncertainty.median()),
    }


def facet_table(facets: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract one full-query GBIF facet with within-species shares."""
    result = facets[facets["field"].eq(field)].copy()
    if result.empty:
        return result
    totals = result.groupby("query_name")["count"].transform("sum")
    result["share"] = result["count"] / totals.replace(0, np.nan) * 100
    return result.sort_values(["query_name", "count"], ascending=[True, False])


def spatial_clusters(data: pd.DataFrame, radius_km: int = 75, minimum_records: int = 8) -> pd.DataFrame:
    """Group bounded-sample coordinates with haversine DBSCAN per species."""
    frames = []
    for species, group in data.groupby("query_name"):
        valid = group.dropna(subset=["latitude", "longitude"]).copy()
        if len(valid) < minimum_records:
            valid["cluster"] = -1
        else:
            coordinates = np.radians(valid[["latitude", "longitude"]].to_numpy())
            valid["cluster"] = DBSCAN(
                eps=float(radius_km) / EARTH_RADIUS_KM,
                min_samples=max(int(minimum_records), 2),
                metric="haversine",
            ).fit_predict(coordinates)
        valid["cluster_label"] = np.where(
            valid["cluster"].eq(-1), "Outside dense clusters", species + " · C" + valid["cluster"].astype(str)
        )
        frames.append(valid)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cluster_summary(clustered: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-noise spatial observation clusters."""
    valid = clustered[clustered["cluster"] >= 0]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["query_name", "cluster", "cluster_label"], as_index=False)
        .agg(
            records=("record_id", "nunique"),
            center_latitude=("latitude", "mean"),
            center_longitude=("longitude", "mean"),
            countries=("country", "nunique"),
            median_uncertainty_m=("coordinate_uncertainty_m", "median"),
        )
        .sort_values("records", ascending=False)
    )


def grid_overlap(data: pd.DataFrame, grid_degrees: float = 1.0) -> pd.DataFrame:
    """Calculate pairwise Jaccard overlap of occupied sample grid cells."""
    frame = data.dropna(subset=["latitude", "longitude"]).copy()
    size = max(float(grid_degrees), 0.25)
    frame["cell"] = (
        np.floor(frame["latitude"] / size).astype(int).astype(str)
        + ":"
        + np.floor(frame["longitude"] / size).astype(int).astype(str)
    )
    cell_sets = {name: set(group["cell"]) for name, group in frame.groupby("query_name")}
    rows = []
    names = sorted(cell_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            union = cell_sets[left] | cell_sets[right]
            intersection = cell_sets[left] & cell_sets[right]
            rows.append(
                {
                    "species_a": left,
                    "species_b": right,
                    "shared_cells": len(intersection),
                    "union_cells": len(union),
                    "jaccard_overlap": len(intersection) / len(union) * 100 if union else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values("jaccard_overlap", ascending=False)


def quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """Audit sample completeness, uncertainty, issues and record licensing."""
    rows = []
    for species, group in data.groupby("query_name"):
        uncertainty = pd.to_numeric(group["coordinate_uncertainty_m"], errors="coerce")
        rows.append(
            {
                "query_name": species,
                "sample_records": len(group),
                "event_date_coverage": group["has_event_date"].mean() * 100,
                "uncertainty_coverage": group["has_uncertainty"].mean() * 100,
                "within_10km": group["uncertainty_acceptable"].mean() * 100,
                "issue_free": group["issue_free"].mean() * 100,
                "median_uncertainty_m": uncertainty.median(),
                "median_quality_score": group["quality_score"].median(),
            }
        )
    return pd.DataFrame(rows)
