import pandas as pd

from biodiversity_observation_intelligence.src.analytics import (
    add_quality_features,
    cluster_summary,
    facet_table,
    grid_overlap,
    quality_report,
    spatial_clusters,
    summary_metrics,
)
from biodiversity_observation_intelligence.src.data import build_demo_data, parse_occurrences


NAMES = ["Erinaceus europaeus", "Lutra lutra"]


def test_parser_flattens_occurrence_and_license():
    frame = parse_occurrences(
        [
            {
                "key": 1,
                "gbifID": "1",
                "taxonKey": 2,
                "scientificName": "Species test",
                "decimalLatitude": 48.7,
                "decimalLongitude": 9.1,
                "eventDate": "2026-07-01",
                "basisOfRecord": "HUMAN_OBSERVATION",
                "license": "http://creativecommons.org/licenses/by/4.0/legalcode",
                "issues": ["COORDINATE_ROUNDED"],
            }
        ],
        "Species test",
    )
    assert frame.iloc[0]["license"] == "CC BY"
    assert frame.iloc[0]["issues_count"] == 1


def test_quality_score_is_bounded_and_reconcilable():
    data, _, _ = build_demo_data(NAMES)
    scored = add_quality_features(data)
    assert scored["quality_score"].between(0, 100).all()
    assert scored.loc[scored["issue_free"], "quality_score"].median() >= scored.loc[~scored["issue_free"], "quality_score"].median()


def test_summary_keeps_indexed_total_separate_from_sample():
    data, _, metadata = build_demo_data(NAMES)
    scored = add_quality_features(data)
    metrics = summary_metrics(scored, metadata)
    assert metrics["sample_records"] == len(data)
    assert metrics["indexed_records"] > metrics["sample_records"]


def test_facets_are_normalized_within_species():
    _, facets, _ = build_demo_data(NAMES)
    months = facet_table(facets, "MONTH")
    totals = months.groupby("query_name")["share"].sum()
    assert totals.round(6).eq(100).all()


def test_spatial_clusters_and_summary_reconcile():
    data, _, _ = build_demo_data(NAMES)
    clustered = spatial_clusters(add_quality_features(data), radius_km=180, minimum_records=5)
    summary = cluster_summary(clustered)
    assert len(clustered) == len(data)
    assert not summary.empty
    assert summary["records"].sum() <= len(data)


def test_grid_overlap_has_one_pair_and_bounded_score():
    data, _, _ = build_demo_data(NAMES)
    overlap = grid_overlap(data, 2.0)
    assert len(overlap) == 1
    assert overlap.iloc[0]["jaccard_overlap"] >= 0
    assert overlap.iloc[0]["jaccard_overlap"] <= 100


def test_quality_report_has_one_row_per_species():
    data, _, _ = build_demo_data(NAMES)
    report = quality_report(add_quality_features(data))
    assert set(report["query_name"]) == set(NAMES)
    assert report["event_date_coverage"].between(0, 100).all()


def test_demo_data_is_deterministic():
    first = build_demo_data(NAMES)
    second = build_demo_data(NAMES)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
