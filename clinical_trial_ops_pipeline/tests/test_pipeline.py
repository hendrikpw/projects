"""Data engineering contract, idempotency and fallback tests."""

from __future__ import annotations

import requests

from clinical_trial_ops_pipeline.src.data import _request_payload, build_demo_studies, fetch_studies, safe_condition
from clinical_trial_ops_pipeline.src.pipeline import (
    FEATURE_COLUMNS, feature_view, normalize_snapshot, quality_report, snapshot_table, stable_hash,
)


def test_custom_condition_is_bounded_and_plain() -> None:
    value = safe_condition("kidney disease; DROP TABLE trials!!! " + "x" * 100)
    assert ";" not in value
    assert len(value) <= 80
    assert "kidney disease" in value


def test_demo_source_and_hashes_are_reproducible() -> None:
    first = build_demo_studies(size=120)
    second = build_demo_studies(size=120)
    assert stable_hash(first) == stable_hash(second)
    assert len(first) == 120


def test_snapshot_is_idempotent_and_contract_reconciles() -> None:
    studies = build_demo_studies(size=160)
    first = snapshot_table(studies)
    second = snapshot_table(studies)
    assert first["payload_hash"].tolist() == second["payload_hash"].tolist()
    validated = normalize_snapshot(first)
    features = feature_view(validated)
    assert len(first) == len(validated) == len(features)
    assert validated["nct_id"].is_unique
    assert set(validated["discontinued"]) == {0, 1}


def test_contract_rejects_duplicate_and_invalid_records() -> None:
    studies = build_demo_studies(size=80)
    studies.append(studies[0])
    invalid = build_demo_studies(size=80)[1]
    invalid["protocolSection"]["identificationModule"]["nctId"] = "BAD"
    studies.append(invalid)
    validated = normalize_snapshot(snapshot_table(studies))
    assert len(validated) == 80


def test_feature_view_excludes_post_outcome_leakage() -> None:
    forbidden = {"overall_status", "completion_date", "has_results", "last_update_date"}
    assert not forbidden.intersection(FEATURE_COLUMNS)
    features = feature_view(normalize_snapshot(snapshot_table(build_demo_studies(size=100))))
    assert not forbidden.intersection(FEATURE_COLUMNS)
    assert features["enrollment_log"].gt(0).all()


def test_quality_report_has_explicit_denominators_and_passes() -> None:
    snapshot = snapshot_table(build_demo_studies(size=120))
    validated = normalize_snapshot(snapshot)
    features = feature_view(validated)
    report = quality_report(snapshot, validated, features)
    assert report["passed"].all()
    assert set(report.columns) == {"check", "passed", "detail"}
    assert len(report) >= 9


def test_source_failure_uses_deterministic_fallback(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fail)
    studies, metadata = fetch_studies("asthma", 120)
    assert metadata["mode"] == "demo"
    assert "ConnectionError" in metadata["fallback_reason"]
    assert len(studies) == 120


def test_transient_http_failure_is_retried_once(monkeypatch) -> None:
    calls = []

    class Response:
        headers = {}
        url = "https://clinicaltrials.gov/api/v2/studies"

        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return {"studies": []}

    def respond(*_args, **_kwargs):
        calls.append(1)
        return Response(503 if len(calls) == 1 else 200)

    monkeypatch.setattr(requests, "get", respond)
    monkeypatch.setattr("clinical_trial_ops_pipeline.src.data.time.sleep", lambda _seconds: None)
    response, payload = _request_payload({"pageSize": 5})
    assert response.status_code == 200
    assert payload == {"studies": []}
    assert len(calls) == 2
