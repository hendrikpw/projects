import pandas as pd
import pytest

from federal_procurement_resolution.src.data import fallback_data
from federal_procurement_resolution.src.pipeline import build_product, contract_silver, frame_hash, make_bronze
from federal_procurement_resolution.src.resolution import corrupt_name, normalize_name, resolve_name, train_and_evaluate


@pytest.fixture(scope="module")
def product():
    raw, metadata = fallback_data()
    return build_product(raw, metadata)


@pytest.fixture(scope="module")
def model(product):
    return train_and_evaluate(product.recipients)


def test_fallback_is_reproducible():
    first, a = fallback_data(); second, b = fallback_data()
    pd.testing.assert_frame_equal(first, second); assert a["source_hash"] == b["source_hash"]


def test_replays_are_idempotent(product):
    assert product.metadata["duplicates"] == 15
    assert product.silver.award_id.is_unique
    assert len(product.bronze) == len(product.silver) + len(product.quarantine) + 15


def test_content_hash_is_order_independent(product):
    assert frame_hash(product.silver) == frame_hash(product.silver.sample(frac=1, random_state=9))


def test_publication_gates_pass(product):
    assert product.quality.passed.all() and len(product.quality) == 8


def test_gold_foreign_keys_reconcile(product):
    assert set(product.awards.recipient_uei) == set(product.recipients.recipient_uei)
    assert product.recipients.recipient_uei.is_unique


def test_invalid_contract_rows_are_quarantined():
    raw, _ = fallback_data(); raw["award_amount"] = raw["award_amount"].astype(object); raw.loc[0, "recipient_uei"] = "BAD"; raw.loc[1, "award_amount"] = "not-a-number"
    bronze = make_bronze(raw, replay_rows=0); silver, quarantine, duplicates = contract_silver(bronze)
    assert duplicates == 0 and len(quarantine) == 2
    assert set(quarantine.quarantine_reason) == {"invalid_uei", "invalid_amount"}
    assert len(silver) == len(raw) - 2


def test_normalization_and_corruptions_are_nonempty():
    assert normalize_name(" ACME, Inc. ") == "ACME INC"
    assert all(corrupt_name("NORTHSTAR ANALYTICS LLC", i) for i in range(5))


def test_model_beats_exact_baseline(model):
    assert model.metrics["top1_accuracy"] >= model.metrics["exact_baseline_accuracy"]
    assert model.metrics["hit_at_5"] >= .90


def test_model_reports_selective_failure_metrics(model):
    required = {"coverage", "selective_accuracy", "false_merge_rate", "unknown_rejection_rate"}
    assert required <= model.metrics.keys()
    assert all(0 <= model.metrics[key] <= 1 for key in required)


def test_matching_is_reproducible(product):
    one = train_and_evaluate(product.recipients); two = train_and_evaluate(product.recipients)
    assert one.metrics == two.metrics and one.thresholds == two.thresholds


def test_resolution_output_shape(model):
    query = corrupt_name(model.reference.iloc[0].canonical_name, 3)
    result = resolve_name(model, query)
    assert len(result) == 5
    assert {"recipient_uei", "canonical_name", "score", "margin", "route"} <= set(result.columns)


def test_empty_query_fails_closed(model):
    assert resolve_name(model, "   ").empty


def test_small_reference_is_rejected(product):
    with pytest.raises(RuntimeError, match="at least 25"):
        train_and_evaluate(product.recipients.head(10))
