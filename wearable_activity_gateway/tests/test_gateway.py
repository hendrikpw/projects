import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from wearable_activity_gateway.src.data import ACTIVITIES, _safe_archive, fallback_data
from wearable_activity_gateway.src.model import _partition, score_window, train_and_evaluate
from wearable_activity_gateway.src.pipeline import build_product, contract_silver, frame_hash, make_bronze


@pytest.fixture(scope="module")
def product():
    raw, metadata = fallback_data(); return build_product(raw, metadata)


@pytest.fixture(scope="module")
def model(product):
    return train_and_evaluate(product.gold, product.features, trees=80)


def test_fallback_is_deterministic():
    first, a = fallback_data(); second, b = fallback_data(); pd.testing.assert_frame_equal(first, second); assert a["source_hash"] == b["source_hash"]


def test_archive_contract_rejects_missing_files():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive: archive.writestr("README.txt", "incomplete")
    with pytest.raises(ValueError, match="archive contract missing"):
        _safe_archive(buffer.getvalue())


def test_archive_contract_rejects_path_traversal():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive: archive.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe archive path"):
        _safe_archive(buffer.getvalue())


def test_replay_delivery_is_idempotent(product):
    assert product.metadata["duplicates"] == 20 and product.gold.window_id.is_unique
    assert len(product.bronze) == len(product.silver) + len(product.quarantine) + 20


def test_hash_is_row_order_invariant(product):
    assert frame_hash(product.gold) == frame_hash(product.gold.sample(frac=1, random_state=4))


def test_all_publication_gates_pass(product):
    assert len(product.quality) == 10 and product.quality.passed.all()


def test_micro_batches_reconcile(product):
    assert product.batches.deliveries.sum() == len(product.bronze)
    assert product.batches.replays.sum() == 20


def test_invalid_sensor_rows_are_quarantined():
    raw, _ = fallback_data(); feature = next(c for c in raw if c.startswith("f")); raw.loc[0, feature] = np.nan; raw.loc[1, feature] = 5
    bronze = make_bronze(raw, replay_rows=0); silver, quarantine, duplicates = contract_silver(bronze, [c for c in raw if c.startswith("f")])
    assert duplicates == 0 and len(quarantine) == 2
    assert set(quarantine.quarantine_reason) == {"missing_sensor_value", "sensor_range_violation"}
    assert len(silver) == len(raw) - 2


def test_subjects_do_not_cross_model_splits(product):
    split = _partition(product.gold); groups = product.gold.assign(split=split).groupby("split").subject_id.apply(set)
    assert groups["train"].isdisjoint(groups["calibration"] | groups["test"])
    assert groups["calibration"].isdisjoint(groups["test"])


def test_model_beats_majority_baseline(model):
    assert model.metrics["macro_f1"] > model.metrics["baseline_macro_f1"] + .2
    assert model.metrics["top2_accuracy"] >= .9


def test_model_output_and_classes(model):
    assert set(model.classes) == set(ACTIVITIES.values())
    assert {"confidence", "route", "correct"} <= set(model.evaluation)
    assert len(model.confusion) == 36


def test_model_is_reproducible(product):
    a = train_and_evaluate(product.gold, product.features, trees=40); b = train_and_evaluate(product.gold, product.features, trees=40)
    assert a.metrics["macro_f1"] == b.metrics["macro_f1"] and a.confidence_threshold == b.confidence_threshold


def test_normal_window_can_be_scored(model, product):
    result = score_window(model, product.gold[product.gold.source_split == "test"].iloc[0])
    assert result["prediction"] in ACTIVITIES.values() and len(result["ranking"]) == 6


def test_missing_sensor_guardrail_fails_closed(model, product):
    result = score_window(model, product.gold.iloc[0], missing_share=.10)
    assert result["route"] == "sensor-fault-review"


def test_small_subject_universe_is_rejected(product):
    with pytest.raises(RuntimeError, match="subject-isolated"):
        train_and_evaluate(product.gold[product.gold.subject_id <= 4], product.features, trees=10)
