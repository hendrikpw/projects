from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from digit_recognition_gateway.src.data import PIXELS, _parse, _safe_archive, fallback_data
from digit_recognition_gateway.src.model import corrupt, score_image, train_and_evaluate
from digit_recognition_gateway.src.pipeline import contract_silver, make_bronze, run_pipeline


@pytest.fixture(scope="module")
def product():
    return run_pipeline(force_fallback=True)


@pytest.fixture(scope="module")
def model(product):
    return train_and_evaluate(product.gold)


def test_fallback_is_deterministic():
    first, first_meta = fallback_data(); second, second_meta = fallback_data()
    pd.testing.assert_frame_equal(first, second)
    assert first_meta["source_hash"] == second_meta["source_hash"]


def test_fallback_contract_shape():
    frame, _ = fallback_data()
    assert frame.shape == (5600, 67)
    assert set(frame.source_split) == {"train", "test"}
    assert frame[PIXELS].min().min() >= 0 and frame[PIXELS].max().max() <= 16


def test_zip_allowlist_rejects_extra_member():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("optdigits.tra", "0")
        archive.writestr("optdigits.tes", "0")
        archive.writestr("unexpected.txt", "no")
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        with pytest.raises(ValueError, match="unexpected archive members"):
            _safe_archive(buffer.getvalue())


def test_zip_path_traversal_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../optdigits.tra", "0")
        archive.writestr("optdigits.tes", "0")
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        with pytest.raises(ValueError):
            _safe_archive(buffer.getvalue())


def test_parser_requires_64_pixels_and_label():
    with pytest.raises(ValueError, match="64 pixels"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive: archive.writestr("bad", "1,2,3\n")
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive: _parse(archive, "bad", "train")


def test_replay_deliveries_are_idempotent():
    source, _ = fallback_data(); source = source.head(100)
    bronze = make_bronze(source, replay_rows=20, batch_size=32)
    silver, _, replays = contract_silver(bronze)
    assert len(bronze) == 120 and len(silver) == 100
    assert replays == 20 and silver.sample_id.is_unique


def test_pipeline_reconciles_and_passes_gates(product):
    assert product.quality.passed.all()
    assert product.metadata["deliveries"] == product.metadata["accepted"] + product.metadata["quarantined"] + product.metadata["replays"]
    assert product.metadata["replays"] == 20


def test_pipeline_is_content_idempotent():
    first = run_pipeline(force_fallback=True); second = run_pipeline(force_fallback=True)
    assert first.metadata["run_id"] == second.metadata["run_id"]
    assert first.metadata["gold_hash"] == second.metadata["gold_hash"]


def test_invalid_pixels_are_quarantined():
    source, _ = fallback_data(); source = source.head(80).copy(); source.loc[source.index[0], "px_0_0"] = 99
    bronze = make_bronze(source, replay_rows=0); silver, quarantine, _ = contract_silver(bronze)
    assert len(silver) == 79
    assert "pixel_out_of_range" in quarantine.quarantine_reason.iloc[0]


def test_model_uses_official_test_partition(model):
    assert model.metadata["test_images"] == 1800
    assert model.metadata["train_people"] == 30 and model.metadata["test_people"] == 13


def test_model_beats_baseline_and_has_valid_outputs(model):
    assert model.metrics["macro_f1"] > .90
    assert model.metrics["macro_f1"] > model.metrics["baseline_macro_f1"] + .70
    assert 0 <= model.metrics["ece"] <= 1
    assert len(model.evaluation) == model.metadata["test_images"]


def test_model_is_reproducible(product, model):
    again = train_and_evaluate(product.gold)
    np.testing.assert_allclose(model.evaluation.confidence, again.evaluation.confidence)
    assert model.metrics["accuracy"] == again.metrics["accuracy"]


def test_corruption_is_reproducible_and_bounded():
    images = np.ones((4, 64)) * .5
    first = corrupt(images, .2); second = corrupt(images, .2)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0 and first.max() <= 1


def test_robustness_evaluation_exposes_failure_state(model):
    assert list(model.robustness.severity) == [0, .05, .10, .15, .20, .25]
    assert model.robustness.iloc[-1].accuracy < model.robustness.iloc[0].accuracy


def test_serving_output_shape_and_withholding(product, model):
    row = product.gold[product.gold.source_split == "test"].iloc[0]
    normal = score_image(model, row)
    assert normal["image"].shape == (8, 8) and normal["sensitivity"].shape == (8, 8)
    assert np.isclose(normal["ranking"].probability.sum(), 1)
    withheld = score_image(model, row, dropout=.30)
    assert withheld["route"] == "input-withheld"
