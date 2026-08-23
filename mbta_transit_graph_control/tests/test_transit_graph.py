from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from mbta_transit_graph_control.src.data import _safe_archive, fallback_extract
from mbta_transit_graph_control.src.model import score_node, train_and_evaluate
from mbta_transit_graph_control.src.pipeline import build_product, run_pipeline


@pytest.fixture(scope="module")
def product(): return run_pipeline(force_fallback=True)


@pytest.fixture(scope="module")
def model(product): return train_and_evaluate(product)


def test_fallback_is_deterministic():
    first=fallback_extract(); second=fallback_extract()
    pd.testing.assert_frame_equal(first.stops,second.stops)
    assert first.metadata["source_hash"]==second.metadata["source_hash"]


def test_fallback_relations_have_expected_keys():
    source=fallback_extract()
    assert source.stops.stop_id.is_unique and source.routes.route_id.is_unique and source.patterns.route_pattern_id.is_unique
    assert set(source.stop_times.trip_id)<=set(source.patterns.representative_trip_id)


def test_archive_rejects_missing_required_members():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w") as archive: archive.writestr("stops.txt","stop_id\n1\n")
    with pytest.raises(ValueError,match="missing GTFS members"): _safe_archive(buffer.getvalue())


def test_archive_rejects_path_traversal():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w") as archive: archive.writestr("../stops.txt","bad")
    with pytest.raises(ValueError,match="unsafe archive path"): _safe_archive(buffer.getvalue())


def test_pipeline_passes_all_publication_gates(product):
    assert product.quality.passed.all()
    assert product.metadata["replays"]==20 and product.metadata["quarantined"]==0


def test_graph_has_referential_integrity(product):
    assert product.edges.node_id.isin(product.nodes.node_id).all()
    assert product.edges.route_id.isin(product.routes.route_id).all()
    assert product.edges.edge_id.is_unique


def test_pipeline_is_content_idempotent():
    first=run_pipeline(force_fallback=True); second=run_pipeline(force_fallback=True)
    assert first.metadata["run_id"]==second.metadata["run_id"]
    assert first.metadata["edges_hash"]==second.metadata["edges_hash"]


def test_invalid_coordinate_is_quarantined():
    source=fallback_extract(); source.stops.loc[0,"stop_lat"]="99"
    product=build_product(source)
    assert "coordinate_out_of_region" in set(product.quarantine.reason)
    assert "S000" not in set(product.nodes.node_id)


def test_model_uses_disjoint_calibration_and_test_edges(model):
    cal=set(model.evaluation.loc[model.evaluation.split=="calibration","node_id"]); test=set(model.evaluation.loc[model.evaluation.split=="test","node_id"])
    assert cal.isdisjoint(test) and len(cal)>50 and len(test)>50


def test_model_beats_popularity_baseline(model):
    assert model.metrics["hit_at_5"]>model.metrics["baseline_hit_at_5"]+.05
    assert model.metrics["hit_at_5"]>.25


def test_model_outputs_have_valid_shapes(model):
    assert model.svd.components_.shape[1]==len(model.routes)
    assert len(model.embeddings)==len(model.routes)
    assert model.evaluation["rank"].ge(1).all()


def test_model_is_reproducible(product,model):
    again=train_and_evaluate(product)
    assert model.metadata==again.metadata
    np.testing.assert_array_equal(model.evaluation["rank"],again.evaluation["rank"])


def test_edge_loss_exposes_measurable_failure(model):
    assert list(model.robustness.edge_dropout)==[0,.1,.2,.3,.4]
    assert model.robustness.iloc[-1].hit_at_1<=model.robustness.iloc[0].hit_at_1


def test_link_recovery_returns_ranked_candidates(product,model):
    node=product.edges.groupby("node_id").filter(lambda group:len(group)>=2).iloc[0].node_id; hidden=product.edges[product.edges.node_id==node].iloc[0].route_id; result=score_node(model,node,hide_route=hidden)
    assert len(result["ranking"])==10 and result["recovered_rank"]>=1
    assert result["ranking"]["score"].is_monotonic_decreasing


def test_serving_rejects_unknown_node(model):
    with pytest.raises(ValueError,match="unknown node"): score_node(model,"not-a-node")
