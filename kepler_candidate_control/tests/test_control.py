import hashlib
import pandas as pd
import pytest
from kepler_candidate_control.src import data
from kepler_candidate_control.src.model import _bucket,score_candidate,train_and_evaluate
from kepler_candidate_control.src.pipeline import FEATURES,_bronze,_silver,run_pipeline

@pytest.fixture(scope="module")
def products():
    patch=pytest.MonkeyPatch(); patch.setattr(data,"_download",lambda:(_ for _ in ()).throw(ConnectionError("offline"))); p=run_pipeline(); m=train_and_evaluate(p.gold); patch.undo(); return p,m
def test_fallback_is_deterministic(): assert data._fallback(100).equals(data._fallback(100))
def test_source_failure_is_atomic_demo(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(TimeoutError("timeout"))); frame,meta=data.load_source(); assert meta["mode"]=="demo" and len(frame)==7800
def test_schema_change_fails_closed():
    with pytest.raises(ValueError): data._parse(b"wrong,column\n1,2\n")
def test_replays_are_suppressed(products):
    p,_=products; assert p.metadata["duplicates"]==20 and not p.silver.event_id.duplicated().any()
def test_invalid_physical_row_is_quarantined():
    raw=data._fallback(100); raw.loc[0,"koi_depth"]=-1; s,q=_silver(_bronze(raw)); assert "invalid_depth" in set(q.reason)
def test_pipeline_is_content_idempotent(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(ConnectionError("offline"))); a=run_pipeline(); b=run_pipeline(); assert a.metadata["gold_hash"]==b.metadata["gold_hash"] and a.metadata["run_id"]==b.metadata["run_id"]
def test_publication_gates_pass(products): assert products[0].quality.passed.all()
def test_disposition_and_scores_are_blocked(): assert not {"koi_disposition","koi_pdisposition","koi_score"}&set(FEATURES)
def test_star_groups_never_cross_splits(products):
    p,_=products; mapping=p.gold.assign(bucket=p.gold.kepid.map(_bucket)).groupby("kepid").bucket.nunique(); assert mapping.max()==1
def test_model_metrics_and_shapes(products):
    _,m=products; assert m.metrics["average_precision"]>m.metrics["baseline_average_precision"]+.05 and m.evaluation.probability.between(0,1).all() and set(m.importance.feature)==set(FEATURES)
def test_ood_and_missing_serving(products):
    p,m=products; row=p.gold.iloc[0].copy(); result=score_candidate(m,row,{"koi_period":1e9,"koi_prad":1e9}); assert result["route"]=="ood-review" and len(result["outside_features"])>=2
