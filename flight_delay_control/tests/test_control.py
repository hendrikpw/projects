import io,zipfile
import numpy as np
import pandas as pd
import pytest
from flight_delay_control.src import data
from flight_delay_control.src.model import score_scenario,train_and_evaluate
from flight_delay_control.src.pipeline import FEATURES,_bronze,_silver,run_pipeline

@pytest.fixture(scope="module")
def products():
    patch=pytest.MonkeyPatch(); patch.setattr(data,"_download",lambda:(_ for _ in ()).throw(ConnectionError("offline")))
    p=run_pipeline(); model=train_and_evaluate(p.gold); patch.undo(); return p,model
def test_fallback_is_deterministic(): assert data._fallback(100).equals(data._fallback(100))
def test_failure_uses_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(TimeoutError("timeout"))); frame,meta=data.load_source(); assert meta["mode"]=="demo" and len(frame)==72000
def test_zip_path_traversal_fails_closed():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as z:z.writestr("../unsafe.csv","x\n1")
    with pytest.raises(ValueError): data._read_zip(buf.getvalue())
def test_replay_is_suppressed(products):
    p,_=products; assert p.metadata["duplicates"]==25 and not p.silver.event_id.duplicated().any()
def test_invalid_contract_row_is_quarantined():
    raw=data._fallback(100); raw.loc[0,"Origin"]="BAD1"; s,q=_silver(_bronze(raw)); assert len(q)>=1 and "invalid_origin" in set(q.reason)
def test_hashes_are_idempotent(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(ConnectionError("offline"))); a=run_pipeline(); b=run_pipeline(); assert a.metadata["gold_hash"]==b.metadata["gold_hash"] and a.metadata["run_id"]==b.metadata["run_id"]
def test_all_publication_gates_pass(products): assert products[0].quality.passed.all()
def test_outcomes_are_excluded(): assert not {"ArrDelayMinutes","ArrDel15","Cancelled","Diverted"}&set(FEATURES)
def test_temporal_evaluation_and_metrics(products):
    _,m=products; assert m.metrics["average_precision"]>m.metrics["baseline_average_precision"] and 0<=m.metrics["brier"]<=1 and m.metadata["test_days"].startswith("25")
def test_model_output_and_scenario(products):
    p,m=products; assert m.evaluation.probability.between(0,1).all(); result=score_scenario(m,p.gold.iloc[-1],{"Origin":"ZZZ"}); assert 0<=result["probability"]<=1 and result["status"] in {"monitor","watch","review"}
