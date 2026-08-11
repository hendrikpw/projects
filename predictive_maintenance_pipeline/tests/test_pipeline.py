from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from predictive_maintenance_pipeline.src import data
from predictive_maintenance_pipeline.src.model import decision_table, score_case, train_and_evaluate
from predictive_maintenance_pipeline.src.pipeline import FAILURE_MODES, FEATURES, _gold, _silver, run_pipeline


def test_fallback_is_deterministic_and_source_shaped():
    pd.testing.assert_frame_equal(data._fallback(),data._fallback())
    assert len(data._fallback())==10_000 and data._fallback()["Machine failure"].sum()>20


def test_contract_quarantines_invalid_and_duplicate_rows():
    frame=data._fallback(2500); bad=frame.iloc[0].copy(); bad["UDI"]=3001; bad["Product ID"]="M03001"; bad["Torque [Nm]"]=-1; duplicate=frame.iloc[1].copy()
    silver,quarantine=_silver(pd.concat([frame,pd.DataFrame([bad,duplicate])],ignore_index=True))
    assert {"torque_out_of_range","duplicate_udi"}.issubset(set(quarantine.invalid_reason))
    assert len(silver)+len(quarantine)==len(frame)+2


def test_gold_contract_prevents_target_leakage():
    silver,_=_silver(data._fallback()); gold=_gold(silver)
    assert not set(FAILURE_MODES).intersection(gold.columns)
    assert FEATURES==[c for c in gold if c not in ["udi","machine_failure"]]


def test_pipeline_is_content_idempotent(monkeypatch):
    frame=data._fallback(); meta={"mode":"demo","fallback_reason":"test","source_url":"x","dataset_page":"x","source_hash":"a"*64,"retrieved_at":"ignored"}
    monkeypatch.setattr("predictive_maintenance_pipeline.src.pipeline.load_dataset",lambda:(frame,meta))
    a,b=run_pipeline(),run_pipeline()
    assert a.metadata["run_id"]==b.metadata["run_id"] and a.metadata["gold_hash"]==b.metadata["gold_hash"]
    assert a.quality.passed.all()


def test_download_failure_uses_fallback(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(RuntimeError("offline")))
    frame,meta=data.load_dataset(); assert meta["mode"]=="demo" and len(frame)==10_000 and "offline" in meta["fallback_reason"]


def test_archive_allowlist_rejects_wrong_member():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as archive: archive.writestr("wrong.csv","a,b\n1,2")
    with pytest.raises(ValueError,match="allowlist"): data._parse_archive(stream.getvalue())


def test_retry_recovers(monkeypatch):
    calls={"n":0}
    class Response:
        content=b"x"*2000
        def raise_for_status(self): return None
    def get(*_a,**_k):
        calls["n"]+=1
        if calls["n"]==1: raise data.requests.Timeout("temporary")
        return Response()
    monkeypatch.setattr(data.requests,"get",get); monkeypatch.setattr(data.time,"sleep",lambda *_:None)
    assert len(data._download())==2000 and calls["n"]==2


def test_model_is_reproducible_evaluated_and_calibrated():
    silver,_=_silver(data._fallback()); gold=_gold(silver)
    first=train_and_evaluate(gold); second=train_and_evaluate(gold)
    assert first.metrics==second.metrics
    assert first.evaluation.failure_probability.between(0,1).all()
    assert first.metrics["average_precision"]>=first.metrics["baseline_ap"]
    assert 0<=first.metrics["brier"]<=1 and len(first.importance)==len(FEATURES)


def test_decision_cost_changes_with_false_negative_penalty():
    y=pd.Series([0,0,0,1,1]).to_numpy(); p=pd.Series([.1,.3,.6,.4,.8]).to_numpy()
    low=decision_table(y,p,1,1); high=decision_table(y,p,50,1)
    assert high.loc[high.cost.idxmin(),"threshold"]<=low.loc[low.cost.idxmin(),"threshold"]


def test_score_case_shape_and_range():
    silver,_=_silver(data._fallback()); bundle=train_and_evaluate(_gold(silver))
    values={"type":"M","air_temperature_k":300.0,"process_temperature_k":310.0,"rotational_speed_rpm":1500,"torque_nm":40.0,"tool_wear_min":100,"temperature_gap_k":10.0,"power_proxy":60000.0}
    assert 0<=score_case(bundle,values)<=1
