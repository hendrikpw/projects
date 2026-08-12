from __future__ import annotations

import gzip
import io

import numpy as np
import pandas as pd
import pytest

from storm_impact_pipeline.src import data
from storm_impact_pipeline.src.model import score_case,train_and_evaluate
from storm_impact_pipeline.src.pipeline import FEATURES,_gold,_silver,parse_damage,run_pipeline


def test_damage_parser_handles_units_and_missingness():
    assert parse_damage("10.00K")==10_000 and parse_damage("1.50M")==1_500_000 and parse_damage("2B")==2_000_000_000
    assert np.isnan(parse_damage("")) and np.isnan(parse_damage("unknown"))


def test_fallback_is_deterministic_and_source_shaped():
    pd.testing.assert_frame_equal(data._fallback(),data._fallback()); assert set(data.USECOLS)==set(data._fallback().columns)


def test_contract_quarantines_duplicate_and_incomplete_label():
    frame=data._fallback(3000); duplicate=frame.iloc[0].copy(); incomplete=frame.iloc[1].copy(); incomplete["EVENT_ID"]=99_999_999; incomplete["DAMAGE_CROPS"]=""
    silver,q=_silver(pd.concat([frame,pd.DataFrame([duplicate,incomplete])],ignore_index=True)); assert {"duplicate_event_id","incomplete_damage_label"}.issubset(set(q.invalid_reason)); assert len(silver)+len(q)==len(frame)+2


def test_gold_leakage_contract():
    silver,_=_silver(data._fallback()); gold=_gold(silver); assert FEATURES==[c for c in gold if c in FEATURES]; assert not {"event_narrative","deaths_direct","injuries_direct","tor_f_scale"}.intersection(gold.columns)


def test_pipeline_is_content_idempotent(monkeypatch):
    frame=data._fallback(); meta={"mode":"demo","fallback_reason":"test","filename":"x","source_url":"x","source_hash":"a"*64,"retrieved_at":"ignored","year":2025}; monkeypatch.setattr("storm_impact_pipeline.src.pipeline.load_dataset",lambda:(frame,meta)); a,b=run_pipeline(),run_pipeline(); assert a.metadata["run_id"]==b.metadata["run_id"] and a.metadata["gold_hash"]==b.metadata["gold_hash"] and a.quality.passed.all()


def test_source_failure_uses_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data,"_discover",lambda:(_ for _ in ()).throw(RuntimeError("offline"))); frame,meta=data.load_dataset(); assert meta["mode"]=="demo" and len(frame)==24_000 and "offline" in meta["fallback_reason"]


def test_discovery_chooses_latest_revision(monkeypatch):
    html=b'x StormEvents_details-ftp_v1.0_d2025_c20260101.csv.gz y StormEvents_details-ftp_v1.0_d2025_c20260728.csv.gz'; monkeypatch.setattr(data,"_request",lambda *_:html); assert data._discover()=="StormEvents_details-ftp_v1.0_d2025_c20260728.csv.gz"


def test_decompression_size_guard(monkeypatch):
    payload=gzip.compress(b"x"*2000); monkeypatch.setattr(data,"MAX_UNCOMPRESSED_BYTES",1000)
    with pytest.raises(ValueError,match="exceeds"): data._parse(payload)


def test_model_reproducible_and_beats_probability_baseline():
    silver,_=_silver(data._fallback()); gold=_gold(silver); a=train_and_evaluate(gold); b=train_and_evaluate(gold); assert a.metrics==b.metrics; assert a.metrics["average_precision"]>=a.metrics["prevalence"]; assert a.evaluation.damage_probability.between(0,1).all(); assert len(a.importance)==len(FEATURES)


def test_serving_output_shape_and_bounds():
    silver,_=_silver(data._fallback()); model=train_and_evaluate(_gold(silver)); values={"state":"TEXAS","event_type":"Tornado","cz_type":"C","month":8,"begin_hour":16,"magnitude":1.0,"begin_lat":31.0,"begin_lon":-99.0}; result=score_case(model,values); assert set(result)=={"damage_probability","extreme_probability","conditional_damage_usd","expected_damage_usd"}; assert 0<=result["damage_probability"]<=1 and 0<=result["extreme_probability"]<=1 and result["expected_damage_usd"]>=0
