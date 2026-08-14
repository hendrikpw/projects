from __future__ import annotations
import pandas as pd
from stackoverflow_tag_router.src import data
from stackoverflow_tag_router.src.model import _split,suggest_tags,train_and_evaluate
from stackoverflow_tag_router.src.pipeline import _transform,clean_html,redact_text,run_pipeline


def test_fallback_is_deterministic(): pd.testing.assert_frame_equal(data._fallback(),data._fallback())
def test_html_and_privacy(): assert "hello" in clean_html("<p>hello</p>") and "<URL>" in redact_text("see https://x.test") and "<EMAIL>" in redact_text("a@b.com")
def test_contract_and_bridge():
    silver,q,gold,bridge=_transform(data._fallback()); assert len(q)==0 and len(gold)==len(silver) and len(bridge)==silver.tags_clean.map(len).sum() and silver.event_id.is_unique
def test_quarantine_invalid_rows():
    frame=data._fallback(600); bad=frame.iloc[[0]].copy(); bad["question_id"]=frame.question_id.iloc[1]; bad["tags"]=[[]]; silver,q,_,_=_transform(pd.concat([frame,bad],ignore_index=True)); assert len(q)>=1 and len(silver)+len(q)==601
def test_temporal_split_order():
    _,_,gold,_=_transform(data._fallback()); train,cal,test=_split(gold); assert train.created_at.max()<cal.created_at.min()<test.created_at.min()
def test_pipeline_idempotency_and_replay(monkeypatch):
    frame=data._fallback(); meta={"mode":"demo","fallback_reason":"test","source_hash":"a"*64}; monkeypatch.setattr("stackoverflow_tag_router.src.pipeline.load_questions",lambda:(frame,meta)); a,b=run_pipeline(),run_pipeline(); assert a.metadata["run_id"]==b.metadata["run_id"] and a.metadata["gold_hash"]==b.metadata["gold_hash"] and a.metadata["replayed_deliveries"]==20 and a.quality.passed.all()
def test_source_failure_fallback(monkeypatch):
    monkeypatch.setattr(data,"_request_page",lambda *_args,**_kwargs:(_ for _ in ()).throw(RuntimeError("offline"))); frame,meta=data.load_questions(); assert meta["mode"]=="demo" and len(frame)==1800 and "offline" in meta["fallback_reason"]
def test_model_reproducible_and_promoted():
    _,_,gold,_=_transform(data._fallback()); a,b=train_and_evaluate(gold),train_and_evaluate(gold); assert a.metrics==b.metrics and a.metrics["precision_at_3"]>a.metrics["baseline_precision_at_3"] and 0<=a.metrics["brier"]<=1
def test_model_output_shape():
    _,_,gold,_=_transform(data._fallback()); model=train_and_evaluate(gold); result=suggest_tags(model,"Pandas merge issue","My Python dataframe has duplicate rows"); assert result["route"] in {"auto-suggest","review"} and len(result["suggestions"])==5 and result["suggestions"].confidence.between(0,1).all()
def test_empty_serving_input():
    _,_,gold,_=_transform(data._fallback()); result=suggest_tags(train_and_evaluate(gold),"",""); assert len(result["suggestions"])==5
