from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from message_trust_gateway.src import data
from message_trust_gateway.src.model import _split,adversarial_text,score_message,train_and_evaluate
from message_trust_gateway.src.pipeline import _gold,_silver,normalize_text,redact_text,run_pipeline


def test_fallback_deterministic_and_shaped():
    pd.testing.assert_frame_equal(data._fallback(),data._fallback()); assert set(data._fallback().columns)=={"source_row","label","message"}


def test_normalization_and_privacy_tokens():
    assert normalize_text("  hello\n world ")=="hello world"; safe=redact_text("Mail me at a@b.com or call +44 7700 900123 and open https://x.test for £50"); assert "<EMAIL>" in safe and "<PHONE>" in safe and "<URL>" in safe and "<MONEY>" in safe


def test_contract_quarantine():
    frame=data._fallback(); bad=pd.DataFrame([{"source_row":99999,"label":"maybe","message":"hello"},{"source_row":100000,"label":"ham","message":""}]); silver,q=_silver(pd.concat([frame,bad],ignore_index=True)); assert {"invalid_label","empty_message"}.issubset(set(q.invalid_reason)); assert len(silver)+len(q)==len(frame)+2


def test_duplicate_groups_never_cross_splits():
    silver,_=_silver(data._fallback()); gold=_gold(silver); train,cal,test=_split(gold); assert not set(train.group_hash)&set(cal.group_hash) and not set(train.group_hash)&set(test.group_hash) and not set(cal.group_hash)&set(test.group_hash)


def test_pipeline_idempotent(monkeypatch):
    frame=data._fallback(); meta={"mode":"demo","fallback_reason":"test","source_url":"x","dataset_url":"x","source_hash":"a"*64,"retrieved_at":"ignored"}; monkeypatch.setattr("message_trust_gateway.src.pipeline.load_dataset",lambda:(frame,meta)); a,b=run_pipeline(),run_pipeline(); assert a.metadata["run_id"]==b.metadata["run_id"] and a.metadata["gold_hash"]==b.metadata["gold_hash"] and a.quality.passed.all(); assert a.metadata["replayed_deliveries"]==25 and a.batches.duplicate_deliveries.sum()==25


def test_download_failure_fallback(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(RuntimeError("offline"))); frame,meta=data.load_dataset(); assert meta["mode"]=="demo" and len(frame)==5_000 and "offline" in meta["fallback_reason"]


def test_unsafe_archive_rejected():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as z: z.writestr("wrong.txt","ham\thello")
    with pytest.raises(ValueError,match="allowlist"): data._parse(stream.getvalue())


def test_model_reproducibility_and_baseline():
    silver,_=_silver(data._fallback()); gold=_gold(silver); a,b=train_and_evaluate(gold),train_and_evaluate(gold); assert a.metrics==b.metrics; assert a.metrics["average_precision"]>=a.metrics["baseline_average_precision"] and 0<=a.metrics["brier"]<=1; assert a.evaluation.spam_probability.between(0,1).all()


def test_adversarial_transform_is_deterministic():
    assert adversarial_text("URGENT free prize") == adversarial_text("URGENT free prize") and adversarial_text("URGENT free prize")!="URGENT free prize"


def test_serving_output_and_empty_input():
    silver,_=_silver(data._fallback()); model=train_and_evaluate(_gold(silver)); result=score_message(model,"FREE prize! Call +44 7700 900123"); assert result["decision"] in {"allow","review","block"} and 0<=result["spam_probability"]<=1 and "<PHONE>" in result["redacted_text"]; empty=score_message(model,""); assert 0<=empty["spam_probability"]<=1
