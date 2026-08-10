from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from movielens_recommendation_pipeline.src import data
from movielens_recommendation_pipeline.src.pipeline import _gold, _silver, run_pipeline


def test_fallback_is_deterministic_and_source_shaped():
    first, second=data._fallback(),data._fallback()
    for name in first:
        pd.testing.assert_frame_equal(first[name],second[name])
    assert set(first)=={"movies","ratings","tags","links"}


def test_contract_and_references():
    silver,quarantine=_silver(data._fallback()); gold=_gold(silver["movies"],silver["ratings"],silver["tags"])
    assert quarantine.empty
    assert silver["movies"]["movieId"].is_unique
    assert silver["ratings"]["movieId"].isin(silver["movies"]["movieId"]).all()
    assert pd.api.types.is_datetime64_any_dtype(silver["ratings"]["rated_at"])
    assert len(gold)==len(silver["movies"])


def test_invalid_and_duplicate_ratings_are_quarantined():
    tables=data._fallback(); bad=tables["ratings"].iloc[0].copy(); bad["rating"]=9; duplicate=tables["ratings"].iloc[1].copy()
    tables["ratings"]=pd.concat([tables["ratings"],pd.DataFrame([bad,duplicate])],ignore_index=True)
    silver,quarantine=_silver(tables)
    assert {"rating_out_of_range","duplicate_rating_event"}.issubset(set(quarantine["invalid_reason"]))
    assert len(silver["ratings"])+len(quarantine)==len(tables["ratings"])


def test_pipeline_idempotency(monkeypatch):
    tables=data._fallback(); meta={"mode":"demo","fallback_reason":"test","archive_hash":"a"*64,"retrieved_at":"ignored","source_url":"test","dataset_version":"test"}
    monkeypatch.setattr("movielens_recommendation_pipeline.src.pipeline.load_dataset",lambda:(tables,meta))
    first,second=run_pipeline(),run_pipeline()
    assert first.metadata["run_id"]==second.metadata["run_id"]
    assert first.metadata["gold_hash"]==second.metadata["gold_hash"]
    assert first.quality["passed"].all()


def test_download_failure_uses_atomic_fallback(monkeypatch):
    monkeypatch.setattr(data,"_download",lambda:(_ for _ in ()).throw(RuntimeError("offline")))
    tables,meta=data.load_dataset()
    assert meta["mode"]=="demo" and len(tables["ratings"])>5000
    assert "offline" in meta["fallback_reason"]


def test_unsafe_or_incomplete_archive_fails():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as archive: archive.writestr("wrong.csv","a,b\n1,2")
    with pytest.raises(ValueError,match="missing required"):
        data._parse_archive(stream.getvalue())


def test_retry_recovers(monkeypatch):
    calls={"n":0}
    class Response:
        status_code=200; content=b"x"*100001
        def raise_for_status(self): return None
    def get(*_a,**_k):
        calls["n"]+=1
        if calls["n"]==1: raise data.requests.Timeout("temporary")
        return Response()
    monkeypatch.setattr(data.requests,"get",get); monkeypatch.setattr(data.time,"sleep",lambda *_:None)
    assert len(data._download())==100001 and calls["n"]==2
