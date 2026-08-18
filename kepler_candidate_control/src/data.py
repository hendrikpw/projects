"""Bounded NASA Exoplanet Archive TAP ingestion and deterministic fallback."""
from __future__ import annotations
import hashlib,io
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import requests

TAP="https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DOCS="https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html"
FIELDS="https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html"
KOI_DOCS="https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html"
COLUMNS=["kepid","kepoi_name","koi_disposition","koi_period","koi_duration","koi_depth","koi_prad","koi_teq","koi_insol","koi_steff","koi_slogg","koi_srad","koi_kepmag","koi_count","koi_num_transits","koi_model_snr","ra","dec"]
QUERY="select "+",".join(COLUMNS)+" from cumulative"

def _download():
    response=requests.get(TAP,params={"query":QUERY,"format":"csv"},headers={"User-Agent":"hendrikpw-projects kepler-control/1.0"},timeout=(5,45))
    response.raise_for_status()
    if not 500_000<len(response.content)<5_000_000: raise ValueError("TAP response outside safety bounds")
    return response.content

def _parse(raw):
    frame=pd.read_csv(io.BytesIO(raw),low_memory=False)
    if list(frame.columns)!=COLUMNS: raise ValueError("NASA TAP schema changed")
    if not 7_000<=len(frame)<=12_000: raise ValueError("KOI row count outside contract")
    return frame

def _fallback(n=7800):
    rng=np.random.default_rng(20260818); y=rng.random(n)<.52; star=rng.integers(1_000_000,13_000_000,n); period=np.exp(rng.normal(2.3,1.35,n)); duration=np.exp(rng.normal(1.05,.55,n)); depth=np.exp(rng.normal(6.4+1.25*(~y),1.05,n)); prad=np.exp(rng.normal(1.25+1.1*(~y),.75,n)); snr=np.exp(rng.normal(3.5+.55*(~y),.85,n)); steff=rng.normal(5600,700,n); srad=np.exp(rng.normal(-.03,.38,n)); insol=np.clip((steff/5778)**4*srad**2/(period/365.25)**(4/3),.01,1e6)
    return pd.DataFrame({"kepid":star,"kepoi_name":[f"K{i%99999:05d}.{i%3+1:02d}" for i in range(n)],"koi_disposition":np.where(y,rng.choice(["CONFIRMED","CANDIDATE"],n,p=[.55,.45]),"FALSE POSITIVE"),"koi_period":period,"koi_duration":duration,"koi_depth":depth,"koi_prad":prad,"koi_teq":np.clip(278*insol**.25,100,3000),"koi_insol":insol,"koi_steff":steff,"koi_slogg":rng.normal(4.35,.35,n),"koi_srad":srad,"koi_kepmag":rng.normal(14.3,1.4,n),"koi_count":rng.integers(1,6,n),"koi_num_transits":np.maximum(3,(1470/period).astype(int)),"koi_model_snr":snr,"ra":rng.uniform(279,302,n),"dec":rng.uniform(36,53,n)})

def load_source():
    try: raw=_download(); frame=_parse(raw); mode="live"; reason=""
    except Exception as exc: frame=_fallback(); raw=frame.to_csv(index=False).encode(); mode="demo"; reason=str(exc)
    return frame,{"mode":mode,"fallback_reason":reason,"source_hash":hashlib.sha256(raw).hexdigest(),"source_bytes":len(raw),"source_url":TAP,"query":QUERY,"retrieved_at":datetime.now(timezone.utc).isoformat()}
