"""Revision-aware NOAA bulk-file discovery, bounded ingestion and fallback."""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

DIRECTORY_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
DATABASE_URL = "https://www.ncei.noaa.gov/stormevents/"
FORMAT_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf"
YEAR = 2025
MAX_COMPRESSED_BYTES = 20_000_000
MAX_UNCOMPRESSED_BYTES = 90_000_000
USECOLS = ["EVENT_ID","STATE","YEAR","MONTH_NAME","EVENT_TYPE","CZ_TYPE","BEGIN_DATE_TIME","DAMAGE_PROPERTY","DAMAGE_CROPS","MAGNITUDE","BEGIN_LAT","BEGIN_LON"]


def _request(url: str, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response=requests.get(url,timeout=(6,45),headers={"User-Agent":"hendrikpw-storm-impact-pipeline/1.0 (portfolio; public NOAA data)"})
            response.raise_for_status(); return response.content
        except requests.RequestException as exc:
            last=exc
            if attempt<retries-1: time.sleep(.3*2**attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def _discover(year: int = YEAR) -> str:
    html=_request(DIRECTORY_URL).decode("utf-8",errors="replace")
    pattern=rf"StormEvents_details-ftp_v1\.0_d{year}_c(\d{{8}})\.csv\.gz"
    matches=re.findall(pattern,html)
    if not matches: raise ValueError(f"no NOAA details revision found for {year}")
    revision=max(matches)
    return f"StormEvents_details-ftp_v1.0_d{year}_c{revision}.csv.gz"


def _download(filename: str) -> bytes:
    if not re.fullmatch(r"StormEvents_details-ftp_v1\.0_d\d{4}_c\d{8}\.csv\.gz",filename): raise ValueError("unsafe filename")
    payload=_request(DIRECTORY_URL+filename)
    if not 100_000 < len(payload) <= MAX_COMPRESSED_BYTES or payload[:2]!=b"\x1f\x8b": raise ValueError("compressed payload failed size or gzip check")
    return payload


def _parse(payload: bytes) -> pd.DataFrame:
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        raw=stream.read(MAX_UNCOMPRESSED_BYTES+1)
    if len(raw)>MAX_UNCOMPRESSED_BYTES: raise ValueError("decompressed payload exceeds safety bound")
    frame=pd.read_csv(io.BytesIO(raw),usecols=USECOLS,low_memory=False)
    if set(frame.columns)!=set(USECOLS): raise ValueError("NOAA schema mismatch")
    return frame


def _fallback(n: int = 24_000) -> pd.DataFrame:
    rng=np.random.default_rng(20250812); months=np.tile(np.arange(1,13),int(np.ceil(n/12)))[:n]; rng.shuffle(months)
    month_names=pd.to_datetime(pd.Series(months),format="%m").dt.month_name()
    kinds=np.array(["Thunderstorm Wind","Hail","Flash Flood","Flood","Tornado","Winter Storm","High Wind","Heavy Rain"])
    event=rng.choice(kinds,n,p=[.31,.26,.12,.09,.06,.05,.06,.05]); state=rng.choice(["TEXAS","KANSAS","FLORIDA","CALIFORNIA","OKLAHOMA","NEW YORK","COLORADO","ILLINOIS"],n)
    magnitude=np.where(event=="Hail",rng.uniform(.75,3,n),np.where(np.isin(event,["Thunderstorm Wind","High Wind"]),rng.uniform(35,90,n),np.nan))
    base=np.select([event=="Tornado",event=="Flood",event=="Flash Flood",event=="Thunderstorm Wind"],[.36,.28,.25,.18],default=.09)
    positive=rng.random(n)<base; scale=np.select([event=="Tornado",event=="Flood",event=="Flash Flood"],[11.4,10.8,10.4],default=9.4)
    total=np.where(positive,np.maximum(100,np.expm1(rng.normal(scale,1.5,n))),0); crop_share=rng.beta(1.2,5,n)
    begin=pd.to_datetime({"year":np.full(n,YEAR),"month":months,"day":rng.integers(1,28,n)})+pd.to_timedelta(rng.integers(0,24,n),unit="h")
    def token(values):
        return [f"{v/1_000_000:.2f}M" if v>=1_000_000 else f"{v/1000:.2f}K" for v in values]
    return pd.DataFrame({"EVENT_ID":np.arange(9_000_000,9_000_000+n),"STATE":state,"YEAR":YEAR,"MONTH_NAME":month_names,"EVENT_TYPE":event,"CZ_TYPE":rng.choice(["C","Z"],n,p=[.72,.28]),"BEGIN_DATE_TIME":begin.dt.strftime("%d-%b-%y %H:%M:%S"),"DAMAGE_PROPERTY":token(total*(1-crop_share)),"DAMAGE_CROPS":token(total*crop_share),"MAGNITUDE":magnitude,"BEGIN_LAT":rng.uniform(25,49,n).round(4),"BEGIN_LON":rng.uniform(-124,-67,n).round(4)})


def load_dataset() -> tuple[pd.DataFrame,dict]:
    try:
        filename=_discover(); payload=_download(filename); frame=_parse(payload); mode="live"; reason=""; source_hash=hashlib.sha256(payload).hexdigest()
    except Exception as exc:
        filename=f"deterministic-fallback-{YEAR}.csv"; frame=_fallback(); mode="demo"; reason=str(exc); source_hash=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return frame,{"mode":mode,"fallback_reason":reason,"filename":filename,"source_url":DIRECTORY_URL+filename if mode=="live" else DATABASE_URL,"source_hash":source_hash,"retrieved_at":datetime.now(timezone.utc).isoformat(),"year":YEAR}
