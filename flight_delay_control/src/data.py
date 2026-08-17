"""Safe BTS monthly ZIP ingestion with deterministic bounded sampling."""
from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

MONTH = "2026_6"
URL = f"https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{MONTH}.zip"
DOCS = "https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD"
FIELDS = "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ"
DEFINITION = "https://www.transtats.bts.gov/ot_delay/OT_DelayCause1.asp?20=E"
USECOLS = ["FlightDate","DayofMonth","DayOfWeek","Reporting_Airline","Flight_Number_Reporting_Airline",
           "Origin","Dest","CRSDepTime","CRSArrTime","CRSElapsedTime","Distance","Cancelled","Diverted","ArrDel15","ArrDelayMinutes"]


def _download() -> bytes:
    response = requests.get(URL, headers={"User-Agent":"hendrikpw-projects flight-delay-control/1.0"}, timeout=(5,75))
    response.raise_for_status()
    if not 10_000_000 < len(response.content) < 45_000_000: raise ValueError("BTS archive outside safety bounds")
    return response.content


def _read_zip(raw: bytes, limit: int = 90_000) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv") and ".." not in n and not n.startswith("/")]
        if len(names) != 1: raise ValueError("expected exactly one safe CSV member")
        parts=[]
        with archive.open(names[0]) as handle:
            for chunk in pd.read_csv(handle, usecols=USECOLS, chunksize=60_000, low_memory=False):
                key=(chunk.FlightDate.astype(str)+"|"+chunk.Reporting_Airline.astype(str)+"|"+chunk.Flight_Number_Reporting_Airline.astype(str)+"|"+chunk.Origin.astype(str)+"|"+chunk.Dest.astype(str)+"|"+chunk.CRSDepTime.astype(str))
                mask=key.map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:8],16)%7==0)
                parts.append(chunk.loc[mask])
                if sum(len(x) for x in parts)>=limit: break
    frame=pd.concat(parts,ignore_index=True).head(limit)
    if len(frame)<50_000: raise ValueError("bounded sample unexpectedly small")
    return frame


def _fallback(n: int = 72_000) -> pd.DataFrame:
    rng=np.random.default_rng(20260817); carriers=np.array(["AA","DL","UA","WN","AS","B6","NK"]); airports=np.array(["ATL","DFW","DEN","ORD","LAX","JFK","SEA","MIA","BOS","PHX"])
    day=rng.integers(1,31,n); origin=rng.choice(airports,n); dest=rng.choice(airports,n); same=origin==dest; dest[same]=np.roll(dest,1)[same]
    carrier=rng.choice(carriers,n); dep=rng.integers(5,23,n)*100+rng.choice([0,15,30,45],n); distance=rng.integers(180,2600,n)
    risk=-2.2+.55*(dep//100>=18)+.45*np.isin(origin,["ORD","JFK"])+.25*np.isin(carrier,["B6","NK"])
    delayed=rng.random(n)<1/(1+np.exp(-risk)); cancelled=rng.random(n)<.012
    dates=pd.to_datetime({"year":2026,"month":6,"day":day})
    return pd.DataFrame({"FlightDate":dates,"DayofMonth":day,"DayOfWeek":dates.dt.dayofweek+1,
        "Reporting_Airline":carrier,"Flight_Number_Reporting_Airline":rng.integers(1,7000,n),"Origin":origin,"Dest":dest,"CRSDepTime":dep,"CRSArrTime":((dep//100+distance//500+1)%24)*100+dep%100,
        "CRSElapsedTime":distance/7+40,"Distance":distance,"Cancelled":cancelled.astype(float),"Diverted":np.zeros(n),"ArrDel15":np.where(cancelled,np.nan,delayed.astype(float)),"ArrDelayMinutes":np.where(delayed,rng.gamma(2,22,n),0)})


def load_source() -> tuple[pd.DataFrame,dict]:
    try:
        raw=_download(); frame=_read_zip(raw); mode="live"; reason=""; source_hash=hashlib.sha256(raw).hexdigest(); size=len(raw)
    except Exception as exc:
        frame=_fallback(); mode="demo"; reason=str(exc); payload=frame.to_csv(index=False).encode(); source_hash=hashlib.sha256(payload).hexdigest(); size=len(payload)
    return frame,{"mode":mode,"fallback_reason":reason,"source_hash":source_hash,"source_bytes":size,"source_url":URL,"retrieved_at":datetime.now(timezone.utc).isoformat()}
