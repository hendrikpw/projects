"""Safe UCI archive extraction with deterministic source-shaped fallback."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import requests

SOURCE_URL="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_URL="https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection"
DOI_URL="https://doi.org/10.24432/C5CC84"
MEMBERS={"SMSSpamCollection","readme"}; MAX_BYTES=1_000_000


def _download(retries=3):
    last=None
    for attempt in range(retries):
        try:
            r=requests.get(SOURCE_URL,timeout=(5,25),headers={"User-Agent":"hendrikpw-message-trust-gateway/1.0"}); r.raise_for_status()
            if not 10_000<len(r.content)<=MAX_BYTES: raise ValueError("archive size outside safety bounds")
            return r.content
        except (requests.RequestException,ValueError) as exc:
            last=exc
            if attempt<retries-1: time.sleep(.25*2**attempt)
    raise RuntimeError(f"source unavailable after {retries} attempts: {last}")


def _parse(payload: bytes):
    if not zipfile.is_zipfile(io.BytesIO(payload)): raise ValueError("payload is not ZIP")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=set(z.namelist())
        if names!=MEMBERS or any(".." in n or n.startswith("/") for n in names): raise ValueError("archive allowlist failed")
        if sum(z.getinfo(n).file_size for n in names)>MAX_BYTES: raise ValueError("expanded archive too large")
        rows=[]
        for i,line in enumerate(z.read("SMSSpamCollection").decode("utf-8",errors="replace").splitlines(),1):
            label,sep,text=line.partition("\t"); rows.append({"source_row":i,"label":label,"message":text if sep else None})
    return pd.DataFrame(rows)


def _fallback(n=5_000):
    rng=np.random.default_rng(228); ham=["Are we still meeting at six?","Call me when you arrive","Can you pick up milk please","Happy birthday! Hope you have a great day","I will be home in twenty minutes","Thanks, see you tomorrow","The train is delayed again","Lunch at the usual place?"]; spam=["URGENT! Claim your free prize now. Call 09001234567","WIN cash today, text WIN to 80080","Exclusive offer! Click http://win.example now","You have been selected for a £1000 reward","FREE entry in our weekly draw. Reply YES","Final notice: claim your bonus before midnight"]
    labels=rng.choice(["ham","spam"],n,p=[.865,.135]); texts=[rng.choice(spam if y=="spam" else ham)+(" "+str(i%17) if rng.random()<.72 else "") for i,y in enumerate(labels)]
    return pd.DataFrame({"source_row":np.arange(1,n+1),"label":labels,"message":texts})


def load_dataset():
    try:
        payload=_download(); frame=_parse(payload); mode,reason,digest="live","",hashlib.sha256(payload).hexdigest()
    except Exception as exc:
        frame=_fallback(); mode,reason="demo",str(exc); digest=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return frame,{"mode":mode,"fallback_reason":reason,"source_url":SOURCE_URL,"dataset_url":DATASET_URL,"source_hash":digest,"retrieved_at":datetime.now(timezone.utc).isoformat()}
