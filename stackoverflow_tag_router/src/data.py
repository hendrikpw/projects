"""Bounded Stack Exchange ingestion with backoff handling and deterministic fallback."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import requests

API_URL="https://api.stackexchange.com/2.3/questions"
DOCS_URL="https://api.stackexchange.com/docs/questions"
LICENSE_URL="https://stackoverflow.com/help/licensing"


def _request_page(page:int,page_size:int=100,retries:int=3):
    params={"site":"stackoverflow","page":page,"pagesize":page_size,"order":"desc","sort":"creation","filter":"withbody"}; last=None
    for attempt in range(retries):
        try:
            response=requests.get(API_URL,params=params,timeout=(5,25),headers={"User-Agent":"hendrikpw-tag-router/1.0"}); response.raise_for_status(); payload=response.json()
            if "error_id" in payload: raise RuntimeError(payload.get("error_message","API error"))
            if payload.get("backoff"): time.sleep(min(float(payload["backoff"]),10.0))
            return payload
        except (requests.RequestException,ValueError,RuntimeError) as exc:
            last=exc
            if attempt<retries-1: time.sleep(.35*2**attempt)
    raise RuntimeError(f"Stack Exchange unavailable after {retries} attempts: {last}")


def _fallback(n:int=1_800):
    rng=np.random.default_rng(42); start=pd.Timestamp("2026-01-01",tz="UTC")
    specs=[
        (("python","pandas"),"Python pandas dataframe merge returns duplicate rows","How can I merge two dataframes by id and keep only the newest row? <pre><code>df.merge(other)</code></pre>"),
        (("javascript","reactjs"),"React state does not update after fetch","Why does my component render stale state after an async fetch call?"),
        (("java","spring-boot"),"Spring Boot dependency injection fails","My service bean is not found when the application context starts."),
        (("c#",".net"),"Deserialize JSON response in C#","How should I map this API response to a typed record in .NET?"),
        (("sql","mysql"),"SQL query group by latest timestamp","I need one row per customer using the maximum created date."),
        (("python","django"),"Django migration cannot add field","The migration fails after adding a non nullable model field."),
        (("javascript","node.js"),"Node API returns promise instead of value","My async route sends before the database promise resolves."),
        (("html","css"),"CSS grid overflows mobile screen","A responsive card grid creates horizontal scrolling on small screens."),
        (("android","kotlin"),"Kotlin coroutine crashes Android view","The lifecycle scope call updates a view after the fragment is destroyed."),
        (("docker","linux"),"Docker container cannot access mounted file","The Linux container reports permission denied for a bind mount."),
        (("git","github"),"Git rebase created merge conflicts","How can I keep the branch changes and finish an interactive rebase?"),
        (("typescript","angular"),"Angular observable type mismatch","The TypeScript compiler rejects the observable returned by my service."),
    ]; rows=[]
    for i in range(n):
        tags,title,body=specs[i%len(specs)]; noise=["after upgrade","in production","minimal example","unit test fails"][i%4]
        if i%5==0: body="How can I debug this unexpected behavior? The documented example works but my reduced case fails after an upgrade."
        if i%7==0: title=f"Unexpected runtime behavior — {noise}"
        label_tags=specs[(i+1)%len(specs)][0] if i%17==0 else tags
        created=start+pd.to_timedelta(37*i,unit="min"); rows.append({"question_id":90_000_000+i,"creation_date":int(created.timestamp()),"last_activity_date":int((created+pd.to_timedelta(i%36,unit="h")).timestamp()),"title":f"{title} — {noise}","body":f"<p>{body}</p><p>Example {i%29}</p>","tags":list(label_tags),"score":int(rng.integers(-2,18)),"view_count":int(rng.integers(5,3000)),"answer_count":int(rng.integers(0,8)),"is_answered":bool(rng.random()>.35),"link":f"https://stackoverflow.com/questions/{90_000_000+i}"})
    return pd.DataFrame(rows)


def load_questions(pages:int=12,page_size:int=100):
    try:
        items=[]; quota=None; has_more=True
        for page in range(1,pages+1):
            if not has_more: break
            payload=_request_page(page,page_size); items.extend(payload.get("items",[])); quota=payload.get("quota_remaining",quota); has_more=bool(payload.get("has_more",False))
        if len(items)<500: raise ValueError(f"only {len(items)} questions returned")
        frame=pd.DataFrame(items); mode,reason="live",""
    except Exception as exc:
        frame=_fallback(); mode,reason,quota="demo",str(exc),None
    digest=hashlib.sha256(frame.to_json(orient="records",date_format="iso",default_handler=str).encode()).hexdigest()
    return frame,{"mode":mode,"fallback_reason":reason,"source_url":API_URL,"docs_url":DOCS_URL,"source_hash":digest,"quota_remaining":quota,"retrieved_at":datetime.now(timezone.utc).isoformat(),"requested_pages":pages,"page_size":page_size}
