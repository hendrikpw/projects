"""Contracted Bronze/Silver/Gold flight operations pipeline."""
from __future__ import annotations
import hashlib,time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from flight_delay_control.src.data import load_source

@dataclass(frozen=True)
class DataProduct:
    bronze:pd.DataFrame; silver:pd.DataFrame; gold:pd.DataFrame; quarantine:pd.DataFrame; quality:pd.DataFrame; stages:pd.DataFrame; metadata:dict

def _hash(frame):
    x=frame.copy()
    for c in x.select_dtypes(include=["datetime"]): x[c]=x[c].astype(str)
    return hashlib.sha256(x.sort_index(axis=1).sort_values(list(x.columns),kind="stable").to_csv(index=False).encode()).hexdigest()

def _bronze(frame):
    x=frame.copy(); x.columns=[c.strip() for c in x.columns]
    x["event_id"]=[hashlib.sha256(f"{r.FlightDate}|{r.Reporting_Airline}|{r.Flight_Number_Reporting_Airline}|{r.Origin}|{r.Dest}|{r.CRSDepTime}".encode()).hexdigest()[:24] for r in x.itertuples()]
    x["payload_hash"]=[hashlib.sha256("|".join(map(str,r)).encode()).hexdigest() for r in x.astype(str).itertuples(index=False,name=None)]
    replay=x.sort_values("event_id").head(25).copy(); replay["payload_hash"]=replay.payload_hash
    return pd.concat([x,replay],ignore_index=True)

def _silver(bronze):
    x=bronze.copy(); x["FlightDate"]=pd.to_datetime(x.FlightDate,errors="coerce")
    for c in ["DayofMonth","DayOfWeek","CRSDepTime","CRSArrTime","CRSElapsedTime","Distance","Cancelled","Diverted","ArrDel15"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    valid=x.FlightDate.notna()&x.Reporting_Airline.notna()&x.Origin.str.fullmatch(r"[A-Z]{3}",na=False)&x.Dest.str.fullmatch(r"[A-Z]{3}",na=False)&x.Distance.between(20,6000)&x.CRSDepTime.between(0,2359)&x.Cancelled.isin([0,1])&x.Diverted.isin([0,1])
    x["reason"]=np.select([x.FlightDate.isna(),~x.Origin.str.fullmatch(r"[A-Z]{3}",na=False),~x.Dest.str.fullmatch(r"[A-Z]{3}",na=False),~x.Distance.between(20,6000),~x.CRSDepTime.between(0,2359)],["invalid_date","invalid_origin","invalid_destination","invalid_distance","invalid_schedule_time"],default="contract_failure")
    q=x.loc[~valid].copy(); s=x.loc[valid].drop(columns="reason").sort_values("payload_hash").drop_duplicates("event_id").copy()
    s["is_operated"]=(s.Cancelled==0)&(s.Diverted==0); s["is_delayed_15"]=np.where(s.is_operated,s.ArrDel15,np.nan)
    s["scheduled_hour"]=(s.CRSDepTime//100).astype(int); s["route"]=s.Origin+"–"+s.Dest
    return s.sort_values(["FlightDate","event_id"]).reset_index(drop=True),q.reset_index(drop=True)

FEATURES=["Reporting_Airline","Origin","Dest","DayOfWeek","scheduled_hour","CRSElapsedTime","Distance"]
def run_pipeline():
    start=time.perf_counter(); source,meta=load_source(); elapsed=(time.perf_counter()-start)*1000; bronze=_bronze(source); silver,q=_silver(bronze); gold=silver[silver.is_operated&silver.is_delayed_15.notna()][["event_id","FlightDate","route","ArrDelayMinutes","is_delayed_15"]+FEATURES].copy()
    dup=len(bronze)-bronze.event_id.nunique(); checks=[("sample_volume",len(silver)>=50_000,f"{len(silver):,} unique flights"),("unique_event_id",not silver.event_id.duplicated().any(),"stable natural-key hash"),("replay_suppression",dup==25,f"{dup} duplicates suppressed"),("row_reconciliation",len(bronze)==len(silver)+len(q)+dup,f"{len(bronze):,} deliveries reconciled"),("schedule_ranges",silver.CRSDepTime.between(0,2359).all(),"HHMM bounds"),("airport_contract",silver.Origin.str.len().eq(3).all()&silver.Dest.str.len().eq(3).all(),"IATA-like codes"),("label_contract",gold.is_delayed_15.isin([0,1]).all(),"operated-flight label"),("no_outcome_leakage",not set(["ArrDelayMinutes","ArrDel15","Cancelled","Diverted"])&set(FEATURES),"outcomes excluded"),("temporal_coverage",gold.FlightDate.dt.day.nunique()>=28,f"{gold.FlightDate.dt.day.nunique()} days"),("source_bound",meta["source_bytes"]<45_000_000,f"{meta['source_bytes']:,} bytes")]
    quality=pd.DataFrame(checks,columns=["check","passed","detail"])
    if not quality.passed.all(): raise RuntimeError("flight data product failed publication gates")
    hashes={"bronze_hash":_hash(bronze),"silver_hash":_hash(silver),"gold_hash":_hash(gold)}; run_id=hashlib.sha256((meta["source_hash"]+"".join(hashes.values())).encode()).hexdigest()[:12]
    stages=pd.DataFrame([{"stage":"Extract","input":meta["source_bytes"],"output":len(source),"rejected":0,"duration_ms":round(elapsed,1),"hash":meta["source_hash"][:12]},{"stage":"Bronze","input":len(source),"output":len(bronze),"rejected":0,"duration_ms":0,"hash":hashes["bronze_hash"][:12]},{"stage":"Silver","input":len(bronze),"output":len(silver),"rejected":len(q)+dup,"duration_ms":0,"hash":hashes["silver_hash"][:12]},{"stage":"Gold","input":len(silver),"output":len(gold),"rejected":len(silver)-len(gold),"duration_ms":0,"hash":hashes["gold_hash"][:12]}])
    return DataProduct(bronze,silver,gold,q,quality,stages,{**meta,**hashes,"run_id":run_id,"duplicates":dup,"quarantine":len(q),"cancelled":int((silver.Cancelled==1).sum()),"diverted":int((silver.Diverted==1).sum())})
