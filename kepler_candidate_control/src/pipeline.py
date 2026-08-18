"""Content-addressed Bronze/Silver/Gold KOI pipeline."""
from __future__ import annotations
import hashlib,time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from kepler_candidate_control.src.data import COLUMNS,load_source

FEATURES=["koi_period","koi_duration","koi_depth","koi_prad","koi_teq","koi_insol","koi_steff","koi_slogg","koi_srad","koi_kepmag","koi_count","koi_num_transits","koi_model_snr"]
@dataclass(frozen=True)
class DataProduct:
    bronze:pd.DataFrame; silver:pd.DataFrame; gold:pd.DataFrame; quarantine:pd.DataFrame; quality:pd.DataFrame; stages:pd.DataFrame; metadata:dict
def _hash(frame): return hashlib.sha256(frame.sort_index(axis=1).sort_values(list(frame.columns),kind="stable").to_csv(index=False).encode()).hexdigest()
def _bronze(raw):
    x=raw.copy(); x["event_id"]=x.kepoi_name.astype(str).map(lambda v:hashlib.sha256(v.encode()).hexdigest()[:24]); x["payload_hash"]=[hashlib.sha256("|".join(map(str,r)).encode()).hexdigest() for r in x[COLUMNS].itertuples(index=False,name=None)]; replay=x[x.koi_period.gt(0)&x.koi_duration.gt(0)&x.koi_depth.gt(0)].sort_values("event_id").head(20); return pd.concat([x,replay],ignore_index=True)
def _silver(bronze):
    x=bronze.copy();
    for c in ["kepid"]+FEATURES+["ra","dec"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    valid=x.kepoi_name.str.fullmatch(r"K\d{5}\.?\d{2}",na=False)&x.koi_disposition.isin(["CONFIRMED","CANDIDATE","FALSE POSITIVE"])&x.kepid.notna()&x.koi_period.gt(0)&x.koi_duration.gt(0)&x.koi_depth.gt(0)&x.ra.between(0,360)&x.dec.between(-90,90)
    x["reason"]=np.select([~x.kepoi_name.str.fullmatch(r"K\d{5}\.?\d{2}",na=False),~x.koi_disposition.isin(["CONFIRMED","CANDIDATE","FALSE POSITIVE"]),x.kepid.isna(),~x.koi_period.gt(0),~x.koi_duration.gt(0),~x.koi_depth.gt(0),~x.ra.between(0,360)|~x.dec.between(-90,90)],["invalid_koi_name","invalid_disposition","missing_star_id","invalid_period","invalid_duration","invalid_depth","invalid_coordinates"],default="contract_failure")
    q=x[~valid].copy(); s=x[valid].sort_values("payload_hash").drop_duplicates("event_id").drop(columns="reason").copy(); s["planet_like"]=s.koi_disposition.isin(["CONFIRMED","CANDIDATE"]).astype(int); s["missing_features"]=s[FEATURES].isna().sum(axis=1); return s.sort_values("event_id").reset_index(drop=True),q.reset_index(drop=True)
def run_pipeline():
    started=time.perf_counter(); raw,meta=load_source(); extract_ms=(time.perf_counter()-started)*1000; bronze=_bronze(raw); silver,q=_silver(bronze); gold=silver[["event_id","kepid","kepoi_name","koi_disposition","planet_like","missing_features","ra","dec"]+FEATURES].copy(); dup=len(bronze)-bronze.event_id.nunique()
    rate=float(gold.planet_like.mean()); gates=[("source_volume",7000<=len(silver)<=12000,f"{len(silver):,} unique KOIs"),("unique_event_id",not silver.event_id.duplicated().any(),"stable KOI key"),("replay_suppression",dup==20,f"{dup} duplicates suppressed"),("row_reconciliation",len(bronze)==len(silver)+len(q)+dup,f"{len(bronze):,} deliveries reconciled"),("disposition_contract",silver.koi_disposition.isin(["CONFIRMED","CANDIDATE","FALSE POSITIVE"]).all(),"three documented states"),("physical_ranges",(silver.koi_period.gt(0)&silver.koi_duration.gt(0)&silver.koi_depth.gt(0)).all(),"positive transit quantities"),("coordinate_contract",silver.ra.between(0,360).all()&silver.dec.between(-90,90).all(),"ICRS bounds"),("label_balance",.2<=rate<=.8,f"{rate:.1%} planet-like"),("feature_coverage",gold[FEATURES].notna().mean().mean()>.90,f"{gold[FEATURES].notna().mean().mean():.1%} populated"),("leakage_block",not {"koi_disposition","koi_pdisposition","koi_score"}&set(FEATURES),"vetting outcomes excluded")]
    quality=pd.DataFrame(gates,columns=["check","passed","detail"])
    if not quality.passed.all(): raise RuntimeError("KOI data product failed publication gates")
    hashes={"bronze_hash":_hash(bronze),"silver_hash":_hash(silver),"gold_hash":_hash(gold)}; run_id=hashlib.sha256((meta["source_hash"]+"".join(hashes.values())).encode()).hexdigest()[:12]
    stages=pd.DataFrame([{"stage":"Extract","input":meta["source_bytes"],"output":len(raw),"rejected":0,"duration_ms":round(extract_ms,1),"hash":meta["source_hash"][:12]},{"stage":"Bronze","input":len(raw),"output":len(bronze),"rejected":0,"duration_ms":0,"hash":hashes["bronze_hash"][:12]},{"stage":"Silver","input":len(bronze),"output":len(silver),"rejected":len(q)+dup,"duration_ms":0,"hash":hashes["silver_hash"][:12]},{"stage":"Gold","input":len(silver),"output":len(gold),"rejected":0,"duration_ms":0,"hash":hashes["gold_hash"][:12]}])
    return DataProduct(bronze,silver,gold,q,quality,stages,{**meta,**hashes,"run_id":run_id,"duplicates":dup,"quarantine":len(q)})
