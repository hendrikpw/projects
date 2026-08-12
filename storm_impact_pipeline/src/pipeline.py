"""Bronze/Silver/Gold transformations, contracts, quarantine and lineage."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from storm_impact_pipeline.src.data import USECOLS, load_dataset

FEATURES=["state","event_type","cz_type","month","begin_hour","magnitude","begin_lat","begin_lon"]
OUTCOMES=["damage_property_usd","damage_crops_usd","total_damage_usd","has_damage","log_damage"]


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame; silver: pd.DataFrame; gold: pd.DataFrame; quarantine: pd.DataFrame
    stages: pd.DataFrame; quality: pd.DataFrame; metadata: dict


def _hash_frame(frame: pd.DataFrame) -> str:
    out=frame.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]): out[c]=out[c].astype(str)
    if len(out): out=out.sort_index(axis=1).sort_values(list(out.columns),kind="stable")
    return hashlib.sha256(out.to_csv(index=False,float_format="%.8g").encode()).hexdigest()


def parse_damage(value) -> float:
    if pd.isna(value) or str(value).strip()=="": return np.nan
    text=str(value).strip().upper().replace("$","").replace(",","")
    match=re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMB]?)",text)
    if not match: return np.nan
    return float(match.group(1))*{"":1,"K":1_000,"M":1_000_000,"B":1_000_000_000}[match.group(2)]


def _silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    f=bronze.rename(columns={c:c.lower() for c in bronze.columns}).copy()
    for c in ["event_id","year","magnitude","begin_lat","begin_lon"]: f[c]=pd.to_numeric(f[c],errors="coerce")
    for c in ["state","month_name","event_type","cz_type"]: f[c]=f[c].astype("string").str.strip()
    f["begin_at"]=pd.to_datetime(f["begin_date_time"],format="mixed",errors="coerce")
    f["damage_property_usd"]=f["damage_property"].map(parse_damage); f["damage_crops_usd"]=f["damage_crops"].map(parse_damage)
    reason=pd.Series(pd.NA,index=f.index,dtype="string")
    rules=[(f.event_id.isna(),"invalid_event_id"),(f.event_id.duplicated(keep="first"),"duplicate_event_id"),(f.begin_at.isna(),"invalid_begin_time"),(f.event_type.isna()|f.event_type.eq(""),"missing_event_type"),(~f.cz_type.isin(["C","Z","M"]),"invalid_geography_type"),(~f.begin_lat.between(15,75),"invalid_latitude"),(~f.begin_lon.between(-180,-50),"invalid_longitude"),(f.damage_property_usd.isna()|f.damage_crops_usd.isna(),"incomplete_damage_label")]
    for mask,label in rules: reason.loc[mask.fillna(True)&reason.isna()]=label
    quarantine=f.loc[reason.notna()].copy(); quarantine["invalid_reason"]=reason.dropna()
    s=f.loc[reason.isna()].sort_values("event_id").reset_index(drop=True); s["event_id"]=s.event_id.astype(int); s["total_damage_usd"]=s.damage_property_usd+s.damage_crops_usd; s["has_damage"]=s.total_damage_usd.gt(0).astype(int); s["month"]=s.begin_at.dt.month; s["begin_hour"]=s.begin_at.dt.hour
    return s,quarantine.reset_index(drop=True)


def _gold(silver: pd.DataFrame) -> pd.DataFrame:
    gold=silver[["event_id","begin_at",*FEATURES,"damage_property_usd","damage_crops_usd","total_damage_usd","has_damage"]].copy(); gold["log_damage"]=np.log1p(gold.total_damage_usd); return gold


def _checks(bronze,silver,quarantine,gold):
    checks=[("source_schema",set(USECOLS).issubset(bronze.columns),f"{len(USECOLS)} required fields"),("row_reconciliation",len(bronze)==len(silver)+len(quarantine),f"{len(bronze):,} = {len(silver):,} + {len(quarantine):,}"),("event_identity",silver.event_id.is_unique and silver.event_id.notna().all(),f"{len(silver):,} unique events"),("damage_contract",gold.total_damage_usd.notna().all() and gold.total_damage_usd.ge(0).all(),"non-negative complete USD labels"),("target_reconciliation",(gold.has_damage==gold.total_damage_usd.gt(0)).all(),"binary label matches monetary outcome"),("feature_contract",FEATURES==[c for c in gold if c in FEATURES],f"{len(FEATURES)} ordered inputs"),("leakage_contract",not {"damage_property","damage_crops","injuries_direct","deaths_direct","event_narrative","tor_f_scale"}.intersection(FEATURES),"post-event fields excluded"),("time_coverage",set(range(1,13)).issubset(set(gold.month)),"all 12 months represented"),("minimum_scale",len(gold)>=10_000 and gold.has_damage.sum()>=500,f"{len(gold):,} labeled / {gold.has_damage.sum():,} damaged"),("gold_reconciliation",len(gold)==len(silver),f"{len(gold):,} publishable events")]
    return pd.DataFrame(checks,columns=["check","passed","detail"])


def run_pipeline():
    bronze,meta=load_dataset(); ledger=[]; t=time.perf_counter(); bh=_hash_frame(bronze); ledger.append(("Bronze",len(bronze),len(bronze),0,(time.perf_counter()-t)*1000,bh)); t=time.perf_counter(); silver,quarantine=_silver(bronze); sh=_hash_frame(silver); ledger.append(("Silver",len(bronze),len(silver),len(quarantine),(time.perf_counter()-t)*1000,sh)); t=time.perf_counter(); gold=_gold(silver); gh=_hash_frame(gold); ledger.append(("Gold",len(silver),len(gold),0,(time.perf_counter()-t)*1000,gh)); quality=_checks(bronze,silver,quarantine,gold)
    if not quality.passed.all(): raise RuntimeError("data product withheld: "+", ".join(quality.loc[~quality.passed,"check"]))
    run_id=hashlib.sha256(f"{meta['source_hash']}:{gh}".encode()).hexdigest()[:12]; stages=pd.DataFrame(ledger,columns=["stage","input_rows","output_rows","rejected_rows","duration_ms","content_hash"]); stages["status"]="passed"; meta={**meta,"run_id":run_id,"bronze_hash":bh,"silver_hash":sh,"gold_hash":gh,"manifest_hash":hashlib.sha256(json.dumps({"run_id":run_id,"features":FEATURES},sort_keys=True).encode()).hexdigest()}
    return PipelineBundle(bronze,silver,gold,quarantine,stages,quality,meta)
