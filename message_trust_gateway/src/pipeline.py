"""Idempotent message pipeline, privacy transforms, contracts and observability."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from message_trust_gateway.src.data import load_dataset

PII_PATTERNS=[(r"https?://\S+|www\.\S+","<URL>"),(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}","<EMAIL>"),(r"(?<!\w)(?:\+?\d[\d ()-]{6,}\d)(?!\w)","<PHONE>"),(r"£\s?\d+(?:[.,]\d+)?|\$\s?\d+(?:[.,]\d+)?","<MONEY>")]


@dataclass(frozen=True)
class PipelineBundle:
    bronze: pd.DataFrame; silver: pd.DataFrame; gold: pd.DataFrame; quarantine: pd.DataFrame
    batches: pd.DataFrame; stages: pd.DataFrame; quality: pd.DataFrame; metadata: dict


def normalize_text(value):
    text=unicodedata.normalize("NFKC",str(value)).replace("\x00",""); return re.sub(r"\s+"," ",text).strip()


def redact_text(text):
    out=text
    for pattern,replacement in PII_PATTERNS: out=re.sub(pattern,replacement,out,flags=re.I)
    return out


def _hash_frame(frame):
    f=frame.copy()
    if len(f): f=f.sort_index(axis=1).sort_values(list(f.columns),kind="stable")
    return hashlib.sha256(f.to_csv(index=False,float_format="%.8g").encode()).hexdigest()


def _silver(bronze):
    f=bronze.copy(); f["label"]=f.label.astype("string").str.strip().str.lower(); f["message_normalized"]=f.message.map(normalize_text); f["group_hash"]=f.message_normalized.str.lower().map(lambda x:hashlib.sha256(x.encode()).hexdigest()); f["message_id"]=[hashlib.sha256(f"{h}:{r}".encode()).hexdigest()[:20] for h,r in zip(f.group_hash,f.source_row)]
    reason=pd.Series(pd.NA,index=f.index,dtype="string"); rules=[(~f.label.isin(["ham","spam"]),"invalid_label"),(f.message.isna()|f.message_normalized.eq(""),"empty_message"),(f.message_normalized.str.len().gt(2_000),"message_too_long"),(f.message_id.duplicated(),"duplicate_event_id")]
    for mask,label in rules: reason.loc[mask.fillna(True)&reason.isna()]=label
    q=f.loc[reason.notna()].copy(); q["invalid_reason"]=reason.dropna(); s=f.loc[reason.isna()].sort_values("source_row").reset_index(drop=True); s["target"]=s.label.eq("spam").astype(int); return s,q.reset_index(drop=True)


def _gold(silver):
    g=silver[["message_id","group_hash","target","message_normalized"]].copy(); g["message_redacted"]=g.message_normalized.map(redact_text); g["length"]=g.message_redacted.str.len(); g["digit_ratio"]=g.message_redacted.str.count(r"\d")/g.length.clip(lower=1); g["upper_ratio"]=g.message_redacted.str.count(r"[A-Z]")/g.length.clip(lower=1); g["token_count"]=g.message_redacted.str.split().str.len(); g["pii_tokens"]=g.message_redacted.str.count(r"<(?:URL|EMAIL|PHONE|MONEY)>"); return g.drop(columns="message_normalized")


def _checks(bronze,silver,q,gold):
    prevalence=float(gold.target.mean())
    checks=[("source_schema",{"source_row","label","message"}.issubset(bronze.columns),"three required fields"),("row_reconciliation",len(bronze)==len(silver)+len(q),f"{len(bronze):,} = {len(silver):,} + {len(q):,}"),("event_identity",silver.message_id.is_unique,f"{len(silver):,} unique ingestion events"),("label_domain",silver.target.isin([0,1]).all(),"ham/spam only"),("privacy_contract",not gold.message_redacted.str.contains(r"https?://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",regex=True).any(),"URLs and emails tokenized"),("group_integrity",silver.groupby("group_hash").target.nunique().max()==1,"duplicate groups have one label"),("feature_completeness",gold[["message_redacted","length","digit_ratio","upper_ratio","token_count","pii_tokens"]].notna().all().all(),"all serving inputs complete"),("minimum_scale",len(gold)>=4_000 and gold.target.sum()>=500,f"{len(gold):,} events / {gold.target.sum():,} spam"),("class_balance",.05<=prevalence<=.30,f"spam prevalence {prevalence:.1%}"),("gold_reconciliation",len(gold)==len(silver),f"{len(gold):,} publishable events")]; return pd.DataFrame(checks,columns=["check","passed","detail"])


def run_pipeline(batch_size=500):
    bronze,meta=load_dataset(); batch_rows=[]; seen=set()
    # Append a bounded replay to the delivery stream. The governed Bronze layer still
    # contains each source row once; this makes idempotency visible and testable.
    replay_count=min(25,len(bronze)); deliveries=pd.concat([bronze,bronze.head(replay_count)],ignore_index=True)
    for batch_id,start in enumerate(range(0,len(deliveries),batch_size),1):
        batch=deliveries.iloc[start:start+batch_size]; ids=[hashlib.sha256(f"{r.source_row}:{r.label}:{r.message}".encode()).hexdigest() for r in batch.itertuples()]; accepted=sum(i not in seen for i in ids); duplicates=len(ids)-accepted; seen.update(ids); batch_rows.append((batch_id,len(batch),accepted,duplicates,hashlib.sha256("".join(ids).encode()).hexdigest()))
    batches=pd.DataFrame(batch_rows,columns=["batch_id","received","accepted","duplicate_deliveries","payload_hash"])
    ledger=[]; t=time.perf_counter(); bh=_hash_frame(bronze); ledger.append(("Bronze",len(bronze),len(bronze),0,(time.perf_counter()-t)*1000,bh)); t=time.perf_counter(); silver,q=_silver(bronze); sh=_hash_frame(silver); ledger.append(("Silver",len(bronze),len(silver),len(q),(time.perf_counter()-t)*1000,sh)); t=time.perf_counter(); gold=_gold(silver); gh=_hash_frame(gold); ledger.append(("Gold",len(silver),len(gold),0,(time.perf_counter()-t)*1000,gh)); quality=_checks(bronze,silver,q,gold)
    if not quality.passed.all(): raise RuntimeError("data product withheld: "+", ".join(quality.loc[~quality.passed,"check"]))
    run_id=hashlib.sha256(f"{meta['source_hash']}:{gh}".encode()).hexdigest()[:12]; stages=pd.DataFrame(ledger,columns=["stage","input_rows","output_rows","rejected_rows","duration_ms","content_hash"]); stages["status"]="passed"; meta={**meta,"run_id":run_id,"bronze_hash":bh,"silver_hash":sh,"gold_hash":gh,"manifest_hash":hashlib.sha256(json.dumps({"run_id":run_id,"pii_patterns":len(PII_PATTERNS)},sort_keys=True).encode()).hexdigest(),"duplicate_groups":int(silver.group_hash.duplicated().sum()),"replayed_deliveries":int(batches.duplicate_deliveries.sum())}; return PipelineBundle(bronze,silver,gold,q,batches,stages,quality,meta)
