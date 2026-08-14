"""Replay-safe relational question pipeline with contracts, lineage and attribution."""

from __future__ import annotations

import hashlib,html,json,re,time
from dataclasses import dataclass

import pandas as pd

from stackoverflow_tag_router.src.data import load_questions


@dataclass(frozen=True)
class PipelineBundle:
    bronze:pd.DataFrame; silver:pd.DataFrame; gold:pd.DataFrame; tag_bridge:pd.DataFrame; quarantine:pd.DataFrame; batches:pd.DataFrame; stages:pd.DataFrame; quality:pd.DataFrame; metadata:dict


def clean_html(value):
    text=re.sub(r"<pre><code>(.*?)</code></pre>",r" CODE_BLOCK \1 ",str(value),flags=re.I|re.S); text=re.sub(r"<[^>]+>"," ",text); return re.sub(r"\s+"," ",html.unescape(text)).strip()


def redact_text(value):
    out=re.sub(r"https?://\S+|www\.\S+"," <URL> ",value,flags=re.I); out=re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"," <EMAIL> ",out); return re.sub(r"\s+"," ",out).strip()


def _hash_frame(frame):
    return hashlib.sha256(frame.sort_index(axis=1).to_json(orient="records",date_format="iso",default_handler=str).encode()).hexdigest()


def _transform(bronze):
    f=bronze.copy(); required=["question_id","creation_date","title","body","tags"]
    for column in required:
        if column not in f: f[column]=None
    f["question_id"]=pd.to_numeric(f.question_id,errors="coerce").astype("Int64"); f["created_at"]=pd.to_datetime(f.creation_date,unit="s",utc=True,errors="coerce"); f["title_clean"]=f.title.map(clean_html); f["body_clean"]=f.body.map(clean_html); f["tags_clean"]=f.tags.map(lambda x:sorted({str(t).strip().lower() for t in x}) if isinstance(x,list) else []); f["text_redacted"]=(f.title_clean+" "+f.body_clean).map(redact_text)
    reason=pd.Series(pd.NA,index=f.index,dtype="string"); rules=[(f.question_id.isna(),"missing_question_id"),(f.question_id.duplicated(),"duplicate_question_id"),(f.created_at.isna(),"invalid_creation_date"),(f.title_clean.eq("")|f.body_clean.eq(""),"empty_text"),(f.tags_clean.map(len).eq(0)|f.tags_clean.map(len).gt(5),"invalid_tag_count"),(f.text_redacted.str.len().gt(30_000),"text_too_long")]
    for mask,label in rules: reason.loc[mask.fillna(True)&reason.isna()]=label
    q=f.loc[reason.notna()].copy(); q["invalid_reason"]=reason.dropna(); s=f.loc[reason.isna()].sort_values(["created_at","question_id"]).reset_index(drop=True); s["event_id"]=s.question_id.map(lambda x:hashlib.sha256(f"stackoverflow:{x}".encode()).hexdigest()[:20]); s["source_link"]=s.apply(lambda r:str(r.get("link")) if pd.notna(r.get("link")) and str(r.get("link")).strip() else f"https://stackoverflow.com/questions/{r.question_id}",axis=1)
    bridge=s[["question_id","created_at","tags_clean"]].explode("tags_clean").rename(columns={"tags_clean":"tag"}).reset_index(drop=True)
    g=s[["event_id","question_id","created_at","source_link","tags_clean","text_redacted"]].copy(); g["text_length"]=g.text_redacted.str.len(); g["token_count"]=g.text_redacted.str.split().str.len(); g["code_blocks"]=g.text_redacted.str.count("CODE_BLOCK"); g["question_marks"]=g.text_redacted.str.count(r"\?"); return s,q.reset_index(drop=True),g,bridge


def _checks(bronze,silver,q,gold,bridge):
    checks=[("source_schema",{"question_id","creation_date","title","body","tags"}.issubset(bronze.columns),"required API fields available"),("row_reconciliation",len(bronze)==len(silver)+len(q),f"{len(bronze):,} = {len(silver):,} + {len(q):,}"),("event_identity",silver.event_id.is_unique,f"{silver.event_id.nunique():,} unique events"),("time_contract",silver.created_at.notna().all(),"all timestamps parsed as UTC"),("tag_contract",silver.tags_clean.map(lambda x:1<=len(x)<=5).all(),"one to five normalized tags"),("tag_reconciliation",len(bridge)==silver.tags_clean.map(len).sum(),f"{len(bridge):,} question-tag edges"),("privacy_contract",not gold.text_redacted.str.contains(r"https?://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",regex=True).any(),"URLs and emails tokenized"),("feature_completeness",gold[["text_redacted","text_length","token_count","code_blocks","question_marks"]].notna().all().all(),"all model inputs complete"),("minimum_scale",len(gold)>=500 and bridge.tag.nunique()>=10,f"{len(gold):,} questions / {bridge.tag.nunique():,} tags"),("gold_reconciliation",len(gold)==len(silver),f"{len(gold):,} publishable questions")]; return pd.DataFrame(checks,columns=["check","passed","detail"])


def run_pipeline(batch_size:int=200):
    bronze,meta=load_questions(); deliveries=pd.concat([bronze,bronze.head(min(20,len(bronze)))],ignore_index=True); seen=set(); rows=[]
    for batch_id,start in enumerate(range(0,len(deliveries),batch_size),1):
        batch=deliveries.iloc[start:start+batch_size]; ids=[hashlib.sha256(f"{r.get('question_id')}:{r.get('creation_date')}".encode()).hexdigest() for _,r in batch.iterrows()]; accepted=sum(i not in seen for i in ids); duplicates=len(ids)-accepted; seen.update(ids); rows.append((batch_id,len(ids),accepted,duplicates,hashlib.sha256("".join(ids).encode()).hexdigest()))
    batches=pd.DataFrame(rows,columns=["batch_id","received","accepted","duplicate_deliveries","payload_hash"]); ledger=[]; t=time.perf_counter(); bh=_hash_frame(bronze); ledger.append(("Bronze",len(bronze),len(bronze),0,(time.perf_counter()-t)*1000,bh)); t=time.perf_counter(); silver,q,gold,bridge=_transform(bronze); sh=_hash_frame(silver); ledger.append(("Silver",len(bronze),len(silver),len(q),(time.perf_counter()-t)*1000,sh)); t=time.perf_counter(); gh=_hash_frame(gold); ledger.append(("Gold",len(silver),len(gold),0,(time.perf_counter()-t)*1000,gh)); quality=_checks(bronze,silver,q,gold,bridge)
    if not quality.passed.all(): raise RuntimeError("data product withheld: "+", ".join(quality.loc[~quality.passed,"check"]))
    run_id=hashlib.sha256(f"{meta['source_hash']}:{gh}".encode()).hexdigest()[:12]; stages=pd.DataFrame(ledger,columns=["stage","input_rows","output_rows","rejected_rows","duration_ms","content_hash"]); stages["status"]="passed"; metadata={**meta,"run_id":run_id,"bronze_hash":bh,"silver_hash":sh,"gold_hash":gh,"replayed_deliveries":int(batches.duplicate_deliveries.sum()),"manifest_hash":hashlib.sha256(json.dumps({"run_id":run_id,"contract":"v1","license":"CC BY-SA 4.0"},sort_keys=True).encode()).hexdigest()}; return PipelineBundle(bronze,silver,gold,bridge,q,batches,stages,quality,metadata)
