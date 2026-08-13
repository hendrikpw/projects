"""Group-isolated, calibrated and adversarially evaluated text gateway."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.pipeline import FeatureUnion,Pipeline

from message_trust_gateway.src.pipeline import redact_text,normalize_text


@dataclass(frozen=True)
class ModelBundle:
    model: Pipeline; baseline: Pipeline; calibrator: LogisticRegression
    calibration: pd.DataFrame; evaluation: pd.DataFrame; metrics: dict
    drift: pd.DataFrame; features: pd.DataFrame; metadata: dict


def _split(gold):
    bucket=gold.group_hash.str[:8].map(lambda x:int(x,16)%100); return gold[bucket<70].copy(),gold[bucket.between(70,84)].copy(),gold[bucket>=85].copy()


def _pipeline(robust=True):
    if robust:
        vector=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.995,sublinear_tf=True,max_features=18_000)),("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=28_000))])
    else: vector=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=20_000)
    return Pipeline([("vectorizer",vector),("classifier",LogisticRegression(C=4.0,class_weight="balanced",max_iter=1_500,random_state=42))])


def _thresholds(y,p,spam_precision=.95,ham_precision=.99):
    candidates=np.linspace(.01,.99,99); high=[]; low=[]
    for t in candidates:
        block=p>=t; allow=p<=t
        if block.any() and y[block].mean()>=spam_precision: high.append(t)
        if allow.any() and (1-y[allow]).mean()>=ham_precision: low.append(t)
    low_t,high_t=(max(low) if low else .10),(min(high) if high else .90)
    if low_t>=high_t:
        midpoint=(low_t+high_t)/2; low_t=max(0.0,midpoint-.01); high_t=min(1.0,midpoint+.01)
    return low_t,high_t


def adversarial_text(text):
    replacements={"free":"fr33","prize":"pr!ze","win":"w i n","call":"c@ll","claim":"cla1m","cash":"ca$h","urgent":"urg ent"}; out=text
    for old,new in replacements.items(): out=out.replace(old,new).replace(old.title(),new.title()).replace(old.upper(),new.upper())
    return out


def _psi(ref,cur):
    edges=np.unique(np.quantile(ref,np.linspace(0,1,11)))
    if len(edges)<2: return 0.0 if np.all(np.asarray(cur)==edges[0]) else 1.0
    edges[0],edges[-1]=-np.inf,np.inf; a=pd.cut(ref,edges,include_lowest=True).value_counts(normalize=True,sort=False); b=pd.cut(cur,edges,include_lowest=True).value_counts(normalize=True,sort=False); a=np.clip(np.asarray(a),1e-6,None); b=np.clip(np.asarray(b),1e-6,None); return float(np.sum((b-a)*np.log(b/a)))


def _feature_names(model):
    vector=model.named_steps["vectorizer"]; names=vector.get_feature_names_out(); coef=model.named_steps["classifier"].coef_[0]; order=np.argsort(np.abs(coef))[::-1][:40]; return pd.DataFrame({"feature":names[order],"weight":coef[order],"direction":np.where(coef[order]>0,"spam","ham")})


def train_and_evaluate(gold):
    train,cal,test=_split(gold)
    if min(len(train),len(cal),len(test))<300 or min(train.target.sum(),cal.target.sum(),test.target.sum())<30: raise ValueError("insufficient group-isolated split")
    model=_pipeline(True).fit(train.message_redacted,train.target); baseline=_pipeline(False).fit(train.message_redacted,train.target)
    raw_cal=model.decision_function(cal.message_redacted); calibrator=LogisticRegression(C=10.0,random_state=42).fit(raw_cal.reshape(-1,1),cal.target); cal_p=calibrator.predict_proba(raw_cal.reshape(-1,1))[:,1]; low,high=_thresholds(cal.target.to_numpy(),cal_p)
    raw=model.decision_function(test.message_redacted); p=calibrator.predict_proba(raw.reshape(-1,1))[:,1]; base=baseline.predict_proba(test.message_redacted)[:,1]; decision=np.select([p>=high,p<=low],["block","allow"],default="review"); predicted=p>=.5
    adv=test.message_redacted.map(adversarial_text); adv_raw=model.decision_function(adv); adv_p=calibrator.predict_proba(adv_raw.reshape(-1,1))[:,1]; spam=test.target.eq(1).to_numpy(); clean_block=(p>=high); adv_block=(adv_p>=high)
    tn,fp,fn,tp=confusion_matrix(test.target,predicted,labels=[0,1]).ravel(); metrics={"average_precision":average_precision_score(test.target,p),"baseline_average_precision":average_precision_score(test.target,base),"roc_auc":roc_auc_score(test.target,p),"brier":brier_score_loss(test.target,p),"precision":precision_score(test.target,predicted,zero_division=0),"recall":recall_score(test.target,predicted,zero_division=0),"f1":f1_score(test.target,predicted,zero_division=0),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"low_threshold":low,"high_threshold":high,"auto_coverage":float(np.mean(decision!="review")),"review_rate":float(np.mean(decision=="review")),"blocked_spam_recall":float(clean_block[spam].mean()),"adversarial_blocked_recall":float(adv_block[spam].mean()),"adversarial_ap":average_precision_score(test.target,adv_p),"prevalence":float(test.target.mean())}
    evaluation=test[["message_id","group_hash","target","message_redacted","length","digit_ratio","upper_ratio","token_count","pii_tokens"]].copy(); evaluation["spam_probability"]=p; evaluation["baseline_probability"]=base; evaluation["decision"]=decision; evaluation["adversarial_probability"]=adv_p
    drift=pd.DataFrame({"feature":["length","digit_ratio","upper_ratio","token_count","pii_tokens"],"psi":[_psi(train[c],test[c]) for c in ["length","digit_ratio","upper_ratio","token_count","pii_tokens"]]}); drift["status"]=np.select([drift.psi>=.25,drift.psi>=.1],["high","watch"],default="stable")
    calibration=cal[["message_id","target"]].copy(); calibration["raw_probability"]=raw_cal; calibration["spam_probability"]=cal_p
    return ModelBundle(model,baseline,calibrator,calibration,evaluation,metrics,drift,_feature_names(model),{"seed":42,"candidate":"word 1-2 + char 3-5 TF-IDF / class-weighted logistic","baseline":"word 1-2 TF-IDF / class-weighted logistic","calibration":"Platt logistic C=10 on calibration groups","split":"duplicate-group hash 70/15/15","train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test),"train_groups":train.group_hash.nunique(),"calibration_groups":cal.group_hash.nunique(),"test_groups":test.group_hash.nunique()})


def score_message(bundle,text):
    safe=redact_text(normalize_text(text)); raw=float(bundle.model.decision_function([safe])[0]); p=float(bundle.calibrator.predict_proba([[raw]])[:,1][0]); decision="block" if p>=bundle.metrics["high_threshold"] else "allow" if p<=bundle.metrics["low_threshold"] else "review"; vector=bundle.model.named_steps["vectorizer"].transform([safe]); coef=bundle.model.named_steps["classifier"].coef_[0]; names=bundle.model.named_steps["vectorizer"].get_feature_names_out(); contrib=vector.multiply(coef).toarray()[0]; idx=np.argsort(np.abs(contrib))[::-1][:8]; evidence=pd.DataFrame({"feature":names[idx],"contribution":contrib[idx]}); evidence=evidence[evidence.contribution.ne(0)]; return {"redacted_text":safe,"spam_probability":p,"decision":decision,"evidence":evidence}
