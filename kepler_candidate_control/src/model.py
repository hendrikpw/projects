"""Group-isolated, calibrated Kepler candidate vetting model."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from kepler_candidate_control.src.pipeline import FEATURES

@dataclass(frozen=True)
class ModelProduct:
    model:Pipeline; calibrator:IsotonicRegression; evaluation:pd.DataFrame; metrics:dict; drift:pd.DataFrame; importance:pd.DataFrame; metadata:dict
def _bucket(v): return int(hashlib.sha256(str(int(v)).encode()).hexdigest()[:8],16)%10
def _psi(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); fill=np.nanmedian(a); a=np.nan_to_num(a,nan=fill); b=np.nan_to_num(b,nan=fill); edges=np.unique(np.quantile(a,np.linspace(0,1,11)))
    if len(edges)<3:return 0.0
    edges[0],edges[-1]=-np.inf,np.inf; x=np.clip(np.histogram(a,edges)[0]/len(a),1e-6,None); y=np.clip(np.histogram(b,edges)[0]/len(b),1e-6,None); return float(np.sum((y-x)*np.log(y/x)))
def train_and_evaluate(gold):
    split=gold.kepid.map(_bucket); train=gold[split<6].copy(); cal=gold[split.between(6,7)].copy(); test=gold[split>=8].copy()
    if min(len(train),len(cal),len(test))<1200: raise ValueError("insufficient group-isolated split")
    model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",RobustScaler()),("model",HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=24,min_samples_leaf=35,l2_regularization=1.5,random_state=42))]).fit(train[FEATURES],train.planet_like)
    cal_raw=model.predict_proba(cal[FEATURES])[:,1]; calibrator=IsotonicRegression(out_of_bounds="clip").fit(cal_raw,cal.planet_like); rank=model.predict_proba(test[FEATURES])[:,1]; prob=calibrator.predict(rank); y=test.planet_like.to_numpy(); base=np.repeat(train.planet_like.mean(),len(test)); budget=max(1,int(np.ceil(.1*len(test)))); selected=np.argsort(rank)[-budget:]
    evaluation=test[["event_id","kepid","kepoi_name","koi_disposition","planet_like","missing_features","koi_period","koi_prad","koi_model_snr","ra","dec"]].copy(); evaluation["ranking_score"]=rank; evaluation["probability"]=prob; evaluation["route"]=np.select([prob<.22,prob<.48],["reject-review","uncertain-review"],default="planet-like")
    metrics={"average_precision":average_precision_score(y,rank),"baseline_average_precision":average_precision_score(y,base),"roc_auc":roc_auc_score(y,rank),"brier":brier_score_loss(y,prob),"recall_at_10pct":float(y[selected].sum()/y.sum()),"event_rate":float(y.mean()),"test_rows":len(test),"uncertain_rows":int((evaluation.route=="uncertain-review").sum())}
    if metrics["average_precision"]<=metrics["baseline_average_precision"]+.05: raise RuntimeError("candidate failed promotion baseline")
    rng=np.random.default_rng(42); base_ap=metrics["average_precision"]; scores=[]
    for c in FEATURES:
        shuffled=test[FEATURES].copy(); shuffled[c]=rng.permutation(shuffled[c].to_numpy()); scores.append(max(0,base_ap-average_precision_score(y,model.predict_proba(shuffled)[:,1])))
    importance=pd.DataFrame({"feature":FEATURES,"ap_drop":scores}).sort_values("ap_drop",ascending=False)
    drift=pd.DataFrame({"feature":FEATURES,"psi":[_psi(train[c],test[c]) for c in FEATURES]}); drift["status"]=np.select([drift.psi>=.25,drift.psi>=.1],["high","watch"],default="stable")
    bounds={c:(float(train[c].quantile(.01)),float(train[c].quantile(.99))) for c in FEATURES}
    return ModelProduct(model,calibrator,evaluation,metrics,drift,importance,{"algorithm":"Histogram gradient boosting + isotonic calibration","split":"star-group SHA-256 60/20/20","train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test),"features":FEATURES,"bounds":bounds})
def score_candidate(bundle,row,updates):
    values=row[FEATURES].copy()
    for k,v in updates.items(): values[k]=v
    outside=[c for c in FEATURES if pd.notna(values[c]) and not bundle.metadata["bounds"][c][0]<=float(values[c])<=bundle.metadata["bounds"][c][1]]
    raw=bundle.model.predict_proba(pd.DataFrame([values],columns=FEATURES))[:,1]; p=float(bundle.calibrator.predict(raw)[0]); route="ood-review" if len(outside)>=2 else "reject-review" if p<.22 else "uncertain-review" if p<.48 else "planet-like"; return {"probability":p,"route":route,"outside_features":outside}
