"""Time-split calibrated departure-time delay classifier."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from sklearn.preprocessing import OrdinalEncoder
from flight_delay_control.src.pipeline import FEATURES

CAT=["Reporting_Airline","Origin","Dest"]; NUM=[x for x in FEATURES if x not in CAT]
@dataclass(frozen=True)
class ModelProduct:
    transformer:ColumnTransformer; model:HistGradientBoostingClassifier; calibrator:IsotonicRegression; evaluation:pd.DataFrame; metrics:dict; carrier_metrics:pd.DataFrame; drift:pd.DataFrame; metadata:dict; category_maps:dict

def _psi(a,b):
    e=np.unique(np.quantile(a,np.linspace(0,1,11))); e[0],e[-1]=-np.inf,np.inf
    x=np.clip(pd.cut(a,e).value_counts(normalize=True,sort=False).to_numpy(),1e-6,None); y=np.clip(pd.cut(b,e).value_counts(normalize=True,sort=False).to_numpy(),1e-6,None)
    return float(np.sum((y-x)*np.log(y/x)))

def _collapse(frame, category_maps):
    x=frame.copy()
    for column, known in category_maps.items():
        x[column]=x[column].where(x[column].isin(known),"__OTHER__")
    return x

def train_and_evaluate(gold):
    train=gold[gold.FlightDate.dt.day<=18].copy(); cal=gold[gold.FlightDate.dt.day.between(19,24)].copy(); test=gold[gold.FlightDate.dt.day>=25].copy()
    if min(len(train),len(cal),len(test))<5000: raise ValueError("insufficient temporal split")
    category_maps={"Reporting_Airline":set(train.Reporting_Airline.value_counts().head(50).index),"Origin":set(train.Origin.value_counts().head(100).index),"Dest":set(train.Dest.value_counts().head(100).index)}
    train_x=_collapse(train[FEATURES],category_maps); cal_x=_collapse(cal[FEATURES],category_maps); test_x=_collapse(test[FEATURES],category_maps)
    transformer=ColumnTransformer([("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),CAT),("num","passthrough",NUM)])
    tx=transformer.fit_transform(train_x); cx=transformer.transform(cal_x); vx=transformer.transform(test_x)
    model=HistGradientBoostingClassifier(max_iter=160,learning_rate=.07,max_leaf_nodes=28,min_samples_leaf=60,l2_regularization=1,random_state=42,categorical_features=[0,1,2]).fit(tx,train.is_delayed_15)
    raw_cal=model.predict_proba(cx)[:,1]; calibrator=IsotonicRegression(out_of_bounds="clip").fit(raw_cal,cal.is_delayed_15)
    ranking=model.predict_proba(vx)[:,1]; probability=calibrator.predict(ranking); baseline=np.repeat(train.is_delayed_15.mean(),len(test)); y=test.is_delayed_15.astype(int).to_numpy(); budget=max(1,int(np.ceil(len(y)*.1))); selected=np.argsort(ranking)[-budget:]
    evaluation=test[["event_id","FlightDate","Reporting_Airline","Origin","Dest","route","scheduled_hour","Distance","is_delayed_15","ArrDelayMinutes"]].copy(); evaluation["ranking_score"]=ranking; evaluation["probability"]=probability; evaluation["status"]=np.select([probability>=.45,probability>=.30],["review","watch"],default="monitor")
    metrics={"average_precision":average_precision_score(y,ranking),"baseline_average_precision":average_precision_score(y,baseline),"roc_auc":roc_auc_score(y,ranking),"brier":brier_score_loss(y,probability),"recall_at_10pct":float(y[selected].sum()/y.sum()),"event_rate":float(y.mean()),"test_rows":len(test),"review_rows":int((probability>=.45).sum())}
    if metrics["average_precision"]<=metrics["baseline_average_precision"]: raise RuntimeError("candidate failed baseline gate")
    carrier=evaluation.groupby("Reporting_Airline").apply(lambda x:pd.Series({"flights":len(x),"delay_rate":x.is_delayed_15.mean(),"average_precision":average_precision_score(x.is_delayed_15,x.ranking_score) if x.is_delayed_15.nunique()>1 else np.nan,"review_rate":(x.status=="review").mean()}),include_groups=False).reset_index()
    drift=pd.DataFrame({"feature":["scheduled_hour","CRSElapsedTime","Distance"],"psi":[_psi(train[c],test[c]) for c in ["scheduled_hour","CRSElapsedTime","Distance"]]}); drift["status"]=np.select([drift.psi>=.25,drift.psi>=.1],["high","watch"],default="stable")
    return ModelProduct(transformer,model,calibrator,evaluation,metrics,carrier,drift,{"algorithm":"HistGradientBoosting + isotonic calibration","train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test),"train_days":"1–18 June 2026","calibration_days":"19–24 June 2026","test_days":"25–30 June 2026","features":FEATURES},category_maps)

def score_scenario(bundle,row,updates):
    values=row[FEATURES].copy()
    for k,v in updates.items(): values[k]=v
    x=bundle.transformer.transform(_collapse(pd.DataFrame([values],columns=FEATURES),bundle.category_maps)); raw=bundle.model.predict_proba(x)[:,1]; p=float(bundle.calibrator.predict(raw)[0])
    return {"probability":p,"status":"review" if p>=.45 else "watch" if p>=.30 else "monitor"}
