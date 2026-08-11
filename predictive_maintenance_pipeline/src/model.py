"""Leakage-safe, calibrated and cost-sensitive failure decision system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predictive_maintenance_pipeline.src.pipeline import FEATURES, TARGET


@dataclass(frozen=True)
class ModelBundle:
    model: Pipeline
    calibrator: IsotonicRegression
    calibration: pd.DataFrame
    evaluation: pd.DataFrame
    metrics: dict
    importance: pd.DataFrame
    drift: pd.DataFrame
    metadata: dict


def decision_table(labels: np.ndarray, probabilities: np.ndarray, fn_cost: float = 25, fp_cost: float = 1) -> pd.DataFrame:
    rows=[]
    for threshold in np.linspace(.01,.99,99):
        pred=probabilities>=threshold; tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
        rows.append((threshold,tn,fp,fn,tp,fn*fn_cost+fp*fp_cost))
    return pd.DataFrame(rows,columns=["threshold","tn","fp","fn","tp","cost"])


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    if reference.dtype.name in {"string","object"}:
        cats=sorted(set(reference.dropna()).union(current.dropna())); a=reference.value_counts(normalize=True).reindex(cats,fill_value=0); b=current.value_counts(normalize=True).reindex(cats,fill_value=0)
    else:
        edges=np.unique(np.quantile(reference,np.linspace(0,1,bins+1))); edges[0],edges[-1]=-np.inf,np.inf
        a=pd.cut(reference,edges,include_lowest=True).value_counts(normalize=True,sort=False); b=pd.cut(current,edges,include_lowest=True).value_counts(normalize=True,sort=False)
    a=np.clip(np.asarray(a,float),1e-6,None); b=np.clip(np.asarray(b,float),1e-6,None)
    return float(np.sum((b-a)*np.log(b/a)))


def train_and_evaluate(gold: pd.DataFrame, fn_cost: float = 25, fp_cost: float = 1) -> ModelBundle:
    ordered=gold.sort_values("udi").reset_index(drop=True); n=len(ordered); i,j=int(n*.6),int(n*.8)
    train,cal,test=ordered.iloc[:i],ordered.iloc[i:j],ordered.iloc[j:]
    if min(train[TARGET].sum(),cal[TARGET].sum(),test[TARGET].sum()) < 2: raise ValueError("insufficient failures in ordered split")
    prep=ColumnTransformer([("type",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["type"]),
        ("numeric","passthrough",[f for f in FEATURES if f!="type"])],verbose_feature_names_out=False)
    ratio=(train[TARGET].eq(0).sum()/train[TARGET].eq(1).sum())
    model=Pipeline([("prep",prep),("classifier",HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1.0,random_state=42))])
    weights=np.where(train[TARGET].eq(1),ratio,1.0); model.fit(train[FEATURES],train[TARGET],classifier__sample_weight=weights)
    raw_cal=model.predict_proba(cal[FEATURES])[:,1]
    calibrator=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(raw_cal,cal[TARGET])
    cal_prob=calibrator.predict(raw_cal); decisions=decision_table(cal[TARGET].to_numpy(),cal_prob,fn_cost,fp_cost)
    threshold=float(decisions.loc[decisions["cost"].idxmin(),"threshold"])
    raw_test=model.predict_proba(test[FEATURES])[:,1]; probability=calibrator.predict(raw_test); pred=probability>=threshold
    tn,fp,fn,tp=confusion_matrix(test[TARGET],pred,labels=[0,1]).ravel()
    metrics={"average_precision":average_precision_score(test[TARGET],probability),"roc_auc":roc_auc_score(test[TARGET],probability),
        "brier":brier_score_loss(test[TARGET],probability),"precision":precision_score(test[TARGET],pred,zero_division=0),
        "recall":recall_score(test[TARGET],pred,zero_division=0),"f1":f1_score(test[TARGET],pred,zero_division=0),
        "balanced_accuracy":balanced_accuracy_score(test[TARGET],pred),"threshold":threshold,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "prevalence":float(test[TARGET].mean()),"baseline_ap":float(test[TARGET].mean()),"expected_cost":float(fn*fn_cost+fp*fp_cost)}
    evaluation=test[["udi",TARGET,*FEATURES]].copy(); evaluation["raw_probability"]=raw_test; evaluation["failure_probability"]=probability; evaluation["alert"]=pred
    perm=permutation_importance(model,test[FEATURES],test[TARGET],scoring="average_precision",n_repeats=5,random_state=42)
    importance=pd.DataFrame({"feature":FEATURES,"importance":perm.importances_mean,"std":perm.importances_std}).sort_values("importance",ascending=False)
    drift=pd.DataFrame({"feature":FEATURES,"psi":[_psi(train[f],test[f]) for f in FEATURES]}); drift["status"]=np.select([drift.psi>=.25,drift.psi>=.1],["high","watch"],default="stable")
    calibration=pd.DataFrame({"label":cal[TARGET].to_numpy(),"raw_probability":raw_cal,"probability":cal_prob})
    return ModelBundle(model,calibrator,calibration,evaluation,metrics,importance,drift,{"seed":42,"train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test),"positive_weight":ratio,"fn_cost":fn_cost,"fp_cost":fp_cost})


def score_case(bundle: ModelBundle, values: dict) -> float:
    frame=pd.DataFrame([{feature:values[feature] for feature in FEATURES}])
    return float(bundle.calibrator.predict(bundle.model.predict_proba(frame)[:,1])[0])
