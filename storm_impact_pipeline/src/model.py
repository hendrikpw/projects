"""Time-separated hurdle model, evaluation, uncertainty and serving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score,brier_score_loss,mean_absolute_error,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from storm_impact_pipeline.src.pipeline import FEATURES

CATEGORICAL=["state","event_type","cz_type"]; NUMERIC=[f for f in FEATURES if f not in CATEGORICAL]


@dataclass(frozen=True)
class ModelBundle:
    classifier: Pipeline; regressor: Pipeline; calibrator: IsotonicRegression
    extreme_classifier: Pipeline; extreme_calibrator: IsotonicRegression
    calibration: pd.DataFrame; evaluation: pd.DataFrame; metrics: dict
    importance: pd.DataFrame; drift: pd.DataFrame; metadata: dict


def _preprocessor():
    return ColumnTransformer([("category",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=10),CATEGORICAL),("numeric","passthrough",NUMERIC)],verbose_feature_names_out=False)


def _psi(reference: pd.Series,current: pd.Series) -> float:
    if reference.dtype.name in {"string","object"}:
        cats=sorted(set(reference.dropna()).union(current.dropna())); a=reference.value_counts(normalize=True).reindex(cats,fill_value=0); b=current.value_counts(normalize=True).reindex(cats,fill_value=0)
    else:
        ref=reference.fillna(reference.median()); cur=current.fillna(reference.median()); edges=np.unique(np.quantile(ref,np.linspace(0,1,11))); edges[0],edges[-1]=-np.inf,np.inf; a=pd.cut(ref,edges,include_lowest=True).value_counts(normalize=True,sort=False); b=pd.cut(cur,edges,include_lowest=True).value_counts(normalize=True,sort=False)
    a=np.clip(np.asarray(a,float),1e-6,None); b=np.clip(np.asarray(b,float),1e-6,None); return float(np.sum((b-a)*np.log(b/a)))


def _ranking_metrics(actual,predicted,severe_threshold=50_000,capacity=.10):
    n=max(1,int(len(actual)*capacity)); order=np.argsort(-predicted)[:n]; actual=np.asarray(actual); severe=actual>=severe_threshold
    return {"damage_capture_at_capacity":float(actual[order].sum()/max(actual.sum(),1)),"severe_precision_at_capacity":float(severe[order].mean()),"severe_recall_at_capacity":float(severe[order].sum()/max(severe.sum(),1)),"review_rows":n}


def train_and_evaluate(gold: pd.DataFrame,severe_threshold: float=50_000,extreme_threshold: float=250_000,capacity: float=.10) -> ModelBundle:
    train=gold[gold.month<=6].copy(); cal=gold[gold.month.between(7,9)].copy(); test=gold[gold.month>=10].copy()
    if min(len(train),len(cal),len(test))<500 or min(train.has_damage.sum(),cal.has_damage.sum(),test.has_damage.sum())<20: raise ValueError("insufficient rows or positives in month split")
    classifier=Pipeline([("prep",_preprocessor()),("model",HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=19,learning_rate=.07,l2_regularization=1.5,random_state=42))]); ratio=train.has_damage.eq(0).sum()/train.has_damage.eq(1).sum(); weights=np.where(train.has_damage.eq(1),ratio,1.0); classifier.fit(train[FEATURES],train.has_damage,model__sample_weight=weights)
    raw_cal=classifier.predict_proba(cal[FEATURES])[:,1]; calibrator=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(raw_cal,cal.has_damage); cal_prob=calibrator.predict(raw_cal)
    positive=train[train.has_damage.eq(1)]; regressor=Pipeline([("prep",_preprocessor()),("model",HistGradientBoostingRegressor(loss="absolute_error",max_iter=150,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1.5,random_state=42))]); regressor.fit(positive[FEATURES],positive.log_damage)
    extreme_train=train.total_damage_usd.ge(extreme_threshold).astype(int); extreme_cal=cal.total_damage_usd.ge(extreme_threshold).astype(int); extreme_ratio=extreme_train.eq(0).sum()/extreme_train.eq(1).sum(); extreme_classifier=Pipeline([("prep",_preprocessor()),("model",HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=19,learning_rate=.07,l2_regularization=2.0,random_state=42))]); extreme_classifier.fit(train[FEATURES],extreme_train,model__sample_weight=np.where(extreme_train.eq(1),extreme_ratio,1.0)); extreme_raw_cal=extreme_classifier.predict_proba(cal[FEATURES])[:,1]; extreme_calibrator=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(extreme_raw_cal,extreme_cal)
    cal_conditional=np.maximum(0,np.expm1(regressor.predict(cal[FEATURES]))); residual=np.abs(cal.loc[cal.has_damage.eq(1),"log_damage"]-regressor.predict(cal.loc[cal.has_damage.eq(1),FEATURES])); q90=float(np.quantile(residual,.90,method="higher"))
    raw=classifier.predict_proba(test[FEATURES])[:,1]; probability=calibrator.predict(raw); conditional=np.maximum(0,np.expm1(regressor.predict(test[FEATURES]))); expected=probability*conditional; extreme_probability=extreme_calibrator.predict(extreme_classifier.predict_proba(test[FEATURES])[:,1])
    stats=train.groupby("event_type").agg(p=("has_damage","mean"),amount=("total_damage_usd",lambda x: x[x>0].median())).fillna(0); global_p=train.has_damage.mean(); global_amount=train.loc[train.has_damage.eq(1),"total_damage_usd"].median(); base_p=test.event_type.map(stats.p).fillna(global_p); base_amount=test.event_type.map(stats.amount).fillna(global_amount); baseline=np.asarray(base_p*base_amount)
    evaluation=test[["event_id","begin_at",*FEATURES,"total_damage_usd","has_damage"]].copy(); evaluation["damage_probability"]=probability; evaluation["extreme_probability"]=extreme_probability; evaluation["conditional_damage_usd"]=conditional; evaluation["expected_damage_usd"]=expected; evaluation["baseline_expected_usd"]=baseline; evaluation["conditional_lower_usd"]=np.maximum(0,np.expm1(np.log1p(conditional)-q90)); evaluation["conditional_upper_usd"]=np.expm1(np.log1p(conditional)+q90)
    rank=_ranking_metrics(test.total_damage_usd,extreme_probability,severe_threshold,capacity); base_rank=_ranking_metrics(test.total_damage_usd,baseline,severe_threshold,capacity); extreme_test=test.total_damage_usd.ge(extreme_threshold)
    metrics={"average_precision":average_precision_score(test.has_damage,probability),"extreme_average_precision":average_precision_score(extreme_test,extreme_probability),"roc_auc":roc_auc_score(test.has_damage,probability),"brier":brier_score_loss(test.has_damage,probability),"damage_mae":mean_absolute_error(test.total_damage_usd,expected),"baseline_damage_mae":mean_absolute_error(test.total_damage_usd,baseline),"wape":float(np.abs(test.total_damage_usd-expected).sum()/max(test.total_damage_usd.sum(),1)),"baseline_wape":float(np.abs(test.total_damage_usd-baseline).sum()/max(test.total_damage_usd.sum(),1)),"prevalence":float(test.has_damage.mean()),"extreme_prevalence":float(extreme_test.mean()),"q90_log_residual":q90,**rank,"baseline_damage_capture":base_rank["damage_capture_at_capacity"]}
    perm=permutation_importance(extreme_classifier,test[FEATURES],extreme_test,scoring="average_precision",n_repeats=4,random_state=42); importance=pd.DataFrame({"feature":FEATURES,"importance":perm.importances_mean,"std":perm.importances_std}).sort_values("importance",ascending=False)
    drift=pd.DataFrame({"feature":FEATURES,"psi":[_psi(train[f],test[f]) for f in FEATURES]}); drift["status"]=np.select([drift.psi>=.25,drift.psi>=.1],["high","watch"],default="stable")
    calibration=cal[["event_id","has_damage","total_damage_usd"]].copy(); calibration["raw_probability"]=raw_cal; calibration["damage_probability"]=cal_prob; calibration["conditional_damage_usd"]=cal_conditional
    return ModelBundle(classifier,regressor,calibrator,extreme_classifier,extreme_calibrator,calibration,evaluation,metrics,importance,drift,{"seed":42,"train_months":"Jan–Jun","calibration_months":"Jul–Sep","test_months":"Oct–Dec","train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test),"positive_weight":ratio,"extreme_positive_weight":extreme_ratio,"severe_threshold_usd":severe_threshold,"extreme_threshold_usd":extreme_threshold,"capacity":capacity})


def score_case(bundle: ModelBundle,values: dict) -> dict:
    frame=pd.DataFrame([{f:values[f] for f in FEATURES}]); p=float(bundle.calibrator.predict(bundle.classifier.predict_proba(frame)[:,1])[0]); extreme=float(bundle.extreme_calibrator.predict(bundle.extreme_classifier.predict_proba(frame)[:,1])[0]); conditional=float(max(0,np.expm1(bundle.regressor.predict(frame)[0]))); return {"damage_probability":p,"extreme_probability":extreme,"conditional_damage_usd":conditional,"expected_damage_usd":p*conditional}
