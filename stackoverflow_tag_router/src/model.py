"""Time-separated multi-label tag recommendation, evaluation and serving."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from stackoverflow_tag_router.src.pipeline import clean_html,redact_text


@dataclass(frozen=True)
class ModelBundle:
    model:Pipeline; encoder:MultiLabelBinarizer; evaluation:pd.DataFrame; tag_metrics:pd.DataFrame; metrics:dict; drift:pd.DataFrame; metadata:dict


def _split(gold):
    f=gold.sort_values(["created_at","question_id"]).reset_index(drop=True); a=int(len(f)*.70); b=int(len(f)*.85); return f.iloc[:a].copy(),f.iloc[a:b].copy(),f.iloc[b:].copy()


def _vectorizer():
    return FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=18_000,sublinear_tf=True)),("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,max_features=22_000,sublinear_tf=True))])


def _scores(y,p,k=3):
    idx=np.argsort(p,axis=1)[:,-k:]; hit=np.take_along_axis(y,idx,axis=1).sum(axis=1); relevant=y.sum(axis=1); return hit/k,np.divide(hit,relevant,out=np.zeros_like(hit,dtype=float),where=relevant>0)


def _psi(ref,cur):
    edges=np.unique(np.quantile(ref,np.linspace(0,1,8)))
    if len(edges)<2:return 0.0
    edges[0],edges[-1]=-np.inf,np.inf; a=np.clip(np.asarray(pd.cut(ref,edges).value_counts(normalize=True,sort=False)),1e-6,None); b=np.clip(np.asarray(pd.cut(cur,edges).value_counts(normalize=True,sort=False)),1e-6,None); return float(np.sum((b-a)*np.log(b/a)))


def train_and_evaluate(gold,top_n:int=12):
    train,cal,test=_split(gold); counts=train.tags_clean.explode().value_counts(); tags=list(counts.head(top_n).index); tag_set=set(tags); enc=MultiLabelBinarizer(classes=tags); enc.fit([tags]); encode=lambda series:enc.transform(series.map(lambda row:[tag for tag in row if tag in tag_set])); y_train=encode(train.tags_clean); y_cal=encode(cal.tags_clean); y_test=encode(test.tags_clean)
    if min(y_train.sum(axis=0))<12 or min(len(train),len(cal),len(test))<70: raise ValueError("insufficient time-separated tag support")
    model=Pipeline([("vectorizer",_vectorizer()),("classifier",OneVsRestClassifier(LogisticRegression(C=4,class_weight="balanced",max_iter=1200,random_state=42)))]).fit(train.text_redacted,y_train); cal_p=model.predict_proba(cal.text_redacted); p=model.predict_proba(test.text_redacted); prevalence=y_train.mean(axis=0); baseline=np.tile(prevalence,(len(test),1)); cal_precision,_=_scores(y_cal,cal_p); candidates=[]
    for threshold in np.linspace(.2,.8,31):
        accepted=cal_p.max(axis=1)>=threshold
        if accepted.mean()>=.2 and cal_precision[accepted].mean()>=.45: candidates.append(threshold)
    threshold=min(candidates) if candidates else .5; accepted=p.max(axis=1)>=threshold; precision,recall=_scores(y_test,p); base_precision,base_recall=_scores(y_test,baseline); predicted=(p>=.35).astype(int); brier=float(np.mean((p-y_test)**2)); metrics={"precision_at_3":float(precision.mean()),"recall_at_3":float(recall.mean()),"baseline_precision_at_3":float(base_precision.mean()),"baseline_recall_at_3":float(base_recall.mean()),"micro_f1":float(f1_score(y_test,predicted,average="micro",zero_division=0)),"macro_f1":float(f1_score(y_test,predicted,average="macro",zero_division=0)),"brier":brier,"abstain_threshold":float(threshold),"auto_coverage":float(accepted.mean()),"review_rate":float(1-accepted.mean())}
    if metrics["precision_at_3"]<=metrics["baseline_precision_at_3"]: raise RuntimeError("candidate failed popularity promotion gate")
    eval_frame=test[["question_id","created_at","source_link","tags_clean","text_redacted"]].copy(); eval_frame["recommended_tags"]=[[tags[j] for j in np.argsort(row)[-3:][::-1]] for row in p]; eval_frame["precision_at_3"]=precision; eval_frame["recall_at_3"]=recall; eval_frame["max_confidence"]=p.max(axis=1); eval_frame["route"]=np.where(accepted,"auto-suggest","review")
    tag_metrics=pd.DataFrame({"tag":tags,"support":y_test.sum(axis=0),"f1":f1_score(y_test,predicted,average=None,zero_division=0),"train_share":y_train.mean(axis=0),"test_share":y_test.mean(axis=0)}); tag_metrics["share_shift_pp"]=(tag_metrics.test_share-tag_metrics.train_share)*100
    features=["text_length","token_count","code_blocks","question_marks"]; drift=pd.DataFrame({"feature":features,"psi":[_psi(train[c],test[c]) for c in features]}); drift["status"]=np.select([drift.psi>=.25,drift.psi>=.1],["high","watch"],default="stable")
    return ModelBundle(model,enc,eval_frame,tag_metrics,metrics,drift,{"seed":42,"split":"oldest 70% train / next 15% policy / newest 15% test","top_tags":tags,"train_rows":len(train),"calibration_rows":len(cal),"test_rows":len(test)})


def suggest_tags(bundle,title,body):
    safe=redact_text(clean_html(title)+" "+clean_html(body)); p=bundle.model.predict_proba([safe])[0]; order=np.argsort(p)[::-1][:5]; suggestions=pd.DataFrame({"tag":bundle.encoder.classes_[order],"confidence":p[order]}); route="auto-suggest" if float(p.max())>=bundle.metrics["abstain_threshold"] else "review"; top=int(order[0]); vector=bundle.model.named_steps["vectorizer"].transform([safe]); classifier=bundle.model.named_steps["classifier"].estimators_[top]; names=bundle.model.named_steps["vectorizer"].get_feature_names_out(); contribution=vector.multiply(classifier.coef_[0]).toarray()[0]; evidence_order=np.argsort(contribution)[::-1][:8]; evidence=pd.DataFrame({"feature":names[evidence_order],"contribution":contribution[evidence_order]}); evidence=evidence[evidence.contribution.gt(0)]; return {"safe_text":safe,"route":route,"suggestions":suggestions,"evidence":evidence}
