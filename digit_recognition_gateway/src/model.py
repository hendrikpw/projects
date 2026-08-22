"""Calibrated digit classifier, corruption evaluation and inference guardrails."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score, top_k_accuracy_score
from sklearn.svm import SVC

from digit_recognition_gateway.src.data import PIXELS


@dataclass(frozen=True)
class DigitModel:
    estimator: SVC
    temperature: float
    threshold: float
    evaluation: pd.DataFrame
    metrics: dict
    class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    robustness: pd.DataFrame
    importance: pd.DataFrame
    metadata: dict
    ink_bounds: tuple[float, float]


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / temperature; scaled -= scaled.max(axis=1, keepdims=True); exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1); predicted = probabilities.argmax(axis=1); correct = predicted == y; result = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any(): result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def _threshold(y: np.ndarray, probabilities: np.ndarray, target: float = .985) -> float:
    confidence = probabilities.max(axis=1); predicted = probabilities.argmax(axis=1); best, coverage = .99, -1.0
    for value in np.arange(.80, 1, .01):
        accepted = confidence >= value
        if accepted.any() and (predicted[accepted] == y[accepted]).mean() >= target and accepted.mean() >= coverage: best, coverage = float(value), float(accepted.mean())
    return best


def corrupt(images: np.ndarray, severity: float, seed: int = 42) -> np.ndarray:
    if severity <= 0: return images.copy()
    rng = np.random.default_rng(seed); noisy = np.clip(images + rng.normal(0, severity * .75, images.shape), 0, 1)
    dropout = rng.random(images.shape) < severity * .45; noisy[dropout] = 0
    return noisy


def _evaluate_probabilities(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities.argmax(axis=1); confidence = probabilities.max(axis=1); accepted = confidence >= threshold
    return {"accuracy":float(accuracy_score(y,predicted)),"macro_f1":float(f1_score(y,predicted,average="macro")),"coverage":float(accepted.mean()),"selective_accuracy":float((predicted[accepted]==y[accepted]).mean()) if accepted.any() else 0.0}


def train_and_evaluate(gold: pd.DataFrame) -> DigitModel:
    train_source = gold[gold.source_split == "train"].sort_values(["label", "source_row"]).copy(); test = gold[gold.source_split == "test"].copy()
    calibration_mask = train_source.groupby("label").cumcount() % 5 == 0; calibration = train_source[calibration_mask]; train = train_source[~calibration_mask]
    x_train, y_train = train[PIXELS].to_numpy(float), train.label.to_numpy(int); x_cal, y_cal = calibration[PIXELS].to_numpy(float), calibration.label.to_numpy(int); x_test, y_test = test[PIXELS].to_numpy(float), test.label.to_numpy(int)
    if min(pd.Series(y_train).value_counts().min(), pd.Series(y_cal).value_counts().min(), pd.Series(y_test).value_counts().min()) < 50: raise RuntimeError("train/calibration/test class coverage is incomplete")
    estimator = SVC(C=5.0, gamma="scale", kernel="rbf", decision_function_shape="ovr", random_state=42)
    started = time.perf_counter(); estimator.fit(x_train, y_train); fit_seconds = time.perf_counter() - started
    cal_scores = estimator.decision_function(x_cal); candidates = np.arange(.20, 3.01, .05); losses = [log_loss(y_cal, _softmax(cal_scores, value), labels=estimator.classes_) for value in candidates]; temperature = float(candidates[int(np.argmin(losses))])
    cal_probability = _softmax(cal_scores, temperature); threshold = _threshold(y_cal, cal_probability)
    started = time.perf_counter(); test_scores = estimator.decision_function(x_test); inference_ms = (time.perf_counter() - started) * 1000 / len(test)
    probability = _softmax(test_scores, temperature); predicted = estimator.classes_[probability.argmax(axis=1)]; confidence = probability.max(axis=1); accepted = confidence >= threshold
    majority = int(pd.Series(y_train).mode().iloc[0]); baseline = np.repeat(majority, len(y_test))
    corrupted = corrupt(x_test, .15); corrupted_probability = _softmax(estimator.decision_function(corrupted), temperature); corrupted_metrics = _evaluate_probabilities(y_test, corrupted_probability, threshold)
    metrics = {"test_images":len(test),"accuracy":float(accuracy_score(y_test,predicted)),"macro_f1":float(f1_score(y_test,predicted,average="macro")),"baseline_macro_f1":float(f1_score(y_test,baseline,average="macro")),"top3_accuracy":float(top_k_accuracy_score(y_test,probability,k=3,labels=estimator.classes_)),"log_loss":float(log_loss(y_test,probability,labels=estimator.classes_)),"ece":_ece(y_test,probability),"coverage":float(accepted.mean()),"selective_accuracy":float((predicted[accepted]==y_test[accepted]).mean()) if accepted.any() else 0.0,"review_rate":float((~accepted).mean()),"corrupted_accuracy":corrupted_metrics["accuracy"],"corrupted_macro_f1":corrupted_metrics["macro_f1"],"inference_ms_per_image":float(inference_ms)}
    if metrics["macro_f1"] < .90 or metrics["macro_f1"] <= metrics["baseline_macro_f1"] + .70 or metrics["corrupted_macro_f1"] < .70: raise RuntimeError(f"digit model failed promotion gates: {metrics}")
    evaluation = test[["sample_id","image_hash","label"]].copy(); evaluation["predicted"] = predicted; evaluation["confidence"] = confidence; evaluation["route"] = np.where(accepted,"auto-read","human-review"); evaluation["correct"] = predicted == y_test
    class_rows = []
    for label in estimator.classes_:
        mask = y_test == label; class_rows.append({"digit":int(label),"images":int(mask.sum()),"recall":float(recall_score(y_test==label,predicted==label)),"mean_confidence":float(confidence[mask].mean()),"review_rate":float((~accepted[mask]).mean())})
    matrix = confusion_matrix(y_test,predicted,labels=estimator.classes_,normalize="true"); confusion = pd.DataFrame(matrix,index=estimator.classes_,columns=estimator.classes_).rename_axis("actual").reset_index().melt("actual",var_name="predicted",value_name="rate")
    robust_rows = []
    for severity in [0,.05,.10,.15,.20,.25]: robust_rows.append({"severity":severity,**_evaluate_probabilities(y_test,_softmax(estimator.decision_function(corrupt(x_test,severity)),temperature),threshold)})
    sample_idx = np.linspace(0,len(test)-1,min(900,len(test)),dtype=int); raw_importance = permutation_importance(estimator,x_test[sample_idx],y_test[sample_idx],scoring="accuracy",n_repeats=2,random_state=42,n_jobs=-1)
    importance = pd.DataFrame({"pixel":PIXELS,"importance":raw_importance.importances_mean,"row":np.repeat(range(8),8),"column":np.tile(range(8),8)})
    ink = x_train.mean(axis=1); metadata = {"train_images":len(train),"calibration_images":len(calibration),"test_images":len(test),"train_people":30,"test_people":13,"temperature":temperature,"confidence_threshold":threshold,"fit_seconds":round(fit_seconds,2)}
    return DigitModel(estimator,temperature,threshold,evaluation,metrics,pd.DataFrame(class_rows),confusion,pd.DataFrame(robust_rows),importance,metadata,(float(np.quantile(ink,.001)),float(np.quantile(ink,.999))))


def score_image(model: DigitModel, row: pd.Series, noise: float = 0.0, dropout: float = 0.0) -> dict:
    rng = np.random.default_rng(42); original = row[PIXELS].to_numpy(float); image = np.clip(original + rng.normal(0,noise,len(original)),0,1); mask = rng.random(len(image)) < dropout; image[mask] = 0
    probability = _softmax(model.estimator.decision_function(image.reshape(1,-1)),model.temperature)[0]; order = np.argsort(-probability); confidence = float(probability[order[0]]); ink = float(image.mean()); ink_ood = not (model.ink_bounds[0] <= ink <= model.ink_bounds[1])
    route = "input-withheld" if dropout > .20 or ink_ood else "auto-read" if confidence >= model.threshold else "human-review"
    variants = np.repeat(image.reshape(1,-1),64,axis=0); variants[np.arange(64),np.arange(64)] = 0; occluded = _softmax(model.estimator.decision_function(variants),model.temperature)[:,order[0]]; sensitivity = np.maximum(0,confidence-occluded).reshape(8,8)
    ranking = pd.DataFrame({"digit":model.estimator.classes_[order].astype(int),"probability":probability[order]})
    return {"prediction":int(model.estimator.classes_[order[0]]),"confidence":confidence,"route":route,"ink_density":ink,"ink_ood":ink_ood,"dropout_share":float(mask.mean()),"image":image.reshape(8,8),"sensitivity":sensitivity,"ranking":ranking}
