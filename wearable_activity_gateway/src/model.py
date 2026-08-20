"""Subject-isolated activity model, calibration, drift and serving guardrails."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, top_k_accuracy_score


@dataclass(frozen=True)
class ActivityModel:
    estimator: ExtraTreesClassifier
    features: list[str]
    classes: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    temperature: float
    confidence_threshold: float
    evaluation: pd.DataFrame
    metrics: dict
    class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    drift: pd.DataFrame
    importance: pd.DataFrame
    metadata: dict


def _partition(gold: pd.DataFrame) -> pd.Series:
    calibration_subject = (gold.source_split == "train") & (gold.subject_id.astype(int) % 5 == 0)
    return pd.Series(np.select([gold.source_split == "test", calibration_subject], ["test", "calibration"], default="train"), index=gold.index)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1)
    adjusted = np.exp(np.log(clipped) / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _ece(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 10) -> float:
    predicted = classes[probabilities.argmax(axis=1)]; confidence = probabilities.max(axis=1); correct = predicted == y
    result = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any(): result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def _confidence_threshold(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, target_accuracy: float = .95) -> float:
    predicted = classes[probabilities.argmax(axis=1)]; confidence = probabilities.max(axis=1); best = .99; best_coverage = -1.0
    for threshold in np.arange(.40, 1.0, .01):
        accepted = confidence >= threshold
        if accepted.any() and (predicted[accepted] == y[accepted]).mean() >= target_accuracy and accepted.mean() > best_coverage:
            best, best_coverage = float(threshold), float(accepted.mean())
    return best


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0] / len(reference); b = np.histogram(current, bins=edges)[0] / len(current)
    a, b = np.clip(a, 1e-5, None), np.clip(b, 1e-5, None)
    return float(np.sum((b - a) * np.log(b / a)))


def train_and_evaluate(gold: pd.DataFrame, features: list[str], trees: int = 220) -> ActivityModel:
    data = gold.copy(); data["model_split"] = _partition(data)
    train = data[data.model_split == "train"]; calibration = data[data.model_split == "calibration"]; test = data[data.model_split == "test"]
    if min(train.subject_id.nunique(), calibration.subject_id.nunique(), test.subject_id.nunique()) < 3:
        raise RuntimeError("subject-isolated train/calibration/test partitions are incomplete")
    medians = train[features].median().to_numpy(float); means = train[features].mean().to_numpy(float); stds = train[features].std().replace(0, 1).to_numpy(float)
    def matrix(frame): return np.where(np.isfinite(frame[features].to_numpy(float)), frame[features].to_numpy(float), medians)
    x_train, x_cal, x_test = matrix(train), matrix(calibration), matrix(test)
    y_train, y_cal, y_test = train.activity.to_numpy(), calibration.activity.to_numpy(), test.activity.to_numpy()
    model = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    started = time.perf_counter(); model.fit(x_train, y_train); fit_ms = (time.perf_counter() - started) * 1000
    raw_cal = model.predict_proba(x_cal)
    candidates = np.arange(.55, 2.51, .05)
    losses = [log_loss(y_cal, _temperature_scale(raw_cal, value), labels=model.classes_) for value in candidates]
    temperature = float(candidates[int(np.argmin(losses))])
    cal_prob = _temperature_scale(raw_cal, temperature)
    threshold = _confidence_threshold(y_cal, cal_prob, model.classes_)
    started = time.perf_counter(); raw_test = model.predict_proba(x_test); inference_ms = (time.perf_counter() - started) * 1000 / len(test)
    test_prob = _temperature_scale(raw_test, temperature); predicted = model.classes_[test_prob.argmax(axis=1)]; confidence = test_prob.max(axis=1)
    accepted = confidence >= threshold
    majority = train.activity.mode().iloc[0]; baseline = np.repeat(majority, len(test))
    metrics = {
        "test_windows": int(len(test)), "accuracy": float(accuracy_score(y_test, predicted)), "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")), "baseline_macro_f1": float(f1_score(y_test, baseline, average="macro")),
        "top2_accuracy": float(top_k_accuracy_score(y_test, test_prob, k=2, labels=model.classes_)), "log_loss": float(log_loss(y_test, test_prob, labels=model.classes_)),
        "ece": _ece(y_test, test_prob, model.classes_), "coverage": float(accepted.mean()),
        "selective_accuracy": float((predicted[accepted] == y_test[accepted]).mean()) if accepted.any() else 0.0,
        "review_rate": float((~accepted).mean()), "inference_ms_per_window": float(inference_ms),
    }
    if metrics["macro_f1"] <= metrics["baseline_macro_f1"] + .20 or metrics["balanced_accuracy"] < .75 or metrics["top2_accuracy"] < .90:
        raise RuntimeError(f"activity model failed promotion gates: {metrics}")
    evaluation = test[["window_id", "subject_id", "activity"]].copy()
    evaluation["predicted"] = predicted; evaluation["confidence"] = confidence; evaluation["route"] = np.where(accepted, "auto-inference", "human-review"); evaluation["correct"] = predicted == y_test
    for index, label in enumerate(model.classes_): evaluation[f"p_{label.lower()}"] = test_prob[:, index]
    class_rows = []
    for label in model.classes_:
        mask = y_test == label; class_rows.append({"activity": label, "windows": int(mask.sum()), "recall": float((predicted[mask] == label).mean()), "mean_confidence": float(confidence[mask].mean()), "review_rate": float((~accepted[mask]).mean())})
    matrix_values = confusion_matrix(y_test, predicted, labels=model.classes_, normalize="true")
    confusion = pd.DataFrame(matrix_values, index=model.classes_, columns=model.classes_).rename_axis("actual").reset_index().melt("actual", var_name="predicted", value_name="rate")
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)
    drift_rows = []
    for feature in importance.feature.head(15):
        idx = features.index(feature); psi = _psi(x_train[:, idx], x_test[:, idx]); drift_rows.append({"feature": feature, "psi": psi, "status": "high" if psi >= .25 else "watch" if psi >= .10 else "stable"})
    metadata = {"train_windows": len(train), "calibration_windows": len(calibration), "test_windows": len(test), "train_subjects": train.subject_id.nunique(), "calibration_subjects": calibration.subject_id.nunique(), "test_subjects": test.subject_id.nunique(), "fit_ms": round(fit_ms, 1), "temperature": temperature, "confidence_threshold": threshold}
    return ActivityModel(model, features, model.classes_, medians, means, stds, temperature, threshold, evaluation, metrics, pd.DataFrame(class_rows), confusion, pd.DataFrame(drift_rows), importance, metadata)


def score_window(model: ActivityModel, row: pd.Series, *, scale: float = 1.0, noise: float = 0.0, missing_share: float = 0.0) -> dict:
    rng = np.random.default_rng(42)
    values = row[model.features].to_numpy(float) * scale + rng.normal(0, noise, len(model.features))
    missing = np.zeros(len(values), dtype=bool); missing[: int(round(len(values) * missing_share))] = True
    values[missing] = np.nan
    finite = np.where(np.isfinite(values), values, model.medians)
    z = np.abs((finite - model.means) / model.stds)
    ood_share = float((z > 4).mean()); ood = missing_share > .05 or ood_share > .02 or float(z.max()) > 8
    probability = _temperature_scale(model.estimator.predict_proba(finite.reshape(1, -1)), model.temperature)[0]
    order = np.argsort(-probability)
    confidence = float(probability[order[0]])
    route = "sensor-fault-review" if ood else "auto-inference" if confidence >= model.confidence_threshold else "human-review"
    ranking = pd.DataFrame({"activity": model.classes[order], "probability": probability[order]})
    return {"prediction": model.classes[order[0]], "confidence": confidence, "route": route, "missing_share": float(missing_share), "ood_share": ood_share, "max_z": float(z.max()), "ranking": ranking}
