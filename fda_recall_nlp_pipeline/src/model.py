"""Evaluated multi-class NLP model with selective prediction and drift checks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from fda_recall_nlp_pipeline.src.data import CLASSES


@dataclass(frozen=True)
class ModelBundle:
    estimator: Pipeline
    metrics: dict[str, float]
    holdout: pd.DataFrame
    confusion: pd.DataFrame
    per_class: pd.DataFrame
    selective: pd.DataFrame
    top_terms: pd.DataFrame
    drift: pd.DataFrame
    metadata: dict[str, Any]


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ordered = frame.sort_values(["report_date", "record_id"]).reset_index(drop=True)
    cut = max(1, int(len(ordered) * 0.75))
    train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    if train["classification"].nunique() == 3 and test["classification"].nunique() == 3:
        return train, test, "newest 25% holdout"
    train, test = train_test_split(
        ordered, test_size=0.25, random_state=42, stratify=ordered["classification"]
    )
    return train.sort_index(), test.sort_index(), "deterministic stratified fallback"


def _estimator() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
            max_df=0.98, max_features=12_000, sublinear_tf=True, stop_words="english",
        )),
        ("char", TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=10_000, sublinear_tf=True,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1_500, class_weight="balanced", random_state=42)),
    ])


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def selective_report(actual: np.ndarray, predicted: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in (0.0, 0.45, 0.55, 0.65, 0.75):
        accepted = confidence >= threshold
        rows.append({
            "threshold": threshold,
            "coverage": float(accepted.mean()),
            "accepted_rows": int(accepted.sum()),
            "selective_accuracy": float((predicted[accepted] == actual[accepted]).mean()) if accepted.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _top_terms(estimator: Pipeline) -> pd.DataFrame:
    names = estimator.named_steps["features"].get_feature_names_out()
    classifier = estimator.named_steps["classifier"]
    rows = []
    for class_name, weights in zip(classifier.classes_, classifier.coef_):
        for index in np.argsort(weights)[-10:][::-1]:
            rows.append({
                "classification": class_name,
                "term": str(names[index]).replace("word__", "").replace("char__", "char: "),
                "weight": float(weights[index]),
            })
    return pd.DataFrame(rows)


def _drift_report(train: pd.DataFrame, test: pd.DataFrame, estimator: Pipeline) -> pd.DataFrame:
    word_vectorizer = dict(estimator.named_steps["features"].transformer_list)["word"]
    vocabulary = set(word_vectorizer.vocabulary_)
    unigram_vocabulary = {token for token in vocabulary if " " not in token}
    test_tokens = [token for text in test["document_text"] for token in re.findall(r"(?u)\b\w\w+\b", text.lower())]
    oov = np.mean([token not in unigram_vocabulary for token in test_tokens]) if test_tokens else 1.0
    train_domains = train["domain"].value_counts(normalize=True)
    test_domains = test["domain"].value_counts(normalize=True)
    domain_tvd = 0.5 * sum(abs(train_domains.get(key, 0) - test_domains.get(key, 0)) for key in set(train_domains) | set(test_domains))
    train_labels = train["classification"].value_counts(normalize=True)
    test_labels = test["classification"].value_counts(normalize=True)
    label_tvd = 0.5 * sum(abs(train_labels.get(key, 0) - test_labels.get(key, 0)) for key in set(train_labels) | set(test_labels))
    length_shift = test["document_text"].str.len().median() / max(train["document_text"].str.len().median(), 1) - 1
    return pd.DataFrame([
        {"signal": "Holdout word OOV", "value": float(oov), "warning_threshold": 0.35, "direction": "above"},
        {"signal": "Domain mix TVD", "value": float(domain_tvd), "warning_threshold": 0.20, "direction": "above"},
        {"signal": "Label mix TVD", "value": float(label_tvd), "warning_threshold": 0.20, "direction": "above"},
        {"signal": "Median text-length shift", "value": float(abs(length_shift)), "warning_threshold": 0.30, "direction": "above"},
    ]).assign(status=lambda frame: np.where(frame["value"] > frame["warning_threshold"], "Watch", "Healthy"))


def train_and_evaluate(features: pd.DataFrame) -> ModelBundle:
    required = {"record_id", "domain", "report_date", "classification", "document_text"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Model input missing columns: {sorted(missing)}")
    frame = features.dropna(subset=list(required)).copy()
    if len(frame) < 90 or set(frame["classification"]) != set(CLASSES):
        raise ValueError("At least 90 records across all three classes are required")
    train, test, split_strategy = _split(frame)
    estimator = _estimator()
    estimator.fit(train["document_text"], train["classification"])
    probabilities = estimator.predict_proba(test["document_text"])
    classes = estimator.named_steps["classifier"].classes_
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    actual = test["classification"].to_numpy()
    correct = predicted == actual
    precision, recall, class_f1, support = precision_recall_fscore_support(actual, predicted, labels=CLASSES, zero_division=0)
    per_class = pd.DataFrame({
        "classification": CLASSES, "precision": precision, "recall": recall,
        "f1": class_f1, "support": support,
    })
    confusion = pd.DataFrame(confusion_matrix(actual, predicted, labels=CLASSES), index=CLASSES, columns=CLASSES)
    holdout = test[["record_id", "domain", "report_date", "classification", "document_text", "record_url"]].copy()
    holdout["prediction"] = predicted
    holdout["confidence"] = confidence
    holdout["correct"] = correct
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "log_loss": float(log_loss(actual, probabilities, labels=classes)),
        "expected_calibration_error": _ece(confidence, correct),
    }
    metadata = {
        "model": "word + character TF-IDF / balanced multinomial logistic regression",
        "training_rows": len(train),
        "holdout_rows": len(test),
        "split_strategy": split_strategy,
        "classes": list(classes),
        "random_state": 42,
        "feature_count": len(estimator.named_steps["features"].get_feature_names_out()),
    }
    return ModelBundle(
        estimator=estimator,
        metrics=metrics,
        holdout=holdout.reset_index(drop=True),
        confusion=confusion,
        per_class=per_class,
        selective=selective_report(actual, predicted, confidence),
        top_terms=_top_terms(estimator),
        drift=_drift_report(train, test, estimator),
        metadata=metadata,
    )


def score_text(model: ModelBundle, domain: str, product: str, reason: str, firm: str, threshold: float = 0.55) -> dict[str, Any]:
    text = " ".join(f"domain {domain}. product {product}. reason {reason}. firm {firm}".split())
    if len(product.strip()) < 4 or len(reason.strip()) < 12:
        raise ValueError("Product and a sufficiently detailed recall reason are required")
    probabilities = model.estimator.predict_proba([text])[0]
    classes = model.estimator.named_steps["classifier"].classes_
    position = int(np.argmax(probabilities))
    confidence = float(probabilities[position])
    return {
        "prediction": str(classes[position]),
        "confidence": confidence,
        "abstained": confidence < threshold,
        "threshold": float(threshold),
        "probabilities": {str(label): float(value) for label, value in zip(classes, probabilities)},
    }
