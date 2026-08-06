"""Explainable trial-discontinuation classifier, evaluation and drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clinical_trial_ops_pipeline.src.pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


@dataclass
class ModelBundle:
    estimator: Pipeline
    metrics: dict[str, Any]
    holdout: pd.DataFrame
    calibration: pd.DataFrame
    coefficients: pd.DataFrame
    drift: pd.DataFrame
    metadata: dict[str, Any]


def _split(frame: pd.DataFrame, holdout_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ordered = frame.sort_values(["first_post_date", "nct_id"]).reset_index(drop=True)
    cut = int(np.clip(round(len(ordered) * (1 - holdout_share)), 30, len(ordered) - 20))
    train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    strategy = "temporal"
    if train["discontinued"].nunique() < 2 or test["discontinued"].nunique() < 2:
        positive = ordered[ordered["discontinued"] == 1]
        negative = ordered[ordered["discontinued"] == 0]
        test = pd.concat([positive.iloc[::4], negative.iloc[::4]]).sort_values("nct_id")
        train = ordered.drop(test.index)
        strategy = "deterministic stratified fallback"
    return train.reset_index(drop=True), test.reset_index(drop=True), strategy


def _estimator() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("features", preprocessing),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", C=0.75, random_state=42)),
    ])


def _calibration_table(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": y_true, "score": probabilities})
    frame["bin"] = pd.cut(frame["score"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    result = frame.groupby("bin", observed=False).agg(mean_score=("score", "mean"), observed_rate=("observed", "mean"), records=("observed", "size")).reset_index()
    return result.dropna(subset=["mean_score"]).assign(bin=lambda value: value["bin"].astype(str))


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 8) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    exp = np.histogram(expected, bins=edges)[0] / len(expected)
    act = np.histogram(actual, bins=edges)[0] / len(actual)
    exp, act = np.clip(exp, 1e-5, None), np.clip(act, 1e-5, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def drift_report(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        psi = _psi(train[feature], holdout[feature])
        rows.append({"feature": feature, "drift_score": psi, "level": "High" if psi >= 0.25 else "Watch" if psi >= 0.10 else "Stable"})
    train_rate, holdout_rate = train["discontinued"].mean(), holdout["discontinued"].mean()
    label_shift = float(abs(train_rate - holdout_rate))
    rows.append({"feature": "outcome prevalence", "drift_score": label_shift, "level": "High" if label_shift >= 0.20 else "Watch" if label_shift >= 0.10 else "Stable"})
    return pd.DataFrame(rows).sort_values("drift_score", ascending=False).reset_index(drop=True)


def train_and_evaluate(frame: pd.DataFrame) -> ModelBundle:
    if len(frame) < 50 or frame["discontinued"].nunique() < 2:
        raise ValueError("At least 50 validated trials and both outcome classes are required")
    train, test, strategy = _split(frame)
    estimator = _estimator()
    estimator.fit(train[FEATURE_COLUMNS], train["discontinued"])
    probability = estimator.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    y = test["discontinued"].to_numpy()
    metrics = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    holdout = test[["nct_id", "title", "first_post_date", "overall_status", "record_url", "discontinued"]].copy()
    holdout["risk_score"] = probability
    holdout["predicted_class"] = prediction
    transformer = estimator.named_steps["features"]
    names = transformer.get_feature_names_out()
    values = estimator.named_steps["classifier"].coef_[0]
    coefficients = pd.DataFrame({"feature": names, "coefficient": values})
    coefficients["direction"] = np.where(coefficients["coefficient"] >= 0, "Higher discontinuation signal", "Lower discontinuation signal")
    coefficients["magnitude"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("magnitude", ascending=False).reset_index(drop=True)
    calibration = _calibration_table(y, probability)
    drift = drift_report(train, test)
    metadata = {
        "split_strategy": strategy,
        "training_rows": len(train),
        "holdout_rows": len(test),
        "training_end": train["first_post_date"].max(),
        "holdout_start": test["first_post_date"].min(),
        "threshold": 0.5,
        "model": "L2-regularized logistic regression",
        "class_weight": "balanced",
    }
    return ModelBundle(estimator, metrics, holdout, calibration, coefficients, drift, metadata)


def score_scenario(model: ModelBundle, values: dict[str, Any]) -> float:
    row = {column: values.get(column) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([row])
    return float(model.estimator.predict_proba(frame[FEATURE_COLUMNS])[:, 1][0])
