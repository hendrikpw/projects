"""Leakage-aware resolution-time quantiles, calibration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL = ["agency", "complaint_type", "descriptor", "location_type", "borough", "open_data_channel_type"]
NUMERIC = ["created_hour", "created_dow", "created_month", "is_weekend", "is_overnight"]


@dataclass(frozen=True)
class ModelBundle:
    median_model: Pipeline
    upper_model: Pipeline
    metrics: dict[str, float]
    predictions: pd.DataFrame
    scorecard: pd.DataFrame
    drift: pd.DataFrame
    metadata: dict[str, Any]


def _pipeline(quantile: float) -> Pipeline:
    prep = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=4, sparse_output=False), CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ])
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, learning_rate=.065, max_iter=180,
        max_leaf_nodes=24, min_samples_leaf=18, l2_regularization=1.2, random_state=42,
    )
    return Pipeline([("prepare", prep), ("model", model)])


def _predict_hours(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, np.expm1(model.predict(frame[CATEGORICAL + NUMERIC])))


def _temporal_split(frame: pd.DataFrame, calibration_days: int = 45, test_days: int = 45) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end = frame["created_at"].max().normalize()
    test_start = end - pd.offsets.Day(test_days - 1)
    calibration_start = test_start - pd.offsets.Day(calibration_days)
    train = frame[frame["created_at"] < calibration_start].copy()
    calibration = frame[(frame["created_at"] >= calibration_start) & (frame["created_at"] < test_start)].copy()
    test = frame[frame["created_at"] >= test_start].copy()
    if min(len(train), len(calibration), len(test)) < 120:
        raise ValueError("insufficient rows in chronological train/calibration/test windows")
    return train, calibration, test


def _population_stability(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = pd.cut(train, edges, include_lowest=True).value_counts(normalize=True, sort=False).clip(1e-5)
    b = pd.cut(test, edges, include_lowest=True).value_counts(normalize=True, sort=False).clip(1e-5)
    return float(((b - a) * np.log(b / a)).sum())


def _category_drift(train: pd.Series, test: pd.Series) -> float:
    labels = sorted(set(train.astype(str)) | set(test.astype(str)))
    a = train.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0).clip(1e-5)
    b = test.astype(str).value_counts(normalize=True).reindex(labels, fill_value=0).clip(1e-5)
    return float(((b - a) * np.log(b / a)).sum())


def train_and_evaluate(gold: pd.DataFrame) -> ModelBundle:
    if len(gold) < 1000:
        raise ValueError("at least 1,000 validated records are required")
    train, calibration, test = _temporal_split(gold)
    median_model, upper_model = _pipeline(.5), _pipeline(.9)
    median_model.fit(train[CATEGORICAL + NUMERIC], train["target_log_hours"])
    upper_model.fit(train[CATEGORICAL + NUMERIC], train["target_log_hours"])
    cal_upper = _predict_hours(upper_model, calibration)
    correction = float(max(0.0, np.quantile(calibration["resolution_hours"].to_numpy() - cal_upper, .9, method="higher")))
    predicted = _predict_hours(median_model, test)
    upper = np.maximum(predicted, _predict_hours(upper_model, test) + correction)
    actual = test["resolution_hours"].to_numpy(float)

    fallback = train.groupby(["agency", "complaint_type"])["resolution_hours"].median()
    agency_fallback = train.groupby("agency")["resolution_hours"].median()
    global_fallback = float(train["resolution_hours"].median())
    baseline = np.array([
        fallback.get((row.agency, row.complaint_type), agency_fallback.get(row.agency, global_fallback))
        for row in test.itertuples()
    ], dtype=float)
    mae, baseline_mae = mean_absolute_error(actual, predicted), mean_absolute_error(actual, baseline)
    metrics = {
        "mae_hours": float(mae), "rmse_hours": float(mean_squared_error(actual, predicted) ** .5),
        "median_ae_hours": float(np.median(np.abs(actual - predicted))),
        "baseline_mae_hours": float(baseline_mae), "skill_vs_group_median": float(1 - mae / max(baseline_mae, 1e-9)),
        "pinball_q50": float(mean_pinball_loss(actual, predicted, alpha=.5)),
        "pinball_q90": float(mean_pinball_loss(actual, upper, alpha=.9)),
        "upper_coverage": float((actual <= upper).mean()), "mean_upper_hours": float(upper.mean()),
        "within_24h_accuracy": float(((predicted <= 24) == (actual <= 24)).mean()),
    }
    predictions = test[["unique_key", "created_at", "agency", "complaint_type", "borough", "open_data_channel_type", "resolution_hours"]].copy()
    predictions["predicted_median_hours"] = predicted
    predictions["predicted_upper_hours"] = upper
    predictions["baseline_hours"] = baseline
    predictions["absolute_error"] = np.abs(actual - predicted)
    predictions["upper_breach"] = actual > upper
    predictions["risk_band"] = pd.cut(upper, [-np.inf, 24, 72, 168, np.inf], labels=["Under 1 day", "1–3 days", "3–7 days", "Over 7 days"])

    rows = []
    for agency, part in predictions.groupby("agency"):
        rows.append({
            "agency": agency, "test_requests": len(part),
            "mae_hours": part["absolute_error"].mean(),
            "baseline_mae_hours": np.abs(part["resolution_hours"] - part["baseline_hours"]).mean(),
            "upper_coverage": (~part["upper_breach"]).mean(), "median_actual_hours": part["resolution_hours"].median(),
        })
    scorecard = pd.DataFrame(rows).sort_values("test_requests", ascending=False)
    scorecard["skill_vs_baseline"] = 1 - scorecard["mae_hours"] / scorecard["baseline_mae_hours"].clip(lower=1e-9)
    drift = pd.DataFrame([
        {"feature": "created_hour", "psi": _population_stability(train["created_hour"], test["created_hour"])},
        {"feature": "agency", "psi": _category_drift(train["agency"], test["agency"])},
        {"feature": "complaint_type", "psi": _category_drift(train["complaint_type"], test["complaint_type"])},
        {"feature": "borough", "psi": _category_drift(train["borough"], test["borough"])},
        {"feature": "channel", "psi": _category_drift(train["open_data_channel_type"], test["open_data_channel_type"])},
    ])
    drift["status"] = pd.cut(drift["psi"], [-np.inf, .1, .25, np.inf], labels=["Stable", "Watch", "Investigate"])
    metadata = {
        "model": "HistGradientBoosting quantile regressors on log1p hours", "random_state": 42,
        "training_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
        "training_end": (test["created_at"].min().normalize() - pd.offsets.Day(46)).date().isoformat(),
        "calibration_start": (test["created_at"].min().normalize() - pd.offsets.Day(45)).date().isoformat(),
        "test_start": test["created_at"].min().date().isoformat(), "test_end": test["created_at"].max().date().isoformat(),
        "upper_correction_hours": correction, "nominal_upper_coverage": .9,
    }
    return ModelBundle(median_model, upper_model, metrics, predictions.reset_index(drop=True), scorecard, drift, metadata)


def score_case(bundle: ModelBundle, gold: pd.DataFrame, values: dict[str, Any]) -> dict[str, float | str]:
    row = pd.DataFrame([{**values,
        "created_month": int(values.get("created_month", gold["created_month"].mode().iloc[0])),
        "is_weekend": int(int(values["created_dow"]) >= 5),
        "is_overnight": int(int(values["created_hour"]) <= 5),
    }])
    median = float(_predict_hours(bundle.median_model, row)[0])
    upper = float(max(median, _predict_hours(bundle.upper_model, row)[0] + bundle.metadata["upper_correction_hours"]))
    band = "Under 1 day" if upper <= 24 else "1–3 days" if upper <= 72 else "3–7 days" if upper <= 168 else "Over 7 days"
    return {"median_hours": median, "upper_hours": upper, "risk_band": band}
