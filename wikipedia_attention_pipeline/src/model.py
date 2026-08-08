"""Rolling-origin attention forecasting, conformal intervals and anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES = [
    "article_index", "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_28",
    "rolling_std_28", "day_sin", "day_cos", "trend",
]


@dataclass(frozen=True)
class ModelBundle:
    estimator: HistGradientBoostingRegressor
    article_map: dict[str, int]
    metrics: dict[str, float]
    per_article: pd.DataFrame
    predictions: pd.DataFrame
    future: pd.DataFrame
    anomalies: pd.DataFrame
    residuals: pd.DataFrame
    metadata: dict[str, Any]


def _design(frame: pd.DataFrame, article_map: dict[str, int]) -> pd.DataFrame:
    result = frame.copy()
    result["article_index"] = result["article"].map(article_map)
    return result[FEATURES].astype(float)


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=220, max_leaf_nodes=24,
        min_samples_leaf=12, l2_regularization=1.0, random_state=42,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(actual)), 1)),
        "smape": float(np.mean(2 * np.abs(error) / np.maximum(np.abs(actual) + np.abs(predicted), 1))),
    }


def _future_features(article: str, history: pd.DataFrame, article_index: int, trend: int) -> dict[str, Any]:
    values = history.sort_values("event_time")["views"].astype(float).tolist()
    next_date = history["event_time"].max() + pd.offsets.Day(1)
    day = next_date.dayofweek
    return {
        "article": article, "event_time": next_date, "article_index": article_index,
        "lag_1": values[-1], "lag_7": values[-7], "lag_14": values[-14],
        "rolling_7": float(np.median(values[-7:])), "rolling_28": float(np.median(values[-28:])),
        "rolling_std_28": float(np.std(values[-28:], ddof=1)),
        "day_sin": float(np.sin(2 * np.pi * day / 7)), "day_cos": float(np.cos(2 * np.pi * day / 7)),
        "trend": trend,
    }


def forecast_future(
    estimator: HistGradientBoostingRegressor,
    silver: pd.DataFrame,
    article_map: dict[str, int],
    interval_radius: float,
    horizon: int = 14,
) -> pd.DataFrame:
    outputs = []
    for article, group in silver.groupby("article", sort=True):
        history = group[["event_time", "views"]].sort_values("event_time").copy()
        base_trend = len(history)
        for step in range(horizon):
            row = _future_features(article, history, article_map[article], base_trend + step)
            vector = pd.DataFrame([row])[FEATURES].astype(float)
            prediction = max(0.0, float(np.expm1(estimator.predict(vector)[0])))
            row.update({
                "prediction": prediction,
                "lower": max(0.0, prediction - interval_radius),
                "upper": prediction + interval_radius,
                "horizon_day": step + 1,
            })
            outputs.append(row)
            history = pd.concat([history, pd.DataFrame([{"event_time": row["event_time"], "views": prediction}])], ignore_index=True)
    return pd.DataFrame(outputs)


def train_and_evaluate(gold: pd.DataFrame, silver: pd.DataFrame, horizon: int = 14) -> ModelBundle:
    if len(gold) < 240 or gold["article"].nunique() < 2:
        raise ValueError("At least 240 feature rows across two articles are required")
    article_map = {article: index for index, article in enumerate(sorted(gold["article"].unique()))}
    max_date = gold["event_time"].max()
    test_start = max_date - pd.offsets.Day(27)
    calibration_start = test_start - pd.offsets.Day(28)
    train = gold[gold["event_time"] < calibration_start].copy()
    calibration = gold[(gold["event_time"] >= calibration_start) & (gold["event_time"] < test_start)].copy()
    test = gold[gold["event_time"] >= test_start].copy()
    if min(len(train), len(calibration), len(test)) < gold["article"].nunique() * 14:
        raise ValueError("Insufficient rolling-origin train, calibration or test window")

    evaluator = _model()
    evaluator.fit(_design(train, article_map), np.log1p(train["views"]))
    calibration_prediction = np.maximum(0, np.expm1(evaluator.predict(_design(calibration, article_map))))
    calibration_error = np.abs(calibration["views"].to_numpy() - calibration_prediction)
    interval_radius = float(np.quantile(calibration_error, 0.90, method="higher"))
    prediction = np.maximum(0, np.expm1(evaluator.predict(_design(test, article_map))))
    actual = test["views"].to_numpy(dtype=float)
    baseline = test["lag_7"].to_numpy(dtype=float)
    lower, upper = np.maximum(0, prediction - interval_radius), prediction + interval_radius
    metrics = _metrics(actual, prediction)
    baseline_metrics = _metrics(actual, baseline)
    metrics.update({
        "baseline_wape": baseline_metrics["wape"],
        "skill_vs_weekly_naive": float(1 - metrics["wape"] / max(baseline_metrics["wape"], 1e-12)),
        "interval_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "mean_interval_width": float(np.mean(upper - lower)),
    })
    predictions = test[["article", "event_time", "views", "lag_7"]].copy()
    predictions["prediction"] = prediction
    predictions["lower"] = lower
    predictions["upper"] = upper
    predictions["residual"] = actual - prediction
    predictions["absolute_error"] = np.abs(predictions["residual"])
    predictions["outside_interval"] = (actual < lower) | (actual > upper)
    predictions["interval_direction"] = np.where(actual > upper, "Above", np.where(actual < lower, "Below", "Inside"))
    predictions["severity"] = predictions["absolute_error"] / max(interval_radius, 1)

    rows = []
    for article, part in predictions.groupby("article"):
        item = _metrics(part["views"].to_numpy(float), part["prediction"].to_numpy(float))
        item.update({
            "article": article, "baseline_wape": _metrics(part["views"].to_numpy(float), part["lag_7"].to_numpy(float))["wape"],
            "interval_coverage": float((~part["outside_interval"]).mean()), "test_days": len(part),
        })
        item["skill_vs_weekly_naive"] = 1 - item["wape"] / max(item["baseline_wape"], 1e-12)
        rows.append(item)
    per_article = pd.DataFrame(rows).sort_values("wape")

    production = _model()
    production.fit(_design(gold, article_map), np.log1p(gold["views"]))
    future = forecast_future(production, silver, article_map, interval_radius, horizon)
    anomalies = predictions[predictions["outside_interval"]].sort_values("severity", ascending=False).reset_index(drop=True)
    residuals = predictions.groupby("event_time", as_index=False).agg(
        mean_residual=("residual", "mean"), mean_absolute_error=("absolute_error", "mean"), anomaly_count=("outside_interval", "sum")
    )
    metadata = {
        "model": "global histogram gradient boosting on log pageviews",
        "training_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
        "training_end": (calibration_start - pd.offsets.Day(1)).date().isoformat(),
        "calibration_start": calibration_start.date().isoformat(), "test_start": test_start.date().isoformat(),
        "test_end": max_date.date().isoformat(), "interval_level": 0.90, "interval_radius": interval_radius,
        "future_refit_rows": len(gold), "horizon_days": horizon, "random_state": 42,
    }
    return ModelBundle(production, article_map, metrics, per_article, predictions.reset_index(drop=True), future, anomalies, residuals, metadata)
