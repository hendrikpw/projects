"""Leakage-safe day-ahead forecasting, conformal intervals, drift and serving guards."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score


@dataclass(frozen=True)
class ForecastModel:
    point_model: HistGradientBoostingRegressor
    lower_model: HistGradientBoostingRegressor
    upper_model: HistGradientBoostingRegressor
    features: list[str]
    evaluation: pd.DataFrame
    metrics: dict
    drift: pd.DataFrame
    importance: pd.DataFrame
    metadata: dict
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    blend_weight: float
    conformal_adjustment: float


def make_features(gold: pd.DataFrame, horizon: int = 24) -> tuple[pd.DataFrame, list[str]]:
    x = gold.sort_values("timestamp").reset_index(drop=True).copy()
    x["current_load"] = x.load_kw
    for lag in [1, 2, 3, 24, 48, 168]: x[f"load_lag_{lag}"] = x.load_kw.shift(lag)
    x["load_mean_24"] = x.load_kw.rolling(24, min_periods=24).mean()
    x["load_std_24"] = x.load_kw.rolling(24, min_periods=24).std()
    x["load_mean_168"] = x.load_kw.rolling(168, min_periods=168).mean()
    x["load_std_168"] = x.load_kw.rolling(168, min_periods=168).std()
    hour = x.timestamp.dt.hour; dow = x.timestamp.dt.dayofweek; doy = x.timestamp.dt.dayofyear
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24); x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["dow_sin"] = np.sin(2 * np.pi * dow / 7); x["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    x["year_sin"] = np.sin(2 * np.pi * doy / 365.25); x["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    x["submeter_share"] = (x.kitchen_wh + x.laundry_wh + x.climate_wh) / (x.load_kw * 1000 + 1)
    x["target"] = x.load_kw.shift(-horizon); x["target_time"] = x.timestamp + pd.to_timedelta(horizon, unit="h")
    features = ["current_load", "load_lag_1", "load_lag_2", "load_lag_3", "load_lag_24", "load_lag_48", "load_lag_168", "load_mean_24", "load_std_24", "load_mean_168", "load_std_168", "voltage_v", "intensity_a", "submeter_share", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "year_sin", "year_cos"]
    return x.dropna(subset=[*features, "target"]).reset_index(drop=True), features


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0] / len(reference); b = np.histogram(current, bins=edges)[0] / len(current)
    a, b = np.clip(a, 1e-5, None), np.clip(b, 1e-5, None)
    return float(np.sum((b - a) * np.log(b / a)))


def _peak_capture(actual: np.ndarray, score: np.ndarray, fraction: float = .10) -> float:
    count = max(1, int(np.ceil(len(actual) * fraction)))
    actual_peak = set(np.argpartition(actual, -count)[-count:]); selected = set(np.argpartition(score, -count)[-count:])
    return len(actual_peak & selected) / count


def train_and_evaluate(gold: pd.DataFrame, horizon: int = 24, interval_coverage: float = .80) -> ForecastModel:
    data, features = make_features(gold, horizon); n = len(data); train_end = int(n * .70); calibration_end = int(n * .85)
    train, calibration, test = data.iloc[:train_end], data.iloc[train_end:calibration_end], data.iloc[calibration_end:]
    if min(len(train), len(calibration), len(test)) < 1000:
        raise RuntimeError("temporal train/calibration/test windows are incomplete")
    x_train, x_cal, x_test = train[features], calibration[features], test[features]
    y_train, y_cal, y_test = train.target.to_numpy(), calibration.target.to_numpy(), test.target.to_numpy()
    residual_train = y_train - train.current_load.to_numpy()
    params = dict(max_iter=170, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=.25, random_state=42)
    started = time.perf_counter(); point = HistGradientBoostingRegressor(loss="squared_error", **params).fit(x_train, residual_train)
    lower = HistGradientBoostingRegressor(loss="quantile", quantile=(1 - interval_coverage) / 2, **params).fit(x_train, residual_train)
    upper = HistGradientBoostingRegressor(loss="quantile", quantile=1 - (1 - interval_coverage) / 2, **params).fit(x_train, residual_train)
    fit_seconds = time.perf_counter() - started
    cal_residual = point.predict(x_cal); cal_base = calibration.current_load.to_numpy()
    weights = np.linspace(0, 1, 21); losses = [mean_absolute_error(y_cal, cal_base + weight * cal_residual) for weight in weights]
    blend_weight = float(weights[int(np.argmin(losses))])
    cal_low = cal_base + blend_weight * lower.predict(x_cal); cal_high = cal_base + blend_weight * upper.predict(x_cal)
    nonconformity = np.maximum.reduce([cal_low - y_cal, y_cal - cal_high, np.zeros(len(y_cal))])
    adjustment = float(np.quantile(nonconformity, interval_coverage, method="higher"))
    started = time.perf_counter(); residual_test = point.predict(x_test); inference_ms = (time.perf_counter() - started) * 1000 / len(test)
    baseline = test.current_load.to_numpy(); forecast = np.clip(baseline + blend_weight * residual_test, 0, None)
    lower_prediction = np.clip(baseline + blend_weight * lower.predict(x_test) - adjustment, 0, None)
    upper_prediction = np.maximum(forecast, baseline + blend_weight * upper.predict(x_test) + adjustment)
    covered = (y_test >= lower_prediction) & (y_test <= upper_prediction)
    threshold = np.quantile(forecast, .90)
    evaluation = test[["timestamp", "target_time", "target", "current_load"]].rename(columns={"timestamp": "issue_time", "target": "actual", "current_load": "baseline"}).copy()
    evaluation["forecast"] = forecast; evaluation["lower"] = lower_prediction; evaluation["upper"] = upper_prediction; evaluation["covered"] = covered; evaluation["peak_review"] = forecast >= threshold; evaluation["absolute_error"] = np.abs(y_test - forecast)
    metrics = {
        "test_hours": len(test), "mae_kw": float(mean_absolute_error(y_test, forecast)), "baseline_mae_kw": float(mean_absolute_error(y_test, baseline)),
        "rmse_kw": float(mean_squared_error(y_test, forecast) ** .5), "mape": float(np.mean(np.abs(y_test - forecast) / np.clip(y_test, .1, None))), "r2": float(r2_score(y_test, forecast)),
        "interval_coverage": float(covered.mean()), "mean_interval_width_kw": float(np.mean(upper_prediction - lower_prediction)),
        "lower_pinball": float(mean_pinball_loss(y_test, lower_prediction, alpha=(1 - interval_coverage) / 2)), "upper_pinball": float(mean_pinball_loss(y_test, upper_prediction, alpha=1 - (1 - interval_coverage) / 2)),
        "peak_capture_at_10pct": _peak_capture(y_test, forecast), "baseline_peak_capture_at_10pct": _peak_capture(y_test, baseline), "inference_ms_per_hour": float(inference_ms),
    }
    if metrics["mae_kw"] > metrics["baseline_mae_kw"] + 1e-9 or not (.70 <= metrics["interval_coverage"] <= .95):
        raise RuntimeError(f"forecast model failed promotion gates: {metrics}")
    bounds_low = train[features].quantile(.001).to_numpy(); bounds_high = train[features].quantile(.999).to_numpy()
    drift = pd.DataFrame([{"feature": feature, "psi": _psi(train[feature].to_numpy(), test[feature].to_numpy())} for feature in features]).sort_values("psi", ascending=False).reset_index(drop=True)
    sample = test.iloc[np.linspace(0, len(test) - 1, min(1200, len(test)), dtype=int)]
    importance_raw = permutation_importance(point, sample[features], sample.target.to_numpy() - sample.current_load.to_numpy(), scoring="neg_mean_absolute_error", n_repeats=3, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({"feature": features, "importance": importance_raw.importances_mean}).sort_values("importance", ascending=False).reset_index(drop=True)
    metadata = {"train_hours": len(train), "calibration_hours": len(calibration), "test_hours": len(test), "train_until": str(train.target_time.max()), "calibration_until": str(calibration.target_time.max()), "test_until": str(test.target_time.max()), "horizon_hours": horizon, "nominal_coverage": interval_coverage, "blend_weight": blend_weight, "conformal_adjustment_kw": adjustment, "fit_seconds": round(fit_seconds, 2)}
    return ForecastModel(point, lower, upper, features, evaluation, metrics, drift, importance, metadata, bounds_low, bounds_high, blend_weight, adjustment)


def score_scenario(model: ForecastModel, row: pd.Series, load_scale: float = 1.0, missing_lags: int = 0) -> dict:
    values = row[model.features].to_numpy(float).copy(); load_positions = [index for index, name in enumerate(model.features) if name.startswith(("current_load", "load_lag", "load_mean", "load_std"))]
    values[load_positions] *= load_scale
    lag_positions = [index for index, name in enumerate(model.features) if name.startswith("load_lag")]
    values[lag_positions[:missing_lags]] = np.nan
    missing_share = float(np.isnan(values).mean()); finite = np.where(np.isfinite(values), values, (model.lower_bounds + model.upper_bounds) / 2)
    ood_count = int(((finite < model.lower_bounds) | (finite > model.upper_bounds)).sum())
    frame = pd.DataFrame([values], columns=model.features); baseline = float(frame.current_load.iloc[0])
    point = max(0.0, baseline + model.blend_weight * float(model.point_model.predict(frame)[0]))
    low = max(0.0, baseline + model.blend_weight * float(model.lower_model.predict(frame)[0]) - model.conformal_adjustment)
    high = max(point, baseline + model.blend_weight * float(model.upper_model.predict(frame)[0]) + model.conformal_adjustment)
    route = "forecast-withheld" if missing_share > .10 or ood_count >= 4 else "review" if missing_share > 0 or ood_count > 0 else "auto-forecast"
    return {"forecast_kw": point, "lower_kw": low, "upper_kw": high, "route": route, "missing_share": missing_share, "ood_features": ood_count}
