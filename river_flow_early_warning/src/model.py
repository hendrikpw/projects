"""Leakage-safe three-day high-flow classifier, calibration and monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FEATURES = ["log_lag_1", "log_lag_2", "log_lag_3", "log_lag_7", "log_lag_14", "log_lag_30",
            "log_roll_mean_7", "log_roll_std_7", "log_roll_mean_30", "change_1d", "change_7d", "doy_sin", "doy_cos"]


@dataclass(frozen=True)
class ModelProduct:
    model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    thresholds: dict
    evaluation: pd.DataFrame
    station_metrics: pd.DataFrame
    drift: pd.DataFrame
    metrics: dict
    metadata: dict


def _psi(reference: pd.Series, current: pd.Series) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 2: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(pd.cut(reference, edges).value_counts(normalize=True, sort=False).to_numpy(), 1e-6, None)
    cur = np.clip(pd.cut(current, edges).value_counts(normalize=True, sort=False).to_numpy(), 1e-6, None)
    return float(np.sum((cur-ref)*np.log(cur/ref)))


def train_and_evaluate(gold: pd.DataFrame) -> ModelProduct:
    data = gold.dropna(subset=FEATURES+["future_max_3d"]).copy()
    train = data[data.event_date < "2024-01-01"].copy()
    calibration = data[data.event_date.between("2024-01-01", "2024-12-31")].copy()
    test = data[data.event_date >= "2025-01-01"].copy()
    if min(len(train), len(calibration), len(test)) < 1000: raise ValueError("insufficient temporal split")
    thresholds = train.groupby("site_no").discharge_cfs.quantile(.90).to_dict()
    for frame in (train, calibration, test):
        frame["high_flow_next_3d"] = frame.future_max_3d >= frame.site_no.map(thresholds)
        frame["month"] = frame.event_date.dt.month
    climatology = train.groupby(["site_no", "month"]).high_flow_next_3d.mean().to_dict()
    global_rate = float(train.high_flow_next_3d.mean())
    for frame in (train, calibration, test):
        frame["baseline_score"] = [climatology.get((site, month), global_rate) for site, month in zip(frame.site_no, frame.month)]
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=.06, max_leaf_nodes=24,
                                            min_samples_leaf=35, l2_regularization=1.2, random_state=42).fit(train[FEATURES], train.high_flow_next_3d)
    raw_cal = model.predict_proba(calibration[FEATURES])[:,1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, calibration.high_flow_next_3d.astype(int))
    ranking_score = model.predict_proba(test[FEATURES])[:,1]
    probability = calibrator.predict(ranking_score)
    evaluation = test[["site_no", "site_name", "event_date", "discharge_cfs", "future_max_3d", "high_flow_next_3d", "baseline_score"]].copy()
    evaluation["probability"] = probability
    evaluation["ranking_score"] = ranking_score
    evaluation["threshold_cfs"] = evaluation.site_no.map(thresholds)
    evaluation["status"] = np.select([probability >= .65, probability >= .35], ["alert", "watch"], default="normal")
    y = evaluation.high_flow_next_3d.astype(int).to_numpy(); budget=max(1,int(np.ceil(len(y)*.10)))
    selected=np.argsort(ranking_score)[-budget:]; selected_base=np.argsort(evaluation.baseline_score.to_numpy())[-budget:]
    metrics = {"average_precision":average_precision_score(y, ranking_score),
               "baseline_average_precision":average_precision_score(y, evaluation.baseline_score),
               "roc_auc":roc_auc_score(y, ranking_score), "brier":brier_score_loss(y, probability),
               "recall_at_10pct":float(y[selected].sum()/y.sum()), "baseline_recall_at_10pct":float(y[selected_base].sum()/y.sum()),
               "event_rate":float(y.mean()), "test_rows":len(test), "alerts":int((probability>=.65).sum())}
    if metrics["average_precision"] <= metrics["baseline_average_precision"]:
        raise RuntimeError(f"model failed baseline promotion gate: {metrics}")
    station_metrics = evaluation.groupby(["site_no","site_name"]).apply(lambda x: pd.Series({
        "rows":len(x), "events":int(x.high_flow_next_3d.sum()), "average_precision":average_precision_score(x.high_flow_next_3d, x.ranking_score) if x.high_flow_next_3d.nunique()>1 else np.nan,
        "alert_rate":float((x.probability>=.65).mean())}), include_groups=False).reset_index()
    drift = pd.DataFrame({"feature":FEATURES, "psi":[_psi(train[f], test[f]) for f in FEATURES]})
    drift["status"] = np.select([drift.psi>=.25, drift.psi>=.10], ["high","watch"], default="stable")
    metadata = {"algorithm":"HistGradientBoosting + isotonic calibration", "target":"next 3 days exceed station training 90th percentile",
                "train_rows":len(train), "calibration_rows":len(calibration), "test_rows":len(test),
                "train_period":"2018–2023", "calibration_period":"2024", "test_period":f"2025–{test.event_date.max().date()}"}
    return ModelProduct(model, calibrator, thresholds, evaluation, station_metrics, drift, metrics, metadata)


def score_latest(bundle: ModelProduct, gold: pd.DataFrame, site_no: str, flow_multiplier: float = 1.0) -> dict:
    row = gold[(gold.site_no==site_no) & gold[FEATURES].notna().all(axis=1)].sort_values("event_date").iloc[-1].copy()
    row["log_lag_1"] = np.log1p(max(0, np.expm1(row.log_lag_1)*flow_multiplier))
    raw = bundle.model.predict_proba(pd.DataFrame([row[FEATURES]], columns=FEATURES))[:,1]
    probability = float(bundle.calibrator.predict(raw)[0])
    return {"probability":probability, "status":"alert" if probability>=.65 else "watch" if probability>=.35 else "normal",
            "event_date":row.event_date, "discharge_cfs":float(np.expm1(row.log_lag_1)), "threshold_cfs":bundle.thresholds[site_no]}
