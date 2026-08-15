"""Peer-local anomaly detection, stress evaluation, drift and explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

FEATURES = ["revenue_growth_yoy", "net_margin", "liability_ratio", "asset_turnover_quarterly", "return_on_assets_quarterly", "margin_change_yoy"]
LABELS = {
    "revenue_growth_yoy": "Revenue growth YoY", "net_margin": "Net margin",
    "liability_ratio": "Implied liabilities / assets", "asset_turnover_quarterly": "Quarterly asset turnover",
    "return_on_assets_quarterly": "Quarterly return on assets", "margin_change_yoy": "Net-margin change YoY",
}


@dataclass(frozen=True)
class ModelBundle:
    model: LocalOutlierFactor
    scaler: RobustScaler
    train: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: pd.DataFrame
    stress_evaluation: pd.DataFrame
    drift: pd.DataFrame
    metrics: dict
    metadata: dict


def _split(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = gold.dropna(subset=FEATURES).sort_values(["calendar_year", "calendar_quarter", "ticker"]).copy()
    periods = sorted(data.frame.unique())
    if len(periods) < 16:
        raise ValueError("insufficient quarterly history")
    cal_periods = set(periods[-16:-8]); test_periods = set(periods[-8:])
    train = data[~data.frame.isin(cal_periods | test_periods)].copy()
    calibration = data[data.frame.isin(cal_periods)].copy()
    test = data[data.frame.isin(test_periods)].copy()
    return train, calibration, test


def _psi(reference: pd.Series, current: pd.Series) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return 0.0 if np.allclose(current, edges[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = pd.cut(reference, edges, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy()
    cur = pd.cut(current, edges, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy()
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur-ref) * np.log(cur/ref)))


def _risk(model: LocalOutlierFactor, matrix: np.ndarray) -> np.ndarray:
    return -model.score_samples(matrix)


def _baseline(train_scaled: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    center = np.median(train_scaled, axis=0)
    return np.max(np.abs(matrix-center), axis=1)


def _inject(test_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    clean = test_scaled.copy(); stressed = test_scaled.copy(); scenarios = []
    for index in range(len(stressed)):
        primary = index % len(FEATURES)
        secondary = (primary + 2) % len(FEATURES)
        direction = -1 if primary in {0, 1, 4, 5} else 1
        stressed[index, primary] += direction * 2.5
        stressed[index, secondary] += direction * 2.0
        scenarios.append(f"{LABELS[FEATURES[primary]]} stress")
    combined = np.vstack([clean, stressed])
    labels = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(stressed), dtype=int)]
    return combined, labels, ["Observed"]*len(clean) + scenarios


def train_and_evaluate(gold: pd.DataFrame) -> ModelBundle:
    train, calibration, test = _split(gold)
    if min(len(train), len(calibration), len(test)) < 24:
        raise ValueError("insufficient train/calibration/test rows")
    scaler = RobustScaler(quantile_range=(10, 90)).fit(train[FEATURES])
    train_x = scaler.transform(train[FEATURES]); cal_x = scaler.transform(calibration[FEATURES]); test_x = scaler.transform(test[FEATURES])
    neighbors = min(30, max(12, len(train)//8))
    model = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, contamination="auto", metric="minkowski", p=2).fit(train_x)
    cal_risk = _risk(model, cal_x); threshold = float(np.quantile(cal_risk, .95))
    test_risk = _risk(model, test_x); flagged = test_risk >= threshold
    evaluation = test[["gold_id", "ticker", "frame", "latest_filed"] + FEATURES].copy()
    evaluation["anomaly_score"] = test_risk
    evaluation["status"] = np.where(flagged, "review", "monitor")
    evaluation["score_percentile"] = pd.Series(test_risk).rank(pct=True).to_numpy()
    injected_x, labels, scenario = _inject(test_x)
    candidate = _risk(model, injected_x); baseline = _baseline(train_x, injected_x)
    budget = max(1, int(np.ceil(len(labels)*.10)))
    selected = np.argsort(candidate)[-budget:]; selected_base = np.argsort(baseline)[-budget:]
    stress_eval = pd.DataFrame({"label": labels, "scenario": scenario, "candidate_score": candidate, "baseline_score": baseline})
    stress_eval["candidate_rank"] = stress_eval.candidate_score.rank(ascending=False, method="min")
    stress_eval["baseline_rank"] = stress_eval.baseline_score.rank(ascending=False, method="min")
    candidate_ap = average_precision_score(labels, candidate); baseline_ap = average_precision_score(labels, baseline)
    metrics = {
        "stress_average_precision": candidate_ap, "baseline_average_precision": baseline_ap,
        "stress_roc_auc": roc_auc_score(labels, candidate),
        "recall_at_10pct": float(labels[selected].sum()/labels.sum()),
        "baseline_recall_at_10pct": float(labels[selected_base].sum()/labels.sum()),
        "natural_review_rate": float(flagged.mean()), "natural_review_count": int(flagged.sum()),
        "threshold": threshold, "calibration_false_alert_rate": float((cal_risk >= threshold).mean()),
        "test_rows": len(test), "stress_rows": int(labels.sum()),
    }
    if candidate_ap < baseline_ap:
        raise RuntimeError("candidate failed stress promotion gate")
    drift = pd.DataFrame({"feature": FEATURES, "psi": [_psi(train[c], test[c]) for c in FEATURES]})
    drift["label"] = drift.feature.map(LABELS)
    drift["status"] = np.select([drift.psi >= .25, drift.psi >= .10], ["high", "watch"], default="stable")
    calibration_out = calibration[["gold_id", "ticker", "frame"]].copy(); calibration_out["anomaly_score"] = cal_risk
    metadata = {
        "features": FEATURES, "algorithm": "Local Outlier Factor novelty detection",
        "n_neighbors": neighbors, "scaler": "RobustScaler 10th-90th percentile",
        "split": "latest 8 reported calendar frames test / prior 8 calibration / earlier train",
        "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
        "train_periods": f"{train.frame.min()}–{train.frame.max()}", "calibration_periods": f"{calibration.frame.min()}–{calibration.frame.max()}", "test_periods": f"{test.frame.min()}–{test.frame.max()}",
    }
    return ModelBundle(model, scaler, train, calibration_out, evaluation, stress_eval, drift, metrics, metadata)


def score_scenario(bundle: ModelBundle, row: pd.Series, adjustments: dict[str, float]) -> dict:
    values = row[FEATURES].astype(float).copy()
    for feature, delta in adjustments.items():
        values[feature] += delta
    scaled = bundle.scaler.transform(pd.DataFrame([values], columns=FEATURES))
    risk = float(_risk(bundle.model, scaled)[0])
    train_x = bundle.scaler.transform(bundle.train[FEATURES])
    distances = np.sqrt(((train_x-scaled[0])**2).sum(axis=1)); nearest = np.argsort(distances)[:8]
    peer_center = np.median(train_x[nearest], axis=0); contribution = np.abs(scaled[0]-peer_center)
    evidence = pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "local_deviation": contribution, "scenario_value": values.to_numpy()}).sort_values("local_deviation", ascending=False)
    peers = bundle.train.iloc[nearest][["ticker", "frame"]].copy(); peers["distance"] = distances[nearest]
    return {"anomaly_score": risk, "status": "review" if risk >= bundle.metrics["threshold"] else "monitor", "evidence": evidence, "peers": peers}
