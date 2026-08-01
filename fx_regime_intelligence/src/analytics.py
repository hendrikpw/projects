"""FX returns, risk diagnostics, anomaly detection and allocation analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest


TRADING_DAYS = 252


def rate_matrix(data: pd.DataFrame, currencies: list[str] | None = None) -> pd.DataFrame:
    """Pivot long ECB observations to an aligned date × currency rate matrix."""
    frame = data.copy()
    if currencies:
        frame = frame[frame["currency"].isin(currencies)]
    matrix = frame.pivot(index="date", columns="currency", values="rate_per_eur")
    return matrix.sort_index().ffill(limit=3).dropna(how="all")


def normalized_rates(rates: pd.DataFrame) -> pd.DataFrame:
    """Rebase each valid rate series to 100 on its first observation."""
    if rates.empty:
        return rates.copy()
    first = rates.apply(lambda series: series.dropna().iloc[0] if series.notna().any() else np.nan)
    return rates.divide(first).multiply(100)


def log_returns(rates: pd.DataFrame) -> pd.DataFrame:
    """Calculate log changes in quoted foreign-currency units per euro."""
    return np.log(rates / rates.shift(1)).replace([np.inf, -np.inf], np.nan)


def risk_summary(rates: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Summarize current rate, momentum, volatility, VaR and drawdown per currency."""
    returns = log_returns(rates)
    rows = []
    for currency in rates.columns:
        series = rates[currency].dropna()
        changes = returns[currency].dropna()
        if len(series) < 2 or changes.empty:
            continue
        recent_window = changes.tail(max(int(window), 2))
        wealth = series / series.iloc[0]
        drawdown = wealth / wealth.cummax() - 1
        momentum_days = min(30, len(series) - 1)
        rows.append(
            {
                "currency": currency,
                "latest_rate": float(series.iloc[-1]),
                "change_30d": float((series.iloc[-1] / series.iloc[-momentum_days - 1] - 1) * 100),
                "annualized_volatility": float(recent_window.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
                "historical_var_95": float(-changes.quantile(0.05) * 100),
                "worst_day": float(changes.min() * 100),
                "max_drawdown": float(drawdown.min() * 100),
                "observations": int(series.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("annualized_volatility", ascending=False)


def rolling_volatility(rates: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Annualize rolling standard deviation of daily log returns."""
    return log_returns(rates).rolling(max(int(window), 5)).std() * np.sqrt(TRADING_DAYS) * 100


def market_regimes(rates: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Classify cross-currency calm, normal and stress states using expanding quantiles."""
    returns = log_returns(rates)
    volatility = returns.rolling(max(int(window), 5)).std() * np.sqrt(TRADING_DAYS)
    features = pd.DataFrame(
        {
            "absolute_move": returns.abs().median(axis=1) * 100,
            "dispersion": returns.std(axis=1) * 100,
            "market_volatility": volatility.median(axis=1) * 100,
        }
    ).dropna()
    baseline = features["market_volatility"].expanding(min_periods=60)
    low = baseline.quantile(0.35)
    high = baseline.quantile(0.80)
    features["regime"] = np.select(
        [features["market_volatility"] >= high, features["market_volatility"] <= low],
        ["Stress", "Calm"],
        default="Normal",
    )
    return features


def detect_anomalies(
    rates: pd.DataFrame,
    contamination: float = 0.03,
    window: int = 20,
) -> pd.DataFrame:
    """Use Isolation Forest on multi-currency returns and rolling market features."""
    returns = log_returns(rates)
    features = returns.add_prefix("return_").copy()
    features["median_abs_return"] = returns.abs().median(axis=1)
    features["cross_section_dispersion"] = returns.std(axis=1)
    features["rolling_market_volatility"] = (
        returns.rolling(max(int(window), 5)).std().median(axis=1)
    )
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < 40:
        return pd.DataFrame(columns=["date", "anomaly_score", "is_anomaly"])
    standardized = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
    model = IsolationForest(
        n_estimators=250,
        contamination=float(np.clip(contamination, 0.005, 0.15)),
        random_state=42,
        n_jobs=-1,
    )
    prediction = model.fit_predict(standardized)
    raw_score = -model.decision_function(standardized)
    result = pd.DataFrame(
        {
            "date": features.index,
            "anomaly_score": pd.Series(raw_score).rank(pct=True).to_numpy() * 100,
            "is_anomaly": prediction.eq(-1) if isinstance(prediction, pd.Series) else prediction == -1,
            "market_move": features["median_abs_return"].to_numpy() * 100,
            "dispersion": features["cross_section_dispersion"].to_numpy() * 100,
        }
    )
    return result.sort_values("date").reset_index(drop=True)


def correlation_and_clusters(
    rates: pd.DataFrame, clusters: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return currency correlations and hierarchical behavior groups."""
    returns = log_returns(rates).dropna(how="all")
    correlation = returns.corr()
    if len(correlation) < 2:
        groups = pd.DataFrame({"currency": correlation.columns, "cluster": 1})
        return correlation, groups
    distance = (1 - correlation).clip(0, 2).fillna(1)
    count = max(2, min(int(clusters), len(correlation)))
    model = AgglomerativeClustering(n_clusters=count, metric="precomputed", linkage="average")
    labels = model.fit_predict(distance)
    groups = pd.DataFrame(
        {"currency": correlation.columns, "cluster": labels + 1}
    ).sort_values(["cluster", "currency"])
    return correlation, groups


def inverse_volatility_allocation(rates: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Create a transparent inverse-volatility reference allocation."""
    returns = log_returns(rates).tail(max(int(window), 20))
    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    valid = volatility.replace(0, np.nan).dropna()
    inverse = 1 / valid
    weights = inverse / inverse.sum()
    return pd.DataFrame(
        {
            "currency": weights.index,
            "annualized_volatility": volatility.loc[weights.index].to_numpy() * 100,
            "weight": weights.to_numpy() * 100,
        }
    ).sort_values("weight", ascending=False)


def shock_scenario(rate: float, eur_amount: float, shock_percent: float) -> dict:
    """Translate a hypothetical FX-rate shock into foreign-currency exposure values."""
    shocked_rate = float(rate) * (1 + float(shock_percent) / 100)
    current_value = float(eur_amount) * float(rate)
    shocked_value = float(eur_amount) * shocked_rate
    return {
        "current_rate": float(rate),
        "shocked_rate": shocked_rate,
        "current_foreign_value": current_value,
        "shocked_foreign_value": shocked_value,
        "foreign_value_change": shocked_value - current_value,
    }
