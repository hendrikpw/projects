"""Transparent scoring, clustering and scenario analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_transition_intelligence.src.data import INDICATORS


SCORE_CODES = [code for code, meta in INDICATORS.items() if meta["weight"] > 0]


def latest_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    """Build one latest-available row per country with value and year columns."""
    required = {"country", "country_code", "year", "indicator_code", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    valid = data.dropna(subset=["value"]).sort_values("year")
    latest = valid.groupby(["country", "country_code", "indicator_code"], as_index=False).tail(1)
    values = latest.pivot(
        index=["country", "country_code"], columns="indicator_code", values="value"
    )
    years = latest.pivot(
        index=["country", "country_code"], columns="indicator_code", values="year"
    ).add_suffix("_year")
    snapshot = values.join(years).reset_index()
    snapshot.columns.name = None
    for code in SCORE_CODES:
        if code not in snapshot:
            snapshot[code] = np.nan
    snapshot["coverage"] = snapshot[SCORE_CODES].notna().mean(axis=1)
    return snapshot


def _minmax(series: pd.Series, inverse: bool = False) -> pd.Series:
    values = series.astype(float)
    spread = values.max() - values.min()
    scaled = pd.Series(0.5, index=values.index) if spread == 0 else (values - values.min()) / spread
    return 1 - scaled if inverse else scaled


def score_countries(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Calculate a documented 0–100 relative transition score."""
    scored = snapshot.copy()
    imputed = []
    for code in SCORE_CODES:
        if code not in scored:
            scored[code] = np.nan
        missing = scored[code].isna()
        fill_value = scored[code].median()
        if pd.isna(fill_value):
            raise ValueError(f"No usable values for {code}")
        scored.loc[missing, code] = fill_value
        imputed.append(missing)

    components = []
    for code in SCORE_CODES:
        meta = INDICATORS[code]
        component = _minmax(scored[code], inverse=meta["direction"] == "lower")
        components.append(component * float(meta["weight"]))
        scored[f"{code}_component"] = component * 100

    scored["transition_score"] = sum(components) * 100
    scored["imputed_fields"] = np.column_stack(imputed).sum(axis=1)
    scored["rank"] = scored["transition_score"].rank(method="min", ascending=False).astype(int)
    return scored.sort_values(["rank", "country"]).reset_index(drop=True)


def cluster_countries(scored: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Create reproducible K-means profiles and a two-dimensional PCA projection."""
    if not 2 <= n_clusters <= 5:
        raise ValueError("n_clusters must be between 2 and 5")
    if len(scored) < n_clusters:
        raise ValueError("Not enough countries for the selected cluster count")

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = scored[SCORE_CODES].astype(float)
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(scaled)
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)

    result = scored.copy()
    result["cluster"] = [f"Profile {label + 1}" for label in labels]
    result["pca_x"] = projection[:, 0]
    result["pca_y"] = projection[:, 1]
    return result


def scenario_score(
    country_row: pd.Series,
    comparison: pd.DataFrame,
    renewable_change_pp: float,
    co2_reduction_pct: float,
    intensity_reduction_pct: float,
) -> tuple[float, float, dict]:
    """Recalculate the relative score after explicit user-defined improvements."""
    changed = {
        "EG.ELC.RNEW.ZS": float(country_row["EG.ELC.RNEW.ZS"]) + renewable_change_pp,
        "EN.ATM.CO2E.PC": float(country_row["EN.ATM.CO2E.PC"]) * (1 - co2_reduction_pct / 100),
        "EG.EGY.PRIM.PP.KD": float(country_row["EG.EGY.PRIM.PP.KD"])
        * (1 - intensity_reduction_pct / 100),
    }

    def calculate(values: dict[str, float]) -> float:
        total = 0.0
        for code in SCORE_CODES:
            minimum = float(comparison[code].min())
            maximum = float(comparison[code].max())
            component = 0.5 if maximum == minimum else (values[code] - minimum) / (maximum - minimum)
            component = float(np.clip(component, 0, 1))
            if INDICATORS[code]["direction"] == "lower":
                component = 1 - component
            total += component * float(INDICATORS[code]["weight"])
        return total * 100

    baseline_values = {code: float(country_row[code]) for code in SCORE_CODES}
    return calculate(baseline_values), calculate(changed), changed
