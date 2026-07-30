"""Transparent product comparison, similarity and data-quality analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_FEATURES = {
    "Lower sugar": ("sugars_100g", False),
    "Lower salt": ("salt_100g", False),
    "Lower saturated fat": ("saturated_fat_100g", False),
    "Higher fibre": ("fiber_100g", True),
    "Higher protein": ("proteins_100g", True),
}
SIMILARITY_FEATURES = [
    "energy_kcal_100g",
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
]


def filter_products(
    data: pd.DataFrame,
    minimum_coverage: int = 60,
    nutri_grades: list[str] | None = None,
) -> pd.DataFrame:
    """Filter by declared data coverage and optional official Nutri-Score grades."""
    result = data[data["nutrition_coverage"] >= int(minimum_coverage)].copy()
    if nutri_grades:
        result = result[result["nutri_grade"].isin(nutri_grades)]
    return result.reset_index(drop=True)


def choice_fit_score(
    data: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Create a within-sample percentile score from visible user priorities."""
    result = data.copy()
    total_weight = sum(max(float(value), 0) for value in weights.values())
    if result.empty or total_weight <= 0:
        result["choice_fit_score"] = np.nan
        return result
    score = pd.Series(0.0, index=result.index)
    used_weight = pd.Series(0.0, index=result.index)
    for label, raw_weight in weights.items():
        weight = max(float(raw_weight), 0)
        if weight <= 0 or label not in SCORE_FEATURES:
            continue
        field, higher_is_better = SCORE_FEATURES[label]
        values = pd.to_numeric(result[field], errors="coerce")
        percentile = values.rank(pct=True, method="average") * 100
        if not higher_is_better:
            percentile = 100 - percentile
        available = values.notna()
        score.loc[available] += percentile.loc[available] * weight
        used_weight.loc[available] += weight
    result["choice_fit_score"] = (score / used_weight.replace(0, np.nan)).round(1)
    return result


def summary_metrics(data: pd.DataFrame) -> dict:
    """Return headline product, label and data-quality metrics."""
    grade_numeric = data["nutri_grade"].map({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    median_grade_value = grade_numeric.median()
    median_grade = (
        {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}.get(int(round(median_grade_value)))
        if pd.notna(median_grade_value)
        else "n/a"
    )
    return {
        "products": len(data),
        "brands": int(data["brands"].nunique()),
        "median_grade": median_grade,
        "median_sugar": float(data["sugars_100g"].median()),
        "nova_4_share": float(data["nova_group"].eq(4).mean() * 100),
        "median_coverage": float(data["nutrition_coverage"].median()),
    }


def similar_products(data: pd.DataFrame, code: str, limit: int = 8) -> pd.DataFrame:
    """Find nutritionally similar products using median-imputed z-score distance."""
    if code not in set(data["code"]) or len(data) < 2:
        return pd.DataFrame()
    matrix = data[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_columns = matrix.columns[matrix.notna().sum() >= 2]
    matrix = matrix[valid_columns]
    if matrix.empty:
        return pd.DataFrame()
    matrix = matrix.fillna(matrix.median())
    scale = matrix.std(ddof=0).replace(0, 1)
    standardized = (matrix - matrix.mean()) / scale
    selected_index = data.index[data["code"].eq(code)][0]
    distances = np.sqrt(
        ((standardized - standardized.loc[selected_index]) ** 2).mean(axis=1)
    )
    result = data.copy()
    result["nutrition_distance"] = distances
    return (
        result[~result["code"].eq(code)]
        .sort_values(["nutrition_distance", "choice_fit_score"], ascending=[True, False])
        .head(int(limit))
        .reset_index(drop=True)
    )


def brand_summary(data: pd.DataFrame, minimum_products: int = 2) -> pd.DataFrame:
    """Aggregate brand-level product choice and data coverage signals."""
    result = (
        data.groupby("brands", as_index=False)
        .agg(
            products=("code", "nunique"),
            median_fit=("choice_fit_score", "median"),
            median_sugar=("sugars_100g", "median"),
            median_protein=("proteins_100g", "median"),
            nova_4_share=("nova_group", lambda values: values.eq(4).mean() * 100),
            median_coverage=("nutrition_coverage", "median"),
        )
    )
    return result[result["products"] >= int(minimum_products)].sort_values(
        ["median_fit", "products"], ascending=False
    )


def missingness_report(data: pd.DataFrame) -> pd.DataFrame:
    """Quantify field availability instead of silently hiding missing labels."""
    labels = {
        "energy_kcal_100g": "Energy",
        "fat_100g": "Fat",
        "saturated_fat_100g": "Saturated fat",
        "carbohydrates_100g": "Carbohydrates",
        "sugars_100g": "Sugars",
        "fiber_100g": "Fibre",
        "proteins_100g": "Protein",
        "salt_100g": "Salt",
        "nutri_grade": "Nutri-Score",
        "nova_group": "NOVA group",
        "ingredients_text": "Ingredients",
    }
    rows = []
    for field, label in labels.items():
        values = data[field]
        available = values.notna()
        if values.dtype == object:
            available &= values.astype(str).str.strip().ne("")
            if field == "nutri_grade":
                available &= values.ne("unknown")
        rows.append(
            {
                "field": label,
                "available_products": int(available.sum()),
                "coverage": float(available.mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values("coverage")
