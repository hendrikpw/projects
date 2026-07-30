"""Open Food Facts ingestion, validation and deterministic fallback data."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


API_URL = "https://world.openfoodfacts.org/api/v2/search"
DOCS_URL = "https://openfoodfacts.github.io/openfoodfacts-server/api/"
TERMS_URL = "https://world.openfoodfacts.org/terms-of-use"
USER_AGENT = "HendrikDataPortfolio/1.0 (https://github.com/hendrikpw/projects)"

CATEGORIES = [
    "Breakfast cereals",
    "Yogurts",
    "Plant-based milks",
    "Protein bars",
    "Soft drinks",
    "Chocolate",
    "Frozen pizzas",
    "Crisps",
]
COUNTRIES = ["Germany", "France", "Spain", "Italy", "United Kingdom", "United States"]
NUTRIENT_FIELDS = [
    "energy_kcal_100g",
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
]


def _number(nutriments: dict, field: str) -> float:
    api_field = field.replace("energy_kcal", "energy-kcal").replace(
        "saturated_fat", "saturated-fat"
    )
    value = nutriments.get(api_field)
    if value is None:
        value = nutriments.get(field)
    return pd.to_numeric(value, errors="coerce")


def parse_products(products: list[dict], category: str, country: str, is_demo: bool) -> pd.DataFrame:
    """Flatten documented product and nutriment fields into one row per barcode."""
    rows = []
    for product in products:
        nutriments = product.get("nutriments") or {}
        rows.append(
            {
                "code": str(product.get("code") or "").strip(),
                "product_name": str(
                    product.get("product_name_en")
                    or product.get("product_name")
                    or "Unnamed product"
                ).strip(),
                "brands": str(product.get("brands") or "Unknown brand").strip(),
                "quantity": str(product.get("quantity") or "").strip(),
                "nutri_grade": str(product.get("nutrition_grades") or "").lower(),
                "nova_group": pd.to_numeric(product.get("nova_group"), errors="coerce"),
                "completeness": pd.to_numeric(product.get("completeness"), errors="coerce"),
                "ingredients_text": str(
                    product.get("ingredients_text_en")
                    or product.get("ingredients_text")
                    or ""
                ).strip(),
                "additives_n": pd.to_numeric(product.get("additives_n"), errors="coerce"),
                "additives": ", ".join(product.get("additives_tags") or []),
                "allergens": ", ".join(product.get("allergens_tags") or []),
                "labels": ", ".join(product.get("labels_tags") or []),
                "last_modified": pd.to_datetime(
                    product.get("last_modified_t"), unit="s", utc=True, errors="coerce"
                ),
                "category_query": category,
                "country_query": country,
                "is_demo": bool(is_demo),
                **{
                    field: _number(nutriments, field)
                    for field in NUTRIENT_FIELDS
                },
            }
        )
    return pd.DataFrame(rows)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent schema, range and duplicate rules."""
    required = {"code", "product_name", "brands", *NUTRIENT_FIELDS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Open Food Facts fields: {sorted(missing)}")
    result = frame.copy()
    result = result[result["code"].ne("") & result["product_name"].ne("")].copy()
    for field in NUTRIENT_FIELDS:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    result["completeness"] = pd.to_numeric(result["completeness"], errors="coerce").clip(0, 1)
    result["nova_group"] = pd.to_numeric(result["nova_group"], errors="coerce")
    result["additives_n"] = pd.to_numeric(result["additives_n"], errors="coerce")
    result = result[
        result["energy_kcal_100g"].isna()
        | result["energy_kcal_100g"].between(0, 1_000)
    ]
    for field in [item for item in NUTRIENT_FIELDS if item != "energy_kcal_100g"]:
        result.loc[~result[field].between(0, 100), field] = np.nan
    result["nutrition_coverage"] = result[NUTRIENT_FIELDS].notna().mean(axis=1) * 100
    result["nutri_grade"] = result["nutri_grade"].where(
        result["nutri_grade"].isin(list("abcde")), "unknown"
    )
    result = result.sort_values(
        ["nutrition_coverage", "completeness"], ascending=False
    ).drop_duplicates("code", keep="first")
    if result.empty:
        raise ValueError("No valid food products remained after validation")
    return result.reset_index(drop=True)


def fetch_products(
    category: str,
    country: str,
    page_size: int = 100,
    timeout: int = 35,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one bounded structured search from Open Food Facts."""
    fields = [
        "code",
        "product_name",
        "product_name_en",
        "brands",
        "quantity",
        "nutrition_grades",
        "nova_group",
        "nutriments",
        "ingredients_text",
        "ingredients_text_en",
        "additives_n",
        "additives_tags",
        "allergens_tags",
        "labels_tags",
        "completeness",
        "last_modified_t",
    ]
    response = requests.get(
        API_URL,
        params={
            "categories_tags_en": category,
            "countries_tags_en": country,
            "page_size": min(max(int(page_size), 10), 100),
            "fields": ",".join(fields),
            "sort_by": "unique_scans_n",
        },
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    products = payload.get("products") if isinstance(payload, dict) else None
    if not isinstance(products, list) or not products:
        raise ValueError("Open Food Facts returned no products")
    data = _prepare_frame(parse_products(products, category, country, False))
    return data, {
        "mode": "live",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "returned_products": len(data),
        "matching_products": int(payload.get("count") or len(data)),
        "source_url": response.url,
    }


def build_demo_data(category: str = "Breakfast cereals", country: str = "Germany") -> pd.DataFrame:
    """Generate stable category-shaped products for resilient UI demonstration."""
    seed = 20260730 + sum(ord(char) for char in f"{category}|{country}")
    rng = np.random.default_rng(seed)
    profiles = {
        "Breakfast cereals": (380, 12, 18, 9, 8),
        "Yogurts": (105, 4, 11, 4, 5),
        "Plant-based milks": (52, 2, 4, 1, 1),
        "Protein bars": (365, 20, 15, 8, 16),
        "Soft drinks": (42, 0, 10, 0, 0),
        "Chocolate": (535, 7, 48, 2, 30),
        "Frozen pizzas": (235, 10, 3, 3, 10),
        "Crisps": (520, 6, 2, 4, 32),
    }
    calories, protein, sugar, fiber, fat = profiles.get(category, profiles["Breakfast cereals"])
    products = []
    for index in range(140):
        product_protein = max(0, rng.normal(protein, max(protein * 0.35, 1)))
        product_sugar = max(0, rng.normal(sugar, max(sugar * 0.45, 1.5)))
        product_fiber = max(0, rng.normal(fiber, max(fiber * 0.4, 0.8)))
        product_fat = max(0, rng.normal(fat, max(fat * 0.3, 1)))
        salt = max(0, rng.lognormal(-1.0 if category not in {"Frozen pizzas", "Crisps"} else 0.2, 0.45))
        grade_score = product_sugar / 12 + salt * 1.8 + product_fat / 22 - product_fiber / 8 - product_protein / 25
        grade = "abcde"[int(np.clip(np.floor(grade_score + 2), 0, 4))]
        missing_probability = rng.uniform(0.01, 0.16)
        nutriments = {
            "energy-kcal_100g": max(0, rng.normal(calories, max(calories * 0.09, 8))),
            "fat_100g": product_fat,
            "saturated-fat_100g": product_fat * rng.uniform(0.2, 0.65),
            "carbohydrates_100g": max(0, rng.normal(55, 12)),
            "sugars_100g": product_sugar,
            "fiber_100g": product_fiber,
            "proteins_100g": product_protein,
            "salt_100g": salt,
        }
        for key in list(nutriments):
            if rng.random() < missing_probability:
                nutriments[key] = None
        products.append(
            {
                "code": f"demo-{seed}-{index:04d}",
                "product_name": f"Synthetic {category} #{index + 1}",
                "brands": f"Demo Brand {chr(65 + index % 12)}",
                "quantity": "Synthetic",
                "nutrition_grades": grade,
                "nova_group": int(rng.choice([1, 2, 3, 4], p=[0.12, 0.12, 0.2, 0.56])),
                "nutriments": nutriments,
                "ingredients_text": "Synthetic ingredients for interface demonstration only.",
                "additives_n": int(rng.integers(0, 5)),
                "additives_tags": [f"en:demo-additive-{n}" for n in range(int(rng.integers(0, 3)))],
                "allergens_tags": [],
                "labels_tags": ["en:synthetic-demo"],
                "completeness": float(rng.uniform(0.55, 1)),
                "last_modified_t": 1785398400 - index * 3_600,
            }
        )
    return _prepare_frame(parse_products(products, category, country, True))


def load_data(category: str, country: str) -> tuple[pd.DataFrame, dict]:
    """Return live product data or a clearly labelled synthetic fallback."""
    try:
        return fetch_products(category, country)
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        data = build_demo_data(category, country)
        return data, {
            "mode": "demo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "returned_products": len(data),
            "matching_products": len(data),
            "fallback_reason": type(exc).__name__,
        }
