"""Evaluated character-level supplier entity resolution with abstention."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LEGAL = re.compile(r"\b(LLC|L L C|INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LTD|LIMITED)\b")


@dataclass(frozen=True)
class ResolutionModel:
    vectorizer: TfidfVectorizer
    matrix: object
    reference: pd.DataFrame
    evaluation: pd.DataFrame
    metrics: dict
    thresholds: dict
    drift: pd.DataFrame


def normalize_name(value: str) -> str:
    text = re.sub(r"[^A-Z0-9 ]+", " ", str(value).upper())
    return re.sub(r"\s+", " ", text).strip()


def _representation(value: str) -> str:
    """Expose original, token-order-invariant and acronym views to one index."""
    clean = normalize_name(value)
    tokens = clean.split()
    return " | ".join([clean, " ".join(sorted(tokens)), "".join(token[0] for token in tokens)])


def corrupt_name(name: str, variant: int) -> str:
    """Deterministically emulate common upstream spelling/format defects."""
    value = normalize_name(name)
    if variant == 0:
        return re.sub(r"\s+", "", value).lower()
    if variant == 1:
        return re.sub(r"\s+", " ", LEGAL.sub("", value)).strip().title()
    if variant == 2:
        tokens = value.split()
        if len(tokens) > 1:
            tokens[0], tokens[1] = tokens[1], tokens[0]
        return " ".join(tokens)
    if variant == 3:
        vowels = [i for i, ch in enumerate(value) if ch in "AEIOU"]
        return value[: vowels[len(vowels) // 2]] + value[vowels[len(vowels) // 2] + 1 :] if vowels else value[:-1]
    tokens = value.split()
    return f"{tokens[0]} {''.join(token[0] for token in tokens[1:])}" if len(tokens) > 1 else value


def _score_queries(vectorizer, matrix, queries: list[str], top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    q = vectorizer.transform([_representation(item) for item in queries])
    similarities = (q @ matrix.T).toarray()
    order = np.argsort(-similarities, axis=1)[:, :top_k]
    scores = np.take_along_axis(similarities, order, axis=1)
    return order, scores


def _split(uei: str) -> str:
    return "calibration" if int(hashlib.sha256(uei.encode()).hexdigest()[:8], 16) % 5 < 2 else "test"


def _select_threshold(calibration: pd.DataFrame, minimum_accuracy: float = .95) -> tuple[float, float]:
    # A non-trivial floor also protects against zero-vector and unrelated-name links.
    best = (.40, .025, -1.0)
    for score_threshold in np.arange(.40, .86, .05):
        for margin_threshold in np.arange(.025, .21, .025):
            accepted = calibration[(calibration.score >= score_threshold) & (calibration.margin >= margin_threshold)]
            if len(accepted) == 0:
                continue
            accuracy = accepted.correct.mean()
            coverage = len(accepted) / len(calibration)
            if accuracy >= minimum_accuracy and coverage > best[2]:
                best = (float(score_threshold), float(margin_threshold), float(coverage))
    return best[0], best[1]


def train_and_evaluate(recipients: pd.DataFrame) -> ResolutionModel:
    reference = recipients[["recipient_uei", "canonical_name", "award_count", "total_award_value"]].drop_duplicates("recipient_uei").sort_values("recipient_uei").reset_index(drop=True)
    if len(reference) < 25:
        raise RuntimeError("at least 25 contracted recipient identities are required")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform(reference["canonical_name"].map(_representation))
    examples = []
    for row in reference.itertuples(index=False):
        for variant in range(5):
            examples.append({"recipient_uei": row.recipient_uei, "canonical_name": row.canonical_name, "query_name": corrupt_name(row.canonical_name, variant), "corruption": ["spacing_case", "legal_suffix", "token_order", "character_loss", "abbreviation"][variant], "split": _split(row.recipient_uei)})
    evaluation = pd.DataFrame(examples)
    order, scores = _score_queries(vectorizer, matrix, evaluation.query_name.tolist())
    evaluation["predicted_uei"] = reference.iloc[order[:, 0]].recipient_uei.to_numpy()
    evaluation["predicted_name"] = reference.iloc[order[:, 0]].canonical_name.to_numpy()
    evaluation["score"] = scores[:, 0]
    evaluation["margin"] = scores[:, 0] - np.where(scores.shape[1] > 1, scores[:, 1], 0)
    evaluation["rank"] = [next((rank + 1 for rank, idx in enumerate(indices) if reference.iloc[idx].recipient_uei == truth), 6) for indices, truth in zip(order, evaluation.recipient_uei)]
    evaluation["correct"] = evaluation.predicted_uei == evaluation.recipient_uei
    evaluation["exact_baseline"] = evaluation.query_name.map(normalize_name) == evaluation.canonical_name.map(normalize_name)
    calibration = evaluation[evaluation.split == "calibration"]
    score_threshold, margin_threshold = _select_threshold(calibration)
    evaluation["accepted"] = (evaluation.score >= score_threshold) & (evaluation.margin >= margin_threshold)
    evaluation["route"] = np.where(evaluation.accepted, "auto-link", "human-review")
    test = evaluation[evaluation.split == "test"].copy()
    accepted = test[test.accepted]
    negative_queries = ["UNRELATED SUPPLIER ZXQ 999", "UNKNOWN VENDOR PLACEHOLDER", "N A", "ACME TEMP RECORD"]
    _, negative_scores = _score_queries(vectorizer, matrix, negative_queries)
    negative_margins = negative_scores[:, 0] - negative_scores[:, 1]
    rejected_negative = (negative_scores[:, 0] < score_threshold) | (negative_margins < margin_threshold)
    metrics = {
        "test_queries": int(len(test)), "top1_accuracy": float(test.correct.mean()), "hit_at_5": float((test["rank"] <= 5).mean()),
        "mrr_at_5": float(np.where(test["rank"] <= 5, 1 / test["rank"], 0).mean()), "exact_baseline_accuracy": float(test.exact_baseline.mean()),
        "coverage": float(test.accepted.mean()), "selective_accuracy": float(accepted.correct.mean()) if len(accepted) else 0.0,
        "false_merge_rate": float((~accepted.correct).sum() / max(1, len(test))), "unknown_rejection_rate": float(rejected_negative.mean()),
    }
    if metrics["top1_accuracy"] < metrics["exact_baseline_accuracy"] or metrics["hit_at_5"] < .90 or metrics["selective_accuracy"] < .90 or metrics["unknown_rejection_rate"] < .75:
        raise RuntimeError(f"entity-resolution evaluation failed promotion gates: {metrics}")
    drift_rows = []
    for corruption, group in test.groupby("corruption"):
        drift_rows.append({"corruption": corruption, "queries": len(group), "top1_accuracy": group.correct.mean(), "coverage": group.accepted.mean(), "mean_score": group.score.mean()})
    return ResolutionModel(vectorizer, matrix, reference, evaluation, metrics, {"score": score_threshold, "margin": margin_threshold}, pd.DataFrame(drift_rows))


def resolve_name(model: ResolutionModel, query: str, top_k: int = 5) -> pd.DataFrame:
    if not normalize_name(query):
        return pd.DataFrame(columns=["recipient_uei", "canonical_name", "score", "margin", "route"])
    order, scores = _score_queries(model.vectorizer, model.matrix, [query], min(top_k, len(model.reference)))
    result = model.reference.iloc[order[0]].copy().reset_index(drop=True)
    result["score"] = scores[0]
    result["margin"] = scores[0] - (scores[0, 1] if len(scores[0]) > 1 else 0)
    result["route"] = "candidate"
    accepted = scores[0, 0] >= model.thresholds["score"] and (scores[0, 0] - scores[0, 1] if len(scores[0]) > 1 else scores[0, 0]) >= model.thresholds["margin"]
    result.loc[0, "route"] = "auto-link" if accepted else "human-review"
    return result
