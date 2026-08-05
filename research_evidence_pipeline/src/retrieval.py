"""Keyless semantic retrieval, evaluation and extractive evidence synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass
class RetrievalIndex:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD | None
    lexical_matrix: object
    semantic_matrix: np.ndarray
    documents: pd.DataFrame


def build_index(documents: pd.DataFrame, max_features: int = 8_000, dimensions: int = 64) -> RetrievalIndex:
    """Build deterministic sparse and latent-semantic representations."""
    if documents.empty or "document_text" not in documents:
        raise ValueError("AI-ready documents are required to build the retrieval index")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98 if len(documents) >= 20 else 1.0,
        max_features=max_features,
        sublinear_tf=True,
    )
    lexical = vectorizer.fit_transform(documents["document_text"].fillna(""))
    max_components = min(dimensions, lexical.shape[0] - 1, lexical.shape[1] - 1)
    if max_components >= 2:
        svd = TruncatedSVD(n_components=max_components, random_state=42)
        semantic = normalize(svd.fit_transform(lexical))
    else:
        svd = None
        semantic = normalize(lexical).toarray()
    return RetrievalIndex(vectorizer, svd, lexical, np.asarray(semantic), documents.reset_index(drop=True).copy())


def _query_vectors(index: RetrievalIndex, query: str) -> tuple[object, np.ndarray]:
    lexical = index.vectorizer.transform([query])
    if index.svd is None:
        semantic = normalize(lexical).toarray()
    else:
        semantic = normalize(index.svd.transform(lexical))
    return lexical, np.asarray(semantic)


def search(index: RetrievalIndex, query: str, top_k: int = 8, semantic_weight: float = 0.70) -> tuple[pd.DataFrame, dict]:
    """Rank publications with a visible lexical/latent-semantic hybrid score."""
    clean_query = re.sub(r"\s+", " ", str(query)).strip()
    if not clean_query:
        return pd.DataFrame(), {"query": "", "zero_vector": True, "top_score": 0.0, "confidence": "No query"}
    query_lexical, query_semantic = _query_vectors(index, clean_query)
    lexical_scores = cosine_similarity(query_lexical, index.lexical_matrix).ravel()
    semantic_scores = cosine_similarity(query_semantic, index.semantic_matrix).ravel()
    weight = float(np.clip(semantic_weight, 0, 1))
    hybrid = weight * semantic_scores + (1 - weight) * lexical_scores
    take = min(max(int(top_k), 1), len(index.documents))
    order = np.argsort(-hybrid)[:take]
    result = index.documents.iloc[order].copy()
    result["semantic_score"] = semantic_scores[order]
    result["lexical_score"] = lexical_scores[order]
    result["relevance_score"] = hybrid[order]
    result["rank"] = np.arange(1, len(result) + 1)
    result["relevance_percent"] = (result["relevance_score"].clip(lower=0) * 100).round(1)
    top_score = float(hybrid[order[0]]) if len(order) else 0.0
    confidence = "Strong match" if top_score >= 0.42 else "Moderate match" if top_score >= 0.20 else "Weak match"
    return result, {
        "query": clean_query,
        "zero_vector": query_lexical.nnz == 0,
        "top_score": top_score,
        "confidence": confidence,
        "semantic_weight": weight,
    }


def _first_sentences(text: str, limit: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", str(text)).strip())
    selected = [part for part in parts if len(part) >= 35][:limit]
    return " ".join(selected) if selected else str(text)[:420]


def evidence_brief(results: pd.DataFrame, query: str, max_sources: int = 3) -> dict:
    """Create a citation-bound extractive brief; no unsupported text is generated."""
    if results.empty:
        return {"headline": "No evidence retrieved", "findings": [], "sources": [], "caveat": "Broaden the question or ingest a larger corpus."}
    sources = []
    findings = []
    for citation, (_, row) in enumerate(results.head(max_sources).iterrows(), start=1):
        sources.append(
            {
                "citation": citation,
                "title": row["title"],
                "url": row["epmc_url"],
                "authors": row["authors"],
                "year": int(row["publication_year"]),
                "score": float(row["relevance_score"]),
            }
        )
        findings.append(f"[{citation}] {_first_sentences(row['abstract'], 2)}")
    return {
        "headline": f"Evidence retrieved for: {query}",
        "findings": findings,
        "sources": sources,
        "caveat": "Extractive summary of retrieved abstracts, not a clinical conclusion or systematic review.",
    }


def evaluate_retrieval(documents: pd.DataFrame, sample_size: int = 30) -> dict:
    """Evaluate title-to-abstract retrieval with deterministic self-document relevance labels."""
    eligible = documents[
        documents["title"].str.len().ge(12) & documents["abstract"].str.len().ge(120)
    ].head(max(int(sample_size), 1)).reset_index(drop=True)
    if len(eligible) < 4:
        return {"evaluated_queries": 0, "hit_rate_at_5": 0.0, "mrr_at_10": 0.0, "median_rank": np.nan, "zero_query_rate": 1.0}
    corpus = (eligible["abstract"] + " " + eligible["mesh_terms"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    query_matrix = vectorizer.transform(eligible["title"].tolist())
    similarities = cosine_similarity(query_matrix, matrix)
    ranks = []
    zero_queries = 0
    for index in range(len(eligible)):
        if query_matrix[index].nnz == 0:
            zero_queries += 1
            ranks.append(len(eligible) + 1)
            continue
        ordering = np.argsort(-similarities[index])
        ranks.append(int(np.where(ordering == index)[0][0]) + 1)
    ranks_array = np.asarray(ranks)
    reciprocal = np.where(ranks_array <= 10, 1 / ranks_array, 0)
    return {
        "evaluated_queries": len(eligible),
        "hit_rate_at_5": float(np.mean(ranks_array <= 5)),
        "mrr_at_10": float(np.mean(reciprocal)),
        "median_rank": float(np.median(ranks_array)),
        "zero_query_rate": float(zero_queries / len(eligible)),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
