from __future__ import annotations

import numpy as np
import pytest

from movielens_recommendation_pipeline.src.data import _fallback
from movielens_recommendation_pipeline.src.model import cold_start_recommendations, recommend_user, train_and_evaluate
from movielens_recommendation_pipeline.src.pipeline import _silver


def inputs():
    silver,_=_silver(_fallback()); return silver["ratings"],silver["movies"]


def test_model_metrics_and_shapes():
    ratings,movies=inputs(); bundle=train_and_evaluate(ratings,movies,factors=24,k=20)
    assert bundle.scores.shape==(bundle.metadata["users"],bundle.metadata["catalog_movies"])
    assert len(bundle.evaluation)==bundle.metadata["holdout_users"]
    for key in ["hit_rate_at_k","mrr_at_k","ndcg_at_k","catalog_coverage","explained_variance"]:
        assert 0<=bundle.metrics[key]<=1


def test_model_is_reproducible():
    ratings,movies=inputs(); first=train_and_evaluate(ratings,movies,24,20); second=train_and_evaluate(ratings,movies,24,20)
    np.testing.assert_allclose(first.scores,second.scores)
    assert first.metrics==second.metrics


def test_recommendations_exclude_seen_items():
    ratings,movies=inputs(); bundle=train_and_evaluate(ratings,movies,24,20); user=int(bundle.holdout.iloc[0]["userId"])
    recs=recommend_user(bundle,movies,user,10,.25); seen=set(bundle.train.loc[bundle.train["userId"].eq(user),"movieId"].astype(int))
    assert len(recs)==10 and not set(recs["movieId"]).intersection(seen)
    assert recs["movieId"].is_unique


def test_cold_start_output_and_unknown_user_guard():
    ratings,movies=inputs(); bundle=train_and_evaluate(ratings,movies,24,20)
    recs=cold_start_recommendations(bundle,movies,["Action","Comedy"],8)
    assert len(recs)==8
    assert recs["movieId"].is_unique
    with pytest.raises(ValueError,match="unknown user"): recommend_user(bundle,movies,999999)


def test_small_data_failure():
    ratings,movies=inputs()
    with pytest.raises(ValueError,match="fewer than 100"):
        train_and_evaluate(ratings[ratings["userId"].le(20)],movies,12,10)
