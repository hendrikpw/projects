"""Collaborative recommendation, temporal holdout evaluation and serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class RecommenderBundle:
    svd: TruncatedSVD
    matrix: csr_matrix
    scores: np.ndarray
    train: pd.DataFrame
    holdout: pd.DataFrame
    evaluation: pd.DataFrame
    metrics: dict[str, float]
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_map: dict[int, int]
    movie_map: dict[int, int]
    popularity: np.ndarray
    metadata: dict[str, Any]


def _split(ratings: pd.DataFrame, min_ratings: int = 20, positive: float = 4.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = ratings.groupby("userId").size()
    eligible = counts[counts.ge(min_ratings)].index
    candidates = ratings[ratings["userId"].isin(eligible) & ratings["rating"].ge(positive)]
    if candidates["userId"].nunique() < 100:
        raise ValueError("fewer than 100 users have an eligible chronological positive holdout")
    hold_indices = candidates.sort_values(["userId", "rated_at", "movieId"]).groupby("userId").tail(1).index
    holdout = ratings.loc[hold_indices].sort_values("userId").copy()
    train = ratings.drop(index=hold_indices).copy()
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def _top_k(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(values))
    candidates = np.argpartition(values, -k)[-k:]
    return candidates[np.argsort(values[candidates])[::-1]]


def train_and_evaluate(ratings: pd.DataFrame, movies: pd.DataFrame, factors: int = 48, k: int = 20) -> RecommenderBundle:
    train, holdout = _split(ratings)
    user_ids = np.sort(ratings["userId"].astype(int).unique()); movie_ids = np.sort(movies["movieId"].astype(int).unique())
    user_map = {value:index for index,value in enumerate(user_ids)}; movie_map = {value:index for index,value in enumerate(movie_ids)}
    rows = train["userId"].astype(int).map(user_map).to_numpy(); cols = train["movieId"].astype(int).map(movie_map).to_numpy()
    weights = np.clip((train["rating"].to_numpy(float) - 2.5) / 2.5, 0, 1)
    matrix = csr_matrix((weights, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    factors = min(factors, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=factors, n_iter=9, random_state=42)
    user_factors = svd.fit_transform(matrix); scores = user_factors @ svd.components_
    positive_counts = train[train["rating"].ge(4)].groupby("movieId").size().reindex(movie_ids, fill_value=0).to_numpy(float)
    popularity = np.log1p(positive_counts)
    popularity_share = (positive_counts + 1) / (positive_counts.sum() + len(movie_ids))
    novelty = -np.log2(popularity_share)
    records=[]; model_catalog=set(); baseline_catalog=set()
    for hold in holdout.itertuples():
        user_index=user_map[int(hold.userId)]; target=movie_map[int(hold.movieId)]
        seen_ids=train.loc[train["userId"].eq(hold.userId),"movieId"].astype(int)
        seen=np.array([movie_map[item] for item in seen_ids if item in movie_map],dtype=int)
        model_values=scores[user_index].copy(); baseline_values=popularity.copy()
        model_values[seen]=-np.inf; baseline_values[seen]=-np.inf
        model_top=_top_k(model_values,k); baseline_top=_top_k(baseline_values,k)
        model_catalog.update(model_top.tolist()); baseline_catalog.update(baseline_top.tolist())
        model_positions=np.where(model_top==target)[0]; baseline_positions=np.where(baseline_top==target)[0]
        rank=int(model_positions[0]+1) if len(model_positions) else 0; baseline_rank=int(baseline_positions[0]+1) if len(baseline_positions) else 0
        records.append({
            "userId":int(hold.userId),"heldout_movieId":int(hold.movieId),"heldout_rating":float(hold.rating),
            "rank":rank,"hit":int(rank>0),"reciprocal_rank":1/rank if rank else 0,"ndcg":1/np.log2(rank+1) if rank else 0,
            "baseline_rank":baseline_rank,"baseline_hit":int(baseline_rank>0),"baseline_rr":1/baseline_rank if baseline_rank else 0,
            "mean_novelty_bits":float(novelty[model_top].mean()),"candidate_count":int(len(movie_ids)-len(seen)),
        })
    evaluation=pd.DataFrame(records)
    metrics={
        "hit_rate_at_k":float(evaluation["hit"].mean()),"mrr_at_k":float(evaluation["reciprocal_rank"].mean()),
        "ndcg_at_k":float(evaluation["ndcg"].mean()),"baseline_hit_rate_at_k":float(evaluation["baseline_hit"].mean()),
        "baseline_mrr_at_k":float(evaluation["baseline_rr"].mean()),
        "hit_lift_vs_popularity":float(evaluation["hit"].mean()-evaluation["baseline_hit"].mean()),
        "catalog_coverage":float(len(model_catalog)/len(movie_ids)),"baseline_catalog_coverage":float(len(baseline_catalog)/len(movie_ids)),
        "mean_novelty_bits":float(evaluation["mean_novelty_bits"].mean()),"explained_variance":float(svd.explained_variance_ratio_.sum()),
    }
    metadata={
        "model":"TruncatedSVD implicit latent-factor recommender","random_state":42,"factors":factors,"evaluation_k":k,
        "training_ratings":len(train),"holdout_users":len(holdout),"users":len(user_ids),"catalog_movies":len(movie_ids),
        "positive_threshold":4.0,"split":"last positive interaction per eligible user","candidate_policy":"full unseen catalog",
    }
    return RecommenderBundle(svd,matrix,scores,train,holdout,evaluation,metrics,user_ids,movie_ids,user_map,movie_map,popularity,metadata)


def _normalize(values: np.ndarray) -> np.ndarray:
    finite=values[np.isfinite(values)]
    if len(finite)==0 or finite.max()==finite.min(): return np.zeros_like(values)
    return (values-finite.min())/(finite.max()-finite.min())


def recommend_user(bundle: RecommenderBundle, movies: pd.DataFrame, user_id: int, limit: int = 10, novelty_weight: float = .25) -> pd.DataFrame:
    if user_id not in bundle.user_map: raise ValueError("unknown user; use cold_start_recommendations")
    user_index=bundle.user_map[user_id]; values=bundle.scores[user_index].copy()
    seen_ids=bundle.train.loc[bundle.train["userId"].eq(user_id),"movieId"].astype(int)
    seen=np.array([bundle.movie_map[item] for item in seen_ids if item in bundle.movie_map],dtype=int); values[seen]=-np.inf
    popularity_share=(np.expm1(bundle.popularity)+1)/(np.expm1(bundle.popularity).sum()+len(bundle.movie_ids))
    novelty=-np.log2(popularity_share); objective=(1-novelty_weight)*_normalize(values)+novelty_weight*_normalize(novelty)
    objective[seen]=-np.inf
    candidates=_top_k(objective,max(limit*8,80)); movie_lookup=movies.set_index("movieId")
    liked=bundle.train[(bundle.train["userId"].eq(user_id)) & (bundle.train["rating"].ge(4))].sort_values("rating",ascending=False)
    liked_genres=set("|".join(movie_lookup.loc[liked["movieId"],"genres"].astype(str)).split("|")) if len(liked) else set()
    selected=[]; used_genres=set()
    for candidate in candidates:
        movie_id=int(bundle.movie_ids[candidate]); genres=set(str(movie_lookup.loc[movie_id,"genres"]).split("|"))
        diversity_bonus=.08*len(genres-used_genres)/max(len(genres),1)
        selected.append((objective[candidate]+diversity_bonus,candidate,genres))
    selected=sorted(selected,key=lambda item:item[0],reverse=True)[:limit]
    rows=[]
    for rank,(_,index,genres) in enumerate(selected,1):
        movie_id=int(bundle.movie_ids[index]); shared=sorted((genres & liked_genres)-{"(no genres listed)"})
        rows.append({"rank":rank,"movieId":movie_id,"title":movie_lookup.loc[movie_id,"title"],"genres":movie_lookup.loc[movie_id,"genres"],
                     "model_score":float(values[index]),"novelty_bits":float(novelty[index]),"reason":f"Matches {', '.join(shared[:3])}" if shared else "Latent taste-neighbor signal"})
    return pd.DataFrame(rows)


def cold_start_recommendations(bundle: RecommenderBundle, movies: pd.DataFrame, genres: list[str], limit: int = 10) -> pd.DataFrame:
    frame=movies.copy(); frame["genre_match"]=frame["genres"].map(lambda value: len(set(str(value).split("|")) & set(genres)))
    frame["popularity"]=frame["movieId"].astype(int).map(lambda value: bundle.popularity[bundle.movie_map[value]] if value in bundle.movie_map else 0)
    result=frame[frame["genre_match"].gt(0)].sort_values(["genre_match","popularity"],ascending=False).head(limit).copy()
    result.insert(0,"rank",range(1,len(result)+1)); result["reason"]=result["genre_match"].map(lambda count:f"Matches {count} selected genre(s)")
    return result[["rank","movieId","title","genres","popularity","reason"]]
