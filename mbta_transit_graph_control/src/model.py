"""Leakage-safe graph embeddings, masked-link evaluation and review routing."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from mbta_transit_graph_control.src.pipeline import TransitGraphProduct


@dataclass(frozen=True)
class GraphModel:
    svd: TruncatedSVD
    matrix: sparse.csr_matrix
    masked_matrix: sparse.csr_matrix
    similarity: np.ndarray
    geo_score: np.ndarray
    nodes: list[str]
    routes: list[str]
    node_index: dict[str, int]
    route_index: dict[str, int]
    weights: tuple[float, float, float]
    threshold: float
    evaluation: pd.DataFrame
    metrics: dict
    robustness: pd.DataFrame
    route_metrics: pd.DataFrame
    embeddings: pd.DataFrame
    metadata: dict


def _stable(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16)


def _minmax(values: np.ndarray) -> np.ndarray:
    low = np.nanmin(values, axis=1, keepdims=True); high = np.nanmax(values, axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-9)


def _score(matrix: sparse.csr_matrix, svd: TruncatedSVD, similarity: np.ndarray, geo: np.ndarray, weights: tuple[float,float,float]) -> np.ndarray:
    topology = np.asarray(matrix @ similarity); latent = svd.transform(matrix) @ svd.components_
    return weights[0]*_minmax(topology) + weights[1]*_minmax(latent) + weights[2]*_minmax(geo)


def _rank_rows(scores: np.ndarray, rows: np.ndarray, truth: np.ndarray, visible: sparse.csr_matrix, split: str) -> pd.DataFrame:
    result = []
    for row, actual in zip(rows, truth):
        values = scores[row].copy(); values[visible.getrow(row).indices] = -np.inf; order = np.argsort(-values); position = int(np.where(order == actual)[0][0]) + 1; margin = float(values[order[0]] - values[order[1]])
        result.append({"row":int(row),"actual_route_index":int(actual),"predicted_route_index":int(order[0]),"rank":position,"hit_at_5":position<=5,"reciprocal_rank":1/position,"ndcg_at_5":1/np.log2(position+1) if position<=5 else 0.0,"margin":margin,"split":split})
    return pd.DataFrame(result)


def _metrics(frame: pd.DataFrame) -> dict:
    return {"hit_at_1":float((frame["rank"]==1).mean()),"hit_at_5":float(frame.hit_at_5.mean()),"mrr":float(frame.reciprocal_rank.mean()),"ndcg_at_5":float(frame.ndcg_at_5.mean())}


def train_and_evaluate(product: TransitGraphProduct) -> GraphModel:
    nodes = sorted(product.edges.node_id.unique()); routes = sorted(product.edges.route_id.unique()); node_index = {value:i for i,value in enumerate(nodes)}; route_index = {value:i for i,value in enumerate(routes)}
    row = product.edges.node_id.map(node_index).to_numpy(); col = product.edges.route_id.map(route_index).to_numpy(); matrix = sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(len(nodes),len(routes)))
    eligible = np.flatnonzero(np.asarray(matrix.sum(axis=1)).ravel() >= 2); holds = []
    for index in eligible:
        candidates = matrix.getrow(index).indices; held = max(candidates, key=lambda route:_stable(f"{nodes[index]}|{routes[route]}")); split = "calibration" if _stable(nodes[index]) % 5 < 2 else "test"; holds.append((index,held,split))
    holdout = pd.DataFrame(holds, columns=["row","route","split"]); masked = matrix.tolil(copy=True)
    for item in holdout.itertuples(index=False): masked[item.row,item.route] = 0
    masked = masked.tocsr(); masked.eliminate_zeros()
    co = (masked.T @ masked).toarray(); degree = np.sqrt(np.maximum(np.diag(co),1)); similarity = co / np.outer(degree,degree); np.fill_diagonal(similarity,0)
    node_frame = product.nodes.set_index("node_id").reindex(nodes); coords = node_frame[["latitude","longitude"]].to_numpy(float); centroids = np.zeros((len(routes),2))
    for route, index in route_index.items():
        attached = masked[:,index].nonzero()[0]; centroids[index] = coords[attached].mean(axis=0) if len(attached) else coords.mean(axis=0)
    lat_scale = np.cos(np.deg2rad(coords[:,0].mean())); delta = coords[:,None,:] - centroids[None,:,:]; distance = np.sqrt(delta[:,:,0]**2 + (delta[:,:,1]*lat_scale)**2); geo = -distance
    cal = holdout[holdout.split=="calibration"]; candidates = []
    started = time.perf_counter()
    for factors in [8,16,24,32]:
        if factors >= min(masked.shape): continue
        svd = TruncatedSVD(n_components=factors, random_state=42, n_iter=7).fit(masked)
        for weights in [(1,0,0),(.5,.5,0),(.4,.4,.2),(.25,.5,.25),(0,1,0)]:
            scores = _score(masked,svd,similarity,geo,weights); ranked = _rank_rows(scores,cal.row.to_numpy(),cal.route.to_numpy(),masked,"calibration"); metric = _metrics(ranked); candidates.append((metric["hit_at_5"],metric["mrr"],factors,weights,svd,scores))
    best = max(candidates,key=lambda x:(x[0],x[1],-x[2])); _,_,factors,weights,svd,scores = best
    test = holdout[holdout.split=="test"]; cal_eval = _rank_rows(scores,cal.row.to_numpy(),cal.route.to_numpy(),masked,"calibration"); test_eval = _rank_rows(scores,test.row.to_numpy(),test.route.to_numpy(),masked,"test")
    popularity = np.asarray(masked.sum(axis=0)).ravel(); baseline_scores = np.tile(popularity,(len(nodes),1)); baseline_eval = _rank_rows(baseline_scores,test.row.to_numpy(),test.route.to_numpy(),masked,"test")
    threshold = float(cal_eval.margin.quantile(.85)); auto = test_eval.margin >= threshold; test_metrics = _metrics(test_eval); baseline_metrics = _metrics(baseline_eval)
    test_metrics.update({"baseline_hit_at_5":baseline_metrics["hit_at_5"],"baseline_mrr":baseline_metrics["mrr"],"coverage":float(auto.mean()),"selective_top1_accuracy":float((test_eval.loc[auto,"rank"]==1).mean()) if auto.any() else 0.0,"test_edges":len(test_eval),"eligible_nodes":len(eligible),"fit_seconds":round(time.perf_counter()-started,2)})
    if test_metrics["hit_at_5"] <= test_metrics["baseline_hit_at_5"] + .05 or test_metrics["hit_at_5"] < .25: raise RuntimeError(f"graph model failed promotion gates: {test_metrics}")
    evaluation = pd.concat([cal_eval,test_eval],ignore_index=True); evaluation["node_id"] = evaluation.row.map(dict(enumerate(nodes))); evaluation["actual_route"] = evaluation.actual_route_index.map(dict(enumerate(routes))); evaluation["predicted_route"] = evaluation.predicted_route_index.map(dict(enumerate(routes))); evaluation["route"] = np.where(evaluation.margin>=threshold,"auto-candidate","human-review")
    robust_rows = []
    rng = np.random.default_rng(42)
    for dropout in [0,.1,.2,.3,.4]:
        damaged = masked.copy().tolil(); nz = list(zip(*damaged.nonzero())); remove = rng.choice(len(nz),int(len(nz)*dropout),replace=False) if dropout else []
        for position in remove: damaged[nz[position][0],nz[position][1]]=0
        damaged = damaged.tocsr(); damaged.eliminate_zeros(); damaged_scores = _score(damaged,svd,similarity,geo,weights); ranked = _rank_rows(damaged_scores,test.row.to_numpy(),test.route.to_numpy(),damaged,"test"); robust_rows.append({"edge_dropout":dropout,**_metrics(ranked)})
    route_table = product.routes.set_index("route_id"); route_rows = []
    for route,index in route_index.items():
        subset=test_eval[test_eval.actual_route_index==index]; route_rows.append({"route_id":route,"route_label":route_table.loc[route,"route_label"],"route_type":int(route_table.loc[route,"route_type"]),"held_edges":len(subset),"hit_at_5":float(subset.hit_at_5.mean()) if len(subset) else np.nan,"graph_degree":int(matrix[:,index].sum())})
    embedding = svd.components_.T; emb = pd.DataFrame({"route_id":routes,"x":embedding[:,0],"y":embedding[:,1] if embedding.shape[1]>1 else 0,"graph_degree":np.asarray(matrix.sum(axis=0)).ravel()}); emb = emb.merge(product.routes[["route_id","route_label","route_type"]],on="route_id",how="left")
    metadata = {"factors":factors,"weights":{"topology":weights[0],"embedding":weights[1],"geography":weights[2]},"confidence_margin":threshold,"calibration_edges":len(cal_eval),"test_edges":len(test_eval),"nodes":len(nodes),"routes":len(routes),"visible_edges":int(masked.nnz)}
    return GraphModel(svd,matrix,masked,similarity,geo,nodes,routes,node_index,route_index,weights,threshold,evaluation,test_metrics,pd.DataFrame(robust_rows),pd.DataFrame(route_rows),emb,metadata)


def score_node(model: GraphModel, node_id: str, hide_route: str | None = None) -> dict:
    if node_id not in model.node_index: raise ValueError("unknown node")
    index=model.node_index[node_id]; row=model.matrix.getrow(index).tolil(copy=True); hidden=None
    if hide_route is not None:
        if hide_route not in model.route_index or model.matrix[index,model.route_index[hide_route]]==0: raise ValueError("hidden route is not attached to this node")
        hidden=model.route_index[hide_route]; row[0,hidden]=0
    row=row.tocsr(); topology=np.asarray(row @ model.similarity); latent=model.svd.transform(row) @ model.svd.components_; geo=model.geo_score[index:index+1]; values=(model.weights[0]*_minmax(topology)+model.weights[1]*_minmax(latent)+model.weights[2]*_minmax(geo))[0]; values[row.indices]=-np.inf; order=np.argsort(-values); margin=float(values[order[0]]-values[order[1]])
    ranking=pd.DataFrame({"route_id":[model.routes[i] for i in order[:10]],"score":values[order[:10]],"rank":range(1,11)}); recovered_rank=int(np.where(order==hidden)[0][0])+1 if hidden is not None else None
    return {"ranking":ranking,"margin":margin,"route":"auto-candidate" if margin>=model.threshold else "human-review","hidden_route":hide_route,"recovered_rank":recovered_rank,"hit_at_5":recovered_rank<=5 if recovered_rank is not None else None}
