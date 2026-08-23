"""Relational GTFS contracts, graph construction, replay safety and lineage."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mbta_transit_graph_control.src.data import GTFSExtract, load_source


@dataclass(frozen=True)
class TransitGraphProduct:
    nodes: pd.DataFrame
    routes: pd.DataFrame
    patterns: pd.DataFrame
    edges: pd.DataFrame
    quarantine: pd.DataFrame
    quality: pd.DataFrame
    stages: pd.DataFrame
    metadata: dict


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1)
    if len(ordered): ordered = ordered.sort_values(list(ordered.columns), kind="stable", na_position="first")
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def _quarantine(table: str, frame: pd.DataFrame, reasons: pd.Series, identifier: pd.Series) -> pd.DataFrame:
    mask = reasons.ne("")
    return pd.DataFrame({"table":table,"record_id":identifier[mask].astype(str),"reason":reasons[mask].astype(str)})


def build_product(source: GTFSExtract, replay_edges: int = 20) -> TransitGraphProduct:
    started = time.perf_counter(); quarantine_parts = []
    stops = source.stops.copy(); stops["stop_lat"] = pd.to_numeric(stops.stop_lat, errors="coerce"); stops["stop_lon"] = pd.to_numeric(stops.stop_lon, errors="coerce")
    stop_reason = pd.Series("", index=stops.index, dtype="string"); stop_reason = stop_reason.mask(stops.stop_id.isna() | stops.stop_id.duplicated(keep=False), "invalid_or_duplicate_stop_id"); stop_reason = stop_reason.mask(stop_reason.eq("") & (~stops.stop_lat.between(41,44) | ~stops.stop_lon.between(-73,-69)), "coordinate_out_of_region")
    quarantine_parts.append(_quarantine("stops", stops, stop_reason, stops.stop_id)); stops = stops[stop_reason.eq("")].copy()

    routes = source.routes.copy(); routes["route_type"] = pd.to_numeric(routes.route_type, errors="coerce"); route_reason = pd.Series("", index=routes.index, dtype="string"); route_reason = route_reason.mask(routes.route_id.isna() | routes.route_id.duplicated(keep=False), "invalid_or_duplicate_route_id"); route_reason = route_reason.mask(route_reason.eq("") & (routes.route_type.isna() | (routes.route_type < 0)), "invalid_route_type")
    quarantine_parts.append(_quarantine("routes", routes, route_reason, routes.route_id)); routes = routes[route_reason.eq("")].copy(); routes["route_label"] = routes.route_short_name.fillna("").where(routes.route_short_name.fillna("").ne(""), routes.route_long_name).fillna(routes.route_id)

    patterns = source.patterns.copy(); pattern_reason = pd.Series("", index=patterns.index, dtype="string"); pattern_reason = pattern_reason.mask(patterns.route_pattern_id.isna() | patterns.route_pattern_id.duplicated(keep=False), "invalid_or_duplicate_pattern_id"); pattern_reason = pattern_reason.mask(pattern_reason.eq("") & ~patterns.route_id.isin(routes.route_id), "unknown_route"); pattern_reason = pattern_reason.mask(pattern_reason.eq("") & patterns.representative_trip_id.isna(), "missing_representative_trip")
    quarantine_parts.append(_quarantine("route_patterns", patterns, pattern_reason, patterns.route_pattern_id)); patterns = patterns[pattern_reason.eq("")].copy()

    times = source.stop_times.copy(); times["stop_sequence"] = pd.to_numeric(times.stop_sequence, errors="coerce"); valid_trips = set(patterns.representative_trip_id); valid_stops = set(stops.stop_id); time_reason = pd.Series("", index=times.index, dtype="string"); time_reason = time_reason.mask(~times.trip_id.isin(valid_trips), "unknown_representative_trip"); time_reason = time_reason.mask(time_reason.eq("") & ~times.stop_id.isin(valid_stops), "unknown_stop"); time_reason = time_reason.mask(time_reason.eq("") & (times.stop_sequence.isna() | (times.stop_sequence < 0)), "invalid_stop_sequence")
    quarantine_parts.append(_quarantine("stop_times", times, time_reason, times.trip_id.astype(str) + "|" + times.stop_id.astype(str))); times = times[time_reason.eq("")].copy()

    parent_map = stops.set_index("stop_id").parent_station.fillna(""); times["node_id"] = times.stop_id.map(parent_map).fillna(""); times["node_id"] = times.node_id.where(times.node_id.ne(""), times.stop_id)
    stop_node = stops.assign(node_id=lambda x: x.parent_station.fillna("").where(x.parent_station.fillna("").ne(""), x.stop_id)); canonical = stops.set_index("stop_id")
    rows = []
    for node_id, group in stop_node.groupby("node_id", sort=True):
        base = canonical.loc[node_id] if node_id in canonical.index else group.iloc[0]
        rows.append({"node_id":node_id,"stop_name":base.stop_name,"latitude":float(base.stop_lat),"longitude":float(base.stop_lon),"municipality":base.municipality,"wheelchair_boarding":base.wheelchair_boarding,"platforms":int(len(group))})
    nodes = pd.DataFrame(rows)

    linked = times.merge(patterns[["representative_trip_id","route_pattern_id","route_id"]], left_on="trip_id", right_on="representative_trip_id", how="inner")
    raw_edges = linked.groupby(["node_id","route_id"], as_index=False).agg(pattern_count=("route_pattern_id","nunique"), sequence_min=("stop_sequence","min"), sequence_max=("stop_sequence","max"))
    raw_edges["edge_id"] = [hashlib.sha256(f"{node}|{route}".encode()).hexdigest()[:24] for node, route in raw_edges[["node_id","route_id"]].itertuples(index=False, name=None)]
    replay = raw_edges.sort_values("edge_id").head(min(replay_edges, len(raw_edges))).copy(); delivered = pd.concat([raw_edges, replay], ignore_index=True); delivered["delivery_sequence"] = np.arange(len(delivered)); duplicate_mask = delivered.duplicated("edge_id", keep="first"); duplicates = int(duplicate_mask.sum()); edges = delivered[~duplicate_mask].drop(columns="delivery_sequence").sort_values(["node_id","route_id"]).reset_index(drop=True)

    quarantine = pd.concat(quarantine_parts, ignore_index=True); node_degree = edges.groupby("node_id").size(); route_degree = edges.groupby("route_id").size(); reconciliation = len(delivered) == len(edges) + duplicates
    gates = [("source_relations", len(source.stops)>=250 and len(source.routes)>=10 and len(source.stop_times)>=500, "required relational source volume"),("unique_entities", nodes.node_id.is_unique and routes.route_id.is_unique and patterns.route_pattern_id.is_unique, "node, route and pattern keys unique"),("foreign_keys", edges.node_id.isin(nodes.node_id).all() and edges.route_id.isin(routes.route_id).all(), "all Gold edge references resolve"),("edge_identity", edges.edge_id.is_unique, "stable edge IDs unique"),("replay_suppression", duplicates==min(replay_edges,len(raw_edges)), f"{duplicates} repeated edges suppressed"),("row_reconciliation", reconciliation, f"{len(delivered):,} edge deliveries reconciled"),("graph_volume", len(nodes)>=250 and len(routes)>=10 and len(edges)>=500, f"{len(nodes):,} nodes / {len(routes):,} routes / {len(edges):,} edges"),("link_prediction_population", int((node_degree>=2).sum())>=50, f"{int((node_degree>=2).sum()):,} multi-route nodes"),("coordinate_contract", nodes.latitude.between(41,44).all() and nodes.longitude.between(-73,-69).all(), "all published nodes within regional bounds"),("route_coverage", route_degree.min()>=1, f"all routes connect at least {int(route_degree.min())} node")]
    quality = pd.DataFrame(gates, columns=["check","passed","detail"])
    if not quality.passed.all(): raise RuntimeError("transit graph failed publication gates: " + ", ".join(quality.loc[~quality.passed,"check"]))
    hashes = {name:frame_hash(frame) for name,frame in {"nodes":nodes,"routes":routes,"patterns":patterns,"edges":edges}.items()}; run_id = hashlib.sha256((source.metadata["source_hash"]+"".join(hashes.values())).encode()).hexdigest()[:12]
    stages = pd.DataFrame([{"stage":"Extract","input":source.metadata["source_bytes"],"output":len(source.stops)+len(source.routes)+len(source.patterns)+len(source.stop_times),"rejected":0,"hash":source.metadata["source_hash"][:12]},{"stage":"Contract","input":len(source.stops)+len(source.routes)+len(source.patterns)+len(source.stop_times),"output":len(nodes)+len(routes)+len(patterns)+len(times),"rejected":len(quarantine),"hash":hashes["nodes"][:12]},{"stage":"Graph","input":len(times),"output":len(delivered),"rejected":0,"hash":hashes["edges"][:12]},{"stage":"Gold","input":len(delivered),"output":len(edges),"rejected":duplicates,"hash":hashes["edges"][:12]}])
    metadata = {**source.metadata, **{f"{key}_hash":value for key,value in hashes.items()},"run_id":run_id,"source_stops":len(source.stops),"source_routes":len(source.routes),"representative_patterns":len(patterns),"filtered_stop_times":len(times),"nodes":len(nodes),"routes":len(routes),"edges":len(edges),"multi_route_nodes":int((node_degree>=2).sum()),"replays":duplicates,"quarantined":len(quarantine),"duration_ms":round((time.perf_counter()-started)*1000,1)}
    return TransitGraphProduct(nodes.reset_index(drop=True), routes.reset_index(drop=True), patterns.reset_index(drop=True), edges, quarantine, quality, stages, metadata)


def run_pipeline(force_fallback: bool = False) -> TransitGraphProduct:
    return build_product(load_source(force_fallback=force_fallback))
