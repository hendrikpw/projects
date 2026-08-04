"""Cycle-hire service levels, spatial pressure and rebalancing analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6371.0088


def add_service_features(
    stations: pd.DataFrame,
    critical_threshold: float = 0.15,
    target_fill: float = 0.50,
) -> pd.DataFrame:
    """Create availability, service-level, imbalance and quality fields."""
    frame = stations.copy()
    threshold = float(np.clip(critical_threshold, 0.02, 0.45))
    target = float(np.clip(target_fill, threshold, 1 - threshold))
    frame["operational"] = frame["installed"].fillna(True) & ~frame["locked"].fillna(False)
    frame["fill_ratio"] = frame["bikes"].div(frame["docks"].replace(0, np.nan)).clip(0, 1)
    frame["fill_percent"] = frame["fill_ratio"] * 100
    frame["effective_capacity"] = (frame["docks"] - frame["unavailable_docks"]).clip(lower=0)
    frame["desired_bikes"] = np.rint(frame["effective_capacity"] * target).astype(int)
    frame["bike_deficit"] = (frame["desired_bikes"] - frame["bikes"]).clip(lower=0).astype(int)
    frame["bike_surplus"] = (frame["bikes"] - frame["desired_bikes"]).clip(lower=0).astype(int)
    frame["imbalance_bikes"] = frame["bikes"] - frame["desired_bikes"]
    frame["service_status"] = np.select(
        [
            ~frame["operational"],
            frame["fill_ratio"].le(threshold),
            frame["fill_ratio"].ge(1 - threshold),
        ],
        ["Unavailable", "Empty risk", "Full risk"],
        default="Balanced",
    )
    frame["pressure_score"] = (
        (frame["fill_ratio"] - target).abs().div(max(target, 1 - target)).clip(0, 1) * 100
    )
    frame.loc[~frame["operational"], "pressure_score"] = 100
    frame["quality_score"] = 100
    frame.loc[frame["capacity_inconsistent"], "quality_score"] -= 45
    frame.loc[frame["bike_type_inconsistent"], "quality_score"] -= 25
    frame.loc[frame["station_updated_at"].isna(), "quality_score"] -= 20
    frame.loc[frame["unavailable_docks"].gt(0), "quality_score"] -= 10
    frame["quality_score"] = frame["quality_score"].clip(0, 100)
    return frame


def network_metrics(stations: pd.DataFrame) -> dict:
    """Summarize the current operational network snapshot."""
    operational = stations[stations["operational"]]
    total_bikes = int(operational["bikes"].sum()) if not operational.empty else 0
    total_docks = int(operational["docks"].sum()) if not operational.empty else 0
    return {
        "stations": int(len(stations)),
        "operational_stations": int(len(operational)),
        "bikes": total_bikes,
        "ebikes": int(operational["ebikes"].sum()) if not operational.empty else 0,
        "empty_docks": int(operational["empty_docks"].sum()) if not operational.empty else 0,
        "empty_risk": int(operational["service_status"].eq("Empty risk").sum()),
        "full_risk": int(operational["service_status"].eq("Full risk").sum()),
        "balanced_share": float(operational["service_status"].eq("Balanced").mean() * 100) if not operational.empty else 0.0,
        "network_fill": float(total_bikes / total_docks * 100) if total_docks else 0.0,
        "unavailable_docks": int(stations["unavailable_docks"].sum()),
    }


def pressure_clusters(
    stations: pd.DataFrame,
    radius_km: float = 0.65,
    min_stations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster nearby empty-risk and full-risk stations separately with haversine DBSCAN."""
    frame = stations.copy()
    frame["cluster_id"] = -1
    frame["cluster_label"] = "No pressure cluster"
    summary_rows = []
    next_cluster = 0
    for status in ["Empty risk", "Full risk"]:
        subset = frame[frame["service_status"].eq(status)]
        if len(subset) < max(int(min_stations), 2):
            continue
        coordinates = np.radians(subset[["latitude", "longitude"]].to_numpy())
        labels = DBSCAN(
            eps=float(radius_km) / EARTH_RADIUS_KM,
            min_samples=max(int(min_stations), 2),
            metric="haversine",
        ).fit_predict(coordinates)
        for local_label in sorted(set(labels) - {-1}):
            indexes = subset.index[labels == local_label]
            global_label = next_cluster
            next_cluster += 1
            frame.loc[indexes, "cluster_id"] = global_label
            frame.loc[indexes, "cluster_label"] = f"{status} cluster {global_label + 1}"
            group = frame.loc[indexes]
            summary_rows.append(
                {
                    "cluster_id": global_label,
                    "cluster_label": f"{status} cluster {global_label + 1}",
                    "pressure_type": status,
                    "stations": len(group),
                    "center_latitude": float(group["latitude"].mean()),
                    "center_longitude": float(group["longitude"].mean()),
                    "bikes": int(group["bikes"].sum()),
                    "empty_docks": int(group["empty_docks"].sum()),
                    "mean_fill_percent": float(group["fill_percent"].mean()),
                    "required_bikes": int(group["bike_deficit"].sum()),
                    "releasable_bikes": int(group["bike_surplus"].sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["pressure_type", "stations"], ascending=[True, False]).reset_index(drop=True)
    return frame, summary


def _haversine_pairs(donors: pd.DataFrame, receivers: pd.DataFrame) -> pd.DataFrame:
    """Build all donor-receiver great-circle distances for the bounded network."""
    if donors.empty or receivers.empty:
        return pd.DataFrame(columns=["donor_id", "receiver_id", "distance_km"])
    donor_lat = np.radians(donors["latitude"].to_numpy())[:, None]
    donor_lon = np.radians(donors["longitude"].to_numpy())[:, None]
    receiver_lat = np.radians(receivers["latitude"].to_numpy())[None, :]
    receiver_lon = np.radians(receivers["longitude"].to_numpy())[None, :]
    delta_lat = receiver_lat - donor_lat
    delta_lon = receiver_lon - donor_lon
    a = np.sin(delta_lat / 2) ** 2 + np.cos(donor_lat) * np.cos(receiver_lat) * np.sin(delta_lon / 2) ** 2
    distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    donor_index, receiver_index = np.indices(distances.shape)
    return pd.DataFrame(
        {
            "donor_id": donors.iloc[donor_index.ravel()]["station_id"].to_numpy(),
            "receiver_id": receivers.iloc[receiver_index.ravel()]["station_id"].to_numpy(),
            "distance_km": distances.ravel(),
        }
    ).sort_values("distance_km").reset_index(drop=True)


def build_rebalancing_plan(
    stations: pd.DataFrame,
    van_capacity: int = 10,
    max_moves: int = 30,
    max_distance_km: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily match nearby surplus and deficit stations under visible constraints."""
    eligible = stations[
        stations["operational"]
        & ~stations["capacity_inconsistent"]
        & stations["docks"].gt(0)
    ].copy()
    donors = eligible[eligible["bike_surplus"].gt(0)].copy()
    receivers = eligible[eligible["bike_deficit"].gt(0)].copy()
    pairs = _haversine_pairs(donors, receivers)
    pairs = pairs[pairs["distance_km"].le(float(max_distance_km))]
    surplus = donors.set_index("station_id")["bike_surplus"].astype(int).to_dict()
    deficit = receivers.set_index("station_id")["bike_deficit"].astype(int).to_dict()
    station_names = stations.set_index("station_id")["station_name"].to_dict()
    moves = []
    capacity = max(int(van_capacity), 1)
    for pair in pairs.itertuples(index=False):
        if len(moves) >= max(int(max_moves), 0):
            break
        available = surplus.get(pair.donor_id, 0)
        needed = deficit.get(pair.receiver_id, 0)
        moved = min(available, needed, capacity)
        if moved <= 0:
            continue
        surplus[pair.donor_id] -= moved
        deficit[pair.receiver_id] -= moved
        moves.append(
            {
                "move": len(moves) + 1,
                "from_station_id": pair.donor_id,
                "from_station": station_names.get(pair.donor_id, pair.donor_id),
                "to_station_id": pair.receiver_id,
                "to_station": station_names.get(pair.receiver_id, pair.receiver_id),
                "bikes_to_move": int(moved),
                "distance_km": float(pair.distance_km),
                "bike_km": float(pair.distance_km * moved),
            }
        )
    plan = pd.DataFrame(moves)
    simulated = stations.copy()
    simulated["simulated_bikes"] = simulated["bikes"].astype(int)
    if not plan.empty:
        changes: dict[str, int] = {}
        for move in plan.itertuples(index=False):
            changes[move.from_station_id] = changes.get(move.from_station_id, 0) - move.bikes_to_move
            changes[move.to_station_id] = changes.get(move.to_station_id, 0) + move.bikes_to_move
        simulated["simulated_bikes"] += simulated["station_id"].map(changes).fillna(0).astype(int)
    simulated["simulated_fill_ratio"] = simulated["simulated_bikes"].div(simulated["docks"].replace(0, np.nan)).clip(0, 1)
    return plan, simulated


def scenario_summary(stations: pd.DataFrame, simulated: pd.DataFrame, critical_threshold: float) -> dict:
    """Compare critical station counts before and after a rebalancing scenario."""
    threshold = float(np.clip(critical_threshold, 0.02, 0.45))
    operational = stations["operational"]
    before_empty = operational & stations["fill_ratio"].le(threshold)
    before_full = operational & stations["fill_ratio"].ge(1 - threshold)
    after_empty = operational & simulated["simulated_fill_ratio"].le(threshold)
    after_full = operational & simulated["simulated_fill_ratio"].ge(1 - threshold)
    return {
        "before_empty": int(before_empty.sum()),
        "before_full": int(before_full.sum()),
        "after_empty": int(after_empty.sum()),
        "after_full": int(after_full.sum()),
        "critical_resolved": int((before_empty | before_full).sum() - (after_empty | after_full).sum()),
    }


def quality_report(stations: pd.DataFrame) -> pd.DataFrame:
    """Return explicit station-level schema and operational quality indicators."""
    checks = [
        ("Capacity arithmetic inconsistent", stations["capacity_inconsistent"]),
        ("Bike-type total inconsistent", stations["bike_type_inconsistent"]),
        ("Missing station update time", stations["station_updated_at"].isna()),
        ("Locked or not installed", ~stations["installed"].fillna(True) | stations["locked"].fillna(False)),
        ("At least one unavailable dock", stations["unavailable_docks"].gt(0)),
    ]
    return pd.DataFrame(
        [
            {
                "check": label,
                "affected_stations": int(mask.sum()),
                "share": float(mask.mean() * 100),
            }
            for label, mask in checks
        ]
    )
