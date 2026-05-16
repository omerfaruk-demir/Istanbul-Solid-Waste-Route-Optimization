"""
data_loader.py
--------------
Loads all EDA outputs and returns a single standardised problem instance
that GA, SA, and MILP solvers can consume directly.

Usage
-----
    from data_loader import load_instance
    inst = load_instance()          # uses default paths
    inst = load_instance(eda_dir="path/to/outputs/eda_output")

Returned dict keys
------------------
    nodes            : list of dicts  – all nodes (depot + stations + disposal)
    node_ids         : list[str]      – ordered node IDs  ["D0","S1",…,"F1",…]
    idx              : dict[str,int]  – node_id → matrix index
    n                : int            – total number of nodes
    stations         : list[str]      – node IDs of transfer stations
    disposal         : list[str]      – node IDs of disposal facilities
    depot            : str            – depot node ID ("D0")
    distance_matrix  : list[list[float]]  – NxN distances in km
    time_matrix      : list[list[float]]  – NxN travel times in minutes (free-flow baseline)
    alpha_r          : dict[int,float]    – congestion factor per hour {0:…, 1:…, …, 23:…}
    v_free_kmh       : float          – free-flow speed used to build time matrix
    fleet            : dict           – {"total":82, "europe":39, "asia":43}
    params           : dict           – capacity, horizon, depot/disposal ids, etc.
    demand           : dict[str,float]   – node_id → demand in tonnes (0 for depot/disposal)
"""

from pathlib import Path
import json
import csv


# ---------------------------------------------------------------------------
# Default paths (relative to repo root; override via argument)
# ---------------------------------------------------------------------------
_DEFAULT_EDA_DIR = Path(__file__).parent.parent / "outputs" / "eda_output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    """Read a square CSV matrix with a header row and index column.
    Returns (row_labels, matrix) where matrix[i][j] is the float value."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    labels = reader[0][1:]          # first row, skip empty corner cell
    matrix = []
    for row in reader[1:]:
        matrix.append([float(v) for v in row[1:]])
    return labels, matrix


def _travel_time_td(dist_km: float, depart_hour: int, alpha_r: dict) -> float:
    """Return time-dependent travel time in minutes.

    τ_ij(t) = (dist_km / v_free_kmh) * 60 * alpha_r[depart_hour]

    This matches Equation (1) in the project proposal.
    """
    raise NotImplementedError(
        "Use inst['alpha_r'] and inst['time_matrix'] directly; "
        "call compute_td_time() for a single arc."
    )


def compute_td_time(dist_km: float, v_free_kmh: float,
                    depart_minute: float, alpha_r: dict) -> float:
    """Compute time-dependent travel time for a single arc.

    Parameters
    ----------
    dist_km       : arc distance in km
    v_free_kmh    : free-flow speed (inst['v_free_kmh'])
    depart_minute : absolute departure time in minutes from horizon start
                    (e.g. 480 = 08:00 if horizon starts at 00:00)
    alpha_r       : congestion dict {hour_int: factor_float}

    Returns
    -------
    travel time in minutes (float)
    """
    hour = int(depart_minute // 60) % 24
    alpha = alpha_r[hour]
    free_flow_min = (dist_km / v_free_kmh) * 60.0
    return free_flow_min * alpha


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_instance(
    eda_dir: str | Path | None = None,
    capacity_tonnes: float = 15.0,
    horizon_min: int = 960,
    service_time_min: float = 30.0,
    time_window_open_min: float = 60.0,
    time_window_close_min: float = 480.0,
) -> dict:
    """Load the full Istanbul CVRP instance.

    Parameters
    ----------
    eda_dir             : path to outputs/eda_output/  (default: auto-detected)
    capacity_tonnes     : truck capacity in tonnes (default 15 t — standard
                          compactor; IBB does not publish this figure)
    horizon_min         : planning horizon in minutes (default 960 = 16 h)
    service_time_min    : service/unloading time at each station (minutes)
    time_window_open    : earliest service time at stations (minutes from
                          horizon start; default 60 → 01:00 if starts 00:00)
    time_window_close   : latest service time at stations (default 480 → 08:00)
    """

    eda_dir = Path(eda_dir) if eda_dir else _DEFAULT_EDA_DIR

    # ------------------------------------------------------------------
    # 1. Load raw files
    # ------------------------------------------------------------------
    instance_raw   = _load_json(eda_dir / "cvrp_instance.json")
    alpha_json     = _load_json(eda_dir / "alpha_r.json")
    dist_labels, dist_matrix = _load_csv_matrix(eda_dir / "distance_matrix_km.csv")
    time_labels, time_matrix = _load_csv_matrix(eda_dir / "time_matrix_baseline_min.csv")

    # ------------------------------------------------------------------
    # 2. Build ordered node list: depot first, then stations, then disposal
    # ------------------------------------------------------------------
    depot_raw      = instance_raw["depot"]
    stations_raw   = instance_raw["transfer_stations"]
    disposal_raw   = instance_raw["disposal_facilities"]

    depot_node = {
        "node_id":   depot_raw["node_id"],
        "name":      depot_raw["name"],
        "lat":       depot_raw["lat"],
        "lon":       depot_raw["lon"],
        "node_type": "Depot",
        "demand_tonnes": 0.0,
        "side":      "Avrupa",
        "time_window": [0.0, float(horizon_min)],
        "service_time_min": 0.0,
    }

    station_nodes = []
    for s in stations_raw:
        station_nodes.append({
            "node_id":   s["node_id"],
            "name":      s["name"],
            "lat":       s["lat"],
            "lon":       s["lon"],
            "node_type": "Transfer Station",
            "demand_tonnes": s["total_waste_tonnes"] / s["total_trips"],
            "side":      s["side"],
            "time_window": [time_window_open_min, time_window_close_min],
            "service_time_min": service_time_min,
        })

    disposal_nodes = []
    for d in disposal_raw:
        disposal_nodes.append({
            "node_id":   d["node_id"],
            "name":      d["name"],
            "lat":       d["lat"],
            "lon":       d["lon"],
            "node_type": "Disposal",
            "demand_tonnes": 0.0,
            "side":      d.get("side", ""),
            "time_window": [0.0, float(horizon_min)],
            "service_time_min": service_time_min,
        })

    all_nodes = [depot_node] + station_nodes + disposal_nodes
    node_ids  = [n["node_id"] for n in all_nodes]
    idx       = {nid: i for i, nid in enumerate(node_ids)}

    # ------------------------------------------------------------------
    # 3. Re-index distance / time matrices to match node_ids order
    # ------------------------------------------------------------------
    # dist_labels comes from the CSV header — may differ in order
    csv_idx = {label: i for i, label in enumerate(dist_labels)}

    n = len(node_ids)
    dist_reindexed = [[0.0] * n for _ in range(n)]
    time_reindexed = [[0.0] * n for _ in range(n)]

    for i, ni in enumerate(node_ids):
        for j, nj in enumerate(node_ids):
            if ni in csv_idx and nj in csv_idx:
                ci, cj = csv_idx[ni], csv_idx[nj]
                dist_reindexed[i][j] = dist_matrix[ci][cj]
                time_reindexed[i][j] = time_matrix[ci][cj]
            # else: 0.0 (node not in matrix — shouldn't happen)

    # ------------------------------------------------------------------
    # 4. Alpha_r — keyed by int hour
    # ------------------------------------------------------------------
    alpha_r = {int(k): float(v) for k, v in alpha_json["alpha_r"].items()}
    v_free  = float(alpha_json["v_free_kmh"])

    # ------------------------------------------------------------------
    # 5. Fleet
    # ------------------------------------------------------------------
    fleet_raw = instance_raw["fleet_2025"]
    fleet = {
        "total":   fleet_raw["total"],
        "europe":  fleet_raw["europe"],
        "asia":    fleet_raw["asia"],
    }

    # ------------------------------------------------------------------
    # 6. Demand dict
    # ------------------------------------------------------------------
    demand = {n["node_id"]: n["demand_tonnes"] for n in all_nodes}

    # ------------------------------------------------------------------
    # 7. Params
    # ------------------------------------------------------------------
    params = {
        "capacity_tonnes":    capacity_tonnes,
        "horizon_min":        horizon_min,
        "service_time_min":   service_time_min,
        "depot_id":           depot_node["node_id"],
        "disposal_ids":       [d["node_id"] for d in disposal_nodes],
        "station_ids":        [s["node_id"] for s in station_nodes],
        "num_trucks":         fleet["total"],
        "note_capacity":      (
            "15 t assumed (standard compactor); "
            "IBB does not publish per-truck capacity."
        ),
        "note_demand":        "Jan 2025 monthly totals aggregated from 48 municipalities.",
    }

    # ------------------------------------------------------------------
    # 8. Assemble and return
    # ------------------------------------------------------------------
    return {
        "nodes":           all_nodes,
        "node_ids":        node_ids,
        "idx":             idx,
        "n":               n,
        "depot":           depot_node["node_id"],
        "stations":        [s["node_id"] for s in station_nodes],
        "disposal":        [d["node_id"] for d in disposal_nodes],
        "distance_matrix": dist_reindexed,
        "time_matrix":     time_reindexed,
        "alpha_r":         alpha_r,
        "v_free_kmh":      v_free,
        "fleet":           fleet,
        "demand":          demand,
        "params":          params,
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    inst = load_instance()

    print(f"Nodes          : {inst['n']}  ({len(inst['stations'])} stations, "
          f"{len(inst['disposal'])} disposal, 1 depot)")
    print(f"Depot          : {inst['depot']}")
    print(f"Stations       : {inst['stations']}")
    print(f"Disposal       : {inst['disposal']}")
    print(f"Fleet          : {inst['fleet']}")
    print(f"Capacity       : {inst['params']['capacity_tonnes']} t/truck")
    print(f"Horizon        : {inst['params']['horizon_min']} min")
    print(f"v_free         : {inst['v_free_kmh']} km/h")
    print(f"alpha peak(18h): {inst['alpha_r'][18]}")
    print()
    print("Distance matrix sample (first 4 nodes):")
    ids = inst['node_ids'][:4]
    header = f"{'':6}" + "".join(f"{i:>10}" for i in ids)
    print(header)
    for ni in ids:
        row = inst['distance_matrix'][inst['idx'][ni]]
        vals = "".join(f"{row[inst['idx'][nj]]:10.2f}" for nj in ids)
        print(f"{ni:6}{vals}")
    print()
    print("Demand (tonnes):")
    for sid in inst['stations']:
        print(f"  {sid}: {inst['demand'][sid]:,.1f} t")
    print()
    print("Time-dependent travel time example:")
    print(f"  D0→S1 at 08:00 (480 min): "
          f"{compute_td_time(inst['distance_matrix'][inst['idx']['D0']][inst['idx']['S1']], inst['v_free_kmh'], 480, inst['alpha_r']):.1f} min")
    print(f"  D0→S1 at 02:00 (120 min): "
          f"{compute_td_time(inst['distance_matrix'][inst['idx']['D0']][inst['idx']['S1']], inst['v_free_kmh'], 120, inst['alpha_r']):.1f} min")


# ---------------------------------------------------------------------------
# Istanbul instance loader — same format as milp_test_case.load_instance()
# ---------------------------------------------------------------------------

def load_istanbul_instance(
    eda_dir: str | Path | None = None,
    capacity_tonnes: float = 15.0,
    horizon_min: int = 960,
    service_time_min: float = 30.0,
    time_window_open_min: float = 60.0,
    time_window_close_min: float = 480.0,
    num_trucks: int | None = None,
    start_hour: int = 7,
) -> dict:
    """Load the full Istanbul CVRP instance in the SAME format as
    milp_test_case.load_instance() so that GA and SA can use it without
    any code changes.

    Drop-in replacement:
        # Before (test case):
        from src.milp_test_case import load_instance
        instance = load_instance()

        # After (full Istanbul):
        from src.data_loader import load_istanbul_instance as load_instance
        instance = load_instance()

    Parameters
    ----------
    eda_dir               : path to outputs/eda_output/
    capacity_tonnes       : truck capacity in tonnes (default 15 t)
    horizon_min           : planning horizon in minutes (default 960 = 16 h)
    service_time_min      : service time at each station (minutes)
    time_window_open_min  : earliest service time for stations (minutes)
    time_window_close_min : latest service time for stations (minutes)
    num_trucks            : override fleet size (default: total fleet from data)
    start_hour            : hour the horizon starts (default 7 = 07:00)
    """
    import math

    eda_dir = Path(eda_dir) if eda_dir else _DEFAULT_EDA_DIR

    # ── Raw files ──────────────────────────────────────────────────────────
    instance_raw = _load_json(eda_dir / "cvrp_instance.json")
    alpha_json   = _load_json(eda_dir / "alpha_r.json")
    _, dist_matrix_raw = _load_csv_matrix(eda_dir / "distance_matrix_km.csv")
    dist_labels, _     = _load_csv_matrix(eda_dir / "distance_matrix_km.csv")

    alpha_r  = {int(k): float(v) for k, v in alpha_json["alpha_r"].items()}
    v_free   = float(alpha_json["v_free_kmh"])

    # ── Node ordering: depot → stations → disposal ──────────────────────────
    depot_raw    = instance_raw["depot"]
    stations_raw = instance_raw["transfer_stations"]
    disposal_raw = instance_raw["disposal_facilities"]

    all_node_dicts = (
        [{"node_id": depot_raw["node_id"], "lat": depot_raw["lat"],
          "lon": depot_raw["lon"], "node_type": "Depot",
          "demand_tonnes": 0.0, "service_time_min": 0.0,
          "tw_early": 0.0, "tw_late": float(horizon_min)}]
        + [{"node_id": s["node_id"], "lat": s["lat"], "lon": s["lon"],
            "node_type": "TransferStation",
            "demand_tonnes": s["total_waste_tonnes"] / s["total_trips"],
            "service_time_min": service_time_min,
            "tw_early": time_window_open_min,
            "tw_late": time_window_close_min}
           for s in stations_raw]
        + [{"node_id": d["node_id"], "lat": d["lat"], "lon": d["lon"],
            "node_type": "Disposal",
            "demand_tonnes": 0.0,
            "service_time_min": service_time_min,
            "tw_early": 0.0, "tw_late": float(horizon_min)}
           for d in disposal_raw]
    )

    idx_to_id = [nd["node_id"] for nd in all_node_dicts]
    id_to_idx = {nid: i for i, nid in enumerate(idx_to_id)}
    n         = len(idx_to_id)

    depot_idx    = id_to_idx[depot_raw["node_id"]]
    disposal_idx = [id_to_idx[d["node_id"]] for d in disposal_raw]
    station_idx  = [id_to_idx[s["node_id"]] for s in stations_raw]

    # ── Re-index distance matrix ────────────────────────────────────────────
    csv_pos = {label: i for i, label in enumerate(dist_labels)}
    _, dist_raw = _load_csv_matrix(eda_dir / "distance_matrix_km.csv")

    dist_km  = {}
    time_min = {}
    for i, ni in enumerate(idx_to_id):
        for j, nj in enumerate(idx_to_id):
            if ni in csv_pos and nj in csv_pos:
                d = dist_raw[csv_pos[ni]][csv_pos[nj]]
            else:
                # Haversine fallback for nodes missing from CSV
                n1, n2 = all_node_dicts[i], all_node_dicts[j]
                R = 6371.0
                lat1, lon1 = math.radians(n1["lat"]), math.radians(n1["lon"])
                lat2, lon2 = math.radians(n2["lat"]), math.radians(n2["lon"])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                d = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist_km[i, j]  = d
            # baseline free-flow travel time (no congestion)
            time_min[i, j] = (d / v_free) * 60.0

    # ── traffic_profile compatible with src/traffic.py ──────────────────────
    # traffic.py expects load_traffic_profile() output — a list of 24 alpha
    # values indexed by hour, or a dict. We pass the alpha_r dict directly.
    traffic_profile = alpha_r   # {0: 1.49, 1: 1.44, …, 23: 1.49}

    # ── Fleet ───────────────────────────────────────────────────────────────
    fleet_raw   = instance_raw["fleet_2025"]
    total_fleet = fleet_raw["total"]
    if num_trucks is None:
        num_trucks = min(total_fleet, len(stations_raw))

    return {
        # ── Identifiers (same keys as milp_test_case) ──────────────────────
        "n":          n,
        "idx_to_id":  idx_to_id,
        "id_to_idx":  id_to_idx,
        "depot":      depot_idx,
        "disposal":   disposal_idx,
        "stations":   station_idx,
        "all_nodes":  list(range(n)),
        # ── Problem parameters ─────────────────────────────────────────────
        "num_trucks": num_trucks,
        "capacity":   capacity_tonnes,
        "horizon":    horizon_min,
        # ── Distance / time matrices ───────────────────────────────────────
        "dist_km":    dist_km,
        "time_min":   time_min,
        # ── Node attributes ────────────────────────────────────────────────
        "demand":       {i: nd["demand_tonnes"]    for i, nd in enumerate(all_node_dicts)},
        "service_time": {i: nd["service_time_min"] for i, nd in enumerate(all_node_dicts)},
        "tw_early":     {i: nd["tw_early"]         for i, nd in enumerate(all_node_dicts)},
        "tw_late":      {i: nd["tw_late"]          for i, nd in enumerate(all_node_dicts)},
        # ── Traffic ────────────────────────────────────────────────────────
        "traffic_profile":    traffic_profile,
        "traffic_start_hour": start_hour,
        "alpha_r":            alpha_r,
        "v_free_kmh":         v_free,
        # ── Extra metadata ─────────────────────────────────────────────────
        "fleet": fleet_raw,
        "params": {
            "capacity_tonnes_per_truck": capacity_tonnes,
            "horizon_min":               horizon_min,
            "num_trucks":                num_trucks,
            "avg_speed_kmh":             v_free,
            "depot_node_id":             depot_raw["node_id"],
            "disposal_node_ids":         [d["node_id"] for d in disposal_raw],
            "start_hour":                start_hour,
        },
    }


if __name__ == "__main__":
    # Quick sanity check for load_istanbul_instance
    inst = load_istanbul_instance()
    print(f"\nIstanbul instance — solver-compatible format")
    print(f"  n            : {inst['n']}")
    print(f"  stations     : {[inst['idx_to_id'][i] for i in inst['stations']]}")
    print(f"  disposal     : {[inst['idx_to_id'][i] for i in inst['disposal']]}")
    print(f"  depot        : {inst['idx_to_id'][inst['depot']]}")
    print(f"  num_trucks   : {inst['num_trucks']}")
    print(f"  capacity     : {inst['capacity']} t")
    print(f"  dist D0→S1   : {inst['dist_km'][inst['depot'], inst['stations'][0]]:.2f} km")
    print(f"  time D0→S1   : {inst['time_min'][inst['depot'], inst['stations'][0]]:.2f} min (free-flow)")