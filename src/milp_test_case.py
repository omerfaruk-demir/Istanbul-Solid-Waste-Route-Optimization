from __future__ import annotations

import json
import sys
import time
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR    = ROOT / "test_case"
NODES_CSV   = DATA_DIR / "test_nodes.csv"
PARAMS_JSON = DATA_DIR / "test_case_params.json"
RESULTS_DIR = ROOT / "results"
TRAFFIC_JSON = ROOT / "outputs" / "eda_output" / "alpha_r.json"
RESULTS_DIR.mkdir(exist_ok=True)

from src.traffic import load_traffic_profile


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# Instance loader
# ---------------------------------------------------------------------------

def load_instance(nodes_csv: Path = NODES_CSV,
                  params_json: Path = PARAMS_JSON) -> dict:
    """Load CSV + JSON into a structured instance dict."""
    nodes = pd.read_csv(nodes_csv)
    with open(params_json) as f:
        params = json.load(f)

    idx_to_id = list(nodes["node_id"])
    id_to_idx = {nid: i for i, nid in enumerate(idx_to_id)}
    n = len(nodes)

    depot_id     = params["depot_node_id"]
    disposal_ids = params["disposal_node_ids"]
    depot_idx    = id_to_idx[depot_id]
    disposal_idx = [id_to_idx[d] for d in disposal_ids]
    station_idx  = [i for i, nid in enumerate(idx_to_id)
                    if nodes.loc[i, "node_type"] == "TransferStation"]

    speed_kmh = params["avg_speed_kmh"]
    lats = nodes["lat"].values
    lons = nodes["lon"].values

    dist_km  = {}
    time_min = {}
    for i in range(n):
        for j in range(n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist_km[i, j]  = d
            time_min[i, j] = (d / speed_kmh) * 60.0

    return {
        "n": n,
        "idx_to_id": idx_to_id,
        "id_to_idx": id_to_idx,
        "depot":    depot_idx,
        "disposal": disposal_idx,
        "stations": station_idx,
        "all_nodes": list(range(n)),
        "num_trucks": params["num_trucks"],
        "capacity":   params["capacity_tonnes_per_truck"],
        "horizon":    params["horizon_min"],
        "dist_km":    dist_km,
        "time_min":   time_min,
        "demand":     {i: float(nodes.loc[i, "demand_tonnes"]) for i in range(n)},
        "service_time": {i: float(nodes.loc[i, "service_time_min"]) for i in range(n)},
        "tw_early":   {i: float(nodes.loc[i, "time_window_early"]) for i in range(n)},
        "tw_late":    {i: float(nodes.loc[i, "time_window_late"])  for i in range(n)},
        "traffic_profile": load_traffic_profile(TRAFFIC_JSON),
        "traffic_start_hour": int(params.get("start_hour", 7)),
        "nodes_df":   nodes,
        "params":     params,
    }


# ---------------------------------------------------------------------------
# MILP model
# ---------------------------------------------------------------------------

def build_and_solve(instance: dict,
                    time_limit: int = 60,
                    verbose: bool = False) -> dict:
    """
    Build the MILP and solve with CBC.

    Decision variables (proposal eqs. 2–4)
    ----------------------------------------
    x[i,j,k]  ∈ {0,1}  truck k travels arc i→j             (eq. 2)
    w[i,k]    ≥ 0       load on truck k departing station i  (eq. 3)
    t[i,k]    ≥ 0       arrival time of truck k at node i    (eq. 4)

    Also introduced for a tighter formulation:
    u[i,k]    ∈ {0,1}   = 1 if truck k visits node i

    Objective (eq. 5 simplified)
    -----------------------------
    min  Σ_{i,j,k, i≠j}  dist_km[i,j] · x[i,j,k]

    Constraints implemented
    ------------------------
    eq. 6  — Coverage
    eq. 7  — Capacity
    eq. 8  — Flow conservation
    eq. 9  — Depot origination (≤ 1; unused trucks stay feasible)
    eq. 10 — Time windows
    eq. 11 — Schedule consistency (Big-M per arc, skip return-to-depot)
    eq. 12 — Waste balance
    """
    t0_solve = time.time()

    N   = instance["all_nodes"]
    S   = instance["stations"]
    F   = instance["disposal"]
    d0  = instance["depot"]
    K   = list(range(instance["num_trucks"]))
    Q   = instance["capacity"]
    H   = instance["horizon"]
    dist = instance["dist_km"]
    tau  = instance["time_min"]
    svc  = instance["service_time"]
    dem  = instance["demand"]
    a    = instance["tw_early"]
    b    = instance["tw_late"]

    # Per-arc Big-M: arrival at j can be at most H; departure from i at least 0.
    # Tight M_ij = H so the LP relaxation stays as tight as possible.
    M_arc = H

    prob = pulp.LpProblem("Istanbul_CVRP", pulp.LpMinimize)

    # ── Variables ────────────────────────────────────────────────────────────
    # Arc variables (eq. 2)
    x = {(i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
         for i in N for j in N for k in K if i != j}

    # Load variables (eq. 3) — only for station nodes
    w = {(i, k): pulp.LpVariable(f"w_{i}_{k}", lowBound=0, upBound=Q)
         for i in S for k in K}

    # Arrival-time variables (eq. 4) — not needed at depot (fixed 0) or return
    t = {(i, k): pulp.LpVariable(f"t_{i}_{k}", lowBound=0, upBound=H)
         for i in N if i != d0 for k in K}

    # Visit-indicator variables (tighter Big-M formulation)
    u = {(i, k): pulp.LpVariable(f"u_{i}_{k}", cat="Binary")
         for i in N for k in K}

    # ── Objective (eq. 5 simplified) ────────────────────────────────────────
    prob += pulp.lpSum(dist[i, j] * x[i, j, k]
                       for i in N for j in N for k in K if i != j)

    # ── Link u to x (visit indicator) ───────────────────────────────────────
    for i in N:
        for k in K:
            in_arcs = pulp.lpSum(x[j, i, k] for j in N if j != i)
            # u[i,k] = 1 iff truck k enters node i
            prob += (u[i, k] == in_arcs, f"link_u_{i}_k{k}")

    # Depot: truck k is "used" iff it leaves the depot
    for k in K:
        out_d0 = pulp.lpSum(x[d0, j, k] for j in N if j != d0)
        # out-degree of d0 equals in-degree (flow), and u[d0,k]=in-degree
        # Force u[d0,k] to also reflect departure
        prob += (u[d0, k] == out_d0, f"link_u_depot_k{k}")

    # ── Structure tightening (valid for test-case route structure) ───────────
    # Disposal nodes cannot precede station visits (D0→S*→F→D0 structure).
    for f in F:
        for s in S:
            for k in K:
                prob += (x[f, s, k] == 0, f"no_f_to_s_f{f}_s{s}_k{k}")
        # Stations cannot go directly back to depot — must pass through F first.
        for s in S:
            for k in K:
                prob += (x[s, d0, k] == 0, f"no_s_to_depot_s{s}_k{k}")

    # ── Constraint (eq. 6): Coverage ────────────────────────────────────────
    for i in S:
        prob += (
            pulp.lpSum(u[i, k] for k in K) == 1,
            f"eq6_coverage_{i}"
        )

    # ── Constraint (eq. 7): Capacity ────────────────────────────────────────
    for k in K:
        prob += (
            pulp.lpSum(w[i, k] for i in S) <= Q,
            f"eq7_capacity_k{k}"
        )

    # ── Constraint (eq. 8): Flow conservation ───────────────────────────────
    for i in N:
        for k in K:
            prob += (
                pulp.lpSum(x[i, j, k] for j in N if j != i) ==
                pulp.lpSum(x[j, i, k] for j in N if j != i),
                f"eq8_flow_{i}_k{k}"
            )

    # ── Constraint (eq. 9): Depot origination (relaxed ≤ 1) ─────────────────
    for k in K:
        prob += (u[d0, k] <= 1, f"eq9_depot_k{k}")

    # ── Fix depot departure time = 0 (avoids circular time constraint) ───────
    # t[d0, k] is not a model variable; departure is implicitly at time 0.

    # ── Constraint (eq. 10): Time windows ───────────────────────────────────
    for i in N:
        if i == d0:
            continue
        for k in K:
            prob += (t[i, k] >= a[i] * u[i, k],                   f"eq10_early_{i}_k{k}")
            prob += (t[i, k] <= b[i] + M_arc * (1 - u[i, k]),     f"eq10_late_{i}_k{k}")

    # ── Constraint (eq. 11): Schedule consistency ────────────────────────────
    # Skip arcs returning to depot (return time not tracked).
    for i in N:
        for j in N:
            if i == j or j == d0:
                continue
            for k in K:
                # Departure from depot: t[j,k] >= 0 + svc[d0] + tau[d0,j]
                if i == d0:
                    prob += (
                        t[j, k] >= tau[d0, j] - M_arc * (1 - x[i, j, k]),
                        f"eq11_{i}_{j}_k{k}"
                    )
                else:
                    prob += (
                        t[j, k] >= t[i, k] + svc[i] + tau[i, j] - M_arc * (1 - x[i, j, k]),
                        f"eq11_{i}_{j}_k{k}"
                    )

    # Ensure each truck's tour fits in the horizon (return leg)
    for k in K:
        for f in F:
            prob += (
                t[f, k] + svc[f] + dist[f, d0] / instance["params"]["avg_speed_kmh"] * 60
                <= H + M_arc * (1 - u[f, k]),
                f"eq11_horizon_f{f}_k{k}"
            )

    # ── Constraint (eq. 12): Waste balance ───────────────────────────────────
    for i in S:
        # Load collected at i across all trucks ≤ available waste
        prob += (
            pulp.lpSum(w[i, k] for k in K) <= dem[i],
            f"eq12_balance_{i}"
        )
        for k in K:
            # w[i,k] = dem[i] when visited, 0 otherwise
            prob += (w[i, k] <= dem[i] * u[i, k],   f"eq12_ub_{i}_k{k}")
            prob += (w[i, k] >= dem[i] * u[i, k],   f"eq12_lb_{i}_k{k}")

    # Each used truck must visit a disposal facility
    for k in K:
        for f in F:
            prob += (u[f, k] >= u[d0, k], f"must_visit_disposal_f{f}_k{k}")

    # ── Solve ────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    prob.solve(solver)
    elapsed = time.time() - t0_solve

    status    = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective) or 0.0

    # ── Extract solution ─────────────────────────────────────────────────────
    x_val = {
        (i, j, k): (pulp.value(x[i, j, k]) or 0.0) > 0.5
        for i in N for j in N for k in K if i != j
    }
    w_val = {(i, k): pulp.value(w[i, k]) or 0.0 for i in S for k in K}
    t_val = {(i, k): (pulp.value(t[i, k]) or 0.0) if i != d0 else 0.0
             for i in N for k in K}

    routes: dict[int, list[int]] = {}
    for k in K:
        edges = [(i, j) for (i, j, kk), v in x_val.items() if kk == k and v]
        if not edges:
            routes[k] = []
            continue
        route = [d0]
        used  = set()
        while True:
            last = route[-1]
            nxt  = next((j for i, j in edges if i == last and (last, j) not in used), None)
            if nxt is None:
                break
            used.add((last, nxt))
            route.append(nxt)
        routes[k] = route

    total_dist  = sum(dist[i, j] for (i, j, k), v in x_val.items() if v)
    trucks_used = sum(1 for k in K if routes[k])
    idx_to_id   = instance["idx_to_id"]

    return {
        "status":           status,
        "objective_value":  round(objective, 4),
        "total_distance_km": round(total_dist, 4),
        "trucks_used":      trucks_used,
        "runtime_seconds":  round(elapsed, 4),
        "routes":           {k: [idx_to_id[n] for n in r] for k, r in routes.items()},
        "routes_idx":       routes,
        "x_val":            x_val,
        "w_val":            {(i, k): w_val.get((i, k), 0.0) for i in N for k in K},
        "t_val":            t_val,
        "instance":         instance,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_solution(sol: dict) -> None:
    inst      = sol["instance"]
    idx_to_id = inst["idx_to_id"]
    dist      = inst["dist_km"]
    S         = inst["stations"]
    dem       = inst["demand"]
    print("\n" + "=" * 60)
    print("MILP SOLUTION SUMMARY")
    print("=" * 60)
    print(f"  Status          : {sol['status']}")
    print(f"  Objective (km)  : {sol['objective_value']}")
    print(f"  Total distance  : {sol['total_distance_km']} km")
    print(f"  Trucks used     : {sol['trucks_used']} / {inst['num_trucks']}")
    print(f"  Runtime         : {sol['runtime_seconds']} s")
    print()
    for k, route in sol["routes_idx"].items():
        if not route:
            print(f"  Truck {k}: (unused)")
            continue
        load = sum(dem[n] for n in route if n in S)
        leg  = sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))
        names = " -> ".join(idx_to_id[n] for n in route)
        print(f"  Truck {k}: {names}")
        print(f"           load {load:.1f} t | route dist {leg:.2f} km")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading test instance...")
    inst = load_instance()
    print(f"  {inst['n']} nodes | {len(inst['stations'])} stations | "
          f"{inst['num_trucks']} trucks | cap {inst['capacity']} t")

    print("\nSolving MILP (CBC)...")
    sol = build_and_solve(inst, time_limit=60, verbose=False)
    print_solution(sol)

    out = {k: v for k, v in sol.items()
           if k not in ("x_val", "w_val", "t_val", "routes_idx", "instance")}
    with open(RESULTS_DIR / "milp_solution.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSolution saved to results/milp_solution.json")
