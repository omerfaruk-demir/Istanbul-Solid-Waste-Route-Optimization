from __future__ import annotations

import json
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _reconstruct_from_routes(sol: dict) -> dict:
    """
    If x_val / w_val / t_val are absent (GA/SA), reconstruct x_val and w_val
    from routes_idx. t_val is computed from travel times.
    """
    inst = sol["instance"]
    if "x_val" in sol and sol["x_val"] is not None:
        return sol

    N = inst["all_nodes"]
    K = list(range(inst["num_trucks"]))
    tau = inst["time_min"]
    svc = inst["service_time"]
    dem = inst["demand"]
    S   = inst["stations"]

    x_val = {(i, j, k): False for i in N for j in N for k in K if i != j}
    w_val = {(i, k): 0.0 for i in N for k in K}
    t_val = {(i, k): 0.0 for i in N for k in K}

    for k, route in sol["routes_idx"].items():
        for pos in range(len(route) - 1):
            i, j = route[pos], route[pos + 1]
            x_val[(i, j, k)] = True
        # arrival times: accumulate from depot, waiting if a node is not open yet.
        arr = 0.0
        for pos, node in enumerate(route):
            if pos > 0:
                arr = max(arr, inst["tw_early"][node])
            if node != inst["depot"] or pos == 0:
                t_val[(node, k)] = arr
            if pos < len(route) - 1:
                nxt = route[pos + 1]
                arr += svc[node] + tau[node, nxt]
        # loads
        for node in route:
            if node in S:
                w_val[(node, k)] = dem[node]

    return {**sol, "x_val": x_val, "w_val": w_val, "t_val": t_val}


def _pass(msg: str) -> tuple[bool, str]:
    line = f"  [PASS] {msg}"
    print(line)
    return True, line


def _fail(msg: str) -> tuple[bool, str]:
    line = f"  [FAIL] {msg}"
    print(line)
    return False, line


# ---------------------------------------------------------------------------
# Individual constraint checks
# ---------------------------------------------------------------------------

def check_coverage(sol: dict, instance: dict) -> bool:
    """
    Eq. (6): Every transfer station must be served by exactly one truck.
    """
    print("\n--- Eq. (6) Coverage ---")
    sol = _reconstruct_from_routes(sol)
    S = instance["stations"]
    K = list(range(instance["num_trucks"]))
    N = instance["all_nodes"]
    x = sol["x_val"]
    idx_to_id = instance["idx_to_id"]
    all_pass = True

    for i in S:
        visits = sum(x.get((i, j, k), False) for j in N for k in K if j != i)
        if visits == 1:
            _, line = _pass(f"Station {idx_to_id[i]} served exactly once")
        else:
            _, line = _fail(f"Station {idx_to_id[i]} served {visits} times (expected 1)")
            all_pass = False

    return all_pass


def check_capacity(sol: dict, instance: dict) -> bool:
    """
    Eq. (7): Load on each truck must not exceed its capacity Q_k.
    """
    print("\n--- Eq. (7) Capacity ---")
    sol = _reconstruct_from_routes(sol)
    S = instance["stations"]
    K = list(range(instance["num_trucks"]))
    Q = instance["capacity"]
    w = sol["w_val"]
    all_pass = True

    for k in K:
        load = sum(w.get((i, k), 0.0) for i in S)
        if load <= Q + 1e-6:
            _, _ = _pass(f"Truck {k}: load {load:.2f} t <= capacity {Q} t")
        else:
            _, _ = _fail(f"Truck {k}: load {load:.2f} t EXCEEDS capacity {Q} t")
            all_pass = False

    return all_pass


def check_flow_conservation(sol: dict, instance: dict) -> bool:
    """
    Eq. (8): For every node and every truck, in-degree == out-degree.
    """
    print("\n--- Eq. (8) Flow conservation ---")
    sol = _reconstruct_from_routes(sol)
    N = instance["all_nodes"]
    K = list(range(instance["num_trucks"]))
    x = sol["x_val"]
    idx_to_id = instance["idx_to_id"]
    all_pass = True

    for k in K:
        for i in N:
            out_deg = sum(x.get((i, j, k), False) for j in N if j != i)
            in_deg  = sum(x.get((j, i, k), False) for j in N if j != i)
            if out_deg == in_deg:
                _pass(f"Truck {k} node {idx_to_id[i]}: in={in_deg} out={out_deg}")
            else:
                _fail(f"Truck {k} node {idx_to_id[i]}: in={in_deg} ≠ out={out_deg}")
                all_pass = False

    return all_pass


def check_depot_constraints(sol: dict, instance: dict) -> bool:
    """
    Eq. (9): Each truck leaves and returns to depot at most once.
    (Relaxed from strict equality so unused trucks stay feasible.)
    """
    print("\n--- Eq. (9) Depot origination ---")
    sol = _reconstruct_from_routes(sol)
    d0 = instance["depot"]
    N  = instance["all_nodes"]
    K  = list(range(instance["num_trucks"]))
    x  = sol["x_val"]
    all_pass = True

    for k in K:
        dep_out = sum(x.get((d0, j, k), False) for j in N if j != d0)
        dep_in  = sum(x.get((j, d0, k), False) for j in N if j != d0)
        ok = (dep_out <= 1) and (dep_in <= 1)
        msg = f"Truck {k}: depot_out={int(dep_out)} depot_in={int(dep_in)} (each <= 1)"
        if ok:
            _pass(msg)
        else:
            _fail(msg)
            all_pass = False

    return all_pass


def check_time_windows(sol: dict, instance: dict) -> bool:
    """
    Eq. (10): Arrival at each visited node must be within [a_i, b_i].
    """
    print("\n--- Eq. (10) Time windows ---")
    sol = _reconstruct_from_routes(sol)
    N   = instance["all_nodes"]
    K   = list(range(instance["num_trucks"]))
    x   = sol["x_val"]
    t   = sol["t_val"]
    a   = instance["tw_early"]
    b   = instance["tw_late"]
    idx_to_id = instance["idx_to_id"]
    all_pass = True

    for k in K:
        for i in N:
            visited = any(x.get((j, i, k), False) for j in N if j != i)
            if not visited:
                continue
            arr = t.get((i, k), 0.0)
            if a[i] - 1e-4 <= arr <= b[i] + 1e-4:
                _pass(f"Truck {k} node {idx_to_id[i]}: arrival {arr:.1f} in [{a[i]}, {b[i]}]")
            else:
                _fail(f"Truck {k} node {idx_to_id[i]}: arrival {arr:.1f} OUTSIDE [{a[i]}, {b[i]}]")
                all_pass = False

    return all_pass


def check_schedule_consistency(sol: dict, instance: dict) -> bool:
    """
    Eq. (11): t[j,k] >= t[i,k] + service_i + tau[i,j]  for each arc x[i,j,k]=1.
    """
    print("\n--- Eq. (11) Schedule consistency ---")
    sol = _reconstruct_from_routes(sol)
    N   = instance["all_nodes"]
    K   = list(range(instance["num_trucks"]))
    x   = sol["x_val"]
    t   = sol["t_val"]
    tau = instance["time_min"]
    svc = instance["service_time"]
    idx_to_id = instance["idx_to_id"]
    all_pass = True

    d0 = instance["depot"]
    for k in K:
        for i in N:
            for j in N:
                if i == j or j == d0:   # return-to-depot arcs not tracked
                    continue
                if not x.get((i, j, k), False):
                    continue
                required = t.get((i, k), 0.0) + svc[i] + tau[i, j]
                actual   = t.get((j, k), 0.0)
                if actual >= required - 1e-3:
                    _pass(
                        f"Truck {k}: {idx_to_id[i]}->{idx_to_id[j]}  "
                        f"arr_j={actual:.1f} >= {required:.1f}"
                    )
                else:
                    _fail(
                        f"Truck {k}: {idx_to_id[i]}->{idx_to_id[j]}  "
                        f"arr_j={actual:.1f} < required {required:.1f}"
                    )
                    all_pass = False

    return all_pass


def check_waste_balance(sol: dict, instance: dict) -> bool:
    """
    Eq. (12): Total load collected at station i (across all trucks) <= q_i.
    Also checks that w[i,k] > 0 only when truck k visits i.
    """
    print("\n--- Eq. (12) Waste balance ---")
    sol = _reconstruct_from_routes(sol)
    S   = instance["stations"]
    N   = instance["all_nodes"]
    K   = list(range(instance["num_trucks"]))
    x   = sol["x_val"]
    w   = sol["w_val"]
    dem = instance["demand"]
    idx_to_id = instance["idx_to_id"]
    all_pass = True

    for i in S:
        total_collected = sum(w.get((i, k), 0.0) for k in K)
        if total_collected <= dem[i] + 1e-6:
            _pass(
                f"Station {idx_to_id[i]}: collected {total_collected:.2f} t <= "
                f"available {dem[i]:.2f} t"
            )
        else:
            _fail(
                f"Station {idx_to_id[i]}: collected {total_collected:.2f} t EXCEEDS "
                f"available {dem[i]:.2f} t"
            )
            all_pass = False

        for k in K:
            visited = any(x.get((j, i, k), False) for j in N if j != i)
            load_k  = w.get((i, k), 0.0)
            if not visited and load_k > 1e-6:
                _fail(f"  Truck {k} has load {load_k:.2f} at unvisited station {idx_to_id[i]}")
                all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Aggregate validator
# ---------------------------------------------------------------------------

def validate_all(sol: dict, instance: dict) -> dict[str, bool]:
    """
    Run all constraint checks and return a dict of {check_name: passed}.
    Prints PASS / FAIL for each individual check.
    """
    print("\n" + "=" * 50)
    print("CONSTRAINT VALIDATION REPORT")
    print("=" * 50)

    results = {
        "coverage":             check_coverage(sol, instance),
        "capacity":             check_capacity(sol, instance),
        "flow_conservation":    check_flow_conservation(sol, instance),
        "depot_constraints":    check_depot_constraints(sol, instance),
        "time_windows":         check_time_windows(sol, instance),
        "schedule_consistency": check_schedule_consistency(sol, instance),
        "waste_balance":        check_waste_balance(sol, instance),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, passed in results.items():
        label = "PASS" if passed else "FAIL"
        print(f"  {label}  {name}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  ALL CONSTRAINTS SATISFIED")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  FAILED: {', '.join(failed)}")
    print("=" * 50)

    return results


# ---------------------------------------------------------------------------
# Entry point — runs validation on the last MILP solution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from src.milp_test_case import load_instance, build_and_solve

    print("Loading instance and solving MILP...")
    instance = load_instance()
    sol = build_and_solve(instance, time_limit=120, verbose=False)

    if sol["status"] not in ("Optimal", "Not Solved"):
        print(f"Solver returned status: {sol['status']}")

    results = validate_all(sol, instance)
    all_pass = all(results.values())
    sys.exit(0 if all_pass else 1)
