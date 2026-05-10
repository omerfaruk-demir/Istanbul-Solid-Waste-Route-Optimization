from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.milp_test_case import load_instance
from src.validate import validate_all

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _route_distance(route: list[int], dist: dict[tuple[int, int], float]) -> float:
    return sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))


def _truck_load(route: list[int], stations: set[int], demand: dict[int, float]) -> float:
    return sum(demand[node] for node in route if node in stations)


def _arrival_times(route: list[int], instance: dict) -> dict[int, float]:
    tau = instance["time_min"]
    service_time = instance["service_time"]
    tw_early = instance["tw_early"]
    arrivals: dict[int, float] = {}
    elapsed = 0.0
    for pos, node in enumerate(route):
        if pos > 0:
            elapsed = max(elapsed, tw_early[node])
        arrivals[node] = elapsed
        if pos < len(route) - 1:
            elapsed += service_time[node] + tau[node, route[pos + 1]]
    return arrivals


def _route_time_penalty(route: list[int], instance: dict) -> float:
    if not route:
        return 0.0
    arrivals = _arrival_times(route, instance)
    penalty = 0.0
    for node, arrival in arrivals.items():
        if arrival > instance["tw_late"][node]:
            penalty += arrival - instance["tw_late"][node]
    return_end = (
        arrivals[route[-2]]
        + instance["service_time"][route[-2]]
        + instance["time_min"][route[-2], route[-1]]
    )
    if return_end > instance["horizon"]:
        penalty += return_end - instance["horizon"]
    return penalty


def _build_routes(assignments: list[list[int]], instance: dict) -> list[list[int]]:
    depot = instance["depot"]
    disposal = instance["disposal"][0]
    return [
        [depot, *stations, disposal, depot] if stations else []
        for stations in assignments
    ]


def _objective(assignments: list[list[int]], instance: dict) -> tuple[float, list[list[int]]]:
    routes = _build_routes(assignments, instance)
    station_set = set(instance["stations"])
    capacity = instance["capacity"]
    demand = instance["demand"]
    dist = instance["dist_km"]

    total_distance = 0.0
    penalty = 0.0
    visited: list[int] = []

    for route in routes:
        if not route:
            continue
        total_distance += _route_distance(route, dist)
        load = _truck_load(route, station_set, demand)
        if load > capacity:
            penalty += (load - capacity) * 1000.0
        penalty += _route_time_penalty(route, instance) * 1000.0
        visited.extend(node for node in route if node in station_set)

    missing = set(instance["stations"]).difference(visited)
    duplicates = len(visited) - len(set(visited))
    penalty += (len(missing) + duplicates) * 10000.0
    return total_distance + penalty, routes


def _initial_assignments(instance: dict) -> list[list[int]]:
    assignments = [[] for _ in range(instance["num_trucks"])]
    loads = [0.0 for _ in range(instance["num_trucks"])]
    demand = instance["demand"]

    for station in sorted(instance["stations"], key=lambda node: demand[node], reverse=True):
        feasible_trucks = [
            k for k, load in enumerate(loads)
            if load + demand[station] <= instance["capacity"] + 1e-6
        ]
        if feasible_trucks:
            truck = min(feasible_trucks, key=lambda k: loads[k])
        else:
            truck = min(range(instance["num_trucks"]), key=lambda k: loads[k])
        assignments[truck].append(station)
        loads[truck] += demand[station]

    return assignments


def _neighbor(assignments: list[list[int]]) -> list[list[int]]:
    candidate = [route[:] for route in assignments]
    non_empty = [idx for idx, route in enumerate(candidate) if route]
    if not non_empty:
        return candidate

    move = random.choice(["swap", "reverse", "relocate"])

    if move == "swap" and sum(len(route) for route in candidate) >= 2:
        truck_a, truck_b = random.sample(non_empty, 2) if len(non_empty) >= 2 else [non_empty[0], non_empty[0]]
        pos_a = random.randrange(len(candidate[truck_a]))
        pos_b = random.randrange(len(candidate[truck_b]))
        candidate[truck_a][pos_a], candidate[truck_b][pos_b] = (
            candidate[truck_b][pos_b],
            candidate[truck_a][pos_a],
        )
    elif move == "reverse":
        truck = random.choice(non_empty)
        if len(candidate[truck]) >= 2:
            left, right = sorted(random.sample(range(len(candidate[truck])), 2))
            candidate[truck][left:right + 1] = reversed(candidate[truck][left:right + 1])
    else:
        source = random.choice(non_empty)
        station = candidate[source].pop(random.randrange(len(candidate[source])))
        target = random.randrange(len(candidate))
        insert_at = random.randrange(len(candidate[target]) + 1)
        candidate[target].insert(insert_at, station)

    return candidate


def _solution_from_routes(routes: Iterable[list[int]], instance: dict,
                          objective: float, runtime_seconds: float) -> dict:
    routes_idx = {k: route for k, route in enumerate(routes)}
    idx_to_id = instance["idx_to_id"]
    dist = instance["dist_km"]
    station_set = set(instance["stations"])
    demand = instance["demand"]
    total_distance = sum(_route_distance(route, dist) for route in routes_idx.values() if route)

    return {
        "status": "Feasible",
        "objective_value": round(objective, 4),
        "total_distance_km": round(total_distance, 4),
        "trucks_used": sum(1 for route in routes_idx.values() if route),
        "runtime_seconds": round(runtime_seconds, 4),
        "routes": {
            k: [idx_to_id[node] for node in route]
            for k, route in routes_idx.items()
        },
        "routes_idx": routes_idx,
        "route_loads_tonnes": {
            k: round(_truck_load(route, station_set, demand), 4)
            for k, route in routes_idx.items()
        },
        "instance": instance,
    }


def solve_sa(instance: dict, initial_temp: float = 40.0, cooling_rate: float = 0.995,
             min_temp: float = 0.001, max_iterations: int = 15000,
             seed: int = 42) -> dict:
    random.seed(seed)
    started = time.time()

    current = _initial_assignments(instance)
    current_cost, current_routes = _objective(current, instance)
    best = [route[:] for route in current]
    best_cost = current_cost
    best_routes = current_routes
    temperature = initial_temp

    for _ in range(max_iterations):
        if temperature < min_temp:
            break

        candidate = _neighbor(current)
        candidate_cost, candidate_routes = _objective(candidate, instance)
        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current = candidate
            current_cost = candidate_cost
            current_routes = candidate_routes

        if current_cost < best_cost:
            best = [route[:] for route in current]
            best_cost = current_cost
            best_routes = current_routes

        temperature *= cooling_rate

    final_cost, final_routes = _objective(best, instance)
    if final_cost <= best_cost:
        best_cost = final_cost
        best_routes = final_routes

    return _solution_from_routes(
        best_routes,
        instance,
        best_cost,
        time.time() - started,
    )


def print_solution(sol: dict) -> None:
    print("\n" + "=" * 60)
    print("SA SOLUTION SUMMARY")
    print("=" * 60)
    print(f"  Status          : {sol['status']}")
    print(f"  Objective       : {sol['objective_value']}")
    print(f"  Total distance  : {sol['total_distance_km']} km")
    print(f"  Trucks used     : {sol['trucks_used']} / {sol['instance']['num_trucks']}")
    print(f"  Runtime         : {sol['runtime_seconds']} s")
    print()
    for k, route in sol["routes"].items():
        if not route:
            print(f"  Truck {k}: (unused)")
            continue
        print(f"  Truck {k}: {' -> '.join(route)}")
        print(f"           load {sol['route_loads_tonnes'][k]:.1f} t")
    print("=" * 60)


def main() -> None:
    instance = load_instance()
    sol = solve_sa(instance)
    print_solution(sol)

    print("\nRunning constraint validation...")
    validation_results = validate_all(sol, instance)

    out = {
        k: v for k, v in sol.items()
        if k not in ("routes_idx", "instance")
    }
    out["validation_passed"] = all(validation_results.values())
    out["validation_details"] = {
        k: ("PASS" if v else "FAIL") for k, v in validation_results.items()
    }

    out_path = RESULTS_DIR / "sa_solution.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSolution saved to {out_path}")


if __name__ == "__main__":
    main()
