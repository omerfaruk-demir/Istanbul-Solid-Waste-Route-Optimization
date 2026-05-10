from __future__ import annotations

import json
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


def _arrival_times(route: list[int], tau: dict[tuple[int, int], float],
                   service_time: dict[int, float]) -> dict[int, float]:
    arrivals: dict[int, float] = {}
    elapsed = 0.0
    for pos, node in enumerate(route):
        arrivals[node] = elapsed
        if pos < len(route) - 1:
            elapsed += service_time[node] + tau[node, route[pos + 1]]
    return arrivals


def _is_route_time_feasible(route: list[int], instance: dict) -> bool:
    arrivals = _arrival_times(route, instance["time_min"], instance["service_time"])
    for node, arrival in arrivals.items():
        if arrival < instance["tw_early"][node] - 1e-6:
            return False
        if arrival > instance["tw_late"][node] + 1e-6:
            return False
    return arrivals[route[-2]] + instance["service_time"][route[-2]] + instance["time_min"][route[-2], route[-1]] <= instance["horizon"] + 1e-6


def _build_routes(chromosome: list[int], split_points: list[int], instance: dict) -> list[list[int]]:
    depot = instance["depot"]
    disposal = instance["disposal"][0]
    routes: list[list[int]] = []
    start = 0
    for split in split_points:
        chunk = chromosome[start:split]
        routes.append([depot, *chunk, disposal, depot] if chunk else [])
        start = split
    return routes


def _random_split(chromosome: list[int], num_trucks: int) -> list[int]:
    if num_trucks <= 1:
        return [len(chromosome)]
    cuts = sorted(random.sample(range(len(chromosome) + 1), num_trucks - 1))
    return [*cuts, len(chromosome)]


def _repair_split(chromosome: list[int], split_points: list[int], instance: dict) -> list[int]:
    stations = set(instance["stations"])
    demand = instance["demand"]
    capacity = instance["capacity"]
    repaired: list[int] = []
    route_load = 0.0
    trucks_left = instance["num_trucks"]

    for pos, node in enumerate(chromosome, start=1):
        node_load = demand[node]
        remaining_stations = len(chromosome) - pos
        must_leave_for_remaining = max(0, trucks_left - 1)
        should_split = route_load + node_load > capacity + 1e-6
        original_split = pos in split_points and remaining_stations >= must_leave_for_remaining
        if route_load > 0 and (should_split or original_split) and trucks_left > 1:
            repaired.append(pos - 1)
            route_load = 0.0
            trucks_left -= 1
        route_load += node_load

    repaired.append(len(chromosome))

    while len(repaired) < instance["num_trucks"]:
        repaired.insert(-1, repaired[-2] if len(repaired) > 1 else 0)
    return repaired[:instance["num_trucks"]]


def _fitness(chromosome: list[int], split_points: list[int], instance: dict) -> tuple[float, list[list[int]]]:
    split_points = _repair_split(chromosome, split_points, instance)
    routes = _build_routes(chromosome, split_points, instance)
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    dist = instance["dist_km"]
    demand = instance["demand"]

    total_distance = 0.0
    penalty = 0.0
    visited: list[int] = []

    for route in routes:
        if not route:
            continue
        route_load = _truck_load(route, stations, demand)
        total_distance += _route_distance(route, dist)
        visited.extend(node for node in route if node in stations)
        if route_load > capacity:
            penalty += (route_load - capacity) * 1000.0
        if not _is_route_time_feasible(route, instance):
            penalty += 10000.0

    missing = set(stations).difference(visited)
    duplicates = len(visited) - len(set(visited))
    penalty += (len(missing) + duplicates) * 10000.0
    return total_distance + penalty, routes


def _ordered_crossover(parent_a: list[int], parent_b: list[int]) -> list[int]:
    left, right = sorted(random.sample(range(len(parent_a)), 2))
    child = [None] * len(parent_a)
    child[left:right + 1] = parent_a[left:right + 1]
    fill = [node for node in parent_b if node not in child]
    fill_idx = 0
    for idx, node in enumerate(child):
        if node is None:
            child[idx] = fill[fill_idx]
            fill_idx += 1
    return [int(node) for node in child]


def _mutate(chromosome: list[int], split_points: list[int],
            mutation_rate: float) -> tuple[list[int], list[int]]:
    child = chromosome[:]
    splits = split_points[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(child)), 2))
        child[i:j + 1] = reversed(child[i:j + 1])
    if random.random() < mutation_rate:
        splits = _random_split(child, len(splits))
    return child, splits


def _tournament(population: list[dict], size: int = 3) -> dict:
    candidates = random.sample(population, min(size, len(population)))
    return min(candidates, key=lambda item: item["fitness"])


def _solution_from_routes(routes: Iterable[list[int]], instance: dict,
                          objective: float, runtime_seconds: float) -> dict:
    routes_idx = {k: route for k, route in enumerate(routes)}
    idx_to_id = instance["idx_to_id"]
    dist = instance["dist_km"]
    stations = set(instance["stations"])
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
            k: round(_truck_load(route, stations, demand), 4)
            for k, route in routes_idx.items()
        },
        "instance": instance,
    }


def solve_ga(instance: dict, population_size: int = 80, generations: int = 400,
             elite_size: int = 8, mutation_rate: float = 0.2,
             seed: int = 42) -> dict:
    random.seed(seed)
    started = time.time()
    stations = instance["stations"][:]
    num_trucks = instance["num_trucks"]

    population: list[dict] = []
    for _ in range(population_size):
        chromosome = stations[:]
        random.shuffle(chromosome)
        split_points = _random_split(chromosome, num_trucks)
        fitness, routes = _fitness(chromosome, split_points, instance)
        population.append({
            "chromosome": chromosome,
            "split_points": split_points,
            "fitness": fitness,
            "routes": routes,
        })

    best = min(population, key=lambda item: item["fitness"])

    for _ in range(generations):
        population.sort(key=lambda item: item["fitness"])
        next_population = population[:elite_size]

        while len(next_population) < population_size:
            parent_a = _tournament(population)
            parent_b = _tournament(population)
            child_chromosome = _ordered_crossover(parent_a["chromosome"], parent_b["chromosome"])
            source_splits = random.choice([parent_a["split_points"], parent_b["split_points"]])
            child_chromosome, child_splits = _mutate(child_chromosome, source_splits, mutation_rate)
            fitness, routes = _fitness(child_chromosome, child_splits, instance)
            next_population.append({
                "chromosome": child_chromosome,
                "split_points": child_splits,
                "fitness": fitness,
                "routes": routes,
            })

        population = next_population
        current_best = min(population, key=lambda item: item["fitness"])
        if current_best["fitness"] < best["fitness"]:
            best = current_best

    return _solution_from_routes(
        best["routes"],
        instance,
        best["fitness"],
        time.time() - started,
    )


def print_solution(sol: dict) -> None:
    print("\n" + "=" * 60)
    print("GA SOLUTION SUMMARY")
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
    sol = solve_ga(instance)
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

    out_path = RESULTS_DIR / "ga_solution.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSolution saved to {out_path}")


if __name__ == "__main__":
    main()
