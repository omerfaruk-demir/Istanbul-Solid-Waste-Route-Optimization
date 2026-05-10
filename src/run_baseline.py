from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.milp_test_case import load_instance, build_and_solve, print_solution
from src.validate import validate_all

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_RUNS = 3

def main():
    print("Loading test instance...")
    instance = load_instance()

    runs = []
    validation_results = None
    first_sol = None

    for run_idx in range(1, N_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx} of {N_RUNS}")
        print(f"{'='*60}")
        sol = build_and_solve(instance, time_limit=120, verbose=False)
        print_solution(sol)

        run_record = {
            "run": run_idx,
            "status": sol["status"],
            "objective_value": sol["objective_value"],
            "total_distance_km": sol["total_distance_km"],
            "trucks_used": sol["trucks_used"],
            "runtime_seconds": sol["runtime_seconds"],
            "routes": sol["routes"],
        }
        runs.append(run_record)

        if run_idx == 1 and sol["status"] == "Optimal":
            first_sol = sol
            print("\nRunning constraint validation on run 1 solution...")
            validation_results = validate_all(sol, instance)

    # Aggregate stats
    costs     = [r["objective_value"]    for r in runs]
    distances = [r["total_distance_km"]  for r in runs]
    runtimes  = [r["runtime_seconds"]    for r in runs]
    statuses  = [r["status"]             for r in runs]

    ground_truth = {
        "total_distance_km": distances[0] if distances else None,
        "objective_value":   costs[0]     if costs else None,
        "trucks_used":       runs[0]["trucks_used"] if runs else None,
        "routes":            runs[0]["routes"] if runs else None,
    }

    def safe_mean(lst):
        return round(statistics.mean(lst), 4) if lst else None

    def safe_stdev(lst):
        return round(statistics.stdev(lst), 4) if len(lst) > 1 else 0.0

    baseline = {
        "description": (
            "MILP baseline on Kadikoy synthetic test case. "
            "Ground truth for comparing GA and SA solutions."
        ),
        "test_case": "data/test_case/test_nodes.csv",
        "solver": "PuLP CBC",
        "n_runs": N_RUNS,
        "statuses": statuses,
        "aggregate": {
            "objective_mean":      safe_mean(costs),
            "objective_stdev":     safe_stdev(costs),
            "distance_mean_km":    safe_mean(distances),
            "distance_stdev_km":   safe_stdev(distances),
            "runtime_mean_s":      safe_mean(runtimes),
            "runtime_stdev_s":     safe_stdev(runtimes),
        },
        "ground_truth": ground_truth,
        "validation_passed": (
            all(validation_results.values()) if validation_results else None
        ),
        "validation_details": (
            {k: ("PASS" if v else "FAIL") for k, v in validation_results.items()}
            if validation_results else None
        ),
        "runs": runs,
    }

    out_path = RESULTS_DIR / "milp_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Baseline saved to {out_path}")
    print(f"  Objective (mean): {baseline['aggregate']['objective_mean']} km")
    print(f"  Distance  (mean): {baseline['aggregate']['distance_mean_km']} km")
    print(f"  Runtime   (mean): {baseline['aggregate']['runtime_mean_s']} s")
    print(f"  All constraints : {'PASS' if baseline['validation_passed'] else 'FAIL or N/A'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
