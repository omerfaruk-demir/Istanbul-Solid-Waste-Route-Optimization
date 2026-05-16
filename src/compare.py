"""
compare.py  —  Compare GA / SA / MILP solutions on common metrics.

Usage
-----
    # test case (synthetic Kadikoy nodes):
    python src/compare.py --test test_case/test_nodes.csv --save results/comparison_test.json

    # full Istanbul instance:
    python src/compare.py --save results/comparison_istanbul.json
"""

import csv as _csv
import json
import math
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ---------------------------------------------------------------------------
# Time-dependent travel time  τ_ij(t) = (d/v_free) * 60 * α_r
# ---------------------------------------------------------------------------

def td_time(dist_km, v_free_kmh, depart_min, alpha_r) -> float:
    hour  = int(depart_min // 60) % 24
    return (dist_km / v_free_kmh) * 60.0 * alpha_r[hour]


# ---------------------------------------------------------------------------
# Solution normaliser  —  handles GA/SA (single-run) and MILP (multi-run)
# ---------------------------------------------------------------------------

def extract_solution(raw: dict) -> dict:
    if "ground_truth" in raw:           # MILP baseline
        gt = raw["ground_truth"]
        return {
            "status":                     raw.get("statuses", ["Optimal"])[0],
            "routes":                     gt["routes"],
            "route_loads_tonnes":         gt.get("route_loads_tonnes", {}),
            "runtime_seconds":            raw["aggregate"]["runtime_mean_s"],
            "runtime_stdev":              raw["aggregate"].get("runtime_stdev_s", 0.0),
            "n_runs":                     raw.get("n_runs", 1),
            "validation_passed":          raw.get("validation_passed", None),
            "validation_details":         raw.get("validation_details", {}),
            "uses_time_dependent_travel": False,
        }
    return {                             # GA / SA
        "status":                     raw.get("status", "Unknown"),
        "routes":                     raw.get("routes", {}),
        "route_loads_tonnes":         raw.get("route_loads_tonnes", {}),
        "runtime_seconds":            raw.get("runtime_seconds", None),
        "runtime_stdev":              None,
        "n_runs":                     1,
        "validation_passed":          raw.get("validation_passed", None),
        "validation_details":         raw.get("validation_details", {}),
        "uses_time_dependent_travel": raw.get("uses_time_dependent_travel", False),
    }


# ---------------------------------------------------------------------------
# Node loaders
# ---------------------------------------------------------------------------

def load_nodes_from_csv(csv_path: Path) -> dict:
    """Load nodes_by_id from test_case/test_nodes.csv."""
    nodes_by_id = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            ntype = row["node_type"]
            if ntype == "TransferStation":
                ntype = "Transfer Station"
            nodes_by_id[row["node_id"]] = {
                "node_id":       row["node_id"],
                "name":          row["name"],
                "node_type":     ntype,
                "lat":           float(row["lat"]),
                "lon":           float(row["lon"]),
                "demand_tonnes": float(row["demand_tonnes"]),
            }
    return nodes_by_id


def load_nodes_from_instance(json_path: Path) -> dict:
    """Load nodes_by_id from cvrp_instance.json (full Istanbul)."""
    with open(json_path, encoding="utf-8") as f:
        inst = json.load(f)
    nodes_by_id = {}
    d = inst["depot"]
    nodes_by_id[d["node_id"]] = {**d, "node_type": "Depot", "demand_tonnes": 0.0}
    for s in inst["transfer_stations"]:
        per_trip = s["total_waste_tonnes"] / s["total_trips"] if s["total_trips"] > 0 else 0.0
        nodes_by_id[s["node_id"]] = {
            **s, "node_type": "Transfer Station",
            "demand_tonnes": per_trip}
    for f in inst["disposal_facilities"]:
        nodes_by_id[f["node_id"]] = {**f, "node_type": "Disposal", "demand_tonnes": 0.0}
    return nodes_by_id


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(sol, nodes_by_id, dist_lut, alpha_r, v_free, capacity) -> dict:
    routes = sol["routes"]
    total_dist = 0.0
    total_cost = 0.0
    stations_covered   = set()
    capacity_violations = 0
    route_details = {}

    for tid, route in routes.items():
        rdist = rtime = rload = 0.0
        t = 0.0
        for k in range(len(route) - 1):
            ni, nj = route[k], route[k+1]
            d   = dist_lut.get((ni, nj), 0.0)
            tt  = td_time(d, v_free, t, alpha_r)
            rdist += d
            rtime += tt
            t     += tt
            node = nodes_by_id.get(nj, {})
            if node.get("node_type") == "Transfer Station":
                stations_covered.add(nj)
                rload += node.get("demand_tonnes", 0.0)

        total_dist += rdist
        total_cost += rtime
        if rload > capacity:
            capacity_violations += 1
        route_details[tid] = {
            "distance_km": round(rdist, 4),
            "td_time_min": round(rtime, 2),
            "load_tonnes": round(rload, 2),
        }

    all_stations = {nid for nid, n in nodes_by_id.items()
                    if n.get("node_type") == "Transfer Station"}
    missing = all_stations - stations_covered

    return {
        "status":               sol["status"],
        "total_distance_km":    round(total_dist, 4),
        "total_td_cost_min":    round(total_cost, 4),
        "trucks_used":          len(routes),
        "runtime_seconds":      sol["runtime_seconds"],
        "runtime_stdev":        sol["runtime_stdev"],
        "n_runs":               sol["n_runs"],
        "uses_td_travel":       sol["uses_time_dependent_travel"],
        "capacity_violations":  capacity_violations,
        "coverage_violations":  len(missing),
        "missing_stations":     sorted(missing),
        "validation_passed":    sol["validation_passed"],
        "validation_details":   sol["validation_details"],
        "route_details":        route_details,
        "gap_to_milp_pct":      None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(results_dir="results", eda_dir="outputs/eda_output",
            test_case=None, save_path=None, skip_milp=False, verbose=True) -> dict:
    results_dir = Path(results_dir)
    eda_dir     = Path(eda_dir)

    # 1. Solutions
    if skip_milp:
        files = {
            "ga":   results_dir / "ga_solution_istanbul.json",
            "sa":   results_dir / "sa_solution_istanbul.json",
        }
        print("[INFO] Skipping MILP (intractable for full Istanbul)")
    else:
        files = {
            "ga":   results_dir / "ga_solution.json",
            "sa":   results_dir / "sa_solution.json",
            "milp": results_dir / "milp_baseline.json",
        }
    solutions = {}
    for name, path in files.items():
        if not path.exists():
            print(f"[WARN] {path} not found — skipping {name.upper()}")
            continue
        with open(path, encoding="utf-8") as f:
            solutions[name] = extract_solution(json.load(f))
    if not solutions:
        raise FileNotFoundError(f"No solution files in {results_dir}")

    # 2. Traffic data
    with open(eda_dir / "alpha_r.json", encoding="utf-8") as f:
        aj = json.load(f)
    alpha_r = {int(k): float(v) for k, v in aj["alpha_r"].items()}
    v_free  = float(aj["v_free_kmh"])

    # 3. Nodes
    if test_case:
        nodes_by_id = load_nodes_from_csv(Path(test_case))
        capacity = 15.0
        print(f"[INFO] Test case mode — {len(nodes_by_id)} nodes from {test_case}")
    else:
        nodes_by_id = load_nodes_from_instance(eda_dir / "cvrp_instance.json")
        capacity = 15.0
        print(f"[INFO] Full Istanbul mode — {len(nodes_by_id)} nodes")

    # 4. Distance lookup
    dist_lut = {}
    for ni, n1 in nodes_by_id.items():
        for nj, n2 in nodes_by_id.items():
            dist_lut[(ni, nj)] = 0.0 if ni == nj else haversine_km(
                n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    # 5. Metrics
    metrics = {}
    for name, sol in solutions.items():
        metrics[name] = compute_metrics(
            sol, nodes_by_id, dist_lut, alpha_r, v_free, capacity)

    # 6. Gap to MILP
    if "milp" in metrics:
        base = metrics["milp"]["total_td_cost_min"]
        for name, m in metrics.items():
            if name != "milp" and base and base > 0:
                m["gap_to_milp_pct"] = round(
                    (m["total_td_cost_min"] - base) / base * 100, 2)

    if verbose:
        _print_table(metrics)

    report = {"metrics": metrics, "summary": _build_summary(metrics)}

    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved → {sp}")

    return report


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def _print_table(metrics):
    order   = ["milp", "ga", "sa"]
    present = [m for m in order if m in metrics]
    col, lbl = 22, 32

    sep = "=" * (lbl + col * len(present))
    print()
    print(sep)
    print(f"{'METRIC':<{lbl}}" + "".join(f"{m.upper():>{col}}" for m in present))
    print(sep)

    def row(label, key, fmt="{}", suffix=""):
        vals = []
        for m in present:
            v = metrics[m].get(key)
            s = "—" if v is None else fmt.format(v) + suffix
            vals.append(f"{s:>{col}}")
        print(f"{label:<{lbl}}" + "".join(vals))

    row("Status",                 "status")
    row("Total distance (km)",    "total_distance_km",  "{:.2f}")
    row("TD travel cost (min)",   "total_td_cost_min",  "{:.2f}")
    row("Trucks used",            "trucks_used")
    row("Runtime (s)",            "runtime_seconds",    "{:.4f}")
    row("Runtime stdev (s)",      "runtime_stdev",      "{:.4f}")
    row("Runs averaged",          "n_runs")
    row("Uses TD travel",         "uses_td_travel")
    row("Gap to MILP (%)",        "gap_to_milp_pct",    "{:+.2f}", "%")
    row("Capacity violations",    "capacity_violations")
    row("Coverage violations",    "coverage_violations")
    row("Validation passed",      "validation_passed")

    print("-" * (lbl + col * len(present)))
    print()
    print("CONSTRAINT DETAILS")
    print("-" * (lbl + col * len(present)))
    for check in ["coverage", "capacity", "flow_conservation",
                  "depot_constraints", "time_windows",
                  "schedule_consistency", "waste_balance"]:
        vals = [f"{metrics[m].get('validation_details',{}).get(check,'—'):>{col}}"
                for m in present]
        print(f"  {check:<{lbl-2}}" + "".join(vals))
    print()

    for model in present:
        rd = metrics[model].get("route_details", {})
        if rd:
            print(f"ROUTE DETAILS — {model.upper()}")
            print(f"  {'Truck':<8}{'Distance (km)':>16}{'TD Time (min)':>16}{'Load (t)':>12}")
            for tid, r in rd.items():
                print(f"  {tid:<8}{r['distance_km']:>16.2f}"
                      f"{r['td_time_min']:>16.2f}{r['load_tonnes']:>12.2f}")
            print()


def _build_summary(metrics):
    def best(key, lower=True):
        c = {m: v for m in metrics if (v := metrics[m].get(key)) is not None}
        return (min if lower else max)(c, key=lambda m: c[m]) if c else None

    return {
        "best_cost":     best("total_td_cost_min"),
        "best_distance": best("total_distance_km"),
        "fastest":       best("runtime_seconds"),
        "fewest_trucks": best("trucks_used"),
        "all_feasible":  all(
            m.get("capacity_violations", 1) == 0 and
            m.get("coverage_violations",  1) == 0
            for m in metrics.values()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare GA / SA / MILP solutions")
    p.add_argument("--results", default="results",
                   help="Dir with solution JSONs (default: results/)")
    p.add_argument("--eda",     default="outputs/eda_output",
                   help="Dir with EDA outputs (default: outputs/eda_output/)")
    p.add_argument("--test",    default=None,
                   help="Path to test_nodes.csv for test-case mode "
                        "(e.g. test_case/test_nodes.csv)")
    p.add_argument("--skip-milp", action="store_true",
                   help="Skip MILP (use for full Istanbul where MILP is intractable)")
    p.add_argument("--save",    default=None,
                   help="Save JSON report to this path")
    args = p.parse_args()

    report  = compare(results_dir=args.results, eda_dir=args.eda,
                      test_case=args.test, save_path=args.save,
                      skip_milp=args.skip_milp, verbose=True)
    summary = report["summary"]
    print("SUMMARY")
    print(f"  Best cost model    : {str(summary.get('best_cost','—')).upper()}")
    print(f"  Best distance model: {str(summary.get('best_distance','—')).upper()}")
    print(f"  Fastest model      : {str(summary.get('fastest','—')).upper()}")
    print(f"  All feasible       : {summary.get('all_feasible')}")