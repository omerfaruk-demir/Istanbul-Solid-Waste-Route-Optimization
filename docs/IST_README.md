# Full Istanbul Pipeline & Model Comparison

**Author:** Muhammed Nusret Yavuz  
**Scope:** Data pipeline for the full Istanbul instance + comparison framework for MILP / GA / SA

---

## What I Did

### 1. Data Pipeline (`src/data_loader.py`)

Reads the EDA outputs (`outputs/eda_output/`) produced from IBB open data and converts them into a standardised Python dict that all three solvers (MILP, GA, SA) can consume directly.

**Demand calculation:** The raw `total_waste_tonnes` and `total_trips` fields in `cvrp_instance.json` were used to compute **average load per trip** (`waste / trips`). This brings every station's demand below the 15-tonne truck capacity.

| Station | Monthly Total (t) | Trips | Per-Trip (t) |
|---------|-------------------|-------|--------------|
| S1 Baruthane | 92,017 | 19,490 | 4.72 |
| S2 Yenibosna | 112,836 | 17,639 | 6.40 |
| S3 Başakşehir | 18,331 | 2,429 | 7.55 |
| S4 Silivri | 6,838 | 1,104 | 6.19 |
| S5 Hasdal | 25,427 | 3,711 | 6.85 |
| S6 K.Bakkalköy | 75,808 | 13,447 | 5.64 |
| S7 Hekimbaşı | 9,170 | 1,261 | 7.27 |
| S8 Aydınlı | 30,617 | 4,350 | 7.04 |
| S9 Şile | 1,900 | 433 | 4.39 |

**Capacity note:** IBB does not publish per-truck capacity. A standard compactor capacity of **15 tonnes** was assumed.

The `load_istanbul_instance()` function returns the exact same dict format as `milp_test_case.load_instance()`, so switching GA and SA from the test case to full Istanbul requires a single import change:

```python
# Test case:
from src.milp_test_case import load_instance

# Full Istanbul:
from src.data_loader import load_istanbul_instance as load_instance
```

---

### 2. Istanbul Solver Scripts

| File | Description |
|------|-------------|
| `src/ga_istanbul.py` | Runs GA on the full Istanbul instance → `results/ga_solution_istanbul.json` |
| `src/sa_istanbul.py` | Runs SA on the full Istanbul instance → `results/sa_solution_istanbul.json` |

These are copies of the test-case scripts (`ga_test_case.py`, `sa_test_case.py`) with only the import changed.

---

### 3. Comparison Framework (`src/compare.py`)

Reads the solution JSONs from all three models and evaluates them on a common set of metrics.

**Computed metrics:**
- Total distance (km)
- Time-dependent travel cost (min) — using hourly congestion factors `α_r`
- Number of trucks used
- Runtime (seconds)
- Gap to MILP (%)
- Constraint violations (capacity, coverage)
- Detailed PASS/FAIL status for all 7 constraints

**Usage:**
```bash
# Test case — 3-model comparison (MILP as ground truth)
python src/compare.py --test test_case/test_nodes.csv --save results/comparison_test_case.json

# Full Istanbul — GA vs SA only (MILP intractable)
python src/compare.py --skip-milp --save results/comparison_istanbul.json
```

---

## Results

### Test Case (Kadıköy, 6 stations)

| Metric | MILP | GA | SA |
|--------|------|----|----|
| Status | Optimal | Feasible | Feasible |
| Distance (km) | 135.87 | 135.87 | 135.87 |
| TD Cost (min) | 138.17 | 138.17 | 138.17 |
| Trucks | 2 | 2 | 2 |
| Runtime (s) | 2.13 | 2.58 | 0.20 |
| Gap to MILP | — | 0.00% | 0.00% |
| Constraints | ALL PASS | ALL PASS | ALL PASS |

All three models found the same optimal solution.

### Full Istanbul (9 stations, 9 disposal facilities)

| Metric | GA | SA |
|--------|----|----|
| Status | Feasible | Feasible |
| Distance (km) | 454.41 | 465.64 |
| TD Cost (min) | 454.44 | 467.53 |
| Active Trucks | 4 / 9 | 5 / 9 |
| Runtime (s) | 3.80 | 0.37 |
| Constraints | ALL PASS | ALL PASS |

**GA found the better solution** (454 km vs 466 km); **SA was significantly faster** (0.37s vs 3.80s).

### MILP on Full Istanbul

The MILP solver (PuLP CBC) **failed to find a feasible solution** for the full Istanbul instance within a 300-second time limit. The combination of 19 nodes and 9 trucks produces too many binary variables for CBC to handle. This is an expected result — as stated in the project proposal, MILP is only practical for small instances, and metaheuristic methods (GA, SA) are needed for city-scale problems.

---

## MILP Bug Fixes (`src/milp_test_case.py`)

Three bugs were discovered and fixed when scaling MILP from the test case to full Istanbul:

1. **`no_s_to_depot` indentation error** — The constraint was nested inside the `for f in F` loop, causing duplicate constraint names when multiple disposal facilities exist. Moved outside the loop.
2. **Missing capacity constraint** — The `Eq. (7)` section contained a disposal-visit constraint instead of the actual capacity limit. The real capacity constraint was added.
3. **No disposal-to-disposal arc ban** — Trucks were visiting all 9 disposal facilities in sequence. Added `x[f1, f2, k] == 0` constraints to prevent this.

Even with these fixes, MILP remained intractable on the full Istanbul instance.

---

## File Map

```
src/
  data_loader.py            ← Istanbul data pipeline (NEW)
  ga_istanbul.py            ← GA on full Istanbul (NEW)
  sa_istanbul.py            ← SA on full Istanbul (NEW)
  compare.py                ← Comparison framework (NEW)
  milp_test_case.py         ← Bug fixes applied
  ga_test_case.py           ← Unchanged
  sa_test_case.py           ← Unchanged

results/
  ga_solution_istanbul.json ← GA Istanbul output (NEW)
  sa_solution_istanbul.json ← SA Istanbul output (NEW)
  comparison_test_case.json ← 3-model comparison (NEW)
  comparison_istanbul.json  ← GA vs SA comparison (NEW)
  ga_solution.json          ← Test case (unchanged)
  sa_solution.json          ← Test case (unchanged)
  milp_baseline.json        ← Test case (unchanged)
```