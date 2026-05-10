# Test Case — Kadıköy Cluster

Shared synthetic instance for validating all three optimization methods (MILP, GA, SA).

## Files

| File                    | Contents                                                       |
| ----------------------- | -------------------------------------------------------------- |
| `test_nodes.csv`        | Node table (1 depot, 6 transfer stations, 1 disposal facility) |
| `test_case_params.json` | Fleet size, truck capacity, speed, time horizon                |

## Node layout

```
D0  — Depot              (41.0082, 29.0378)  — Kadıköy centre proxy
S1  — Moda               (41.0031, 29.0260)  — demand 4.5 t
S2  — Kadıköy Merkez     (41.0086, 29.0304)  — demand 6.0 t
S3  — Yeldeğirmeni       (41.0051, 29.0355)  — demand 3.5 t
S4  — Fikirtepe          (41.0003, 29.0433)  — demand 5.0 t
S5  — Kozyatağı          (40.9905, 29.0661)  — demand 4.0 t
S6  — Bostancı           (40.9679, 29.0913)  — demand 5.5 t
F1  — Kömürcüoda         (41.1509, 29.3693)  — disposal (no demand)
```

**Total demand:** 28.5 tonnes  
**Fleet:** 2 trucks × 15 t capacity = 30 t total (comfortably feasible)

## Distances

All distances computed from Haversine formula inside the solver scripts.
Average speed assumed **40 km/h** for travel-time estimation.

## Time windows

- Stations: 60 – 480 min from start of horizon (service 07:00–15:00 window)
- Depot / Disposal: 0 – 960 min (open all day)
- Horizon: 960 minutes (16-hour operating day)

## How to use

```bash
# run MILP and validation
python src/milp_test_case.py

# run only validation on an existing solution dict
python src/validate.py
```

## Coordinate notes

Coordinates are **real geographic positions** in the Kadıköy–Kozyatağı corridor approximated
to the nearest transfer-station function. They are **not** official IBB station addresses —
this is a synthetic test cluster whose geometry is representative of the Anadolu side.
The disposal site **F1** uses the actual IBB Kömürcüoda Düzenli Depolama coordinates.
