from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_traffic_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"v_free_kmh": None, "alpha_r": {str(hour): 1.0 for hour in range(24)}}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    alpha_r = {
        str(hour): float(data.get("alpha_r", {}).get(str(hour), 1.0))
        for hour in range(24)
    }
    return {
        "v_free_kmh": data.get("v_free_kmh"),
        "alpha_r": alpha_r,
    }


def clock_hour(departure_min: float, start_hour: int = 7) -> int:
    return int((start_hour + departure_min // 60) % 24)


def congestion_factor(instance: dict, departure_min: float) -> float:
    traffic = instance.get("traffic_profile") or {}
    alpha_r = traffic.get("alpha_r") or {}
    hour = clock_hour(departure_min, instance.get("traffic_start_hour", 7))
    return float(alpha_r.get(str(hour), 1.0))


def travel_time_min(instance: dict, i: int, j: int, departure_min: float,
                    use_traffic: bool = True) -> float:
    base_time = instance["time_min"][i, j]
    if not use_traffic:
        return base_time
    return base_time * congestion_factor(instance, departure_min)


def route_schedule(route: list[int], instance: dict,
                   use_traffic: bool = True) -> dict[str, Any]:
    arrivals: dict[int, float] = {}
    travel_total = 0.0
    elapsed = 0.0

    for pos, node in enumerate(route):
        if pos > 0:
            elapsed = max(elapsed, instance["tw_early"][node])
        arrivals[node] = elapsed
        if pos < len(route) - 1:
            nxt = route[pos + 1]
            arc_time = travel_time_min(instance, node, nxt, elapsed, use_traffic)
            travel_total += arc_time
            elapsed += instance["service_time"][node] + arc_time

    return {
        "arrivals": arrivals,
        "travel_time_min": travel_total,
        "route_end_min": elapsed,
    }
