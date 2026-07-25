from __future__ import annotations

import csv

from validate_controller_promotion import GateConfig, _load_runs, evaluate_campaign


def row(controller: str, seed: int, arrived: float):
    return {
        "controller": controller,
        "seed": seed,
        "fixed_demand": 1,
        "scheduled_total": 100,
        "demand_route_sha256": f"route-{seed}",
        "demand_network_sha256": "network",
        "demand_scenario": "medium",
        "demand_map_id": "map-a",
        "total_arrived": arrived,
        "mean_global_wait": 10.0,
        "mean_global_queue": 5.0,
        "mean_avg_speed_mps": 8.0,
    }


def test_existing_paired_csv_is_accepted(tmp_path):
    rows = [row("native_sumo", 1, 80), row("candidate", 1, 82)]
    path = tmp_path / "paired_runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    loaded = _load_runs(path)
    result = evaluate_campaign(
        {"map-a-medium": loaded},
        baseline="native_sumo",
        candidate="candidate",
        config=GateConfig(bootstrap_samples=0),
    )
    assert result["promoted"] is True
