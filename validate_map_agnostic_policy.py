#!/usr/bin/env python3
"""Balanced validation of one checkpoint on maps excluded from training."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from train_map_agnostic_multimap import (
    ROOT,
    choose_topology_balanced,
    discover_tls,
    load_maps,
    parse_csv,
    passenger_lane_km,
)


EVAL_MARKER = "MAP_AGNOSTIC_EVAL_JSON="


def evaluate_task(
    args: argparse.Namespace,
    net_file: Path,
    tls_id: str,
    seed: int,
    target_center: int,
) -> dict[str, Any]:
    max_center = max(target_center, int(round(target_center * 1.20)))
    initial_center = min(target_center, max(40, int(round(target_center * 0.30))))
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_map_agnostic_policy.py"),
        "--eval-only",
        "--net-file",
        str(net_file),
        "--tls-id",
        tls_id,
        "--model-path",
        args.model_path,
        "--seed",
        str(seed),
        "--episode-seconds",
        str(args.episode_seconds),
        "--eval-steps",
        str(args.eval_steps),
        "--eval-print-every",
        str(max(args.eval_steps + 1, 10_000)),
        "--max-vehicle-center",
        str(max_center),
        "--target-vehicle-center",
        str(target_center),
        "--initial-vehicle-center",
        str(initial_center),
        "--spawn-batch-center",
        str(args.spawn_batch_center),
        "--density-spread",
        "0",
        "--initial-spread",
        "0",
        "--observation-noise-std",
        "0",
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    env["TRAFFIC_NET_FILE"] = str(net_file)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    marker_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith(EVAL_MARKER)),
        None,
    )
    if marker_line is None:
        raise RuntimeError(
            f"Validation produced no result for {net_file.name}/{tls_id}:\n{result.stdout}"
        )
    record = dict(json.loads(marker_line[len(EVAL_MARKER) :]))
    print(
        f"  {net_file.name} :: {tls_id} seed={seed} "
        f"reward/step={float(record['mean_reward_per_step']):.5f} "
        f"queue={float(record['mean_queue_density']):.4f}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="validation")
    parser.add_argument("--model-path", default="models/traffic_signal_map_agnostic_v2")
    parser.add_argument("--output-json", type=Path, default=Path("map_agnostic_validation.json"))
    parser.add_argument("--seeds", default="9001,9002")
    parser.add_argument("--tls-per-map", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=1701)
    parser.add_argument("--episode-seconds", type=int, default=600)
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument("--target-density", type=float, default=6.0)
    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/map_agnostic_tls"))
    parser.add_argument("--refresh-tls-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.splits = set(parse_csv(args.splits))
    seeds = [int(value) for value in parse_csv(args.seeds)]
    if not seeds:
        parser.error("--seeds cannot be empty")
    if args.tls_per_map <= 0:
        parser.error("--tls-per-map must be positive")

    maps = load_maps(args)
    by_map: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    import random

    selection_rng = random.Random(args.selection_seed)
    for net_file in maps:
        tls_records = discover_tls(net_file, args.cache_dir, args.refresh_tls_cache)
        selected = choose_topology_balanced(
            tls_records,
            min(args.tls_per_map, len(tls_records)),
            selection_rng,
        )
        if not selected:
            print(f"WARNING: no usable validation TLS in {net_file}")
            continue
        target_center = min(
            args.max_vehicle_center,
            max(40, int(round(passenger_lane_km(net_file) * args.target_density))),
        )
        map_results: list[dict[str, Any]] = []
        print(f"Validating {net_file.name}: {len(selected)} TLS x {len(seeds)} seeds")
        for record in selected:
            tls_id = str(record["tls_id"])
            for seed in seeds:
                result = evaluate_task(args, net_file, tls_id, seed, target_center)
                result["topology"] = {
                    key: record.get(key)
                    for key in (
                        "movements",
                        "phases",
                        "incoming_edges",
                        "outgoing_edges",
                    )
                }
                map_results.append(result)
                all_records.append(result)
        by_map[str(net_file)] = map_results

    if not by_map:
        raise RuntimeError("No validation map produced a usable task.")
    map_scores = {
        name: statistics.fmean(float(row["mean_reward_per_step"]) for row in rows)
        for name, rows in by_map.items()
    }
    mean_map_score = statistics.fmean(map_scores.values())
    worst_map_score = min(map_scores.values())
    # Bias checkpoint selection toward the weakest held-out domain while still
    # retaining a signal from all validation maps.
    selection_score = 0.75 * worst_map_score + 0.25 * mean_map_score
    payload = {
        "schema_version": 1,
        "model_path": str(Path(args.model_path).resolve()),
        "splits": sorted(args.splits),
        "seeds": seeds,
        "map_scores": map_scores,
        "mean_map_score": mean_map_score,
        "worst_map_score": worst_map_score,
        "selection_score": selection_score,
        "records": all_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("MAP_AGNOSTIC_VALIDATION_JSON=" + json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
