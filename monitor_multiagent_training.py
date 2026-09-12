#!/usr/bin/env python3
"""Display live collection, optimization, throughput, and ETA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def render(path: Path) -> bool:
    if not path.exists():
        print(f"Waiting for {path}...")
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status", "unknown"))
    percent = float(payload.get("overall_percent", 0.0))
    completed = int(payload.get("completed_updates", 0))
    total = int(payload.get("total_updates", 0))
    transitions = int(payload.get("agent_transitions", 0))
    print(
        f"status={status}  progress={percent:6.2f}%  "
        f"updates={completed}/{total}  agent_transitions={transitions:,}"
    )
    active = dict(payload.get("active_rollouts", {}))
    if active:
        print("Active rollouts:")
        for name, fraction in sorted(active.items()):
            print(f"  {name:<32} {100.0 * float(fraction):6.2f}%")
    if status == "optimizing":
        optimization_completed = int(
            payload.get("optimization_steps_completed", 0)
        )
        optimization_total = int(payload.get("optimization_steps_total", 0))
        optimization_percent = float(payload.get("optimization_percent", 0.0))
        optimization_samples = int(payload.get("optimization_samples", 0))
        elapsed = max(
            0.0,
            float(payload.get("optimization_elapsed_seconds", 0.0)),
        )
        print(
            "Optimization: "
            f"{optimization_percent:6.2f}% "
            f"({optimization_completed}/{optimization_total} minibatches, "
            f"{optimization_samples:,} samples, {elapsed:.1f}s elapsed)"
        )
        optimization_eta = payload.get(
            "optimization_estimated_seconds_remaining"
        )
        if optimization_eta is not None:
            optimization_eta = max(0.0, float(optimization_eta))
            hours, remainder = divmod(int(optimization_eta), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                "Optimization ETA: "
                f"{hours:d}h {minutes:02d}m {seconds:02d}s"
            )
    metrics = payload.get("last_collection_metrics", [])
    if metrics:
        transitions_last = sum(int(item.get("transitions", 0)) for item in metrics)
        wall = max(float(item.get("wall_seconds", 0.0)) for item in metrics)
        throughput = transitions_last / max(1e-9, wall)
        print(
            f"Last collection: {transitions_last:,} transitions in "
            f"{wall:.1f}s ({throughput:.2f} transitions/s)"
        )
    eta = payload.get("estimated_seconds_remaining")
    if eta is not None and status not in {"complete", "failed"}:
        eta = max(0.0, float(eta))
        hours, remainder = divmod(int(eta), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Rough ETA from last update: {hours:d}h {minutes:02d}m {seconds:02d}s")
    return status in {"complete", "failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "progress_file",
        nargs="?",
        type=Path,
        default=Path("map_agnostic_multiagent_progress.json"),
    )
    parser.add_argument("--refresh", type=float, default=10.0)
    args = parser.parse_args()
    while True:
        done = render(args.progress_file)
        if done or args.refresh <= 0:
            break
        print()
        time.sleep(max(1.0, args.refresh))


if __name__ == "__main__":
    main()
