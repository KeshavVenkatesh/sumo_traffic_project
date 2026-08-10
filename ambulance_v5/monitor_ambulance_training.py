#!/usr/bin/env python3
"""Display live schema-v5 ambulance-override training progress."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


TERMINAL = {"complete", "failed"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def render(path: Path) -> bool:
    if not path.is_file():
        print(f"Waiting for {path}...")
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Progress file is temporarily unreadable: {exc}")
        return False

    status = str(payload.get("status", "unknown"))
    completed = int(payload.get("completed_updates", 0))
    total = int(payload.get("total_updates", 0))
    percentage = _number(payload.get("percentage"))
    active = int(payload.get("active_transitions", 0))
    print(
        f"status={status}  progress={percentage:6.2f}%  "
        f"updates={completed}/{total}  "
        f"ambulance-active transitions={active:,}"
    )
    best = _number(
        payload.get("best_validation_score"),
        float("-inf"),
    )
    validated = int(payload.get("last_validated_round", 0))
    if best != float("-inf"):
        print(
            f"best constrained validation={best:.4f}  "
            f"last validated round={validated}"
        )

    metrics = list(payload.get("last_rollout_metrics", ()))
    if metrics:
        completion = [
            _number(item.get("ambulance", {}).get("completion_rate"))
            for item in metrics
        ]
        trip_time = [
            _number(
                item.get("ambulance", {}).get(
                    "mean_response_time_s",
                    item.get("ambulance", {}).get(
                        "mean_trip_time_s"
                    ),
                )
            )
            for item in metrics
        ]
        reward = [
            _number(item.get("mean_emergency_reward"))
            for item in metrics
        ]
        print(
            "last rollout means: "
            f"completion={sum(completion) / len(completion):.3f}  "
            f"response_time={sum(trip_time) / len(trip_time):.1f}s  "
            f"reward={sum(reward) / len(reward):.4f}"
        )
        for item in metrics:
            ambulance = item.get("ambulance", {})
            print(
                f"  {Path(str(item.get('net_file', 'map'))).name:<28} "
                f"active={int(item.get('active_transitions', 0)):6d} "
                f"arrived={int(ambulance.get('arrived_total', 0)):2d}/"
                f"{int(ambulance.get('scheduled_total', 0)):2d} "
                f"failed={int(ambulance.get('failed_total', 0)):2d} "
                f"censored={int(ambulance.get('censored_total', 0)):2d}"
            )
    update = payload.get("last_update")
    if isinstance(update, dict):
        print(
            "last PPO update: "
            f"policy={_number(update.get('policy_loss')):.4f}  "
            f"value={_number(update.get('value_loss')):.4f}  "
            f"entropy={_number(update.get('entropy')):.4f}  "
            f"authority={_number(update.get('authority')):.3f}"
        )
    return status in TERMINAL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "progress_file",
        nargs="?",
        type=Path,
        default=Path("ambulance_v5_progress.json"),
    )
    parser.add_argument("--refresh", type=float, default=10.0)
    args = parser.parse_args()
    while True:
        done = render(args.progress_file)
        if done or args.refresh <= 0.0:
            break
        print()
        time.sleep(max(1.0, args.refresh))


if __name__ == "__main__":
    main()
