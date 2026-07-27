#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import signal
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
ROOT_POINTER = PROJECT / "latest_two_map_four_model_eval_root.txt"
EXPECTED_CAMPAIGNS = 10
DEFAULT_SEEDS_PER_CAMPAIGN = 30
PROGRESS_RE = re.compile(r"\[progress\]\s+([0-9.]+)% complete")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def parse_manifest(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    path = root / "evaluation_manifest.txt"
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def latest_progress(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    matches = PROGRESS_RE.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    return float(matches[-1]) if matches else 0.0


def bar(percent: float, width: int = 40) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> None:
    print(f"Two-map RL evaluation — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 92)

    if not ROOT_POINTER.exists():
        print(f"Waiting for {ROOT_POINTER.name} ...")
        return

    root = Path(ROOT_POINTER.read_text(encoding="utf-8").strip())
    if not root.exists():
        print(f"Evaluation root does not exist: {root}")
        return

    manifest = parse_manifest(root)
    seeds_per_campaign = int(
        manifest.get("seed_count", DEFAULT_SEEDS_PER_CAMPAIGN)
    )
    expected_runs = EXPECTED_CAMPAIGNS * seeds_per_campaign

    campaign_dirs = sorted(path.parent for path in root.glob("*/*/wrapper.pid"))
    rows = []
    total_finished = 0
    running_campaigns = 0
    failed_campaigns = 0

    for campaign in campaign_dirs:
        pid_path = campaign / "wrapper.pid"
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1

        finished = len(list((campaign / "seed_logs").glob("*_seed*.json")))
        merged = (campaign / "merged.json").exists()
        alive = process_alive(pid)
        progress = latest_progress(campaign / "wrapper.log")

        if merged:
            status = "DONE"
            progress = 100.0
        elif alive:
            status = "RUNNING"
            running_campaigns += 1
        else:
            status = "FAILED/EXITED"
            failed_campaigns += 1

        total_finished += min(finished, seeds_per_campaign)
        map_name = campaign.parent.name
        controller = campaign.name
        rows.append(
            (map_name, controller, status, finished, progress, pid)
        )

    overall = 100.0 * total_finished / max(1, expected_runs)
    print(f"Root:               {root}")
    print(f"Completed runs:     {total_finished} / {expected_runs}")
    print(f"Overall completion: {bar(overall)} {overall:6.2f}%")
    print(f"Campaigns running:  {running_campaigns}")
    print(f"Campaigns failed:   {failed_campaigns}")
    print()
    print(
        f"{'MAP':12s} {'CONTROLLER':28s} {'STATUS':14s} "
        f"{'SEEDS':>9s} {'INNER %':>9s} {'PID':>8s}"
    )
    print("-" * 92)
    for map_name, controller, status, finished, progress, pid in rows:
        print(
            f"{map_name:12s} {controller:28.28s} {status:14s} "
            f"{finished:2d}/{seeds_per_campaign:<6d} {progress:8.2f}% {pid:8d}"
        )

    if len(campaign_dirs) < EXPECTED_CAMPAIGNS:
        print()
        print(
            f"Only {len(campaign_dirs)}/{EXPECTED_CAMPAIGNS} campaign PID files "
            "exist; the launcher may not have finished starting them."
        )

    if failed_campaigns:
        print()
        print("Latest output from failed/exited campaigns without merged results:")
        for campaign in campaign_dirs:
            if (campaign / "merged.json").exists():
                continue
            try:
                pid = int((campaign / "wrapper.pid").read_text().strip())
            except (OSError, ValueError):
                pid = -1
            if process_alive(pid):
                continue
            log = campaign / "wrapper.log"
            tail = log.read_text(errors="replace").splitlines()[-4:] if log.exists() else []
            print(f"\n{campaign.parent.name}/{campaign.name}:")
            for line in tail:
                print(f"  {line}")


if __name__ == "__main__":
    main()

