#!/usr/bin/env python3
"""Show per-job and aggregate partial percentages for map-agnostic evaluations."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


STEP_RE = re.compile(r"\[(?:native|all-model) seed\s+(\d+)\].*?step=\s*(\d+).*?t=\s*([0-9.]+)")
NAME_RE = re.compile(r"(native|allmodel)_seed(-?\d+)\.log$")


def default_root() -> Path:
    pointer = Path(".last_map_agnostic_eval_dir")
    if pointer.exists():
        return Path(pointer.read_text(encoding="utf-8").strip())
    raise FileNotFoundError("Pass an output directory or run launch_map_agnostic_eval.sh first.")


def progress_for(log: Path, episode_seconds: float) -> tuple[float, str]:
    result = log.with_suffix(".json")
    if result.exists():
        return 100.0, "done"
    if not log.exists():
        return 0.0, "pending"
    try:
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return 0.0, "unreadable"
    for line in reversed(lines[-2000:]):
        match = STEP_RE.search(line)
        if match:
            sim_time = float(match.group(3))
            return min(99.9, 100.0 * sim_time / max(1.0, episode_seconds)), "running"
    if any("Traceback (most recent call last)" in line for line in lines[-200:]):
        return 0.0, "failed"
    return 0.0, "starting"


def show(root: Path, episode_seconds: float) -> bool:
    jobs = []
    for log in sorted(root.glob("*_seed*.log")):
        match = NAME_RE.match(log.name)
        if not match:
            continue
        percent, status = progress_for(log, episode_seconds)
        jobs.append((match.group(1), int(match.group(2)), percent, status))
    if not jobs:
        print(f"No evaluation logs found in {root}")
        return False

    total = sum(job[2] for job in jobs) / len(jobs)
    print(f"[{root}] aggregate: {total:6.2f}% ({len(jobs)} jobs)")
    for controller, seed, percent, status in jobs:
        print(f"  {controller:8s} seed={seed:4d}: {percent:6.2f}%  {status}")
    return all(job[2] >= 100.0 for job in jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--episode-seconds", type=float, default=1200.0)
    parser.add_argument("--refresh", type=float, default=0.0, help="Refresh interval; zero prints once.")
    args = parser.parse_args()
    root = args.root or default_root()
    while True:
        done = show(root, args.episode_seconds)
        if done or args.refresh <= 0.0:
            break
        print()
        time.sleep(max(1.0, args.refresh))


if __name__ == "__main__":
    main()
