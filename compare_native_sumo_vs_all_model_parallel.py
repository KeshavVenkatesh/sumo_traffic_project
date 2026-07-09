#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace


STEP_RE = re.compile(
    r"\[(native|all-model|fixed-cycle|fixed) seed\s+(\d+)\]\s+step=\s*(\d+)"
)


class ProgressTracker:
    def __init__(self, seeds: list[int], stages: list[str], eval_steps: int):
        self.seeds = list(seeds)
        self.stages = list(stages)
        self.eval_steps = max(1, int(eval_steps))
        self.total_units = len(self.seeds) * len(self.stages) * self.eval_steps
        self.stage_index = {stage: i for i, stage in enumerate(self.stages)}

        # accepted aliases from log labels
        if "fixed_cycle" in self.stage_index:
            self.stage_index["fixed"] = self.stage_index["fixed_cycle"]
            self.stage_index["fixed-cycle"] = self.stage_index["fixed_cycle"]

        self.units_by_seed = {seed: 0 for seed in self.seeds}
        self.finished_seeds: set[int] = set()
        self.lock = threading.Lock()
        self.last_print_time = 0.0
        self.last_print_pct = -1.0

    def _print_locked(self, reason: str = "") -> None:
        done = sum(self.units_by_seed.values())
        pct = 100.0 * done / max(1, self.total_units)

        now = time.time()
        should_print = (
            pct >= 100.0
            or pct - self.last_print_pct >= 0.25
            or now - self.last_print_time >= 20.0
        )

        if not should_print:
            return

        self.last_print_time = now
        self.last_print_pct = pct

        seed_status = ", ".join(
            f"{seed}:{100.0 * self.units_by_seed[seed] / max(1, len(self.stages) * self.eval_steps):5.1f}%"
            for seed in self.seeds
        )

        msg = f"[progress] {pct:6.2f}% complete"
        if reason:
            msg += f" | {reason}"
        msg += f" | per-seed: {seed_status}"
        print(msg, flush=True)

    def update_from_line(self, line: str) -> None:
        m = STEP_RE.search(line)
        if not m:
            return

        stage_raw, seed_raw, step_raw = m.groups()
        seed = int(seed_raw)
        step = min(int(step_raw), self.eval_steps)

        stage = stage_raw.replace("-", "_")
        if stage not in self.stage_index:
            return

        idx = self.stage_index[stage]
        units = idx * self.eval_steps + step

        with self.lock:
            if seed in self.units_by_seed:
                self.units_by_seed[seed] = max(self.units_by_seed[seed], units)
                self._print_locked(f"seed {seed} {stage_raw} step {step}/{self.eval_steps}")

    def mark_seed_done(self, seed: int) -> None:
        with self.lock:
            self.finished_seeds.add(seed)
            self.units_by_seed[seed] = len(self.stages) * self.eval_steps
            self._print_locked(f"seed {seed} finished")

    def mark_seed_failed(self, seed: int) -> None:
        with self.lock:
            self._print_locked(f"seed {seed} failed")


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [42]


def infer_stages(extra_args: list[str]) -> list[str]:
    skip_native = "--skip-native" in extra_args
    skip_all_model = "--skip-all-model" in extra_args
    include_fixed = "--include-fixed-cycle" in extra_args

    stages = []
    if not skip_native:
        stages.append("native")
    if include_fixed:
        stages.append("fixed_cycle")
    if not skip_all_model:
        stages.append("all_model")

    if not stages:
        stages = ["native"]

    return stages


def get_extra_value(extra_args: list[str], flag: str, default: int) -> int:
    if flag not in extra_args:
        return default
    idx = extra_args.index(flag)
    if idx + 1 >= len(extra_args):
        return default
    try:
        return int(float(extra_args[idx + 1]))
    except Exception:
        return default


def run_one_seed(
    seed: int,
    args: argparse.Namespace,
    extra_args: list[str],
    tracker: ProgressTracker,
) -> tuple[int, Path, Path, Path, int]:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.stats_json).with_suffix("").name
    seed_json = log_dir / f"{stem}_seed{seed}.json"
    seed_csv = log_dir / f"{stem}_seed{seed}.csv"
    seed_log = log_dir / f"{stem}_seed{seed}.log"

    cmd = [
        sys.executable,
        "-u",
        args.script,
        *extra_args,
        "--compare-seeds",
        str(seed),
        "--stats-csv",
        str(seed_csv),
        "--stats-json",
        str(seed_json),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with seed_log.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            tracker.update_from_line(line)

        code = proc.wait()

    if code == 0:
        tracker.mark_seed_done(seed)
    else:
        tracker.mark_seed_failed(seed)

    return seed, seed_json, seed_csv, seed_log, code


def merge_outputs(results, args: argparse.Namespace) -> None:
    rows = []

    for seed, seed_json, _seed_csv, seed_log, code in sorted(results):
        if code != 0:
            raise RuntimeError(f"Seed {seed} failed. Check {seed_log}")
        if not seed_json.exists():
            raise FileNotFoundError(f"Missing seed JSON for seed {seed}: {seed_json}")

        payload = json.loads(seed_json.read_text(encoding="utf-8"))
        rows.extend(payload.get("runs", []))

    if not rows:
        raise RuntimeError("No rows found in seed JSON files.")

    import compare_native_sumo_vs_all_model as base

    print()
    base.print_native_vs_model_table(rows)

    # Write merged outputs using the original helper if possible.
    out_args = SimpleNamespace(stats_csv=args.stats_csv, stats_json=args.stats_json)
    try:
        base.write_outputs(rows, out_args)
        return
    except Exception as exc:
        print(f"Original write_outputs failed, writing manually: {exc}")

    fieldnames = sorted(set().union(*(row.keys() for row in rows)))

    csv_path = Path(args.stats_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(args.stats_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "runs": rows,
                "aggregate": base.aggregate_by_controller(rows),
                "parallel_eval": {
                    "stats_csv": str(csv_path),
                    "stats_json": str(json_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote merged CSV: {csv_path}")
    print(f"Wrote merged JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run native-vs-all-model eval seeds in parallel with live percent progress."
    )

    parser.add_argument("--script", default="compare_native_sumo_vs_all_model.py")
    parser.add_argument("--compare-seeds", default="42,43,44")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--stats-csv", default="native_vs_model_parallel.csv")
    parser.add_argument("--stats-json", default="native_vs_model_parallel.json")
    parser.add_argument("--log-dir", default="parallel_eval_logs")

    args, extra_args = parser.parse_known_args()

    seeds = parse_seeds(args.compare_seeds)
    jobs = max(1, min(int(args.jobs), len(seeds)))
    eval_steps = get_extra_value(extra_args, "--eval-steps", 10000)
    stages = infer_stages(extra_args)

    tracker = ProgressTracker(seeds=seeds, stages=stages, eval_steps=eval_steps)

    print("=" * 100, flush=True)
    print("Parallel native-vs-all-model evaluation", flush=True)
    print("=" * 100, flush=True)
    print(f"seeds: {seeds}", flush=True)
    print(f"jobs: {jobs}", flush=True)
    print(f"stages per seed: {stages}", flush=True)
    print(f"eval_steps per stage: {eval_steps}", flush=True)
    print(f"script: {args.script}", flush=True)
    print(f"log dir: {args.log_dir}", flush=True)
    print("=" * 100, flush=True)

    started = time.time()
    tracker._print_locked("started")

    results = []

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(run_one_seed, seed, args, extra_args, tracker)
            for seed in seeds
        ]

        for fut in as_completed(futures):
            result = fut.result()
            seed, seed_json, seed_csv, seed_log, code = result
            print(
                f"[seed {seed}] finished with returncode={code}; "
                f"json={seed_json}; csv={seed_csv}; log={seed_log}",
                flush=True,
            )
            results.append(result)

    merge_outputs(results, args)

    elapsed = time.time() - started
    print(f"[progress] 100.00% complete | total elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
