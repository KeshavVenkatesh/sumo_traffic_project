#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time


FATAL_RE = re.compile(
    r"Traceback|FileNotFoundError|FatalTraCIError|"
    r"Connection closed by SUMO|CalledProcessError|Segmentation fault",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\bt\s*=\s*([0-9]+(?:\.[0-9]+)?)")
STEP_RE = re.compile(r"\bstep\s*[=:]?\s*([0-9]+)", re.IGNORECASE)


def read_config(root: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    path = root / "run_config.env"
    if not path.exists():
        return config

    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def read_log_tail(path: Path, max_bytes: int = 262_144) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def running_process_text() -> str:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
        matching = []
        for line in result.stdout.splitlines():
            if re.search(
                r"\bpython(?:3)?\b.*\bcompare_native_sumo_vs_all_model\.py\b",
                line,
            ):
                matching.append(line)
        return "\n".join(matching)
    except OSError:
        return ""


def calculate_progress(
    root: Path,
    job: dict[str, str],
    episode_seconds: float,
    eval_steps: int,
    process_text: str,
) -> tuple[float, str, str]:
    json_path = root / job["json"]
    log_path = root / job["log"]

    if json_path.exists():
        return 100.0, "DONE", ""

    if not log_path.exists():
        return 0.0, "QUEUED", ""

    text = read_log_tail(log_path)
    time_values = [float(value) for value in TIME_RE.findall(text)]
    step_values = [int(value) for value in STEP_RE.findall(text)]

    if time_values:
        current = max(time_values)
        percent = min(99.9, 100.0 * current / episode_seconds)
        detail = f"t={current:.0f}/{episode_seconds:.0f}s"
    elif step_values:
        current_step = max(step_values)
        percent = min(99.9, 100.0 * current_step / eval_steps)
        detail = f"step={current_step}/{eval_steps}"
    else:
        percent = 0.0
        detail = "starting"

    if FATAL_RE.search(text):
        return percent, "FAILED", detail

    json_text = str(json_path)
    json_relative = job["json"]
    if json_text in process_text or json_relative in process_text:
        return percent, "RUNNING", detail

    return percent, "STOPPED", detail


def load_jobs(root: Path) -> list[dict[str, str]]:
    jobs_path = root / "jobs.tsv"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing job manifest: {jobs_path}")

    with jobs_path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render(root: Path, details: bool) -> bool:
    config = read_config(root)
    episode_seconds = float(config.get("EPISODE_SECONDS", "1200"))
    eval_steps = int(config.get("EVAL_STEPS", "2500"))
    jobs = load_jobs(root)
    process_text = running_process_text()
    process_count = len([line for line in process_text.splitlines() if line.strip()])

    rows = []
    for job in jobs:
        percent, status, detail = calculate_progress(
            root,
            job,
            episode_seconds,
            eval_steps,
            process_text,
        )
        rows.append({**job, "percent": percent, "status": status, "detail": detail})

    map_order = list(dict.fromkeys(row["map"] for row in rows))
    controller_order = list(dict.fromkeys(row["controller"] for row in rows))

    print(f"Evaluation directory: {root}")
    print(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for map_tag in map_order:
        print("=" * 86)
        print(map_tag.upper())
        print("=" * 86)

        for controller in controller_order:
            group = [
                row
                for row in rows
                if row["map"] == map_tag and row["controller"] == controller
            ]
            if not group:
                continue

            completed = sum(row["status"] == "DONE" for row in group)
            failed = sum(row["status"] == "FAILED" for row in group)
            stopped = sum(row["status"] == "STOPPED" for row in group)
            average = sum(float(row["percent"]) for row in group) / len(group)

            print(
                f"{controller:<24} "
                f"{completed:>3}/{len(group):<3} complete   "
                f"avg={average:6.1f}%   "
                f"failed={failed:<2} stopped={stopped:<2}"
            )

            if details:
                for row in sorted(group, key=lambda item: int(item["seed"])):
                    print(
                        f"  seed {int(row['seed']):>4}: "
                        f"{float(row['percent']):6.1f}%  "
                        f"{row['status']:<8} {row['detail']}"
                    )
        print()

    total = len(rows)
    completed = sum(row["status"] == "DONE" for row in rows)
    failed = sum(row["status"] == "FAILED" for row in rows)
    stopped = sum(row["status"] == "STOPPED" for row in rows)
    overall = sum(float(row["percent"]) for row in rows) / total if total else 0.0

    print("=" * 86)
    print(f"OVERALL: {completed}/{total} complete   average progress={overall:.1f}%")
    print(f"Running processes: {process_count}   failed={failed}   stopped={stopped}")
    print("=" * 86)

    return completed == total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor the two-map, multi-seed SUMO evaluation campaign."
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="Evaluation directory. Defaults to .last_two_map_eval_dir.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=10.0,
        help="Seconds between refreshes. Use 0 for a one-time report.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show every seed instead of only grouped progress.",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        pointer = Path(".last_two_map_eval_dir")
        if not pointer.exists():
            print("Missing .last_two_map_eval_dir", file=sys.stderr)
            return 1
        root = Path(pointer.read_text().strip())

    if not root.exists():
        print(f"Evaluation directory does not exist: {root}", file=sys.stderr)
        return 1

    try:
        while True:
            if args.refresh > 0:
                print("\033[2J\033[H", end="")

            complete = render(root, args.details)

            if args.refresh <= 0 or complete:
                return 0

            print(f"\nRefreshing every {args.refresh:g} seconds. Press Ctrl+C to stop.")
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Evaluations are still running.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Monitor error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
