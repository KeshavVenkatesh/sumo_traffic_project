#!/usr/bin/env python3
"""Generate paired demand and evaluate Native, MaxPressure, and learned control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from generate_fixed_demand import find_random_trips, generate_one
from train_map_agnostic_multimap import parse_csv, passenger_lane_km


ROOT = Path(__file__).resolve().parent
PROGRESS_LOCK = threading.Lock()


def parse_benchmarks(raw: str) -> dict[str, Path]:
    result = {}
    for item in parse_csv(raw):
        if "=" not in item:
            raise ValueError(
                f"Benchmark must be NAME=NET_FILE, received {item!r}"
            )
        name, value = item.split("=", 1)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name.strip()] = path
    return result


def parse_legacy_models(values: list[str]) -> dict[str, Path]:
    result = {}
    reserved = {"native_sumo", "max_pressure", "all_model"}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Legacy model must be NAME=MODEL_PATH, received {item!r}"
            )
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if (
            not name
            or name in reserved
            or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
        ):
            raise ValueError(f"Invalid/reserved legacy model name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def add_manifest_maps(
    benchmarks: dict[str, Path], manifest: str, splits: set[str]
) -> None:
    if not manifest:
        return
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for record in payload.get("maps", []):
        if str(record.get("split", "")) not in splits:
            continue
        path = Path(record["net_file"]).resolve()
        benchmarks.setdefault(str(record.get("name", path.stem)), path)


def write_progress(
    path: Path,
    completed: int,
    total: int,
    failed: int,
    current: str,
    status: str,
) -> None:
    with PROGRESS_LOCK:
        payload = {
            "status": status,
            "completed_jobs": completed,
            "total_jobs": total,
            "failed_jobs": failed,
            "percent": 100.0 * completed / max(1, total),
            "current": current,
            "updated_at": time.time(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def run_job(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_csv = job["output_csv"]
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and all(
            row.get("controller") == job["controller"] for row in rows
        ):
            return {"status": "reused", "job": job}

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_csv.with_suffix(".json")
    log = output_csv.with_suffix(".log")
    common = [
        "--episode-seconds",
        str(args.episode_seconds),
        "--eval-steps",
        str(args.eval_steps),
        "--eval-print-every",
        str(args.print_interval),
        "--metrics-interval",
        str(args.metrics_interval),
        "--model-update-period",
        str(args.model_update_period),
        "--max-vehicle-center",
        str(args.max_vehicles),
        "--target-vehicle-center",
        "0",
        "--initial-vehicle-center",
        "0",
        "--spawn-batch-center",
        "1",
        "--compare-seeds",
        str(job["seed"]),
        "--demand-route-file",
        str(job["demand_route"]),
        "--stats-csv",
        str(output_csv),
        "--stats-json",
        str(output_json),
    ]
    if job["controller"] == "native_sumo":
        script = ROOT / "compare_native_sumo_vs_map_agnostic.py"
        command = [sys.executable, "-u", str(script), *common, "--skip-all-model"]
    elif job["controller"] == "max_pressure":
        script = ROOT / "compare_native_sumo_vs_max_pressure.py"
        command = [
            sys.executable,
            "-u",
            str(script),
            *common,
            "--skip-native",
        ]
    elif job["controller"] == "all_model":
        script = ROOT / "compare_native_sumo_vs_map_agnostic.py"
        command = [
            sys.executable,
            "-u",
            str(script),
            *common,
            "--skip-native",
            "--model-path",
            str(Path(args.model_path).resolve()),
        ]
    else:
        script = ROOT / "compare_native_sumo_vs_all_model.py"
        command = [
            sys.executable,
            "-u",
            str(script),
            *common,
            "--skip-native",
            "--model-path",
            str(job["model_path"]),
        ]

    environment = os.environ.copy()
    environment["TRAFFIC_NET_FILE"] = str(job["net_file"])
    environment["MAP_AGNOSTIC_MAX_ACTIVE_CAP"] = str(args.max_vehicles)
    start_wall = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    wall_seconds = time.monotonic() - start_wall
    if result.returncode != 0:
        return {
            "status": "failed",
            "job": job,
            "returncode": result.returncode,
            "log": str(log),
            "wall_seconds": wall_seconds,
        }
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if job["controller"] not in {
            "native_sumo",
            "max_pressure",
            "all_model",
        }:
            row["controller"] = job["controller"]
        row["evaluation_wall_seconds"] = f"{wall_seconds:.6f}"
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "status": "completed",
        "job": job,
        "wall_seconds": wall_seconds,
    }


def merge_condition(directory: Path) -> Path:
    rows = []
    for path in sorted((directory / "raw").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    output = directory / "paired_runs.csv"
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return output


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def protect_campaign_identity(
    args: argparse.Namespace,
    benchmarks: dict[str, Path],
    rates: list[float],
    seeds: list[int],
    legacy_models: dict[str, Path],
) -> None:
    model = Path(args.model_path).expanduser().resolve()
    if model.suffix != ".zip":
        model = model.with_suffix(".zip")
    if not model.is_file():
        raise FileNotFoundError(model)
    payload = {
        "schema_version": 1,
        "model_path": str(model),
        "model_sha256": file_sha256(model),
        "legacy_models": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in sorted(legacy_models.items())
        },
        "benchmarks": {
            name: str(path.resolve()) for name, path in sorted(benchmarks.items())
        },
        "rates": rates,
        "seeds": seeds,
        "episode_seconds": args.episode_seconds,
        "metrics_interval": args.metrics_interval,
        "model_update_period": args.model_update_period,
        "max_vehicles": args.max_vehicles,
        "min_distance": args.min_distance,
        "fringe_factor": args.fringe_factor,
    }
    path = args.output_dir / "campaign.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(
                f"{path} belongs to a different model/map/demand campaign. "
                "Use a new --output-dir to prevent mixed results."
            )
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default="fremont=new_map.net.xml,santaclara=santa_clara.net.xml",
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--manifest-splits", default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--legacy-model",
        action="append",
        default=[],
        metavar="NAME=MODEL_PATH",
        help=(
            "Optional historical five-action model; repeat the flag to add "
            "several direct fixed-demand comparisons."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("comprehensive_eval"))
    parser.add_argument("--rates", default="6,12,18")
    parser.add_argument("--seeds", default="1001,1002,1003")
    parser.add_argument("--episode-seconds", type=int, default=1200)
    parser.add_argument("--eval-steps", type=int, default=2500)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--demand-generation-workers", type=int, default=4)
    parser.add_argument("--max-vehicles", type=int, default=3000)
    parser.add_argument("--min-distance", type=float, default=300.0)
    parser.add_argument("--fringe-factor", type=float, default=3.0)
    parser.add_argument("--random-trips", default="")
    parser.add_argument("--force-demand", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--metrics-interval", type=float, default=20.0)
    parser.add_argument("--model-update-period", type=float, default=10.0)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("comprehensive_eval_progress.json"),
    )
    args = parser.parse_args()
    benchmarks = parse_benchmarks(args.benchmarks)
    add_manifest_maps(
        benchmarks,
        args.manifest,
        set(parse_csv(args.manifest_splits)),
    )
    rates = [float(value) for value in parse_csv(args.rates)]
    seeds = [int(value) for value in parse_csv(args.seeds)]
    legacy_models = parse_legacy_models(args.legacy_model)
    if not rates or not seeds:
        parser.error("--rates and --seeds cannot be empty")
    if min(rates) <= 0:
        parser.error("--rates must be positive")
    if (
        args.episode_seconds <= 0
        or args.eval_steps <= 0
        or args.max_parallel <= 0
        or args.demand_generation_workers <= 0
    ):
        parser.error("durations, steps, and worker counts must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protect_campaign_identity(
        args, benchmarks, rates, seeds, legacy_models
    )

    random_trips = find_random_trips(args.random_trips)
    demand_jobs = []
    demand_lookup = {}
    for map_name, net_file in benchmarks.items():
        lane_km = passenger_lane_km(net_file)
        for rate in rates:
            rate_tag = str(rate).replace(".", "p")
            period = max(0.05, 3600.0 / max(1e-9, rate * lane_km))
            for seed in seeds:
                route_file = (
                    args.output_dir
                    / "demand"
                    / map_name
                    / f"rate_{rate_tag}"
                    / f"seed_{seed}.rou.xml"
                )
                demand_lookup[(map_name, rate, seed)] = route_file
                demand_jobs.append(
                    (
                        map_name,
                        rate,
                        dict(
                            random_trips=random_trips,
                            net_file=net_file,
                            output=route_file,
                            seed=seed,
                            episode_seconds=args.episode_seconds,
                            period=period,
                            min_distance=args.min_distance,
                            fringe_factor=args.fringe_factor,
                            force=args.force_demand,
                        ),
                    )
                )

    print(f"Generating/reusing {len(demand_jobs)} paired demand files...")
    with ThreadPoolExecutor(
        max_workers=max(1, args.demand_generation_workers)
    ) as executor:
        futures = [
            executor.submit(generate_one, **parameters)
            for _map_name, _rate, parameters in demand_jobs
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"[demand {index}/{len(futures)}]", flush=True)

    jobs = []
    for map_name, net_file in benchmarks.items():
        for rate in rates:
            rate_tag = str(rate).replace(".", "p")
            condition = args.output_dir / map_name / f"rate_{rate_tag}"
            for seed in seeds:
                controller_names = [
                    "native_sumo",
                    "max_pressure",
                    "all_model",
                    *legacy_models,
                ]
                for controller in controller_names:
                    jobs.append(
                        {
                            "map_name": map_name,
                            "net_file": net_file,
                            "rate": rate,
                            "seed": seed,
                            "controller": controller,
                            "model_path": legacy_models.get(controller),
                            "demand_route": demand_lookup[
                                (map_name, rate, seed)
                            ],
                            "output_csv": condition
                            / "raw"
                            / f"{controller}_seed{seed}.csv",
                        }
                    )

    completed = 0
    failed = 0
    write_progress(
        args.progress_file, completed, len(jobs), failed, "", "running"
    )
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {executor.submit(run_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            failed += int(result["status"] == "failed")
            job = result["job"]
            current = (
                f"{job['map_name']} rate={job['rate']} "
                f"{job['controller']} seed={job['seed']}"
            )
            print(
                f"[evaluation {completed}/{len(jobs)}] "
                f"{result['status']}: {current}",
                flush=True,
            )
            write_progress(
                args.progress_file,
                completed,
                len(jobs),
                failed,
                current,
                "running",
            )

    for map_name in benchmarks:
        for rate in rates:
            rate_tag = str(rate).replace(".", "p")
            merge_condition(args.output_dir / map_name / f"rate_{rate_tag}")
    status = "complete" if failed == 0 else "failed"
    write_progress(
        args.progress_file,
        completed,
        len(jobs),
        failed,
        "",
        status,
    )
    if failed:
        raise RuntimeError(f"{failed} evaluation jobs failed")
    if not args.skip_analysis:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "analyze_comprehensive_evaluation.py"),
                str(args.output_dir),
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"Comprehensive evaluation complete: {args.output_dir}")


if __name__ == "__main__":
    main()
