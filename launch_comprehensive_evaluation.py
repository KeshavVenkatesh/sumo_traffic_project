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
from fixed_demand import count_scheduled_vehicles, sha256_file
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
    reserved = {"native_sumo", "max_pressure", "all_model", "schema_v3"}
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


def parse_optional_model(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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
        identity_fields = {
            "fixed_demand",
            "scheduled_total",
            "demand_route_sha256",
            "demand_network_sha256",
            "demand_scenario",
            "demand_map_id",
        }
        if rows and all(
            row.get("controller") == job["controller"]
            and identity_fields.issubset(row)
            for row in rows
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
        script = ROOT / (
            "compare_native_sumo_vs_detector_realistic.py"
            if args.all_model_runner == "detector_realistic"
            else "compare_native_sumo_vs_map_agnostic.py"
        )
        command = [
            sys.executable,
            "-u",
            str(script),
            *common,
            "--skip-native",
            "--model-path",
            str(Path(args.model_path).resolve()),
        ]
    elif job["controller"] == "schema_v3":
        script = ROOT / "compare_native_sumo_vs_map_agnostic.py"
        command = [
            sys.executable,
            "-u",
            str(script),
            *common,
            "--skip-native",
            "--model-path",
            str(job["model_path"]),
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
    environment["DETECTOR_SENSOR_PROFILE"] = args.sensor_profile
    environment["DETECTOR_DECISION_SECONDS"] = str(args.model_update_period)
    environment["DETECTOR_NOISE_STD"] = str(args.detector_noise_std)
    environment["DETECTOR_CALIBRATION_JITTER"] = str(
        args.detector_calibration_jitter
    )
    environment["DETECTOR_DROPOUT_PROB"] = str(args.detector_dropout_prob)
    environment["DETECTOR_STUCK_PROB"] = str(args.detector_stuck_prob)
    environment["DETECTOR_MAX_LATENCY_DECISIONS"] = str(
        args.max_detector_latency_decisions
    )
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
    route_file = Path(job["demand_route"]).resolve()
    network_file = Path(job["net_file"]).resolve()
    scheduled_total = count_scheduled_vehicles(route_file)
    for row in rows:
        if job["controller"] not in {
            "native_sumo",
            "max_pressure",
            "all_model",
        }:
            row["controller"] = job["controller"]
        row["evaluation_wall_seconds"] = f"{wall_seconds:.6f}"
        # Promotion gates need controller-independent demand identity in every
        # raw row, not merely in the launcher job description.
        row.update(
            fixed_demand="1",
            scheduled_total=str(scheduled_total),
            demand_route_sha256=sha256_file(route_file),
            demand_network_sha256=sha256_file(network_file),
            demand_scenario=f"rate_{job['rate']:g}",
            demand_map_id=str(job["map_name"]),
        )
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


def verify_split_protocol_lock(
    lock_value: str,
    manifest_value: str,
    manifest_splits: set[str],
    benchmarks: dict[str, Path],
) -> dict[str, Any] | None:
    """Bind a primary evaluation to the frozen, post-generation test split."""
    if not lock_value:
        return None
    if not manifest_value:
        raise RuntimeError("--split-protocol-lock requires --manifest")
    if manifest_splits != {"test"}:
        raise RuntimeError(
            "A locked final evaluation requires --manifest-splits test"
        )
    lock_path = Path(lock_value).expanduser().resolve()
    manifest_path = Path(manifest_value).expanduser().resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "valid":
        raise RuntimeError(f"Split protocol lock is not valid: {lock_path}")
    generated = lock.get("generated_manifest")
    if not isinstance(generated, dict) or not generated.get("sha256"):
        raise RuntimeError(
            "Split protocol lock is preflight-only; rerun validation with "
            "--manifest after map generation"
        )
    actual_manifest_sha = file_sha256(manifest_path)
    if actual_manifest_sha != str(generated["sha256"]):
        raise RuntimeError(
            "Evaluation manifest does not match the frozen protocol lock"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_names = {
        str(record.get("name"))
        for record in manifest.get("maps", [])
        if str(record.get("split")) in manifest_splits
    }
    expected_names = set(
        str(name)
        for name in lock.get("new_corpus", {}).get("final_test_maps", [])
    )
    if not expected_names or selected_names != expected_names:
        raise RuntimeError(
            "Manifest test maps differ from the frozen protocol lock: "
            f"selected={sorted(selected_names)}, expected={sorted(expected_names)}"
        )
    if set(benchmarks) != expected_names:
        raise RuntimeError(
            "A locked primary evaluation may contain only frozen final-test "
            "maps; run legacy benchmarks in a separate output directory"
        )
    return {
        "path": str(lock_path),
        "sha256": file_sha256(lock_path),
        "manifest_sha256": actual_manifest_sha,
        "final_test_maps": sorted(expected_names),
    }


def protect_campaign_identity(
    args: argparse.Namespace,
    benchmarks: dict[str, Path],
    rates: list[float],
    seeds: list[int],
    schema_v3_model: Path | None,
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
        "all_model_runner": args.all_model_runner,
        "sensor_profile": args.sensor_profile,
        "detector_noise_std": args.detector_noise_std,
        "detector_calibration_jitter": args.detector_calibration_jitter,
        "detector_dropout_prob": args.detector_dropout_prob,
        "detector_stuck_prob": args.detector_stuck_prob,
        "max_detector_latency_decisions": args.max_detector_latency_decisions,
        "split_protocol_lock": args.verified_split_protocol,
        "schema_v3_model": (
            {
                "path": str(schema_v3_model),
                "sha256": file_sha256(schema_v3_model),
            }
            if schema_v3_model is not None
            else None
        ),
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
    parser.add_argument(
        "--split-protocol-lock",
        default="",
        help=(
            "Post-generation lock from validate_map_split_protocol.py. When "
            "set, only its exact frozen test maps may be evaluated."
        ),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--all-model-runner",
        choices=("map_agnostic", "detector_realistic"),
        default="map_agnostic",
        help="Observation/controller adapter used for the learned checkpoint.",
    )
    parser.add_argument(
        "--sensor-profile",
        choices=("loops", "camera", "mixed"),
        default="mixed",
        help="Used only with --all-model-runner detector_realistic.",
    )
    parser.add_argument("--detector-noise-std", type=float, default=0.0)
    parser.add_argument("--detector-calibration-jitter", type=float, default=0.0)
    parser.add_argument("--detector-dropout-prob", type=float, default=0.0)
    parser.add_argument("--detector-stuck-prob", type=float, default=0.0)
    parser.add_argument("--max-detector-latency-decisions", type=int, default=0)
    parser.add_argument(
        "--schema-v3-model",
        default="",
        help=(
            "Optional previous full-state schema-v3 checkpoint. When set, "
            "it is evaluated as a fourth controller on the exact same fixed "
            "demand as detector v4, MaxPressure, and native SUMO."
        ),
    )
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
    manifest_splits = set(parse_csv(args.manifest_splits))
    add_manifest_maps(
        benchmarks,
        args.manifest,
        manifest_splits,
    )
    args.verified_split_protocol = verify_split_protocol_lock(
        args.split_protocol_lock,
        args.manifest,
        manifest_splits,
        benchmarks,
    )
    rates = [float(value) for value in parse_csv(args.rates)]
    seeds = [int(value) for value in parse_csv(args.seeds)]
    schema_v3_model = parse_optional_model(args.schema_v3_model)
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
    detector_probabilities = (
        args.detector_dropout_prob,
        args.detector_stuck_prob,
    )
    if any(value < 0.0 or value > 1.0 for value in detector_probabilities):
        parser.error("Detector probabilities must be in [0, 1]")
    if args.detector_noise_std < 0.0 or args.detector_calibration_jitter < 0.0:
        parser.error("Detector noise and calibration jitter must be nonnegative")
    if args.max_detector_latency_decisions < 0:
        parser.error("--max-detector-latency-decisions must be nonnegative")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protect_campaign_identity(
        args, benchmarks, rates, seeds, schema_v3_model, legacy_models
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
                    *(["schema_v3"] if schema_v3_model is not None else []),
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
                            "model_path": (
                                schema_v3_model
                                if controller == "schema_v3"
                                else legacy_models.get(controller)
                            ),
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
