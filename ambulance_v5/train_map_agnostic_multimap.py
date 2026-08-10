#!/usr/bin/env python3
"""Balanced round-robin training across maps and heterogeneous intersections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VALIDATION_MARKER = "MAP_AGNOSTIC_VALIDATION_JSON="


def parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def load_maps(args: argparse.Namespace) -> list[Path]:
    records: list[dict[str, Any]] = []
    if args.manifest:
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        records = list(payload if isinstance(payload, list) else payload.get("maps", []))

    paths: list[Path] = []
    for record in records:
        if str(record.get("split", "train")) not in args.splits:
            continue
        value = record.get("net_file") or record.get("path")
        if value:
            paths.append(Path(value))
    paths.extend(Path(value) for value in parse_csv(args.maps))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise RuntimeError("No training maps selected. Pass --manifest or --maps.")
    return resolved


def cache_path(net_file: Path, cache_dir: Path) -> Path:
    stat = net_file.stat()
    token = f"{net_file}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return cache_dir / (hashlib.sha256(token).hexdigest()[:20] + ".json")


def passenger_lane_km(net_file: Path) -> float:
    total_meters = 0.0
    for _event, element in ET.iterparse(net_file, events=("end",)):
        if element.tag != "edge":
            continue
        if not str(element.get("id", "")).startswith(":"):
            for lane in element.findall("lane"):
                allow = set(str(lane.get("allow", "")).split())
                disallow = set(str(lane.get("disallow", "")).split())
                if "passenger" in disallow:
                    continue
                if allow and not ({"passenger", "private"} & allow):
                    continue
                total_meters += float(lane.get("length", 0.0) or 0.0)
        element.clear()
    return max(0.001, total_meters / 1000.0)


def topology_bucket(record: dict[str, Any]) -> tuple[str, str, str]:
    """Coarse shape bucket used to keep rare junction types in the gradient."""
    approaches = max(
        int(record.get("incoming_edges", 0)),
        int(record.get("outgoing_edges", 0)),
    )
    movements = int(record.get("movements", 0))
    phases = int(record.get("phases", 0))
    approach_bin = "2-" if approaches <= 2 else "3" if approaches == 3 else "4" if approaches == 4 else "5+"
    movement_bin = "1-4" if movements <= 4 else "5-8" if movements <= 8 else "9+"
    phase_bin = "2" if phases <= 2 else "3" if phases == 3 else "4" if phases == 4 else "5+"
    return approach_bin, movement_bin, phase_bin


def choose_topology_balanced(
    records: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Round-robin coarse topology buckets, cycling only when needed."""
    if not records or count <= 0:
        return []
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(topology_bucket(record), []).append(record)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)

    result: list[dict[str, Any]] = []
    offsets = {key: 0 for key in keys}
    while len(result) < count:
        for key in keys:
            values = buckets[key]
            result.append(values[offsets[key] % len(values)])
            offsets[key] += 1
            if len(result) >= count:
                break
    return result


def discover_tls(net_file: Path, cache_dir: Path, refresh: bool) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_path(net_file, cache_dir)
    if cache.exists() and not refresh:
        return list(json.loads(cache.read_text(encoding="utf-8")))

    env = os.environ.copy()
    env["TRAFFIC_NET_FILE"] = str(net_file)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "traffic_rl_map_agnostic_env.py"),
        "--net-file",
        str(net_file),
        "--list-tls-json",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    marker = "MAP_AGNOSTIC_TLS_JSON="
    line = next((line for line in result.stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        raise RuntimeError(f"TLS discovery produced no JSON for {net_file}:\n{result.stdout}")
    records = list(json.loads(line[len(marker) :]))
    cache.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def training_command(
    args: argparse.Namespace,
    net_file: Path,
    tls_id: str,
    first_task: bool,
    max_vehicle_center: int,
    target_vehicle_center: int,
    initial_vehicle_center: int,
    task_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_map_agnostic_policy.py"),
        "--net-file",
        str(net_file),
        "--tls-id",
        tls_id,
        "--model-path",
        args.model_path,
        "--timesteps",
        str(args.steps_per_tls),
        "--episode-seconds",
        str(args.episode_seconds),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(task_seed),
        "--max-vehicle-center",
        str(max_vehicle_center),
        "--target-vehicle-center",
        str(target_vehicle_center),
        "--initial-vehicle-center",
        str(initial_vehicle_center),
        "--spawn-batch-center",
        str(args.spawn_batch_center),
        "--density-spread",
        str(args.density_spread),
        "--initial-spread",
        str(args.initial_spread),
        "--observation-noise-std",
        str(args.observation_noise_std),
        "--sensor-scale-jitter",
        str(args.sensor_scale_jitter),
        "--sensor-dropout-prob",
        str(args.sensor_dropout_prob),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--device",
        args.device,
        "--torch-threads",
        str(args.torch_threads),
        "--no-progress-bar",
    ]
    if first_task and args.restart:
        command.append("--no-resume")
    else:
        command.append("--resume")
    return command


def checkpoint_zip(path: str | Path) -> Path:
    value = Path(path)
    return value if value.suffix == ".zip" else value.with_suffix(".zip")


def checkpoint_metadata(path: str | Path) -> Path:
    value = Path(path)
    base = value.with_suffix("") if value.suffix == ".zip" else value
    return base.parent / f"{base.name}_map_agnostic.json"


def validate_checkpoint(
    args: argparse.Namespace,
    round_number: int,
) -> dict[str, Any]:
    output = args.validation_dir / f"round_{round_number:03d}.json"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "validate_map_agnostic_policy.py"),
        "--manifest",
        str(args.validation_manifest or args.manifest),
        "--splits",
        args.validation_splits,
        "--model-path",
        args.model_path,
        "--output-json",
        str(output),
        "--seeds",
        args.validation_seeds,
        "--tls-per-map",
        str(args.validation_tls_per_map),
        "--episode-seconds",
        str(args.validation_episode_seconds),
        "--eval-steps",
        str(args.validation_eval_steps),
        "--target-density",
        str(args.validation_target_density),
        "--max-vehicle-center",
        str(args.max_vehicle_center),
        "--spawn-batch-center",
        str(args.spawn_batch_center),
        "--device",
        args.device,
    ]
    print(f"\n[validation after round {round_number}]")
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    marker_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith(VALIDATION_MARKER)),
        None,
    )
    if marker_line is None:
        raise RuntimeError(f"Validation returned no summary:\n{result.stdout}")
    return dict(json.loads(marker_line[len(VALIDATION_MARKER) :]))


def save_best_checkpoint(
    source: str | Path,
    destination: str | Path,
    validation: dict[str, Any],
) -> None:
    source_zip = checkpoint_zip(source)
    destination_zip = checkpoint_zip(destination)
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_zip, destination_zip)

    source_metadata = checkpoint_metadata(source)
    destination_metadata = checkpoint_metadata(destination)
    if source_metadata.exists():
        shutil.copy2(source_metadata, destination_metadata)
    validation_path = destination_zip.with_name(
        destination_zip.stem + "_validation.json"
    )
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(f"New best validation checkpoint: {destination_zip}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="train")
    parser.add_argument("--model-path", default="models/traffic_signal_map_agnostic_v2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-tls", type=int, default=10_000)
    parser.add_argument("--max-tls-per-map", type=int, default=24)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--topology-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample coarse intersection shapes evenly within each map.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--refresh-tls-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/map_agnostic_tls"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-every-round",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select the best checkpoint on manifest validation maps after each round.",
    )
    parser.add_argument("--validation-manifest", default="")
    parser.add_argument("--validation-splits", default="validation")
    parser.add_argument("--validation-seeds", default="9001,9002")
    parser.add_argument("--validation-tls-per-map", type=int, default=4)
    parser.add_argument("--validation-episode-seconds", type=int, default=600)
    parser.add_argument("--validation-eval-steps", type=int, default=120)
    parser.add_argument("--validation-target-density", type=float, default=6.0)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("runs/map_agnostic_validation"),
    )
    parser.add_argument(
        "--best-model-path",
        default="",
        help="Defaults to <model-path>_best when validation selection is enabled.",
    )

    parser.add_argument("--episode-seconds", type=int, default=900)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument(
        "--target-density-range",
        default="2.0,10.0",
        help="Sampled active vehicles per passenger lane-km; map-specific targets are capped by --max-vehicle-center.",
    )
    parser.add_argument("--initial-vehicle-center", type=int, default=300)
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--density-spread", type=float, default=0.35)
    parser.add_argument("--initial-spread", type=float, default=0.65)
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument("--sensor-scale-jitter", type=float, default=0.05)
    parser.add_argument("--sensor-dropout-prob", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    args.splits = set(parse_csv(args.splits))
    density_values = [float(value) for value in parse_csv(args.target_density_range)]
    if len(density_values) != 2 or density_values[0] <= 0 or density_values[1] < density_values[0]:
        parser.error("--target-density-range must be MIN,MAX with 0 < MIN <= MAX")
    args.target_density_min, args.target_density_max = density_values
    if args.validate_every_round and not (args.validation_manifest or args.manifest):
        parser.error("--validate-every-round requires --manifest or --validation-manifest")
    if not args.best_model_path:
        model_base = str(args.model_path)
        args.best_model_path = (
            model_base[:-4] if model_base.endswith(".zip") else model_base
        ) + "_best"

    maps = load_maps(args)
    per_map: dict[Path, list[dict[str, Any]]] = {}
    lane_km_by_map: dict[Path, float] = {}
    print("Discovering map-compatible TLS controllers...")
    for net_file in maps:
        records = discover_tls(net_file, args.cache_dir, args.refresh_tls_cache)
        if not records:
            print(f"WARNING: no usable TLS in {net_file}")
            continue
        per_map[net_file] = records
        lane_km_by_map[net_file] = passenger_lane_km(net_file)
        shapes = sorted({(r["movements"], r["phases"]) for r in records})
        buckets = sorted({topology_bucket(r) for r in records})
        print(
            f"  {net_file.name}: {len(records)} TLS, "
            f"passenger_lane_km={lane_km_by_map[net_file]:.1f}, "
            f"shapes={shapes}, topology_buckets={len(buckets)}"
        )

    if not per_map:
        raise RuntimeError("None of the selected maps has a usable TLS.")

    tasks_per_map = (
        args.max_tls_per_map
        if args.max_tls_per_map > 0
        else max(len(records) for records in per_map.values())
    )
    total_tasks = args.rounds * len(per_map) * tasks_per_map
    print("\nBalanced multi-map plan")
    print(f"  maps:              {len(per_map)}")
    print(f"  rounds:            {args.rounds}")
    print(f"  tasks/map/round:   {tasks_per_map}")
    print(f"  tasks:             {total_tasks}")
    print(f"  steps/task:        {args.steps_per_tls}")
    print(f"  requested steps:  {total_tasks * args.steps_per_tls}")
    print(f"  model:             {args.model_path}.zip")
    print(
        "  parallelism:       one learner; --num-envs parallel SUMO workers per task "
        "(never concurrent writers to one checkpoint)"
    )

    first_task = True
    best_validation_score = float("-inf")
    best_validation_path = checkpoint_zip(args.best_model_path).with_name(
        checkpoint_zip(args.best_model_path).stem + "_validation.json"
    )
    if not args.restart and best_validation_path.exists():
        try:
            best_validation_score = float(
                json.loads(best_validation_path.read_text(encoding="utf-8"))[
                    "selection_score"
                ]
            )
            print(f"Resuming best validation score: {best_validation_score:.6f}")
        except Exception:
            best_validation_score = float("-inf")
    for round_index in range(args.rounds):
        round_rng = random.Random(args.seed + round_index)
        tasks = []
        for net_file, records in per_map.items():
            if args.topology_balanced:
                chosen = choose_topology_balanced(records, tasks_per_map, round_rng)
            else:
                order = list(records)
                round_rng.shuffle(order)
                chosen = [order[index % len(order)] for index in range(tasks_per_map)]
            # Every map contributes the same number of tasks. Within a map the
            # default sampler also prevents a common topology from suppressing
            # uncommon asymmetric or higher-phase controllers.
            tasks.extend((net_file, str(record["tls_id"])) for record in chosen)
        if args.shuffle:
            round_rng.shuffle(tasks)

        for task_index, (net_file, tls_id) in enumerate(tasks, start=1):
            target_density = round_rng.uniform(
                args.target_density_min, args.target_density_max
            )
            target_center = min(
                args.max_vehicle_center,
                max(40, int(round(lane_km_by_map[net_file] * target_density))),
            )
            max_center = min(
                args.max_vehicle_center,
                max(target_center, int(round(target_center * 1.20))),
            )
            initial_center = min(
                target_center,
                max(40, min(args.initial_vehicle_center, int(round(target_center * 0.30)))),
            )
            command = training_command(
                args,
                net_file,
                tls_id,
                first_task,
                max_vehicle_center=max_center,
                target_vehicle_center=target_center,
                initial_vehicle_center=initial_center,
                task_seed=args.seed + round_index * 100_003 + task_index * 1009,
            )
            print(
                f"\n[round {round_index + 1}/{args.rounds} "
                f"task {task_index}/{len(tasks)}] {net_file.name} :: {tls_id} "
                f"density={target_density:.2f} veh/lane-km target={target_center}"
            )
            print(" ".join(command), flush=True)
            if not args.dry_run:
                env = os.environ.copy()
                env["TRAFFIC_NET_FILE"] = str(net_file)
                subprocess.run(command, cwd=ROOT, env=env, check=True)
            first_task = False

        if args.validate_every_round and not args.dry_run:
            validation = validate_checkpoint(args, round_index + 1)
            selection_score = float(validation["selection_score"])
            print(
                f"Validation selection score={selection_score:.6f} "
                f"(worst-map={float(validation['worst_map_score']):.6f}, "
                f"mean-map={float(validation['mean_map_score']):.6f})"
            )
            if selection_score > best_validation_score:
                best_validation_score = selection_score
                validation["selected_after_round"] = round_index + 1
                save_best_checkpoint(
                    args.model_path,
                    args.best_model_path,
                    validation,
                )

    print("\nMulti-map training complete.")


if __name__ == "__main__":
    main()
