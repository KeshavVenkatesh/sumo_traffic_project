#!/usr/bin/env python3
"""Fast held-out validation with the shared policy controlling every TLS."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO

from map_agnostic_multiagent_worker import rollout_worker_main
from train_map_agnostic_multimap import load_maps, parse_csv, passenger_lane_km


def model_zip(path: str | Path) -> Path:
    value = Path(path)
    return value if value.suffix == ".zip" else value.with_suffix(".zip")


def load_policy(path: str | Path):
    model = MaskablePPO.load(str(model_zip(path)), device="cpu")
    network = model.policy.map_network
    return (
        {
            key: value.detach().cpu().numpy().copy()
            for key, value in network.state_dict().items()
        },
        int(network.movement_encoder[0].out_features),
        len(network.graph_blocks),
    )


def worker_config(
    args: argparse.Namespace,
    net_file: Path,
    seed: int,
    rank: int,
    embed_dim: int,
    graph_layers: int,
) -> dict[str, Any]:
    return {
        "net_file": str(net_file),
        "passenger_lane_km": passenger_lane_km(net_file),
        "seed": seed,
        "worker_rank": rank,
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        "target_density_range": (
            args.target_density,
            args.target_density,
        ),
        "max_vehicle_center": args.max_vehicle_center,
        "spawn_batch_center": args.spawn_batch_center,
        "observation_noise_std": 0.0,
        "sensor_scale_jitter": 0.0,
        "sensor_dropout_prob": 0.0,
        "embed_dim": embed_dim,
        "graph_layers": graph_layers,
        "demand_routes": [],
        "use_libsumo": args.use_libsumo,
    }


def run_wave(
    ctx,
    tasks: list[tuple[Path, int, int]],
    args: argparse.Namespace,
    state_dict,
    embed_dim: int,
    graph_layers: int,
) -> list[dict[str, Any]]:
    workers = []
    try:
        for net_file, seed, rank in tasks:
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=rollout_worker_main,
                args=(
                    child,
                    worker_config(
                        args,
                        net_file,
                        seed,
                        rank,
                        embed_dim,
                        graph_layers,
                    ),
                ),
            )
            process.start()
            child.close()
            workers.append(
                {
                    "process": process,
                    "connection": parent,
                    "net_file": net_file,
                    "seed": seed,
                }
            )

        rollout_steps = int(
            math.ceil(args.episode_seconds / args.decision_seconds)
        )
        pending = {}
        for worker in workers:
            connection = worker["connection"]
            if not connection.poll(args.worker_start_timeout):
                raise TimeoutError(
                    f"Validation worker did not start: {worker['net_file']}"
                )
            ready = connection.recv()
            if ready.get("type") == "error":
                raise RuntimeError(ready["traceback"])
            if ready.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker message: {ready}")
            print(
                f"[validation ready] {worker['net_file'].name} "
                f"seed={worker['seed']}: {ready['tls']} TLS",
                flush=True,
            )
            connection.send(
                {
                    "cmd": "rollout",
                    "state_dict": state_dict,
                    "rollout_steps": rollout_steps,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "progress_interval": max(1, rollout_steps // 4),
                    "deterministic": True,
                }
            )
            pending[connection] = worker

        records = []
        while pending:
            ready_connections = wait(list(pending), timeout=60.0)
            if not ready_connections:
                dead = [
                    item["net_file"].name
                    for item in pending.values()
                    if not item["process"].is_alive()
                ]
                if dead:
                    raise RuntimeError(f"Validation workers exited: {dead}")
                print("[validation] workers still active", flush=True)
                continue
            for connection in ready_connections:
                message = connection.recv()
                worker = pending[connection]
                if message.get("type") == "progress":
                    print(
                        f"[validation] {worker['net_file'].name} "
                        f"seed={worker['seed']}: "
                        f"{100.0 * message['step'] / message['total']:.1f}%",
                        flush=True,
                    )
                    continue
                if message.get("type") == "error":
                    raise RuntimeError(message["traceback"])
                if message.get("type") != "rollout":
                    raise RuntimeError(f"Unexpected worker message: {message}")
                metrics = dict(message["metrics"])
                records.append(
                    {
                        "net_file": str(worker["net_file"]),
                        "seed": worker["seed"],
                        "tls": metrics["tls"],
                        "score": metrics["mean_topology_balanced_reward"],
                        "unweighted_mean_reward": metrics["mean_reward"],
                        "wall_seconds": metrics["wall_seconds"],
                        "transitions": metrics["transitions"],
                    }
                )
                del pending[connection]
        return records
    finally:
        for worker in workers:
            try:
                worker["connection"].send({"cmd": "close"})
            except Exception:
                pass
        for worker in workers:
            worker["process"].join(timeout=20)
            if worker["process"].is_alive():
                worker["process"].terminate()
                worker["process"].join(timeout=10)
            worker["connection"].close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="validation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seeds", default="9001,9002")
    parser.add_argument("--episode-seconds", type=int, default=600)
    parser.add_argument("--decision-seconds", type=float, default=10.0)
    parser.add_argument("--target-density", type=float, default=6.0)
    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-start-timeout", type=float, default=900.0)
    parser.add_argument(
        "--use-libsumo", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    del args.device  # workers intentionally use one CPU thread each
    args.splits = set(parse_csv(args.splits))
    seeds = [int(value) for value in parse_csv(args.seeds)]
    maps = load_maps(args)
    if not seeds:
        parser.error("--seeds cannot be empty")
    if args.decision_seconds <= 0 or args.episode_seconds <= 0:
        parser.error("episode and decision durations must be positive")

    state_dict, embed_dim, graph_layers = load_policy(args.model_path)
    tasks = [
        (net_file, seed, index)
        for index, (net_file, seed) in enumerate(
            (net_file, seed) for net_file in maps for seed in seeds
        )
    ]
    records: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    for start in range(0, len(tasks), max(1, args.workers)):
        records.extend(
            run_wave(
                ctx,
                tasks[start : start + max(1, args.workers)],
                args,
                state_dict,
                embed_dim,
                graph_layers,
            )
        )

    map_scores = {
        str(net_file): statistics.fmean(
            float(record["score"])
            for record in records
            if Path(record["net_file"]).resolve() == net_file.resolve()
        )
        for net_file in maps
    }
    mean_map_score = statistics.fmean(map_scores.values())
    worst_map_score = min(map_scores.values())
    selection_score = 0.75 * worst_map_score + 0.25 * mean_map_score
    payload = {
        "schema_version": 3,
        "validation_mode": "deterministic_all_tls_exact_sumo",
        "model_path": str(model_zip(args.model_path).resolve()),
        "splits": sorted(args.splits),
        "seeds": seeds,
        "map_scores": map_scores,
        "mean_map_score": mean_map_score,
        "worst_map_score": worst_map_score,
        "selection_score": selection_score,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "MAP_AGNOSTIC_VALIDATION_JSON="
        + json.dumps(payload, separators=(",", ":"))
    )
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
