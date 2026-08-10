#!/usr/bin/env python3
"""Paired five-way evaluation for ambulance routing and signal priority.

Every ablation for a scenario receives the same immutable background route
file, SUMO seed, ambulance O/D schedule, spawn times, and free-flow reference
routes.  The evaluator fails if those schedule fingerprints differ.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fixed_demand import (
    count_scheduled_vehicles,
    fixed_demand_vehicle_type_is_safe,
    sha256_file,
)
from ambulance_system import AMBULANCE_ID_PREFIX
from map_agnostic_tls import (
    DEFAULT_MAX_GREEN,
    DEFAULT_MIN_GREEN,
    DEFAULT_REQUIRED_EXIT_GAP_METERS,
)
from train_map_agnostic_multimap import (
    load_maps,
    parse_csv,
    passenger_lane_km,
)


STEP_LENGTH_SECONDS = 1.0
VALIDATION_MARKER = "AMBULANCE_VALIDATION_JSON="


# LIVE_VALIDATION_TEE_V1
class _ValidationTee:
    """Write evaluator output both to the parent pipe and a live file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    def fileno(self):
        return self.streams[0].fileno()

ABLATIONS = (
    {
        "name": "free_flow_route_base_signals",
        "routing_mode": "free_flow",
        "controller_mode": "base",
    },
    {
        "name": "traffic_aware_route_base_signals",
        "routing_mode": "traffic_aware",
        "controller_mode": "base",
    },
    {
        "name": "free_flow_route_learned_signals",
        "routing_mode": "free_flow",
        "controller_mode": "learned",
    },
    {
        "name": "traffic_aware_route_learned_signals",
        "routing_mode": "traffic_aware",
        "controller_mode": "learned",
    },
    {
        "name": "traffic_aware_route_deterministic_preemption",
        "routing_mode": "traffic_aware",
        "controller_mode": "deterministic_preemption",
    },
)
ABLATION_BY_NAME = {item["name"]: item for item in ABLATIONS}
PRIMARY_BASELINE = "traffic_aware_route_base_signals"
PRIMARY_LEARNED = "traffic_aware_route_learned_signals"
COMBINED_BASELINE = "free_flow_route_base_signals"

# Final three-way benchmark.  This is deliberately opt-in so the checkpoint
# selector keeps using the original five ablations above.  All three arms use
# the identical traffic-aware ambulance router, fixed background demand, and
# deterministic ambulance schedule; only signal control differs.
BASELINE_BENCHMARK_ABLATIONS = (
    {
        "name": "traffic_aware_route_native_sumo",
        "routing_mode": "traffic_aware",
        "controller_mode": "native_sumo",
    },
    {
        "name": "traffic_aware_route_max_pressure",
        "routing_mode": "traffic_aware",
        "controller_mode": "max_pressure",
    },
    ABLATION_BY_NAME[PRIMARY_LEARNED],
)
ALL_ABLATION_BY_NAME = {
    **ABLATION_BY_NAME,
    **{item["name"]: item for item in BASELINE_BENCHMARK_ABLATIONS},
}


@dataclass(frozen=True)
class DemandRecord:
    map_id: str
    net_file: Path
    route_file: Path
    seed: int
    scenario: str
    network_sha256: str
    route_sha256: str
    scheduled_vehicles: int

    @property
    def scenario_key(self) -> str:
        return (
            f"{self.net_file.resolve()}|{self.scenario}|{self.seed}|"
            f"{self.network_sha256}|{self.route_sha256}"
        )


@dataclass(frozen=True)
class EvaluationTask:
    index: int
    demand: DemandRecord
    ablation_name: str

    @property
    def ablation(self) -> Mapping[str, str]:
        return ALL_ABLATION_BY_NAME[self.ablation_name]


def _resolve_manifest_path(raw: str, manifest_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _validate_hash(
    path: Path, expected: str | None, label: str
) -> str:
    actual = sha256_file(path)
    if expected and str(expected) != actual:
        raise RuntimeError(
            f"{label} hash mismatch for {path}: "
            f"manifest={expected}, actual={actual}"
        )
    return actual


def load_paired_demand(
    path: str | Path,
    maps: Sequence[Path],
    seeds: set[int],
    episode_seconds: float,
) -> list[DemandRecord]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 2:
        raise RuntimeError(
            "Ambulance evaluation requires a schema-v2 checksummed demand "
            f"manifest: {manifest_path}"
        )
    duration = float(payload.get("episode_seconds", 0.0) or 0.0)
    if duration + 1e-9 < float(episode_seconds):
        raise RuntimeError(
            f"Demand bank lasts {duration:g}s but evaluation lasts "
            f"{episode_seconds:g}s"
        )
    manifest_dir = manifest_path.parent
    selected_maps = {map_path.resolve() for map_path in maps}
    records: list[DemandRecord] = []
    validated_networks: dict[Path, str] = {}

    rich_records = list(payload.get("records", ()))
    raw_records = rich_records or list(payload.get("routes", ()))
    if not raw_records:
        raise RuntimeError(f"Demand manifest has no route records: {path}")
    for index, raw in enumerate(raw_records):
        if "net_file" not in raw or "route_file" not in raw:
            raise RuntimeError(
                f"Demand record {index} lacks net_file/route_file"
            )
        net_file = _resolve_manifest_path(
            str(raw["net_file"]), manifest_dir
        )
        if net_file not in selected_maps:
            continue
        seed = int(raw["seed"])
        if seed not in seeds:
            continue
        route_file = _resolve_manifest_path(
            str(raw["route_file"]), manifest_dir
        )
        if not net_file.is_file():
            raise FileNotFoundError(net_file)
        if not route_file.is_file():
            raise FileNotFoundError(route_file)
        if not raw.get("network_sha256") or not raw.get("route_sha256"):
            raise RuntimeError(
                "Every ambulance-evaluation demand record must include "
                f"network_sha256 and route_sha256: record {index}"
            )
        expected_network_hash = str(raw["network_sha256"])
        if net_file in validated_networks:
            network_hash = validated_networks[net_file]
            if expected_network_hash != network_hash:
                raise RuntimeError(
                    "Demand records disagree about the network hash for "
                    f"{net_file}"
                )
        else:
            network_hash = _validate_hash(
                net_file,
                expected_network_hash,
                "Network",
            )
            validated_networks[net_file] = network_hash
        route_hash = _validate_hash(
            route_file,
            raw.get("route_sha256"),
            "Route",
        )
        if not fixed_demand_vehicle_type_is_safe(route_file):
            raise RuntimeError(
                "Ambulance evaluation demand must bind every vehicle to "
                "the audited passenger vType with "
                f"jmIgnoreKeepClearTime=-1: {route_file}"
            )
        scheduled = int(
            raw.get("scheduled_vehicles")
            or raw.get("scheduled_records")
            or count_scheduled_vehicles(route_file)
        )
        scenario = str(
            raw.get("scenario")
            or raw.get("trips_per_lane_km_hour")
            or raw.get("period_seconds")
            or f"record_{index:04d}"
        )
        map_id = str(
            raw.get("map_id")
            or net_file.stem.replace(".net", "")
        )
        records.append(
            DemandRecord(
                map_id=map_id,
                net_file=net_file,
                route_file=route_file,
                seed=seed,
                scenario=scenario,
                network_sha256=network_hash,
                route_sha256=route_hash,
                scheduled_vehicles=scheduled,
            )
        )

    missing = []
    for net_file in maps:
        for seed in sorted(seeds):
            if not any(
                record.net_file == net_file.resolve()
                and record.seed == seed
                for record in records
            ):
                missing.append(f"{net_file.name}:seed={seed}")
    if missing:
        raise RuntimeError(
            "The immutable demand bank lacks exact requested map/seed "
            "records: " + ", ".join(missing)
        )
    identities = [
        (
            record.net_file,
            record.seed,
            record.scenario,
            record.route_sha256,
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError(
            "Demand manifest contains duplicate map/seed/scenario records"
        )
    return sorted(
        records,
        key=lambda item: (
            item.map_id,
            item.scenario,
            item.seed,
            str(item.route_file),
        ),
    )


def _corridor_config(
    args: argparse.Namespace,
    contract: AmbulanceCheckpointContract,
) -> dict[str, float]:
    saved = dict(contract.corridor)
    return {
        "recovery_seconds": float(
            saved.get("recovery_seconds", args.recovery_seconds)
        ),
        "max_preemption_seconds": float(
            saved.get(
                "max_preemption_seconds",
                args.max_preemption_seconds,
            )
        ),
        "clearance_buffer_seconds": float(
            saved.get(
                "clearance_buffer_seconds",
                args.clearance_buffer_seconds,
            )
        ),
        "prepare_eta_seconds": float(
            saved.get(
                "prepare_eta_seconds",
                args.prepare_eta_seconds,
            )
        ),
        "serve_eta_seconds": float(
            saved.get(
                "serve_eta_seconds",
                args.serve_eta_seconds,
            )
        ),
    }


def worker_config(
    args: argparse.Namespace,
    task: EvaluationTask,
    base_embed_dim: int,
    base_graph_layers: int,
    contract: AmbulanceCheckpointContract,
) -> dict[str, Any]:
    demand = task.demand
    ablation = task.ablation
    saved = dict(contract.ambulance_system)
    ambulance_system = {
        "routing_mode": ablation["routing_mode"],
        "step_length_seconds": STEP_LENGTH_SECONDS,
        "first_spawn_seconds": args.ambulance_first_spawn,
        "spawn_interval_seconds": args.ambulance_interval_seconds,
        "spawn_jitter_seconds": args.ambulance_spawn_jitter,
        "max_ambulances": args.max_ambulances,
        "max_active_ambulances": args.max_active_ambulances,
        "planned_active_duration_factor": float(
            saved.get(
                "planned_active_duration_factor",
                args.planned_active_duration_factor,
            )
        ),
        "min_euclidean_distance": float(
            saved.get(
                "min_euclidean_distance",
                args.ambulance_min_euclidean_distance,
            )
        ),
        "min_route_distance": float(
            saved.get(
                "min_route_distance",
                args.ambulance_min_route_distance,
            )
        ),
        "min_route_edges": int(
            saved.get(
                "min_route_edges", args.ambulance_min_route_edges
            )
        ),
        "min_route_tls": int(
            saved.get("min_route_tls", args.ambulance_min_route_tls)
        ),
        "route_attempts_per_ambulance": int(
            saved.get(
                "route_attempts_per_ambulance",
                args.ambulance_route_attempts,
            )
        ),
        "reroute_interval_seconds": float(
            saved.get(
                "reroute_interval_seconds",
                args.reroute_interval,
            )
        ),
        "reroute_jitter_seconds": float(
            saved.get(
                "reroute_jitter_seconds", args.reroute_jitter
            )
        ),
        "reroute_min_savings_seconds": float(
            saved.get(
                "reroute_min_savings_seconds",
                args.reroute_min_savings_seconds,
            )
        ),
        "reroute_min_savings_fraction": float(
            saved.get(
                "reroute_min_savings_fraction",
                args.reroute_min_savings_fraction,
            )
        ),
        "no_reroute_within_tls_meters": float(
            saved.get(
                "no_reroute_within_tls_meters",
                args.no_reroute_within_tls,
            )
        ),
        "last_spawn_buffer_seconds": float(
            saved.get(
                "last_spawn_buffer_seconds",
                args.ambulance_last_spawn_buffer,
            )
        ),
    }
    return {
        "net_file": str(demand.net_file),
        "passenger_lane_km": passenger_lane_km(demand.net_file),
        # The same scenario seed is intentionally reused by all five
        # ablations. worker_rank only selects a distinct SUMO connection.
        "seed": int(demand.seed),
        "worker_rank": int(task.index),
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        "target_density_range": (1.0, 1.0),
        "max_vehicle_center": args.max_vehicle_center,
        "spawn_batch_center": 1,
        "observation_noise_std": 0.0,
        "sensor_scale_jitter": 0.0,
        "sensor_dropout_prob": 0.0,
        "base_embed_dim": base_embed_dim,
        "base_graph_layers": base_graph_layers,
        "emergency_embed_dim": contract.emergency_embed_dim,
        "emergency_graph_layers": contract.emergency_graph_layers,
        "residual_bound": contract.residual_bound,
        "demand_routes": [str(demand.route_file)],
        "scheduled_ordinary_vehicles": demand.scheduled_vehicles,
        "use_libsumo": args.use_libsumo,
        "time_to_teleport": -1,
        "strict_exit_space": True,
        "required_exit_gap_meters": (
            DEFAULT_REQUIRED_EXIT_GAP_METERS
        ),
        "allow_unsafe_hard_max_fallback": False,
        # Native SUMO must never pass through the custom TLS action executor.
        # The worker still builds read-only phase metadata for ambulance route
        # indexing, but SUMO advances the original <tlLogic> by itself.
        "native_sumo_signals": (
            ablation["controller_mode"] == "native_sumo"
        ),
        "sumo_error_log": str(
            args.sumo_log_dir
            / f"eval_task_{int(task.index):06d}.log"
        ),
        "ambulance_system": ambulance_system,
        "emergency_observation": dict(
            contract.emergency_observation
        ),
        "corridor": _corridor_config(args, contract),
        "traffic_excluded_vehicle_prefixes": [AMBULANCE_ID_PREFIX],
    }


def _close_workers(workers: Sequence[dict[str, Any]]) -> None:
    for worker in workers:
        try:
            worker["connection"].send({"cmd": "close"})
        except Exception:
            pass
    for worker in workers:
        worker["process"].join(timeout=20.0)
        if worker["process"].is_alive():
            worker["process"].terminate()
            worker["process"].join(timeout=10.0)
        worker["connection"].close()


def run_wave(
    ctx: Any,
    tasks: Sequence[EvaluationTask],
    args: argparse.Namespace,
    base_state: Mapping[str, np.ndarray],
    override_state: Mapping[str, np.ndarray],
    base_embed_dim: int,
    base_graph_layers: int,
    contract: AmbulanceCheckpointContract,
    authority: float,
) -> list[dict[str, Any]]:
    from ambulance_multiagent_worker import (
        emergency_rollout_worker_main,
    )

    workers: list[dict[str, Any]] = []
    try:
        for task in tasks:
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=emergency_rollout_worker_main,
                args=(
                    child,
                    worker_config(
                        args,
                        task,
                        base_embed_dim,
                        base_graph_layers,
                        contract,
                    ),
                ),
            )
            process.start()
            child.close()
            workers.append(
                {
                    "task": task,
                    "process": process,
                    "connection": parent,
                }
            )

        rollout_steps = int(
            math.ceil(args.episode_seconds / args.decision_seconds)
        )
        pending: dict[Any, dict[str, Any]] = {}
        for worker in workers:
            connection = worker["connection"]
            task = worker["task"]
            if not connection.poll(args.worker_start_timeout):
                raise TimeoutError(
                    "Ambulance evaluation worker did not start: "
                    f"{task.demand.map_id}/{task.ablation_name}"
                )
            ready = connection.recv()
            if ready.get("type") == "error":
                detail = ready["traceback"]
                if ready.get("sumo_error_log"):
                    detail += (
                        "\nSUMO error log: "
                        + str(ready["sumo_error_log"])
                    )
                raise RuntimeError(detail)
            if ready.get("type") != "ready":
                raise RuntimeError(
                    f"Unexpected worker message: {ready}"
                )
            worker["schedule_sha256"] = str(
                ready["schedule_sha256"]
            )
            print(
                f"[ready] {task.demand.map_id} "
                f"{task.demand.scenario} seed={task.demand.seed} "
                f"{task.ablation_name}: {ready['tls']} TLS, "
                f"schedule={ready['schedule_sha256'][:12]}",
                flush=True,
            )
            connection.send(
                {
                    "cmd": "rollout",
                    "base_state_dict": base_state,
                    "override_state_dict": override_state,
                    "rollout_steps": rollout_steps,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "authority": authority,
                    "controller_mode": task.ablation[
                        "controller_mode"
                    ],
                    "metrics_only": True,
                    "progress_interval": max(1, rollout_steps // 4),
                    "deterministic": True,
                }
            )
            pending[connection] = worker

        results: list[dict[str, Any]] = []
        while pending:
            ready_connections = wait(list(pending), timeout=60.0)
            if not ready_connections:
                dead = [
                    item["task"].ablation_name
                    for item in pending.values()
                    if not item["process"].is_alive()
                ]
                if dead:
                    raise RuntimeError(
                        f"Ambulance evaluation workers exited: {dead}"
                    )
                print("[evaluation] workers still active", flush=True)
                continue
            for connection in ready_connections:
                message = connection.recv()
                worker = pending[connection]
                task = worker["task"]
                if message.get("type") == "progress":
                    print(
                        f"[evaluation] {task.demand.map_id} "
                        f"{task.ablation_name}: "
                        f"{100.0 * message['step'] / message['total']:.1f}%",
                        flush=True,
                    )
                    continue
                if message.get("type") == "error":
                    detail = message["traceback"]
                    if message.get("sumo_error_log"):
                        detail += (
                            "\nSUMO error log: "
                            + str(message["sumo_error_log"])
                        )
                    raise RuntimeError(detail)
                if message.get("type") != "rollout":
                    raise RuntimeError(
                        f"Unexpected worker message: {message}"
                    )
                metrics = dict(message["metrics"])
                results.append(
                    {
                        "scenario_key": task.demand.scenario_key,
                        "map_id": task.demand.map_id,
                        "net_file": str(task.demand.net_file),
                        "scenario": task.demand.scenario,
                        "seed": task.demand.seed,
                        "route_file": str(task.demand.route_file),
                        "route_sha256": (
                            task.demand.route_sha256
                        ),
                        "network_sha256": (
                            task.demand.network_sha256
                        ),
                        "scheduled_ordinary_vehicles": (
                            task.demand.scheduled_vehicles
                        ),
                        "ablation": task.ablation_name,
                        "routing_mode": task.ablation[
                            "routing_mode"
                        ],
                        "controller_mode": task.ablation[
                            "controller_mode"
                        ],
                        "schedule_sha256": worker[
                            "schedule_sha256"
                        ],
                        "tls": metrics["tls"],
                        "ambulance": metrics["ambulance"],
                        "ordinary_traffic": metrics.get(
                            "ordinary_traffic", {}
                        ),
                        "recovery": metrics.get("recovery", {}),
                    }
                )
                del pending[connection]
        return results
    finally:
        _close_workers(workers)


def verify_pairing(
    records: Sequence[Mapping[str, Any]],
    expected_names: set[str] | None = None,
) -> None:
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(
            str(record["scenario_key"]), []
        ).append(record)
    expected = set(ABLATION_BY_NAME) if expected_names is None else set(expected_names)
    for scenario_key, group in by_scenario.items():
        names = {str(record["ablation"]) for record in group}
        if names != expected or len(group) != len(expected):
            raise RuntimeError(
                f"Incomplete ablation group for {scenario_key}: "
                f"{sorted(names)}"
            )
        schedules = {
            str(record["schedule_sha256"]) for record in group
        }
        if len(schedules) != 1:
            raise RuntimeError(
                "Ambulance O/D/spawn/free-flow schedules differ across "
                f"controllers for {scenario_key}: {sorted(schedules)}"
            )
        route_hashes = {
            str(record["route_sha256"]) for record in group
        }
        if len(route_hashes) != 1:
            raise RuntimeError(
                f"Background route hashes differ for {scenario_key}"
            )
        network_hashes = {
            str(record["network_sha256"]) for record in group
        }
        if len(network_hashes) != 1:
            raise RuntimeError(
                f"Network hashes differ for {scenario_key}"
            )


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0.0


def _ordinary_delay(summary: Mapping[str, Any]) -> float:
    """Use the censor-resistant all-departed metric when available."""

    return float(
        summary.get(
            "mean_time_loss_all_departed_s",
            summary.get("mean_time_loss_s", 0.0),
        )
    )


def _ordinary_delay_weight(summary: Mapping[str, Any]) -> int:
    if "mean_time_loss_all_departed_s" in summary:
        return int(summary.get("departed_total", 0))
    return int(summary.get("arrived_total", 0))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def aggregate_ablation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ambulance_records = [
        item
        for record in records
        for item in record["ambulance"].get("records", ())
    ]
    arrived = [
        item
        for item in ambulance_records
        if item.get("status") == "arrived"
        and item.get("actual_departure") is not None
        and item.get("end_time") is not None
    ]
    trip_times = [
        max(
            0.0,
            float(item["end_time"])
            - float(item["actual_departure"]),
        )
        for item in arrived
    ]
    response_times = [
        max(
            0.0,
            float(item["end_time"])
            - float(
                item.get(
                    "requested_departure",
                    item["actual_departure"],
                )
            ),
        )
        for item in arrived
    ]
    departure_delays = [
        max(
            0.0,
            float(item["actual_departure"])
            - float(
                item.get(
                    "requested_departure",
                    item["actual_departure"],
                )
            ),
        )
        for item in arrived
    ]
    ordinary_arrived = sum(
        int(record["ordinary_traffic"].get("arrived_total", 0))
        for record in records
    )
    ordinary_departed = sum(
        int(record["ordinary_traffic"].get("departed_total", 0))
        for record in records
    )
    scheduled_ordinary = sum(
        int(record["scheduled_ordinary_vehicles"])
        for record in records
    )
    ordinary_delay_numerator = sum(
        _ordinary_delay(record["ordinary_traffic"])
        * _ordinary_delay_weight(record["ordinary_traffic"])
        for record in records
    )
    ordinary_delay_denominator = sum(
        _ordinary_delay_weight(record["ordinary_traffic"])
        for record in records
    )
    collision_ids = [
        vehicle_id
        for record in records
        for vehicle_id in record["ambulance"].get(
            "collision_vehicle_ids", ()
        )
    ]
    teleport_ids = [
        vehicle_id
        for record in records
        for vehicle_id in record["ambulance"].get(
            "teleported_vehicle_ids", ()
        )
    ]
    invalid_policy_actions = sum(
        int(
            record["ambulance"].get("signal_safety", {}).get(
                "invalid_policy_actions", 0
            )
        )
        for record in records
    )
    invalid_signal_transitions = sum(
        int(
            record["ambulance"].get("signal_safety", {}).get(
                "invalid_signal_transitions", 0
            )
        )
        for record in records
    )
    scheduled_ambulances = len(ambulance_records)
    return {
        "runs": len(records),
        "ambulance_scheduled_total": scheduled_ambulances,
        "ambulance_arrived_total": len(arrived),
        "ambulance_completion_rate": (
            len(arrived) / max(1, scheduled_ambulances)
        ),
        "ambulance_failed_total": sum(
            str(item.get("status"))
            in {
                "teleported",
                "collided",
                "insertion_failed",
                "removed",
            }
            for item in ambulance_records
        ),
        "ambulance_censored_total": sum(
            item.get("status") == "censored"
            for item in ambulance_records
        ),
        "ambulance_mean_trip_time_s": _mean(trip_times),
        "ambulance_p95_trip_time_s": _percentile(
            trip_times, 95.0
        ),
        "ambulance_mean_response_time_s": _mean(response_times),
        "ambulance_p95_response_time_s": _percentile(
            response_times, 95.0
        ),
        "ambulance_mean_departure_delay_s": _mean(
            departure_delays
        ),
        "ambulance_mean_time_loss_s": _mean(
            float(item.get("time_loss", 0.0))
            for item in arrived
        ),
        "ambulance_mean_stopped_seconds": _mean(
            float(item.get("stopped_seconds", 0.0))
            for item in arrived
        ),
        "ordinary_scheduled_total": scheduled_ordinary,
        "ordinary_departed_total": ordinary_departed,
        "ordinary_arrived_total": ordinary_arrived,
        "ordinary_throughput_rate": (
            ordinary_arrived / max(1, scheduled_ordinary)
        ),
        "ordinary_mean_time_loss_s": (
            ordinary_delay_numerator
            / max(1, ordinary_delay_denominator)
        ),
        "ordinary_mean_time_loss_all_departed_s": (
            ordinary_delay_numerator
            / max(1, ordinary_delay_denominator)
        ),
        "ordinary_mean_queue_vehicles": _mean(
            float(
                record["ordinary_traffic"].get(
                    "mean_queue_vehicles", 0.0
                )
            )
            for record in records
        ),
        "ordinary_mean_speed_mps": _mean(
            float(
                record["ordinary_traffic"].get(
                    "mean_speed_mps", 0.0
                )
            )
            for record in records
        ),
        "mean_recovery_seconds": _mean(
            float(record["recovery"].get("mean_seconds", 0.0))
            for record in records
            if int(
                record["recovery"].get("completed_events", 0)
            )
            > 0
        ),
        "unrecovered_events": sum(
            int(record["recovery"].get("unrecovered_events", 0))
            for record in records
        ),
        # AmbulanceSystem records SUMO's global event lists, including
        # ordinary vehicles, so adding the ordinary monitor here would double
        # count the same collision/teleport.
        "collision_events": len(collision_ids),
        "teleport_events": len(teleport_ids),
        "invalid_policy_actions": invalid_policy_actions,
        "invalid_signal_transitions": invalid_signal_transitions,
        "guarded_fallback_green_activations": sum(
            int(
                record["ambulance"].get(
                    "signal_safety", {}
                ).get("guarded_fallback_green_activations", 0)
            )
            for record in records
        ),
        "blocked_green_activation_seconds": sum(
            float(
                record["ambulance"].get(
                    "signal_safety", {}
                ).get("blocked_green_activation_seconds", 0.0)
            )
            for record in records
        ),
    }


def _relative_change(
    candidate: float,
    baseline: float,
    lower_is_better: bool,
) -> float:
    if abs(float(baseline)) <= 1e-9:
        if abs(float(candidate)) <= 1e-9:
            return 0.0
        return (
            -1.0e9
            if lower_is_better and candidate > baseline
            else 1.0e9
        )
    raw = 100.0 * (float(candidate) - float(baseline)) / abs(
        float(baseline)
    )
    return -raw if lower_is_better else raw


def _run_response_time(summary: Mapping[str, Any]) -> float:
    return float(
        summary.get(
            "mean_response_time_s",
            summary.get("mean_trip_time_s", 0.0),
        )
    )


def pairwise_analysis(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["scenario_key"]), {})[
            str(record["ablation"])
        ] = record
    output = []
    for scenario_key, group in sorted(groups.items()):
        base = group[PRIMARY_BASELINE]
        learned = group[PRIMARY_LEARNED]
        combined_base = group[COMBINED_BASELINE]
        free_flow_learned = group[
            "free_flow_route_learned_signals"
        ]
        deterministic = group[
            "traffic_aware_route_deterministic_preemption"
        ]
        base_ambulance = base["ambulance"]
        learned_ambulance = learned["ambulance"]
        combined_ambulance = combined_base["ambulance"]
        output.append(
            {
                "scenario_key": scenario_key,
                "map_id": base["map_id"],
                "scenario": base["scenario"],
                "seed": base["seed"],
                "signal_ambulance_gain_percent": _relative_change(
                    _run_response_time(learned_ambulance),
                    _run_response_time(base_ambulance),
                    lower_is_better=True,
                ),
                "combined_ambulance_gain_percent": _relative_change(
                    _run_response_time(learned_ambulance),
                    _run_response_time(combined_ambulance),
                    lower_is_better=True,
                ),
                "routing_only_gain_percent": _relative_change(
                    _run_response_time(base_ambulance),
                    _run_response_time(combined_ambulance),
                    lower_is_better=True,
                ),
                "free_flow_signal_gain_percent": _relative_change(
                    _run_response_time(
                        free_flow_learned["ambulance"]
                    ),
                    _run_response_time(combined_ambulance),
                    lower_is_better=True,
                ),
                "learned_vs_deterministic_gain_percent": (
                    _relative_change(
                        _run_response_time(learned_ambulance),
                        _run_response_time(
                            deterministic["ambulance"]
                        ),
                        lower_is_better=True,
                    )
                ),
                "ordinary_delay_change_percent": _relative_change(
                    _ordinary_delay(learned["ordinary_traffic"]),
                    _ordinary_delay(base["ordinary_traffic"]),
                    lower_is_better=False,
                ),
                "throughput_change_percent": _relative_change(
                    float(
                        learned["ordinary_traffic"].get(
                            "arrived_total", 0
                        )
                    ),
                    float(
                        base["ordinary_traffic"].get(
                            "arrived_total", 0
                        )
                    ),
                    lower_is_better=False,
                ),
                "schedule_sha256": base["schedule_sha256"],
                "route_sha256": base["route_sha256"],
            }
        )
    return output


def build_summary(
    records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aggregates = {
        name: aggregate_ablation(
            [
                record
                for record in records
                if record["ablation"] == name
            ]
        )
        for name in ABLATION_BY_NAME
    }
    baseline = aggregates[PRIMARY_BASELINE]
    learned = aggregates[PRIMARY_LEARNED]
    combined_baseline = aggregates[COMBINED_BASELINE]
    free_flow_learned = aggregates[
        "free_flow_route_learned_signals"
    ]
    deterministic = aggregates[
        "traffic_aware_route_deterministic_preemption"
    ]
    ambulance_gain = _relative_change(
        learned["ambulance_mean_response_time_s"],
        baseline["ambulance_mean_response_time_s"],
        lower_is_better=True,
    )
    combined_gain = _relative_change(
        learned["ambulance_mean_response_time_s"],
        combined_baseline["ambulance_mean_response_time_s"],
        lower_is_better=True,
    )
    routing_gain = _relative_change(
        baseline["ambulance_mean_response_time_s"],
        combined_baseline["ambulance_mean_response_time_s"],
        lower_is_better=True,
    )
    free_flow_signal_gain = _relative_change(
        free_flow_learned["ambulance_mean_response_time_s"],
        combined_baseline["ambulance_mean_response_time_s"],
        lower_is_better=True,
    )
    deterministic_gain = _relative_change(
        learned["ambulance_mean_response_time_s"],
        deterministic["ambulance_mean_response_time_s"],
        lower_is_better=True,
    )
    delay_change = _relative_change(
        learned["ordinary_mean_time_loss_s"],
        baseline["ordinary_mean_time_loss_s"],
        lower_is_better=False,
    )
    throughput_change = _relative_change(
        learned["ordinary_arrived_total"],
        baseline["ordinary_arrived_total"],
        lower_is_better=False,
    )
    paired = pairwise_analysis(records)
    finite_gains = [
        float(item["signal_ambulance_gain_percent"])
        for item in paired
        if math.isfinite(
            float(item["signal_ambulance_gain_percent"])
        )
    ]
    worst_gain = min(finite_gains) if finite_gains else float("-inf")
    mean_gain = _mean(finite_gains)
    selection_score = 0.75 * worst_gain + 0.25 * mean_gain
    completion_not_worse = (
        learned["ambulance_completion_rate"] + 1e-12
        >= baseline["ambulance_completion_rate"]
    )
    safety_ok = (
        learned["collision_events"] == 0
        and learned["teleport_events"] == 0
        and learned["invalid_policy_actions"] == 0
        and learned["invalid_signal_transitions"] == 0
        and learned["ambulance_failed_total"] == 0
        and learned["ambulance_censored_total"] == 0
        and learned["unrecovered_events"] == 0
    )
    delay_ok = (
        math.isfinite(delay_change)
        and delay_change
        <= args.ordinary_delay_budget_percent + 1e-12
    )
    throughput_ok = (
        math.isfinite(throughput_change)
        and throughput_change
        >= -args.throughput_budget_percent - 1e-12
    )
    ambulance_not_worse = (
        math.isfinite(ambulance_gain)
        and ambulance_gain >= -1e-12
        and completion_not_worse
    )
    scenario_robustness = (
        len(finite_gains) == len(paired)
        and bool(finite_gains)
        and worst_gain >= -1e-12
    )
    deterministic_not_worse = (
        math.isfinite(deterministic_gain)
        and deterministic_gain >= -1e-12
    )
    return {
        "eligible": bool(
            safety_ok
            and delay_ok
            and throughput_ok
            and ambulance_not_worse
            and scenario_robustness
            and deterministic_not_worse
        ),
        "selection_score": selection_score,
        "ambulance_gain_percent": ambulance_gain,
        "combined_routing_and_signal_gain_percent": combined_gain,
        "routing_only_gain_percent": routing_gain,
        "free_flow_signal_gain_percent": free_flow_signal_gain,
        "learned_vs_deterministic_gain_percent": (
            deterministic_gain
        ),
        "ordinary_delay_change_percent": delay_change,
        "throughput_change_percent": throughput_change,
        "worst_scenario_ambulance_gain_percent": worst_gain,
        "mean_scenario_ambulance_gain_percent": mean_gain,
        "gates": {
            "safety_and_recovery": safety_ok,
            "ordinary_delay_budget": delay_ok,
            "throughput_budget": throughput_ok,
            "ambulance_not_worse": ambulance_not_worse,
            "completion_not_worse": completion_not_worse,
            "every_scenario_not_worse": scenario_robustness,
            "deterministic_preemption_not_better": (
                deterministic_not_worse
            ),
        },
        "budgets": {
            "ordinary_delay_percent": (
                args.ordinary_delay_budget_percent
            ),
            "throughput_reduction_percent": (
                args.throughput_budget_percent
            ),
        },
        "ablations": aggregates,
        "paired_scenarios": paired,
    }


def build_baseline_benchmark_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize learned-vs-Native and learned-vs-MaxPressure signal control."""

    names = [item["name"] for item in BASELINE_BENCHMARK_ABLATIONS]
    aggregates = {
        name: aggregate_ablation(
            [record for record in records if record["ablation"] == name]
        )
        for name in names
    }
    learned = aggregates[PRIMARY_LEARNED]

    def comparison(baseline_name: str) -> dict[str, float]:
        baseline = aggregates[baseline_name]
        return {
            "ambulance_response_time_improvement_percent": _relative_change(
                learned["ambulance_mean_response_time_s"],
                baseline["ambulance_mean_response_time_s"],
                lower_is_better=True,
            ),
            "ambulance_p95_response_time_improvement_percent": _relative_change(
                learned["ambulance_p95_response_time_s"],
                baseline["ambulance_p95_response_time_s"],
                lower_is_better=True,
            ),
            "ordinary_delay_improvement_percent": _relative_change(
                learned["ordinary_mean_time_loss_all_departed_s"],
                baseline["ordinary_mean_time_loss_all_departed_s"],
                lower_is_better=True,
            ),
            "ordinary_throughput_improvement_percent": _relative_change(
                learned["ordinary_throughput_rate"],
                baseline["ordinary_throughput_rate"],
                lower_is_better=False,
            ),
            "ordinary_queue_improvement_percent": _relative_change(
                learned["ordinary_mean_queue_vehicles"],
                baseline["ordinary_mean_queue_vehicles"],
                lower_is_better=True,
            ),
            "ordinary_speed_improvement_percent": _relative_change(
                learned["ordinary_mean_speed_mps"],
                baseline["ordinary_mean_speed_mps"],
                lower_is_better=False,
            ),
        }

    by_scenario: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(str(record["scenario_key"]), {})[
            str(record["ablation"])
        ] = record
    paired = []
    for scenario_key, group in sorted(by_scenario.items()):
        learned_record = group[PRIMARY_LEARNED]
        item: dict[str, Any] = {
            "scenario_key": scenario_key,
            "map_id": learned_record["map_id"],
            "scenario": learned_record["scenario"],
            "seed": learned_record["seed"],
            "schedule_sha256": learned_record["schedule_sha256"],
            "route_sha256": learned_record["route_sha256"],
        }
        for label, baseline_name in (
            ("native_sumo", "traffic_aware_route_native_sumo"),
            ("max_pressure", "traffic_aware_route_max_pressure"),
        ):
            baseline_record = group[baseline_name]
            item[f"learned_vs_{label}_ambulance_gain_percent"] = _relative_change(
                _run_response_time(learned_record["ambulance"]),
                _run_response_time(baseline_record["ambulance"]),
                lower_is_better=True,
            )
            item[f"learned_vs_{label}_ordinary_delay_gain_percent"] = _relative_change(
                _ordinary_delay(learned_record["ordinary_traffic"]),
                _ordinary_delay(baseline_record["ordinary_traffic"]),
                lower_is_better=True,
            )
        paired.append(item)

    return {
        "controllers": names,
        "aggregates": aggregates,
        "learned_vs_native_sumo": comparison(
            "traffic_aware_route_native_sumo"
        ),
        "learned_vs_max_pressure": comparison(
            "traffic_aware_route_max_pressure"
        ),
        "paired_scenarios": paired,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="validation")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--emergency-model-path", required=True)
    parser.add_argument("--demand-bank-manifest", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seeds", default="9001,9002")
    parser.add_argument("--episode-seconds", type=int, default=1200)
    parser.add_argument("--decision-seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--worker-start-timeout", type=float, default=900.0
    )
    parser.add_argument(
        "--sumo-log-dir",
        type=Path,
        default=Path("runs/ambulance_v5_eval_sumo_logs"),
    )
    parser.add_argument(
        "--use-libsumo",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-vehicle-center", type=int, default=20000)
    parser.add_argument(
        "--ordinary-delay-budget-percent",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--throughput-budget-percent", type=float, default=2.0
    )
    parser.add_argument(
        "--ambulance-first-spawn", type=float, default=30.0
    )
    parser.add_argument(
        "--ambulance-interval-seconds", type=float, default=180.0
    )
    parser.add_argument(
        "--ambulance-spawn-jitter", type=float, default=20.0
    )
    parser.add_argument("--max-ambulances", type=int, default=16)
    parser.add_argument(
        "--max-active-ambulances", type=int, default=2
    )
    parser.add_argument(
        "--planned-active-duration-factor",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--ambulance-last-spawn-buffer",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--ambulance-min-euclidean-distance",
        type=float,
        default=1200.0,
    )
    parser.add_argument(
        "--ambulance-min-route-distance",
        type=float,
        default=1500.0,
    )
    parser.add_argument(
        "--ambulance-min-route-edges", type=int, default=12
    )
    parser.add_argument(
        "--ambulance-min-route-tls", type=int, default=2
    )
    parser.add_argument(
        "--ambulance-route-attempts", type=int, default=120
    )
    parser.add_argument("--reroute-interval", type=float, default=12.0)
    parser.add_argument("--reroute-jitter", type=float, default=2.0)
    parser.add_argument(
        "--reroute-min-savings-seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--reroute-min-savings-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--no-reroute-within-tls", type=float, default=100.0
    )
    parser.add_argument("--recovery-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-preemption-seconds", type=float, default=45.0
    )
    parser.add_argument(
        "--clearance-buffer-seconds", type=float, default=3.0
    )
    parser.add_argument(
        "--prepare-eta-seconds", type=float, default=25.0
    )
    parser.add_argument(
        "--serve-eta-seconds", type=float, default=12.0
    )
    parser.add_argument(
        "--native-maxpressure-benchmark-only",
        action="store_true",
        help=(
            "Run exactly three paired traffic-aware arms: native SUMO, "
            "normalized MaxPressure, and the learned ambulance controller."
        ),
    )
    args = parser.parse_args()

    # The trainer captures evaluator stdout until validation ends. Mirror it
    # to this sidecar so monitoring can read each progress update immediately.
    _validation_live_path = args.output_json.with_suffix(".live.log")
    _validation_live_path.parent.mkdir(parents=True, exist_ok=True)
    _validation_live_stream = _validation_live_path.open(
        "w", encoding="utf-8", buffering=1
    )
    sys.stdout = _ValidationTee(sys.stdout, _validation_live_stream)
    print(f"VALIDATION_LIVE_LOG={_validation_live_path}", flush=True)

    args.splits = set(parse_csv(args.splits))
    if not args.manifest and not args.maps:
        parser.error("Pass --manifest or --maps")
    if args.episode_seconds <= 0 or args.decision_seconds <= 0.0:
        parser.error("Episode and decision durations must be positive")
    decision_steps = args.decision_seconds / STEP_LENGTH_SECONDS
    if abs(decision_steps - round(decision_steps)) > 1e-9:
        parser.error(
            "--decision-seconds must be an exact multiple of the "
            f"{STEP_LENGTH_SECONDS:g}s SUMO step"
        )
    episode_decisions = (
        float(args.episode_seconds) / args.decision_seconds
    )
    if abs(episode_decisions - round(episode_decisions)) > 1e-9:
        parser.error(
            "--episode-seconds must be an exact multiple of "
            "--decision-seconds"
        )
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.ambulance_interval_seconds <= 0.0:
        parser.error("--ambulance-interval-seconds must be positive")
    if args.ambulance_first_spawn + 1e-9 < STEP_LENGTH_SECONDS:
        parser.error(
            "--ambulance-first-spawn must leave at least one SUMO "
            "step for boundary-safe prequeuing"
        )
    if args.max_ambulances <= 0 or args.max_active_ambulances <= 0:
        parser.error("Ambulance limits must be positive")
    if args.planned_active_duration_factor < 1.0:
        parser.error(
            "--planned-active-duration-factor must be at least 1"
        )
    if (
        args.serve_eta_seconds < 0.0
        or args.prepare_eta_seconds <= 0.0
        or args.serve_eta_seconds > args.prepare_eta_seconds
    ):
        parser.error(
            "Require 0 <= --serve-eta-seconds <= "
            "--prepare-eta-seconds"
        )
    if args.max_preemption_seconds < args.decision_seconds:
        parser.error(
            "--max-preemption-seconds must cover at least one complete "
            "decision interval"
        )
    return args


def main() -> None:
    args = parse_args()

    from ambulance_checkpoint import (
        AMBULANCE_SCHEMA_VERSION,
        emergency_model_path,
        load_emergency_checkpoint,
    )
    from ambulance_emergency import EmergencyOverrideNetwork
    from train_ambulance_override import load_base_policy

    maps = load_maps(args)
    seeds = {int(value) for value in parse_csv(args.seeds)}
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    demand_records = load_paired_demand(
        args.demand_bank_manifest,
        maps,
        seeds,
        args.episode_seconds,
    )
    (
        base_checkpoint,
        base_state,
        base_embed_dim,
        base_graph_layers,
    ) = load_base_policy(args.base_model_path)
    checkpoint, contract = load_emergency_checkpoint(
        args.emergency_model_path,
        base_checkpoint=base_checkpoint,
        decision_seconds=args.decision_seconds,
        step_length_seconds=STEP_LENGTH_SECONDS,
        minimum_green_seconds=DEFAULT_MIN_GREEN,
        maximum_green_seconds=DEFAULT_MAX_GREEN,
        device="cpu",
    )
    override = EmergencyOverrideNetwork(
        embed_dim=contract.emergency_embed_dim,
        graph_layers=contract.emergency_graph_layers,
        residual_bound=contract.residual_bound,
    )
    override.load_state_dict(checkpoint["state_dict"])
    override.eval()
    override_state = {
        key: value.detach().cpu().numpy().copy()
        for key, value in override.state_dict().items()
    }
    authority = float(
        checkpoint.get("authority", contract.authority)
    )
    selected_ablations = (
        BASELINE_BENCHMARK_ABLATIONS
        if args.native_maxpressure_benchmark_only
        else ABLATIONS
    )
    tasks = [
        EvaluationTask(
            index=index,
            demand=demand,
            ablation_name=str(ablation["name"]),
        )
        for index, (demand, ablation) in enumerate(
            (demand, ablation)
            for demand in demand_records
            for ablation in selected_ablations
        )
    ]
    print("\nAmbulance evaluation plan")
    print(f"  maps:              {len(maps)}")
    print(f"  demand scenarios:  {len(demand_records)}")
    print(f"  paired runs:       {len(tasks)}")
    print(f"  ablations:         {len(selected_ablations)}")
    print(f"  workers:           {args.workers}")
    print(f"  authority:         {authority:g}")

    records: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    for start in range(0, len(tasks), args.workers):
        records.extend(
            run_wave(
                ctx,
                tasks[start : start + args.workers],
                args,
                base_state,
                override_state,
                base_embed_dim,
                base_graph_layers,
                contract,
                authority,
            )
        )
    # Detect any demand or network mutation that occurred while the long
    # evaluation was running, without re-hashing the same network per task.
    verified_networks: set[Path] = set()
    for demand in demand_records:
        if demand.net_file not in verified_networks:
            _validate_hash(
                demand.net_file,
                demand.network_sha256,
                "Network",
            )
            verified_networks.add(demand.net_file)
        _validate_hash(
            demand.route_file,
            demand.route_sha256,
            "Route",
        )
    selected_names = {str(item["name"]) for item in selected_ablations}
    verify_pairing(records, expected_names=selected_names)
    if args.native_maxpressure_benchmark_only:
        summary = {
            "baseline_benchmark": build_baseline_benchmark_summary(records)
        }
        evaluation_mode = (
            "paired_immutable_demand_native_maxpressure_learned_exact_sumo"
        )
    else:
        summary = build_summary(records, args)
        evaluation_mode = "paired_immutable_demand_five_way_exact_sumo"
    payload = {
        "schema_version": AMBULANCE_SCHEMA_VERSION,
        "evaluation_mode": evaluation_mode,
        "base_model": str(base_checkpoint.resolve()),
        "base_model_sha256": contract.base_checkpoint_sha256,
        "emergency_model": str(
            emergency_model_path(
                args.emergency_model_path
            ).resolve()
        ),
        "demand_bank_manifest": str(
            Path(args.demand_bank_manifest).resolve()
        ),
        "splits": sorted(args.splits),
        "seeds": sorted(seeds),
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        **summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(
        args.output_json.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_json)
    if args.native_maxpressure_benchmark_only:
        compact = payload["baseline_benchmark"]
    else:
        compact = {
            key: payload[key]
            for key in (
                "eligible",
                "selection_score",
                "ambulance_gain_percent",
                "combined_routing_and_signal_gain_percent",
                "routing_only_gain_percent",
                "learned_vs_deterministic_gain_percent",
                "ordinary_delay_change_percent",
                "throughput_change_percent",
                "gates",
            )
        }
    print(
        VALIDATION_MARKER
        + json.dumps(compact, separators=(",", ":"), allow_nan=False)
    )
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
