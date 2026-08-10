#!/usr/bin/env python3
"""Train a schema-v5 ambulance override around a frozen schema-v3 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import shutil
import subprocess
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch.distributions import Categorical

from ambulance_checkpoint import (
    AMBULANCE_SCHEMA_VERSION,
    AmbulanceCheckpointContract,
    emergency_model_path,
    save_emergency_checkpoint,
    sha256_file,
)
from ambulance_curriculum import curriculum_demand_routes
from ambulance_emergency import (
    CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS,
    MIN_EMERGENCY_DOWNSTREAM_SPACE,
    EmergencyOverrideNetwork,
)
from ambulance_multiagent_worker import emergency_rollout_worker_main
from ambulance_system import AMBULANCE_ID_PREFIX
from fixed_demand import fixed_demand_vehicle_type_is_safe
from map_agnostic_tls import (
    DEFAULT_MAX_GREEN,
    DEFAULT_MIN_GREEN,
    DEFAULT_REQUIRED_EXIT_GAP_METERS,
    MAX_PHASES,
)
from train_map_agnostic_multiagent import (
    audit_manifest_splits,
    load_demand_bank,
    model_zip,
)
from train_map_agnostic_multimap import (
    load_maps,
    parse_csv,
    passenger_lane_km,
)


ROOT = Path(__file__).resolve().parent
STEP_LENGTH_SECONDS = 1.0
EMERGENCY_OBSERVATION_CONFIG = {
    "relevance_distance_meters": 650.0,
    "relevance_eta_seconds": 60.0,
    "eta_floor_speed_mps": 5.0,
    "max_active_reference": 3.0,
    "route_horizon": 3,
}


def load_ambulance_demand_bank(
    path: str,
    expected_episode_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    """Preserve fixed-route intensity metadata for staged curriculum."""

    routes_by_map = load_demand_bank(
        path,
        expected_episode_seconds=expected_episode_seconds,
    )
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if int(payload.get("schema_version", 0)) < 2:
        raise RuntimeError(
            "Ambulance training requires a schema-v2 checksummed demand "
            f"manifest: {manifest_path}"
        )
    manifest_dir = manifest_path.parent
    metadata_by_route: dict[str, float | None] = {}
    validated_networks: dict[str, str] = {}
    for record in list(payload.get("records", ())) + list(
        payload.get("routes", ())
    ):
        raw_route = record.get("route_file")
        if not raw_route:
            continue
        route_path = Path(str(raw_route)).expanduser()
        if not route_path.is_absolute():
            route_path = manifest_dir / route_path
        net_path = Path(str(record.get("net_file", ""))).expanduser()
        if not net_path.is_absolute():
            net_path = manifest_dir / net_path
        if not record.get("network_sha256") or not record.get(
            "route_sha256"
        ):
            raise RuntimeError(
                "Every ambulance-training demand record must include "
                "network_sha256 and route_sha256"
            )
        net_key = str(net_path.resolve())
        expected_network_hash = str(record["network_sha256"])
        if net_key in validated_networks:
            if validated_networks[net_key] != expected_network_hash:
                raise RuntimeError(
                    "Demand records disagree about the network hash for "
                    f"{net_path}"
                )
        else:
            actual_network_hash = sha256_file(net_path)
            if actual_network_hash != expected_network_hash:
                raise RuntimeError(
                    f"Network hash mismatch for {net_path}"
                )
            validated_networks[net_key] = actual_network_hash
        if sha256_file(route_path) != str(record["route_sha256"]):
            raise RuntimeError(
                f"Route hash mismatch for {route_path}"
            )
        if not fixed_demand_vehicle_type_is_safe(route_path):
            raise RuntimeError(
                "Ambulance training demand must bind every vehicle to "
                "the audited passenger vType with "
                f"jmIgnoreKeepClearTime=-1: {route_path}"
            )
        intensity: float | None = None
        raw_rate = record.get("trips_per_lane_km_hour")
        if raw_rate is not None:
            intensity = float(raw_rate)
        else:
            period = float(
                record.get("period_seconds", 0.0) or 0.0
            )
            if period > 0.0:
                intensity = 1.0 / period
        key = str(route_path.resolve())
        if intensity is not None or key not in metadata_by_route:
            metadata_by_route[key] = intensity

    return {
        net_file: [
            {
                "route_file": route_file,
                "intensity": metadata_by_route.get(route_file),
            }
            for route_file in route_files
        ]
        for net_file, route_files in routes_by_map.items()
    }


def trainer_path(path: str | Path) -> Path:
    base = emergency_model_path(path).with_suffix("")
    return base.parent / f"{base.name}_trainer.pt"


def load_base_policy(path: str | Path):
    checkpoint = model_zip(path)
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    network = model.policy.map_network
    state = {
        key: value.detach().cpu().numpy().copy()
        for key, value in network.state_dict().items()
    }
    embed_dim = int(network.movement_encoder[0].out_features)
    graph_layers = len(network.graph_blocks)
    del model
    return checkpoint, state, embed_dim, graph_layers


def flatten_rollouts(
    rollouts: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    dynamic_keys = (
        "movements",
        "phase_features",
        "global_features",
    )
    emergency_keys = (
        "emergency_movements",
        "emergency_phase_features",
        "emergency_global_features",
    )
    vector_keys = (
        "actions",
        "base_actions",
        "old_log_probs",
        "old_values",
        "advantages",
        "returns",
        "teacher_actions",
        "active",
    )
    output: dict[str, list[np.ndarray]] = {
        key: [] for key in dynamic_keys + emergency_keys + vector_keys
    }
    output["action_masks"] = []
    output["base_logits"] = []
    output["sample_weights"] = []
    output["static_indices"] = []
    static_banks = {
        "movement_mask": [],
        "movement_adjacency": [],
        "phase_membership": [],
    }
    static_offset = 0
    for rollout in rollouts:
        actions = np.asarray(rollout["actions"])
        steps, agents = actions.shape
        for key in dynamic_keys:
            value = np.asarray(rollout["dynamic"][key])
            output[key].append(
                value.reshape((steps * agents, *value.shape[2:]))
            )
        for key in emergency_keys:
            value = np.asarray(rollout["emergency_dynamic"][key])
            output[key].append(
                value.reshape((steps * agents, *value.shape[2:]))
            )
        for key in static_banks:
            static_banks[key].append(
                np.asarray(rollout["static"][key])
            )
        output["static_indices"].append(
            np.tile(
                np.arange(agents, dtype=np.int32) + static_offset,
                steps,
            )
        )
        static_offset += agents
        output["action_masks"].append(
            np.asarray(rollout["action_masks"]).reshape(
                steps * agents, MAX_PHASES + 1
            )
        )
        output["base_logits"].append(
            np.asarray(rollout["base_logits"]).reshape(
                steps * agents, MAX_PHASES + 1
            )
        )
        for key in vector_keys:
            output[key].append(
                np.asarray(rollout[key]).reshape(-1)
            )
        agent_weights = np.asarray(
            rollout["agent_weights"], dtype=np.float32
        )
        active = np.asarray(
            rollout["active"], dtype=np.float32
        )
        output["sample_weights"].append(
            (
                active
                * agent_weights[None, :]
            ).reshape(-1)
        )
    flattened = {
        key: np.concatenate(values, axis=0)
        for key, values in output.items()
    }
    flattened.update(
        {
            f"static_{key}": np.concatenate(values, axis=0)
            for key, values in static_banks.items()
        }
    )
    return flattened


def weighted_mean(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class EmergencyPPOTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        contract: AmbulanceCheckpointContract,
    ):
        self.args = args
        self.device = device
        self.contract = contract
        self.network = EmergencyOverrideNetwork(
            embed_dim=args.emergency_embed_dim,
            graph_layers=args.emergency_graph_layers,
            residual_bound=args.residual_bound,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )
        self.completed_updates = 0
        self.active_transitions = 0
        self.best_validation_score = float("-inf")
        self.last_validated_round = 0

    def authority(self, total_updates: int) -> float:
        progress = self.completed_updates / max(1, total_updates)
        return (
            self.args.authority_start
            + (self.args.authority_end - self.args.authority_start)
            * min(1.0, max(0.0, progress))
        )

    def state_dict_for_workers(self) -> dict[str, np.ndarray]:
        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in self.network.state_dict().items()
        }

    def update(
        self,
        rollouts: list[dict[str, Any]],
        total_updates: int,
        rollout_authority: float,
    ) -> dict[str, float]:
        batch = flatten_rollouts(rollouts)
        weights_np = batch["sample_weights"].astype(np.float32)
        chosen_samples = np.flatnonzero(weights_np > 0.0)
        if len(chosen_samples) == 0:
            raise RuntimeError(
                "The rollout contained no ambulance-corridor or recovery "
                "transitions. Shorten --ambulance-first-spawn or increase "
                "--rollout-steps."
            )
        self.active_transitions += len(chosen_samples)
        advantages_np = batch["advantages"].astype(np.float32)
        active_weights = weights_np[chosen_samples]
        active_advantages = advantages_np[chosen_samples]
        advantage_mean = float(
            np.sum(active_weights * active_advantages)
            / max(1e-9, active_weights.sum())
        )
        advantage_variance = float(
            np.sum(
                active_weights
                * (active_advantages - advantage_mean) ** 2
            )
            / max(1e-9, active_weights.sum())
        )
        advantages_np = (
            advantages_np - advantage_mean
        ) / math.sqrt(advantage_variance + 1e-8)

        progress = self.completed_updates / max(1, total_updates)
        learning_rate = (
            self.args.learning_rate
            + (
                self.args.final_learning_rate
                - self.args.learning_rate
            )
            * progress
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        teacher_fraction = max(
            0.0,
            1.0
            - progress
            / max(1e-9, self.args.teacher_decay_fraction),
        )
        teacher_coefficient = (
            self.args.teacher_coef * teacher_fraction
        )

        indices = chosen_samples.copy()
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "teacher_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "batches": 0.0,
        }
        stop_early = False
        self.network.train()
        for _epoch in range(self.args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(
                0, len(indices), self.args.minibatch_size
            ):
                selected = indices[
                    start : start + self.args.minibatch_size
                ]
                if len(selected) == 0:
                    continue
                base_observation = {
                    key: torch.as_tensor(
                        batch[key][selected],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for key in (
                        "movements",
                        "phase_features",
                        "global_features",
                    )
                }
                static_indices = batch["static_indices"][selected]
                for key in (
                    "movement_mask",
                    "movement_adjacency",
                    "phase_membership",
                ):
                    base_observation[key] = torch.as_tensor(
                        batch[f"static_{key}"][static_indices],
                        dtype=torch.float32,
                        device=self.device,
                    )
                emergency_observation = {
                    key: torch.as_tensor(
                        batch[key][selected],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for key in (
                        "emergency_movements",
                        "emergency_phase_features",
                        "emergency_global_features",
                    )
                }
                base_logits = torch.as_tensor(
                    batch["base_logits"][selected],
                    dtype=torch.float32,
                    device=self.device,
                )
                action_masks = torch.as_tensor(
                    batch["action_masks"][selected],
                    dtype=torch.bool,
                    device=self.device,
                )
                combined, values, _residual = self.network(
                    base_observation,
                    emergency_observation,
                    base_logits,
                    authority=rollout_authority,
                )
                distribution = Categorical(
                    logits=combined.masked_fill(
                        ~action_masks, -1e8
                    )
                )
                actions = torch.as_tensor(
                    batch["actions"][selected],
                    dtype=torch.long,
                    device=self.device,
                )
                old_log_probs = torch.as_tensor(
                    batch["old_log_probs"][selected],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_values = torch.as_tensor(
                    batch["old_values"][selected],
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.as_tensor(
                    batch["returns"][selected],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    advantages_np[selected],
                    dtype=torch.float32,
                    device=self.device,
                )
                weights = torch.as_tensor(
                    weights_np[selected],
                    dtype=torch.float32,
                    device=self.device,
                )
                teacher_actions = torch.as_tensor(
                    batch["teacher_actions"][selected],
                    dtype=torch.long,
                    device=self.device,
                )
                log_probs = distribution.log_prob(actions)
                entropy = distribution.entropy()
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                policy_loss = -weighted_mean(
                    torch.minimum(
                        ratio * advantages,
                        torch.clamp(
                            ratio,
                            1.0 - self.args.clip_range,
                            1.0 + self.args.clip_range,
                        )
                        * advantages,
                    ),
                    weights,
                )
                values = values.squeeze(-1)
                clipped_values = old_values + torch.clamp(
                    values - old_values,
                    -self.args.value_clip_range,
                    self.args.value_clip_range,
                )
                value_loss = 0.5 * weighted_mean(
                    torch.maximum(
                        (values - returns) ** 2,
                        (clipped_values - returns) ** 2,
                    ),
                    weights,
                )
                entropy_mean = weighted_mean(entropy, weights)
                teacher_loss = weighted_mean(
                    -distribution.log_prob(teacher_actions),
                    weights,
                )
                loss = (
                    policy_loss
                    + self.args.value_coef * value_loss
                    - self.args.entropy_coef * entropy_mean
                    + teacher_coefficient * teacher_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.args.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = weighted_mean(
                        (torch.exp(log_ratio) - 1.0) - log_ratio,
                        weights,
                    )
                    clip_fraction = weighted_mean(
                        (
                            torch.abs(ratio - 1.0)
                            > self.args.clip_range
                        ).float(),
                        weights,
                    )
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy_mean.item())
                totals["teacher_loss"] += float(
                    teacher_loss.item()
                )
                totals["approx_kl"] += float(approx_kl.item())
                totals["clip_fraction"] += float(
                    clip_fraction.item()
                )
                totals["batches"] += 1.0
                if (
                    self.args.target_kl > 0.0
                    and float(approx_kl.item())
                    > 1.5 * self.args.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break
        self.completed_updates += 1
        denominator = max(1.0, totals.pop("batches"))
        result = {
            key: value / denominator
            for key, value in totals.items()
        }
        result.update(
            learning_rate=learning_rate,
            active_samples=float(len(chosen_samples)),
            advantage_mean=advantage_mean,
            advantage_std=math.sqrt(
                advantage_variance + 1e-8
            ),
            teacher_coefficient=teacher_coefficient,
            authority=float(rollout_authority),
            early_stop=float(stop_early),
        )
        return result

    def save(self, total_updates: int) -> None:
        path = trainer_path(self.args.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": AMBULANCE_SCHEMA_VERSION,
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "completed_updates": self.completed_updates,
                "active_transitions": self.active_transitions,
                "best_validation_score": self.best_validation_score,
                "last_validated_round": self.last_validated_round,
                "plan_signature": self.args.plan_signature,
            },
            temporary,
        )
        temporary.replace(path)
        save_emergency_checkpoint(
            self.args.model_path,
            state_dict=self.network.state_dict(),
            contract=self.contract,
            training_state={
                "completed_updates": self.completed_updates,
                "active_transitions": self.active_transitions,
                "planned_updates": total_updates,
                "authority": self.authority(total_updates),
            },
        )

    def load(self) -> None:
        path = trainer_path(self.args.model_path)
        payload = torch.load(path, map_location=self.device)
        if int(payload.get("schema_version", -1)) != (
            AMBULANCE_SCHEMA_VERSION
        ):
            raise RuntimeError(f"Incompatible trainer checkpoint: {path}")
        if payload.get("plan_signature") != self.args.plan_signature:
            raise RuntimeError(
                "The resume checkpoint belongs to a different training "
                "schedule. Repeat the original command, choose a new model "
                "path, or pass --restart."
            )
        self.network.load_state_dict(payload["network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.completed_updates = int(
            payload.get("completed_updates", 0)
        )
        self.active_transitions = int(
            payload.get("active_transitions", 0)
        )
        self.best_validation_score = float(
            payload.get("best_validation_score", float("-inf"))
        )
        self.last_validated_round = int(
            payload.get("last_validated_round", 0)
        )


def curriculum(
    args: argparse.Namespace, progress: float
) -> dict[str, Any]:
    low, high = args.target_density_min, args.target_density_max
    if progress < 0.25:
        density = (low, low + 0.50 * (high - low))
        active_choices = [1]
        interval = (
            max(args.ambulance_interval_min, 180.0),
            max(args.ambulance_interval_max, 240.0),
        )
    elif progress < 0.65:
        density = (low, high)
        active_choices = [1]
        interval = (
            args.ambulance_interval_min,
            args.ambulance_interval_max,
        )
    else:
        density = (
            low + 0.25 * (high - low),
            high,
        )
        active_choices = [1, 2, 2]
        interval = (
            args.ambulance_interval_min,
            args.ambulance_interval_max,
        )
    return {
        "target_density_range": density,
        "ambulance_max_active_choices": active_choices,
        "ambulance_interval_range": interval,
    }


def worker_config(
    args: argparse.Namespace,
    net_file: Path,
    seed: int,
    rank: int,
    base_embed_dim: int,
    base_graph_layers: int,
    demand_routes: list[str],
    progress: float,
) -> dict[str, Any]:
    stage = curriculum(args, progress)
    ambulance_system = {
        "routing_mode": "traffic_aware",
        "step_length_seconds": STEP_LENGTH_SECONDS,
        "first_spawn_seconds": args.ambulance_first_spawn,
        "spawn_interval_seconds": args.ambulance_interval_max,
        "spawn_jitter_seconds": args.ambulance_spawn_jitter,
        "max_ambulances": args.max_ambulances_per_episode,
        "max_active_ambulances": 2,
        "planned_active_duration_factor": (
            args.planned_active_duration_factor
        ),
        "min_euclidean_distance": args.ambulance_min_euclidean_distance,
        "min_route_distance": args.ambulance_min_route_distance,
        "min_route_edges": args.ambulance_min_route_edges,
        "min_route_tls": args.ambulance_min_route_tls,
        "route_attempts_per_ambulance": args.ambulance_route_attempts,
        "reroute_interval_seconds": args.reroute_interval,
        "reroute_jitter_seconds": args.reroute_jitter,
        "reroute_min_savings_seconds": args.reroute_min_savings_seconds,
        "reroute_min_savings_fraction": args.reroute_min_savings_fraction,
        "no_reroute_within_tls_meters": args.no_reroute_within_tls,
        "last_spawn_buffer_seconds": (
            args.ambulance_last_spawn_buffer
        ),
    }
    return {
        "net_file": str(net_file),
        "passenger_lane_km": passenger_lane_km(net_file),
        "seed": int(seed),
        "worker_rank": int(rank),
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        "target_density_range": stage["target_density_range"],
        "max_vehicle_center": args.max_vehicle_center,
        "spawn_batch_center": args.spawn_batch_center,
        "observation_noise_std": args.observation_noise_std,
        "sensor_scale_jitter": args.sensor_scale_jitter,
        "sensor_dropout_prob": args.sensor_dropout_prob,
        "base_embed_dim": base_embed_dim,
        "base_graph_layers": base_graph_layers,
        "emergency_embed_dim": args.emergency_embed_dim,
        "emergency_graph_layers": args.emergency_graph_layers,
        "residual_bound": args.residual_bound,
        "demand_routes": list(demand_routes),
        "use_libsumo": args.use_libsumo,
        "time_to_teleport": -1,
        # DIAGNOSTIC ONLY: unsafe matched ablation.
        "strict_exit_space": False,
        "required_exit_gap_meters": (
            DEFAULT_REQUIRED_EXIT_GAP_METERS
        ),
        "allow_unsafe_hard_max_fallback": False,
        "sumo_error_log": str(
            args.sumo_log_dir / f"train_worker_{int(rank):04d}.log"
        ),
        "ambulance_system": ambulance_system,
        "ambulance_max_active_choices": stage[
            "ambulance_max_active_choices"
        ],
        "ambulance_interval_range": stage[
            "ambulance_interval_range"
        ],
        "corridor": {
            "recovery_seconds": args.recovery_seconds,
            "max_preemption_seconds": args.max_preemption_seconds,
            "clearance_buffer_seconds": args.clearance_buffer_seconds,
            "prepare_eta_seconds": args.prepare_eta_seconds,
            "serve_eta_seconds": args.serve_eta_seconds,
        },
        "emergency_observation": dict(
            EMERGENCY_OBSERVATION_CONFIG
        ),
        "traffic_excluded_vehicle_prefixes": [
            AMBULANCE_ID_PREFIX
        ],
    }


def start_workers(
    ctx,
    args: argparse.Namespace,
    maps: list[Path],
    demand_bank: dict[str, list[dict[str, Any]]],
    base_embed_dim: int,
    base_graph_layers: int,
    rank_offset: int,
    progress: float,
):
    workers = []
    try:
        for index, net_file in enumerate(maps):
            parent, child = ctx.Pipe()
            config = worker_config(
                args,
                net_file,
                args.seed + rank_offset + index,
                rank_offset + index,
                base_embed_dim,
                base_graph_layers,
                curriculum_demand_routes(
                    demand_bank.get(str(net_file.resolve()), []),
                    progress,
                ),
                progress,
            )
            process = ctx.Process(
                target=emergency_rollout_worker_main,
                args=(child, config),
            )
            process.start()
            child.close()
            workers.append(
                {
                    "map": net_file,
                    "process": process,
                    "connection": parent,
                }
            )
        for worker in workers:
            connection = worker["connection"]
            if not connection.poll(args.worker_start_timeout):
                raise TimeoutError(
                    f"Ambulance worker did not start: {worker['map']}"
                )
            message = connection.recv()
            if message.get("type") == "error":
                detail = message["traceback"]
                if message.get("sumo_error_log"):
                    detail += (
                        "\nSUMO error log: "
                        + str(message["sumo_error_log"])
                    )
                raise RuntimeError(detail)
            if message.get("type") != "ready":
                raise RuntimeError(
                    f"Unexpected worker message: {message}"
                )
            print(
                f"[worker ready] {worker['map'].name}: "
                f"{message['tls']} TLS, "
                f"schedule={message['schedule_sha256'][:12]}",
                flush=True,
            )
        return workers
    except BaseException:
        close_workers(workers)
        raise


def close_workers(workers) -> None:
    for worker in workers:
        try:
            worker["connection"].send({"cmd": "close"})
        except Exception:
            pass
    for worker in workers:
        worker["process"].join(timeout=10.0)
        if worker["process"].is_alive():
            worker["process"].terminate()
            worker["process"].join(timeout=5.0)
        worker["connection"].close()


def collect_wave(
    workers,
    trainer: EmergencyPPOTrainer,
    args: argparse.Namespace,
    base_state: dict[str, np.ndarray],
    total_updates: int,
):
    authority = trainer.authority(total_updates)
    pending = {}
    override_state = trainer.state_dict_for_workers()
    for worker in workers:
        connection = worker["connection"]
        connection.send(
            {
                "cmd": "rollout",
                "base_state_dict": base_state,
                "override_state_dict": override_state,
                "rollout_steps": args.rollout_steps,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "authority": authority,
                "controller_mode": "base",
                "progress_interval": args.progress_interval,
                "deterministic": True,
            }
        )
        pending[connection] = worker
    results = []
    while pending:
        ready = wait(list(pending), timeout=60.0)
        if not ready:
            dead = [
                worker["map"].name
                for worker in pending.values()
                if not worker["process"].is_alive()
            ]
            if dead:
                raise RuntimeError(
                    f"Ambulance rollout workers exited: {dead}"
                )
            print(
                "[rollout] workers active; no message in 60s",
                flush=True,
            )
            continue
        for connection in ready:
            message = connection.recv()
            worker = pending[connection]
            if message.get("type") == "error":
                detail = message["traceback"]
                if message.get("sumo_error_log"):
                    detail += (
                        "\nSUMO error log: "
                        + str(message["sumo_error_log"])
                    )
                raise RuntimeError(detail)
            if message.get("type") == "progress":
                print(
                    f"[rollout] {worker['map'].name}: "
                    f"{100.0 * message['step'] / message['total']:.1f}% "
                    f"active={message['active_transitions']}",
                    flush=True,
                )
                continue
            if message.get("type") != "rollout":
                raise RuntimeError(
                    f"Unexpected worker message: {message}"
                )
            results.append(message)
            del pending[connection]
    return results, authority


def write_progress(
    args: argparse.Namespace,
    trainer: EmergencyPPOTrainer,
    total_updates: int,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": AMBULANCE_SCHEMA_VERSION,
        "status": status,
        "updated_unix": time.time(),
        "completed_updates": trainer.completed_updates,
        "total_updates": total_updates,
        "percentage": (
            100.0
            * trainer.completed_updates
            / max(1, total_updates)
        ),
        "active_transitions": trainer.active_transitions,
        "best_validation_score": (
            trainer.best_validation_score
            if math.isfinite(trainer.best_validation_score)
            else None
        ),
        "last_validated_round": trainer.last_validated_round,
    }
    if extra:
        payload.update(extra)
    args.progress_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.progress_file.with_suffix(
        args.progress_file.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.progress_file)


def validation_command(
    args: argparse.Namespace, round_index: int
) -> dict[str, Any]:
    output = (
        args.validation_dir
        / f"round_{round_index:03d}.json"
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "evaluate_ambulance_system.py"),
        "--manifest",
        args.manifest,
        "--splits",
        args.validation_splits,
        "--base-model-path",
        args.base_model_path,
        "--emergency-model-path",
        args.model_path,
        "--demand-bank-manifest",
        args.validation_demand_bank_manifest,
        "--output-json",
        str(output),
        "--seeds",
        args.validation_seeds,
        "--episode-seconds",
        str(args.validation_episode_seconds),
        "--decision-seconds",
        str(args.decision_seconds),
        "--workers",
        str(args.validation_workers),
        "--sumo-log-dir",
        str(args.sumo_log_dir / f"validation_round_{round_index:03d}"),
        "--ordinary-delay-budget-percent",
        str(args.ordinary_delay_budget_percent),
        "--throughput-budget-percent",
        str(args.throughput_budget_percent),
        "--ambulance-first-spawn",
        str(args.ambulance_first_spawn),
        "--ambulance-interval-seconds",
        str(
            0.5
            * (
                args.ambulance_interval_min
                + args.ambulance_interval_max
            )
        ),
        "--ambulance-spawn-jitter",
        str(args.ambulance_spawn_jitter),
        "--max-ambulances",
        str(args.max_ambulances_per_episode),
        "--max-active-ambulances",
        "2",
        "--planned-active-duration-factor",
        str(args.planned_active_duration_factor),
        "--ambulance-last-spawn-buffer",
        str(args.ambulance_last_spawn_buffer),
        "--ambulance-min-euclidean-distance",
        str(args.ambulance_min_euclidean_distance),
        "--ambulance-min-route-distance",
        str(args.ambulance_min_route_distance),
        "--ambulance-min-route-edges",
        str(args.ambulance_min_route_edges),
        "--ambulance-min-route-tls",
        str(args.ambulance_min_route_tls),
        "--ambulance-route-attempts",
        str(args.ambulance_route_attempts),
        "--reroute-interval",
        str(args.reroute_interval),
        "--reroute-jitter",
        str(args.reroute_jitter),
        "--reroute-min-savings-seconds",
        str(args.reroute_min_savings_seconds),
        "--reroute-min-savings-fraction",
        str(args.reroute_min_savings_fraction),
        "--no-reroute-within-tls",
        str(args.no_reroute_within_tls),
        "--recovery-seconds",
        str(args.recovery_seconds),
        "--max-preemption-seconds",
        str(args.max_preemption_seconds),
        "--clearance-buffer-seconds",
        str(args.clearance_buffer_seconds),
        "--prepare-eta-seconds",
        str(args.prepare_eta_seconds),
        "--serve-eta-seconds",
        str(args.serve_eta_seconds),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    marker = "AMBULANCE_VALIDATION_JSON="
    line = next(
        (
            value
            for value in reversed(result.stdout.splitlines())
            if value.startswith(marker)
        ),
        None,
    )
    if line is None:
        raise RuntimeError(
            "Ambulance validation produced no summary"
        )
    return dict(json.loads(line[len(marker) :]))


def save_best(
    trainer: EmergencyPPOTrainer,
    args: argparse.Namespace,
    total_updates: int,
    validation: dict[str, Any],
) -> None:
    trainer.save(total_updates)
    source_model = emergency_model_path(args.model_path)
    source_contract = source_model.with_name(
        source_model.stem + "_contract.json"
    )
    source_trainer = trainer_path(args.model_path)
    destination_model = emergency_model_path(args.best_model_path)
    destination_contract = destination_model.with_name(
        destination_model.stem + "_contract.json"
    )
    destination_trainer = trainer_path(args.best_model_path)
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, destination_model)
    shutil.copy2(source_contract, destination_contract)
    shutil.copy2(source_trainer, destination_trainer)
    destination_model.with_name(
        destination_model.stem + "_validation.json"
    ).write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="train")
    parser.add_argument(
        "--demand-bank-manifest", required=True
    )
    parser.add_argument(
        "--base-model-path",
        default="models/map_agnostic_multiagent_v3_best",
    )
    parser.add_argument(
        "--model-path",
        default="models/map_agnostic_emergency_v5",
    )
    parser.add_argument("--best-model-path", default="")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--num-map-workers", type=int, default=4)
    parser.add_argument(
        "--rollouts-per-map-visit", type=int, default=12
    )
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=4)
    parser.add_argument("--episode-seconds", type=int, default=3600)
    parser.add_argument("--decision-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--use-libsumo",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--worker-start-timeout", type=float, default=900.0
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("ambulance_v5_progress.json"),
    )
    parser.add_argument(
        "--sumo-log-dir",
        type=Path,
        default=Path("runs/ambulance_v5_sumo_logs"),
    )

    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument(
        "--target-density-range", default="2,12"
    )
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument(
        "--observation-noise-std", type=float, default=0.01
    )
    parser.add_argument(
        "--sensor-scale-jitter", type=float, default=0.05
    )
    parser.add_argument(
        "--sensor-dropout-prob", type=float, default=0.01
    )

    parser.add_argument(
        "--emergency-embed-dim", type=int, default=96
    )
    parser.add_argument(
        "--emergency-graph-layers", type=int, default=1
    )
    parser.add_argument("--residual-bound", type=float, default=4.0)
    parser.add_argument("--authority-start", type=float, default=0.50)
    parser.add_argument("--authority-end", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument(
        "--final-learning-rate", type=float, default=3e-5
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--value-clip-range", type=float, default=0.2
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--teacher-coef", type=float, default=0.20)
    parser.add_argument(
        "--teacher-decay-fraction", type=float, default=0.35
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)

    parser.add_argument(
        "--ambulance-first-spawn", type=float, default=30.0
    )
    parser.add_argument(
        "--ambulance-interval-min", type=float, default=90.0
    )
    parser.add_argument(
        "--ambulance-interval-max", type=float, default=240.0
    )
    parser.add_argument(
        "--ambulance-spawn-jitter", type=float, default=20.0
    )
    parser.add_argument(
        "--max-ambulances-per-episode", type=int, default=16
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
        "--validate-every-round",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validation-splits", default="validation"
    )
    parser.add_argument("--validation-seeds", default="9001,9002")
    parser.add_argument(
        "--validation-episode-seconds", type=int, default=1200
    )
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument(
        "--validation-demand-bank-manifest", default=""
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("runs/ambulance_v5_validation"),
    )
    parser.add_argument(
        "--ordinary-delay-budget-percent",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--throughput-budget-percent", type=float, default=2.0
    )
    args = parser.parse_args()
    args.splits = set(parse_csv(args.splits))
    density = [
        float(value)
        for value in parse_csv(args.target_density_range)
    ]
    if (
        len(density) != 2
        or density[0] <= 0.0
        or density[1] < density[0]
    ):
        parser.error("--target-density-range must be MIN,MAX")
    args.target_density_min, args.target_density_max = density
    if not args.best_model_path:
        args.best_model_path = (
            str(emergency_model_path(args.model_path).with_suffix(""))
            + "_best"
        )
    if not args.validation_demand_bank_manifest:
        args.validation_demand_bank_manifest = (
            args.demand_bank_manifest
        )
    if args.ambulance_interval_min <= 0.0 or (
        args.ambulance_interval_max
        < args.ambulance_interval_min
    ):
        parser.error("Invalid ambulance interval range")
    if args.ambulance_first_spawn + 1e-9 < STEP_LENGTH_SECONDS:
        parser.error(
            "--ambulance-first-spawn must leave at least one SUMO "
            "step for boundary-safe prequeuing"
        )
    if not (
        0.0
        <= args.authority_start
        <= args.authority_end
        <= 1.0
    ):
        parser.error("Authority must satisfy 0 <= start <= end <= 1")
    if args.episode_seconds <= 0 or args.decision_seconds <= 0.0:
        parser.error("Episode and decision durations must be positive")
    decision_steps = args.decision_seconds / STEP_LENGTH_SECONDS
    if abs(decision_steps - round(decision_steps)) > 1e-9:
        parser.error(
            "--decision-seconds must be an exact multiple of the "
            f"{STEP_LENGTH_SECONDS:g}s SUMO step"
        )
    for duration, name in (
        (args.episode_seconds, "--episode-seconds"),
        (
            args.validation_episode_seconds,
            "--validation-episode-seconds",
        ),
    ):
        decisions = float(duration) / args.decision_seconds
        if abs(decisions - round(decisions)) > 1e-9:
            parser.error(
                f"{name} must be an exact multiple of "
                "--decision-seconds"
            )
    if args.rounds <= 0 or args.rollout_steps <= 0:
        parser.error("Rounds and rollout steps must be positive")
    if args.residual_bound <= 0.0:
        parser.error("--residual-bound must be positive")
    if args.ambulance_last_spawn_buffer < 0.0:
        parser.error("--ambulance-last-spawn-buffer cannot be negative")
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = (
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Emergency PPO learner device: {device}")
    audit_manifest_splits(
        args.manifest,
        require_validation=args.validate_every_round,
        validation_splits=set(
            parse_csv(args.validation_splits)
        ),
    )
    maps = load_maps(args)
    demand_bank = load_ambulance_demand_bank(
        args.demand_bank_manifest,
        expected_episode_seconds=args.episode_seconds,
    )
    missing = [
        str(path)
        for path in maps
        if str(path.resolve()) not in demand_bank
    ]
    if missing:
        raise RuntimeError(
            "Demand bank is missing selected training maps: "
            + ", ".join(missing)
        )
    (
        base_checkpoint,
        base_state,
        base_embed_dim,
        base_graph_layers,
    ) = load_base_policy(args.base_model_path)

    ambulance_system_contract = {
        "routing_mode": "traffic_aware",
        "step_length_seconds": STEP_LENGTH_SECONDS,
        "spawn_queue_lead_seconds": STEP_LENGTH_SECONDS,
        "time_to_teleport": -1,
        "terminal_censor_penalty": True,
        "ordinary_delay_metric": "mean_time_loss_all_departed_s",
        "first_spawn_seconds": args.ambulance_first_spawn,
        "spawn_interval_range_seconds": [
            args.ambulance_interval_min,
            args.ambulance_interval_max,
        ],
        "spawn_jitter_seconds": args.ambulance_spawn_jitter,
        "max_ambulances": args.max_ambulances_per_episode,
        "max_active_ambulances_trained": [1, 2],
        "planned_active_duration_factor": (
            args.planned_active_duration_factor
        ),
        "min_euclidean_distance": (
            args.ambulance_min_euclidean_distance
        ),
        "min_route_distance": args.ambulance_min_route_distance,
        "min_route_edges": args.ambulance_min_route_edges,
        "min_route_tls": args.ambulance_min_route_tls,
        "route_attempts_per_ambulance": (
            args.ambulance_route_attempts
        ),
        "reroute_interval_seconds": args.reroute_interval,
        "reroute_jitter_seconds": args.reroute_jitter,
        "reroute_jitter_stream": "per_ambulance_sha256",
        "reroute_min_savings_seconds": (
            args.reroute_min_savings_seconds
        ),
        "reroute_min_savings_fraction": (
            args.reroute_min_savings_fraction
        ),
        "no_reroute_within_tls_meters": (
            args.no_reroute_within_tls
        ),
        "last_spawn_buffer_seconds": (
            args.ambulance_last_spawn_buffer
        ),
        "blue_light_device": False,
        "obeys_signals": True,
    }
    corridor_contract = {
        "horizon_tls": 3,
        "recovery_seconds": args.recovery_seconds,
        "max_preemption_seconds": args.max_preemption_seconds,
        "clearance_buffer_seconds": (
            args.clearance_buffer_seconds
        ),
        "prepare_eta_seconds": args.prepare_eta_seconds,
        "serve_eta_seconds": args.serve_eta_seconds,
        "min_emergency_downstream_space": (
            MIN_EMERGENCY_DOWNSTREAM_SPACE
        ),
        "budget_close_eta_exception_seconds": (
            CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS
        ),
        "strict_exit_space": True,
        "required_exit_gap_meters": (
            DEFAULT_REQUIRED_EXIT_GAP_METERS
        ),
        "allow_unsafe_hard_max_fallback": False,
        "recovery_teacher": "normalized_max_pressure",
    }
    contract = AmbulanceCheckpointContract.create(
        base_checkpoint=base_checkpoint,
        decision_seconds=args.decision_seconds,
        step_length_seconds=STEP_LENGTH_SECONDS,
        minimum_green_seconds=DEFAULT_MIN_GREEN,
        maximum_green_seconds=DEFAULT_MAX_GREEN,
        emergency_embed_dim=args.emergency_embed_dim,
        emergency_graph_layers=args.emergency_graph_layers,
        residual_bound=args.residual_bound,
        authority=args.authority_end,
        ambulance_system=ambulance_system_contract,
        emergency_observation=EMERGENCY_OBSERVATION_CONFIG,
        corridor=corridor_contract,
    )

    waves_per_round = math.ceil(
        len(maps) / max(1, args.num_map_workers)
    )
    total_updates = (
        args.rounds
        * waves_per_round
        * args.rollouts_per_map_visit
    )
    plan = {
        "maps": [str(path.resolve()) for path in maps],
        "base_checkpoint_sha256": contract.base_checkpoint_sha256,
        "rounds": args.rounds,
        "workers": args.num_map_workers,
        "rollouts_per_map_visit": (
            args.rollouts_per_map_visit
        ),
        "rollout_steps": args.rollout_steps,
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        "seed": args.seed,
        "emergency_embed_dim": args.emergency_embed_dim,
        "emergency_graph_layers": args.emergency_graph_layers,
        "residual_bound": args.residual_bound,
        "demand_bank_sha256": sha256_file(
            args.demand_bank_manifest
        ),
        "ambulance_system": ambulance_system_contract,
        "emergency_observation": EMERGENCY_OBSERVATION_CONFIG,
        "corridor": corridor_contract,
        "sensor_domain": {
            "observation_noise_std": (
                args.observation_noise_std
            ),
            "sensor_scale_jitter": args.sensor_scale_jitter,
            "sensor_dropout_prob": args.sensor_dropout_prob,
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "final_learning_rate": args.final_learning_rate,
            "ppo_epochs": args.ppo_epochs,
            "minibatch_size": args.minibatch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "value_clip_range": args.value_clip_range,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "teacher_coef": args.teacher_coef,
            "teacher_decay_fraction": (
                args.teacher_decay_fraction
            ),
            "max_grad_norm": args.max_grad_norm,
            "target_kl": args.target_kl,
            "authority_start": args.authority_start,
            "authority_end": args.authority_end,
        },
    }
    args.plan_signature = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode("utf-8")
    ).hexdigest()
    trainer = EmergencyPPOTrainer(args, device, contract)
    if not args.restart and trainer_path(args.model_path).exists():
        trainer.load()
        print(
            f"Resumed update={trainer.completed_updates}, "
            f"active_transitions={trainer.active_transitions}"
        )

    print("\nAmbulance override training plan")
    print(f"  frozen base:             {base_checkpoint}")
    print(f"  maps:                    {len(maps)}")
    print(f"  rounds:                  {args.rounds}")
    print(f"  workers:                 {args.num_map_workers}")
    print(f"  planned PPO updates:     {total_updates}")
    print(f"  decision interval:       {args.decision_seconds:g}s")
    print("  ordinary observations:   ambulances excluded")
    print("  no nearby ambulance:     frozen base + exit-space shield")
    print(f"  output:                  {emergency_model_path(args.model_path)}")

    ctx = mp.get_context("spawn")
    try:
        for round_index in range(args.rounds):
            order = list(maps)
            random.Random(args.seed + round_index).shuffle(order)
            for wave_index in range(waves_per_round):
                wave_maps = order[
                    wave_index * args.num_map_workers :
                    (wave_index + 1) * args.num_map_workers
                ]
                wave_start = (
                    (
                        round_index * waves_per_round
                        + wave_index
                    )
                    * args.rollouts_per_map_visit
                )
                wave_end = (
                    wave_start + args.rollouts_per_map_visit
                )
                if trainer.completed_updates >= wave_end:
                    continue
                progress = trainer.completed_updates / max(
                    1, total_updates
                )
                workers = start_workers(
                    ctx,
                    args,
                    wave_maps,
                    demand_bank,
                    base_embed_dim,
                    base_graph_layers,
                    (
                        round_index * waves_per_round
                        + wave_index
                    )
                    * args.num_map_workers,
                    progress,
                )
                try:
                    for visit in range(
                        args.rollouts_per_map_visit
                    ):
                        planned = wave_start + visit
                        if trainer.completed_updates > planned:
                            continue
                        start = time.monotonic()
                        rollouts, authority = collect_wave(
                            workers,
                            trainer,
                            args,
                            base_state,
                            total_updates,
                        )
                        for diagnostic_index, rollout in enumerate(
                            rollouts, start=1
                        ):
                            print(
                                f"[diagnostic ambulance {diagnostic_index}] "
                                + json.dumps(
                                    rollout["metrics"]["ambulance"],
                                    sort_keys=True,
                                    allow_nan=False,
                                ),
                                flush=True,
                            )
                        update = trainer.update(
                            rollouts,
                            total_updates,
                            authority,
                        )
                        trainer.save(total_updates)
                        elapsed = time.monotonic() - start
                        active = sum(
                            int(
                                rollout["metrics"][
                                    "active_transitions"
                                ]
                            )
                            for rollout in rollouts
                        )
                        ambulance_completion = [
                            float(
                                rollout["metrics"]["ambulance"][
                                    "completion_rate"
                                ]
                            )
                            for rollout in rollouts
                        ]
                        ambulance_arrivals = sum(
                            int(
                                rollout["metrics"]["ambulance"].get(
                                    "arrived_total", 0
                                )
                            )
                            for rollout in rollouts
                        )
                        response_weighted = sum(
                            float(
                                rollout["metrics"]["ambulance"].get(
                                    "mean_response_time_s", 0.0
                                )
                            )
                            * int(
                                rollout["metrics"]["ambulance"].get(
                                    "arrived_total", 0
                                )
                            )
                            for rollout in rollouts
                        )
                        mean_response = (
                            response_weighted / ambulance_arrivals
                            if ambulance_arrivals
                            else float("nan")
                        )
                        ambulance_failures = sum(
                            int(
                                rollout["metrics"]["ambulance"].get(
                                    "failed_total", 0
                                )
                            )
                            + int(
                                rollout["metrics"]["ambulance"].get(
                                    "censored_total", 0
                                )
                            )
                            for rollout in rollouts
                        )
                        print(
                            f"[update {trainer.completed_updates}/"
                            f"{total_updates}] "
                            f"active={active} "
                            f"completion={np.mean(ambulance_completion):.3f} "
                            f"arrived={ambulance_arrivals} "
                            f"response_s={mean_response:.1f} "
                            f"failed_or_censored={ambulance_failures} "
                            f"reward={np.mean([r['metrics']['mean_emergency_reward'] for r in rollouts]):.4f} "
                            f"policy={update['policy_loss']:.4f} "
                            f"value={update['value_loss']:.4f} "
                            f"authority={authority:.3f} "
                            f"wall={elapsed:.1f}s",
                            flush=True,
                        )
                        write_progress(
                            args,
                            trainer,
                            total_updates,
                            "training",
                            {
                                "round": round_index + 1,
                                "wave": wave_index + 1,
                                "visit": visit + 1,
                                "last_update": update,
                                "last_rollout_metrics": [
                                    rollout["metrics"]
                                    for rollout in rollouts
                                ],
                            },
                        )
                finally:
                    close_workers(workers)

            round_end = (
                (round_index + 1)
                * waves_per_round
                * args.rollouts_per_map_visit
            )
            if (
                args.validate_every_round
                and trainer.completed_updates >= round_end
                and trainer.last_validated_round
                < round_index + 1
            ):
                trainer.save(total_updates)
                validation = validation_command(
                    args, round_index + 1
                )
                eligible = bool(validation["eligible"])
                score = float(validation["selection_score"])
                print(
                    f"[validation] eligible={eligible} "
                    f"selection={score:.6f} "
                    f"ambulance_gain={validation['ambulance_gain_percent']:.2f}% "
                    f"delay_change={validation['ordinary_delay_change_percent']:.2f}% "
                    f"throughput_change={validation['throughput_change_percent']:.2f}%",
                    flush=True,
                )
                if (
                    eligible
                    and score > trainer.best_validation_score
                ):
                    trainer.best_validation_score = score
                    validation["selected_after_round"] = (
                        round_index + 1
                    )
                    save_best(
                        trainer,
                        args,
                        total_updates,
                        validation,
                    )
                    print(
                        "[validation] new constrained best: "
                        f"{emergency_model_path(args.best_model_path)}",
                        flush=True,
                    )
                trainer.last_validated_round = round_index + 1
                trainer.save(total_updates)
        trainer.save(total_updates)
        write_progress(
            args, trainer, total_updates, "complete"
        )
        print("\nAmbulance override training complete.")
    except BaseException:
        write_progress(args, trainer, total_updates, "failed")
        raise


if __name__ == "__main__":
    main()
