#!/usr/bin/env python3
"""Fast parameter-sharing multi-agent PPO over persistent exact-SUMO maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from torch.distributions import Categorical

from map_agnostic_multiagent_worker import rollout_worker_main
from map_agnostic_policy import (
    MapAgnosticMaskablePolicy,
    MovementGraphNetwork,
)
from map_agnostic_tls import (
    GLOBAL_FEATURE_NAMES,
    MAX_PHASES,
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_NAMES,
    empty_observation,
    observation_space,
)
from train_map_agnostic_multimap import (
    load_maps,
    parse_csv,
    passenger_lane_km,
)


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 3


class PolicyShapeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(MAX_PHASES + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = empty_observation()
        observation["movement_mask"][0] = 1.0
        observation["phase_membership"][0, 0] = 1.0
        return observation, {}

    def step(self, action):
        observation, _ = self.reset()
        return observation, 0.0, False, False, {}

    def action_masks(self):
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[:2] = True
        return mask


def model_zip(path: str | Path) -> Path:
    value = Path(path)
    return value if value.suffix == ".zip" else value.with_suffix(".zip")


def trainer_checkpoint(path: str | Path) -> Path:
    value = Path(path)
    base = value.with_suffix("") if value.suffix == ".zip" else value
    return base.parent / f"{base.name}_trainer.pt"


def metadata_path(path: str | Path) -> Path:
    value = Path(path)
    base = value.with_suffix("") if value.suffix == ".zip" else value
    return base.parent / f"{base.name}_map_agnostic.json"


def load_demand_bank(
    path: str, expected_episode_seconds: float | None = None
) -> dict[str, list[str]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    duration = float(payload.get("episode_seconds", 0.0) or 0.0)
    if (
        expected_episode_seconds is not None
        and duration + 1e-9 < float(expected_episode_seconds)
    ):
        raise RuntimeError(
            f"Demand bank lasts {duration:g}s but training episodes last "
            f"{expected_episode_seconds:g}s. Regenerate a bank at least as long."
        )
    by_map: dict[str, list[str]] = {}
    for record in payload.get("routes", []):
        net_file = str(Path(record["net_file"]).resolve())
        route_file = str(Path(record["route_file"]).resolve())
        if not Path(route_file).is_file():
            raise FileNotFoundError(route_file)
        by_map.setdefault(net_file, []).append(route_file)
    return by_map


def audit_manifest_splits(
    manifest_path: str,
    require_validation: bool,
    validation_splits: set[str],
) -> None:
    """Fail early on accidental train/validation/test file leakage."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = list(
        payload if isinstance(payload, list) else payload.get("maps", [])
    )
    by_path: dict[Path, set[str]] = {}
    by_split: dict[str, list[Path]] = {}
    for record in records:
        value = record.get("net_file") or record.get("path")
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        split = str(record.get("split", "train"))
        by_path.setdefault(path, set()).add(split)
        by_split.setdefault(split, []).append(path)
    duplicated_paths = {
        str(path): sorted(splits)
        for path, splits in by_path.items()
        if len(splits) > 1
    }
    if duplicated_paths:
        raise RuntimeError(
            "The same map path appears in multiple data splits: "
            + json.dumps(duplicated_paths, sort_keys=True)
        )
    hashes: dict[str, tuple[Path, str]] = {}
    for split, paths in by_split.items():
        for path in paths:
            if not path.is_file():
                continue
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            previous = hashes.get(digest)
            if previous is not None and previous[0] != path:
                raise RuntimeError(
                    "Byte-identical duplicate map files would leak or "
                    "overweight a domain: "
                    f"{previous[0]} ({previous[1]}) and {path} ({split})"
                )
            hashes[digest] = (path, split)
    if require_validation and not any(
        by_split.get(split) for split in validation_splits
    ):
        raise RuntimeError(
            "Validation is enabled but the manifest has no map in "
            f"--validation-splits={','.join(sorted(validation_splits))}."
        )
    print(
        "Manifest split audit: "
        + ", ".join(
            f"{split}={len(paths)}"
            for split, paths in sorted(by_split.items())
        )
        + " (no path/content leakage)",
        flush=True,
    )


def flatten_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {
        "movements": [],
        "phase_features": [],
        "global_features": [],
        "action_masks": [],
        "actions": [],
        "old_log_probs": [],
        "old_values": [],
        "advantages": [],
        "returns": [],
        "sample_weights": [],
        "teacher_actions": [],
        "static_indices": [],
    }
    static_banks: dict[str, list[np.ndarray]] = {
        "movement_mask": [],
        "movement_adjacency": [],
        "phase_membership": [],
    }
    static_offset = 0
    for rollout in rollouts:
        actions = rollout["actions"]
        time_steps, agents = actions.shape
        for key in ("movements", "phase_features", "global_features"):
            values = np.asarray(rollout["dynamic"][key])
            output[key].append(values.reshape((time_steps * agents, *values.shape[2:])))
        for key in static_banks:
            static = np.asarray(rollout["static"][key])
            static_banks[key].append(static)
        output["static_indices"].append(
            np.tile(
                np.arange(agents, dtype=np.int32) + static_offset,
                time_steps,
            )
        )
        static_offset += agents
        output["action_masks"].append(
            np.asarray(rollout["action_masks"]).reshape(
                time_steps * agents, MAX_PHASES + 1
            )
        )
        for key in (
            "actions",
            "old_log_probs",
            "old_values",
            "advantages",
            "returns",
            "teacher_actions",
        ):
            output[key].append(np.asarray(rollout[key]).reshape(-1))
        agent_weights = np.asarray(rollout["agent_weights"], dtype=np.float32)
        output["sample_weights"].append(np.tile(agent_weights, time_steps))
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


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class SharedPPOTrainer:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.network = MovementGraphNetwork(
            embed_dim=args.embed_dim,
            graph_layers=args.graph_layers,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )
        self.completed_updates = 0
        self.agent_transitions = 0
        self.best_validation_score = float("-inf")
        self.last_validated_round = 0

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.network.state_dict().items()
        }

    def state_dict_for_workers(self) -> dict[str, np.ndarray]:
        # NumPy uses the ordinary Pipe payload protocol. Sending torch tensors
        # makes multiprocessing start a separate resource-sharing socket,
        # which is slower and unavailable on some clusters/containers.
        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in self.network.state_dict().items()
        }

    def set_learning_rate(self, progress: float) -> float:
        progress = min(1.0, max(0.0, progress))
        learning_rate = (
            self.args.learning_rate
            + (self.args.final_learning_rate - self.args.learning_rate)
            * progress
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def update(
        self,
        rollouts: list[dict[str, Any]],
        total_planned_updates: int,
    ) -> dict[str, float]:
        batch = flatten_rollouts(rollouts)
        sample_count = len(batch["actions"])
        self.agent_transitions += sample_count
        progress = self.completed_updates / max(1, total_planned_updates)
        learning_rate = self.set_learning_rate(progress)

        weights_np = batch["sample_weights"].astype(np.float32)
        advantages_np = batch["advantages"].astype(np.float32)
        weighted_adv_mean = float(
            np.sum(weights_np * advantages_np) / max(1e-9, weights_np.sum())
        )
        weighted_adv_var = float(
            np.sum(weights_np * (advantages_np - weighted_adv_mean) ** 2)
            / max(1e-9, weights_np.sum())
        )
        advantages_np = (
            advantages_np - weighted_adv_mean
        ) / math.sqrt(weighted_adv_var + 1e-8)

        indices = np.arange(sample_count)
        metric_totals: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "teacher_loss": 0.0,
            "updates": 0.0,
        }
        self.network.train()
        stop_early = False
        for _epoch in range(self.args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, sample_count, self.args.minibatch_size):
                chosen = indices[start : start + self.args.minibatch_size]
                if len(chosen) == 0:
                    continue
                observation = {
                    key: torch.as_tensor(
                        batch[key][chosen],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for key in (
                        "movements",
                        "phase_features",
                        "global_features",
                    )
                }
                static_indices = batch["static_indices"][chosen]
                for key in (
                    "movement_mask",
                    "movement_adjacency",
                    "phase_membership",
                ):
                    observation[key] = torch.as_tensor(
                        batch[f"static_{key}"][static_indices],
                        dtype=torch.float32,
                        device=self.device,
                    )
                action_masks = torch.as_tensor(
                    batch["action_masks"][chosen],
                    dtype=torch.bool,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    batch["actions"][chosen],
                    dtype=torch.long,
                    device=self.device,
                )
                old_log_probs = torch.as_tensor(
                    batch["old_log_probs"][chosen],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_values = torch.as_tensor(
                    batch["old_values"][chosen],
                    dtype=torch.float32,
                    device=self.device,
                )
                returns = torch.as_tensor(
                    batch["returns"][chosen],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    advantages_np[chosen],
                    dtype=torch.float32,
                    device=self.device,
                )
                sample_weights = torch.as_tensor(
                    weights_np[chosen],
                    dtype=torch.float32,
                    device=self.device,
                )
                teacher_actions = torch.as_tensor(
                    batch["teacher_actions"][chosen],
                    dtype=torch.long,
                    device=self.device,
                )

                logits, values = self.network(observation)
                values = values.squeeze(-1)
                distribution = Categorical(
                    logits=logits.masked_fill(~action_masks, -1e8)
                )
                log_probs = distribution.log_prob(actions)
                entropy = distribution.entropy()
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.args.clip_range,
                        1.0 + self.args.clip_range,
                    )
                    * advantages
                )
                policy_loss = -weighted_mean(
                    torch.minimum(unclipped, clipped), sample_weights
                )

                clipped_values = old_values + torch.clamp(
                    values - old_values,
                    -self.args.value_clip_range,
                    self.args.value_clip_range,
                )
                value_loss_unclipped = (values - returns) ** 2
                value_loss_clipped = (clipped_values - returns) ** 2
                value_loss = 0.5 * weighted_mean(
                    torch.maximum(value_loss_unclipped, value_loss_clipped),
                    sample_weights,
                )
                entropy_mean = weighted_mean(entropy, sample_weights)
                teacher_fraction = max(
                    0.0,
                    1.0
                    - progress
                    / max(1e-9, self.args.teacher_decay_fraction),
                )
                teacher_coefficient = (
                    self.args.teacher_coef * teacher_fraction
                )
                teacher_loss = weighted_mean(
                    -distribution.log_prob(teacher_actions), sample_weights
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
                    self.network.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = weighted_mean(
                        (torch.exp(log_ratio) - 1.0) - log_ratio,
                        sample_weights,
                    )
                    clip_fraction = weighted_mean(
                        (torch.abs(ratio - 1.0) > self.args.clip_range).float(),
                        sample_weights,
                    )
                metric_totals["policy_loss"] += float(policy_loss.item())
                metric_totals["value_loss"] += float(value_loss.item())
                metric_totals["entropy"] += float(entropy_mean.item())
                metric_totals["approx_kl"] += float(approx_kl.item())
                metric_totals["clip_fraction"] += float(clip_fraction.item())
                metric_totals["teacher_loss"] += float(teacher_loss.item())
                metric_totals["updates"] += 1.0
                if (
                    self.args.target_kl > 0
                    and float(approx_kl.item()) > 1.5 * self.args.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        self.completed_updates += 1
        denominator = max(1.0, metric_totals.pop("updates"))
        metrics = {
            key: value / denominator for key, value in metric_totals.items()
        }
        metrics.update(
            learning_rate=learning_rate,
            samples=float(sample_count),
            advantage_mean=weighted_adv_mean,
            advantage_std=math.sqrt(weighted_adv_var + 1e-8),
            early_stop=float(stop_early),
            teacher_coefficient=float(
                self.args.teacher_coef
                * max(
                    0.0,
                    1.0
                    - progress
                    / max(1e-9, self.args.teacher_decay_fraction),
                )
            ),
        )
        return metrics

    def save_trainer(self, args: argparse.Namespace) -> None:
        path = trainer_checkpoint(args.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "completed_updates": self.completed_updates,
                "agent_transitions": self.agent_transitions,
                "best_validation_score": self.best_validation_score,
                "last_validated_round": self.last_validated_round,
                "embed_dim": args.embed_dim,
                "graph_layers": args.graph_layers,
                "plan_signature": args.plan_signature,
            },
            path,
        )

    def load_trainer(self, args: argparse.Namespace) -> None:
        path = trainer_checkpoint(args.model_path)
        payload = torch.load(path, map_location=self.device)
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(f"Incompatible trainer checkpoint: {path}")
        checkpoint_plan = str(payload.get("plan_signature", ""))
        if checkpoint_plan and checkpoint_plan != args.plan_signature:
            raise RuntimeError(
                "The resume checkpoint belongs to a different map/update "
                "schedule. Repeat the original command, choose a new "
                "--model-path, or pass --restart to intentionally start over."
            )
        self.network.load_state_dict(payload["network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.completed_updates = int(payload.get("completed_updates", 0))
        self.agent_transitions = int(payload.get("agent_transitions", 0))
        self.best_validation_score = float(
            payload.get("best_validation_score", float("-inf"))
        )
        self.last_validated_round = int(
            payload.get("last_validated_round", 0)
        )


def export_sb3_checkpoint(
    trainer: SharedPPOTrainer,
    args: argparse.Namespace,
    destination: str | Path | None = None,
) -> None:
    destination = destination or args.model_path
    env = DummyVecEnv([PolicyShapeEnv])
    model = MaskablePPO(
        policy=MapAgnosticMaskablePolicy,
        env=env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={
            "embed_dim": args.embed_dim,
            "graph_layers": args.graph_layers,
        },
        device="cpu",
        verbose=0,
    )
    model.policy.map_network.load_state_dict(trainer.state_dict_cpu())
    model.num_timesteps = trainer.agent_transitions
    model.save(str(Path(destination).with_suffix("")))
    env.close()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_class": "map_agnostic_policy.MapAgnosticMaskablePolicy",
        "trainer": "parameter_sharing_multiagent_ppo",
        "analytical_observation_normalization": True,
        "vecnormalize_required": False,
        "movement_features": list(MOVEMENT_FEATURE_NAMES),
        "phase_features": list(PHASE_FEATURE_NAMES),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "embed_dim": args.embed_dim,
        "graph_layers": args.graph_layers,
        "decision_seconds": args.decision_seconds,
        "agent_transitions": trainer.agent_transitions,
        "completed_updates": trainer.completed_updates,
    }
    metadata_path(destination).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def worker_config(
    args: argparse.Namespace,
    net_file: Path,
    rank: int,
    demand_bank: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "net_file": str(net_file),
        "passenger_lane_km": passenger_lane_km(net_file),
        "seed": args.seed + rank * 1009,
        "worker_rank": rank,
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        "target_density_range": (
            args.target_density_min,
            args.target_density_max,
        ),
        "max_vehicle_center": args.max_vehicle_center,
        "spawn_batch_center": args.spawn_batch_center,
        "observation_noise_std": args.observation_noise_std,
        "sensor_scale_jitter": args.sensor_scale_jitter,
        "sensor_dropout_prob": args.sensor_dropout_prob,
        "embed_dim": args.embed_dim,
        "graph_layers": args.graph_layers,
        "demand_routes": demand_bank.get(str(net_file.resolve()), []),
        "use_libsumo": args.use_libsumo,
    }


def start_workers(
    ctx,
    args: argparse.Namespace,
    maps: list[Path],
    demand_bank: dict[str, list[str]],
    rank_offset: int,
):
    workers = []
    for local_rank, net_file in enumerate(maps):
        parent, child = ctx.Pipe()
        config = worker_config(
            args, net_file, rank_offset + local_rank, demand_bank
        )
        process = ctx.Process(
            target=rollout_worker_main,
            args=(child, config),
            daemon=False,
        )
        process.start()
        child.close()
        workers.append(
            {"process": process, "connection": parent, "map": net_file}
        )
    try:
        for worker in workers:
            connection = worker["connection"]
            if not connection.poll(args.worker_start_timeout):
                raise TimeoutError(f"Worker did not start: {worker['map']}")
            message = connection.recv()
            if message.get("type") == "error":
                raise RuntimeError(message["traceback"])
            if message.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker message: {message}")
            print(
                f"[worker ready] {Path(message['net_file']).name}: "
                f"{message['tls']} TLS",
                flush=True,
            )
    except BaseException:
        close_workers(workers)
        raise
    return workers


def close_workers(workers) -> None:
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
        try:
            worker["connection"].close()
        except Exception:
            pass


def write_progress(
    args: argparse.Namespace,
    trainer: SharedPPOTrainer,
    total_updates: int,
    active_progress: dict[str, float],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    partial = (
        sum(active_progress.values()) / len(active_progress)
        if active_progress
        else 0.0
    )
    overall = 100.0 * min(
        1.0,
        (trainer.completed_updates + partial) / max(1, total_updates),
    )
    payload: dict[str, Any] = {
        "status": status,
        "overall_percent": overall,
        "completed_updates": trainer.completed_updates,
        "total_updates": total_updates,
        "agent_transitions": trainer.agent_transitions,
        "active_rollouts": active_progress,
        "updated_at": time.time(),
    }
    if extra:
        payload.update(extra)
    args.progress_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.progress_file.with_suffix(args.progress_file.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.progress_file)


def collect_wave_rollouts(
    workers,
    trainer: SharedPPOTrainer,
    args: argparse.Namespace,
    total_updates: int,
) -> list[dict[str, Any]]:
    state = trainer.state_dict_for_workers()
    pending = {}
    progress = {}
    for worker in workers:
        connection = worker["connection"]
        key = worker["map"].name
        pending[connection] = worker
        progress[key] = 0.0
        connection.send(
            {
                "cmd": "rollout",
                "state_dict": state,
                "rollout_steps": args.rollout_steps,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "progress_interval": args.progress_interval,
                "deterministic": False,
            }
        )

    results = []
    while pending:
        ready_connections = wait(list(pending), timeout=60.0)
        if not ready_connections:
            dead = [
                worker["map"].name
                for worker in pending.values()
                if not worker["process"].is_alive()
            ]
            if dead:
                raise RuntimeError(f"Rollout workers exited: {dead}")
            print("[rollout] workers active; no message in the last 60s", flush=True)
            continue
        for connection in ready_connections:
            message = connection.recv()
            worker = pending[connection]
            key = worker["map"].name
            if message.get("type") == "error":
                raise RuntimeError(message["traceback"])
            if message.get("type") == "progress":
                progress[key] = float(message["step"]) / max(
                    1.0, float(message["total"])
                )
                print(
                    f"[rollout progress] {key}: "
                    f"{100.0 * progress[key]:5.1f}% "
                    f"({message['transitions']} agent transitions, "
                    f"{message['tls']} TLS)",
                    flush=True,
                )
                write_progress(
                    args,
                    trainer,
                    total_updates,
                    progress,
                    status="collecting",
                )
                continue
            if message.get("type") != "rollout":
                raise RuntimeError(f"Unexpected worker message: {message}")
            progress[key] = 1.0
            results.append(message)
            del pending[connection]
    return results


def validation_command(args: argparse.Namespace, round_index: int) -> dict[str, Any]:
    output = args.validation_dir / f"round_{round_index:03d}.json"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "validate_map_agnostic_multiagent.py"),
        "--manifest",
        str(args.manifest),
        "--splits",
        args.validation_splits,
        "--model-path",
        args.model_path,
        "--output-json",
        str(output),
        "--seeds",
        args.validation_seeds,
        "--episode-seconds",
        str(args.validation_episode_seconds),
        "--decision-seconds",
        str(args.decision_seconds),
        "--target-density",
        str(args.validation_target_density),
        "--max-vehicle-center",
        str(args.max_vehicle_center),
        "--spawn-batch-center",
        str(args.spawn_batch_center),
        "--device",
        args.device,
        "--workers",
        str(args.validation_workers),
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
    marker = "MAP_AGNOSTIC_VALIDATION_JSON="
    line = next(
        (value for value in reversed(result.stdout.splitlines()) if value.startswith(marker)),
        None,
    )
    if line is None:
        raise RuntimeError("Validation produced no summary")
    return dict(json.loads(line[len(marker) :]))


def save_best(
    trainer: SharedPPOTrainer,
    args: argparse.Namespace,
    validation: dict[str, Any],
) -> None:
    export_sb3_checkpoint(trainer, args, args.best_model_path)
    destination_trainer = trainer_checkpoint(args.best_model_path)
    destination_trainer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trainer_checkpoint(args.model_path), destination_trainer)
    validation_path = model_zip(args.best_model_path).with_name(
        model_zip(args.best_model_path).stem + "_validation.json"
    )
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--maps",
        default="",
        help="Optional comma-separated .net.xml files in addition to the manifest.",
    )
    parser.add_argument("--splits", default="train")
    parser.add_argument("--model-path", default="models/traffic_signal_map_agnostic_v3")
    parser.add_argument("--best-model-path", default="")
    parser.add_argument("--demand-bank-manifest", default="")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--num-map-workers", type=int, default=4)
    parser.add_argument("--rollouts-per-map-visit", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=2)
    parser.add_argument("--episode-seconds", type=int, default=7200)
    parser.add_argument("--decision-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--use-libsumo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--worker-start-timeout", type=float, default=900.0)
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("map_agnostic_multiagent_progress.json"),
    )

    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument("--target-density-range", default="2,10")
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument("--sensor-scale-jitter", type=float, default=0.05)
    parser.add_argument("--sensor-dropout-prob", type=float, default=0.01)

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--teacher-coef",
        type=float,
        default=0.10,
        help="Early normalized-MaxPressure imitation coefficient (0 disables).",
    )
    parser.add_argument(
        "--teacher-decay-fraction",
        type=float,
        default=0.25,
        help="Fraction of training over which the heuristic teacher decays to zero.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)

    parser.add_argument(
        "--validate-every-round",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-splits", default="validation")
    parser.add_argument("--validation-seeds", default="9001,9002")
    parser.add_argument("--validation-episode-seconds", type=int, default=600)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--validation-target-density", type=float, default=6.0)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("runs/map_agnostic_multiagent_validation"),
    )
    args = parser.parse_args()
    args.splits = set(parse_csv(args.splits))
    density = [float(value) for value in parse_csv(args.target_density_range)]
    if len(density) != 2 or density[0] <= 0 or density[1] < density[0]:
        parser.error("--target-density-range must be MIN,MAX")
    args.target_density_min, args.target_density_max = density
    if not args.best_model_path:
        base = args.model_path[:-4] if args.model_path.endswith(".zip") else args.model_path
        args.best_model_path = base + "_best"
    positive_integer_fields = (
        "rounds",
        "num_map_workers",
        "rollouts_per_map_visit",
        "rollout_steps",
        "progress_interval",
        "episode_seconds",
        "ppo_epochs",
        "minibatch_size",
        "embed_dim",
        "validation_workers",
    )
    for field in positive_integer_fields:
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.decision_seconds <= 0:
        parser.error("--decision-seconds must be positive")
    if args.teacher_coef < 0 or args.teacher_decay_fraction <= 0:
        parser.error(
            "--teacher-coef must be nonnegative and "
            "--teacher-decay-fraction must be positive"
        )
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Central PPO learner device: {device}")

    audit_manifest_splits(
        args.manifest,
        require_validation=args.validate_every_round,
        validation_splits=set(parse_csv(args.validation_splits)),
    )
    maps = load_maps(args)
    demand_bank = load_demand_bank(
        args.demand_bank_manifest,
        expected_episode_seconds=args.episode_seconds,
    )
    if args.demand_bank_manifest:
        missing_demand = [
            str(path)
            for path in maps
            if str(path.resolve()) not in demand_bank
        ]
        if missing_demand:
            raise RuntimeError(
                "Demand bank is missing selected training maps: "
                + ", ".join(missing_demand)
            )
    waves_per_round = math.ceil(len(maps) / max(1, args.num_map_workers))
    total_updates = (
        args.rounds * waves_per_round * args.rollouts_per_map_visit
    )
    plan_payload = {
        "maps": [str(path.resolve()) for path in maps],
        "rounds": args.rounds,
        "num_map_workers": args.num_map_workers,
        "rollouts_per_map_visit": args.rollouts_per_map_visit,
        "rollout_steps": args.rollout_steps,
        "episode_seconds": args.episode_seconds,
        "decision_seconds": args.decision_seconds,
        "seed": args.seed,
        "embed_dim": args.embed_dim,
        "graph_layers": args.graph_layers,
    }
    args.plan_signature = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    trainer = SharedPPOTrainer(args, device)
    if not args.restart and trainer_checkpoint(args.model_path).exists():
        trainer.load_trainer(args)
        print(
            f"Resumed {trainer_checkpoint(args.model_path)} at "
            f"update={trainer.completed_updates}, "
            f"agent_transitions={trainer.agent_transitions}"
        )

    print("\nPersistent multi-agent training plan")
    print(f"  maps:                    {len(maps)}")
    print(f"  map workers at once:     {args.num_map_workers}")
    print(f"  rounds:                  {args.rounds}")
    print(f"  waves/round:             {waves_per_round}")
    print(f"  rollouts/map visit:      {args.rollouts_per_map_visit}")
    print(f"  temporal steps/rollout:  {args.rollout_steps}")
    print(f"  planned PPO updates:     {total_updates}")
    print(f"  decision interval:       {args.decision_seconds:g}s")
    print(f"  episode duration:        {args.episode_seconds}s")
    print("  transition source:       every usable TLS in every map worker")
    print("  map weighting:           equal map weight; balanced topology buckets")
    print(f"  model:                   {model_zip(args.model_path)}")

    if trainer.completed_updates >= total_updates and (
        not args.validate_every_round
        or trainer.last_validated_round >= args.rounds
    ):
        export_sb3_checkpoint(trainer, args)
        write_progress(args, trainer, total_updates, {}, status="complete")
        print("Checkpoint already completed this exact training plan.")
        return

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
                print(
                    f"\n[round {round_index + 1}/{args.rounds} "
                    f"wave {wave_index + 1}/{waves_per_round}] "
                    + ", ".join(path.name for path in wave_maps),
                    flush=True,
                )
                wave_start_update = (
                    (round_index * waves_per_round + wave_index)
                    * args.rollouts_per_map_visit
                )
                wave_end_update = (
                    wave_start_update + args.rollouts_per_map_visit
                )
                if trainer.completed_updates >= wave_end_update:
                    print(
                        f"[resume] wave already complete "
                        f"({wave_end_update}/{total_updates} updates)",
                        flush=True,
                    )
                    continue
                workers = start_workers(
                    ctx,
                    args,
                    wave_maps,
                    demand_bank,
                    (round_index * waves_per_round + wave_index)
                    * args.num_map_workers,
                )
                try:
                    for visit in range(args.rollouts_per_map_visit):
                        planned_update = wave_start_update + visit
                        if trainer.completed_updates > planned_update:
                            print(
                                f"[resume] skip visit {visit + 1}; "
                                f"checkpoint already has update "
                                f"{planned_update + 1}",
                                flush=True,
                            )
                            continue
                        print(
                            f"[collect] visit {visit + 1}/"
                            f"{args.rollouts_per_map_visit}",
                            flush=True,
                        )
                        update_start_wall = time.monotonic()
                        rollouts = collect_wave_rollouts(
                            workers, trainer, args, total_updates
                        )
                        collection_metrics = [
                            rollout["metrics"] for rollout in rollouts
                        ]
                        update_metrics = trainer.update(
                            rollouts, total_updates
                        )
                        trainer.save_trainer(args)
                        export_sb3_checkpoint(trainer, args)
                        update_wall_seconds = (
                            time.monotonic() - update_start_wall
                        )
                        transitions = sum(
                            int(item["transitions"])
                            for item in collection_metrics
                        )
                        wall = max(
                            float(item["wall_seconds"])
                            for item in collection_metrics
                        )
                        throughput = transitions / max(1e-9, wall)
                        overall_percent = (
                            100.0
                            * trainer.completed_updates
                            / max(1, total_updates)
                        )
                        print(
                            f"[update {trainer.completed_updates}/{total_updates}] "
                            f"{overall_percent:6.2f}% | "
                            f"samples={transitions} | "
                            f"collection={wall:.1f}s | "
                            f"throughput={throughput:.2f} agent-transitions/s | "
                            f"reward={np.mean([m['mean_reward'] for m in collection_metrics]):.4f} | "
                            f"policy_loss={update_metrics['policy_loss']:.4f} | "
                            f"value_loss={update_metrics['value_loss']:.4f} | "
                            f"kl={update_metrics['approx_kl']:.5f}",
                            flush=True,
                        )
                        write_progress(
                            args,
                            trainer,
                            total_updates,
                            {},
                            status="training",
                            extra={
                                "round": round_index + 1,
                                "wave": wave_index + 1,
                                "visit": visit + 1,
                                "last_collection_metrics": collection_metrics,
                                "last_update_metrics": update_metrics,
                                "last_update_wall_seconds": update_wall_seconds,
                                "estimated_seconds_remaining": (
                                    max(
                                        0,
                                        total_updates
                                        - trainer.completed_updates,
                                    )
                                    * update_wall_seconds
                                ),
                            },
                        )
                finally:
                    close_workers(workers)

            round_end_update = (
                (round_index + 1)
                * waves_per_round
                * args.rollouts_per_map_visit
            )
            if (
                args.validate_every_round
                and trainer.completed_updates >= round_end_update
                and trainer.last_validated_round < round_index + 1
            ):
                export_sb3_checkpoint(trainer, args)
                validation = validation_command(args, round_index + 1)
                score = float(validation["selection_score"])
                print(
                    f"[validation] round={round_index + 1} "
                    f"selection={score:.6f} "
                    f"worst_map={float(validation['worst_map_score']):.6f} "
                    f"mean_map={float(validation['mean_map_score']):.6f}",
                    flush=True,
                )
                if score > trainer.best_validation_score:
                    trainer.best_validation_score = score
                    trainer.save_trainer(args)
                    validation["selected_after_round"] = round_index + 1
                    save_best(trainer, args, validation)
                    print(
                        f"[validation] new best: {model_zip(args.best_model_path)}",
                        flush=True,
                    )
                trainer.last_validated_round = round_index + 1
                trainer.save_trainer(args)
        export_sb3_checkpoint(trainer, args)
        trainer.save_trainer(args)
        write_progress(
            args,
            trainer,
            total_updates,
            {},
            status="complete",
        )
        print("\nPersistent multi-agent training complete.")
    except BaseException:
        write_progress(
            args,
            trainer,
            total_updates,
            {},
            status="failed",
        )
        raise


if __name__ == "__main__":
    main()
