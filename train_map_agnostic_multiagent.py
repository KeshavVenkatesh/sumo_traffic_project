#!/usr/bin/env python3
"""Map-balanced PPO with synchronized all-TLS SUMO rollout workers.

Each update launches one process per selected map.  Workers control every
compatible signal simultaneously and return trajectories; only then does the
central learner perform a PPO update over the pooled, map-balanced batch.  No
worker writes the model, so checkpoint races and sequential-task forgetting are
avoided.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from checkpoint_contract import (
    CheckpointContract,
    validate_checkpoint_runtime,
    write_checkpoint_contract,
)
from map_agnostic_tls import OBSERVATION_KEYS
from safe_residual_controller import CMPPConfig
from safe_residual_policy import SafeResidualMapAgnosticPolicy
from traffic_rl_map_agnostic_env import MapAgnosticPolicyShapeEnv, legacy, sim
from train_map_agnostic_multimap import passenger_lane_km


ROOT = Path(__file__).resolve().parent


def parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def model_zip(path: str | Path) -> Path:
    value = Path(path)
    return value if value.suffix == ".zip" else value.with_suffix(".zip")


def load_maps(args: argparse.Namespace) -> list[Path]:
    records: list[Mapping[str, Any]] = []
    manifest_path: Path | None = None
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = list(payload if isinstance(payload, list) else payload.get("maps", []))
    paths: list[Path] = []
    for record in records:
        if str(record.get("split", "train")) not in args.splits:
            continue
        value = record.get("net_file") or record.get("path")
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute() and manifest_path is not None:
                path = manifest_path.parent / path
            paths.append(path)
    paths.extend(Path(value).expanduser() for value in parse_csv(args.maps))
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path not in seen:
            seen.add(path)
            result.append(path)
    if not result:
        raise RuntimeError("No maps selected; pass --manifest or --maps")
    return result


def shape_env():
    return DummyVecEnv([lambda: MapAgnosticPolicyShapeEnv()])


def contract_for_policy(policy: SafeResidualMapAgnosticPolicy) -> CheckpointContract:
    return CheckpointContract.create(
        decision_seconds=float(sim.DECISION_INTERVAL),
        step_length_seconds=float(sim.STEP_LENGTH),
        minimum_green_seconds=float(legacy.MIN_GREEN_BEFORE_SWITCH),
        maximum_green_seconds=float(legacy.HARD_MAX_GREEN),
        residual_authority=float(policy.residual_authority),
        residual_bound=float(policy.residual_bound),
        max_baseline_regret=float(policy.max_baseline_regret),
        cmpp_config=policy.cmpp_config,
        adapter_names=policy.adapter_names,
        active_adapter=policy.active_adapter,
    )


def create_or_load_model(args: argparse.Namespace, env) -> MaskablePPO:
    path = Path(args.model_path)
    source_path: Path | None = None
    if args.resume and model_zip(path).exists():
        source_path = path
    elif args.initialize_from:
        source_path = Path(args.initialize_from).expanduser().resolve()
        if not model_zip(source_path).exists():
            raise FileNotFoundError(model_zip(source_path))
    if source_path is not None:
        validate_checkpoint_runtime(
            source_path,
            decision_seconds=float(sim.DECISION_INTERVAL),
            step_length_seconds=float(sim.STEP_LENGTH),
            minimum_green_seconds=float(legacy.MIN_GREEN_BEFORE_SWITCH),
            maximum_green_seconds=float(legacy.HARD_MAX_GREEN),
        )
        model = MaskablePPO.load(str(source_path), env=env, device=args.device)
        if not isinstance(model.policy, SafeResidualMapAgnosticPolicy):
            raise RuntimeError("Checkpoint is not a safe-residual map-agnostic policy")
        return model
    return MaskablePPO(
        policy=SafeResidualMapAgnosticPolicy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=0,
        device=args.device,
        seed=args.seed,
        policy_kwargs={
            "embed_dim": args.embed_dim,
            "graph_layers": args.graph_layers,
            "residual_authority": 0.0,
            "residual_bound": args.residual_bound,
            "max_baseline_regret": args.max_baseline_regret,
            "cmpp_config": CMPPConfig().to_dict(),
        },
    )


def tensor_observation(observation: Mapping[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in observation.items()
    }


def collect_worker(args: argparse.Namespace) -> None:
    """Internal subprocess entry point; writes one map rollout to .npz."""

    from map_agnostic_multiagent_env import MapAgnosticAllTLSEnv

    torch.set_num_threads(max(1, int(args.torch_threads)))
    env_shape = shape_env()
    model = MaskablePPO.load(args.worker_model, env=env_shape, device=args.worker_device)
    if not isinstance(model.policy, SafeResidualMapAgnosticPolicy):
        raise RuntimeError("Worker model is not SafeResidualMapAgnosticPolicy")
    policy = model.policy
    policy.set_training_mode(False)

    environment = MapAgnosticAllTLSEnv(
        net_file=args.worker_net_file,
        episode_seconds=args.episode_seconds,
        seed=args.worker_seed,
        max_vehicle_center=args.worker_max_vehicles,
        target_vehicle_center=args.worker_target_vehicles,
        initial_vehicle_center=args.worker_initial_vehicles,
        spawn_batch_center=args.spawn_batch_center,
    )
    observations_by_key: dict[str, list[np.ndarray]] = {key: [] for key in OBSERVATION_KEYS}
    masks_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    log_probs_list: list[np.ndarray] = []
    values_list: list[np.ndarray] = []
    rewards_list: list[np.ndarray] = []
    dones_list: list[bool] = []
    last_state = None
    try:
        state = environment.reset()
        agent_count = len(state.tls_ids)
        for _step in range(args.rollout_steps):
            obs_tensor = tensor_observation(state.observations, policy.device)
            with torch.no_grad():
                actions, values, log_probs = policy.forward(
                    obs_tensor,
                    deterministic=False,
                    action_masks=state.action_masks,
                )
            action_array = actions.detach().cpu().numpy().reshape(-1)
            next_state = environment.step(action_array)
            for key in OBSERVATION_KEYS:
                observations_by_key[key].append(np.asarray(state.observations[key]))
            masks_list.append(np.asarray(state.action_masks, dtype=bool))
            actions_list.append(action_array.astype(np.int64))
            log_probs_list.append(log_probs.detach().cpu().numpy().reshape(-1))
            values_list.append(values.detach().cpu().numpy().reshape(-1))
            rewards_list.append(np.asarray(next_state.rewards, dtype=np.float32))
            done = bool(next_state.terminated or next_state.truncated)
            dones_list.append(done)
            state = next_state
            last_state = next_state
            if done:
                break

        if not actions_list:
            raise RuntimeError("Worker collected no transitions")
        with torch.no_grad():
            if dones_list[-1]:
                bootstrap = np.zeros(agent_count, dtype=np.float32)
            else:
                bootstrap = (
                    policy.predict_values(
                        tensor_observation(state.observations, policy.device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float32)
                )

        rewards = np.stack(rewards_list)
        values = np.stack(values_list).astype(np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(agent_count, dtype=np.float32)
        for index in reversed(range(len(rewards_list))):
            if index == len(rewards_list) - 1:
                next_values = bootstrap
            else:
                next_values = values[index + 1]
            nonterminal = 0.0 if dones_list[index] else 1.0
            delta = rewards[index] + args.gamma * nonterminal * next_values - values[index]
            last_gae = delta + args.gamma * args.gae_lambda * nonterminal * last_gae
            advantages[index] = last_gae
        returns = advantages + values

        payload: dict[str, np.ndarray] = {
            f"obs__{key}": np.concatenate(values_for_key, axis=0)
            for key, values_for_key in observations_by_key.items()
        }
        payload.update(
            action_masks=np.concatenate(masks_list, axis=0),
            actions=np.concatenate(actions_list, axis=0),
            old_log_probs=np.concatenate(log_probs_list, axis=0).astype(np.float32),
            old_values=values.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            map_name=np.asarray(Path(args.worker_net_file).name),
            agent_count=np.asarray(agent_count, dtype=np.int64),
            rollout_steps=np.asarray(len(actions_list), dtype=np.int64),
            info_json=np.asarray(json.dumps(last_state.info if last_state else {})),
        )
        np.savez_compressed(args.worker_output, **payload)
    finally:
        environment.close()
        env_shape.close()


@dataclass
class RolloutBatch:
    observations: dict[str, np.ndarray]
    action_masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    sample_weights: np.ndarray
    map_names: list[str]
    infos: list[dict[str, Any]]


def load_rollouts(paths: Sequence[Path]) -> RolloutBatch:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    try:
        sizes = [len(item["actions"]) for item in loaded]
        total = sum(sizes)
        map_count = len(sizes)
        weights = [
            np.full(size, total / max(1.0, map_count * size), dtype=np.float32)
            for size in sizes
        ]
        return RolloutBatch(
            observations={
                key: np.concatenate([item[f"obs__{key}"] for item in loaded], axis=0)
                for key in OBSERVATION_KEYS
            },
            action_masks=np.concatenate([item["action_masks"] for item in loaded], axis=0),
            actions=np.concatenate([item["actions"] for item in loaded], axis=0),
            old_log_probs=np.concatenate([item["old_log_probs"] for item in loaded], axis=0),
            old_values=np.concatenate([item["old_values"] for item in loaded], axis=0),
            advantages=np.concatenate([item["advantages"] for item in loaded], axis=0),
            returns=np.concatenate([item["returns"] for item in loaded], axis=0),
            sample_weights=np.concatenate(weights, axis=0),
            map_names=[str(item["map_name"].item()) for item in loaded],
            infos=[json.loads(str(item["info_json"].item())) for item in loaded],
        )
    finally:
        for item in loaded:
            item.close()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-9)


def ppo_update(
    model: MaskablePPO,
    batch: RolloutBatch,
    args: argparse.Namespace,
    update_index: int,
) -> dict[str, float]:
    policy = model.policy
    assert isinstance(policy, SafeResidualMapAgnosticPolicy)
    policy.set_training_mode(True)
    device = policy.device
    advantages = np.asarray(batch.advantages, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
    sample_count = len(batch.actions)
    rng = np.random.default_rng(args.seed + update_index * 10_007)
    teacher_coef = args.teacher_coef * max(
        0.0, 1.0 - update_index / max(1, args.teacher_decay_updates)
    )
    stats: dict[str, list[float]] = {
        "loss": [], "policy_loss": [], "value_loss": [], "entropy": [],
        "teacher_loss": [], "approx_kl": [], "clip_fraction": [],
    }

    for _epoch in range(args.ppo_epochs):
        order = rng.permutation(sample_count)
        for start in range(0, sample_count, args.batch_size):
            indices = order[start : start + args.batch_size]
            obs = tensor_observation(
                {key: value[indices] for key, value in batch.observations.items()},
                device,
            )
            masks = torch.as_tensor(batch.action_masks[indices], dtype=torch.bool, device=device)
            actions = torch.as_tensor(batch.actions[indices], dtype=torch.long, device=device)
            old_log_probs = torch.as_tensor(
                batch.old_log_probs[indices], dtype=torch.float32, device=device
            )
            returns = torch.as_tensor(batch.returns[indices], dtype=torch.float32, device=device)
            batch_advantages = torch.as_tensor(
                advantages[indices], dtype=torch.float32, device=device
            )
            weights = torch.as_tensor(
                batch.sample_weights[indices], dtype=torch.float32, device=device
            )

            values, log_probs, entropy = policy.evaluate_actions(
                obs, actions, action_masks=masks
            )
            values = values.reshape(-1)
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = batch_advantages * ratio
            clipped = batch_advantages * torch.clamp(
                ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
            )
            policy_loss = -weighted_mean(torch.minimum(unclipped, clipped), weights)
            value_loss = weighted_mean((returns - values) ** 2, weights)
            entropy_loss = (
                -weighted_mean(entropy, weights)
                if entropy is not None
                else weighted_mean(log_probs, weights)
            )

            combined_logits, _values_again, baseline_logits = policy.safe_logits(obs, masks)
            baseline_actions = baseline_logits.masked_fill(~masks, -1e8).argmax(dim=-1)
            teacher_per_sample = F.cross_entropy(
                combined_logits, baseline_actions, reduction="none"
            )
            teacher_loss = weighted_mean(teacher_per_sample, weights)
            loss = (
                policy_loss
                + args.vf_coef * value_loss
                + args.ent_coef * entropy_loss
                + teacher_coef * teacher_loss
            )

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            policy.optimizer.step()

            with torch.no_grad():
                log_ratio = log_probs - old_log_probs
                approx_kl = weighted_mean((torch.exp(log_ratio) - 1.0) - log_ratio, weights)
                clip_fraction = weighted_mean(
                    (torch.abs(ratio - 1.0) > args.clip_range).float(), weights
                )
            for name, value in (
                ("loss", loss), ("policy_loss", policy_loss),
                ("value_loss", value_loss), ("entropy", -entropy_loss),
                ("teacher_loss", teacher_loss), ("approx_kl", approx_kl),
                ("clip_fraction", clip_fraction),
            ):
                stats[name].append(float(value.detach().cpu()))
        if stats["approx_kl"] and stats["approx_kl"][-1] > args.target_kl:
            break
    model.num_timesteps += sample_count
    return {name: float(np.mean(values)) for name, values in stats.items() if values}


def worker_command(
    args: argparse.Namespace,
    *,
    net_file: Path,
    output: Path,
    seed: int,
    target: int,
) -> list[str]:
    maximum = min(args.max_vehicle_center, max(target, int(round(1.2 * target))))
    initial = min(target, max(40, int(round(0.25 * target))))
    return [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--collect-worker", "--worker-net-file", str(net_file),
        "--worker-model", str(Path(args.model_path).resolve()),
        "--worker-output", str(output), "--worker-seed", str(seed),
        "--worker-max-vehicles", str(maximum),
        "--worker-target-vehicles", str(target),
        "--worker-initial-vehicles", str(initial),
        "--worker-device", args.rollout_device,
        "--rollout-steps", str(args.rollout_steps),
        "--episode-seconds", str(args.episode_seconds),
        "--spawn-batch-center", str(args.spawn_batch_center),
        "--gamma", str(args.gamma), "--gae-lambda", str(args.gae_lambda),
        "--torch-threads", str(args.worker_torch_threads),
    ]


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, args.torch_threads))
    maps = load_maps(args)
    lane_km = {path: passenger_lane_km(path) for path in maps}
    env_shape = shape_env()
    model = create_or_load_model(args, env_shape)
    policy = model.policy
    assert isinstance(policy, SafeResidualMapAgnosticPolicy)
    if args.adapter_name:
        policy.add_adapter(args.adapter_name)
        policy.freeze_for_adapter(args.adapter_name)
        if len(maps) != 1:
            print(
                "WARNING: adapter training normally uses exactly one certified map; "
                f"received {len(maps)} maps."
            )
    else:
        policy.set_active_adapter(None)
        policy.unfreeze_all()
    for group in policy.optimizer.param_groups:
        group["lr"] = float(args.learning_rate)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_file)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    completed_updates = 0
    if args.resume and progress_path.exists():
        try:
            saved_progress = json.loads(progress_path.read_text())
            same_model = str(saved_progress.get("model_path", "")) == str(
                Path(args.model_path).resolve()
            )
            same_adapter = saved_progress.get("active_adapter") == (
                args.adapter_name or None
            )
            if same_model and same_adapter:
                completed_updates = int(saved_progress["completed_updates"])
        except Exception:
            completed_updates = 0
    rng = random.Random(args.seed)
    total_updates = args.updates
    for update_index in range(completed_updates, total_updates):
        authority_progress = min(1.0, (update_index + 1) / max(1, args.authority_warmup_updates))
        policy.residual_authority = (
            args.residual_authority
            if args.adapter_name
            else args.residual_authority * authority_progress
        )
        model.save(args.model_path)
        write_checkpoint_contract(args.model_path, contract_for_policy(policy))

        shuffled = list(maps)
        rng.shuffle(shuffled)
        selected = shuffled[: min(len(shuffled), args.maps_per_update)]
        with tempfile.TemporaryDirectory(prefix="multicity_rollout_") as temporary:
            temporary_path = Path(temporary)
            processes = []
            outputs: list[Path] = []
            logs: list[Path] = []
            for map_index, net_file in enumerate(selected):
                density = rng.uniform(args.target_density_min, args.target_density_max)
                target = min(
                    args.max_vehicle_center,
                    max(100, int(round(lane_km[net_file] * density))),
                )
                output = temporary_path / f"rollout_{map_index}.npz"
                log = temporary_path / f"rollout_{map_index}.log"
                command = worker_command(
                    args,
                    net_file=net_file,
                    output=output,
                    seed=args.seed + update_index * 100_003 + map_index * 1009,
                    target=target,
                )
                log_handle = log.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT
                )
                processes.append((process, log_handle, net_file))
                outputs.append(output)
                logs.append(log)

            for process, log_handle, net_file in processes:
                return_code = process.wait()
                log_handle.close()
                if return_code != 0:
                    matching_log = logs[[item[2] for item in processes].index(net_file)]
                    raise RuntimeError(
                        f"Rollout worker failed for {net_file}:\n"
                        + matching_log.read_text(encoding="utf-8", errors="replace")
                    )
            batch = load_rollouts(outputs)
            update_stats = ppo_update(model, batch, args, update_index)

        model.save(args.model_path)
        write_checkpoint_contract(args.model_path, contract_for_policy(policy))
        progress = {
            "completed_updates": update_index + 1,
            "total_updates": total_updates,
            "model_path": str(Path(args.model_path).resolve()),
            "total_timesteps": int(model.num_timesteps),
            "residual_authority": float(policy.residual_authority),
            "active_adapter": policy.active_adapter,
            "maps": batch.map_names,
            "worker_infos": batch.infos,
            "ppo": update_stats,
        }
        temporary_progress = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary_progress.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        temporary_progress.replace(progress_path)
        print(
            f"[update {update_index + 1}/{total_updates}] samples={len(batch.actions)} "
            f"maps={batch.map_names} authority={policy.residual_authority:.3f} "
            f"loss={update_stats.get('loss', 0.0):.4f}"
        )
    env_shape.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="train")
    parser.add_argument("--model-path", default="models/traffic_signal_multicity_safe_residual_v1")
    parser.add_argument("--initialize-from", default="",
                        help="Foundation checkpoint used when --model-path does not yet exist (for a new map-adapter bundle).")
    parser.add_argument("--adapter-name", default="",
                        help="Train only this small map adapter while freezing the universal backbone.")
    parser.add_argument("--progress-file", default="multicity_safe_residual_progress.json")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--maps-per-update", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--episode-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-vehicle-center", type=int, default=1800)
    parser.add_argument("--target-density-range", default="2.0,10.0")
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--residual-authority", type=float, default=0.20)
    parser.add_argument("--authority-warmup-updates", type=int, default=20)
    parser.add_argument("--residual-bound", type=float, default=1.0)
    parser.add_argument("--max-baseline-regret", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--teacher-coef", type=float, default=0.05)
    parser.add_argument("--teacher-decay-updates", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--worker-torch-threads", type=int, default=1)

    # Internal worker-only options.
    parser.add_argument("--collect-worker", action="store_true")
    parser.add_argument("--worker-net-file", default="")
    parser.add_argument("--worker-model", default="")
    parser.add_argument("--worker-output", default="")
    parser.add_argument("--worker-seed", type=int, default=42)
    parser.add_argument("--worker-max-vehicles", type=int, default=1500)
    parser.add_argument("--worker-target-vehicles", type=int, default=1200)
    parser.add_argument("--worker-initial-vehicles", type=int, default=300)
    parser.add_argument("--worker-device", default="cpu")
    args = parser.parse_args()
    args.splits = set(parse_csv(args.splits))
    density = [float(value) for value in parse_csv(args.target_density_range)]
    if len(density) != 2 or density[0] <= 0.0 or density[1] < density[0]:
        parser.error("--target-density-range must be MIN,MAX with 0 < MIN <= MAX")
    args.target_density_min, args.target_density_max = density
    if not 0.0 <= args.residual_authority <= 1.0:
        parser.error("--residual-authority must be in [0, 1]")
    args.maps_per_update = max(1, args.maps_per_update)
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.collect_worker:
        collect_worker(arguments)
    else:
        train(arguments)
