#!/usr/bin/env python3
"""Train or evaluate the shared movement-GNN policy on one map/TLS task.

Use train_map_agnostic_multimap.py for the normal balanced multi-map workflow.
This lower-level entry point intentionally handles one map/TLS per process so
SUMO and the large realistic simulator never retain caches from another map.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from map_agnostic_policy import MapAgnosticMaskablePolicy
from map_agnostic_tls import (
    GLOBAL_FEATURE_NAMES,
    MAX_MOVEMENTS,
    MAX_PHASES,
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_NAMES,
)
from traffic_rl_map_agnostic_env import (
    MapAgnosticExactTrafficSignalEnv,
    configure_network,
    legacy,
)


SCHEMA_VERSION = 2


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)

    return schedule


def make_env(rank: int, args: argparse.Namespace, eval_mode: bool = False):
    def _init():
        seed = int(args.seed) + 1009 * rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        fixed_scenario = None
        if eval_mode and args.eval_fixed_scenario:
            # The legacy env otherwise hard-codes seed 42 whenever scenario
            # randomization is disabled, making a multi-seed validation set
            # silently repeat the same demand.
            fixed_scenario = legacy.build_fixed_scenario(seed=seed, args=args)
        env = MapAgnosticExactTrafficSignalEnv(
            tls_id=args.tls_id,
            episode_seconds=args.episode_seconds,
            gui=bool(args.gui and eval_mode and rank == 0),
            randomize_scenarios=not eval_mode or not args.eval_fixed_scenario,
            base_seed=seed,
            env_rank=rank,
            print_scenarios=args.print_scenarios,
            fixed_scenario=fixed_scenario,
            max_vehicle_center=args.max_vehicle_center,
            target_center=args.target_vehicle_center,
            initial_center=args.initial_vehicle_center,
            spawn_batch_center=args.spawn_batch_center,
            green_duration_center=args.green_duration_center,
            density_spread=args.density_spread,
            initial_spread=args.initial_spread,
            observation_noise_std=(0.0 if eval_mode else args.observation_noise_std),
            sensor_scale_jitter=(0.0 if eval_mode else args.sensor_scale_jitter),
            sensor_dropout_prob=(0.0 if eval_mode else args.sensor_dropout_prob),
        )
        return Monitor(env)

    return _init


def build_vec_env(args: argparse.Namespace, eval_mode: bool = False):
    n_envs = 1 if eval_mode else max(1, int(args.num_envs))
    env_fns = [make_env(rank, args, eval_mode=eval_mode) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def model_zip(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def metadata_path(path: Path) -> Path:
    base = path.with_suffix("") if path.suffix == ".zip" else path
    return base.parent / f"{base.name}_map_agnostic.json"


def write_metadata(args: argparse.Namespace, model: MaskablePPO) -> None:
    path = metadata_path(Path(args.model_path))
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

    tasks = list(previous.get("training_tasks", []))
    task = {"net_file": str(Path(args.net_file).resolve()), "tls_id": args.tls_id}
    if task not in tasks:
        tasks.append(task)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_class": "map_agnostic_policy.MapAgnosticMaskablePolicy",
        "analytical_observation_normalization": True,
        "vecnormalize_required": False,
        "max_movements": MAX_MOVEMENTS,
        "max_phases": MAX_PHASES,
        "movement_features": list(MOVEMENT_FEATURE_NAMES),
        "phase_features": list(PHASE_FEATURE_NAMES),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "embed_dim": args.embed_dim,
        "graph_layers": args.graph_layers,
        "observation_noise_std": args.observation_noise_std,
        "sensor_scale_jitter": args.sensor_scale_jitter,
        "sensor_dropout_prob": args.sensor_dropout_prob,
        "training_tasks": tasks,
        "total_timesteps": int(model.num_timesteps),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_or_create_model(env, args: argparse.Namespace) -> MaskablePPO:
    path = Path(args.model_path)
    existing = model_zip(path)
    if args.resume and existing.exists():
        print(f"Resuming map-agnostic checkpoint: {existing}")
        try:
            model = MaskablePPO.load(str(path), env=env, device=args.device)
        except Exception as exc:
            raise RuntimeError(
                "The requested checkpoint is not compatible with schema v2. "
                "Old 30/46-value, five-action checkpoints cannot be converted because "
                "their phase-index semantics are map-specific. Start a new model path."
            ) from exc
        if not isinstance(model.policy, MapAgnosticMaskablePolicy):
            raise RuntimeError("Checkpoint did not contain MapAgnosticMaskablePolicy.")
        model.learning_rate = linear_schedule(args.lr_start, args.lr_end)
        model.lr_schedule = model.learning_rate
        return model

    print("Starting a fresh map-agnostic movement-GNN checkpoint.")
    return MaskablePPO(
        policy=MapAgnosticMaskablePolicy,
        env=env,
        learning_rate=linear_schedule(args.lr_start, args.lr_end),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        verbose=1,
        tensorboard_log=str(args.tensorboard_dir),
        device=args.device,
        seed=args.seed,
        policy_kwargs={
            "embed_dim": args.embed_dim,
            "graph_layers": args.graph_layers,
        },
    )


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, int(args.torch_threads)))
    env = build_vec_env(args, eval_mode=False)
    model = load_or_create_model(env, args)

    save_base = Path(args.model_path)
    save_base.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(1, args.num_envs)),
        save_path=str(save_base.parent),
        name_prefix=save_base.stem + "_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("\nMap-agnostic training task")
    print(f"  map:                  {args.net_file}")
    print(f"  TLS:                  {args.tls_id}")
    print(f"  timesteps:            {args.timesteps}")
    print(f"  parallel SUMO envs:   {args.num_envs}")
    print(f"  analytical normalize: yes (VecNormalize disabled)")
    print(f"  model:                {model_zip(save_base)}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint,
        progress_bar=args.progress_bar,
        reset_num_timesteps=not (args.resume and model_zip(save_base).exists()),
        tb_log_name="map_agnostic_movement_gnn",
    )
    model.save(str(save_base))
    write_metadata(args, model)
    env.close()
    print(f"Saved: {model_zip(save_base)}")
    print(f"Saved: {metadata_path(save_base)}")


def evaluate(args: argparse.Namespace) -> None:
    env = build_vec_env(args, eval_mode=True)
    model = MaskablePPO.load(str(Path(args.model_path)), env=env, device=args.device)
    if not isinstance(model.policy, MapAgnosticMaskablePolicy):
        raise RuntimeError("This evaluator requires a schema-v2 map-agnostic checkpoint.")

    obs = env.reset()
    total_reward = 0.0
    metric_totals = {
        "mean_queue_density": 0.0,
        "mean_vehicle_density": 0.0,
        "mean_downstream_occupancy": 0.0,
        "spillback": 0.0,
        "max_starvation": 0.0,
    }
    metric_samples = 0
    steps_completed = 0
    last_info: dict[str, Any] = {}
    try:
        for step in range(1, args.eval_steps + 1):
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, rewards, dones, infos = env.step(action)
            total_reward += float(rewards[0])
            steps_completed = step
            last_info = dict(infos[0])
            for key in metric_totals:
                metric_totals[key] += float(last_info.get(key, 0.0) or 0.0)
            metric_samples += 1
            if step % args.eval_print_every == 0:
                info = last_info
                print(
                    f"step={step:6d} t={float(info.get('sim_time', 0)):8.1f} "
                    f"reward={total_reward:10.3f} "
                    f"queue_density={float(info.get('mean_queue_density', 0)):.3f} "
                    f"spillback={float(info.get('spillback', 0)):.3f} "
                    f"arrived={int(info.get('total_arrived', 0))}"
                )
            if bool(np.any(dones)):
                break
    finally:
        env.close()

    result = {
        "schema_version": SCHEMA_VERSION,
        "net_file": str(Path(args.net_file).resolve()),
        "tls_id": args.tls_id,
        "seed": int(args.seed),
        "steps": int(steps_completed),
        "mean_reward_per_step": float(total_reward / max(1, steps_completed)),
        "total_reward": float(total_reward),
        "total_arrived": int(last_info.get("total_arrived", 0) or 0),
        **{
            key: float(value / max(1, metric_samples))
            for key, value in metric_totals.items()
        },
    }
    marker = "MAP_AGNOSTIC_EVAL_JSON=" + json.dumps(result, separators=(",", ":"))
    print(marker)
    if args.eval_json:
        args.eval_json.parent.mkdir(parents=True, exist_ok=True)
        args.eval_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-file", default=os.environ.get("TRAFFIC_NET_FILE", "new_map.net.xml"))
    parser.add_argument("--tls-id", required=True)
    parser.add_argument("--model-path", default="models/traffic_signal_map_agnostic_v2")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--episode-seconds", type=int, default=900)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-scenarios", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-fixed-scenario", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-steps", type=int, default=2_500)
    parser.add_argument("--eval-print-every", type=int, default=50)
    parser.add_argument("--eval-json", type=Path)

    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument("--target-vehicle-center", type=int, default=1200)
    parser.add_argument("--initial-vehicle-center", type=int, default=300)
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--green-duration-center", type=float, default=30.0)
    parser.add_argument("--density-spread", type=float, default=0.35)
    parser.add_argument("--initial-spread", type=float, default=0.65)
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--sensor-scale-jitter",
        type=float,
        default=0.05,
        help="Per-episode log-normal detector calibration jitter.",
    )
    parser.add_argument(
        "--sensor-dropout-prob",
        type=float,
        default=0.01,
        help="Per-episode probability of dropping each dynamic sensor channel.",
    )

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--lr-start", type=float, default=2.5e-4)
    parser.add_argument("--lr-end", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/map_agnostic_v2"))
    args = parser.parse_args()

    args.net_file = str(configure_network(args.net_file))
    args.num_envs = max(1, args.num_envs)
    args.target_vehicle_center = min(args.target_vehicle_center, args.max_vehicle_center)
    args.initial_vehicle_center = min(args.initial_vehicle_center, args.target_vehicle_center)
    args.density_spread = min(0.75, max(0.0, args.density_spread))
    args.initial_spread = min(1.5, max(0.0, args.initial_spread))
    args.observation_noise_std = min(0.10, max(0.0, args.observation_noise_std))
    args.sensor_scale_jitter = min(0.25, max(0.0, args.sensor_scale_jitter))
    args.sensor_dropout_prob = min(0.25, max(0.0, args.sensor_dropout_prob))
    if args.batch_size > args.n_steps * args.num_envs:
        parser.error("--batch-size cannot exceed --n-steps * --num-envs")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
