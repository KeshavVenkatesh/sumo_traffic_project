#!/usr/bin/env python3
"""
Fast, strong MaskablePPO trainer for the SUMO traffic-signal model.

This version intentionally uses the fast proxy environment in traffic_rl_model.py
instead of running the full realistic_all_intersections_fixed_cycle.py loop for
training.  That is the same reason the older training file could run around
30 it/s: the expensive visual-simulation helpers are not executed on every PPO
step.

Training goal:
  - keep the traffic centered around the current realistic simulation settings
    (750 max vehicles, 650 target vehicles, 200 initial vehicles, spawn batch 12)
  - disable ambulances during training/evaluation
  - train much longer and more robustly with curriculum, VecNormalize,
    checkpointing, and optional deterministic evaluation
  - support GUI eval if the fast environment exposes a gui constructor argument

Typical strong training command:
    python train_traffic_signal_rl.py \
      --timesteps 2000000 \
      --episode-seconds 900 \
      --model-path models/traffic_signal_maskable_ppo_fast_proxy_strong \
      --no-resume \
      --progress-bar

Typical GUI evaluation command:
    python train_traffic_signal_rl.py \
      --eval-only \
      --gui \
      --model-path models/traffic_signal_maskable_ppo_fast_proxy_strong \
      --eval-steps 20000
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing RL dependencies. Install them with:\n"
        "python -m pip install gymnasium stable-baselines3 sb3-contrib tensorboard tqdm rich"
    ) from exc


DEFAULT_TLS_ID = "cluster_12179861947_12179861948_12179861949_12185616643_#11more"
DEFAULT_MODEL_BASENAME = "models/traffic_signal_maskable_ppo_fast_proxy_strong"
DEFAULT_ENV_MODULE = "traffic_rl_model"

# Match the latest realistic simulation command as the center of training.
SIM_CENTER_MAX_VEHICLES = 750
SIM_CENTER_TARGET_VEHICLES = 650
SIM_CENTER_INITIAL_VEHICLES = 200
SIM_CENTER_SPAWN_BATCH = 12
SIM_CENTER_ROUTE_LOOKAHEAD = 60
SIM_CENTER_GREEN_DURATION = 45.0
SIM_CENTER_NO_LANE_CHANGE_DISTANCE = 100.0
SIM_CENTER_LANE_PREP_DISTANCE = 320.0


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    """Stable-Baselines schedule: progress_remaining goes from 1.0 to 0.0."""

    def schedule(progress_remaining: float) -> float:
        progress_done = 1.0 - progress_remaining
        return start + progress_done * (end - start)

    return schedule


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(default)

    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values or list(default)


def with_zip_suffix(path: str | Path) -> str:
    path = str(path)
    return path if path.endswith(".zip") else path + ".zip"


def parent_dir(path: str | Path) -> Path:
    return Path(path).expanduser().resolve().parent


def set_module_attr_if_present(module: Any, name: str, value: Any) -> None:
    if hasattr(module, name):
        setattr(module, name, value)


def set_first_present(module: Any, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        set_module_attr_if_present(module, name, value)


def constructor_accepts(cls: type, name: str) -> bool:
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def discover_route_variants(module: Any, base_dir: Path) -> list[str]:
    """Find route variants used by the fast proxy environment."""

    if hasattr(module, "discover_background_route_variants"):
        try:
            variants = module.discover_background_route_variants()
            if variants:
                return [str(v) for v in variants]
        except Exception:
            pass

    variants = sorted(base_dir.glob("background_train_*.rou.xml"))
    if variants:
        return [str(path) for path in variants]

    if hasattr(module, "BACKGROUND_ROUTE_FILE"):
        return [str(getattr(module, "BACKGROUND_ROUTE_FILE"))]

    fallback = base_dir / "background_new.rou.xml"
    return [str(fallback)]


# -----------------------------------------------------------------------------
# Strong curriculum
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumStage:
    name: str
    timesteps: int
    max_vehicle_variants: list[int]
    target_vehicles: int
    initial_vehicles: int
    spawn_batch: int
    route_lookahead_edges: int


def build_curriculum(args: argparse.Namespace) -> list[CurriculumStage]:
    """Dense curriculum centered around the current simulation traffic level."""

    if args.no_curriculum:
        variants = parse_int_list(
            args.vehicle_variants,
            [args.target_vehicles, args.max_vehicles],
        )
        return [
            CurriculumStage(
                name="single_stage",
                timesteps=args.timesteps,
                max_vehicle_variants=variants,
                target_vehicles=args.target_vehicles,
                initial_vehicles=args.initial_vehicles,
                spawn_batch=args.spawn_batch,
                route_lookahead_edges=args.route_lookahead_edges,
            )
        ]

    warmup_steps = int(args.timesteps * args.warmup_fraction)
    main_steps = int(args.timesteps * args.main_fraction)
    stress_steps = int(args.timesteps * args.stress_fraction)
    polish_steps = max(0, args.timesteps - warmup_steps - main_steps - stress_steps)

    center_max = min(args.max_vehicles, SIM_CENTER_MAX_VEHICLES)
    center_target = min(args.target_vehicles, center_max)

    stages: list[CurriculumStage] = []

    if warmup_steps > 0:
        stages.append(
            CurriculumStage(
                name="dense_warmup",
                timesteps=warmup_steps,
                max_vehicle_variants=sorted(set([450, 550, min(650, center_max)])),
                target_vehicles=min(560, center_target),
                initial_vehicles=min(max(120, args.initial_vehicles), 220),
                spawn_batch=max(8, min(args.spawn_batch + 4, 20)),
                route_lookahead_edges=max(35, min(args.route_lookahead_edges, 50)),
            )
        )

    if main_steps > 0:
        stages.append(
            CurriculumStage(
                name="simulation_center",
                timesteps=main_steps,
                max_vehicle_variants=sorted(set([550, 650, center_max])),
                target_vehicles=center_target,
                initial_vehicles=args.initial_vehicles,
                spawn_batch=args.spawn_batch,
                route_lookahead_edges=args.route_lookahead_edges,
            )
        )

    if stress_steps > 0:
        stages.append(
            CurriculumStage(
                name="stress_generalization",
                timesteps=stress_steps,
                max_vehicle_variants=sorted(set([650, 700, center_max])),
                target_vehicles=min(center_max, max(center_target, 700)),
                initial_vehicles=min(max(args.initial_vehicles, 250), center_target),
                spawn_batch=max(args.spawn_batch, 16),
                route_lookahead_edges=args.route_lookahead_edges,
            )
        )

    if polish_steps > 0:
        stages.append(
            CurriculumStage(
                name="sim_polish",
                timesteps=polish_steps,
                max_vehicle_variants=sorted(set([center_target, center_max])),
                target_vehicles=center_target,
                initial_vehicles=args.initial_vehicles,
                spawn_batch=args.spawn_batch,
                route_lookahead_edges=args.route_lookahead_edges,
            )
        )

    return [stage for stage in stages if stage.timesteps > 0]


# -----------------------------------------------------------------------------
# Fast proxy environment creation
# -----------------------------------------------------------------------------

def apply_env_globals(module: Any, args: argparse.Namespace, stage: CurriculumStage) -> None:
    """Patch common globals in traffic_rl_model.py when that file exposes them."""

    set_first_present(module, ("TRAIN_EPISODE_SECONDS", "EPISODE_SECONDS"), args.episode_seconds)
    set_first_present(module, ("SIM_END", "DEFAULT_SIM_END"), max(args.episode_seconds, getattr(module, "SIM_END", 0)))

    set_first_present(module, ("MAX_NUM_VEHICLES", "MAX_ACTIVE_VEHICLE_CAP", "MAX_VEHICLES"), args.max_vehicles)
    set_first_present(module, ("MAX_VEHICLE_VARIANTS", "VEHICLE_VARIANTS"), stage.max_vehicle_variants)
    set_first_present(module, ("TARGET_VEHICLES", "TARGET_ACTIVE_VEHICLES"), stage.target_vehicles)
    set_first_present(module, ("INITIAL_VEHICLES", "INITIAL_ACTIVE_VEHICLES"), stage.initial_vehicles)
    set_first_present(module, ("SPAWN_BATCH", "SPAWN_BATCH_SIZE"), stage.spawn_batch)
    set_first_present(module, ("ROUTE_LOOKAHEAD_EDGES", "LOOKAHEAD_EDGES"), stage.route_lookahead_edges)
    set_first_present(module, ("GREEN_DURATION", "DEFAULT_GREEN_DURATION"), args.green_duration)

    # Carry over the latest no-late-lane-change idea when the fast env supports it.
    set_first_present(
        module,
        ("INTERSECTION_NO_LANE_CHANGE_DISTANCE", "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE", "TLS_NO_LANE_CHANGE_DISTANCE"),
        args.intersection_no_lane_change_distance,
    )
    set_first_present(
        module,
        ("INTERSECTION_LANE_PREP_DISTANCE", "TRAFFIC_LIGHT_LANE_PREP_DISTANCE", "TLS_LANE_PREP_DISTANCE"),
        args.intersection_lane_prep_distance,
    )

    # Training difference requested by the user: no ambulances.
    for name in (
        "USE_AMBULANCES",
        "ENABLE_AMBULANCES",
        "AMBULANCES_ENABLED",
        "SPAWN_AMBULANCES",
        "TRAIN_WITH_AMBULANCES",
    ):
        set_module_attr_if_present(module, name, False)
    for name in ("AMBULANCE_INTERVAL", "AMBULANCE_SPAWN_INTERVAL"):
        set_module_attr_if_present(module, name, 10**12)

    # Quiet training.
    set_first_present(module, ("TRAIN_WITH_SUMO_LOGS", "PRINT_SUMO_LOGS"), False)
    set_first_present(module, ("PRINT_TRAINING_SCENARIOS", "PRINT_SCENARIOS"), args.print_scenarios)


def make_raw_env(
    env_class: type,
    module: Any,
    args: argparse.Namespace,
    route_variants: list[str],
    stage: CurriculumStage,
    rank: int = 0,
    eval_mode: bool = False,
):
    def _init():
        seed = args.seed + rank * 1009 + (100_000 if eval_mode else 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        kwargs: dict[str, Any] = {}
        if constructor_accepts(env_class, "tls_id"):
            kwargs["tls_id"] = args.tls_id
        if constructor_accepts(env_class, "gui"):
            kwargs["gui"] = bool(args.gui and eval_mode and rank == 0)
        if constructor_accepts(env_class, "randomize_traffic"):
            kwargs["randomize_traffic"] = not eval_mode or not args.eval_fixed_scenario
        if constructor_accepts(env_class, "route_variants"):
            kwargs["route_variants"] = route_variants
        if constructor_accepts(env_class, "max_vehicle_variants"):
            kwargs["max_vehicle_variants"] = stage.max_vehicle_variants
        if constructor_accepts(env_class, "seed"):
            kwargs["seed"] = seed

        env = env_class(**kwargs)
        return Monitor(env)

    return _init


def make_vec_env_for_stage(
    module: Any,
    env_class: type,
    args: argparse.Namespace,
    route_variants: list[str],
    stage: CurriculumStage,
    vecnorm_path: Path,
    eval_mode: bool = False,
) -> VecNormalize:
    apply_env_globals(module, args, stage)

    n_envs = 1 if eval_mode else max(1, int(args.num_envs))
    env_fns = [
        make_raw_env(env_class, module, args, route_variants, stage, rank=rank, eval_mode=eval_mode)
        for rank in range(n_envs)
    ]

    if n_envs == 1:
        raw_vec_env = DummyVecEnv(env_fns)
    else:
        raw_vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    if vecnorm_path.exists() and args.resume:
        vec_env = VecNormalize.load(str(vecnorm_path), raw_vec_env)
        vec_env.training = not eval_mode
        vec_env.norm_reward = not eval_mode
        return vec_env

    return VecNormalize(
        raw_vec_env,
        norm_obs=True,
        norm_reward=not eval_mode,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        gamma=args.gamma,
    )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

class SaveModelAndVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: Path, model_name: str, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_dir = save_dir
        self.model_name = model_name
        self.vecnorm_path = save_dir / f"{model_name}_vecnormalize.pkl"

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.save_dir / f"{self.model_name}_step_{self.num_timesteps}.zip"
        latest_path = self.save_dir / f"{self.model_name}_latest.zip"

        self.model.save(str(checkpoint_path))
        self.model.save(str(latest_path))

        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(str(self.vecnorm_path))

        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")
            print(f"Saved latest model: {latest_path}")
            print(f"Saved VecNormalize stats: {self.vecnorm_path}")

        return True


class MaskableEvalSaveBestCallback(BaseCallback):
    """Lightweight deterministic eval that supports action masks."""

    def __init__(
        self,
        eval_env: VecNormalize | None,
        eval_freq: int,
        n_eval_episodes: int,
        save_dir: Path,
        model_name: str,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.eval_env is None or self.n_calls % self.eval_freq != 0:
            return True

        try:
            sync_envs_normalization(self.training_env, self.eval_env)
        except Exception:
            pass

        rewards: list[float] = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            ep_reward = 0.0
            steps = 0
            while not bool(done[0]) and steps < 5000:
                try:
                    masks = get_action_masks(self.eval_env)
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
                except Exception:
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _info = self.eval_env.step(action)
                ep_reward += float(reward[0])
                steps += 1
            rewards.append(ep_reward)

        mean_reward = float(np.mean(rewards)) if rewards else -float("inf")
        if self.verbose:
            print(f"Eval after {self.num_timesteps} steps: mean_reward={mean_reward:.3f}, best={self.best_mean_reward:.3f}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.save_dir.mkdir(parents=True, exist_ok=True)
            best_path = self.save_dir / f"{self.model_name}_best.zip"
            self.model.save(str(best_path))
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(str(self.save_dir / f"{self.model_name}_vecnormalize.pkl"))
            if self.verbose:
                print(f"New best model saved: {best_path}")

        return True


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def make_model(env: VecNormalize, args: argparse.Namespace, model_path: Path) -> MaskablePPO:
    model_zip = with_zip_suffix(model_path)

    if args.resume and Path(model_zip).exists():
        print(f"Loading existing model and continuing training: {model_zip}")
        model = MaskablePPO.load(str(model_path), env=env, device=args.device)
        model.learning_rate = linear_schedule(args.lr_start, args.lr_end)
        model.clip_range = linear_schedule(args.clip_start, args.clip_end)
        return model

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
    )

    return MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=args.device,
        learning_rate=linear_schedule(args.lr_start, args.lr_end),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=linear_schedule(args.clip_start, args.clip_end),
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.tensorboard_dir),
        seed=args.seed,
    )


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, int(args.torch_threads)))

    base_dir = Path.cwd()
    module = importlib.import_module(args.env_module)
    if not hasattr(module, args.env_class):
        raise AttributeError(
            f"{args.env_module}.py does not define {args.env_class}. "
            "Use --env-class to point to your Gymnasium env class."
        )

    env_class = getattr(module, args.env_class)
    route_variants = discover_route_variants(module, base_dir)
    stages = build_curriculum(args)

    model_path = Path(args.model_path)
    save_dir = parent_dir(model_path)
    model_name = model_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    vecnorm_path = save_dir / f"{model_name}_vecnormalize.pkl"

    print("\nFast-proxy strong training configuration")
    print(f"  env module:       {args.env_module}")
    print(f"  env class:        {args.env_class}")
    print(f"  target TLS:       {args.tls_id}")
    print(f"  model path:       {model_path}.zip")
    print(f"  vecnormalize:     {vecnorm_path}")
    print(f"  ambulances:       disabled when fast env exposes ambulance flags")
    print(f"  max center:       {args.max_vehicles}")
    print(f"  target center:    {args.target_vehicles}")
    print(f"  initial center:   {args.initial_vehicles}")
    print(f"  spawn batch:      {args.spawn_batch}")
    print(f"  route lookahead:  {args.route_lookahead_edges}")
    print(f"  no-lane-change:   {args.intersection_no_lane_change_distance} m")
    print(f"  route variants:   {len(route_variants)}")
    for route in route_variants[:10]:
        print(f"    - {os.path.basename(route)}")
    if len(route_variants) > 10:
        print(f"    ... {len(route_variants) - 10} more")

    print("\nCurriculum")
    for stage in stages:
        print(
            f"  {stage.name}: {stage.timesteps} steps, "
            f"variants={stage.max_vehicle_variants}, target={stage.target_vehicles}, "
            f"initial={stage.initial_vehicles}, spawn={stage.spawn_batch}"
        )

    model: MaskablePPO | None = None
    total_done = 0

    for i, stage in enumerate(stages):
        print("\n" + "=" * 80)
        print(f"Starting stage {i + 1}/{len(stages)}: {stage.name}")
        print("=" * 80)

        env = make_vec_env_for_stage(
            module=module,
            env_class=env_class,
            args=args,
            route_variants=route_variants,
            stage=stage,
            vecnorm_path=vecnorm_path,
            eval_mode=False,
        )

        if not is_masking_supported(env):
            env.close()
            raise RuntimeError(
                "The environment does not expose action_masks(). MaskablePPO needs "
                "this so it does not choose impossible signal phases."
            )

        if model is None:
            model = make_model(env, args, model_path)
        else:
            model.set_env(env)

        callbacks: list[BaseCallback] = [
            SaveModelAndVecNormalizeCallback(
                save_freq=args.save_freq,
                save_dir=save_dir,
                model_name=model_name,
                verbose=1,
            )
        ]

        eval_env = None
        if args.eval_freq > 0:
            eval_stage = CurriculumStage(
                name="eval",
                timesteps=0,
                max_vehicle_variants=[min(args.max_vehicles, SIM_CENTER_MAX_VEHICLES)],
                target_vehicles=args.target_vehicles,
                initial_vehicles=args.initial_vehicles,
                spawn_batch=args.spawn_batch,
                route_lookahead_edges=args.route_lookahead_edges,
            )
            eval_env = make_vec_env_for_stage(
                module=module,
                env_class=env_class,
                args=args,
                route_variants=route_variants,
                stage=eval_stage,
                vecnorm_path=vecnorm_path,
                eval_mode=True,
            )
            eval_env.training = False
            eval_env.norm_reward = False
            callbacks.append(
                MaskableEvalSaveBestCallback(
                    eval_env=eval_env,
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.n_eval_episodes,
                    save_dir=save_dir,
                    model_name=model_name,
                    verbose=1,
                )
            )

        model.learn(
            total_timesteps=stage.timesteps,
            callback=CallbackList(callbacks),
            progress_bar=args.progress_bar,
            tb_log_name=model_name,
            reset_num_timesteps=(not args.resume and i == 0 and total_done == 0),
        )

        total_done += stage.timesteps
        model.save(str(model_path))
        env.save(str(vecnorm_path))
        env.close()
        if eval_env is not None:
            eval_env.close()

        print(f"Finished stage: {stage.name}")
        print(f"Saved model: {model_path}.zip")
        print(f"Saved normalization stats: {vecnorm_path}")

    print("\nTraining complete.")
    print(f"Final model: {model_path}.zip")
    print(f"Best model if eval enabled: {save_dir / (model_name + '_best.zip')}")
    print(f"VecNormalize stats: {vecnorm_path}")
    print("\nView TensorBoard with:")
    print(f"  tensorboard --logdir {args.tensorboard_dir}")


def eval_model(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, int(args.torch_threads)))

    base_dir = Path.cwd()
    module = importlib.import_module(args.env_module)
    env_class = getattr(module, args.env_class)
    route_variants = discover_route_variants(module, base_dir)

    model_path = Path(args.model_path)
    vecnorm_path = parent_dir(model_path) / f"{model_path.stem}_vecnormalize.pkl"

    stage = CurriculumStage(
        name="eval",
        timesteps=0,
        max_vehicle_variants=[min(args.max_vehicles, SIM_CENTER_MAX_VEHICLES)],
        target_vehicles=args.target_vehicles,
        initial_vehicles=args.initial_vehicles,
        spawn_batch=args.spawn_batch,
        route_lookahead_edges=args.route_lookahead_edges,
    )
    apply_env_globals(module, args, stage)

    raw_vec_env = DummyVecEnv([
        make_raw_env(env_class, module, args, route_variants, stage, rank=0, eval_mode=True)
    ])

    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), raw_vec_env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"Warning: no VecNormalize file found at {vecnorm_path}; eval uses raw observations.")
        env = VecNormalize(raw_vec_env, norm_obs=False, norm_reward=False)
        env.training = False

    model = MaskablePPO.load(str(model_path), env=env, device=args.device)
    obs = env.reset()
    episode_reward = 0.0

    print("\nStarting deterministic fast-proxy evaluation. Press Ctrl+C to stop.")
    try:
        for step in range(args.eval_steps):
            try:
                masks = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            except Exception:
                action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_reward += float(rewards[0])

            if step % args.eval_print_every == 0:
                info = infos[0] if infos else {}
                print(
                    f"step={step:6d} "
                    f"reward={float(rewards[0]):8.3f} "
                    f"episode_reward={episode_reward:10.3f} "
                    f"action={int(action[0]) if np.ndim(action) else int(action)} "
                    f"sim_time={info.get('sim_time', 'n/a')} "
                    f"phase={info.get('phase_name', 'n/a')}"
                )

            if bool(dones[0]):
                print(f"Episode ended. episode_reward={episode_reward:.3f}")
                episode_reward = 0.0
                obs = env.reset()
    finally:
        env.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast strong MaskablePPO trainer using the traffic_rl_model.py proxy environment."
    )

    parser.add_argument("--env-module", default=DEFAULT_ENV_MODULE)
    parser.add_argument("--env-class", default="TrafficSignalEnv")
    parser.add_argument("--tls-id", default=DEFAULT_TLS_ID)

    parser.add_argument("--model-path", default=DEFAULT_MODEL_BASENAME)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--episode-seconds", type=int, default=900)

    parser.add_argument("--max-vehicles", type=int, default=SIM_CENTER_MAX_VEHICLES)
    parser.add_argument("--target-vehicles", type=int, default=SIM_CENTER_TARGET_VEHICLES)
    parser.add_argument("--initial-vehicles", type=int, default=SIM_CENTER_INITIAL_VEHICLES)
    parser.add_argument("--spawn-batch", type=int, default=SIM_CENTER_SPAWN_BATCH)
    parser.add_argument("--route-lookahead-edges", type=int, default=SIM_CENTER_ROUTE_LOOKAHEAD)
    parser.add_argument("--green-duration", type=float, default=SIM_CENTER_GREEN_DURATION)
    parser.add_argument("--intersection-no-lane-change-distance", type=float, default=SIM_CENTER_NO_LANE_CHANGE_DISTANCE)
    parser.add_argument("--intersection-lane-prep-distance", type=float, default=SIM_CENTER_LANE_PREP_DISTANCE)
    parser.add_argument("--vehicle-variants", default=None)

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-scenarios", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--warmup-fraction", type=float, default=0.18)
    parser.add_argument("--main-fraction", type=float, default=0.52)
    parser.add_argument("--stress-fraction", type=float, default=0.20)

    parser.add_argument("--lr-start", type=float, default=2.5e-4)
    parser.add_argument("--lr-end", type=float, default=3e-5)
    parser.add_argument("--clip-start", type=float, default=0.20)
    parser.add_argument("--clip-end", type=float, default=0.08)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.010)
    parser.add_argument("--vf-coef", type=float, default=0.70)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=0, help="0 disables periodic eval. Use 50_000 after the speed is acceptable.")
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs"))

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Use GUI during eval if TrafficSignalEnv supports gui=True.")
    parser.add_argument("--eval-fixed-scenario", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=20_000)
    parser.add_argument("--eval-print-every", type=int, default=25)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.warmup_fraction < 0 or args.main_fraction < 0 or args.stress_fraction < 0:
        raise ValueError("Curriculum fractions must be nonnegative.")
    if args.warmup_fraction + args.main_fraction + args.stress_fraction > 0.98:
        raise ValueError("warmup_fraction + main_fraction + stress_fraction is too large.")

    args.max_vehicles = min(int(args.max_vehicles), SIM_CENTER_MAX_VEHICLES)
    args.target_vehicles = min(int(args.target_vehicles), args.max_vehicles)
    args.initial_vehicles = min(int(args.initial_vehicles), args.target_vehicles)
    args.spawn_batch = max(1, int(args.spawn_batch))
    args.route_lookahead_edges = max(2, int(args.route_lookahead_edges))
    args.torch_threads = max(1, int(args.torch_threads))

    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval_only:
        eval_model(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
