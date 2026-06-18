#!/usr/bin/env python3
"""
Improved training script for the SUMO traffic-signal MaskablePPO model.

This script intentionally keeps the SUMO simulation/environment logic in your
existing module, usually traffic_rl_model.py. It improves the training loop by
adding:

1. observation/reward normalization with VecNormalize
2. curriculum training over dense traffic levels
3. safer MaskablePPO hyperparameters for noisy SUMO rewards
4. checkpointing that saves both model weights and VecNormalize stats
5. deterministic eval-only mode with action masks
6. resume support

Expected project layout:
    sumo_traffic_project/
        traffic_rl_model.py
        train_traffic_signal_rl.py   <- replace with this file if desired
        new_map.net.xml
        background_new.rou.xml
        ambulance_random_new.rou.xml
        background_train_*.rou.xml   <- optional route variants

Typical command:
    python3 train_traffic_signal_rl.py \
      --timesteps 300000 \
      --episode-seconds 900 \
      --max-vehicles 1000 \
      --target-vehicles 700 \
      --initial-vehicles 500 \
      --spawn-batch 40 \
      --route-lookahead-edges 25

Note: target-vehicles, initial-vehicles, spawn-batch, and route-lookahead-edges
are accepted for command compatibility with your other scripts. If the imported
environment module exposes matching globals or constructor parameters, this
script applies them. Otherwise, they are harmless no-ops here.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
except ImportError as exc:
    raise ImportError(
        "Missing RL dependencies. Install them with:\n"
        "python3 -m pip install gymnasium stable-baselines3 sb3-contrib tensorboard tqdm rich"
    ) from exc


DEFAULT_TLS_ID = "cluster_12179861947_12179861948_12179861949_12185616643_#11more"
DEFAULT_MODEL_BASENAME = "models/traffic_signal_maskable_ppo_better"
DEFAULT_ENV_MODULE = "traffic_rl_model"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    """SB3 schedule: progress_remaining goes from 1.0 to 0.0."""

    def schedule(progress_remaining: float) -> float:
        progress_done = 1.0 - progress_remaining
        return start + progress_done * (end - start)

    return schedule


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return default

    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))

    return values or default


def with_zip_suffix(path: str | Path) -> str:
    path = str(path)
    return path if path.endswith(".zip") else path + ".zip"


def parent_dir(path: str | Path) -> Path:
    return Path(path).expanduser().resolve().parent


def set_module_attr_if_present(module: Any, name: str, value: Any) -> None:
    if hasattr(module, name):
        setattr(module, name, value)


def constructor_accepts(cls: type, name: str) -> bool:
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False

    return name in signature.parameters


def discover_route_variants(module: Any, base_dir: Path) -> list[str]:
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
# Curriculum
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumStage:
    name: str
    timesteps: int
    max_vehicle_variants: list[int]


def build_curriculum(args: argparse.Namespace) -> list[CurriculumStage]:
    """
    Keep traffic intense, but do not throw the agent directly into only 1000-car
    stress episodes from the first update. This usually learns better because the
    policy sees congestion before total gridlock.
    """

    if args.no_curriculum:
        variants = parse_int_list(args.vehicle_variants, [args.target_vehicles, args.max_vehicles])
        return [CurriculumStage("single_stage", args.timesteps, variants)]

    warmup_steps = max(0, int(args.timesteps * args.warmup_fraction))
    stress_steps = max(0, int(args.timesteps * args.stress_fraction))
    main_steps = max(0, args.timesteps - warmup_steps - stress_steps)

    # These are still dense. The warmup is not the old weak 150-car debug setup.
    warmup_hi = max(500, min(args.target_vehicles, 700))
    main_hi = max(warmup_hi, min(args.max_vehicles, max(args.target_vehicles, 850)))

    stages = []

    if warmup_steps > 0:
        stages.append(
            CurriculumStage(
                name="dense_warmup",
                timesteps=warmup_steps,
                max_vehicle_variants=sorted(set([500, 600, warmup_hi])),
            )
        )

    if main_steps > 0:
        stages.append(
            CurriculumStage(
                name="main_congestion",
                timesteps=main_steps,
                max_vehicle_variants=sorted(set([700, args.target_vehicles, main_hi])),
            )
        )

    if stress_steps > 0:
        stages.append(
            CurriculumStage(
                name="stress_adaptation",
                timesteps=stress_steps,
                max_vehicle_variants=sorted(set([args.target_vehicles, 900, args.max_vehicles])),
            )
        )

    return [stage for stage in stages if stage.timesteps > 0]


# -----------------------------------------------------------------------------
# Env creation
# -----------------------------------------------------------------------------

def apply_env_globals(module: Any, args: argparse.Namespace, stage: CurriculumStage) -> None:
    """
    Patch common globals used by your existing traffic_rl_model.py environment.
    This keeps the rewritten training loop compatible with your current file
    without requiring us to rewrite all SUMO simulation logic here.
    """

    set_module_attr_if_present(module, "TRAIN_EPISODE_SECONDS", args.episode_seconds)
    set_module_attr_if_present(module, "SIM_END", max(args.episode_seconds, getattr(module, "SIM_END", 0)))
    set_module_attr_if_present(module, "MAX_NUM_VEHICLES", args.max_vehicles)
    set_module_attr_if_present(module, "MAX_VEHICLE_VARIANTS", stage.max_vehicle_variants)

    # These names exist in your realistic simulation script. They may not exist
    # in traffic_rl_model.py; setting only when present keeps this safe.
    set_module_attr_if_present(module, "TARGET_VEHICLES", args.target_vehicles)
    set_module_attr_if_present(module, "INITIAL_VEHICLES", args.initial_vehicles)
    set_module_attr_if_present(module, "SPAWN_BATCH", args.spawn_batch)
    set_module_attr_if_present(module, "ROUTE_LOOKAHEAD_EDGES", args.route_lookahead_edges)

    # Training should stay headless and quiet.
    set_module_attr_if_present(module, "TRAIN_WITH_SUMO_LOGS", False)
    set_module_attr_if_present(module, "PRINT_TRAINING_SCENARIOS", args.print_scenarios)


def make_raw_env(
    env_class: type,
    module: Any,
    args: argparse.Namespace,
    route_variants: list[str],
    stage: CurriculumStage,
    rank: int = 0,
):
    def _init():
        seed = args.seed + rank * 1009
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        kwargs: dict[str, Any] = {}

        if constructor_accepts(env_class, "tls_id"):
            kwargs["tls_id"] = args.tls_id
        if constructor_accepts(env_class, "gui"):
            # Use SUMO GUI only when requested. Training should normally stay headless,
            # but eval-only runs can now open XQuartz / sumo-gui.
            kwargs["gui"] = bool(getattr(args, "gui", False))
        if constructor_accepts(env_class, "randomize_traffic"):
            kwargs["randomize_traffic"] = True
        if constructor_accepts(env_class, "route_variants"):
            kwargs["route_variants"] = route_variants
        if constructor_accepts(env_class, "max_vehicle_variants"):
            kwargs["max_vehicle_variants"] = stage.max_vehicle_variants

        env = env_class(**kwargs)
        env = Monitor(env)
        return env

    return _init


def make_vec_env_for_stage(
    module: Any,
    env_class: type,
    args: argparse.Namespace,
    route_variants: list[str],
    stage: CurriculumStage,
    vecnorm_path: Path,
) -> VecNormalize:
    apply_env_globals(module, args, stage)

    # SUMO/TraCI is the bottleneck. On a Mac, one heavy SUMO world is usually
    # faster and more stable than two huge worlds fighting for CPU.
    if args.num_envs != 1:
        print(
            "Warning: forcing --num-envs to 1. Your SUMO/TraCI environment is "
            "not a cheap GPU batch; multiple heavy SUMO processes can train slower."
        )

    raw_vec_env = DummyVecEnv([
        make_raw_env(env_class, module, args, route_variants, stage, rank=0)
    ])

    if vecnorm_path.exists() and args.resume:
        vec_env = VecNormalize.load(str(vecnorm_path), raw_vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        return vec_env

    return VecNormalize(
        raw_vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        gamma=args.gamma,
    )


# -----------------------------------------------------------------------------
# Checkpoint callback
# -----------------------------------------------------------------------------

class SaveModelAndVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: Path, model_name: str, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_freq = max(1, save_freq)
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

        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(str(self.vecnorm_path))

        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")
            print(f"Saved latest model: {latest_path}")
            print(f"Saved VecNormalize stats: {self.vecnorm_path}")

        return True


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def make_model(env: VecNormalize, args: argparse.Namespace, model_path: Path) -> MaskablePPO:
    model_zip = with_zip_suffix(model_path)

    if args.resume and Path(model_zip).exists():
        print(f"Loading existing model and continuing training: {model_zip}")
        return MaskablePPO.load(str(model_path), env=env, device=args.device)

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

    print("\nTraining configuration")
    print(f"  env module:       {args.env_module}")
    print(f"  env class:        {args.env_class}")
    print(f"  tls id:           {args.tls_id}")
    print(f"  model path:       {model_path}")
    print(f"  vecnormalize:     {vecnorm_path}")
    print(f"  route variants:   {len(route_variants)}")
    for route in route_variants[:12]:
        print(f"    - {os.path.basename(route)}")
    if len(route_variants) > 12:
        print(f"    ... {len(route_variants) - 12} more")

    print("\nCurriculum")
    for stage in stages:
        print(f"  {stage.name}: {stage.timesteps} steps, vehicles={stage.max_vehicle_variants}")

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

        checkpoint_callback = SaveModelAndVecNormalizeCallback(
            save_freq=args.save_freq,
            save_dir=save_dir,
            model_name=model_name,
            verbose=1,
        )

        callbacks = CallbackList([checkpoint_callback])

        model.learn(
            total_timesteps=stage.timesteps,
            callback=callbacks,
            progress_bar=args.progress_bar,
            tb_log_name=model_name,
            reset_num_timesteps=(not args.resume and i == 0 and total_done == 0),
        )

        total_done += stage.timesteps

        model.save(str(model_path))
        env.save(str(vecnorm_path))
        env.close()

        print(f"Finished stage: {stage.name}")
        print(f"Saved model: {model_path}")
        print(f"Saved normalization stats: {vecnorm_path}")

    print("\nTraining complete.")
    print(f"Final model: {model_path}")
    print(f"VecNormalize stats: {vecnorm_path}")
    print("\nView TensorBoard with:")
    print(f"  tensorboard --logdir {args.tensorboard_dir}")


def eval_model(args: argparse.Namespace) -> None:
    base_dir = Path.cwd()
    module = importlib.import_module(args.env_module)
    env_class = getattr(module, args.env_class)

    model_path = Path(args.model_path)
    vecnorm_path = parent_dir(model_path) / f"{model_path.stem}_vecnormalize.pkl"

    stage = CurriculumStage(
        name="eval",
        timesteps=0,
        max_vehicle_variants=[args.max_vehicles],
    )

    # For eval we usually want the same high-density setup every episode.
    route_variants = discover_route_variants(module, base_dir)
    apply_env_globals(module, args, stage)

    raw_vec_env = DummyVecEnv([
        make_raw_env(env_class, module, args, route_variants, stage, rank=0)
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

    print("\nStarting deterministic evaluation. Press Ctrl+C to stop.")

    try:
        for step in range(args.eval_steps):
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
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
        description="Train a better MaskablePPO traffic-signal model for SUMO."
    )

    parser.add_argument("--env-module", default=DEFAULT_ENV_MODULE)
    parser.add_argument("--env-class", default="TrafficSignalEnv")
    parser.add_argument("--tls-id", default=DEFAULT_TLS_ID)

    parser.add_argument("--model-path", default=DEFAULT_MODEL_BASENAME)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--episode-seconds", type=int, default=900)

    parser.add_argument("--max-vehicles", type=int, default=1000)
    parser.add_argument("--target-vehicles", type=int, default=700)
    parser.add_argument("--initial-vehicles", type=int, default=500)
    parser.add_argument("--spawn-batch", type=int, default=40)
    parser.add_argument("--route-lookahead-edges", type=int, default=25)
    parser.add_argument("--vehicle-variants", default=None)

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-scenarios", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--warmup-fraction", type=float, default=0.20)
    parser.add_argument("--stress-fraction", type=float, default=0.30)

    parser.add_argument("--lr-start", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=5e-5)
    parser.add_argument("--clip-start", type=float, default=0.20)
    parser.add_argument("--clip-end", type=float, default=0.10)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.015)
    parser.add_argument("--vf-coef", type=float, default=0.70)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.03)

    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs"))

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Open SUMO GUI/XQuartz during evaluation or debugging.")
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--eval-print-every", type=int, default=25)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.warmup_fraction < 0 or args.stress_fraction < 0:
        raise ValueError("Curriculum fractions must be nonnegative.")
    if args.warmup_fraction + args.stress_fraction > 0.95:
        raise ValueError("warmup_fraction + stress_fraction is too large.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval_only:
        eval_model(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
