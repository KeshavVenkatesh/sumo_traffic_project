#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

import compare_fixed_vs_single_vs_all_model_realistic as cmp


def with_zip(path: str) -> str:
    return path if path.endswith(".zip") else path + ".zip"


class SaveExactModelCallback(BaseCallback):
    def __init__(self, model_path: str, vecnormalize_path: str, save_freq: int):
        super().__init__(verbose=1)
        self.model_path = model_path
        self.vecnormalize_path = vecnormalize_path
        self.save_freq = max(1, int(save_freq))

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        self.model.save(self.model_path)
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(self.vecnormalize_path)

        return True



class ProgressEverySecondsCallback(BaseCallback):
    def __init__(self, total_timesteps: int, print_every_seconds: float):
        super().__init__(verbose=0)
        self.total_timesteps = max(1, int(total_timesteps))
        self.print_every_seconds = max(1.0, float(print_every_seconds))
        self.start_time = None
        self.last_print_time = None

    def _on_training_start(self) -> None:
        now = time.monotonic()
        self.start_time = now
        self.last_print_time = now

    def _on_step(self) -> bool:
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
        if self.last_print_time is None:
            self.last_print_time = now

        if now - self.last_print_time < self.print_every_seconds:
            return True

        elapsed = max(1e-9, now - self.start_time)
        done = min(self.n_calls, self.total_timesteps)
        pct = 100.0 * done / self.total_timesteps
        fps = done / elapsed
        remaining = max(0, self.total_timesteps - done)
        eta_seconds = remaining / fps if fps > 0 else 0.0

        print(
            f"[progress] {done}/{self.total_timesteps} steps "
            f"({pct:.2f}%) | elapsed={elapsed/60:.1f} min "
            f"| fps={fps:.2f} | eta={eta_seconds/60:.1f} min "
            f"| model_timesteps={self.num_timesteps}",
            flush=True,
        )

        self.last_print_time = now
        return True


def make_env(args: argparse.Namespace):
    def _init():
        env = cmp.ExactSimulationTrafficSignalEnv(
            tls_id=args.tls_id,
            episode_seconds=args.episode_seconds,
            gui=False,
            randomize_scenarios=True,
            base_seed=args.seed,
            env_rank=0,
            print_scenarios=args.print_scenarios,
            max_vehicle_center=args.max_vehicle_center,
            target_center=args.target_vehicle_center,
            initial_center=args.initial_vehicle_center,
            spawn_batch_center=args.spawn_batch_center,
            green_duration_center=args.green_duration_center,
            density_spread=args.density_spread,
            initial_spread=args.initial_spread,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tls-id", required=True)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--episode-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)

    parser.add_argument("--init-model", default="models/traffic_signal_maskable_ppo_fast_proxy_strong")
    parser.add_argument("--init-vecnormalize", default="models/traffic_signal_maskable_ppo_fast_proxy_strong_vecnormalize.pkl")
    parser.add_argument("--model-path", default="models/traffic_signal_maskable_ppo_exact_santaclara")
    parser.add_argument("--vecnormalize-path", default="models/traffic_signal_maskable_ppo_exact_santaclara_vecnormalize.pkl")

    parser.add_argument("--max-vehicle-center", type=int, default=1500)
    parser.add_argument("--target-vehicle-center", type=int, default=1500)
    parser.add_argument("--initial-vehicle-center", type=int, default=300)
    parser.add_argument("--spawn-batch-center", type=int, default=20)
    parser.add_argument("--green-duration-center", type=float, default=30.0)
    parser.add_argument("--density-spread", type=float, default=0.18)
    parser.add_argument("--initial-spread", type=float, default=0.40)

    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.04)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--print-scenarios", action="store_true")
    parser.add_argument("--print-every-seconds", type=float, default=5.0)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    raw_env = make_env(args)

    init_vec = Path(args.init_vecnormalize)
    if init_vec.exists() and not args.fresh:
        env = VecNormalize.load(str(init_vec), raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            raw_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
        )

    init_model_zip = Path(with_zip(args.init_model))
    final_model_zip = Path(with_zip(args.model_path))

    if init_model_zip.exists() and not args.fresh:
        model = MaskablePPO.load(str(args.init_model), env=env, device=args.device)
        model.learning_rate = args.learning_rate
    elif final_model_zip.exists() and not args.fresh:
        model = MaskablePPO.load(str(args.model_path), env=env, device=args.device)
        model.learning_rate = args.learning_rate
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            device=args.device,
            learning_rate=args.learning_rate,
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
            tensorboard_log="runs/exact_santaclara",
            seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                activation_fn=torch.nn.Tanh,
                ortho_init=True,
            ),
        )


    model.verbose = 0

    callback = CallbackList([
        ProgressEverySecondsCallback(
            total_timesteps=args.timesteps,
            print_every_seconds=args.print_every_seconds,
        ),
        SaveExactModelCallback(
            model_path=args.model_path,
            vecnormalize_path=args.vecnormalize_path,
            save_freq=args.save_freq,
        ),
    ])

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=args.progress_bar,
        reset_num_timesteps=False,
    )

    model.save(args.model_path)
    env.save(args.vecnormalize_path)
    env.close()


if __name__ == "__main__":
    main()
