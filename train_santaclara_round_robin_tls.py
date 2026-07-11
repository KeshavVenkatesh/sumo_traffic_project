#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import random
import subprocess
import sys
from pathlib import Path


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def trainer_help(trainer: Path) -> str:
    try:
        out = subprocess.check_output(
            [sys.executable, str(trainer), "--help"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out
    except Exception:
        return ""


def add_if_supported(cmd: list[str], help_text: str, flag: str, value=None) -> None:
    if flag not in help_text:
        return
    cmd.append(flag)
    if value is not None:
        cmd.append(str(value))


def discover_usable_tls(env_module_name: str, min_phases: int, cache_file: Path, refresh: bool) -> list[str]:
    if cache_file.exists() and not refresh:
        cached = [line.strip() for line in cache_file.read_text().splitlines() if line.strip()]
        if cached:
            print(f"Loaded {len(cached)} usable TLS ids from {cache_file}")
            return cached

    env_mod = importlib.import_module(env_module_name)
    traci = env_mod.traci

    cmd = [
        env_mod.SUMO_HEADLESS_BINARY,
        "-n",
        env_mod.NET_FILE,
        "--start",
        "--step-length",
        str(env_mod.STEP_LENGTH),
        "--end",
        "5",
        *getattr(env_mod, "QUIET_SUMO_ARGS", []),
    ]

    print("Discovering usable TLS ids...")
    print(" ".join(cmd))

    traci.start(cmd)

    usable = []
    try:
        tls_ids = list(traci.trafficlight.getIDList())
        print(f"Total SUMO TLS ids: {len(tls_ids)}")

        for tls_id in tls_ids:
            try:
                controller = env_mod.build_controller_for_tls(tls_id, activate=False)
            except Exception:
                controller = None

            if controller is None:
                continue

            phases = controller.get("phases", [])
            if len(phases) < min_phases:
                continue

            usable.append(tls_id)
    finally:
        try:
            traci.close(False)
        except Exception:
            pass

    cache_file.write_text("\n".join(usable) + "\n")
    print(f"Usable TLS ids: {len(usable)}")
    print(f"Wrote cache: {cache_file}")

    return usable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin shared-policy trainer over usable Santa Clara TLS ids."
    )

    parser.add_argument("--trainer", default="train_santaclara_proxy.py")
    parser.add_argument("--env-module", default="traffic_rl_model_santaclara_proxy")
    parser.add_argument("--model-path", default="models/traffic_signal_maskable_ppo_santaclara_rr")

    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--steps-per-tls", type=int, default=1500)
    parser.add_argument("--max-tls", type=int, default=20)
    parser.add_argument("--tls-ids", default="")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min-phases", type=int, default=2)
    parser.add_argument("--tls-cache-file", type=Path, default=Path(".usable_santaclara_tls.txt"))
    parser.add_argument("--refresh-tls-cache", action="store_true")

    parser.add_argument("--episode-seconds", type=int, default=900)
    parser.add_argument("--max-vehicles", type=int, default=2000)
    parser.add_argument("--target-vehicles", type=int, default=1500)
    parser.add_argument("--initial-vehicles", type=int, default=300)
    parser.add_argument("--spawn-batch", type=int, default=20)
    parser.add_argument("--vehicle-variants", default="1000,1200,1500,1800,2000")

    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)

    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--lr-start", type=float, default=2.5e-4)
    parser.add_argument("--lr-end", type=float, default=3e-5)
    parser.add_argument("--norm-obs", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    trainer = Path(args.trainer)
    if not trainer.exists():
        raise FileNotFoundError(trainer)

    help_text = trainer_help(trainer)

    explicit_tls = parse_csv(args.tls_ids)
    if explicit_tls:
        tls_ids = explicit_tls
    else:
        tls_ids = discover_usable_tls(
            env_module_name=args.env_module,
            min_phases=args.min_phases,
            cache_file=args.tls_cache_file,
            refresh=args.refresh_tls_cache,
        )

    if args.max_tls > 0:
        tls_ids = tls_ids[: args.max_tls]

    if not tls_ids:
        raise RuntimeError("No usable TLS ids found.")

    print()
    print("=" * 100)
    print("Round-robin TLS training plan")
    print("=" * 100)
    print(f"Model path: {args.model_path}")
    print(f"Rounds: {args.rounds}")
    print(f"TLS count: {len(tls_ids)}")
    print(f"Steps per TLS: {args.steps_per_tls}")
    print(f"Total requested steps: {args.rounds * len(tls_ids) * args.steps_per_tls}")
    print("=" * 100)

    for round_idx in range(args.rounds):
        order = list(tls_ids)
        if args.shuffle:
            random.Random(args.seed + round_idx).shuffle(order)

        print()
        print("#" * 100)
        print(f"ROUND {round_idx + 1}/{args.rounds}")
        print("#" * 100)

        for i, tls_id in enumerate(order, start=1):
            print()
            print("=" * 100)
            print(f"Training TLS {i}/{len(order)} in round {round_idx + 1}: {tls_id}")
            print("=" * 100)

            cmd = [
                sys.executable,
                "-u",
                str(trainer),
            ]

            add_if_supported(cmd, help_text, "--env-module", args.env_module)
            add_if_supported(cmd, help_text, "--tls-id", tls_id)
            add_if_supported(cmd, help_text, "--model-path", args.model_path)
            add_if_supported(cmd, help_text, "--timesteps", args.steps_per_tls)
            add_if_supported(cmd, help_text, "--episode-seconds", args.episode_seconds)

            add_if_supported(cmd, help_text, "--max-vehicles", args.max_vehicles)
            add_if_supported(cmd, help_text, "--target-vehicles", args.target_vehicles)
            add_if_supported(cmd, help_text, "--initial-vehicles", args.initial_vehicles)
            add_if_supported(cmd, help_text, "--spawn-batch", args.spawn_batch)
            add_if_supported(cmd, help_text, "--vehicle-variants", args.vehicle_variants)

            add_if_supported(cmd, help_text, "--save-freq", args.save_freq)
            add_if_supported(cmd, help_text, "--device", args.device)
            add_if_supported(cmd, help_text, "--torch-threads", args.torch_threads)

            add_if_supported(cmd, help_text, "--n-steps", args.n_steps)
            add_if_supported(cmd, help_text, "--batch-size", args.batch_size)
            add_if_supported(cmd, help_text, "--n-epochs", args.n_epochs)
            add_if_supported(cmd, help_text, "--lr-start", args.lr_start)
            add_if_supported(cmd, help_text, "--lr-end", args.lr_end)
            if args.norm_obs:
                add_if_supported(cmd, help_text, "--norm-obs")
            else:
                add_if_supported(cmd, help_text, "--no-norm-obs")

            add_if_supported(cmd, help_text, "--no-curriculum")
            add_if_supported(cmd, help_text, "--no-progress-bar")
            add_if_supported(cmd, help_text, "--resume")

            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    print()
    print("Round-robin all-TLS training complete.")
    print(f"Final model: {args.model_path}.zip")


if __name__ == "__main__":
    main()
