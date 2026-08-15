#!/usr/bin/env python3
"""Lightweight interfaces for detector-realistic schema-v4 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import traffic_rl_map_agnostic_env as map_env
from detector_realistic_tls import (
    MAX_PHASES,
    DetectorRealisticTLSAdapter,
    empty_observation,
    observation_space,
    sensor_config_from_environment,
)


sim = map_env.sim


class DetectorRealisticPolicyShapeEnv(gym.Env):
    """No-SUMO environment used to construct/load MaskablePPO checkpoints."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(MAX_PHASES + 1)
        self.observation_space = observation_space()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        observation = empty_observation()
        observation["movement_mask"][0] = 1.0
        observation["phase_membership"][0, 0] = 1.0
        return observation, {}

    def step(self, action: int):
        observation, _ = self.reset()
        return observation, 0.0, False, False, {}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[:2] = True
        return mask


def discover_usable_tls(net_file: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = map_env.configure_network(net_file)
    command = [
        sim.SUMO_HEADLESS_BINARY,
        "-n",
        str(path),
        "--start",
        "--step-length",
        str(sim.STEP_LENGTH),
        "--end",
        "2",
        *getattr(sim, "QUIET_SUMO_ARGS", []),
    ]
    sim.traci.start(command)
    records: list[dict[str, Any]] = []
    rng = random.Random(0)
    sensor_config = sensor_config_from_environment(False)
    try:
        for tls_id in sim.traci.trafficlight.getIDList():
            try:
                controller = sim.build_map_agnostic_controller_for_tls(
                    tls_id, rng=rng, activate=False
                )
                if controller is None:
                    continue
                adapter = DetectorRealisticTLSAdapter(
                    controller,
                    sim.traci,
                    sim,
                    sensor_config=sensor_config,
                )
                records.append(adapter.validate_controller())
            except Exception as exc:
                print(f"Skipping TLS {tls_id}: {exc}")
    finally:
        try:
            sim.traci.close(False)
        except Exception:
            pass
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect detector-realistic TLS compatibility."
    )
    parser.add_argument(
        "--net-file", default=os.environ.get("TRAFFIC_NET_FILE", sim.NET_FILE)
    )
    parser.add_argument("--list-tls-json", action="store_true")
    args = parser.parse_args()
    records = discover_usable_tls(Path(args.net_file))
    if args.list_tls_json:
        print("DETECTOR_REALISTIC_TLS_JSON=" + json.dumps(records, separators=(",", ":")))
    else:
        print(json.dumps(records, indent=2))
        print(f"Usable TLS: {len(records)}")


if __name__ == "__main__":
    main()
