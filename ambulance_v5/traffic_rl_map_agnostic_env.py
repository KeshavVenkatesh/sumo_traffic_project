#!/usr/bin/env python3
"""Exact-SUMO Gym environment for the movement-centric shared policy."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import compare_fixed_vs_single_vs_all_model_realistic as legacy
from map_agnostic_tls import (
    MAX_PHASES,
    MapAgnosticTLSAdapter,
    adapter_for_controller,
    empty_observation,
    normalized_reward,
    observation_space,
)

sim = legacy.sim
sim.USE_MAP_AGNOSTIC_PHASE_CATALOG = True

# The legacy comparison module captured a 750-vehicle GUI default at import and
# silently clamped larger CLI requests to it.  Schema v3 uses an explicit,
# configurable compute cap so density randomization is not accidentally erased.
MAP_AGNOSTIC_MAX_ACTIVE_CAP = int(os.environ.get("MAP_AGNOSTIC_MAX_ACTIVE_CAP", "2000"))
legacy.SIM_CENTER_MAX_VEHICLES = MAP_AGNOSTIC_MAX_ACTIVE_CAP
sim.MAX_ACTIVE_VEHICLE_CAP = MAP_AGNOSTIC_MAX_ACTIVE_CAP


def configure_network(net_file: str | os.PathLike[str]) -> Path:
    path = Path(net_file).expanduser()
    if not path.is_absolute():
        path = Path(sim.BASE_DIR) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"SUMO network does not exist: {path}")
    os.environ["TRAFFIC_NET_FILE"] = str(path)
    sim.NET_FILE = str(path)
    return path


class MapAgnosticPolicyShapeEnv(gym.Env):
    """No-SUMO environment used when loading a saved MaskablePPO model."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(MAX_PHASES + 1)
        self.observation_space = observation_space()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return empty_observation(), {}

    def step(self, action: int):
        return empty_observation(), 0.0, False, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(MAX_PHASES + 1, dtype=bool)


class MapAgnosticExactTrafficSignalEnv(legacy.ExactSimulationTrafficSignalEnv):
    """The current realistic simulator with invariant state/action/reward.

    Other intersections retain the generated safe fixed-cycle controller while
    the selected TLS is controlled by the shared policy.  The multi-map trainer
    rotates the selected TLS and map between training calls.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.observation_noise_std = max(
            0.0, float(kwargs.pop("observation_noise_std", 0.0))
        )
        self.sensor_scale_jitter = max(
            0.0, float(kwargs.pop("sensor_scale_jitter", 0.0))
        )
        self.sensor_dropout_prob = min(
            0.25, max(0.0, float(kwargs.pop("sensor_dropout_prob", 0.0)))
        )
        self._sensor_columns = 15
        self._movement_sensor_scale = np.ones(
            self._sensor_columns, dtype=np.float32
        )
        self._movement_sensor_bias = np.zeros(
            self._sensor_columns, dtype=np.float32
        )
        self._movement_sensor_dropout = np.zeros(
            self._sensor_columns, dtype=bool
        )
        super().__init__(*args, **kwargs)
        # A tiny map should not be forced to the legacy 250-target/350-cap
        # floors. The multi-map orchestrator already scales demand by passenger
        # lane-kilometers, so retain only a small numerical floor.
        self.sampler.min_max_vehicles = min(50, self.sampler.max_vehicle_center)
        self.sampler.min_target_vehicles = min(40, self.sampler.target_center)
        self.sampler.min_initial_vehicles = min(10, self.sampler.initial_center)
        self.action_space = spaces.Discrete(MAX_PHASES + 1)
        self.observation_space = observation_space()
        self.adapter: MapAgnosticTLSAdapter | None = None
        self._snapshot = None

    def _training_observation(self, observation: dict[str, np.ndarray]):
        if (
            self.observation_noise_std <= 0.0
            and self.sensor_scale_jitter <= 0.0
            and self.sensor_dropout_prob <= 0.0
        ):
            return observation
        augmented = {key: value.copy() for key, value in observation.items()}
        valid_movements = augmented["movement_mask"][:, None]
        augmented["movements"][:, : self._sensor_columns] = np.clip(
            (
                augmented["movements"][:, : self._sensor_columns]
                * self._movement_sensor_scale[None, :]
                + self._movement_sensor_bias[None, :]
            )
            * valid_movements,
            -1.0,
            1.0,
        )
        if self._movement_sensor_dropout.any():
            augmented["movements"][
                :, np.flatnonzero(self._movement_sensor_dropout)
            ] = 0.0
        movement_noise = self.np_random.normal(
            0.0,
            self.observation_noise_std,
            size=augmented["movements"][:, : self._sensor_columns].shape,
        ).astype(np.float32)
        movement_noise *= augmented["movement_mask"][:, None]
        augmented["movements"][:, : self._sensor_columns] = np.clip(
            augmented["movements"][:, : self._sensor_columns]
            + movement_noise,
            -1.0,
            1.0,
        )

        phase_valid = (augmented["phase_membership"].sum(axis=1) > 0.0).astype(np.float32)
        for column in (2, 3, 4, 7):
            noise = self.np_random.normal(
                0.0, self.observation_noise_std, size=phase_valid.shape
            ).astype(np.float32)
            augmented["phase_features"][:, column] = np.clip(
                augmented["phase_features"][:, column] + noise * phase_valid,
                -1.0,
                1.0,
            )

        global_noise = self.np_random.normal(
            0.0, self.observation_noise_std, size=(4,)
        ).astype(np.float32)
        augmented["global_features"][2:6] = np.clip(
            augmented["global_features"][2:6] + global_noise, -1.0, 1.0
        )
        # Graph structure, padding masks, and phase membership are exact map
        # semantics rather than noisy sensors and must never be perturbed.
        return augmented

    def _sample_sensor_domain(self) -> None:
        """Hold one synthetic detector calibration constant for an episode."""
        if self.sensor_scale_jitter > 0.0:
            self._movement_sensor_scale = self.np_random.lognormal(
                mean=0.0,
                sigma=self.sensor_scale_jitter,
                size=(self._sensor_columns,),
            ).astype(np.float32)
            self._movement_sensor_bias = self.np_random.normal(
                0.0,
                self.sensor_scale_jitter * 0.10,
                size=(self._sensor_columns,),
            ).astype(np.float32)
        else:
            self._movement_sensor_scale.fill(1.0)
            self._movement_sensor_bias.fill(0.0)
        self._movement_sensor_dropout = (
            self.np_random.random(self._sensor_columns)
            < self.sensor_dropout_prob
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # Base reset creates the realistic OD scenario and all safe controllers.
        _legacy_observation, info = super().reset(seed=seed, options=options)
        assert self.controller is not None

        # This training target is normal traffic.  Ambulance experiments remain
        # a separate downstream objective and must not alter the shared policy.
        self.sim_state["next_ambulance_spawn"] = float("inf")
        self.sim_state["active_ambulances"] = {}
        if self.args is not None:
            self.args.disable_ambulances = True

        self.adapter = adapter_for_controller(self.controller, sim.traci, sim)
        self.adapter.reset_history()
        self._snapshot = self.adapter.observe(update_history=True)
        self._sample_sensor_domain()
        info.update(self._map_agnostic_info())
        return self._training_observation(self._snapshot.observation), info

    def _map_agnostic_info(self) -> dict[str, Any]:
        if self.adapter is None or self._snapshot is None:
            return {}
        return {
            "map_agnostic_schema": 2,
            "movement_count": len(self.adapter.topology.movements),
            "phase_candidate_count": self.adapter.phase_count,
            "mean_queue_density": self._snapshot.mean_queue_density,
            "mean_vehicle_density": self._snapshot.mean_vehicle_density,
            "mean_downstream_occupancy": self._snapshot.mean_downstream_occupancy,
            "spillback": self._snapshot.spillback,
            "max_starvation": self._snapshot.max_starvation,
            "served_pressure": self._snapshot.served_pressure,
        }

    def _valid_action_mask(self) -> np.ndarray:
        if self.adapter is None:
            mask = np.zeros(MAX_PHASES + 1, dtype=bool)
            mask[0] = True
            return mask
        return self.adapter.action_mask(
            min_green=legacy.MIN_GREEN_BEFORE_SWITCH,
            max_green=legacy.HARD_MAX_GREEN,
        )

    def action_masks(self) -> np.ndarray:
        return self._valid_action_mask()

    def _apply_rl_action(self, action: int) -> tuple[bool, bool]:
        assert self.controller is not None
        assert self.adapter is not None

        mask = self._valid_action_mask()
        action = int(action)
        if action < 0 or action >= len(mask) or not mask[action]:
            action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

        elapsed = float(self.controller.get("phase_elapsed", 0.0))
        if action == 0:
            if (
                self.controller.get("mode") == "green"
                and elapsed >= legacy.HARD_MAX_GREEN
                and not mask[1:].any()
            ):
                return bool(sim.switch_next_fixed_phase(self.controller)), True
            return False, False

        phase_pos = self.adapter.action_to_phase_position(action)
        if (
            phase_pos is None
            or self.controller.get("mode") != "green"
            or elapsed < legacy.MIN_GREEN_BEFORE_SWITCH
        ):
            return False, False
        return bool(sim.request_switch(self.controller, phase_pos)), False

    def step(self, action: int):
        assert self.controller is not None
        assert self.adapter is not None
        assert self.args is not None
        assert self.rng is not None

        self._apply_fixed_cycle_to_other_tls()
        before = self._snapshot or self.adapter.observe(update_history=False)
        switched, forced = self._apply_rl_action(int(action))

        decision_steps = max(1, int(round(sim.DECISION_INTERVAL / sim.STEP_LENGTH)))
        arrived, spawned, extended, recovered = sim.run_simulation_steps(
            num_steps=decision_steps,
            controllers=self.controllers,
            start_edges=self.main_start_edges,
            turn_index=self.turn_index,
            raw_graph=self.raw_graph,
            edge_metadata=self.edge_metadata,
            core_edges=self.core_edges,
            rng=self.rng,
            turn_counts=self.turn_counts,
            sim_state=self.sim_state,
            args=self.args,
        )
        self.total_arrived += arrived

        current = self.adapter.observe(update_history=True)
        self._snapshot = current
        local_cleared = len(before.vehicle_ids - current.vehicle_ids)
        reward, reward_components = normalized_reward(
            previous=before,
            current=current,
            local_cleared=local_cleared,
            decision_seconds=decision_steps * sim.STEP_LENGTH,
            switched=switched,
            forced=forced,
        )

        # These are diagnostics only; raw global magnitudes are intentionally
        # absent from the reward.
        target_wait, target_queue = legacy.target_wait_and_queue(self.controller)
        global_wait, global_queue, avg_speed = legacy.network_wait_queue_speed()
        sim_time = sim.traci.simulation.getTime()
        terminated = False
        truncated = bool(sim_time >= self.episode_seconds)

        info = self._info(
            switched=switched,
            forced=forced,
            spawned=spawned,
            extended=extended,
            recovered=recovered,
            arrived=arrived,
            local_cleared=local_cleared,
            reward_components=reward_components,
            target_wait=target_wait,
            target_queue=target_queue,
            global_wait=global_wait,
            global_queue=global_queue,
            avg_speed=avg_speed,
        )
        info.update(self._map_agnostic_info())
        return (
            self._training_observation(current.observation),
            float(reward),
            terminated,
            truncated,
            info,
        )


def discover_usable_tls(net_file: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = configure_network(net_file)
    cmd = [
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
    sim.traci.start(cmd)
    records: list[dict[str, Any]] = []
    rng = random.Random(0)
    try:
        for tls_id in sim.traci.trafficlight.getIDList():
            try:
                controller = sim.build_map_agnostic_controller_for_tls(
                    tls_id, rng=rng, activate=False
                )
                if controller is None:
                    continue
                adapter = MapAgnosticTLSAdapter(controller, sim.traci, sim)
                if adapter.phase_count < 2:
                    continue
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
    parser = argparse.ArgumentParser(description="Inspect map-agnostic TLS compatibility.")
    parser.add_argument("--net-file", default=os.environ.get("TRAFFIC_NET_FILE", sim.NET_FILE))
    parser.add_argument("--list-tls-json", action="store_true")
    args = parser.parse_args()
    records = discover_usable_tls(args.net_file)
    if args.list_tls_json:
        print("MAP_AGNOSTIC_TLS_JSON=" + json.dumps(records, separators=(",", ":")))
    else:
        print(json.dumps(records, indent=2))
        print(f"Usable TLS: {len(records)}")


if __name__ == "__main__":
    main()
