#!/usr/bin/env python3
"""Persistent exact-SUMO worker that collects one transition per TLS."""

from __future__ import annotations

import os
import random
import time
import traceback
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from map_agnostic_policy import MovementGraphNetwork
from map_agnostic_tls import (
    DEFAULT_MAX_GREEN,
    DEFAULT_MIN_GREEN,
    GLOBAL_FEATURE_DIM,
    MAX_MOVEMENTS,
    MAX_PHASES,
    MOVEMENT_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    MapTrafficSnapshot,
    adapter_for_controller,
    normalized_reward,
    stack_observations,
)


DYNAMIC_KEYS = ("movements", "phase_features", "global_features")
STATIC_KEYS = ("movement_mask", "movement_adjacency", "phase_membership")


def _topology_bucket(adapter) -> tuple[str, str, str]:
    record = adapter.validate_controller()
    approaches = max(record["incoming_edges"], record["outgoing_edges"])
    movements = record["movements"]
    phases = record["phases"]
    approach_bin = "2-" if approaches <= 2 else "3" if approaches == 3 else "4" if approaches == 4 else "5+"
    movement_bin = "1-4" if movements <= 4 else "5-8" if movements <= 8 else "9+"
    phase_bin = "2" if phases <= 2 else "3" if phases == 3 else "4" if phases == 4 else "5+"
    return approach_bin, movement_bin, phase_bin


class ObservationAugmentor:
    """Episode-constant sensor calibration plus per-decision noise."""

    def __init__(
        self,
        rng: np.random.Generator,
        noise_std: float,
        scale_jitter: float,
        dropout_prob: float,
    ):
        self.rng = rng
        self.noise_std = max(0.0, float(noise_std))
        self.scale_jitter = max(0.0, float(scale_jitter))
        self.dropout_prob = min(0.25, max(0.0, float(dropout_prob)))
        self.resample()

    def resample(self) -> None:
        # Columns 0:15 are inferred from traffic sensors.  Current phase,
        # service time, turn type, lane count, and speed limit are exact
        # controller/topology facts and must not be corrupted.
        self.sensor_columns = 15
        self.scale = self.rng.lognormal(
            0.0, self.scale_jitter, size=(self.sensor_columns,)
        ).astype(np.float32)
        self.bias = self.rng.normal(
            0.0,
            self.scale_jitter * 0.10,
            size=(self.sensor_columns,),
        ).astype(np.float32)
        self.dropout = (
            self.rng.random(self.sensor_columns) < self.dropout_prob
        )

    def apply(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        augmented = {key: value.copy() for key, value in observation.items()}
        valid = augmented["movement_mask"][:, None]
        dynamic = augmented["movements"][:, : self.sensor_columns]
        dynamic = np.clip(
            (dynamic * self.scale[None, :] + self.bias[None, :]) * valid,
            -1.0,
            1.0,
        )
        if self.dropout.any():
            dynamic[:, self.dropout] = 0.0
        if self.noise_std > 0.0:
            dynamic = np.clip(
                dynamic
                + self.rng.normal(0.0, self.noise_std, dynamic.shape).astype(
                    np.float32
                )
                * valid,
                -1.0,
                1.0,
            )
        augmented["movements"][:, : self.sensor_columns] = dynamic

        phase_valid = (
            augmented["phase_membership"].sum(axis=1) > 0.0
        ).astype(np.float32)
        if self.noise_std > 0.0:
            for column in (2, 3, 4, 7):
                noise = self.rng.normal(
                    0.0, self.noise_std, size=phase_valid.shape
                ).astype(np.float32)
                augmented["phase_features"][:, column] = np.clip(
                    augmented["phase_features"][:, column]
                    + noise * phase_valid,
                    -1.0,
                    1.0,
                )
            augmented["global_features"][2:6] = np.clip(
                augmented["global_features"][2:6]
                + self.rng.normal(0.0, self.noise_std, size=(4,)).astype(
                    np.float32
                ),
                -1.0,
                1.0,
            )
        return augmented


class PersistentAllTLSEpisode:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.net_file = str(config["net_file"])
        self.lane_km = float(config["passenger_lane_km"])
        self.seed = int(config["seed"])
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.episode_index = 0
        self.demand_routes = list(config.get("demand_routes", []))
        self.py_rng.shuffle(self.demand_routes)
        self.episode = None
        self.adapters = []
        self.snapshots = []
        self.observations = []
        self.tls_ids: list[str] = []
        self.cache = None

        # Import after the worker has selected its map and optional libsumo.
        import traffic_rl_map_agnostic_env as map_env

        self.map_env = map_env
        self.legacy = map_env.legacy
        self.sim = map_env.sim
        self.sim.PRINT_PHASE_LENGTH_DEBUG = False
        self.sim.OD_DESTINATION_ZONE_HISTOGRAM_PRINT_INTERVAL = float("inf")
        self.sim.OD_ACTIVE_APPROACH_HOTSPOT_PRINT_INTERVAL = float("inf")
        self.augmentor = ObservationAugmentor(
            self.np_rng,
            config["observation_noise_std"],
            config["sensor_scale_jitter"],
            config["sensor_dropout_prob"],
        )
        self._start_episode()

    def _route_file(self) -> str:
        if not self.demand_routes:
            return ""
        return str(
            self.demand_routes[
                self.episode_index % len(self.demand_routes)
            ]
        )

    def _scenario(self):
        density_min, density_max = self.config["target_density_range"]
        density = self.py_rng.uniform(float(density_min), float(density_max))
        target = min(
            int(self.config["max_vehicle_center"]),
            max(40, int(round(self.lane_km * density))),
        )
        maximum = min(
            int(self.config["max_vehicle_center"]),
            max(target, int(round(target * 1.20))),
        )
        initial = min(target, max(10, int(round(target * 0.25))))
        args = SimpleNamespace(
            max_vehicle_center=maximum,
            target_vehicle_center=target,
            initial_vehicle_center=initial,
            spawn_batch_center=int(self.config["spawn_batch_center"]),
            green_duration_center=30.0,
        )
        scenario_seed = self.seed + self.episode_index * 100_003
        base = self.legacy.build_fixed_scenario(seed=scenario_seed, args=args)
        return replace(
            base,
            seed=scenario_seed,
            max_vehicles=maximum,
            target_vehicles=target,
            initial_vehicles=initial,
            signal_timing_jitter=self.py_rng.uniform(0.05, 0.25),
        )

    def _start_episode(self) -> None:
        if self.episode is not None:
            self.episode.close()
        scenario = self._scenario()
        route_file = self._route_file()
        outer_args = SimpleNamespace(
            episode_seconds=int(self.config["episode_seconds"]),
            demand_route_file=route_file,
        )
        self.episode = self.legacy.AnchorlessSimulationEpisode(
            scenario=scenario,
            seed=scenario.seed,
            args=outer_args,
            gui=False,
            env_rank=int(self.config["worker_rank"]),
        )
        # A pre-generated demand route already contains the complete,
        # controller-independent departure schedule. Disable the legacy
        # target-population filler so it cannot add global_car vehicles.
        if route_file:
            self.episode.args.initial_vehicles = 0
            self.episode.args.target_vehicles = 0
            self.episode.args.spawn_batch = 0

        self.episode.args.disable_ambulances = True
        self.episode.reset()

        # Some legacy routing/recovery helpers use global_car for findRoute().
        # Supply a compatible alias without adding any dynamic vehicles.
        if route_file:
            type_ids = list(self.sim.traci.vehicletype.getIDList())
            if "global_car" not in type_ids:
                source_type = next(
                    (
                        type_id
                        for type_id in type_ids
                        if type_id.endswith("__passenger")
                    ),
                    None,
                )
                if source_type is None and "DEFAULT_VEHTYPE" in type_ids:
                    source_type = "DEFAULT_VEHTYPE"
                if source_type is None:
                    raise RuntimeError(
                        "Fixed-demand route loaded no passenger vehicle type."
                    )
                self.sim.traci.vehicletype.copy(
                    source_type,
                    "global_car",
                )
        self.episode.sim_state["next_ambulance_spawn"] = float("inf")
        self.episode.sim_state["active_ambulances"] = {}

        self.cache = MapTrafficSnapshot(self.sim.traci, self.sim)
        self.adapters = [
            adapter_for_controller(
                controller,
                self.sim.traci,
                self.sim,
                snapshot_cache=self.cache,
            )
            for controller in self.episode.controllers
            if not controller.get("disabled")
        ]
        if not self.adapters:
            raise RuntimeError(f"No usable map-agnostic TLS found in {self.net_file}")
        new_tls_ids = [adapter.tls_id for adapter in self.adapters]
        if self.tls_ids and new_tls_ids != self.tls_ids:
            raise RuntimeError(
                "TLS ordering changed across persistent episodes: "
                f"{self.tls_ids} != {new_tls_ids}"
            )
        self.tls_ids = new_tls_ids
        for adapter in self.adapters:
            adapter.reset_history()
        self.cache.refresh(self.adapters)
        self.snapshots = [
            adapter.observe(update_history=True) for adapter in self.adapters
        ]
        self.augmentor.resample()
        self.observations = [
            self.augmentor.apply(snapshot.observation)
            for snapshot in self.snapshots
        ]

    def _action_mask(self, adapter) -> np.ndarray:
        return adapter.action_mask(
            min_green=self.legacy.MIN_GREEN_BEFORE_SWITCH,
            max_green=self.legacy.HARD_MAX_GREEN,
        )

    def action_masks(self) -> np.ndarray:
        return np.stack(
            [self._action_mask(adapter) for adapter in self.adapters], axis=0
        ).astype(bool)

    def _apply_action(self, adapter, action: int) -> tuple[bool, bool]:
        controller = adapter.controller
        mask = self._action_mask(adapter)
        action = int(action)
        if action < 0 or action >= len(mask) or not mask[action]:
            action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

        elapsed = float(controller.get("phase_elapsed", 0.0))
        if action == 0:
            if (
                controller.get("mode") == "green"
                and elapsed >= self.legacy.HARD_MAX_GREEN
                and not mask[1:].any()
            ):
                return bool(self.sim.switch_next_fixed_phase(controller)), True
            return False, False
        phase_pos = adapter.action_to_phase_position(action)
        if (
            phase_pos is None
            or controller.get("mode") != "green"
            or elapsed < self.legacy.MIN_GREEN_BEFORE_SWITCH
        ):
            return False, False
        return bool(self.sim.request_switch(controller, phase_pos)), False

    def step(self, actions: np.ndarray, reset_on_done: bool = True):
        before = list(self.snapshots)
        switches = [
            self._apply_action(adapter, int(action))
            for adapter, action in zip(self.adapters, actions)
        ]
        decision_steps = max(
            1,
            int(
                round(
                    float(self.config["decision_seconds"])
                    / float(self.sim.STEP_LENGTH)
                )
            ),
        )
        arrived, _spawned, _extended, _recovered = self.sim.run_simulation_steps(
            num_steps=decision_steps,
            controllers=self.episode.controllers,
            start_edges=self.episode.main_start_edges,
            turn_index=self.episode.turn_index,
            raw_graph=self.episode.raw_graph,
            edge_metadata=self.episode.edge_metadata,
            core_edges=self.episode.core_edges,
            rng=self.episode.rng,
            turn_counts=self.episode.turn_counts,
            sim_state=self.episode.sim_state,
            args=self.episode.args,
        )
        self.episode.total_arrived += int(arrived)
        self.cache.refresh(self.adapters)
        current = [
            adapter.observe(update_history=True) for adapter in self.adapters
        ]
        rewards = np.zeros(len(self.adapters), dtype=np.float32)
        for index, (previous, snapshot, switch) in enumerate(
            zip(before, current, switches)
        ):
            local_cleared = len(previous.vehicle_ids - snapshot.vehicle_ids)
            reward, _components = normalized_reward(
                previous=previous,
                current=snapshot,
                local_cleared=local_cleared,
                decision_seconds=decision_steps * self.sim.STEP_LENGTH,
                switched=switch[0],
                forced=switch[1],
            )
            rewards[index] = reward
        self.snapshots = current
        self.observations = [
            self.augmentor.apply(snapshot.observation) for snapshot in current
        ]

        sim_time = float(self.sim.traci.simulation.getTime())
        done = sim_time >= float(self.config["episode_seconds"])
        if done and reset_on_done:
            self.episode_index += 1
            self._start_episode()
        return rewards, done, sim_time

    def sample_weights(self) -> np.ndarray:
        buckets = [_topology_bucket(adapter) for adapter in self.adapters]
        counts = Counter(buckets)
        bucket_count = max(1, len(counts))
        weights = np.asarray(
            [1.0 / (bucket_count * counts[bucket]) for bucket in buckets],
            dtype=np.float32,
        )
        return weights / max(1e-9, float(weights.sum()))

    def close(self) -> None:
        if self.episode is not None:
            self.episode.close()
            self.episode = None


def _tensor_observation(
    observations: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = stack_observations(observations)
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in batch.items()
    }


@torch.no_grad()
def _policy_step(network, observations, masks, device, deterministic=False):
    tensor_obs = _tensor_observation(observations, device)
    logits, values = network(tensor_obs)
    tensor_masks = torch.as_tensor(masks, dtype=torch.bool, device=device)
    masked_logits = logits.masked_fill(~tensor_masks, -1e8)
    distribution = Categorical(logits=masked_logits)
    actions = (
        masked_logits.argmax(dim=-1)
        if deterministic
        else distribution.sample()
    )
    return (
        actions.cpu().numpy().astype(np.int64),
        distribution.log_prob(actions).cpu().numpy().astype(np.float32),
        values.squeeze(-1).cpu().numpy().astype(np.float32),
    )


def normalized_max_pressure_actions(
    observations: list[dict[str, np.ndarray]], masks: np.ndarray
) -> np.ndarray:
    """Map-scale-free teacher used only as a decaying PPO auxiliary loss."""
    phase = np.stack(
        [observation["phase_features"] for observation in observations],
        axis=0,
    ).astype(np.float32)
    global_features = np.stack(
        [observation["global_features"] for observation in observations],
        axis=0,
    ).astype(np.float32)
    phase_scores = (
        1.00 * phase[..., 2]
        + 1.50 * phase[..., 3]
        + 0.60 * phase[..., 4]
        + 0.80 * phase[..., 5]
        + 0.20 * phase[..., 7]
    )
    scores = np.full(
        (len(observations), MAX_PHASES + 1), -1e9, dtype=np.float32
    )
    scores[:, 1:] = phase_scores
    scores[:, 0] = (phase_scores * phase[..., 0]).sum(axis=1) + 0.15 * (
        1.0 - global_features[:, 1]
    )
    scores[~np.asarray(masks, dtype=bool)] = -1e9
    return np.argmax(scores, axis=1).astype(np.int64)


def collect_rollout(
    episode: PersistentAllTLSEpisode,
    network: MovementGraphNetwork,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    connection,
    progress_interval: int,
    deterministic: bool = False,
) -> dict[str, Any]:
    device = torch.device("cpu")
    network.eval()
    n_agents = len(episode.adapters)
    dynamic = {key: [] for key in DYNAMIC_KEYS}
    actions_list = []
    masks_list = []
    log_probs_list = []
    values_list = []
    rewards_list = []
    dones_list = []
    teacher_actions_list = []
    reset_count = 0
    start_wall = time.monotonic()
    start_sim_time = float(episode.sim.traci.simulation.getTime())

    static = {
        key: np.stack(
            [observation[key] for observation in episode.observations], axis=0
        )
        for key in STATIC_KEYS
    }

    for step in range(rollout_steps):
        observations = episode.observations
        masks = episode.action_masks()
        actions, log_probs, values = _policy_step(
            network,
            observations,
            masks,
            device,
            deterministic=deterministic,
        )
        teacher_actions = normalized_max_pressure_actions(
            observations, masks
        )
        for key in DYNAMIC_KEYS:
            dynamic[key].append(
                np.stack([observation[key] for observation in observations], axis=0)
            )
        rewards, done, _sim_time = episode.step(
            actions, reset_on_done=not deterministic
        )
        actions_list.append(actions)
        masks_list.append(masks)
        log_probs_list.append(log_probs)
        values_list.append(values)
        rewards_list.append(rewards)
        dones_list.append(np.full(n_agents, done, dtype=np.float32))
        teacher_actions_list.append(teacher_actions)
        reset_count += int(done)
        if connection is not None and (
            (step + 1) % max(1, progress_interval) == 0
            or step + 1 == rollout_steps
        ):
            connection.send(
                {
                    "type": "progress",
                    "step": step + 1,
                    "total": rollout_steps,
                    "transitions": (step + 1) * n_agents,
                    "tls": n_agents,
                }
            )

    final_masks = episode.action_masks()
    _final_actions, _final_log_probs, final_values = _policy_step(
        network,
        episode.observations,
        final_masks,
        device,
        deterministic=deterministic,
    )
    values_array = np.asarray(values_list, dtype=np.float32)
    rewards_array = np.asarray(rewards_list, dtype=np.float32)
    dones_array = np.asarray(dones_list, dtype=np.float32)
    advantages = np.zeros_like(rewards_array)
    last_gae = np.zeros(n_agents, dtype=np.float32)
    next_values = final_values
    for step in reversed(range(rollout_steps)):
        nonterminal = 1.0 - dones_array[step]
        delta = (
            rewards_array[step]
            + gamma * next_values * nonterminal
            - values_array[step]
        )
        last_gae = (
            delta + gamma * gae_lambda * nonterminal * last_gae
        )
        advantages[step] = last_gae
        next_values = values_array[step]
    returns = advantages + values_array

    end_sim_time = float(episode.sim.traci.simulation.getTime())
    elapsed = max(1e-9, time.monotonic() - start_wall)
    return {
        "type": "rollout",
        "net_file": episode.net_file,
        "tls_ids": episode.tls_ids,
        "dynamic": {
            key: np.asarray(value, dtype=np.float16)
            for key, value in dynamic.items()
        },
        "static": {
            "movement_mask": static["movement_mask"].astype(np.uint8),
            "movement_adjacency": static["movement_adjacency"].astype(np.uint8),
            "phase_membership": static["phase_membership"].astype(np.float16),
        },
        "action_masks": np.asarray(masks_list, dtype=np.uint8),
        "actions": np.asarray(actions_list, dtype=np.int16),
        "old_log_probs": np.asarray(log_probs_list, dtype=np.float32),
        "old_values": values_array,
        "advantages": advantages,
        "returns": returns,
        "teacher_actions": np.asarray(
            teacher_actions_list, dtype=np.int16
        ),
        "agent_weights": episode.sample_weights(),
        "metrics": {
            "wall_seconds": elapsed,
            "sim_seconds": max(0.0, end_sim_time - start_sim_time)
            + reset_count * float(episode.config["episode_seconds"]),
            "transitions": rollout_steps * n_agents,
            "tls": n_agents,
            "episode_resets": reset_count,
            "mean_reward": float(rewards_array.mean()),
            "mean_topology_balanced_reward": float(
                (
                    rewards_array
                    * episode.sample_weights()[None, :]
                ).sum(axis=1).mean()
            ),
            "transitions_per_second": rollout_steps * n_agents / elapsed,
        },
    }


def rollout_worker_main(connection, config: dict[str, Any]) -> None:
    episode = None
    try:
        os.environ["TRAFFIC_NET_FILE"] = str(config["net_file"])
        if config.get("use_libsumo"):
            os.environ["SUMO_USE_LIBSUMO"] = "1"
        torch.set_num_threads(1)
        network = MovementGraphNetwork(
            embed_dim=int(config["embed_dim"]),
            graph_layers=int(config["graph_layers"]),
        ).cpu()
        episode = PersistentAllTLSEpisode(config)
        connection.send(
            {
                "type": "ready",
                "net_file": config["net_file"],
                "tls": len(episode.adapters),
                "tls_ids": episode.tls_ids,
            }
        )
        while True:
            request = connection.recv()
            command = request.get("cmd")
            if command == "close":
                break
            if command != "rollout":
                raise ValueError(f"Unknown worker command: {command}")
            network.load_state_dict(
                {
                    key: torch.as_tensor(value)
                    for key, value in request["state_dict"].items()
                }
            )
            result = collect_rollout(
                episode=episode,
                network=network,
                rollout_steps=int(request["rollout_steps"]),
                gamma=float(request["gamma"]),
                gae_lambda=float(request["gae_lambda"]),
                connection=connection,
                progress_interval=int(request["progress_interval"]),
                deterministic=bool(request.get("deterministic", False)),
            )
            connection.send(result)
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        if episode is not None:
            episode.close()
        try:
            connection.close()
        except Exception:
            pass
