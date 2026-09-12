#!/usr/bin/env python3
"""Persistent exact-SUMO rollout worker for detector-realistic schema v4."""

from __future__ import annotations

import os
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from detector_realistic_policy import DetectorGraphNetwork
from detector_realistic_tls import (
    DetectorSensorConfig,
    DetectorTrafficSnapshot,
    adapter_for_controller,
    normalized_reward,
)
from map_agnostic_multiagent_worker import (
    PersistentAllTLSEpisode,
    collect_rollout,
)


def sensor_config_from_worker(config: dict[str, Any]) -> DetectorSensorConfig:
    return DetectorSensorConfig(
        profile=str(config.get("sensor_profile", "mixed")),
        nominal_decision_seconds=float(config["decision_seconds"]),
        stopbar_zone_meters=float(config.get("stopbar_zone_meters", 12.0)),
        advance_distance_meters=float(config.get("advance_distance_meters", 80.0)),
        advance_zone_meters=float(config.get("advance_zone_meters", 10.0)),
        downstream_zone_meters=float(config.get("downstream_zone_meters", 30.0)),
        history_seconds=float(config.get("detector_history_seconds", 60.0)),
        observation_noise_std=float(config.get("detector_noise_std", 0.02)),
        calibration_jitter=float(config.get("detector_calibration_jitter", 0.05)),
        transient_dropout_probability=float(config.get("detector_dropout_prob", 0.03)),
        stuck_detector_probability=float(config.get("detector_stuck_prob", 0.01)),
        max_latency_decisions=int(config.get("max_detector_latency_decisions", 1)),
        mixed_speed_probability=float(config.get("mixed_speed_probability", 0.50)),
        mixed_downstream_probability=float(
            config.get("mixed_downstream_probability", 0.35)
        ),
    )


class DetectorRealisticAllTLSEpisode(PersistentAllTLSEpisode):
    """Exact-SUMO episode with a detector-only policy and reward interface."""

    def __init__(self, config: dict[str, Any]):
        # Schema v3's post-observation augmentation must be disabled because
        # schema v4 corrupts physical detector channels before computing its
        # estimates. Applying both would double-count noise and erase missing
        # sensor sentinels.
        adapted = dict(config)
        adapted["observation_noise_std"] = 0.0
        adapted["sensor_scale_jitter"] = 0.0
        adapted["sensor_dropout_prob"] = 0.0
        self.detector_sensor_config = sensor_config_from_worker(adapted)
        self.detector_cache: DetectorTrafficSnapshot | None = None
        super().__init__(adapted)

    def _start_episode(self) -> None:
        # Recreate the parent's exact-SUMO episode lifecycle directly so the
        # schema-v3 oracle adapter is never constructed and no route, waiting-
        # time, or ETA query occurs along the schema-v4 policy path.
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
        if route_file:
            self.episode.args.initial_vehicles = 0
            self.episode.args.target_vehicles = 0
            self.episode.args.spawn_batch = 0

        self.episode.args.disable_ambulances = True
        self.episode.reset()

        # Legacy routing/recovery helpers expect this passenger alias. It is a
        # vehicle type, not a dynamically added vehicle or policy input.
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
                self.sim.traci.vehicletype.copy(source_type, "global_car")
        self.episode.sim_state["next_ambulance_spawn"] = float("inf")
        self.episode.sim_state["active_ambulances"] = {}

        self.detector_cache = DetectorTrafficSnapshot(self.sim.traci)
        detector_adapters = []
        for index, controller in enumerate(self.episode.controllers):
            if controller.get("disabled"):
                continue
            rng = np.random.default_rng(
                self.seed
                + self.episode_index * 1_000_003
                + index * 10_007
            )
            detector_adapters.append(
                adapter_for_controller(
                    controller,
                    self.sim.traci,
                    self.sim,
                    snapshot_cache=self.detector_cache,
                    sensor_config=self.detector_sensor_config,
                    rng=rng,
                )
            )
        if not detector_adapters:
            raise RuntimeError(
                f"No detector-realistic TLS found in {self.net_file}"
            )
        detector_tls_ids = [adapter.tls_id for adapter in detector_adapters]
        if self.tls_ids and detector_tls_ids != self.tls_ids:
            raise RuntimeError(
                "Detector TLS ordering changed across persistent episodes: "
                f"{detector_tls_ids} != {self.tls_ids}"
            )
        self.tls_ids = detector_tls_ids
        self.adapters = detector_adapters
        self.cache = self.detector_cache
        self.detector_cache.refresh(self.adapters)
        self.snapshots = [
            adapter.observe(update_history=True) for adapter in self.adapters
        ]
        self.observations = [snapshot.observation for snapshot in self.snapshots]
        self.augmentor.resample()

    def step(self, actions: np.ndarray, reset_on_done: bool = True):
        """Advance exact SUMO and compute reward from detector aggregates only."""

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
        decision_seconds = decision_steps * self.sim.STEP_LENGTH
        rewards = np.zeros(len(self.adapters), dtype=np.float32)
        for index, (adapter, previous, snapshot, switch) in enumerate(
            zip(self.adapters, before, current, switches)
        ):
            reward, _components = normalized_reward(
                previous=previous,
                current=snapshot,
                local_cleared=int(round(adapter.last_detected_departures)),
                decision_seconds=decision_seconds,
                switched=switch[0],
                forced=switch[1],
            )
            rewards[index] = reward
        self.snapshots = current
        self.observations = [snapshot.observation for snapshot in current]

        sim_time = float(self.sim.traci.simulation.getTime())
        done = sim_time >= float(self.config["episode_seconds"])
        if done and reset_on_done:
            self.episode_index += 1
            self._start_episode()
        return rewards, done, sim_time


def rollout_worker_main(connection, config: dict[str, Any]) -> None:
    episode = None
    try:
        os.environ["TRAFFIC_NET_FILE"] = str(config["net_file"])
        if config.get("use_libsumo"):
            os.environ["SUMO_USE_LIBSUMO"] = "1"
        torch.set_num_threads(1)
        network = DetectorGraphNetwork(
            embed_dim=int(config["embed_dim"]),
            graph_layers=int(config["graph_layers"]),
        ).cpu()
        episode = DetectorRealisticAllTLSEpisode(config)
        connection.send(
            {
                "type": "ready",
                "net_file": config["net_file"],
                "tls": len(episode.adapters),
                "tls_ids": episode.tls_ids,
                "sensor_profile": episode.detector_sensor_config.profile,
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
            connection.send(
                collect_rollout(
                    episode=episode,
                    network=network,
                    rollout_steps=int(request["rollout_steps"]),
                    gamma=float(request["gamma"]),
                    gae_lambda=float(request["gae_lambda"]),
                    connection=connection,
                    progress_interval=int(request["progress_interval"]),
                    deterministic=bool(request.get("deterministic", False)),
                )
            )
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


__all__ = [
    "DetectorRealisticAllTLSEpisode",
    "rollout_worker_main",
    "sensor_config_from_worker",
]
