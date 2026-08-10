#!/usr/bin/env python3
"""Persistent all-TLS rollout worker for the ambulance emergency override."""

from __future__ import annotations

import math
import os
import statistics
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
import torch
from torch.distributions import Categorical

from ambulance_emergency import (
    EmergencyFeatureBuilder,
    EmergencyObservationConfig,
    EmergencyOverrideNetwork,
    RollingGreenCorridor,
    emergency_rewards,
    stack_emergency_observations,
)
from ambulance_system import (
    AMBULANCE_ID_PREFIX,
    AmbulanceSystem,
    AmbulanceSystemConfig,
)
from map_agnostic_multiagent_worker import (
    DYNAMIC_KEYS,
    STATIC_KEYS,
    PersistentAllTLSEpisode,
    _tensor_observation,
    normalized_max_pressure_actions,
)
from map_agnostic_policy import MovementGraphNetwork
from map_agnostic_tls import MAX_PHASES


def _safe_vehicle_value(method, vehicle_id: str, default: float = 0.0) -> float:
    try:
        value = float(method(vehicle_id))
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


@dataclass
class OrdinaryTrafficMonitor:
    """Exact ordinary-vehicle events plus one-second network integrals."""

    traci: Any
    last_time: float = 0.0
    departed_ids: set[str] = field(default_factory=set)
    arrived_ids: set[str] = field(default_factory=set)
    teleported_ids: set[str] = field(default_factory=set)
    collision_ids: set[str] = field(default_factory=set)
    active_time_loss: dict[str, float] = field(default_factory=dict)
    active_wait: dict[str, float] = field(default_factory=dict)
    completed_time_losses: list[float] = field(default_factory=list)
    completed_waits: list[float] = field(default_factory=list)
    queue_vehicle_seconds: float = 0.0
    active_vehicle_seconds: float = 0.0
    speed_vehicle_seconds: float = 0.0
    observation_seconds: float = 0.0

    @staticmethod
    def _ordinary(vehicle_id: str) -> bool:
        return not str(vehicle_id).startswith(AMBULANCE_ID_PREFIX)

    def after_simulation_step(
        self,
        sim_state: dict[str, Any],
        sim_time: float,
        args: Any,
    ) -> None:
        del sim_state, args
        now = float(sim_time)
        elapsed = max(0.0, now - self.last_time)
        self.last_time = now
        simulation = self.traci.simulation

        def event_ids(name: str) -> set[str]:
            method = getattr(simulation, name, None)
            if not callable(method):
                return set()
            try:
                return {
                    str(vehicle_id)
                    for vehicle_id in method()
                    if self._ordinary(str(vehicle_id))
                }
            except Exception:
                return set()

        departed = event_ids("getDepartedIDList")
        arrived = event_ids("getArrivedIDList")
        teleported = event_ids("getStartingTeleportIDList")
        collided = event_ids("getCollidingVehiclesIDList")
        self.departed_ids.update(departed)
        self.arrived_ids.update(arrived)
        self.teleported_ids.update(teleported)
        self.collision_ids.update(collided)
        for vehicle_id in arrived:
            self.completed_time_losses.append(
                self.active_time_loss.pop(vehicle_id, 0.0)
            )
            self.completed_waits.append(
                self.active_wait.pop(vehicle_id, 0.0)
            )
        for vehicle_id in teleported | collided:
            self.active_time_loss.pop(vehicle_id, None)
            self.active_wait.pop(vehicle_id, None)

        try:
            current_ids = [
                str(vehicle_id)
                for vehicle_id in self.traci.vehicle.getIDList()
                if self._ordinary(str(vehicle_id))
            ]
        except Exception:
            current_ids = []
        speeds: list[float] = []
        queue = 0
        for vehicle_id in current_ids:
            speed = max(
                0.0,
                _safe_vehicle_value(
                    self.traci.vehicle.getSpeed,
                    vehicle_id,
                ),
            )
            speeds.append(speed)
            queue += int(speed < 0.1)
            self.active_time_loss[vehicle_id] = max(
                self.active_time_loss.get(vehicle_id, 0.0),
                _safe_vehicle_value(
                    self.traci.vehicle.getTimeLoss,
                    vehicle_id,
                    self.active_time_loss.get(vehicle_id, 0.0),
                ),
            )
            # SUMO's accumulated-wait API uses a configurable rolling memory
            # window and can decrease during a long trip.  Integrating the
            # stopped predicate at every microstep gives a true episode-total
            # wait that cannot silently shrink.
            self.active_wait[vehicle_id] = (
                self.active_wait.get(vehicle_id, 0.0)
                + (elapsed if speed < 0.1 else 0.0)
            )
        self.queue_vehicle_seconds += float(queue) * elapsed
        self.active_vehicle_seconds += float(len(current_ids)) * elapsed
        self.speed_vehicle_seconds += float(sum(speeds)) * elapsed
        self.observation_seconds += elapsed

    def summary(
        self, scheduled_vehicles: int | None = None
    ) -> dict[str, Any]:
        arrived = len(self.arrived_ids)
        departed = len(self.departed_ids)
        all_departed_time_losses = (
            list(self.completed_time_losses)
            + list(self.active_time_loss.values())
        )
        all_departed_waits = (
            list(self.completed_waits)
            + list(self.active_wait.values())
        )
        return {
            "scheduled_total": (
                int(scheduled_vehicles)
                if scheduled_vehicles is not None
                else None
            ),
            "departed_total": departed,
            "arrived_total": arrived,
            "completion_rate": arrived / max(1, departed),
            "scheduled_throughput_rate": (
                arrived / max(1, int(scheduled_vehicles))
                if scheduled_vehicles is not None
                else None
            ),
            "mean_time_loss_s": (
                statistics.fmean(self.completed_time_losses)
                if self.completed_time_losses
                else 0.0
            ),
            "p95_time_loss_s": _percentile(
                self.completed_time_losses, 95.0
            ),
            "mean_time_loss_all_departed_s": (
                statistics.fmean(all_departed_time_losses)
                if all_departed_time_losses
                else 0.0
            ),
            "p95_time_loss_all_departed_s": _percentile(
                all_departed_time_losses, 95.0
            ),
            "mean_wait_s": (
                statistics.fmean(self.completed_waits)
                if self.completed_waits
                else 0.0
            ),
            "mean_wait_all_departed_s": (
                statistics.fmean(all_departed_waits)
                if all_departed_waits
                else 0.0
            ),
            "mean_queue_vehicles": (
                self.queue_vehicle_seconds
                / max(1e-9, self.observation_seconds)
            ),
            "mean_speed_mps": (
                self.speed_vehicle_seconds
                / max(1e-9, self.active_vehicle_seconds)
            ),
            "teleported_vehicle_ids": sorted(self.teleported_ids),
            "collision_vehicle_ids": sorted(self.collision_ids),
        }


class _ClearanceDeltaView:
    """Expose filtered clearances while forwarding every other delta field."""

    def __init__(self, source, cleared_tls):
        self._source = source
        self.cleared_tls = list(cleared_tls)

    def __getattr__(self, name):
        return getattr(self._source, name)


class EmergencyAllTLSEpisode(PersistentAllTLSEpisode):
    """Schema-v3 traffic episode plus deterministic ambulance demand."""

    def __init__(self, config: dict[str, Any]):
        # The frozen schema-v3 controller must retain the action mask and
        # bounded hard-max cycling it was trained with.  The exact exit-gap
        # shield is applied later only to emergency actions that differ from
        # that frozen-base action.
        base_config = dict(config)
        self._emergency_required_exit_gap_meters = float(
            base_config.get(
                "emergency_override_required_exit_gap_meters",
                base_config.get("required_exit_gap_meters", 18.0),
            )
        )
        if not math.isfinite(self._emergency_required_exit_gap_meters):
            raise ValueError(
                "emergency override exit gap must be finite"
            )
        self._emergency_required_exit_gap_meters = max(
            0.0, self._emergency_required_exit_gap_meters
        )
        base_config["strict_exit_space"] = False
        base_config["allow_unsafe_hard_max_fallback"] = True
        base_config[
            "emergency_override_required_exit_gap_meters"
        ] = self._emergency_required_exit_gap_meters
        self.system: AmbulanceSystem | None = None
        self.corridor: RollingGreenCorridor | None = None
        self.feature_builder: EmergencyFeatureBuilder | None = None
        self.ordinary_monitor: OrdinaryTrafficMonitor | None = None
        self.emergency_contexts = []
        self._ordinary_queue_baselines: dict[str, float] = {}
        self._intervention_queue_baselines: dict[str, float] = {}
        self._pending_recoveries: dict[str, dict[str, float]] = {}
        self._completed_recovery_seconds: list[float] = []
        self._decision_base_actions: dict[str, int] = {}
        super().__init__(base_config)

    def emergency_override_exit_space_masks(
        self,
        base_masks: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Return exact-gap legality for emergency-only phase changes.

        Action zero does not activate a phase.  Frozen-base actions are
        restored by ``_policy_actions`` even when the corresponding phase does
        not pass this emergency-only mask.
        """

        base_masks = np.asarray(base_masks, dtype=bool)
        active = np.asarray(active, dtype=bool)
        if base_masks.ndim != 2 or base_masks.shape[0] != len(
            self.adapters
        ):
            raise ValueError(
                "base action-mask shape does not match ambulance TLS order"
            )
        if active.shape != (len(self.adapters),):
            raise ValueError(
                "override-active shape does not match ambulance TLS order"
            )

        safe = np.zeros_like(base_masks, dtype=bool)
        safe[:, 0] = True
        for index, (adapter, is_active) in enumerate(
            zip(self.adapters, active)
        ):
            if not is_active:
                continue
            for action in np.flatnonzero(base_masks[index, 1:]) + 1:
                phase_pos = adapter.action_to_phase_position(int(action))
                if phase_pos is None:
                    continue
                safe[index, int(action)] = bool(
                    adapter.phase_position_has_exit_space(
                        phase_pos,
                        self._emergency_required_exit_gap_meters,
                    )
                )
        return safe

    @staticmethod
    def _clear_emergency_transition(controller: dict[str, Any]) -> None:
        controller.pop("emergency_green_activation_guard", None)
        controller.pop("emergency_override_phase_pos", None)
        controller.pop("emergency_base_fallback_phase_pos", None)

    def _apply_action(self, adapter, action: int) -> tuple[bool, bool]:
        """Arm a post-clearance guard only for a real emergency switch."""

        controller = adapter.controller
        base_action = int(
            self._decision_base_actions.get(
                str(adapter.tls_id), int(action)
            )
        )
        action = int(action)
        armed = False
        if action > 0 and action != base_action:
            override_phase_pos = adapter.action_to_phase_position(action)
            base_phase_pos = (
                adapter.action_to_phase_position(base_action)
                if base_action > 0
                else int(controller.get("phase_pos", 0))
            )
            if override_phase_pos is not None and base_phase_pos is not None:
                gap = self._emergency_required_exit_gap_meters
                controller["emergency_green_activation_guard"] = (
                    lambda phase_position, adapter=adapter, gap=gap: (
                        adapter.phase_position_has_exit_space(
                            phase_position, gap
                        )
                    )
                )
                controller["emergency_override_phase_pos"] = int(
                    override_phase_pos
                )
                controller["emergency_base_fallback_phase_pos"] = int(
                    base_phase_pos
                )
                armed = True

        result = super()._apply_action(adapter, action)
        if armed and not result[0]:
            self._clear_emergency_transition(controller)
        elif armed:
            controller["emergency_override_switch_requests"] = (
                int(
                    controller.get(
                        "emergency_override_switch_requests", 0
                    )
                )
                + 1
            )
        return result

    def _episode_ambulance_config(self) -> AmbulanceSystemConfig:
        payload = dict(self.config.get("ambulance_system", {}))
        choices = [
            int(value)
            for value in self.config.get(
                "ambulance_max_active_choices", ()
            )
            if int(value) > 0
        ]
        if choices:
            payload["max_active_ambulances"] = self.py_rng.choice(choices)
        interval_range = tuple(
            float(value)
            for value in self.config.get(
                "ambulance_interval_range", ()
            )
        )
        if len(interval_range) == 2:
            payload["spawn_interval_seconds"] = self.py_rng.uniform(
                min(interval_range), max(interval_range)
            )
        return AmbulanceSystemConfig(**payload)

    def _start_episode(self) -> None:
        if self.system is not None:
            try:
                self.system.finish_episode()
            except Exception:
                pass
        super()._start_episode()
        # Rebuild the map cache so normal-traffic observations and rewards do
        # not count ambulances a second time.
        self.cache.exclude_vehicle = lambda vehicle_id: str(
            vehicle_id
        ).startswith(AMBULANCE_ID_PREFIX)
        self.cache.refresh(self.adapters)
        self.snapshots = [
            adapter.observe(update_history=True)
            for adapter in self.adapters
        ]
        self.observations = [
            self.augmentor.apply(snapshot.observation)
            for snapshot in self.snapshots
        ]

        corridor_config = dict(self.config.get("corridor", {}))
        self.corridor = RollingGreenCorridor(**corridor_config)
        ambulance_config = self._episode_ambulance_config()
        self.system = AmbulanceSystem(
            traci_module=self.sim.traci,
            simulation_module=self.sim,
            adapters=self.adapters,
            raw_graph=self.episode.raw_graph,
            edge_metadata=self.episode.edge_metadata,
            sim_state=self.episode.sim_state,
            episode_seconds=float(self.config["episode_seconds"]),
            schedule_seed=(
                int(self.seed)
                + int(self.episode_index) * 1_000_003
                + 71_911
            ),
            config=ambulance_config,
        )
        self.ordinary_monitor = OrdinaryTrafficMonitor(
            self.sim.traci,
            last_time=float(self.sim.traci.simulation.getTime()),
        )
        self.episode.sim_state.setdefault(
            "after_simulation_step_hooks", []
        ).append(self.ordinary_monitor.after_simulation_step)
        self._ordinary_queue_baselines = {}
        self._intervention_queue_baselines = {}
        self._pending_recoveries = {}
        self._completed_recovery_seconds = []
        self.feature_builder = EmergencyFeatureBuilder(
            self.system,
            self.adapters,
            self.corridor,
            config=EmergencyObservationConfig(
                **dict(
                    self.config.get(
                        "emergency_observation", {}
                    )
                )
            ),
        )
        sim_time = float(self.sim.traci.simulation.getTime())
        self.emergency_contexts = self.feature_builder.build(sim_time)

    def sample_ordinary_network_metrics(self) -> dict[str, float]:
        assert self.system is not None
        vehicle_ids = [
            str(vehicle_id)
            for vehicle_id in self.sim.traci.vehicle.getIDList()
            if not str(vehicle_id).startswith(AMBULANCE_ID_PREFIX)
        ]
        waits: list[float] = []
        time_losses: list[float] = []
        speeds: list[float] = []
        queue = 0
        for vehicle_id in vehicle_ids:
            speed = max(
                0.0,
                _safe_vehicle_value(
                    self.system.traci.vehicle.getSpeed,
                    vehicle_id,
                ),
            )
            speeds.append(speed)
            queue += int(speed < 0.1)
            waiting_method = getattr(
                self.system.traci.vehicle,
                "getAccumulatedWaitingTime",
                self.system.traci.vehicle.getWaitingTime,
            )
            waits.append(
                max(
                    0.0,
                    _safe_vehicle_value(
                        waiting_method,
                        vehicle_id,
                    ),
                )
            )
            time_losses.append(
                max(
                    0.0,
                    _safe_vehicle_value(
                        self.system.traci.vehicle.getTimeLoss,
                        vehicle_id,
                    ),
                )
            )
        return {
            "ordinary_vehicle_count": float(len(vehicle_ids)),
            "ordinary_queue": float(queue),
            "ordinary_mean_speed": (
                float(np.mean(speeds)) if speeds else 0.0
            ),
            "ordinary_mean_wait": (
                float(np.mean(waits)) if waits else 0.0
            ),
            "ordinary_mean_time_loss": (
                float(np.mean(time_losses)) if time_losses else 0.0
            ),
        }

    def sample_ordinary_tls_queues(self) -> dict[str, float]:
        """Count distinct stopped ordinary vehicles on each TLS approach."""

        queues: dict[str, float] = {}
        for adapter in self.adapters:
            tls_id = str(adapter.tls_id)
            queued_vehicle_ids: set[str] = set()
            try:
                controlled_lanes = set(
                    self.sim.traci.trafficlight.getControlledLanes(tls_id)
                )
            except Exception:
                controlled_lanes = set()

            for lane_id in controlled_lanes:
                try:
                    vehicle_ids = (
                        self.sim.traci.lane.getLastStepVehicleIDs(lane_id)
                    )
                except Exception:
                    continue

                for raw_vehicle_id in vehicle_ids:
                    vehicle_id = str(raw_vehicle_id)
                    if vehicle_id.startswith(AMBULANCE_ID_PREFIX):
                        continue
                    speed = _safe_vehicle_value(
                        self.sim.traci.vehicle.getSpeed,
                        vehicle_id,
                    )
                    if speed <= 0.1:
                        queued_vehicle_ids.add(vehicle_id)

            queues[tls_id] = float(len(queued_vehicle_ids))

        return queues

    def _remember_interventions(
        self,
        actions: np.ndarray,
        base_actions: np.ndarray,
        tls_queues: dict[str, float],
    ) -> None:
        """Freeze a pre-intervention local baseline on real policy divergence."""

        for adapter, action, base_action in zip(
            self.adapters, actions, base_actions
        ):
            if int(action) == int(base_action):
                continue

            tls_id = str(adapter.tls_id)
            if tls_id in self._intervention_queue_baselines:
                continue

            fallback = float(tls_queues.get(tls_id, 0.0))
            self._intervention_queue_baselines[tls_id] = float(
                self._ordinary_queue_baselines.get(tls_id, fallback)
            )

    def _intervened_clearances(self, cleared_tls):
        """Keep at most one clearance per actually affected intersection."""

        filtered = []
        seen_tls: set[str] = set()
        for raw_ambulance_id, raw_tls_id in cleared_tls:
            tls_id = str(raw_tls_id)
            if (
                tls_id not in self._intervention_queue_baselines
                or tls_id in seen_tls
            ):
                continue
            seen_tls.add(tls_id)
            filtered.append((str(raw_ambulance_id), tls_id))
        return filtered

    def _start_recoveries(self, cleared_tls, sim_time: float) -> None:
        """Start or merge one local recovery event per affected TLS."""

        seen_tls: set[str] = set()
        for _ambulance_id, raw_tls_id in cleared_tls:
            tls_id = str(raw_tls_id)
            if tls_id in seen_tls:
                continue
            seen_tls.add(tls_id)

            baseline = self._intervention_queue_baselines.pop(tls_id, None)
            if baseline is None:
                continue

            threshold = max(
                float(baseline) + 1.0,
                1.05 * float(baseline),
            )
            recovery = self._pending_recoveries.get(tls_id)

            if recovery is None:
                self._pending_recoveries[tls_id] = {
                    "start": float(sim_time),
                    "threshold": float(threshold),
                }
            else:
                recovery["start"] = min(
                    float(recovery["start"]),
                    float(sim_time),
                )
                recovery["threshold"] = max(
                    float(recovery["threshold"]),
                    float(threshold),
                )

    def _advance_recoveries(
        self,
        tls_queues: dict[str, float],
        sim_time: float,
    ) -> None:
        completed_tls = []

        for tls_id, recovery in self._pending_recoveries.items():
            queue = float(tls_queues.get(tls_id, float("inf")))
            if (
                sim_time > recovery["start"]
                and queue <= recovery["threshold"]
            ):
                self._completed_recovery_seconds.append(
                    max(
                        0.0,
                        float(sim_time) - float(recovery["start"]),
                    )
                )
                completed_tls.append(tls_id)

        for tls_id in completed_tls:
            self._pending_recoveries.pop(tls_id, None)

    def _update_tls_queue_baselines(
        self,
        tls_queues: dict[str, float],
        active_tls: set[str],
    ) -> None:
        for raw_tls_id, raw_queue in tls_queues.items():
            tls_id = str(raw_tls_id)
            if (
                tls_id in active_tls
                or tls_id in self._pending_recoveries
                or tls_id in self._intervention_queue_baselines
            ):
                continue

            queue = float(raw_queue)
            previous = self._ordinary_queue_baselines.get(tls_id)
            self._ordinary_queue_baselines[tls_id] = (
                queue
                if previous is None
                else 0.95 * previous + 0.05 * queue
            )

    def _recovery_summary(self) -> dict[str, Any]:
        return {
            "mean_seconds": (
                statistics.fmean(
                    self._completed_recovery_seconds
                )
                if self._completed_recovery_seconds
                else 0.0
            ),
            "p95_seconds": _percentile(
                self._completed_recovery_seconds, 95.0
            ),
            "completed_events": len(
                self._completed_recovery_seconds
            ),
            "unrecovered_events": len(
                self._pending_recoveries
            ),
        }

    def finish_measurement_summary(
        self, sim_time: float | None = None
    ) -> dict[str, Any]:
        assert self.system is not None
        assert self.ordinary_monitor is not None
        self.system.finish_episode(sim_time)
        summary = self.system.summary()
        summary["ordinary_traffic"] = self.ordinary_monitor.summary(
            self.config.get("scheduled_ordinary_vehicles")
        )
        summary["recovery"] = self._recovery_summary()
        summary["signal_safety"] = {
            "invalid_policy_actions": sum(
                int(
                    adapter.controller.get(
                        "invalid_policy_actions", 0
                    )
                )
                for adapter in self.adapters
            ),
            "invalid_signal_transitions": sum(
                int(
                    adapter.controller.get(
                        "signal_transition_violations", 0
                    )
                )
                for adapter in self.adapters
            ),
            "emergency_override_switch_requests": sum(
                int(
                    adapter.controller.get(
                        "emergency_override_switch_requests", 0
                    )
                )
                for adapter in self.adapters
            ),
            "emergency_base_fallback_green_activations": sum(
                int(
                    adapter.controller.get(
                        "emergency_base_fallback_green_activations", 0
                    )
                )
                for adapter in self.adapters
            ),
            "emergency_base_fallback_failures": sum(
                int(
                    adapter.controller.get(
                        "emergency_base_fallback_failures", 0
                    )
                )
                for adapter in self.adapters
            ),
            "guarded_fallback_green_activations": sum(
                int(
                    adapter.controller.get(
                        "guarded_fallback_green_activations", 0
                    )
                )
                for adapter in self.adapters
            ),
            "blocked_green_activation_seconds": sum(
                float(
                    adapter.controller.get(
                        "blocked_green_activation_seconds", 0.0
                    )
                )
                for adapter in self.adapters
            ),
        }
        return summary

    def step_emergency(
        self,
        actions: np.ndarray,
        base_actions: np.ndarray,
        contexts,
        reset_on_done: bool,
    ):
        assert self.system is not None
        assert self.corridor is not None
        assert self.feature_builder is not None
        self.system.begin_decision()
        actions = np.asarray(actions, dtype=np.int64)
        base_actions = np.asarray(base_actions, dtype=np.int64)
        if actions.shape != (len(self.adapters),):
            raise ValueError("emergency action count does not match TLS count")
        if base_actions.shape != actions.shape:
            raise ValueError("base action count does not match final actions")

        pre_intervention_tls_queues = self.sample_ordinary_tls_queues()
        self._remember_interventions(
            actions,
            base_actions,
            pre_intervention_tls_queues,
        )
        self._decision_base_actions = {
            str(adapter.tls_id): int(base_action)
            for adapter, base_action in zip(self.adapters, base_actions)
        }
        try:
            traffic_rewards, done, sim_time = super().step(
                actions, reset_on_done=False
            )
        finally:
            self._decision_base_actions = {}
        # Mark unfinished and never-requested trips before closing this
        # decision window.  Their penalty must enter the terminal PPO reward,
        # rather than appearing only later in evaluation statistics.
        if done:
            self.system.finish_episode(sim_time)
        delta = self.system.end_decision()
        intervention_clearances = self._intervened_clearances(
            delta.cleared_tls
        )
        self.corridor.update_cleared(
            _ClearanceDeltaView(delta, intervention_clearances),
            sim_time,
        )
        emergency_reward, reward_components = emergency_rewards(
            system=self.system,
            delta=delta,
            contexts=contexts,
            traffic_rewards=traffic_rewards,
            decision_seconds=float(self.config["decision_seconds"]),
            base_actions=np.asarray(base_actions),
            final_actions=np.asarray(actions),
        )
        ordinary_metrics = self.sample_ordinary_network_metrics()
        tls_queues = self.sample_ordinary_tls_queues()

        self._start_recoveries(intervention_clearances, sim_time)
        self._advance_recoveries(tls_queues, sim_time)

        active_tls = {
            str(adapter.tls_id)
            for adapter, context in zip(self.adapters, contexts)
            if context.active_for_training
        }
        self._update_tls_queue_baselines(tls_queues, active_tls)
        finished_summary = None
        if done:
            finished_summary = self.finish_measurement_summary(
                sim_time
            )
        if done and reset_on_done:
            self.episode_index += 1
            self._start_episode()
        else:
            self.emergency_contexts = self.feature_builder.build(sim_time)
        return (
            emergency_reward,
            traffic_rewards,
            done,
            sim_time,
            delta,
            reward_components,
            ordinary_metrics,
            finished_summary,
        )


def _to_tensor_emergency(
    observations: list[Mapping[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = stack_emergency_observations(observations)
    return {
        key: torch.as_tensor(
            value, dtype=torch.float32, device=device
        )
        for key, value in batch.items()
    }


def _synchronize_corridor(
    episode: EmergencyAllTLSEpisode,
    contexts,
    sim_time: float,
):
    """Advance corridor timers before masking the current decision."""

    synchronized = []
    for context in contexts:
        state = episode.corridor.observe_context(context, sim_time)
        recovery = (
            state.mode == "recovery"
            and float(sim_time) < state.recovery_until
        )
        context.observation["emergency_global_features"][6] = float(
            recovery
        )
        context.observation["emergency_global_features"][7] = (
            episode.corridor.budget_remaining(context.tls_id)
        )
        context.observation["emergency_phase_features"][:, 6] = float(
            recovery
        )
        synchronized.append(
            replace(
                context,
                recovery_active=recovery,
                corridor_mode=state.mode,
                authority_available=(
                    episode.corridor.can_afford_preemption(
                        context.tls_id,
                        float(episode.config["decision_seconds"]),
                    )
                ),
            )
        )
    return synchronized


@torch.no_grad()
def _policy_actions(
    base_network: MovementGraphNetwork,
    override_network: EmergencyOverrideNetwork,
    base_observations: list[dict[str, np.ndarray]],
    emergency_observations: list[dict[str, np.ndarray]],
    masks: np.ndarray,
    override_exit_space_masks: np.ndarray,
    active: np.ndarray,
    authority: float,
    device: torch.device,
    deterministic: bool,
) -> dict[str, np.ndarray]:
    base_tensor = _tensor_observation(base_observations, device)
    emergency_tensor = _to_tensor_emergency(
        emergency_observations, device
    )
    base_logits, _base_values = base_network(base_tensor)
    tensor_masks = torch.as_tensor(
        masks, dtype=torch.bool, device=device
    )
    tensor_override_masks = torch.as_tensor(
        override_exit_space_masks,
        dtype=torch.bool,
        device=device,
    )
    if tensor_override_masks.shape != tensor_masks.shape:
        raise ValueError(
            "emergency exit-space mask shape differs from base mask"
        )
    masked_base = base_logits.masked_fill(~tensor_masks, -1e8)
    base_actions = masked_base.argmax(dim=-1)
    combined, values, _residual = override_network(
        base_tensor,
        emergency_tensor,
        masked_base,
        authority=authority,
    )

    effective_masks = tensor_masks & tensor_override_masks
    # Exact exit-space filtering applies only to actions that differ from the
    # frozen controller.  Its selected action is always restored as the
    # liveness-preserving fallback, even when that phase lacks an 18 m gap.
    effective_masks[
        torch.arange(len(base_actions), device=device),
        base_actions,
    ] = True
    inactive = ~torch.as_tensor(
        active, dtype=torch.bool, device=device
    )
    if inactive.any():
        effective_masks[inactive] = False
        effective_masks[
            torch.arange(len(base_actions), device=device)[inactive],
            base_actions[inactive],
        ] = True
    masked_combined = combined.masked_fill(~effective_masks, -1e8)
    distribution = Categorical(logits=masked_combined)
    actions = (
        masked_combined.argmax(dim=-1)
        if deterministic
        else distribution.sample()
    )
    return {
        "actions": actions.cpu().numpy().astype(np.int64),
        "base_actions": base_actions.cpu().numpy().astype(np.int64),
        "base_logits": masked_base.cpu().numpy().astype(np.float32),
        "log_probs": distribution.log_prob(actions)
        .cpu()
        .numpy()
        .astype(np.float32),
        "values": values.squeeze(-1)
        .cpu()
        .numpy()
        .astype(np.float32),
        "effective_masks": effective_masks.cpu().numpy().astype(bool),
    }


def evaluate_emergency_episode(
    episode: EmergencyAllTLSEpisode,
    base_network: MovementGraphNetwork,
    override_network: EmergencyOverrideNetwork,
    rollout_steps: int,
    authority: float,
    controller_mode: str,
    connection,
    progress_interval: int,
) -> dict[str, Any]:
    """Run one deterministic episode without returning training tensors."""

    if controller_mode not in {
        "learned",
        "base",
        "deterministic_preemption",
        "max_pressure",
        "native_sumo",
    }:
        raise ValueError(f"Unknown controller_mode={controller_mode!r}")
    device = torch.device("cpu")
    base_network.eval()
    override_network.eval()
    agents = len(episode.adapters)
    reward_total = 0.0
    traffic_reward_total = 0.0
    traffic_reward_count = 0
    active_transitions = 0
    actual_steps = 0
    finished_summary: dict[str, Any] | None = None

    for step in range(int(rollout_steps)):
        sim_time = float(episode.sim.traci.simulation.getTime())
        contexts = _synchronize_corridor(
            episode,
            list(episode.emergency_contexts),
            sim_time,
        )
        masks = episode.action_masks()
        active = np.asarray(
            [
                context.override_allowed
                for context in contexts
            ],
            dtype=bool,
        )
        override_exit_space_masks = (
            episode.emergency_override_exit_space_masks(masks, active)
        )
        policy = _policy_actions(
            base_network,
            override_network,
            episode.observations,
            [context.observation for context in contexts],
            masks,
            override_exit_space_masks,
            active,
            authority,
            device,
            deterministic=True,
        )
        recovery_fallbacks = normalized_max_pressure_actions(
            episode.observations,
            policy["effective_masks"],
        )
        standalone_max_pressure = normalized_max_pressure_actions(
            episode.observations,
            masks,
        )
        teacher_actions = np.asarray(
            [
                episode.corridor.teacher_action(
                    context,
                    (
                        int(recovery_action)
                        if context.recovery_active
                        else int(base_action)
                    ),
                    mask,
                    sim_time,
                )
                for context, base_action, recovery_action, mask in zip(
                    contexts,
                    policy["base_actions"],
                    recovery_fallbacks,
                    policy["effective_masks"],
                )
            ],
            dtype=np.int64,
        )
        if controller_mode == "base":
            actions = policy["base_actions"]
        elif controller_mode == "deterministic_preemption":
            actions = teacher_actions
        elif controller_mode == "max_pressure":
            # This is the same normalized MaxPressure/starvation heuristic used
            # by compare_native_sumo_vs_max_pressure.py in the schema-v3 final
            # campaign.  It is intentionally ambulance-unaware.  Use the full
            # ordinary action mask here, not the emergency residual mask (which
            # collapses to the frozen-base action outside an ambulance corridor).
            actions = standalone_max_pressure
        elif controller_mode == "native_sumo":
            # PersistentAllTLSEpisode.step() ignores actions in native mode and
            # leaves the .net.xml <tlLogic> under SUMO's control.
            actions = policy["base_actions"]
        else:
            actions = policy["actions"]

        (
            rewards,
            traffic_rewards,
            done,
            _new_sim_time,
            _delta,
            _components,
            _ordinary_metrics,
            completed,
        ) = episode.step_emergency(
            actions,
            policy["base_actions"],
            contexts,
            reset_on_done=False,
        )
        actual_steps += 1
        active_count = int(active.sum())
        active_transitions += active_count
        reward_total += float(np.asarray(rewards).sum())
        traffic_reward_total += float(
            np.asarray(traffic_rewards).sum()
        )
        traffic_reward_count += len(traffic_rewards)
        if completed is not None:
            finished_summary = completed
        if connection is not None and (
            actual_steps % max(1, progress_interval) == 0
            or done
            or actual_steps == rollout_steps
        ):
            connection.send(
                {
                    "type": "progress",
                    "step": actual_steps,
                    "total": rollout_steps,
                    "transitions": actual_steps * agents,
                    "active_transitions": active_transitions,
                    "tls": agents,
                }
            )
        if done:
            break

    if finished_summary is None:
        finished_summary = episode.finish_measurement_summary(
            float(episode.sim.traci.simulation.getTime())
        )
    ordinary_summary = dict(
        finished_summary.get("ordinary_traffic", {})
    )
    return {
        "type": "rollout",
        "net_file": episode.net_file,
        "tls_ids": episode.tls_ids,
        "metrics": {
            "net_file": episode.net_file,
            "transitions": actual_steps * agents,
            "active_transitions": active_transitions,
            "tls": agents,
            "mean_emergency_reward": (
                reward_total / max(1, active_transitions)
            ),
            "mean_traffic_reward": (
                traffic_reward_total
                / max(1, traffic_reward_count)
            ),
            "ordinary_arrived": int(
                ordinary_summary.get("arrived_total", 0)
            ),
            "ordinary_traffic": ordinary_summary,
            "recovery": finished_summary.get("recovery", {}),
            "ambulance": finished_summary,
            "controller_mode": controller_mode,
            "routing_mode": episode.system.config.routing_mode,
        },
    }


def collect_emergency_rollout(
    episode: EmergencyAllTLSEpisode,
    base_network: MovementGraphNetwork,
    override_network: EmergencyOverrideNetwork,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    authority: float,
    controller_mode: str,
    connection,
    progress_interval: int,
    deterministic: bool,
) -> dict[str, Any]:
    if controller_mode not in {
        "learned",
        "base",
        "deterministic_preemption",
        "max_pressure",
        "native_sumo",
    }:
        raise ValueError(f"Unknown controller_mode={controller_mode!r}")
    device = torch.device("cpu")
    base_network.eval()
    override_network.eval()
    agents = len(episode.adapters)
    dynamic = {key: [] for key in DYNAMIC_KEYS}
    emergency_dynamic = {
        "emergency_movements": [],
        "emergency_phase_features": [],
        "emergency_global_features": [],
    }
    static = {
        key: np.stack(
            [
                observation[key]
                for observation in episode.observations
            ],
            axis=0,
        )
        for key in STATIC_KEYS
    }
    action_masks_list = []
    actions_list = []
    base_actions_list = []
    base_logits_list = []
    old_log_probs_list = []
    values_list = []
    rewards_list = []
    traffic_rewards_list = []
    active_list = []
    teacher_actions_list = []
    dones_list = []
    ordinary_samples: list[dict[str, float]] = []
    completed_episode_summaries: list[dict[str, Any]] = []

    for step in range(int(rollout_steps)):
        sim_time = float(episode.sim.traci.simulation.getTime())
        contexts = _synchronize_corridor(
            episode,
            list(episode.emergency_contexts),
            sim_time,
        )
        observations = episode.observations
        masks = episode.action_masks()
        active = np.asarray(
            [
                context.override_allowed
                for context in contexts
            ],
            dtype=bool,
        )
        override_exit_space_masks = (
            episode.emergency_override_exit_space_masks(masks, active)
        )
        policy = _policy_actions(
            base_network,
            override_network,
            observations,
            [context.observation for context in contexts],
            masks,
            override_exit_space_masks,
            active,
            authority,
            device,
            deterministic,
        )
        recovery_fallbacks = normalized_max_pressure_actions(
            observations,
            policy["effective_masks"],
        )
        standalone_max_pressure = normalized_max_pressure_actions(
            observations,
            masks,
        )
        teacher_actions = np.asarray(
            [
                episode.corridor.teacher_action(
                    context,
                    (
                        int(recovery_action)
                        if context.recovery_active
                        else int(base_action)
                    ),
                    mask,
                    sim_time,
                )
                for context, base_action, recovery_action, mask in zip(
                    contexts,
                    policy["base_actions"],
                    recovery_fallbacks,
                    policy["effective_masks"],
                )
            ],
            dtype=np.int64,
        )
        if controller_mode == "base":
            actions = policy["base_actions"]
            effective_masks = np.zeros_like(masks)
            effective_masks[
                np.arange(agents), actions
            ] = True
            log_probs = np.zeros(agents, dtype=np.float32)
        elif controller_mode == "deterministic_preemption":
            actions = teacher_actions
            effective_masks = policy["effective_masks"]
            log_probs = np.zeros(agents, dtype=np.float32)
        elif controller_mode == "max_pressure":
            actions = standalone_max_pressure
            effective_masks = np.asarray(masks, dtype=bool)
            log_probs = np.zeros(agents, dtype=np.float32)
        elif controller_mode == "native_sumo":
            actions = policy["base_actions"]
            effective_masks = np.zeros_like(masks)
            effective_masks[np.arange(agents), actions] = True
            log_probs = np.zeros(agents, dtype=np.float32)
        else:
            actions = policy["actions"]
            effective_masks = policy["effective_masks"]
            log_probs = policy["log_probs"]

        for key in DYNAMIC_KEYS:
            dynamic[key].append(
                np.stack(
                    [observation[key] for observation in observations],
                    axis=0,
                )
            )
        for key in emergency_dynamic:
            emergency_dynamic[key].append(
                np.stack(
                    [context.observation[key] for context in contexts],
                    axis=0,
                )
            )

        (
            rewards,
            traffic_rewards,
            done,
            _new_sim_time,
            _delta,
            _components,
            ordinary_metrics,
            finished_summary,
        ) = episode.step_emergency(
            actions,
            policy["base_actions"],
            contexts,
            reset_on_done=not deterministic,
        )
        if finished_summary is not None:
            completed_episode_summaries.append(finished_summary)
        ordinary_samples.append(ordinary_metrics)
        action_masks_list.append(effective_masks)
        actions_list.append(actions)
        base_actions_list.append(policy["base_actions"])
        base_logits_list.append(policy["base_logits"])
        old_log_probs_list.append(log_probs)
        values_list.append(policy["values"])
        rewards_list.append(rewards)
        traffic_rewards_list.append(traffic_rewards)
        active_list.append(active.astype(np.float32))
        teacher_actions_list.append(teacher_actions)
        dones_list.append(
            np.full(agents, done, dtype=np.float32)
        )
        if connection is not None and (
            (step + 1) % max(1, progress_interval) == 0
            or step + 1 == rollout_steps
        ):
            connection.send(
                {
                    "type": "progress",
                    "step": step + 1,
                    "total": rollout_steps,
                    "transitions": (step + 1) * agents,
                    "active_transitions": int(
                        np.asarray(active_list).sum()
                    ),
                    "tls": agents,
                }
            )
        if done and deterministic:
            break

    actual_steps = len(actions_list)
    final_contexts = _synchronize_corridor(
        episode,
        list(episode.emergency_contexts),
        float(episode.sim.traci.simulation.getTime()),
    )
    final_masks = episode.action_masks()
    final_active = np.asarray(
        [
            context.active_for_training
            for context in final_contexts
        ],
        dtype=bool,
    )
    final_override_exit_space_masks = (
        episode.emergency_override_exit_space_masks(
            final_masks, final_active
        )
    )
    final_policy = _policy_actions(
        base_network,
        override_network,
        episode.observations,
        [context.observation for context in final_contexts],
        final_masks,
        final_override_exit_space_masks,
        final_active,
        authority,
        device,
        deterministic=True,
    )
    values_array = np.asarray(values_list, dtype=np.float32)
    rewards_array = np.asarray(rewards_list, dtype=np.float32)
    dones_array = np.asarray(dones_list, dtype=np.float32)
    advantages = np.zeros_like(rewards_array)
    last_gae = np.zeros(agents, dtype=np.float32)
    next_values = final_policy["values"]
    for step in reversed(range(actual_steps)):
        nonterminal = 1.0 - dones_array[step]
        delta = (
            rewards_array[step]
            + float(gamma) * next_values * nonterminal
            - values_array[step]
        )
        last_gae = (
            delta
            + float(gamma)
            * float(gae_lambda)
            * nonterminal
            * last_gae
        )
        advantages[step] = last_gae
        next_values = values_array[step]
    returns = advantages + values_array

    if deterministic and episode.system is not None:
        episode.system.finish_episode()
        if not completed_episode_summaries:
            completed_episode_summaries.append(
                episode.system.summary()
            )
    ambulance_summary = (
        completed_episode_summaries[-1]
        if completed_episode_summaries
        else episode.system.summary()
    )
    ordinary = {
        key: float(
            np.mean([sample[key] for sample in ordinary_samples])
        )
        for key in ordinary_samples[0]
    }
    ordinary_summary = dict(
        ambulance_summary.get("ordinary_traffic", {})
    )
    if not ordinary_summary and episode.ordinary_monitor is not None:
        ordinary_summary = episode.ordinary_monitor.summary(
            episode.config.get("scheduled_ordinary_vehicles")
        )
    ordinary_arrived = int(
        ordinary_summary.get("arrived_total", 0)
    )
    active_array = np.asarray(active_list, dtype=np.float32)
    return {
        "type": "rollout",
        "net_file": episode.net_file,
        "tls_ids": episode.tls_ids,
        "dynamic": {
            key: np.asarray(value, dtype=np.float16)
            for key, value in dynamic.items()
        },
        "emergency_dynamic": {
            key: np.asarray(value, dtype=np.float16)
            for key, value in emergency_dynamic.items()
        },
        "static": {
            "movement_mask": static["movement_mask"].astype(np.uint8),
            "movement_adjacency": static[
                "movement_adjacency"
            ].astype(np.uint8),
            "phase_membership": static[
                "phase_membership"
            ].astype(np.float16),
        },
        "action_masks": np.asarray(
            action_masks_list, dtype=np.uint8
        ),
        "actions": np.asarray(actions_list, dtype=np.int16),
        "base_actions": np.asarray(
            base_actions_list, dtype=np.int16
        ),
        "base_logits": np.asarray(
            base_logits_list, dtype=np.float32
        ),
        "old_log_probs": np.asarray(
            old_log_probs_list, dtype=np.float32
        ),
        "old_values": values_array,
        "advantages": advantages,
        "returns": returns,
        "teacher_actions": np.asarray(
            teacher_actions_list, dtype=np.int16
        ),
        "active": active_array,
        "agent_weights": episode.sample_weights(),
        "metrics": {
            "net_file": episode.net_file,
            "transitions": actual_steps * agents,
            "active_transitions": int(active_array.sum()),
            "tls": agents,
            "mean_emergency_reward": float(
                rewards_array.sum()
                / max(1.0, active_array.sum())
            ),
            "mean_traffic_reward": float(
                np.asarray(traffic_rewards_list).mean()
            ),
            "ordinary_arrived": ordinary_arrived,
            "ordinary_traffic": ordinary_summary,
            "recovery": ambulance_summary.get("recovery", {}),
            **ordinary,
            "ambulance": ambulance_summary,
            "controller_mode": controller_mode,
            "routing_mode": episode.system.config.routing_mode,
        },
    }


def emergency_rollout_worker_main(
    connection, config: dict[str, Any]
) -> None:
    episode = None
    try:
        os.environ["TRAFFIC_NET_FILE"] = str(config["net_file"])
        os.environ["MAP_AGNOSTIC_MAX_ACTIVE_CAP"] = str(
            int(config["max_vehicle_center"])
        )
        if config.get("use_libsumo"):
            os.environ["SUMO_USE_LIBSUMO"] = "1"
        torch.set_num_threads(1)
        base_network = MovementGraphNetwork(
            embed_dim=int(config["base_embed_dim"]),
            graph_layers=int(config["base_graph_layers"]),
        ).cpu()
        override_network = EmergencyOverrideNetwork(
            embed_dim=int(config["emergency_embed_dim"]),
            graph_layers=int(config["emergency_graph_layers"]),
            residual_bound=float(config["residual_bound"]),
        ).cpu()
        episode = EmergencyAllTLSEpisode(config)
        connection.send(
            {
                "type": "ready",
                "net_file": config["net_file"],
                "tls": len(episode.adapters),
                "tls_ids": episode.tls_ids,
                "schedule_sha256": episode.system.schedule_hash,
                "sumo_error_log": str(
                    config.get("sumo_error_log", "")
                ),
            }
        )
        while True:
            request = connection.recv()
            command = request.get("cmd")
            if command == "close":
                break
            if command != "rollout":
                raise ValueError(
                    f"Unknown worker command: {command}"
                )
            base_network.load_state_dict(
                {
                    key: torch.as_tensor(value)
                    for key, value in request[
                        "base_state_dict"
                    ].items()
                }
            )
            override_network.load_state_dict(
                {
                    key: torch.as_tensor(value)
                    for key, value in request[
                        "override_state_dict"
                    ].items()
                }
            )
            if bool(request.get("metrics_only", False)):
                result = evaluate_emergency_episode(
                    episode=episode,
                    base_network=base_network,
                    override_network=override_network,
                    rollout_steps=int(request["rollout_steps"]),
                    authority=float(request["authority"]),
                    controller_mode=str(
                        request.get("controller_mode", "learned")
                    ),
                    connection=connection,
                    progress_interval=int(
                        request["progress_interval"]
                    ),
                )
            else:
                result = collect_emergency_rollout(
                    episode=episode,
                    base_network=base_network,
                    override_network=override_network,
                    rollout_steps=int(request["rollout_steps"]),
                    gamma=float(request["gamma"]),
                    gae_lambda=float(request["gae_lambda"]),
                    authority=float(request["authority"]),
                    controller_mode=str(
                        request.get("controller_mode", "learned")
                    ),
                    connection=connection,
                    progress_interval=int(
                        request["progress_interval"]
                    ),
                    deterministic=bool(
                        request.get("deterministic", False)
                    ),
                )
            connection.send(result)
    except BaseException as exc:
        try:
            sumo_error_log = str(
                getattr(
                    getattr(
                        getattr(episode, "episode", None),
                        "outer_args",
                        None,
                    ),
                    "sumo_error_log",
                    config.get("sumo_error_log", ""),
                )
            )
            connection.send(
                {
                    "type": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "sumo_error_log": sumo_error_log,
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
    "EmergencyAllTLSEpisode",
    "collect_emergency_rollout",
    "emergency_rollout_worker_main",
    "evaluate_emergency_episode",
]
