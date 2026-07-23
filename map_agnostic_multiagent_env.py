#!/usr/bin/env python3
"""Synchronized all-traffic-light environment for map-balanced PPO rollouts.

One instance owns one SUMO map.  Every compatible signal is observed, acted on,
and advanced in the same decision step, eliminating the old one-TLS-training /
all-TLS-deployment mismatch.  It is intentionally a lightweight batch API,
not a Gym environment: the number of signals differs by map.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

import compare_fixed_vs_single_vs_all_model_realistic as legacy
from map_agnostic_tls import (
    MAX_PHASES,
    MapAgnosticTLSAdapter,
    ObservationSnapshot,
    adapter_for_controller,
    confirmed_stopline_crossings,
    normalized_reward,
    stack_observations,
)
from traffic_rl_map_agnostic_env import configure_network, sim


@dataclass(frozen=True)
class MultiAgentRewardMix:
    local: float = 0.70
    neighbor: float = 0.20
    network: float = 0.10

    def __post_init__(self) -> None:
        values = (float(self.local), float(self.neighbor), float(self.network))
        if any(value < 0.0 for value in values):
            raise ValueError("Reward mixture weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Reward mixture weights must sum to 1")


@dataclass
class MultiAgentStep:
    tls_ids: tuple[str, ...]
    observations: dict[str, np.ndarray]
    action_masks: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def build_signal_neighbors(
    adapters: Sequence[MapAgnosticTLSAdapter],
) -> tuple[tuple[int, ...], ...]:
    """Connect signals whose controlled movement edges meet directly.

    SUMO edges normally run from one junction to the next, so an outgoing edge
    at signal A is an incoming edge at downstream signal B.  The relationship
    is made undirected for regional reward sharing; the observation still
    preserves downstream occupancy direction locally.
    """

    incoming_owners: dict[str, set[int]] = {}
    for index, adapter in enumerate(adapters):
        for movement in adapter.topology.movements:
            incoming_owners.setdefault(movement.incoming_edge, set()).add(index)

    neighbors: list[set[int]] = [set() for _ in adapters]
    for source, adapter in enumerate(adapters):
        for movement in adapter.topology.movements:
            for target in incoming_owners.get(movement.outgoing_edge, set()):
                if target == source:
                    continue
                neighbors[source].add(target)
                neighbors[target].add(source)
    return tuple(tuple(sorted(values)) for values in neighbors)


class MapAgnosticAllTLSEnv:
    """One entire SUMO map controlled by a shared per-intersection actor."""

    def __init__(
        self,
        *,
        net_file: str | Path,
        episode_seconds: int = 900,
        seed: int = 42,
        max_vehicle_center: int = 1500,
        target_vehicle_center: int = 1200,
        initial_vehicle_center: int = 300,
        spawn_batch_center: int = 20,
        green_duration_center: float = 30.0,
        reward_mix: MultiAgentRewardMix = MultiAgentRewardMix(),
        fixed_demand_route: str | Path | None = None,
        fixed_demand_scheduled_total: int = 0,
        gui: bool = False,
    ):
        self.net_file = configure_network(net_file)
        sim.USE_MAP_AGNOSTIC_PHASE_CATALOG = True
        self.episode_seconds = int(episode_seconds)
        self.seed = int(seed)
        self.reward_mix = reward_mix
        self.outer_args = SimpleNamespace(
            episode_seconds=self.episode_seconds,
            gui=bool(gui),
            max_vehicle_center=int(max_vehicle_center),
            target_vehicle_center=int(target_vehicle_center),
            initial_vehicle_center=int(initial_vehicle_center),
            spawn_batch_center=int(spawn_batch_center),
            green_duration_center=float(green_duration_center),
            density_spread=0.0,
            initial_spread=0.0,
            fixed_demand_route=(str(Path(fixed_demand_route).resolve()) if fixed_demand_route else ""),
            fixed_demand_scheduled_total=int(fixed_demand_scheduled_total),
        )
        self.episode: legacy.AnchorlessSimulationEpisode | None = None
        self.controllers: list[dict[str, Any]] = []
        self.adapters: list[MapAgnosticTLSAdapter] = []
        self.snapshots: list[ObservationSnapshot] = []
        self.neighbors: tuple[tuple[int, ...], ...] = ()
        self.invalid_action_count = 0
        self.policy_decisions = 0
        self.policy_switches = 0
        self.policy_forced_switches = 0
        self._previous_network_queue_fraction = 0.0
        self._started = False

    @property
    def tls_ids(self) -> tuple[str, ...]:
        return tuple(adapter.tls_id for adapter in self.adapters)

    def _network_state(self) -> tuple[float, float, float, int]:
        global_wait, global_queue, avg_speed = legacy.network_wait_queue_speed()
        try:
            active = int(sim.traci.vehicle.getIDCount())
        except Exception:
            active = 0
        queue_fraction = float(np.clip(global_queue / max(1.0, active), 0.0, 1.0))
        wait_ratio = float(
            np.clip(np.log1p(max(0.0, global_wait / max(1.0, active))) / np.log1p(300.0), 0.0, 1.0)
        )
        speed_ratio = float(np.clip(avg_speed / 13.9, 0.0, 1.0))
        return queue_fraction, wait_ratio, speed_ratio, active

    def _batched_observations(self) -> dict[str, np.ndarray]:
        return stack_observations([snapshot.observation for snapshot in self.snapshots])

    def action_masks(self) -> np.ndarray:
        if not self.adapters:
            return np.zeros((0, MAX_PHASES + 1), dtype=bool)
        return np.stack(
            [
                adapter.action_mask(
                    min_green=legacy.MIN_GREEN_BEFORE_SWITCH,
                    max_green=legacy.HARD_MAX_GREEN,
                )
                for adapter in self.adapters
            ]
        ).astype(bool)

    def reset(self) -> MultiAgentStep:
        self.close()
        configure_network(self.net_file)
        scenario = legacy.build_fixed_scenario(seed=self.seed, args=self.outer_args)
        self.episode = legacy.AnchorlessSimulationEpisode(
            scenario=scenario,
            seed=self.seed,
            args=self.outer_args,
            gui=bool(self.outer_args.gui),
            env_rank=0,
        )
        self.episode.reset()
        self._started = True
        assert self.episode.args is not None
        self.episode.args.disable_ambulances = True
        self.episode.sim_state["next_ambulance_spawn"] = float("inf")
        self.episode.sim_state["active_ambulances"] = {}

        self.controllers = [
            controller
            for controller in self.episode.controllers
            if not controller.get("disabled")
        ]
        self.adapters = []
        compatible_controllers: list[dict[str, Any]] = []
        for controller in self.controllers:
            try:
                adapter = adapter_for_controller(controller, sim.traci, sim)
                if adapter.phase_count < 2:
                    continue
                adapter.reset_history()
                self.adapters.append(adapter)
                compatible_controllers.append(controller)
            except Exception as exc:
                print(f"[multiagent] leaving TLS {controller.get('tls_id')} on fixed timing: {exc}")
        self.controllers = compatible_controllers
        if not self.adapters:
            raise RuntimeError(f"No compatible map-agnostic TLS found in {self.net_file}")
        self.snapshots = [adapter.observe(update_history=True) for adapter in self.adapters]
        self.neighbors = build_signal_neighbors(self.adapters)
        self.invalid_action_count = 0
        self.policy_decisions = 0
        self.policy_switches = 0
        self.policy_forced_switches = 0
        self._previous_network_queue_fraction = self._network_state()[0]
        return MultiAgentStep(
            tls_ids=self.tls_ids,
            observations=self._batched_observations(),
            action_masks=self.action_masks(),
            rewards=np.zeros(len(self.adapters), dtype=np.float32),
            terminated=False,
            truncated=False,
            info=self._info(),
        )

    def _apply_action(
        self,
        controller: dict[str, Any],
        adapter: MapAgnosticTLSAdapter,
        requested_action: int,
        mask: np.ndarray,
    ) -> tuple[bool, bool]:
        action = int(requested_action)
        self.policy_decisions += 1
        if action < 0 or action >= len(mask) or not mask[action]:
            self.invalid_action_count += 1
            action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

        elapsed = float(controller.get("phase_elapsed", 0.0))
        if action == 0:
            if (
                controller.get("mode") == "green"
                and elapsed >= legacy.HARD_MAX_GREEN
                and not mask[1:].any()
            ):
                switched = bool(sim.switch_next_fixed_phase(controller))
                return switched, True
            return False, False

        phase_pos = adapter.action_to_phase_position(action)
        if (
            phase_pos is None
            or controller.get("mode") != "green"
            or elapsed < legacy.MIN_GREEN_BEFORE_SWITCH
        ):
            return False, False
        return bool(sim.request_switch(controller, phase_pos)), False

    def step(self, actions: Sequence[int] | np.ndarray) -> MultiAgentStep:
        if self.episode is None or not self._started:
            raise RuntimeError("Call reset() before step().")
        action_array = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(action_array) != len(self.adapters):
            raise ValueError(
                f"Expected {len(self.adapters)} TLS actions, got {len(action_array)}"
            )

        before = list(self.snapshots)
        masks = self.action_masks()
        controlled_object_ids = {id(controller) for controller in self.controllers}
        for controller in self.episode.controllers:
            if id(controller) in controlled_object_ids or controller.get("disabled"):
                continue
            if (
                controller.get("mode") == "green"
                and float(controller.get("phase_elapsed", 0.0))
                >= float(controller.get("green_duration", 30.0))
            ):
                sim.switch_next_fixed_phase(controller)
        switched_flags: list[bool] = []
        forced_flags: list[bool] = []
        for controller, adapter, action, mask in zip(
            self.controllers, self.adapters, action_array, masks
        ):
            switched, forced = self._apply_action(controller, adapter, int(action), mask)
            switched_flags.append(switched)
            forced_flags.append(forced)
            self.policy_switches += int(switched)
            self.policy_forced_switches += int(forced)

        decision_steps = max(1, int(round(sim.DECISION_INTERVAL / sim.STEP_LENGTH)))
        arrived, spawned, extended, recovered = sim.run_simulation_steps(
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
        self.snapshots = [adapter.observe(update_history=True) for adapter in self.adapters]

        local_rewards = np.zeros(len(self.adapters), dtype=np.float64)
        crossing_totals = np.zeros(len(self.adapters), dtype=np.int64)
        for index, (adapter, old, new, switched, forced) in enumerate(
            zip(self.adapters, before, self.snapshots, switched_flags, forced_flags)
        ):
            crossings = confirmed_stopline_crossings(old, new, adapter.topology)
            crossing_totals[index] = crossings.total
            local_rewards[index], _components = normalized_reward(
                previous=old,
                current=new,
                local_cleared=crossings.total,
                decision_seconds=decision_steps * sim.STEP_LENGTH,
                switched=switched,
                forced=forced,
            )

        neighbor_rewards = np.asarray(
            [
                float(np.mean(local_rewards[list(neighbors)]))
                if neighbors
                else float(local_rewards[index])
                for index, neighbors in enumerate(self.neighbors)
            ],
            dtype=np.float64,
        )
        queue_fraction, wait_ratio, speed_ratio, active = self._network_state()
        queue_improvement = float(
            np.clip(self._previous_network_queue_fraction - queue_fraction, -1.0, 1.0)
        )
        self._previous_network_queue_fraction = queue_fraction
        network_reward = float(
            np.clip(
                0.75 * queue_improvement
                - 0.50 * queue_fraction
                - 0.20 * wait_ratio
                + 0.20 * speed_ratio,
                -1.0,
                1.0,
            )
        )
        rewards = (
            self.reward_mix.local * local_rewards
            + self.reward_mix.neighbor * neighbor_rewards
            + self.reward_mix.network * network_reward
        ).astype(np.float32)

        sim_time = float(sim.traci.simulation.getTime())
        truncated = sim_time >= float(self.episode_seconds)
        return MultiAgentStep(
            tls_ids=self.tls_ids,
            observations=self._batched_observations(),
            action_masks=self.action_masks(),
            rewards=rewards,
            terminated=False,
            truncated=bool(truncated),
            info=self._info(
                arrived=int(arrived),
                spawned=int(spawned),
                extended=int(extended),
                recovered=int(recovered),
                confirmed_crossings=int(crossing_totals.sum()),
                network_reward=network_reward,
                queue_fraction=queue_fraction,
                wait_ratio=wait_ratio,
                speed_ratio=speed_ratio,
                active_vehicles=active,
            ),
        )

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "net_file": str(self.net_file),
            "tls_count": len(self.adapters),
            "policy_decisions": self.policy_decisions,
            "policy_switches": self.policy_switches,
            "policy_forced_switches": self.policy_forced_switches,
            "invalid_actions": self.invalid_action_count,
            "controlled_tls_fraction": (
                len(self.adapters) / max(1, len(self.episode.controllers))
                if self.episode is not None
                else 0.0
            ),
            "total_arrived": self.episode.total_arrived if self.episode is not None else 0,
        }
        try:
            info["sim_time"] = float(sim.traci.simulation.getTime())
        except Exception:
            info["sim_time"] = 0.0
        info.update(extra)
        return info

    def close(self) -> None:
        if self.episode is not None:
            self.episode.close()
        self.episode = None
        self.controllers = []
        self.adapters = []
        self.snapshots = []
        self.neighbors = ()
        self._started = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test synchronized all-TLS control.")
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from safe_residual_controller import SafeResidualController

    controller = SafeResidualController(residual_authority=0.0)
    with MapAgnosticAllTLSEnv(net_file=args.net_file, seed=args.seed) as env:
        state = env.reset()
        for _ in range(args.steps):
            actions = [
                controller.select_action(
                    {key: value[index] for key, value in state.observations.items()},
                    state.action_masks[index],
                ).action
                for index in range(len(state.tls_ids))
            ]
            state = env.step(actions)
            print(state.info)
            if state.truncated:
                break


if __name__ == "__main__":
    main()
