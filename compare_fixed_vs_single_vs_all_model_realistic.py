#!/usr/bin/env python3
"""
Evaluate a trained MaskablePPO traffic-signal controller through the exact
realistic_all_intersections_fixed_cycle.py simulation settings.

This training script imports the current simulation file directly.  This file is intended for GUI evaluation of the fast-proxy-trained model inside
the full realistic simulation.  It imports the simulation module directly and
uses the same default settings from realistic_all_intersections_fixed_cycle.py:

- OD fastest-route vehicle generation
- no last-100m lane changes before intersections
- traffic-light no-lane-change zone
- right-turn permissive clearance behavior
- keep-clear / right-of-way logic
- unconnected-lane rescue
- unjustified-stop watchdog
- lane balancing
- full-map geographic spawning

The RL policy controls one selected traffic light.  All other traffic lights use
the same fixed-cycle controller from the simulation file.

Typical training command:
    python train_traffic_signal_rl.py \
      --timesteps 750000 \
      --episode-seconds 900 \
      --model-path models/traffic_signal_maskable_ppo_exact_sim \
      --no-resume

Typical GUI evaluation command:
    python train_traffic_signal_rl.py \
      --eval-only \
      --gui \
      --eval-fixed-scenario \
      --episode-seconds 3600 \
      --model-path models/traffic_signal_maskable_ppo_exact_sim
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing Gym dependency. Install with:\n"
        "python -m pip install gymnasium"
    ) from exc

try:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing RL dependencies. Install them with:\n"
        "python -m pip install gymnasium stable-baselines3 sb3-contrib tensorboard tqdm rich"
    ) from exc

try:
    import realistic_all_intersections_fixed_cycle as sim
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import realistic_all_intersections_fixed_cycle.py.\n"
        "Put this training file in the same folder as the current simulation file."
    ) from exc


TARGET_TLS_ID = "cluster_12179861947_12179861948_12179861949_12185616643_#11more"


def load_maskable_ppo_for_inference(model_path: str, env, device: str):
    """Load a MaskablePPO checkpoint for inference only.

    Checkpoints saved with a non-constant learning-rate/clip-range schedule
    (a Python callable) get cloudpickled together with a reference to the
    module/file path of whoever trained the model. If that checkpoint is
    loaded on a different machine/user account (e.g. a teammate's laptop),
    SB3 still tries to reconstruct that pickled callable during
    `_setup_model()`, which has been observed to crash natively inside
    PyTorch rather than raising a catchable Python exception.

    Since we only need this model for inference (action selection), not to
    resume training, we override those schedule-related fields with inert
    constants. This avoids ever touching the original pickled closures.
    """
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _progress_remaining: 0.0,
        "clip_range": lambda _progress_remaining: 0.2,
    }
    return MaskablePPO.load(
        str(model_path),
        env=env,
        device=device,
        custom_objects=custom_objects,
    )
MODEL_DEFAULT = "models/traffic_signal_maskable_ppo_fast_proxy_strong"

# Center values match the simulation command you have been using for the current
# file.  Training randomizes around these, but evaluation can lock to them.
SIM_CENTER_MAX_VEHICLES = int(getattr(sim, "MAX_ACTIVE_VEHICLE_CAP", 750))
SIM_CENTER_TARGET_VEHICLES = 650
SIM_CENTER_INITIAL_VEHICLES = 200
SIM_CENTER_SPAWN_BATCH = 12
SIM_CENTER_ROUTE_LOOKAHEAD = 60
SIM_CENTER_MIN_REMAINING = 15
SIM_CENTER_GREEN_DURATION = 30.0
SIM_CENTER_SPAWN_GRID_SIZE = 6
SIM_CENTER_TELEPORT = 180
SIM_CENTER_MAX_DEPART_DELAY = 300

# The current simulation logic should already define these, but keep safe
# fallbacks so the training file fails less mysteriously on older copies.
SIM_CENTER_INTERSECTION_NO_LC = float(getattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE", 100.0))
SIM_CENTER_INTERSECTION_PREP = float(getattr(sim, "INTERSECTION_LANE_PREP_DISTANCE", 320.0))
SIM_CENTER_TLS_NO_LC = float(getattr(sim, "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE", 100.0))
SIM_CENTER_TLS_PREP = float(getattr(sim, "TRAFFIC_LIGHT_LANE_PREP_DISTANCE", 320.0))

ROUTING_MODE_OD = getattr(sim, "ROUTING_MODE_OD", "od")
OD_DEPART_LANE = getattr(sim, "OD_DEPART_LANE", "free")

# Action timing. The agent can switch after a minimum green, but it is punished
# and eventually forced if it starves a phase for too long.
MIN_GREEN_BEFORE_SWITCH = 10.0
SOFT_MAX_GREEN = 75.0
HARD_MAX_GREEN = 95.0

# Reward scaling. Values are deliberately moderate because raw SUMO wait-time
# can be very large and noisy.
TARGET_WAIT_DELTA_SCALE = 220.0
TARGET_QUEUE_DELTA_SCALE = 14.0
GLOBAL_WAIT_DELTA_SCALE = 950.0
GLOBAL_QUEUE_DELTA_SCALE = 55.0
TARGET_WAIT_LEVEL_SCALE = 4200.0
TARGET_QUEUE_LEVEL_SCALE = 95.0
GLOBAL_WAIT_LEVEL_SCALE = 28000.0
GLOBAL_QUEUE_LEVEL_SCALE = 500.0
SWITCH_PENALTY = 0.04
FORCED_SWITCH_PENALTY = 0.35
RECOVERY_PENALTY = 0.015
ARRIVAL_BONUS = 0.025
STARVATION_PENALTY_SCALE = 0.015


@dataclass(frozen=True)
class TrafficScenario:
    seed: int
    max_vehicles: int
    target_vehicles: int
    initial_vehicles: int
    spawn_batch: int
    route_lookahead_edges: int
    min_remaining_edges: int
    green_duration: float
    signal_timing_jitter: float
    spawn_grid_size: int
    max_depart_delay: int
    time_to_teleport: int
    local_road_penalty: float
    local_to_local_penalty: float
    leave_local_bonus: float
    non_core_penalty: float
    routing_mode: str
    od_route_attempts: int
    od_boundary_margin_fraction: float
    od_min_euclidean_distance: float
    od_min_route_distance: float
    od_min_zone_separation: int
    od_max_local_middle_fraction: float
    od_local_middle_trim_edges: int
    od_through_trip_probability: float
    od_access_trip_probability: float
    od_long_local_trip_probability: float
    od_min_edge_length: float
    od_random_walk_fallback: bool
    depart_lane: str
    intersection_no_lane_change_distance: float
    intersection_lane_prep_distance: float
    tls_no_lane_change_distance: float
    tls_lane_prep_distance: float
    unjustified_stop_watchdog: bool
    unjustified_stop_check_interval: float
    unjustified_stop_speed: float
    unjustified_stop_min_time: float
    disable_strict_split: bool


class ScenarioSampler:
    """Randomizes density around the current simulation, not around old training defaults."""

    def __init__(
        self,
        base_seed: Optional[int],
        max_vehicle_center: int,
        target_center: int,
        initial_center: int,
        spawn_batch_center: int,
        green_duration_center: float,
        density_spread: float,
        initial_spread: float,
    ):
        self.rng = random.Random(base_seed if base_seed is not None else time.time_ns())
        self.max_vehicle_center = min(max_vehicle_center, SIM_CENTER_MAX_VEHICLES)
        self.target_center = min(target_center, self.max_vehicle_center)
        self.initial_center = min(initial_center, self.target_center)
        self.spawn_batch_center = spawn_batch_center
        self.green_duration_center = green_duration_center
        self.density_spread = max(0.0, density_spread)
        self.initial_spread = max(0.0, initial_spread)

    def _jitter_int(self, center: int, spread: float, lo: int, hi: int) -> int:
        if hi < lo:
            hi = lo
        sigma = max(1.0, abs(center) * spread / 2.0)
        value = round(self.rng.gauss(center, sigma))
        return int(max(lo, min(hi, value)))

    def _jitter_float(self, center: float, spread: float, lo: float, hi: float) -> float:
        if hi < lo:
            hi = lo
        sigma = max(0.01, abs(center) * spread / 2.0)
        value = self.rng.gauss(center, sigma)
        return float(max(lo, min(hi, value)))

    def sample(self) -> TrafficScenario:
        # Vary traffic intensity and cap around the simulation values.  The hard
        # cap remains the simulation hard cap, not an old 1000-car default.
        max_lo = max(350, int(self.max_vehicle_center * (1.0 - self.density_spread)))
        max_hi = self.max_vehicle_center
        max_vehicles = self._jitter_int(self.max_vehicle_center, self.density_spread, max_lo, max_hi)

        target_lo = max(250, int(self.target_center * (1.0 - self.density_spread)))
        target_hi = min(max_vehicles, int(self.target_center * (1.0 + self.density_spread)))
        target_vehicles = self._jitter_int(self.target_center, self.density_spread, target_lo, target_hi)
        target_vehicles = min(target_vehicles, max_vehicles)

        initial_lo = max(40, int(self.initial_center * (1.0 - self.initial_spread)))
        initial_hi = min(target_vehicles, int(self.initial_center * (1.0 + self.initial_spread)))
        initial_vehicles = self._jitter_int(self.initial_center, self.initial_spread, initial_lo, initial_hi)
        initial_vehicles = min(initial_vehicles, target_vehicles)

        spawn_batch = self._jitter_int(self.spawn_batch_center, 0.55, 6, 28)
        route_lookahead = self._jitter_int(SIM_CENTER_ROUTE_LOOKAHEAD, 0.20, 45, 75)
        min_remaining = self._jitter_int(SIM_CENTER_MIN_REMAINING, 0.30, 8, max(9, route_lookahead // 2))
        green_duration = self._jitter_float(self.green_duration_center, 0.12, 34.0, 55.0)
        signal_jitter = self._jitter_float(0.15, 0.40, 0.05, 0.25)

        return TrafficScenario(
            seed=self.rng.randint(1, 2_000_000_000),
            max_vehicles=max_vehicles,
            target_vehicles=target_vehicles,
            initial_vehicles=initial_vehicles,
            spawn_batch=spawn_batch,
            route_lookahead_edges=route_lookahead,
            min_remaining_edges=min_remaining,
            green_duration=green_duration,
            signal_timing_jitter=signal_jitter,
            spawn_grid_size=SIM_CENTER_SPAWN_GRID_SIZE,
            max_depart_delay=SIM_CENTER_MAX_DEPART_DELAY,
            time_to_teleport=SIM_CENTER_TELEPORT,
            local_road_penalty=0.04,
            local_to_local_penalty=0.15,
            leave_local_bonus=8.0,
            non_core_penalty=1.0,
            routing_mode=ROUTING_MODE_OD,
            od_route_attempts=int(getattr(sim, "OD_ROUTE_ATTEMPTS", 120)),
            od_boundary_margin_fraction=float(getattr(sim, "OD_BOUNDARY_MARGIN_FRACTION", 0.13)),
            od_min_euclidean_distance=float(getattr(sim, "OD_MIN_EUCLIDEAN_DISTANCE", 900.0)),
            od_min_route_distance=float(getattr(sim, "OD_MIN_ROUTE_DISTANCE", 1200.0)),
            od_min_zone_separation=int(getattr(sim, "OD_MIN_ZONE_SEPARATION", 2)),
            od_max_local_middle_fraction=float(getattr(sim, "OD_MAX_LOCAL_MIDDLE_FRACTION", 0.35)),
            od_local_middle_trim_edges=int(getattr(sim, "OD_LOCAL_MIDDLE_TRIM_EDGES", 2)),
            od_through_trip_probability=float(getattr(sim, "OD_THROUGH_TRIP_PROBABILITY", 0.72)),
            od_access_trip_probability=float(getattr(sim, "OD_ACCESS_TRIP_PROBABILITY", 0.23)),
            od_long_local_trip_probability=float(getattr(sim, "OD_LONG_LOCAL_TRIP_PROBABILITY", 0.05)),
            od_min_edge_length=float(getattr(sim, "OD_MIN_EDGE_LENGTH", 20.0)),
            od_random_walk_fallback=bool(getattr(sim, "OD_RANDOM_WALK_FALLBACK", True)),
            depart_lane=OD_DEPART_LANE,
            intersection_no_lane_change_distance=SIM_CENTER_INTERSECTION_NO_LC,
            intersection_lane_prep_distance=SIM_CENTER_INTERSECTION_PREP,
            tls_no_lane_change_distance=max(SIM_CENTER_TLS_NO_LC, SIM_CENTER_INTERSECTION_NO_LC),
            tls_lane_prep_distance=max(SIM_CENTER_TLS_PREP, SIM_CENTER_INTERSECTION_PREP),
            unjustified_stop_watchdog=bool(getattr(sim, "UNJUSTIFIED_STOP_WATCHDOG_ENABLED", True)),
            unjustified_stop_check_interval=float(getattr(sim, "UNJUSTIFIED_STOP_CHECK_INTERVAL", 1.0)),
            unjustified_stop_speed=float(getattr(sim, "UNJUSTIFIED_STOP_SPEED", 0.20)),
            unjustified_stop_min_time=float(getattr(sim, "UNJUSTIFIED_STOP_MIN_TIME", 3.0)),
            disable_strict_split=False,
        )


def reset_sim_globals() -> None:
    """Clear mutable module-level state before each SUMO episode."""

    for name in (
        "KEEP_CLEAR_HELD_VEHICLES",
        "KEEP_CLEAR_HOLD_START_TIME",
        "KEEP_CLEAR_FORCE_RELEASE_UNTIL",
        "TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES",
        "APPROACH_TURN_DECISIONS",
        "APPROACH_TURN_COUNTS",
        "VEHICLE_EDGE_HISTORY",
        "VEHICLE_LAST_EDGE",
        "LANE_BALANCE_LAST_CHANGE",
        "UNJUSTIFIED_STOP_TRACKING",
        "UNJUSTIFIED_STOP_LAST_ACTION",
    ):
        obj = getattr(sim, name, None)
        if hasattr(obj, "clear"):
            obj.clear()

    if hasattr(sim, "TURN_LANE_PREFERENCE_INDEX"):
        sim.TURN_LANE_PREFERENCE_INDEX = {}
    if hasattr(sim, "APPROACH_DECISION_INDEX"):
        sim.APPROACH_DECISION_INDEX = {}
    if hasattr(sim, "TRAFFIC_LIGHT_APPROACH_LANES"):
        sim.TRAFFIC_LIGHT_APPROACH_LANES = set()


def apply_sim_distance_globals(scenario: TrafficScenario) -> None:
    """Match the same global-distance updates the simulation CLI performs."""

    if hasattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE"):
        sim.INTERSECTION_NO_LANE_CHANGE_DISTANCE = max(0.0, scenario.intersection_no_lane_change_distance)
    if hasattr(sim, "INTERSECTION_LANE_PREP_DISTANCE"):
        sim.INTERSECTION_LANE_PREP_DISTANCE = max(
            getattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE", 0.0),
            scenario.intersection_lane_prep_distance,
        )
    if hasattr(sim, "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE"):
        sim.TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE = max(
            getattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE", 0.0),
            scenario.tls_no_lane_change_distance,
        )
    if hasattr(sim, "TRAFFIC_LIGHT_LANE_PREP_DISTANCE"):
        sim.TRAFFIC_LIGHT_LANE_PREP_DISTANCE = max(
            getattr(sim, "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE", 0.0),
            getattr(sim, "INTERSECTION_LANE_PREP_DISTANCE", 0.0),
            scenario.tls_lane_prep_distance,
        )


def make_sim_args(scenario: TrafficScenario, route_file: str, episode_seconds: int, gui: bool) -> SimpleNamespace:
    args = SimpleNamespace()
    args.route_file = route_file
    args.end = float(episode_seconds)
    args.seed = scenario.seed
    args.gui = bool(gui)
    args.max_vehicles = scenario.max_vehicles
    args.target_vehicles = scenario.target_vehicles
    args.initial_vehicles = scenario.initial_vehicles
    args.spawn_batch = scenario.spawn_batch
    args.spawn_attempts = 40
    args.route_lookahead_edges = scenario.route_lookahead_edges
    args.min_remaining_edges = scenario.min_remaining_edges
    args.recovery_attempts = 20
    args.spawn_grid_size = scenario.spawn_grid_size
    args.local_road_penalty = scenario.local_road_penalty
    args.local_to_local_penalty = scenario.local_to_local_penalty
    args.leave_local_bonus = scenario.leave_local_bonus
    args.non_core_penalty = scenario.non_core_penalty
    args.signal_timing_jitter = scenario.signal_timing_jitter
    args.max_depart_delay = scenario.max_depart_delay
    args.time_to_teleport = scenario.time_to_teleport
    args.green_duration = scenario.green_duration
    args.max_consecutive_straight = 4
    args.print_every = 60.0
    args.disable_strict_split = scenario.disable_strict_split
    args.generate_only = False
    args.run_existing = False

    # Exact current simulation mode: OD fastest routing, not forced random walk.
    args.routing_mode = scenario.routing_mode
    args.od_route_attempts = scenario.od_route_attempts
    args.od_boundary_margin_fraction = scenario.od_boundary_margin_fraction
    args.od_min_euclidean_distance = scenario.od_min_euclidean_distance
    args.od_min_route_distance = scenario.od_min_route_distance
    args.od_min_zone_separation = scenario.od_min_zone_separation
    args.od_max_local_middle_fraction = scenario.od_max_local_middle_fraction
    args.od_local_middle_trim_edges = scenario.od_local_middle_trim_edges
    args.od_through_trip_probability = scenario.od_through_trip_probability
    args.od_access_trip_probability = scenario.od_access_trip_probability
    args.od_long_local_trip_probability = scenario.od_long_local_trip_probability
    args.od_min_edge_length = scenario.od_min_edge_length
    args.od_random_walk_fallback = scenario.od_random_walk_fallback
    args.od_no_random_walk_fallback = not scenario.od_random_walk_fallback
    args.depart_lane = scenario.depart_lane

    # Exact current lane-change constraints.
    args.intersection_no_lane_change_distance = scenario.intersection_no_lane_change_distance
    args.intersection_lane_prep_distance = scenario.intersection_lane_prep_distance
    args.tls_no_lane_change_distance = scenario.tls_no_lane_change_distance
    args.tls_lane_prep_distance = scenario.tls_lane_prep_distance

    # Exact current phantom-stop watchdog.
    args.unjustified_stop_watchdog = scenario.unjustified_stop_watchdog
    args.unjustified_stop_check_interval = scenario.unjustified_stop_check_interval
    args.unjustified_stop_speed = scenario.unjustified_stop_speed
    args.unjustified_stop_min_time = scenario.unjustified_stop_min_time

    # Match realistic_all_intersections_fixed_cycle.py defaults: ambulances enabled
    # unless the simulation file itself is changed.
    args.disable_ambulances = False
    args.ambulance_interval = float(getattr(sim, "AMBULANCE_SPAWN_INTERVAL", 120.0))
    args.ambulance_min_euclidean_distance = float(getattr(sim, "AMBULANCE_MIN_EUCLIDEAN_DISTANCE", 1500.0))
    args.ambulance_min_route_distance = float(getattr(sim, "AMBULANCE_MIN_ROUTE_DISTANCE", 1800.0))
    args.ambulance_min_route_edges = int(getattr(sim, "AMBULANCE_MIN_ROUTE_EDGES", 20))
    args.ambulance_route_attempts = int(getattr(sim, "AMBULANCE_ROUTE_ATTEMPTS", 100))
    args.ambulance_depart_lane = getattr(sim, "AMBULANCE_DEPART_LANE", "free")
    args.ambulance_depart_pos = getattr(sim, "AMBULANCE_DEPART_POS", "random_free")
    args.ambulance_poi_radius = float(getattr(sim, "AMBULANCE_POI_RADIUS", 250.0))
    args.ambulance_debug = False

    return args


def movement_queue_and_wait(controller: dict[str, Any], movement_label: str) -> tuple[float, float]:
    veh_ids: set[str] = set()
    for lane_id in controller["movement_in_lanes_cache"].get(movement_label, set()):
        try:
            veh_ids.update(sim.traci.lane.getLastStepVehicleIDs(lane_id))
        except sim.traci.TraCIException:
            continue

    queue = 0.0
    wait = 0.0
    for veh_id in veh_ids:
        try:
            speed = sim.traci.vehicle.getSpeed(veh_id)
            if speed < sim.QUEUE_SPEED_THRESHOLD:
                queue += 1.0
            wait += sim.traci.vehicle.getWaitingTime(veh_id)
        except sim.traci.TraCIException:
            continue
    return queue, wait


def target_wait_and_queue(controller: dict[str, Any]) -> tuple[float, float]:
    return sim.total_controlled_wait_and_queue(controller)


def network_wait_queue_speed() -> tuple[float, float, float]:
    total_wait = 0.0
    total_queue = 0.0
    total_speed = 0.0
    count = 0
    for veh_id in sim.traci.vehicle.getIDList():
        try:
            if str(veh_id).startswith("ambulance_"):
                continue
            speed = sim.traci.vehicle.getSpeed(veh_id)
            wait = sim.traci.vehicle.getWaitingTime(veh_id)
        except sim.traci.TraCIException:
            continue
        count += 1
        total_speed += speed
        total_wait += wait
        if speed < sim.QUEUE_SPEED_THRESHOLD:
            total_queue += 1.0
    avg_speed = total_speed / count if count else 0.0
    return total_wait, total_queue, avg_speed


def phase_slot(controller: dict[str, Any]) -> int:
    try:
        return int(controller["phases"][controller["phase_pos"]].get("slot", -1))
    except Exception:
        return -1


def get_observation(controller: dict[str, Any], episode_seconds: int) -> np.ndarray:
    """30-value observation compatible with the fast proxy model.

    Layout: 12 movement queue/wait pairs, 4 phase one-hot values, normalized
    phase elapsed, and normalized simulation time.  The surrounding SUMO
    simulation is the full realistic_all_intersections_fixed_cycle.py logic.
    """
    obs: list[float] = []

    for label in sim.MOVEMENT_LABELS:
        queue, wait = movement_queue_and_wait(controller, label)
        obs.append(queue / 100.0)
        obs.append(wait / 1000.0)

    phase_one_hot = [0.0, 0.0, 0.0, 0.0]
    slot = phase_slot(controller)
    if 0 <= slot < 4:
        phase_one_hot[slot] = 1.0
    obs.extend(phase_one_hot)

    obs.append(float(controller.get("phase_elapsed", 0.0)) / 60.0)
    try:
        sim_time = sim.traci.simulation.getTime()
    except sim.traci.TraCIException:
        sim_time = 0.0
    obs.append(sim_time / max(1.0, float(episode_seconds)))

    return np.array(obs, dtype=np.float32)

class ExactSimulationTrafficSignalEnv(gym.Env):
    """Gym wrapper around the current realistic simulation file.

    One selected traffic light is controlled by RL.  All normal vehicle motion,
    routing, lane changing, anti-gridlock logic, and phantom-stop repair come
    directly from realistic_all_intersections_fixed_cycle.py.  Ambulances are
    disabled through args.disable_ambulances=True.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tls_id: str = TARGET_TLS_ID,
        episode_seconds: int = 900,
        gui: bool = False,
        randomize_scenarios: bool = True,
        base_seed: Optional[int] = None,
        env_rank: int = 0,
        print_scenarios: bool = False,
        fixed_scenario: Optional[TrafficScenario] = None,
        max_vehicle_center: int = SIM_CENTER_MAX_VEHICLES,
        target_center: int = SIM_CENTER_TARGET_VEHICLES,
        initial_center: int = SIM_CENTER_INITIAL_VEHICLES,
        spawn_batch_center: int = SIM_CENTER_SPAWN_BATCH,
        green_duration_center: float = SIM_CENTER_GREEN_DURATION,
        density_spread: float = 0.18,
        initial_spread: float = 0.55,
    ):
        super().__init__()
        self.tls_id = tls_id
        self.episode_seconds = int(episode_seconds)
        self.gui = bool(gui)
        self.randomize_scenarios = bool(randomize_scenarios)
        self.env_rank = int(env_rank)
        self.print_scenarios = bool(print_scenarios)
        self.fixed_scenario = fixed_scenario

        self.sampler = ScenarioSampler(
            base_seed=(base_seed or 0) + 1009 * (env_rank + 1),
            max_vehicle_center=max_vehicle_center,
            target_center=target_center,
            initial_center=initial_center,
            spawn_batch_center=spawn_batch_center,
            green_duration_center=green_duration_center,
            density_spread=density_spread,
            initial_spread=initial_spread,
        )

        self.started = False
        self.controllers: list[dict[str, Any]] = []
        self.controller: Optional[dict[str, Any]] = None
        self.skipped: list[str] = []
        self.scenario: Optional[TrafficScenario] = None
        self.args: Optional[SimpleNamespace] = None
        self.route_file = os.path.join(
            sim.BASE_DIR,
            f"random_drive_dynamic_turns_train_{os.getpid()}_{self.env_rank}.rou.xml",
        )

        self.rng: Optional[random.Random] = None
        self.turn_counts: Counter[str] = Counter()
        self.sim_state: dict[str, Any] = {}
        self.main_start_edges: Any = None
        self.turn_index: Any = None
        self.raw_graph: dict[str, list[str]] = {}
        self.edge_metadata: dict[str, Any] = {}
        self.core_edges: set[str] = set()

        self.prev_target_wait = 0.0
        self.prev_target_queue = 0.0
        self.prev_global_wait = 0.0
        self.prev_global_queue = 0.0
        self.total_arrived = 0

        self.action_space = spaces.Discrete(5)  # 0=hold, 1..4=switch to phase slot 0..3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

    def _choose_scenario(self) -> TrafficScenario:
        if self.fixed_scenario is not None:
            return self.fixed_scenario
        if self.randomize_scenarios:
            return self.sampler.sample()
        return build_fixed_scenario(seed=42 + self.env_rank)

    def _sumo_cmd(self) -> list[str]:
        assert self.scenario is not None
        binary = sim.SUMO_GUI_BINARY if self.gui else sim.SUMO_HEADLESS_BINARY
        return [
            binary,
            "-n", sim.NET_FILE,
            "-r", self.route_file,
            "--start",
            "--step-length", str(sim.STEP_LENGTH),
            "--end", str(float(self.episode_seconds)),
            "--seed", str(self.scenario.seed),
            "--max-num-vehicles", str(self.scenario.max_vehicles),
            "--max-depart-delay", str(self.scenario.max_depart_delay),
            "--time-to-teleport", str(self.scenario.time_to_teleport),
            "--ignore-route-errors", "true",
            "--quit-on-end", "false",
            *sim.QUIET_SUMO_ARGS,
        ]

    def _close_traci(self) -> None:
        if self.started:
            try:
                sim.traci.close(False)
            except Exception:
                pass
        self.started = False

    def close(self) -> None:
        self._close_traci()

    def _build_episode(self) -> None:
        assert self.scenario is not None
        assert self.args is not None
        assert self.rng is not None

        reset_sim_globals()
        apply_sim_distance_globals(self.scenario)
        sim.write_empty_route_file(self.route_file)

        if self.gui:
            sim.ensure_xquartz()

        sim.traci.start(self._sumo_cmd())
        self.started = True

        self.controllers, self.skipped = sim.build_all_fixed_controllers(rng=self.rng, args=self.args)
        sim.rebuild_traffic_light_approach_lanes(self.controllers)

        self.controller = next((c for c in self.controllers if c["tls_id"] == self.tls_id), None)
        if self.controller is None:
            available = [c["tls_id"] for c in self.controllers[:20]]
            raise RuntimeError(
                f"Could not find target TLS {self.tls_id!r}. "
                f"First available controlled TLS IDs: {available}"
            )

        valid_edges = sim.get_valid_passenger_edges()
        self.edge_metadata = sim.build_edge_metadata(valid_edges)
        self.raw_graph = sim.build_raw_successor_graph(valid_edges)
        self.raw_graph = sim.remove_hardcoded_loop_region_from_graph(self.raw_graph)
        self.core_edges = set(self.raw_graph)

        start_candidates = list(self.raw_graph.keys())
        if not start_candidates:
            raise RuntimeError("No valid start edges were found.")

        self.main_start_edges = sim.build_spawn_zones(
            start_edges=start_candidates,
            edge_metadata=self.edge_metadata,
            grid_size=self.args.spawn_grid_size,
        ) or start_candidates

        self.turn_index = sim.build_turn_decision_index(
            controllers=self.controllers,
            raw_graph=self.raw_graph,
        )

        od_context = None
        if sim.use_od_routing(self.args):
            od_context = sim.build_od_context(
                valid_edges=valid_edges,
                raw_graph=self.raw_graph,
                edge_metadata=self.edge_metadata,
                args=self.args,
            )

        sim.APPROACH_DECISION_INDEX = sim.build_approach_decision_index(self.raw_graph)
        sim.TURN_LANE_PREFERENCE_INDEX = sim.APPROACH_DECISION_INDEX

        self.sim_state = {
            "next_vehicle_id": 0,
            "next_route_id": 0,
            "next_spawn_zone_index": 0,
            "next_od_origin_zone_index": 0,
            "next_lane_pref_time": 0.0,
            "next_lane_balance_time": 0.0,
            "next_unconnected_lane_rescue_time": 0.0,
            "next_unjustified_stop_check_time": 0.0,
            # NOTE: this must be a reachable time (0.0), not float("inf").
            # Spawning is gated by `sim_time >= next_ambulance_spawn` in
            # update_ambulances(); if this starts at infinity, that
            # condition can never become true and ambulances never spawn
            # for the entire episode, regardless of ambulance_interval.
            "next_ambulance_spawn": 0.0,
            "active_ambulances": {},
            "od_context": od_context,
            "od_trip_counts": Counter(),
            "od_movement_counts": Counter(),
            "od_route_failures": 0,
        }

        sim.fill_vehicle_population(
            sim_state=self.sim_state,
            target_count=self.args.initial_vehicles,
            max_to_spawn=self.args.initial_vehicles,
            start_edges=self.main_start_edges,
            turn_index=self.turn_index,
            raw_graph=self.raw_graph,
            edge_metadata=self.edge_metadata,
            core_edges=self.core_edges,
            rng=self.rng,
            turn_counts=self.turn_counts,
            args=self.args,
        )

        self.prev_target_wait, self.prev_target_queue = target_wait_and_queue(self.controller)
        self.prev_global_wait, self.prev_global_queue, _ = network_wait_queue_speed()
        self.total_arrived = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self._close_traci()

        if seed is not None:
            self.sampler.rng.seed(seed + 1009 * (self.env_rank + 1))

        self.scenario = self._choose_scenario()
        self.args = make_sim_args(
            scenario=self.scenario,
            route_file=self.route_file,
            episode_seconds=self.episode_seconds,
            gui=self.gui,
        )
        self.rng = random.Random(self.scenario.seed)
        self.turn_counts = Counter()

        if self.print_scenarios:
            print(f"[env {self.env_rank}] scenario: {asdict(self.scenario)}", flush=True)

        self._build_episode()
        assert self.controller is not None
        return get_observation(self.controller, self.episode_seconds), self._info()

    def _phase_pos_for_slot(self, slot: int) -> Optional[int]:
        assert self.controller is not None
        for pos, phase in enumerate(self.controller["phases"]):
            if int(phase.get("slot", -1)) == slot:
                return pos
        return None

    def _valid_action_mask(self) -> np.ndarray:
        assert self.controller is not None
        mask = np.zeros(5, dtype=bool)

        if self.controller.get("disabled") or self.controller["mode"] != "green":
            mask[0] = True
            return mask

        elapsed = float(self.controller.get("phase_elapsed", 0.0))
        if elapsed < MIN_GREEN_BEFORE_SWITCH:
            mask[0] = True
            return mask

        if elapsed < HARD_MAX_GREEN:
            mask[0] = True

        current_pos = self.controller["phase_pos"]
        for action in range(1, 5):
            phase_pos = self._phase_pos_for_slot(action - 1)
            if phase_pos is None or phase_pos == current_pos:
                continue
            mask[action] = True

        if not mask.any():
            mask[0] = True
        return mask

    def action_masks(self) -> np.ndarray:
        return self._valid_action_mask()

    def _apply_rl_action(self, action: int) -> tuple[bool, bool]:
        assert self.controller is not None
        action = int(action)
        mask = self._valid_action_mask()
        if action < 0 or action >= len(mask) or not mask[action]:
            action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

        elapsed = float(self.controller.get("phase_elapsed", 0.0))
        forced = False
        switched = False

        if action == 0:
            if self.controller["mode"] == "green" and elapsed >= HARD_MAX_GREEN:
                switched = sim.switch_next_fixed_phase(self.controller)
                forced = True
            return switched, forced

        if self.controller["mode"] != "green" or elapsed < MIN_GREEN_BEFORE_SWITCH:
            return False, False

        phase_pos = self._phase_pos_for_slot(action - 1)
        if phase_pos is None:
            return False, False

        switched = sim.request_switch(self.controller, phase_pos)
        return switched, forced

    def _apply_fixed_cycle_to_other_tls(self) -> None:
        assert self.controller is not None
        for controller in self.controllers:
            if controller is self.controller or controller.get("disabled"):
                continue
            if controller["mode"] == "green" and controller["phase_elapsed"] >= controller["green_duration"]:
                sim.switch_next_fixed_phase(controller)

    def _starvation_penalty(self) -> float:
        assert self.controller is not None
        elapsed = float(self.controller.get("phase_elapsed", 0.0))
        if elapsed <= SOFT_MAX_GREEN:
            return 0.0
        excess = elapsed - SOFT_MAX_GREEN
        # Penalize holding a phase too long when there are queued vehicles on
        # other movements. This encourages smooth, fair flow without forcing a
        # fixed cycle.
        try:
            active_core = set(self.controller["phases"][self.controller["phase_pos"]].get("core_labels", []))
        except Exception:
            active_core = set()

        waiting_other = 0.0
        for label in sim.MOVEMENT_LABELS:
            # Right turns are usually permissive in the simulation; the useful
            # starvation signal is for protected straight/left movements that
            # are currently not in the active phase.
            if label.endswith("-R") or label in active_core:
                continue
            queue, _wait = movement_queue_and_wait(self.controller, label)
            if queue > 0:
                waiting_other += queue
        return STARVATION_PENALTY_SCALE * excess * waiting_other

    def step(self, action: int):
        assert self.controller is not None
        assert self.args is not None
        assert self.rng is not None

        self._apply_fixed_cycle_to_other_tls()
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

        target_wait, target_queue = target_wait_and_queue(self.controller)
        global_wait, global_queue, avg_speed = network_wait_queue_speed()

        target_wait_delta = self.prev_target_wait - target_wait
        target_queue_delta = self.prev_target_queue - target_queue
        global_wait_delta = self.prev_global_wait - global_wait
        global_queue_delta = self.prev_global_queue - global_queue

        reward = 0.0
        reward += target_wait_delta / TARGET_WAIT_DELTA_SCALE
        reward += target_queue_delta / TARGET_QUEUE_DELTA_SCALE
        reward += global_wait_delta / GLOBAL_WAIT_DELTA_SCALE
        reward += global_queue_delta / GLOBAL_QUEUE_DELTA_SCALE
        reward -= target_wait / TARGET_WAIT_LEVEL_SCALE
        reward -= target_queue / TARGET_QUEUE_LEVEL_SCALE
        reward -= global_wait / GLOBAL_WAIT_LEVEL_SCALE
        reward -= global_queue / GLOBAL_QUEUE_LEVEL_SCALE
        reward += arrived * ARRIVAL_BONUS
        reward -= recovered * RECOVERY_PENALTY
        reward -= self._starvation_penalty()

        if switched:
            reward -= SWITCH_PENALTY
        if forced:
            reward -= FORCED_SWITCH_PENALTY

        self.prev_target_wait = target_wait
        self.prev_target_queue = target_queue
        self.prev_global_wait = global_wait
        self.prev_global_queue = global_queue

        sim_time = sim.traci.simulation.getTime()
        terminated = False
        truncated = bool(sim_time >= self.episode_seconds)
        obs = get_observation(self.controller, self.episode_seconds)

        return obs, float(reward), terminated, truncated, self._info(
            switched=switched,
            forced=forced,
            spawned=spawned,
            extended=extended,
            recovered=recovered,
            arrived=arrived,
            target_wait=target_wait,
            target_queue=target_queue,
            global_wait=global_wait,
            global_queue=global_queue,
            avg_speed=avg_speed,
        )

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self.scenario is not None:
            info["scenario"] = asdict(self.scenario)
        if self.controller is not None and self.started:
            try:
                info.update(
                    sim_time=sim.traci.simulation.getTime(),
                    active_vehicles=sim.traci.vehicle.getIDCount(),
                    phase_name=self.controller["phases"][self.controller["phase_pos"]]["name"],
                    phase_elapsed=float(self.controller.get("phase_elapsed", 0.0)),
                    mode=self.controller.get("mode"),
                    total_arrived=self.total_arrived,
                    ambulance_count=len(self.sim_state.get("active_ambulances", {})),
                )
            except Exception:
                pass
        info.update(extra)
        return info


def build_fixed_scenario(seed: int = 42, args: Optional[argparse.Namespace] = None) -> TrafficScenario:
    max_vehicles = getattr(args, "max_vehicle_center", SIM_CENTER_MAX_VEHICLES) if args is not None else SIM_CENTER_MAX_VEHICLES
    target = getattr(args, "target_vehicle_center", SIM_CENTER_TARGET_VEHICLES) if args is not None else SIM_CENTER_TARGET_VEHICLES
    initial = getattr(args, "initial_vehicle_center", SIM_CENTER_INITIAL_VEHICLES) if args is not None else SIM_CENTER_INITIAL_VEHICLES
    spawn_batch = getattr(args, "spawn_batch_center", SIM_CENTER_SPAWN_BATCH) if args is not None else SIM_CENTER_SPAWN_BATCH
    green = getattr(args, "green_duration_center", SIM_CENTER_GREEN_DURATION) if args is not None else SIM_CENTER_GREEN_DURATION

    max_vehicles = min(int(max_vehicles), SIM_CENTER_MAX_VEHICLES)
    target = min(int(target), max_vehicles)
    initial = min(int(initial), target)

    return TrafficScenario(
        seed=int(seed),
        max_vehicles=max_vehicles,
        target_vehicles=target,
        initial_vehicles=initial,
        spawn_batch=int(spawn_batch),
        route_lookahead_edges=SIM_CENTER_ROUTE_LOOKAHEAD,
        min_remaining_edges=SIM_CENTER_MIN_REMAINING,
        green_duration=float(green),
        signal_timing_jitter=0.15,
        spawn_grid_size=SIM_CENTER_SPAWN_GRID_SIZE,
        max_depart_delay=SIM_CENTER_MAX_DEPART_DELAY,
        time_to_teleport=SIM_CENTER_TELEPORT,
        local_road_penalty=0.04,
        local_to_local_penalty=0.15,
        leave_local_bonus=8.0,
        non_core_penalty=1.0,
        routing_mode=ROUTING_MODE_OD,
        od_route_attempts=int(getattr(sim, "OD_ROUTE_ATTEMPTS", 120)),
        od_boundary_margin_fraction=float(getattr(sim, "OD_BOUNDARY_MARGIN_FRACTION", 0.13)),
        od_min_euclidean_distance=float(getattr(sim, "OD_MIN_EUCLIDEAN_DISTANCE", 900.0)),
        od_min_route_distance=float(getattr(sim, "OD_MIN_ROUTE_DISTANCE", 1200.0)),
        od_min_zone_separation=int(getattr(sim, "OD_MIN_ZONE_SEPARATION", 2)),
        od_max_local_middle_fraction=float(getattr(sim, "OD_MAX_LOCAL_MIDDLE_FRACTION", 0.35)),
        od_local_middle_trim_edges=int(getattr(sim, "OD_LOCAL_MIDDLE_TRIM_EDGES", 2)),
        od_through_trip_probability=float(getattr(sim, "OD_THROUGH_TRIP_PROBABILITY", 0.72)),
        od_access_trip_probability=float(getattr(sim, "OD_ACCESS_TRIP_PROBABILITY", 0.23)),
        od_long_local_trip_probability=float(getattr(sim, "OD_LONG_LOCAL_TRIP_PROBABILITY", 0.05)),
        od_min_edge_length=float(getattr(sim, "OD_MIN_EDGE_LENGTH", 20.0)),
        od_random_walk_fallback=bool(getattr(sim, "OD_RANDOM_WALK_FALLBACK", True)),
        depart_lane=OD_DEPART_LANE,
        intersection_no_lane_change_distance=SIM_CENTER_INTERSECTION_NO_LC,
        intersection_lane_prep_distance=SIM_CENTER_INTERSECTION_PREP,
        tls_no_lane_change_distance=max(SIM_CENTER_TLS_NO_LC, SIM_CENTER_INTERSECTION_NO_LC),
        tls_lane_prep_distance=max(SIM_CENTER_TLS_PREP, SIM_CENTER_INTERSECTION_PREP),
        unjustified_stop_watchdog=bool(getattr(sim, "UNJUSTIFIED_STOP_WATCHDOG_ENABLED", True)),
        unjustified_stop_check_interval=float(getattr(sim, "UNJUSTIFIED_STOP_CHECK_INTERVAL", 1.0)),
        unjustified_stop_speed=float(getattr(sim, "UNJUSTIFIED_STOP_SPEED", 0.20)),
        unjustified_stop_min_time=float(getattr(sim, "UNJUSTIFIED_STOP_MIN_TIME", 3.0)),
        disable_strict_split=False,
    )


def make_env(rank: int, args: argparse.Namespace, eval_mode: bool = False):
    def _init():
        fixed_scenario = None
        randomize = args.randomize_scenarios and not eval_mode
        if eval_mode and args.eval_fixed_scenario:
            fixed_scenario = build_fixed_scenario(seed=args.seed or 42, args=args)

        env = ExactSimulationTrafficSignalEnv(
            tls_id=args.tls_id,
            episode_seconds=args.episode_seconds,
            gui=(args.gui and rank == 0),
            randomize_scenarios=randomize,
            base_seed=args.seed,
            env_rank=rank,
            print_scenarios=args.print_scenarios,
            fixed_scenario=fixed_scenario,
            max_vehicle_center=args.max_vehicle_center,
            target_center=args.target_vehicle_center,
            initial_center=args.initial_vehicle_center,
            spawn_batch_center=args.spawn_batch_center,
            green_duration_center=args.green_duration_center,
            density_spread=args.density_spread,
            initial_spread=args.initial_spread,
        )
        return Monitor(env)
    return _init


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(self.save_path)
        return True


def linear_decay(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        done = 1.0 - progress_remaining
        return start + done * (end - start)
    return schedule


def build_vec_env(args: argparse.Namespace, eval_mode: bool = False):
    n_envs = 1 if eval_mode else max(1, int(args.num_envs))
    env_fns = [make_env(rank=i, args=args, eval_mode=eval_mode) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    vecnorm_path = str(model_path.with_suffix(".vecnormalize.pkl"))

    train_env = build_vec_env(args, eval_mode=False)
    if args.resume and model_path.with_suffix(".zip").exists() and Path(vecnorm_path).exists():
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=args.clip_obs,
            clip_reward=args.clip_reward,
            gamma=args.gamma,
        )

    if args.resume and model_path.with_suffix(".zip").exists():
        print(f"Resuming model from {model_path}.zip")
        model = load_maskable_ppo_for_inference(model_path, train_env, args.device)
        model.learning_rate = linear_decay(args.lr_start, args.lr_end)
    else:
        print("Starting a fresh exact-simulation MaskablePPO model")
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_decay(args.lr_start, args.lr_end),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=linear_decay(args.clip_start, args.clip_end),
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            verbose=1,
            tensorboard_log=str(log_dir),
            device=args.device,
            seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                activation_fn=torch.nn.Tanh,
                ortho_init=True,
            ),
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(1, args.num_envs), 1),
            save_path=str(model_path.parent),
            name_prefix=model_path.name + "_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        ),
        SaveVecNormalizeCallback(args.checkpoint_freq, vecnorm_path, verbose=1),
    ])

    print("\nExact simulation training configuration")
    print("  imported simulation file: realistic_all_intersections_fixed_cycle.py")
    print("  ambulances:              disabled")
    print("  routing mode:            OD fastest routes")
    print(f"  target TLS:              {args.tls_id}")
    print(f"  max vehicle center:      {args.max_vehicle_center}")
    print(f"  target vehicle center:   {args.target_vehicle_center}")
    print(f"  initial vehicle center:  {args.initial_vehicle_center}")
    print(f"  spawn batch center:      {args.spawn_batch_center}")
    print(f"  green duration center:   {args.green_duration_center}")
    print(f"  density spread:          {args.density_spread}")
    print(f"  timesteps:               {args.timesteps}")
    print(f"  model path:              {model_path}.zip")
    print(f"  vecnormalize path:       {vecnorm_path}")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=args.progress_bar,
        tb_log_name="exact_sim_maskable_ppo",
        reset_num_timesteps=not args.resume,
    )

    print(f"Saving model to {model_path}.zip")
    model.save(str(model_path))
    env = model.get_vec_normalize_env()
    if env is not None:
        print(f"Saving VecNormalize stats to {vecnorm_path}")
        env.save(vecnorm_path)
    train_env.close()


def find_vecnormalize_path(model_path: Path, explicit_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend([
        model_path.parent / f"{model_path.stem}_vecnormalize.pkl",
        model_path.with_suffix(".vecnormalize.pkl"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    vecnorm_path = find_vecnormalize_path(model_path, getattr(args, "vecnormalize_path", None))

    eval_env = build_vec_env(args, eval_mode=True)
    if vecnorm_path is not None:
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("Warning: no VecNormalize stats found; using raw observations.")
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
        eval_env.training = False

    model = load_maskable_ppo_for_inference(model_path, eval_env, args.device)
    obs = eval_env.reset()
    total_reward = 0.0

    print("\nStarting evaluation through the same simulation. Press Ctrl+C to stop.")
    try:
        for step in range(1, args.eval_steps + 1):
            masks = get_action_masks(eval_env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, rewards, dones, infos = eval_env.step(action)
            total_reward += float(np.mean(rewards))

            if step % args.eval_print_every == 0:
                info = infos[0] if infos else {}
                print(
                    f"step={step:6d}, "
                    f"t={float(info.get('sim_time', 0.0)):8.1f}, "
                    f"reward={total_reward:10.2f}, "
                    f"active={int(info.get('active_vehicles', 0)):4d}, "
                    f"amb={int(info.get('ambulance_count', 0)):2d}, "
                    f"mode={info.get('mode', '')}, "
                    f"phase={info.get('phase_name', '')}"
                )

            if bool(np.any(dones)):
                print(f"Episode ended at step={step}, total_reward={total_reward:.2f}")
                break
    finally:
        eval_env.close()



# ============================================================
# Model-vs-fixed-cycle comparison
# ============================================================

import csv
import json
import statistics


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw).split(','):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [42]


def safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def collect_sample(samples: list[dict[str, float]], info: dict[str, Any], reward: float = 0.0) -> None:
    samples.append({
        'sim_time': float(info.get('sim_time', 0.0) or 0.0),
        'reward': float(reward),
        'active_vehicles': float(info.get('active_vehicles', 0) or 0),
        'target_queue': float(info.get('target_queue', 0.0) or 0.0),
        'target_wait': float(info.get('target_wait', 0.0) or 0.0),
        'global_queue': float(info.get('global_queue', 0.0) or 0.0),
        'global_wait': float(info.get('global_wait', 0.0) or 0.0),
        'avg_speed': float(info.get('avg_speed', 0.0) or 0.0),
        'arrived': float(info.get('arrived', 0) or 0),
        'spawned': float(info.get('spawned', 0) or 0),
        'extended': float(info.get('extended', 0) or 0),
        'recovered': float(info.get('recovered', 0) or 0),
        'total_arrived': float(info.get('total_arrived', 0) or 0),
        'ambulance_count': float(info.get('ambulance_count', 0) or 0),
    })


def summarize_samples(label: str, seed: int, samples: list[dict[str, float]]) -> dict[str, float | int | str]:
    if not samples:
        return {'controller': label, 'seed': seed, 'sim_seconds': 0.0, 'samples': 0}

    def col(name: str) -> list[float]:
        return [float(s.get(name, 0.0)) for s in samples]

    return {
        'controller': label,
        'seed': seed,
        'sim_seconds': col('sim_time')[-1],
        'samples': len(samples),
        'total_reward': sum(col('reward')),
        'total_arrived': int(col('total_arrived')[-1]),
        'arrived_this_window_sum': int(sum(col('arrived'))),
        'spawned_total': int(sum(col('spawned'))),
        'recovered_total': int(sum(col('recovered'))),
        'extended_total': int(sum(col('extended'))),
        'mean_active_vehicles': safe_mean(col('active_vehicles')),
        'mean_avg_speed_mps': safe_mean(col('avg_speed')),
        'mean_target_queue': safe_mean(col('target_queue')),
        'max_target_queue': safe_max(col('target_queue')),
        'mean_target_wait': safe_mean(col('target_wait')),
        'max_target_wait': safe_max(col('target_wait')),
        'mean_global_queue': safe_mean(col('global_queue')),
        'max_global_queue': safe_max(col('global_queue')),
        'mean_global_wait': safe_mean(col('global_wait')),
        'max_global_wait': safe_max(col('global_wait')),
        'mean_ambulance_count': safe_mean(col('ambulance_count')),
    }


def improvement_percent(model_value: float, fixed_value: float, higher_is_better: bool) -> float:
    if abs(fixed_value) < 1e-12:
        return 0.0
    if higher_is_better:
        return 100.0 * (model_value - fixed_value) / abs(fixed_value)
    return 100.0 * (fixed_value - model_value) / abs(fixed_value)


def run_model_episode(args: argparse.Namespace, scenario: TrafficScenario, seed: int) -> dict[str, float | int | str]:
    raw_env = DummyVecEnv([
        lambda: Monitor(ExactSimulationTrafficSignalEnv(
            tls_id=args.tls_id,
            episode_seconds=args.episode_seconds,
            gui=bool(args.gui),
            randomize_scenarios=False,
            base_seed=seed,
            env_rank=0,
            print_scenarios=args.print_scenarios,
            fixed_scenario=scenario,
            max_vehicle_center=args.max_vehicle_center,
            target_center=args.target_vehicle_center,
            initial_center=args.initial_vehicle_center,
            spawn_batch_center=args.spawn_batch_center,
            green_duration_center=args.green_duration_center,
            density_spread=args.density_spread,
            initial_spread=args.initial_spread,
        ))
    ])

    vecnorm_path = find_vecnormalize_path(Path(args.model_path), getattr(args, 'vecnormalize_path', None))
    if vecnorm_path is not None:
        print(f'[model seed {seed}] Loading VecNormalize stats from {vecnorm_path}')
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = False
        env.norm_reward = False
    else:
        print(f'[model seed {seed}] Warning: no VecNormalize stats found; using raw observations.')
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=False)
        env.training = False

    model = load_maskable_ppo_for_inference(Path(args.model_path), env, args.device)
    obs = env.reset()
    samples: list[dict[str, float]] = []

    try:
        for step in range(1, args.eval_steps + 1):
            try:
                masks = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            except Exception:
                action, _ = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)
            info = infos[0] if infos else {}
            reward = float(rewards[0]) if len(rewards) else 0.0
            collect_sample(samples, info, reward=reward)

            if step % args.eval_print_every == 0:
                print(
                    f'[model seed {seed}] step={step:6d}, '
                    f't={float(info.get("sim_time", 0.0)):8.1f}, '
                    f'active={int(info.get("active_vehicles", 0)):4d}, '
                    f'gq={float(info.get("global_queue", 0.0)):7.1f}, '
                    f'gw={float(info.get("global_wait", 0.0)):9.1f}, '
                    f'arrived={int(info.get("total_arrived", 0)):5d}'
                )

            if bool(np.any(dones)):
                break
    finally:
        env.close()

    return summarize_samples('model', seed, samples)



def phase_pos_for_slot_in_controller(controller: dict[str, Any], slot: int) -> Optional[int]:
    for pos, phase in enumerate(controller.get("phases", [])):
        if int(phase.get("slot", -1)) == int(slot):
            return pos
    return None


def valid_action_mask_for_controller(controller: dict[str, Any]) -> np.ndarray:
    """Same 5-action mask as the single-target RL env, but for any TLS controller."""
    mask = np.zeros(5, dtype=bool)

    if controller.get("disabled") or controller.get("mode") != "green":
        mask[0] = True
        return mask

    elapsed = float(controller.get("phase_elapsed", 0.0))
    if elapsed < MIN_GREEN_BEFORE_SWITCH:
        mask[0] = True
        return mask

    if elapsed < HARD_MAX_GREEN:
        mask[0] = True

    current_pos = controller.get("phase_pos")
    for action in range(1, 5):
        phase_pos = phase_pos_for_slot_in_controller(controller, action - 1)
        if phase_pos is None or phase_pos == current_pos:
            continue
        mask[action] = True

    if not mask.any():
        mask[0] = True
    return mask


def apply_model_action_to_controller(controller: dict[str, Any], action: int) -> tuple[bool, bool]:
    """Apply one trained-policy action to one traffic-light controller."""
    action = int(action)
    mask = valid_action_mask_for_controller(controller)
    if action < 0 or action >= len(mask) or not mask[action]:
        action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

    elapsed = float(controller.get("phase_elapsed", 0.0))
    forced = False
    switched = False

    if action == 0:
        if controller.get("mode") == "green" and elapsed >= HARD_MAX_GREEN:
            switched = sim.switch_next_fixed_phase(controller)
            forced = True
        return switched, forced

    if controller.get("mode") != "green" or elapsed < MIN_GREEN_BEFORE_SWITCH:
        return False, False

    phase_pos = phase_pos_for_slot_in_controller(controller, action - 1)
    if phase_pos is None:
        return False, False

    switched = sim.request_switch(controller, phase_pos)
    return switched, forced


def run_all_model_episode(args: argparse.Namespace, scenario: TrafficScenario, seed: int) -> dict[str, float | int | str]:
    """Copy the same single-intersection policy onto every compatible TLS.

    This is an evaluation experiment only. The policy was trained as a shared
    local controller, not as a coordinated multi-agent controller.
    """
    env = ExactSimulationTrafficSignalEnv(
        tls_id=args.tls_id,
        episode_seconds=args.episode_seconds,
        gui=bool(args.gui),
        randomize_scenarios=False,
        base_seed=seed,
        env_rank=0,
        print_scenarios=args.print_scenarios,
        fixed_scenario=scenario,
        max_vehicle_center=args.max_vehicle_center,
        target_center=args.target_vehicle_center,
        initial_center=args.initial_vehicle_center,
        spawn_batch_center=args.spawn_batch_center,
        green_duration_center=args.green_duration_center,
        density_spread=args.density_spread,
        initial_spread=args.initial_spread,
    )

    # Load VecNormalize stats without starting another SUMO instance.
    dummy_raw_env = DummyVecEnv([
        lambda: Monitor(ExactSimulationTrafficSignalEnv(
            tls_id=args.tls_id,
            episode_seconds=args.episode_seconds,
            gui=False,
            randomize_scenarios=False,
            base_seed=seed,
            env_rank=999,
            print_scenarios=False,
            fixed_scenario=scenario,
            max_vehicle_center=args.max_vehicle_center,
            target_center=args.target_vehicle_center,
            initial_center=args.initial_vehicle_center,
            spawn_batch_center=args.spawn_batch_center,
            green_duration_center=args.green_duration_center,
            density_spread=args.density_spread,
            initial_spread=args.initial_spread,
        ))
    ])

    vecnorm_path = find_vecnormalize_path(Path(args.model_path), getattr(args, 'vecnormalize_path', None))
    if vecnorm_path is not None:
        print(f'[all-model seed {seed}] Loading VecNormalize stats from {vecnorm_path}')
        norm_env = VecNormalize.load(str(vecnorm_path), dummy_raw_env)
        norm_env.training = False
        norm_env.norm_reward = False
    else:
        print(f'[all-model seed {seed}] Warning: no VecNormalize stats found; using raw observations.')
        norm_env = VecNormalize(dummy_raw_env, norm_obs=False, norm_reward=False)
        norm_env.training = False

    model = load_maskable_ppo_for_inference(Path(args.model_path), norm_env, args.device)

    samples: list[dict[str, float]] = []
    try:
        _obs, _info = env.reset(seed=seed)
        decision_steps = max(1, int(round(sim.DECISION_INTERVAL / sim.STEP_LENGTH)))

        controllable_count = sum(1 for c in env.controllers if not c.get('disabled'))
        print(f'[all-model seed {seed}] controlling {controllable_count} traffic lights with copied policy')

        for step in range(1, args.eval_steps + 1):
            active_controllers = [c for c in env.controllers if not c.get('disabled')]
            if active_controllers:
                obs_batch = np.stack([
                    get_observation(controller, args.episode_seconds)
                    for controller in active_controllers
                ]).astype(np.float32)
                masks = np.stack([
                    valid_action_mask_for_controller(controller)
                    for controller in active_controllers
                ]).astype(bool)

                # The model was trained through VecNormalize, so apply the same
                # observation normalization before predicting controller actions.
                try:
                    normalized_obs = norm_env.normalize_obs(obs_batch)
                except Exception:
                    normalized_obs = obs_batch

                try:
                    actions, _ = model.predict(
                        normalized_obs,
                        deterministic=True,
                        action_masks=masks,
                    )
                except Exception:
                    actions, _ = model.predict(normalized_obs, deterministic=True)

                for controller, action in zip(active_controllers, np.asarray(actions).reshape(-1)):
                    apply_model_action_to_controller(controller, int(action))

            arrived, spawned, extended, recovered = sim.run_simulation_steps(
                num_steps=decision_steps,
                controllers=env.controllers,
                start_edges=env.main_start_edges,
                turn_index=env.turn_index,
                raw_graph=env.raw_graph,
                edge_metadata=env.edge_metadata,
                core_edges=env.core_edges,
                rng=env.rng,
                turn_counts=env.turn_counts,
                sim_state=env.sim_state,
                args=env.args,
            )
            env.total_arrived += arrived

            assert env.controller is not None
            target_wait, target_queue = target_wait_and_queue(env.controller)
            global_wait, global_queue, avg_speed = network_wait_queue_speed()
            info = env._info(
                spawned=spawned,
                extended=extended,
                recovered=recovered,
                arrived=arrived,
                target_wait=target_wait,
                target_queue=target_queue,
                global_wait=global_wait,
                global_queue=global_queue,
                avg_speed=avg_speed,
            )
            collect_sample(samples, info, reward=0.0)

            if step % args.eval_print_every == 0:
                print(
                    f'[all-model seed {seed}] step={step:6d}, '
                    f't={float(info.get("sim_time", 0.0)):8.1f}, '
                    f'active={int(info.get("active_vehicles", 0)):4d}, '
                    f'gq={float(info.get("global_queue", 0.0)):7.1f}, '
                    f'gw={float(info.get("global_wait", 0.0)):9.1f}, '
                    f'arrived={int(info.get("total_arrived", 0)):5d}'
                )

            if float(info.get('sim_time', 0.0)) >= float(args.episode_seconds):
                break
    finally:
        env.close()
        norm_env.close()

    return summarize_samples('all_model', seed, samples)

def run_fixed_cycle_episode(args: argparse.Namespace, scenario: TrafficScenario, seed: int) -> dict[str, float | int | str]:
    env = ExactSimulationTrafficSignalEnv(
        tls_id=args.tls_id,
        episode_seconds=args.episode_seconds,
        gui=bool(args.gui),
        randomize_scenarios=False,
        base_seed=seed,
        env_rank=0,
        print_scenarios=args.print_scenarios,
        fixed_scenario=scenario,
        max_vehicle_center=args.max_vehicle_center,
        target_center=args.target_vehicle_center,
        initial_center=args.initial_vehicle_center,
        spawn_batch_center=args.spawn_batch_center,
        green_duration_center=args.green_duration_center,
        density_spread=args.density_spread,
        initial_spread=args.initial_spread,
    )

    samples: list[dict[str, float]] = []
    try:
        _obs, _info = env.reset(seed=seed)
        decision_steps = max(1, int(round(sim.DECISION_INTERVAL / sim.STEP_LENGTH)))

        for step in range(1, args.eval_steps + 1):
            # This is the no-model baseline: every traffic light, including the
            # target TLS, follows the same fixed-cycle rule from the simulation.
            for controller in env.controllers:
                if (
                    not controller.get('disabled')
                    and controller['mode'] == 'green'
                    and controller['phase_elapsed'] >= controller['green_duration']
                ):
                    sim.switch_next_fixed_phase(controller)

            arrived, spawned, extended, recovered = sim.run_simulation_steps(
                num_steps=decision_steps,
                controllers=env.controllers,
                start_edges=env.main_start_edges,
                turn_index=env.turn_index,
                raw_graph=env.raw_graph,
                edge_metadata=env.edge_metadata,
                core_edges=env.core_edges,
                rng=env.rng,
                turn_counts=env.turn_counts,
                sim_state=env.sim_state,
                args=env.args,
            )
            env.total_arrived += arrived

            assert env.controller is not None
            target_wait, target_queue = target_wait_and_queue(env.controller)
            global_wait, global_queue, avg_speed = network_wait_queue_speed()
            info = env._info(
                spawned=spawned,
                extended=extended,
                recovered=recovered,
                arrived=arrived,
                target_wait=target_wait,
                target_queue=target_queue,
                global_wait=global_wait,
                global_queue=global_queue,
                avg_speed=avg_speed,
            )
            collect_sample(samples, info, reward=0.0)

            if step % args.eval_print_every == 0:
                print(
                    f'[fixed seed {seed}] step={step:6d}, '
                    f't={float(info.get("sim_time", 0.0)):8.1f}, '
                    f'active={int(info.get("active_vehicles", 0)):4d}, '
                    f'gq={float(info.get("global_queue", 0.0)):7.1f}, '
                    f'gw={float(info.get("global_wait", 0.0)):9.1f}, '
                    f'arrived={int(info.get("total_arrived", 0)):5d}'
                )

            if float(info.get('sim_time', 0.0)) >= float(args.episode_seconds):
                break
    finally:
        env.close()

    return summarize_samples('fixed_cycle', seed, samples)


def aggregate_by_controller(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controllers = sorted(set(str(row['controller']) for row in rows))
    result = []
    for controller in controllers:
        subset = [row for row in rows if row['controller'] == controller]
        out: dict[str, Any] = {'controller': controller, 'seed': 'mean', 'runs': len(subset)}
        numeric_keys = [
            key for key, value in subset[0].items()
            if key not in {'controller', 'seed'} and isinstance(value, (int, float))
        ]
        for key in numeric_keys:
            out[key] = safe_mean([float(row.get(key, 0.0)) for row in subset])
        result.append(out)
    return result


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    agg = {row['controller']: row for row in aggregate_by_controller(rows)}
    fixed = agg.get('fixed_cycle')
    single = agg.get('model')
    all_model = agg.get('all_model')
    if not fixed:
        print('Not enough data to print comparison table.')
        return

    metrics = [
        ('total_arrived', 'higher'),
        ('mean_avg_speed_mps', 'higher'),
        ('mean_global_queue', 'lower'),
        ('max_global_queue', 'lower'),
        ('mean_global_wait', 'lower'),
        ('max_global_wait', 'lower'),
        ('mean_target_queue', 'lower'),
        ('mean_target_wait', 'lower'),
        ('recovered_total', 'lower'),
    ]

    print('\n' + '=' * 118)
    print('FIXED CYCLE VS SINGLE-TLS MODEL VS COPIED ALL-TLS MODEL')
    print('=' * 118)
    print(
        f"{'metric':28s} {'fixed_cycle':>14s} "
        f"{'single_model':>14s} {'single imp.':>13s} "
        f"{'all_model':>14s} {'all imp.':>13s}"
    )
    print('-' * 118)

    for metric, direction in metrics:
        fixed_value = float(fixed.get(metric, 0.0) or 0.0)
        single_value = float(single.get(metric, 0.0) or 0.0) if single else 0.0
        all_value = float(all_model.get(metric, 0.0) or 0.0) if all_model else 0.0
        single_pct = improvement_percent(single_value, fixed_value, higher_is_better=(direction == 'higher')) if single else 0.0
        all_pct = improvement_percent(all_value, fixed_value, higher_is_better=(direction == 'higher')) if all_model else 0.0
        print(
            f'{metric:28s} {fixed_value:14.3f} '
            f'{single_value:14.3f} {single_pct:12.2f}% '
            f'{all_value:14.3f} {all_pct:12.2f}%'
        )

    print('=' * 118)
    print('Positive improvement means that controller beat the fixed-cycle baseline for that metric.')
    print('For queue/wait/recovery metrics, lower is better. For arrived/speed, higher is better.')
    print('The all_model row is the same single-TLS policy copied independently to every compatible TLS, not a separately trained multi-agent model.\n')

def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    csv_path = Path(args.stats_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote per-run CSV stats: {csv_path}')

    if args.stats_json:
        json_path = Path(args.stats_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'runs': rows,
            'aggregate': aggregate_by_controller(rows),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'Wrote JSON stats: {json_path}')


def compare_model_to_fixed_cycle(args: argparse.Namespace) -> None:
    seeds = parse_seed_list(args.compare_seeds)
    all_rows: list[dict[str, Any]] = []

    if args.gui and len(seeds) > 1:
        print('WARNING: --gui with multiple seeds will open/close SUMO GUI repeatedly and will be slow.')

    for seed in seeds:
        print('\n' + '#' * 92)
        print(f'Comparing seed {seed}')
        print('#' * 92)
        scenario = build_fixed_scenario(seed=seed, args=args)

        # Always use the exact same scenario object for all controller modes.
        if not args.skip_fixed:
            all_rows.append(run_fixed_cycle_episode(args, scenario, seed))
        if not args.skip_single_model:
            all_rows.append(run_model_episode(args, scenario, seed))
        if not args.skip_all_model:
            all_rows.append(run_all_model_episode(args, scenario, seed))

    print_comparison_table(all_rows)
    write_outputs(all_rows, args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare fixed-cycle, one-TLS RL, and copied all-TLS RL control in the realistic SUMO simulation.'
    )
    parser.add_argument('--episode-seconds', type=int, default=3600)
    parser.add_argument('--eval-steps', type=int, default=10_000)
    parser.add_argument('--eval-print-every', type=int, default=20)
    parser.add_argument('--tls-id', default=TARGET_TLS_ID)
    parser.add_argument('--model-path', default=MODEL_DEFAULT)
    parser.add_argument('--vecnormalize-path', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare-seeds', default='42')
    parser.add_argument('--stats-csv', default='model_vs_fixed_cycle_stats.csv')
    parser.add_argument('--stats-json', default='model_vs_fixed_cycle_stats.json')
    parser.add_argument('--run-order', choices=['fixed_first', 'model_first', 'both'], default='fixed_first', help='Deprecated; kept for compatibility.')
    parser.add_argument('--skip-fixed', action='store_true')
    parser.add_argument('--skip-single-model', action='store_true')
    parser.add_argument('--skip-all-model', action='store_true')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI. For statistics, headless mode is much faster.')
    parser.add_argument('--print-scenarios', action='store_true')
    parser.add_argument('--device', default='auto')

    # Same realistic-simulation center settings as the exact evaluator.
    parser.add_argument('--max-vehicle-center', type=int, default=SIM_CENTER_MAX_VEHICLES)
    parser.add_argument('--target-vehicle-center', type=int, default=SIM_CENTER_TARGET_VEHICLES)
    parser.add_argument('--initial-vehicle-center', type=int, default=SIM_CENTER_INITIAL_VEHICLES)
    parser.add_argument('--spawn-batch-center', type=int, default=SIM_CENTER_SPAWN_BATCH)
    parser.add_argument('--green-duration-center', type=float, default=SIM_CENTER_GREEN_DURATION)
    parser.add_argument('--density-spread', type=float, default=0.0)
    parser.add_argument('--initial-spread', type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_vehicle_center = min(int(args.max_vehicle_center), SIM_CENTER_MAX_VEHICLES)
    args.target_vehicle_center = min(int(args.target_vehicle_center), args.max_vehicle_center)
    args.initial_vehicle_center = min(int(args.initial_vehicle_center), args.target_vehicle_center)
    args.spawn_batch_center = max(1, int(args.spawn_batch_center))
    args.density_spread = min(0.60, max(0.0, float(args.density_spread)))
    args.initial_spread = min(1.50, max(0.0, float(args.initial_spread)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    compare_model_to_fixed_cycle(args)


if __name__ == '__main__':
    main()
