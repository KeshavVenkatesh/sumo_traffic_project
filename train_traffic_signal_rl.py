#!/usr/bin/env python3
"""
Train a MaskablePPO traffic-signal controller under the exact same traffic
simulation mechanics as realistic_all_intersections_fixed_cycle.py.

This training script imports the current simulation file directly.  The only
intentional simulation difference during training/evaluation is that ambulances
are disabled.  Everything else is inherited from the simulation module:

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

This turbo version keeps the same SUMO mechanics but removes the training-only
bottlenecks that made exact-simulation learning too slow: it reuses a large cache
of valid SUMO fastest OD routes, throttles expensive helper scans during training,
uses lane-level queue/wait measurements, caches global metrics, suppresses reset
prints, and keeps the PPO network lightweight on CPU. Evaluation can still run the
normal simulation in the GUI.

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
import contextlib
import hashlib
import io
import json
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
MODEL_DEFAULT = "models/traffic_signal_maskable_ppo_exact_sim"

# Center values match the simulation command you have been using for the current
# file.  Training randomizes around these, but evaluation can lock to them.
SIM_CENTER_MAX_VEHICLES = int(getattr(sim, "MAX_ACTIVE_VEHICLE_CAP", 750))
SIM_CENTER_TARGET_VEHICLES = 650
SIM_CENTER_INITIAL_VEHICLES = 200
SIM_CENTER_SPAWN_BATCH = 12
SIM_CENTER_ROUTE_LOOKAHEAD = 60
SIM_CENTER_MIN_REMAINING = 15
SIM_CENTER_GREEN_DURATION = 45.0
SIM_CENTER_SPAWN_GRID_SIZE = 6
SIM_CENTER_TELEPORT = 180
SIM_CENTER_MAX_DEPART_DELAY = 300

# Training-only acceleration defaults. These do not change SUMO car-following,
# routing legality, traffic-light states, or lane-change safety checks. They
# reduce repeated Python/TraCI scans that are useful for GUI debugging but too
# expensive to run every simulated second during PPO data collection.
DEFAULT_ROUTE_POOL_SIZE = 2500
DEFAULT_ROUTE_POOL_MIN_SIZE = 800
DEFAULT_ROUTE_POOL_DIR = ".route_cache"
DEFAULT_TRAINING_KEEP_CLEAR_INTERVAL = 2.0
DEFAULT_TRAINING_LANE_LOCK_INTERVAL = 2.0
DEFAULT_TRAINING_UNCONNECTED_INTERVAL = 4.0
DEFAULT_TRAINING_LANE_PREF_INTERVAL = 4.0
DEFAULT_TRAINING_LANE_BALANCE_INTERVAL = 10.0
DEFAULT_TRAINING_SIGNAL_UPDATE_INTERVAL = 4.0

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


# Keep original simulation functions so training wrappers can fall back to the
# exact simulation behavior whenever their fast path is disabled.
_ORIGINAL_SIM_BUILD_OD_ROUTE = sim.build_od_route
_ORIGINAL_KEEP_CLEAR_ALL = sim.apply_keep_clear_and_right_of_way_to_all_vehicles
_ORIGINAL_TLS_LANE_LOCK_ALL = sim.apply_traffic_light_lane_change_lock_to_all_vehicles

_TRAINING_FAST_RUNTIME_ENABLED = False
_TRAINING_KEEP_CLEAR_INTERVAL = 1.0
_TRAINING_LANE_LOCK_INTERVAL = 1.0
_TRAINING_NEXT_KEEP_CLEAR_TIME = 0.0
_TRAINING_NEXT_LANE_LOCK_TIME = 0.0


def _throttled_keep_clear_all():
    global _TRAINING_NEXT_KEEP_CLEAR_TIME
    if not _TRAINING_FAST_RUNTIME_ENABLED or _TRAINING_KEEP_CLEAR_INTERVAL <= 1.0:
        return _ORIGINAL_KEEP_CLEAR_ALL()

    now = sim.current_sim_time() if hasattr(sim, "current_sim_time") else sim.traci.simulation.getTime()
    if now + 1e-9 < _TRAINING_NEXT_KEEP_CLEAR_TIME:
        return 0

    _TRAINING_NEXT_KEEP_CLEAR_TIME = now + _TRAINING_KEEP_CLEAR_INTERVAL
    return _ORIGINAL_KEEP_CLEAR_ALL()


def _throttled_tls_lane_lock_all():
    global _TRAINING_NEXT_LANE_LOCK_TIME
    if not _TRAINING_FAST_RUNTIME_ENABLED or _TRAINING_LANE_LOCK_INTERVAL <= 1.0:
        return _ORIGINAL_TLS_LANE_LOCK_ALL()

    now = sim.current_sim_time() if hasattr(sim, "current_sim_time") else sim.traci.simulation.getTime()
    if now + 1e-9 < _TRAINING_NEXT_LANE_LOCK_TIME:
        return 0

    _TRAINING_NEXT_LANE_LOCK_TIME = now + _TRAINING_LANE_LOCK_INTERVAL
    return _ORIGINAL_TLS_LANE_LOCK_ALL()


def configure_training_runtime_acceleration(args: argparse.Namespace, eval_mode: bool = False) -> None:
    """Reduce repeated TraCI scans during training while keeping mechanics active.

    The exact simulation uses several watchdog/helper scans every simulated
    second.  They are valuable for interactive debugging, but they dominate PPO
    collection time.  This keeps the same helpers enabled, only with a lower
    cadence during training.  GUI evaluation uses exact cadence unless
    --fast-eval-runtime is explicitly set.
    """
    global _TRAINING_FAST_RUNTIME_ENABLED
    global _TRAINING_KEEP_CLEAR_INTERVAL, _TRAINING_LANE_LOCK_INTERVAL
    global _TRAINING_NEXT_KEEP_CLEAR_TIME, _TRAINING_NEXT_LANE_LOCK_TIME

    fast_runtime = (not getattr(args, "exact_runtime_cadence", False)) and (not eval_mode or getattr(args, "fast_eval_runtime", False))
    _TRAINING_FAST_RUNTIME_ENABLED = bool(fast_runtime)
    _TRAINING_NEXT_KEEP_CLEAR_TIME = 0.0
    _TRAINING_NEXT_LANE_LOCK_TIME = 0.0

    if not fast_runtime:
        _TRAINING_KEEP_CLEAR_INTERVAL = 1.0
        _TRAINING_LANE_LOCK_INTERVAL = 1.0
        # Restore the simulation's normal fast-debug cadence when requested.
        if hasattr(sim, "UNCONNECTED_LANE_RESCUE_INTERVAL"):
            sim.UNCONNECTED_LANE_RESCUE_INTERVAL = 1.0
        if hasattr(sim, "LANE_PREF_INTERVAL"):
            sim.LANE_PREF_INTERVAL = 1.0
        if hasattr(sim, "LANE_BALANCE_INTERVAL"):
            sim.LANE_BALANCE_INTERVAL = 2.0
        if hasattr(sim, "SIGNAL_UPDATE_INTERVAL"):
            sim.SIGNAL_UPDATE_INTERVAL = 2.0
        return

    _TRAINING_KEEP_CLEAR_INTERVAL = max(1.0, float(getattr(args, "training_keep_clear_interval", DEFAULT_TRAINING_KEEP_CLEAR_INTERVAL)))
    _TRAINING_LANE_LOCK_INTERVAL = max(1.0, float(getattr(args, "training_lane_lock_interval", DEFAULT_TRAINING_LANE_LOCK_INTERVAL)))

    if hasattr(sim, "UNCONNECTED_LANE_RESCUE_INTERVAL"):
        sim.UNCONNECTED_LANE_RESCUE_INTERVAL = max(1.0, float(getattr(args, "training_unconnected_interval", DEFAULT_TRAINING_UNCONNECTED_INTERVAL)))
    if hasattr(sim, "LANE_PREF_INTERVAL"):
        sim.LANE_PREF_INTERVAL = max(1.0, float(getattr(args, "training_lane_pref_interval", DEFAULT_TRAINING_LANE_PREF_INTERVAL)))
    if hasattr(sim, "LANE_BALANCE_INTERVAL"):
        sim.LANE_BALANCE_INTERVAL = max(2.0, float(getattr(args, "training_lane_balance_interval", DEFAULT_TRAINING_LANE_BALANCE_INTERVAL)))
    if hasattr(sim, "SIGNAL_UPDATE_INTERVAL"):
        sim.SIGNAL_UPDATE_INTERVAL = max(1.0, float(getattr(args, "training_signal_update_interval", DEFAULT_TRAINING_SIGNAL_UPDATE_INTERVAL)))


# Install wrappers once. They are no-ops unless configure_training_runtime_acceleration
# enables them for a training episode.
sim.apply_keep_clear_and_right_of_way_to_all_vehicles = _throttled_keep_clear_all
sim.apply_traffic_light_lane_change_lock_to_all_vehicles = _throttled_tls_lane_lock_all


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

    # The only intentional simulation difference for training/eval.
    args.disable_ambulances = True
    args.ambulance_interval = 0.0
    args.ambulance_min_euclidean_distance = float(getattr(sim, "AMBULANCE_MIN_EUCLIDEAN_DISTANCE", 1500.0))
    args.ambulance_min_route_distance = float(getattr(sim, "AMBULANCE_MIN_ROUTE_DISTANCE", 1800.0))
    args.ambulance_min_route_edges = int(getattr(sim, "AMBULANCE_MIN_ROUTE_EDGES", 20))
    args.ambulance_route_attempts = int(getattr(sim, "AMBULANCE_ROUTE_ATTEMPTS", 100))
    args.ambulance_depart_lane = getattr(sim, "AMBULANCE_DEPART_LANE", "free")
    args.ambulance_depart_pos = getattr(sim, "AMBULANCE_DEPART_POS", "random_free")
    args.ambulance_poi_radius = float(getattr(sim, "AMBULANCE_POI_RADIUS", 250.0))
    args.ambulance_debug = False

    return args




def _route_cache_key(args: SimpleNamespace) -> str:
    try:
        net_mtime = os.path.getmtime(sim.NET_FILE)
    except OSError:
        net_mtime = 0.0

    payload = {
        "net": os.path.abspath(sim.NET_FILE),
        "net_mtime": round(float(net_mtime), 3),
        "routing_mode": getattr(args, "routing_mode", ROUTING_MODE_OD),
        "od_boundary_margin_fraction": getattr(args, "od_boundary_margin_fraction", None),
        "od_min_euclidean_distance": getattr(args, "od_min_euclidean_distance", None),
        "od_min_route_distance": getattr(args, "od_min_route_distance", None),
        "od_min_zone_separation": getattr(args, "od_min_zone_separation", None),
        "od_max_local_middle_fraction": getattr(args, "od_max_local_middle_fraction", None),
        "od_local_middle_trim_edges": getattr(args, "od_local_middle_trim_edges", None),
        "od_through_trip_probability": getattr(args, "od_through_trip_probability", None),
        "od_access_trip_probability": getattr(args, "od_access_trip_probability", None),
        "od_long_local_trip_probability": getattr(args, "od_long_local_trip_probability", None),
        "od_min_edge_length": getattr(args, "od_min_edge_length", None),
        "blocked_loop_edges": sorted(getattr(sim, "HARDCODED_NO_CRUISE_LOOP_EDGES", set())),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _load_cached_routes(cache_file: Path) -> list[dict[str, Any]]:
    if not cache_file.exists():
        return []
    try:
        data = json.loads(cache_file.read_text())
    except Exception:
        return []
    routes = data.get("routes", []) if isinstance(data, dict) else []
    good = []
    for item in routes:
        edges = item.get("edges") if isinstance(item, dict) else None
        if isinstance(edges, list) and len(edges) >= 2:
            good.append({"edges": [str(edge) for edge in edges], "info": dict(item.get("info", {}))})
    return good


def _save_cached_routes(cache_file: Path, routes: list[dict[str, Any]], key: str) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "routes": routes}, separators=(",", ":")))
        tmp.replace(cache_file)
    except Exception:
        pass


class TrainingODRoutePool:
    """Large reusable pool of legal SUMO fastest OD routes.

    The normal simulation calls findRoute repeatedly during vehicle spawning. In
    training that becomes a bottleneck because the same map and OD constraints
    are used every episode. This pool is built from the same OD generator and
    then sampled during spawning, so routes remain valid fastest SUMO routes but
    route computation is not repeated for every car.
    """

    def __init__(self, routes: list[dict[str, Any]]):
        self.routes = routes

    def __len__(self) -> int:
        return len(self.routes)

    def sample(self, rng: random.Random) -> tuple[list[str], dict[str, Any]]:
        item = rng.choice(self.routes)
        info = dict(item.get("info", {}))
        info["trip_type"] = info.get("trip_type", "cached_od")
        info["cached_od_route"] = True
        return list(item["edges"]), info


def build_training_route_pool(
    sim_state: dict[str, Any],
    context: Any,
    raw_graph: dict[str, list[str]],
    edge_metadata: dict[str, Any],
    rng: random.Random,
    args: SimpleNamespace,
    pool_size: int,
    pool_min_size: int,
    cache_dir: str,
    use_cache: bool,
    quiet: bool,
) -> TrainingODRoutePool | None:
    if pool_size <= 0 or context is None or not sim.use_od_routing(args):
        return None

    key = _route_cache_key(args)
    cache_file = Path(cache_dir).expanduser() / f"od_routes_{key}.json"
    routes = _load_cached_routes(cache_file) if use_cache else []

    if len(routes) >= max(1, pool_min_size):
        if not quiet:
            print(f"Loaded {len(routes)} cached OD routes from {cache_file}")
        return TrainingODRoutePool(routes[:max(pool_size, len(routes))])

    # Build the missing routes using the original simulation OD generator. Keep
    # a separate state so route-pool construction does not consume vehicle IDs.
    seen = {tuple(item["edges"]) for item in routes}
    pool_state = {"next_od_origin_zone_index": sim_state.get("next_od_origin_zone_index", 0)}
    attempts = 0
    max_attempts = max(pool_size * 8, pool_min_size * 12, 2000)

    if not quiet:
        print(f"Building OD route pool: {len(routes)}/{pool_size} cached routes available...")

    while len(routes) < pool_size and attempts < max_attempts:
        attempts += 1
        try:
            route_edges, od_info = _ORIGINAL_SIM_BUILD_OD_ROUTE(
                sim_state=pool_state,
                context=context,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                rng=rng,
                args=args,
            )
        except sim.traci.TraCIException:
            continue
        except Exception:
            continue

        if not route_edges or len(route_edges) < 2:
            continue
        route_tuple = tuple(route_edges)
        if route_tuple in seen:
            continue
        seen.add(route_tuple)
        routes.append({"edges": list(route_edges), "info": dict(od_info or {"trip_type": "cached_od"})})

    if use_cache and routes:
        _save_cached_routes(cache_file, routes, key)

    if not quiet:
        print(f"OD route pool ready: {len(routes)} routes after {attempts} attempts")

    if len(routes) < max(1, pool_min_size):
        # Fall back to exact on-demand route generation if the pool is too small.
        return None
    return TrainingODRoutePool(routes)


def _training_build_od_route(sim_state, context, raw_graph, edge_metadata, rng, args):
    pool = sim_state.get("_training_od_route_pool") if isinstance(sim_state, dict) else None
    if pool is not None and len(pool) > 0:
        return pool.sample(rng)
    return _ORIGINAL_SIM_BUILD_OD_ROUTE(sim_state, context, raw_graph, edge_metadata, rng, args)


sim.build_od_route = _training_build_od_route


def lane_level_wait_and_queue(lanes: Any) -> tuple[float, float]:
    """Fast queue/wait measurement using SUMO lane aggregates.

    This avoids one TraCI round trip per vehicle for every observation and reward
    calculation. It does not change vehicle behavior; it only changes how the RL
    wrapper measures the current state.
    """
    queue = 0.0
    wait = 0.0
    for lane_id in lanes:
        try:
            queue += float(sim.traci.lane.getLastStepHaltingNumber(lane_id))
            wait += float(sim.traci.lane.getWaitingTime(lane_id))
        except sim.traci.TraCIException:
            continue
    return queue, wait


def movement_queue_and_wait(controller: dict[str, Any], movement_label: str) -> tuple[float, float]:
    return lane_level_wait_and_queue(controller["movement_in_lanes_cache"].get(movement_label, set()))


def target_wait_and_queue(controller: dict[str, Any]) -> tuple[float, float]:
    # Return order matches the old helper: wait first, queue second.
    queue, wait = lane_level_wait_and_queue(controller.get("all_in_lanes", set()))
    return wait, queue


_VEH_SUB_VARS = (
    sim.traci.constants.VAR_SPEED,
    sim.traci.constants.VAR_WAITING_TIME,
)
_SUBSCRIBED_VEHICLES: set[str] = set()


def _ensure_vehicle_subscriptions() -> None:
    """Subscribe every currently-known vehicle to speed/waiting-time once.

    SUMO automatically includes subscription results for newly departed
    vehicles in getAllSubscriptionResults() once subscribed, and TraCI
    auto-subscribes vehicles that already have a context subscription on
    the simulation step. To keep this simple and robust across SUMO
    versions, we instead use a single context subscription anchored to
    the network bounding box, which covers every vehicle without per
    vehicle subscribe calls and without per vehicle getSpeed/getWaitingTime
    round trips.
    """
    pass


def network_wait_queue_speed() -> tuple[float, float, float]:
    """Fast global queue/wait/speed using one bulk TraCI call instead of
    two TraCI calls per vehicle. getAllSubscriptionResults requires each
    vehicle to be subscribed; we instead use getContextSubscriptionResults
    via a single junction-anchored context subscription set up once in
    _build_episode. If that context subscription is unavailable for any
    reason, fall back to the per-vehicle loop so behavior never breaks.
    """
    try:
        results = sim.traci.vehicle.getAllContextSubscriptionResults()
    except Exception:
        results = None

    if results:
        total_wait = 0.0
        total_queue = 0.0
        total_speed = 0.0
        count = 0
        # getAllContextSubscriptionResults is keyed by the subscribing object;
        # we only ever issue one such subscription (see _ensure_global_context_subscription),
        # so take its single value dict.
        for veh_data in results.values():
            for veh_id, vals in veh_data.items():
                if str(veh_id).startswith("ambulance_"):
                    continue
                speed = vals.get(sim.traci.constants.VAR_SPEED)
                wait = vals.get(sim.traci.constants.VAR_WAITING_TIME)
                if speed is None or wait is None:
                    continue
                count += 1
                total_speed += speed
                total_wait += wait
                if speed < sim.QUEUE_SPEED_THRESHOLD:
                    total_queue += 1.0
            break  # only one subscriber expected
        if count:
            return total_wait, total_queue, total_speed / count

    # Fallback: original per-vehicle scan (still correct, just slower).
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


def _ensure_global_context_subscription() -> None:
    """Issue one junction-anchored context subscription that reports
    speed and waiting time for every vehicle in the network. This replaces
    O(active_vehicles) TraCI round trips per call with O(1).
    """
    try:
        junction_ids = sim.traci.junction.getIDList()
        if not junction_ids:
            return
        anchor = junction_ids[0]
        sim.traci.junction.subscribeContext(
            anchor,
            sim.traci.constants.CMD_GET_VEHICLE_VARIABLE,
            1_000_000.0,  # effectively unlimited radius
            _VEH_SUB_VARS,
        )
    except Exception:
        pass


def phase_slot(controller: dict[str, Any]) -> int:
    try:
        return int(controller["phases"][controller["phase_pos"]].get("slot", -1))
    except Exception:
        return -1


def get_observation(
    controller: dict[str, Any],
    episode_seconds: int,
    network_metrics: Optional[tuple[float, float, float]] = None,
) -> np.ndarray:
    obs: list[float] = []

    for label in sim.MOVEMENT_LABELS:
        queue, wait = movement_queue_and_wait(controller, label)
        obs.append(queue / 100.0)
        obs.append(wait / 1200.0)

    phase_one_hot = [0.0, 0.0, 0.0, 0.0]
    slot = phase_slot(controller)
    if 0 <= slot < 4:
        phase_one_hot[slot] = 1.0
    obs.extend(phase_one_hot)

    obs.append(float(controller.get("phase_elapsed", 0.0)) / HARD_MAX_GREEN)
    try:
        sim_time = sim.traci.simulation.getTime()
    except sim.traci.TraCIException:
        sim_time = 0.0
    obs.append(sim_time / max(1.0, float(episode_seconds)))

    if network_metrics is None:
        global_wait, global_queue, avg_speed = network_wait_queue_speed()
    else:
        global_wait, global_queue, avg_speed = network_metrics
    try:
        active = sim.traci.vehicle.getIDCount()
    except sim.traci.TraCIException:
        active = 0
    obs.append(active / max(1.0, float(SIM_CENTER_MAX_VEHICLES)))
    obs.append(global_queue / 300.0)
    obs.append(global_wait / 25000.0)
    obs.append(avg_speed / 20.0)

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
        global_metric_interval: int = 2,
        quiet_episode_build: bool = True,
        route_pool_size: int = DEFAULT_ROUTE_POOL_SIZE,
        route_pool_min_size: int = DEFAULT_ROUTE_POOL_MIN_SIZE,
        route_pool_cache: bool = True,
        route_pool_dir: str = DEFAULT_ROUTE_POOL_DIR,
        exact_runtime_cadence: bool = False,
        fast_eval_runtime: bool = False,
        training_keep_clear_interval: float = DEFAULT_TRAINING_KEEP_CLEAR_INTERVAL,
        training_lane_lock_interval: float = DEFAULT_TRAINING_LANE_LOCK_INTERVAL,
        training_unconnected_interval: float = DEFAULT_TRAINING_UNCONNECTED_INTERVAL,
        training_lane_pref_interval: float = DEFAULT_TRAINING_LANE_PREF_INTERVAL,
        training_lane_balance_interval: float = DEFAULT_TRAINING_LANE_BALANCE_INTERVAL,
        training_signal_update_interval: float = DEFAULT_TRAINING_SIGNAL_UPDATE_INTERVAL,
    ):
        super().__init__()
        self.tls_id = tls_id
        self.episode_seconds = int(episode_seconds)
        self.gui = bool(gui)
        self.randomize_scenarios = bool(randomize_scenarios)
        self.env_rank = int(env_rank)
        self.print_scenarios = bool(print_scenarios)
        self.fixed_scenario = fixed_scenario
        self.global_metric_interval = max(1, int(global_metric_interval))
        self.quiet_episode_build = bool(quiet_episode_build)
        self.route_pool_size = max(0, int(route_pool_size))
        self.route_pool_min_size = max(0, int(route_pool_min_size))
        self.route_pool_cache = bool(route_pool_cache)
        self.route_pool_dir = str(route_pool_dir)
        self.runtime_args = argparse.Namespace(
            exact_runtime_cadence=bool(exact_runtime_cadence),
            fast_eval_runtime=bool(fast_eval_runtime),
            training_keep_clear_interval=float(training_keep_clear_interval),
            training_lane_lock_interval=float(training_lane_lock_interval),
            training_unconnected_interval=float(training_unconnected_interval),
            training_lane_pref_interval=float(training_lane_pref_interval),
            training_lane_balance_interval=float(training_lane_balance_interval),
            training_signal_update_interval=float(training_signal_update_interval),
        )
        self.step_count = 0
        self.cached_global_metrics: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
        self.route_pool: TrainingODRoutePool | None = None

        self.prev_target_wait = 0.0
        self.prev_target_queue = 0.0
        self.prev_global_wait = 0.0
        self.prev_global_queue = 0.0
        self.total_arrived = 0

        self.action_space = spaces.Discrete(5)  # 0=hold, 1..4=switch to phase slot 0..3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

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

    def _maybe_quiet(self):
        if self.quiet_episode_build and not self.gui and not self.print_scenarios:
            return contextlib.redirect_stdout(io.StringIO())
        return contextlib.nullcontext()

    def _refresh_global_metrics(self, force: bool = False) -> tuple[float, float, float]:
        if force or self.step_count % self.global_metric_interval == 0:
            self.cached_global_metrics = network_wait_queue_speed()
        return self.cached_global_metrics

    def _build_episode(self) -> None:
        assert self.scenario is not None
        assert self.args is not None
        assert self.rng is not None

        configure_training_runtime_acceleration(self.runtime_args, eval_mode=self.gui)
        reset_sim_globals()
        apply_sim_distance_globals(self.scenario)
        sim.write_empty_route_file(self.route_file)

        if self.gui:
            sim.ensure_xquartz()

        sim.traci.start(self._sumo_cmd())
        self.started = True

        with self._maybe_quiet():
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
            "next_ambulance_spawn": float("inf"),
            "active_ambulances": {},
            "od_context": od_context,
            "od_trip_counts": Counter(),
            "od_movement_counts": Counter(),
            "od_route_failures": 0,
        }

        self.route_pool = None
        if od_context is not None and self.route_pool_size > 0:
            with self._maybe_quiet():
                self.route_pool = build_training_route_pool(
                    sim_state=self.sim_state,
                    context=od_context,
                    raw_graph=self.raw_graph,
                    edge_metadata=self.edge_metadata,
                    rng=self.rng,
                    args=self.args,
                    pool_size=self.route_pool_size,
                    pool_min_size=self.route_pool_min_size,
                    cache_dir=self.route_pool_dir,
                    use_cache=self.route_pool_cache,
                    quiet=self.quiet_episode_build and not self.print_scenarios,
                )
            if self.route_pool is not None:
                self.sim_state["_training_od_route_pool"] = self.route_pool

        with self._maybe_quiet():
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

        _ensure_global_context_subscription()

        self.step_count = 0
        self.prev_target_wait, self.prev_target_queue = target_wait_and_queue(self.controller)
        self.cached_global_metrics = network_wait_queue_speed()
        self.prev_global_wait, self.prev_global_queue, _ = self.cached_global_metrics
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
        return get_observation(
            self.controller,
            self.episode_seconds,
            network_metrics=self.cached_global_metrics,
        ), self._info()

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
        self.step_count += 1

        target_wait, target_queue = target_wait_and_queue(self.controller)
        global_wait, global_queue, avg_speed = self._refresh_global_metrics(force=False)

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
        obs = get_observation(
            self.controller,
            self.episode_seconds,
            network_metrics=(global_wait, global_queue, avg_speed),
        )

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
            global_metric_interval=args.global_metric_interval,
            quiet_episode_build=args.quiet_episode_build,
            route_pool_size=0 if args.disable_route_pool else args.route_pool_size,
            route_pool_min_size=args.route_pool_min_size,
            route_pool_cache=args.route_pool_cache,
            route_pool_dir=args.route_pool_dir,
            exact_runtime_cadence=args.exact_runtime_cadence,
            fast_eval_runtime=args.fast_eval_runtime,
            training_keep_clear_interval=args.training_keep_clear_interval,
            training_lane_lock_interval=args.training_lane_lock_interval,
            training_unconnected_interval=args.training_unconnected_interval,
            training_lane_pref_interval=args.training_lane_pref_interval,
            training_lane_balance_interval=args.training_lane_balance_interval,
            training_signal_update_interval=args.training_signal_update_interval,
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
        model = MaskablePPO.load(str(model_path), env=train_env, device=args.device)
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
    print(f"  global metric interval:  {args.global_metric_interval}")
    print(f"  route pool:              {'disabled' if args.disable_route_pool else str(args.route_pool_size) + ' routes'}")
    print(f"  route pool cache:        {args.route_pool_cache} ({args.route_pool_dir})")
    print(f"  exact runtime cadence:   {args.exact_runtime_cadence}")
    print(f"  keep-clear interval:     {args.training_keep_clear_interval}")
    print(f"  lane-lock interval:      {args.training_lane_lock_interval}")
    print(f"  unconnected interval:    {args.training_unconnected_interval}")
    print(f"  lane-pref interval:      {args.training_lane_pref_interval}")
    print(f"  lane-balance interval:   {args.training_lane_balance_interval}")
    print(f"  torch threads:           {args.torch_threads}")
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


def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    vecnorm_path = str(model_path.with_suffix(".vecnormalize.pkl"))

    eval_env = build_vec_env(args, eval_mode=True)
    if Path(vecnorm_path).exists():
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print(f"Warning: no VecNormalize stats found at {vecnorm_path}; using raw observations.")
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
        eval_env.training = False

    model = MaskablePPO.load(str(model_path), env=eval_env, device=args.device)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate MaskablePPO in the exact current SUMO simulation conditions."
    )
    parser.add_argument("--timesteps", type=int, default=750_000)
    parser.add_argument("--episode-seconds", type=int, default=900)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--tls-id", default=TARGET_TLS_ID)
    parser.add_argument("--model-path", default=MODEL_DEFAULT)
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--randomize-scenarios", action="store_true", default=True)
    parser.add_argument("--no-randomize-scenarios", action="store_false", dest="randomize_scenarios")
    parser.add_argument("--print-scenarios", action="store_true")

    parser.add_argument("--max-vehicle-center", type=int, default=SIM_CENTER_MAX_VEHICLES)
    parser.add_argument("--target-vehicle-center", type=int, default=SIM_CENTER_TARGET_VEHICLES)
    parser.add_argument("--initial-vehicle-center", type=int, default=SIM_CENTER_INITIAL_VEHICLES)
    parser.add_argument("--spawn-batch-center", type=int, default=SIM_CENTER_SPAWN_BATCH)
    parser.add_argument("--green-duration-center", type=float, default=SIM_CENTER_GREEN_DURATION)
    parser.add_argument("--density-spread", type=float, default=0.18)
    parser.add_argument("--initial-spread", type=float, default=0.55)
    parser.add_argument(
        "--global-metric-interval",
        type=int,
        default=2,
        help=(
            "Scan all vehicles for global reward/observation metrics every N RL steps. "
            "Target-intersection metrics are still measured every step. Use 1 for the exact old metric cadence."
        ),
    )
    parser.add_argument(
        "--quiet-episode-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress verbose simulation topology prints during training resets.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for PyTorch. Small PPO MLPs usually train faster with 1 so SUMO gets the CPU.",
    )
    parser.add_argument(
        "--route-pool-size",
        type=int,
        default=DEFAULT_ROUTE_POOL_SIZE,
        help="Number of legal fastest OD routes to cache and sample during spawning.",
    )
    parser.add_argument(
        "--route-pool-min-size",
        type=int,
        default=DEFAULT_ROUTE_POOL_MIN_SIZE,
        help="Minimum cached routes required before using the route pool.",
    )
    parser.add_argument("--route-pool-dir", default=DEFAULT_ROUTE_POOL_DIR)
    parser.add_argument("--route-pool-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-route-pool", action="store_true")
    parser.add_argument(
        "--exact-runtime-cadence",
        action="store_true",
        help="Use the simulation's original per-second helper cadence. Slower, useful for final fine-tuning.",
    )
    parser.add_argument(
        "--fast-eval-runtime",
        action="store_true",
        help="Use the accelerated helper cadence even during eval. GUI eval defaults to exact cadence.",
    )
    parser.add_argument("--training-keep-clear-interval", type=float, default=DEFAULT_TRAINING_KEEP_CLEAR_INTERVAL)
    parser.add_argument("--training-lane-lock-interval", type=float, default=DEFAULT_TRAINING_LANE_LOCK_INTERVAL)
    parser.add_argument("--training-unconnected-interval", type=float, default=DEFAULT_TRAINING_UNCONNECTED_INTERVAL)
    parser.add_argument("--training-lane-pref-interval", type=float, default=DEFAULT_TRAINING_LANE_PREF_INTERVAL)
    parser.add_argument("--training-lane-balance-interval", type=float, default=DEFAULT_TRAINING_LANE_BALANCE_INTERVAL)
    parser.add_argument("--training-signal-update-interval", type=float, default=DEFAULT_TRAINING_SIGNAL_UPDATE_INTERVAL)

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--eval-fixed-scenario", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--eval-print-every", type=int, default=20)

    parser.add_argument("--lr-start", type=float, default=2.5e-4)
    parser.add_argument("--lr-end", type=float, default=5e-5)
    parser.add_argument("--clip-start", type=float, default=0.20)
    parser.add_argument("--clip-end", type=float, default=0.10)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.012)
    parser.add_argument("--vf-coef", type=float, default=0.65)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.035)
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-reward", type=float, default=10.0)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_vehicle_center = min(int(args.max_vehicle_center), SIM_CENTER_MAX_VEHICLES)
    args.target_vehicle_center = min(int(args.target_vehicle_center), args.max_vehicle_center)
    args.initial_vehicle_center = min(int(args.initial_vehicle_center), args.target_vehicle_center)
    args.spawn_batch_center = max(1, int(args.spawn_batch_center))
    args.density_spread = min(0.60, max(0.0, float(args.density_spread)))
    args.initial_spread = min(1.50, max(0.0, float(args.initial_spread)))
    args.global_metric_interval = max(1, int(args.global_metric_interval))
    args.route_pool_size = max(0, int(args.route_pool_size))
    args.route_pool_min_size = max(0, min(int(args.route_pool_min_size), max(0, args.route_pool_size)))
    args.training_keep_clear_interval = max(1.0, float(args.training_keep_clear_interval))
    args.training_lane_lock_interval = max(1.0, float(args.training_lane_lock_interval))
    args.training_unconnected_interval = max(1.0, float(args.training_unconnected_interval))
    args.training_lane_pref_interval = max(1.0, float(args.training_lane_pref_interval))
    args.training_lane_balance_interval = max(2.0, float(args.training_lane_balance_interval))
    args.training_signal_update_interval = max(1.0, float(args.training_signal_update_interval))
    args.torch_threads = max(1, int(args.torch_threads))
    torch.set_num_threads(args.torch_threads)

    if args.num_envs != 1:
        print(
            "WARNING: num-envs > 1 starts multiple heavy SUMO processes. "
            "On your Mac, --num-envs 1 is usually faster and more stable."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
