#!/usr/bin/env python3
"""
Generalized traffic-signal proxy environment.

This module keeps the same external interface as traffic_rl_model_santaclara_proxy.py:
    - TrafficSignalEnv
    - build_controller_for_tls
    - generate_route_variants
    - discover_background_route_variants
    - list_tls-style constants

But it changes the RL state/reward to be more map-invariant:
    1. phase-relative features instead of raw NB/SB/EB/WB queue slots only
    2. queue/vehicle counts normalized by local lane storage capacity
    3. downstream occupancy and blocked-exit features
    4. max-pressure-style reward terms
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import traffic_rl_model_santaclara_proxy as base
from traffic_rl_model_santaclara_proxy import *  # reuse SUMO setup, geometry, phase builders, etc.


# ======================================================================================
# Map/file configuration
# ======================================================================================
# Default stays Santa Clara. For new_map, run with:
#   export TRAFFIC_NET_FILE="new_map.net.xml"
#   export TRAFFIC_ROUTE_PREFIX="new_map"
TRAFFIC_NET_FILE = os.environ.get("TRAFFIC_NET_FILE", os.path.basename(base.NET_FILE))
ROUTE_PREFIX = os.environ.get(
    "TRAFFIC_ROUTE_PREFIX",
    Path(TRAFFIC_NET_FILE).stem.replace(".net", ""),
)

NET_FILE = os.path.join(BASE_DIR, TRAFFIC_NET_FILE)
BACKGROUND_ROUTE_FILE = os.path.join(BASE_DIR, f"background_{ROUTE_PREFIX}.rou.xml")
AMBULANCE_ROUTE_FILE = os.path.join(BASE_DIR, f"empty_ambulance_{ROUTE_PREFIX}.rou.xml")
SUMO_RUN_LOG = os.path.join(BASE_DIR, f"sumo_{ROUTE_PREFIX}_general_proxy_run.log")
SUMO_ERROR_LOG = os.path.join(BASE_DIR, f"sumo_{ROUTE_PREFIX}_general_proxy_error.log")
MODEL_FILE = os.path.join(BASE_DIR, f"models/traffic_signal_maskable_ppo_{ROUTE_PREFIX}_general_proxy")

# Keep base functions that we reuse pointed at the same files.
for _name in (
    "NET_FILE",
    "BACKGROUND_ROUTE_FILE",
    "AMBULANCE_ROUTE_FILE",
    "SUMO_RUN_LOG",
    "SUMO_ERROR_LOG",
    "MODEL_FILE",
):
    setattr(base, _name, globals()[_name])


def ensure_empty_ambulance_file() -> None:
    path = Path(AMBULANCE_ROUTE_FILE)
    if path.exists():
        return
    path.write_text(
        '<routes>\n'
        '  <vType id="ambulance" vClass="emergency" accel="3.0" decel="5.0" '
        'sigma="0.3" length="6.5" maxSpeed="55" color="1,0,0"/>\n'
        '</routes>\n'
    )


def discover_background_route_variants():
    variants = sorted(Path(BASE_DIR).glob(f"background_{ROUTE_PREFIX}_train_*.rou.xml"))
    if variants:
        return [str(path) for path in variants]
    return [BACKGROUND_ROUTE_FILE]


def generate_route_variants(periods, route_seeds):
    ensure_empty_ambulance_file()

    if not os.path.exists(RANDOM_TRIPS_SCRIPT):
        raise FileNotFoundError(f"Could not find randomTrips.py at {RANDOM_TRIPS_SCRIPT}")
    if not os.path.exists(NET_FILE):
        raise FileNotFoundError(f"Missing network file: {NET_FILE}")

    generated_files = []
    for period in periods:
        period_name = safe_period_name(period)
        for seed in route_seeds:
            output_file = os.path.join(
                BASE_DIR,
                f"background_{ROUTE_PREFIX}_train_{period_name}_seed{seed}.rou.xml",
            )
            prefix = f"{ROUTE_PREFIX}_{period_name}_s{seed}_car_"
            cmd = [
                sys.executable,
                RANDOM_TRIPS_SCRIPT,
                "-n",
                NET_FILE,
                "-r",
                output_file,
                "--no-validate",
                "-b",
                "0",
                "-e",
                str(SIM_END),
                "-p",
                str(period),
                "--seed",
                str(seed),
                "--prefix",
                prefix,
                "-t",
                'type="car" departLane="best" departPos="free" departSpeed="max"',
            ]
            print("\nGenerating route file:")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            patch_car_vtype(output_file)
            generated_files.append(output_file)
            print(f"Generated and patched: {output_file}")

    return generated_files


# ======================================================================================
# Generalized observation constants
# ======================================================================================
# For each of the four canonical phase slots, we expose:
#   0 available
#   1 current
#   2 left_phase
#   3 straight_phase
#   4 served_queue_ratio
#   5 served_wait_per_vehicle_ratio
#   6 served_vehicle_ratio
#   7 pressure_ratio
#   8 downstream_space_ratio
#   9 blocked_exit_ratio
#   10 starvation_ratio
PHASE_FEATURES = 11
OBSERVATION_SIZE = 4 * PHASE_FEATURES + 2

APPROACH_STORAGE_METERS = 120.0
VEHICLE_STORAGE_METERS = 7.5
WAIT_NORMALIZER_SECONDS = 120.0

DOWNSTREAM_BLOCKED_SPACE_THRESHOLD = 0.12
DOWNSTREAM_BLOCKED_RATIO_THRESHOLD = 0.65

PRESSURE_REWARD_WEIGHT = 1.20
PRESSURE_DELTA_WEIGHT = 2.00
DOWNSTREAM_BLOCK_PENALTY = 2.50
STARVATION_REWARD_PENALTY = 1.20


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_div(a: float, b: float) -> float:
    return float(a) / max(1e-6, float(b))


def lane_storage_capacity(lane_id: str, max_storage_m: float = APPROACH_STORAGE_METERS) -> float:
    try:
        length = float(traci.lane.getLength(lane_id))
    except traci.TraCIException:
        length = max_storage_m
    usable = max(7.5, min(length, max_storage_m))
    return max(1.0, usable / VEHICLE_STORAGE_METERS)


def lane_vehicle_count_queue_wait(lanes: set[str], snapshot=None):
    if snapshot is None:
        snapshot = lane_vehicle_snapshot(lanes)

    lane_ids, speed_cache, wait_cache = snapshot

    veh_ids = set()
    for lane_id in lanes:
        veh_ids.update(lane_ids.get(lane_id, ()))

    count = 0.0
    queue = 0.0
    wait = 0.0

    for veh_id in veh_ids:
        count += 1.0
        speed = speed_cache.get(veh_id)
        if speed is None:
            try:
                speed = float(traci.vehicle.getSpeed(veh_id))
                speed_cache[veh_id] = speed
            except traci.TraCIException:
                speed = 0.0

        if speed < 0.1:
            queue += 1.0

        value = wait_cache.get(veh_id)
        if value is None:
            try:
                value = float(traci.vehicle.getWaitingTime(veh_id))
                wait_cache[veh_id] = value
            except traci.TraCIException:
                value = 0.0
        wait += value

    return count, queue, wait


def build_lane_caches_general(movement_map):
    movement_in_lanes_cache = {}
    movement_out_lanes_cache = {}

    all_in_lanes = set()
    all_out_lanes = set()

    for label in MOVEMENT_LABELS:
        in_lanes = set()
        out_lanes = set()

        for lane_sets in movement_map[label].values():
            in_lanes.update(lane_sets["in"])
            out_lanes.update(lane_sets["out"])

        movement_in_lanes_cache[label] = in_lanes
        movement_out_lanes_cache[label] = out_lanes
        all_in_lanes.update(in_lanes)
        all_out_lanes.update(out_lanes)

    return movement_in_lanes_cache, movement_out_lanes_cache, all_in_lanes, all_out_lanes


def build_controller_for_tls(tls_id, activate=True):
    state_length, movement_map = classify_tls_movements(tls_id)
    phases = build_safe_phase_plan(movement_map)

    if len(phases) < 2:
        return None

    slot_to_pos = {phase["slot"]: i for i, phase in enumerate(phases)}
    movement_in_lanes_cache, movement_out_lanes_cache, all_in_lanes, all_out_lanes = (
        build_lane_caches_general(movement_map)
    )

    controller = {
        "tls_id": tls_id,
        "state_length": state_length,
        "movement_map": movement_map,
        "movement_in_lanes_cache": movement_in_lanes_cache,
        "movement_out_lanes_cache": movement_out_lanes_cache,
        "all_in_lanes": all_in_lanes,
        "all_out_lanes": all_out_lanes,
        "phases": phases,
        "slot_to_pos": slot_to_pos,
        "phase_pos": 0,
        "mode": "green",
        "remaining": 0.0,
        "phase_elapsed": 0.0,
        "last_active_indices": set(),
        "phase_last_served_time": [0.0 for _ in phases],
    }

    validate_four_way_target(controller, tls_id)

    if activate:
        start_green(tls_id, controller, phase_pos=0)

    return controller


def start_green(tls_id, controller, phase_pos=None):
    base.start_green(tls_id, controller, phase_pos=phase_pos)
    try:
        pos = int(controller["phase_pos"])
        controller["phase_last_served_time"][pos] = float(traci.simulation.getTime())
    except Exception:
        pass


def switch_to_phase(tls_id, controller, new_phase_pos):
    if new_phase_pos == controller["phase_pos"]:
        return False

    base.start_yellow(tls_id, controller)
    base.run_steps(T_YELLOW)

    base.start_all_red(tls_id, controller)
    base.run_steps(T_ALL_RED)

    start_green(tls_id, controller, phase_pos=new_phase_pos)
    return True


def movement_stats(controller, movement_label: str, snapshot=None):
    in_lanes = controller.get("movement_in_lanes_cache", {}).get(movement_label, set())
    out_lanes = controller.get("movement_out_lanes_cache", {}).get(movement_label, set())

    if snapshot is None:
        snapshot = lane_vehicle_snapshot(set(in_lanes) | set(out_lanes))

    in_count, in_queue, in_wait = lane_vehicle_count_queue_wait(in_lanes, snapshot=snapshot)
    out_count, out_queue, _out_wait = lane_vehicle_count_queue_wait(out_lanes, snapshot=snapshot)

    in_capacity = sum(lane_storage_capacity(lane_id) for lane_id in in_lanes)
    out_capacity = sum(lane_storage_capacity(lane_id) for lane_id in out_lanes)

    queue_ratio = clamp01(safe_div(in_queue, in_capacity))
    vehicle_ratio = clamp01(safe_div(in_count, in_capacity))
    wait_per_vehicle_ratio = clamp01(safe_div(in_wait, max(1.0, in_count)) / WAIT_NORMALIZER_SECONDS)

    downstream_occupancy = clamp01(safe_div(out_count, out_capacity))
    downstream_space = clamp01(1.0 - downstream_occupancy)

    blocked = 0.0
    for lane_id in out_lanes:
        if not outgoing_lane_has_space(lane_id):
            blocked += 1.0
    blocked_ratio = clamp01(safe_div(blocked, max(1.0, len(out_lanes))))

    # Max-pressure style signal:
    # positive means upstream is backed up and downstream still has room.
    pressure = max(-1.0, min(1.0, queue_ratio - downstream_occupancy))

    return {
        "count": in_count,
        "queue": in_queue,
        "wait": in_wait,
        "in_capacity": max(1.0, in_capacity),
        "out_capacity": max(1.0, out_capacity),
        "queue_ratio": queue_ratio,
        "vehicle_ratio": vehicle_ratio,
        "wait_per_vehicle_ratio": wait_per_vehicle_ratio,
        "downstream_occupancy": downstream_occupancy,
        "downstream_space": downstream_space,
        "blocked_ratio": blocked_ratio,
        "pressure": pressure,
    }


def phase_core_labels(phase):
    labels = []
    for label, signal_char in phase.get("rules", {}).items():
        if label.endswith("-R"):
            continue
        if signal_char == "G":
            labels.append(label)
    return labels


def aggregate_phase_stats(controller, phase, snapshot=None):
    labels = phase_core_labels(phase)

    if not labels:
        return {
            "queue_ratio": 0.0,
            "wait_per_vehicle_ratio": 0.0,
            "vehicle_ratio": 0.0,
            "pressure": 0.0,
            "downstream_space": 1.0,
            "blocked_ratio": 0.0,
            "starvation_ratio": 0.0,
        }

    total_queue = 0.0
    total_count = 0.0
    total_wait = 0.0
    total_in_capacity = 0.0
    pressure_sum = 0.0
    downstream_space_sum = 0.0
    blocked_sum = 0.0

    for label in labels:
        stats = movement_stats(controller, label, snapshot=snapshot)
        total_queue += stats["queue"]
        total_count += stats["count"]
        total_wait += stats["wait"]
        total_in_capacity += stats["in_capacity"]
        pressure_sum += stats["pressure"]
        downstream_space_sum += stats["downstream_space"]
        blocked_sum += stats["blocked_ratio"]

    n = max(1.0, float(len(labels)))

    queue_ratio = clamp01(safe_div(total_queue, total_in_capacity))
    vehicle_ratio = clamp01(safe_div(total_count, total_in_capacity))
    wait_per_vehicle_ratio = clamp01(
        safe_div(total_wait, max(1.0, total_count)) / WAIT_NORMALIZER_SECONDS
    )
    pressure = max(-1.0, min(1.0, pressure_sum / n))
    downstream_space = clamp01(downstream_space_sum / n)
    blocked_ratio = clamp01(blocked_sum / n)

    starvation_ratio = wait_per_vehicle_ratio
    return {
        "queue_ratio": queue_ratio,
        "wait_per_vehicle_ratio": wait_per_vehicle_ratio,
        "vehicle_ratio": vehicle_ratio,
        "pressure": pressure,
        "downstream_space": downstream_space,
        "blocked_ratio": blocked_ratio,
        "starvation_ratio": starvation_ratio,
    }


def phase_is_downstream_blocked(controller, phase, snapshot=None):
    stats = aggregate_phase_stats(controller, phase, snapshot=snapshot)
    return (
        stats["downstream_space"] < DOWNSTREAM_BLOCKED_SPACE_THRESHOLD
        or stats["blocked_ratio"] > DOWNSTREAM_BLOCKED_RATIO_THRESHOLD
    )


def total_local_pressure(controller, snapshot=None) -> float:
    if snapshot is None:
        lanes = set(controller.get("all_in_lanes", set())) | set(controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

    pressure = 0.0
    for label in MOVEMENT_LABELS:
        if label.endswith("-R"):
            continue
        stats = movement_stats(controller, label, snapshot=snapshot)
        pressure += max(0.0, stats["pressure"])

    return float(pressure)


def total_downstream_blocking(controller, snapshot=None) -> float:
    if snapshot is None:
        lanes = set(controller.get("all_in_lanes", set())) | set(controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

    blocked = 0.0
    count = 0.0

    for phase in controller.get("phases", []):
        stats = aggregate_phase_stats(controller, phase, snapshot=snapshot)
        blocked += stats["blocked_ratio"]
        count += 1.0

    return clamp01(safe_div(blocked, max(1.0, count)))


def get_observation(controller, snapshot=None):
    if snapshot is None:
        lanes = set(controller.get("all_in_lanes", set())) | set(controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

    obs = []

    current_pos = int(controller.get("phase_pos", 0))
    current_slot = int(controller["phases"][current_pos]["slot"])
    slot_to_pos = controller.get("slot_to_pos", {})

    for slot in range(4):
        if slot not in slot_to_pos:
            obs.extend([0.0] * PHASE_FEATURES)
            continue

        phase_pos = slot_to_pos[slot]
        phase = controller["phases"][phase_pos]
        stats = aggregate_phase_stats(controller, phase, snapshot=snapshot)

        available = 1.0
        is_current = 1.0 if slot == current_slot else 0.0
        is_left_phase = 1.0 if slot in (0, 2) else 0.0
        is_straight_phase = 1.0 if slot in (1, 3) else 0.0

        # Only count starvation when this phase is not currently green.
        starvation = 0.0 if is_current else stats["starvation_ratio"]

        obs.extend(
            [
                available,
                is_current,
                is_left_phase,
                is_straight_phase,
                stats["queue_ratio"],
                stats["wait_per_vehicle_ratio"],
                stats["vehicle_ratio"],
                stats["pressure"],
                stats["downstream_space"],
                stats["blocked_ratio"],
                starvation,
            ]
        )

    obs.append(float(controller.get("phase_elapsed", 0.0)) / MAX_GREEN_HOLD)

    try:
        sim_time = float(traci.simulation.getTime())
    except traci.TraCIException:
        sim_time = 0.0
    obs.append(sim_time / max(1.0, float(SIM_END)))

    return np.array(obs, dtype=np.float32)


def inactive_core_queue(controller, snapshot=None):
    if snapshot is None:
        lanes = set(controller.get("all_in_lanes", set())) | set(controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

    try:
        active_rules = controller["phases"][controller["phase_pos"]]["rules"]
    except Exception:
        active_rules = {}

    waiting = 0.0
    for label in MOVEMENT_LABELS:
        if label.endswith("-R"):
            continue
        if active_rules.get(label) == "G":
            continue
        stats = movement_stats(controller, label, snapshot=snapshot)
        waiting += stats["queue"]

    return waiting


def compute_reward(
    controller,
    switched,
    arrived,
    prev_wait,
    prev_queue,
    prev_pressure=0.0,
    local_cleared=0,
    snapshot=None,
):
    if snapshot is None:
        lanes = set(controller.get("all_in_lanes", set())) | set(controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

    total_wait, total_queue = total_controlled_wait_and_queue(controller, snapshot=snapshot)
    current_pressure = total_local_pressure(controller, snapshot=snapshot)
    blocking = total_downstream_blocking(controller, snapshot=snapshot)

    wait_delta = float(prev_wait) - float(total_wait)
    queue_delta = float(prev_queue) - float(total_queue)
    pressure_delta = float(prev_pressure) - float(current_pressure)

    inactive_queue = inactive_core_queue(controller, snapshot=snapshot)
    elapsed = float(controller.get("phase_elapsed", 0.0))
    excess_green = max(0.0, elapsed - STARVATION_START)

    reward = 0.0

    # Local causal throughput.
    reward += 0.80 * float(local_cleared)

    # Local improvement.
    reward += 0.012 * clipped(wait_delta, -250.0, 250.0)
    reward += 0.110 * clipped(queue_delta, -35.0, 35.0)

    # Max-pressure shaping.
    reward += PRESSURE_DELTA_WEIGHT * clipped(pressure_delta, -2.5, 2.5)
    reward -= PRESSURE_REWARD_WEIGHT * current_pressure

    # Do not dump cars into blocked downstream lanes.
    reward -= DOWNSTREAM_BLOCK_PENALTY * blocking

    # Weak network-level hint only.
    reward += 0.03 * float(arrived)

    # Level penalties.
    reward -= total_wait / 1100.0
    reward -= total_queue / 55.0

    # Starvation / excessive hold.
    reward -= STARVATION_REWARD_PENALTY * (excess_green * inactive_queue) / 420.0

    if switched:
        reward -= 0.08

    if elapsed > MAX_GREEN_HOLD:
        reward -= 0.85

    return float(reward), float(total_wait), float(total_queue), float(current_pressure)


class TrafficSignalEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tls_id=None,
        gui=False,
        randomize_traffic=False,
        route_variants=None,
        max_vehicle_variants=None,
    ):
        if gym is None or spaces is None:
            raise ImportError(
                "Missing dependencies. Run:\n"
                "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
            )

        self.tls_id = tls_id
        self.gui = gui
        self.randomize_traffic = randomize_traffic
        self.route_variants = route_variants or discover_background_route_variants()
        self.max_vehicle_variants = max_vehicle_variants or MAX_VEHICLE_VARIANTS

        self.current_background_route_file = BACKGROUND_ROUTE_FILE
        self.current_max_num_vehicles = MAX_NUM_VEHICLES
        self.current_sumo_seed = 42

        self.started = False
        self.controller = None

        self.prev_wait = 0.0
        self.prev_queue = 0.0
        self.prev_pressure = 0.0
        self.episode_arrived = 0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def choose_episode_scenario(self):
        if self.randomize_traffic:
            self.current_background_route_file = random.choice(self.route_variants)
            self.current_max_num_vehicles = random.choice(self.max_vehicle_variants)
            self.current_sumo_seed = random.randint(SUMO_SEED_MIN, SUMO_SEED_MAX)
        else:
            self.current_background_route_file = BACKGROUND_ROUTE_FILE
            self.current_max_num_vehicles = MAX_NUM_VEHICLES
            self.current_sumo_seed = 42

    def _sumo_cmd(self):
        ensure_empty_ambulance_file()

        binary = SUMO_GUI_BINARY if self.gui else SUMO_HEADLESS_BINARY
        route_file = f"{self.current_background_route_file},{AMBULANCE_ROUTE_FILE}"

        cmd = [
            binary,
            "-n",
            NET_FILE,
            "-r",
            route_file,
            "--start",
            "--step-length",
            str(STEP_LENGTH),
            "--end",
            str(TRAIN_EPISODE_SECONDS),
            "--max-num-vehicles",
            str(self.current_max_num_vehicles),
            "--max-depart-delay",
            str(MAX_DEPART_DELAY),
            "--time-to-teleport",
            str(TIME_TO_TELEPORT),
            "--seed",
            str(self.current_sumo_seed),
            *QUIET_SUMO_ARGS,
        ]

        if TRAIN_WITH_SUMO_LOGS:
            cmd.extend(["--log", SUMO_RUN_LOG, "--error-log", SUMO_ERROR_LOG])

        return cmd

    def reset(self, seed=None, options=None):
        if gym is not None:
            super().reset(seed=seed)

        if self.started:
            try:
                traci.close()
            except Exception:
                pass

        if self.gui:
            ensure_xquartz()

        self.choose_episode_scenario()

        if PRINT_TRAINING_SCENARIOS:
            print(
                "Training scenario: "
                f"route={os.path.basename(self.current_background_route_file)}, "
                f"max_vehicles={self.current_max_num_vehicles}, "
                f"sumo_seed={self.current_sumo_seed}"
            )

        traci.start(self._sumo_cmd())
        self.started = True

        chosen_tls = TARGET_TLS_ID if self.tls_id is None else self.tls_id
        self.controller = build_controller_for_tls(chosen_tls, activate=True)

        if self.controller is None:
            raise RuntimeError(f"Traffic light {chosen_tls} is not usable.")

        lanes = set(self.controller.get("all_in_lanes", set())) | set(self.controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

        self.prev_wait, self.prev_queue = total_controlled_wait_and_queue(
            self.controller,
            snapshot=snapshot,
        )
        self.prev_pressure = total_local_pressure(self.controller, snapshot=snapshot)

        self.episode_arrived = 0
        reset_run_step_arrival_counter()

        return get_observation(self.controller, snapshot=snapshot), {}

    def action_masks(self):
        mask = np.zeros(5, dtype=bool)

        if self.controller is None:
            mask[0] = True
            return mask

        if self.controller.get("mode") != "green":
            mask[0] = True
            return mask

        elapsed = float(self.controller.get("phase_elapsed", 0.0))

        if elapsed < MIN_GREEN_BEFORE_SWITCH:
            mask[0] = True
            return mask

        if elapsed < MAX_GREEN_HOLD:
            mask[0] = True

        current_pos = self.controller["phase_pos"]

        lanes = set(self.controller.get("all_in_lanes", set())) | set(self.controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

        for phase in self.controller["phases"]:
            phase_pos = self.controller["slot_to_pos"].get(phase["slot"])
            if phase_pos is None or phase_pos == current_pos:
                continue

            action_index = phase["slot"] + 1
            if phase_is_downstream_blocked(self.controller, phase, snapshot=snapshot):
                continue

            mask[action_index] = True

        # If every switch was blocked, fall back to the old legal mask so the env never deadlocks.
        if not mask.any() or (not mask[1:].any() and elapsed >= MAX_GREEN_HOLD):
            if elapsed < MAX_GREEN_HOLD:
                mask[0] = True
            for phase in self.controller["phases"]:
                phase_pos = self.controller["slot_to_pos"].get(phase["slot"])
                if phase_pos is None or phase_pos == current_pos:
                    continue
                mask[phase["slot"] + 1] = True

        if not mask.any():
            mask[0] = True

        return mask

    def step(self, action):
        action = int(action)
        mask = self.action_masks()

        if action < 0 or action >= len(mask) or not mask[action]:
            action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

        switched = False

        before_controlled_ids = controlled_vehicle_ids(self.controller)
        reset_run_step_arrival_counter()

        if action > 0:
            desired_slot = action - 1

            if desired_slot in self.controller["slot_to_pos"]:
                desired_phase_pos = self.controller["slot_to_pos"][desired_slot]
                can_switch = self.controller["phase_elapsed"] >= MIN_GREEN_BEFORE_SWITCH

                if can_switch and desired_phase_pos != self.controller["phase_pos"]:
                    switched = switch_to_phase(
                        self.controller["tls_id"],
                        self.controller,
                        desired_phase_pos,
                    )

        if self.controller["phase_elapsed"] >= MAX_GREEN_HOLD:
            next_pos = (self.controller["phase_pos"] + 1) % len(self.controller["phases"])
            switched = switch_to_phase(
                self.controller["tls_id"],
                self.controller,
                next_pos,
            )

        base.run_steps(
            DECISION_INTERVAL,
            self.controller["tls_id"],
            self.controller,
        )

        arrived = consume_run_step_arrivals()
        self.episode_arrived += int(arrived)

        after_controlled_ids = controlled_vehicle_ids(self.controller)
        local_cleared = len(before_controlled_ids - after_controlled_ids)

        lanes = set(self.controller.get("all_in_lanes", set())) | set(self.controller.get("all_out_lanes", set()))
        snapshot = lane_vehicle_snapshot(lanes)

        obs = get_observation(self.controller, snapshot=snapshot)

        reward, total_wait, total_queue, current_pressure = compute_reward(
            self.controller,
            switched=switched,
            arrived=arrived,
            prev_wait=self.prev_wait,
            prev_queue=self.prev_queue,
            prev_pressure=self.prev_pressure,
            local_cleared=local_cleared,
            snapshot=snapshot,
        )

        self.prev_wait = total_wait
        self.prev_queue = total_queue
        self.prev_pressure = current_pressure

        sim_time = traci.simulation.getTime()
        terminated = sim_time >= TRAIN_EPISODE_SECONDS
        truncated = traci.simulation.getMinExpectedNumber() <= 0

        current_phase = self.controller["phases"][self.controller["phase_pos"]]

        info = {
            "tls_id": self.controller["tls_id"],
            "sim_time": sim_time,
            "phase_pos": self.controller["phase_pos"],
            "phase_slot": current_phase["slot"],
            "phase_name": current_phase["name"],
            "switched": switched,
            "valid_action_mask": self.action_masks(),
            "background_route": os.path.basename(self.current_background_route_file),
            "max_num_vehicles": self.current_max_num_vehicles,
            "sumo_seed": self.current_sumo_seed,
            "arrived": arrived,
            "episode_arrived": self.episode_arrived,
            "local_cleared": local_cleared,
            "controlled_wait": total_wait,
            "controlled_queue": total_queue,
            "local_pressure": current_pressure,
            "downstream_blocking": total_downstream_blocking(self.controller, snapshot=snapshot),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.started:
            try:
                traci.close()
            except Exception:
                pass
        self.started = False


def main():
    # Reuse the old CLI modes where possible.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["generate-routes", "list-tls"],
        default="list-tls",
    )
    parser.add_argument(
        "--traffic-periods",
        default=",".join(str(x) for x in DEFAULT_TRAINING_TRAFFIC_PERIODS),
    )
    parser.add_argument(
        "--route-seeds",
        default=",".join(str(x) for x in DEFAULT_TRAINING_ROUTE_SEEDS),
    )
    args = parser.parse_args()

    if args.mode == "generate-routes":
        periods = parse_float_list(args.traffic_periods)
        route_seeds = parse_int_list(args.route_seeds)
        generate_route_variants(periods, route_seeds)
    elif args.mode == "list-tls":
        base.NET_FILE = NET_FILE
        base.BACKGROUND_ROUTE_FILE = BACKGROUND_ROUTE_FILE
        base.AMBULANCE_ROUTE_FILE = AMBULANCE_ROUTE_FILE
        ensure_empty_ambulance_file()
        list_tls()


if __name__ == "__main__":
    main()
