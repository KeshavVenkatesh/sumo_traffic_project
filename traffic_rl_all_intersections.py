import argparse
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# SUMO path setup BEFORE importing traci
# ============================================================

DEFAULT_SUMO_HOME = (
    "/Library/Frameworks/EclipseSUMO.framework/"
    "Versions/1.26.0/EclipseSUMO/share/sumo"
)

os.environ.setdefault("SUMO_HOME", DEFAULT_SUMO_HOME)

SUMO_TOOLS = os.path.join(os.environ["SUMO_HOME"], "tools")
if os.path.isdir(SUMO_TOOLS) and SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

for proj_candidate in (
    "/opt/homebrew/share/proj",
    "/usr/local/share/proj",
    os.path.join(os.environ["SUMO_HOME"], "proj"),
):
    if os.path.exists(os.path.join(proj_candidate, "proj.db")):
        os.environ.setdefault("PROJ_DATA", proj_candidate)
        os.environ.setdefault("PROJ_LIB", proj_candidate)
        break

import numpy as np
import traci

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


# ============================================================
# Files / SUMO setup
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMO_BIN_DIR = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin"
SUMO_GUI_BINARY = os.path.join(SUMO_BIN_DIR, "sumo-gui")
SUMO_HEADLESS_BINARY = os.path.join(SUMO_BIN_DIR, "sumo")

if not os.path.exists(SUMO_GUI_BINARY):
    SUMO_GUI_BINARY = "sumo-gui"

if not os.path.exists(SUMO_HEADLESS_BINARY):
    SUMO_HEADLESS_BINARY = "sumo"

NET_FILE = os.path.join(BASE_DIR, "new_map.net.xml")
BACKGROUND_ROUTE_FILE = os.path.join(BASE_DIR, "background_new.rou.xml")

RANDOM_TRIPS_SCRIPT = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

SUMO_RUN_LOG = os.path.join(BASE_DIR, "sumo_all_intersections_run.log")
SUMO_ERROR_LOG = os.path.join(BASE_DIR, "sumo_all_intersections_error.log")

MODEL_FILE = os.path.join(BASE_DIR, "all_intersections_shared_policy")

QUIET_SUMO_ARGS = [
    "--no-warnings", "true",
    "--no-step-log", "true",
]

TRAIN_WITH_SUMO_LOGS = False


# ============================================================
# Simulation / training constants
# ============================================================

# Old/slower scheme, because it learned better in your tests.
STEP_LENGTH = 0.5
DECISION_INTERVAL = 5.0
TRAIN_EPISODE_SECONDS = 1800
SIM_END = 7200

T_YELLOW = 4.0
T_ALL_RED = 2.0

MIN_GREEN_BEFORE_SWITCH = 8.0
MAX_GREEN_HOLD = 60.0

REQUIRED_EXIT_GAP = 14.0

# Watched-intersection upstream-line metric settings.
# This counts cars farther upstream from the stop line, which better matches
# what you visually see in SUMO GUI.
UPSTREAM_APPROACH_DISTANCE = 500.0
QUEUE_SPEED_THRESHOLD = 0.1
SLOW_SPEED_THRESHOLD = 2.0

MAX_NUM_VEHICLES = 2000
MAX_DEPART_DELAY = 60
TIME_TO_TELEPORT = 300

PRINT_TRAINING_SCENARIOS = False

DEFAULT_TRAINING_TRAFFIC_PERIODS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
]

DEFAULT_TRAINING_ROUTE_SEEDS = [
    1,
    2,
    3,
]

MAX_VEHICLE_VARIANTS = [
    800,
    1000,
    1200,
    1500,
    1800,
    2000,
]

SUMO_SEED_MIN = 1
SUMO_SEED_MAX = 2_000_000_000

APPROACHES = ["NB", "SB", "EB", "WB"]
MOVEMENTS = ["L", "S", "R"]

MOVEMENT_LABELS = [
    "NB-L", "NB-S", "NB-R",
    "SB-L", "SB-S", "SB-R",
    "EB-L", "EB-S", "EB-R",
    "WB-L", "WB-S", "WB-R",
]

NON_RIGHT_MOVEMENTS = [
    "NB-L", "NB-S",
    "SB-L", "SB-S",
    "EB-L", "EB-S",
    "WB-L", "WB-S",
]

PHASE_SLOT_NAMES = [
    "PHASE 1: N/S Protected Lefts",
    "PHASE 2: N/S Straights",
    "PHASE 3: E/W Protected Lefts",
    "PHASE 4: E/W Straights",
]

ALL_RIGHT_TURNS = {
    "NB-R": "g",
    "SB-R": "g",
    "EB-R": "g",
    "WB-R": "g",
}

# 12 movements * 4 features each:
# queue, wait, exists, blocked
# + 4 phase one-hot
# + 4 valid phase-slot mask
# + phase elapsed
# + sim time
# + total queue
# + total wait
# + active vehicle count
OBS_DIM = 12 * 4 + 4 + 4 + 5


# ============================================================
# Route generation helpers
# ============================================================

def safe_period_name(period):
    return str(period).replace(".", "p")


def parse_float_list(raw):
    values = []

    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))

    return values


def parse_int_list(raw):
    values = []

    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))

    return values


def discover_background_route_variants():
    variants = sorted(Path(BASE_DIR).glob("background_train_all_*.rou.xml"))

    if variants:
        return [str(path) for path in variants]

    return [BACKGROUND_ROUTE_FILE]


def patch_car_vtype(route_file):
    path = Path(route_file)
    text = path.read_text()

    text = re.sub(r'\s*<vType id="car"[\s\S]*?/>', '', text)

    car_vtype = '''    <vType id="car"
           accel="2.6"
           decel="4.5"
           emergencyDecel="9.0"
           maxSpeed="13.9"
           sigma="0.5"
           lcCooperative="1.0"
           lcStrategic="1.0"
           lcSpeedGain="1.0"
           jmIgnoreKeepClearTime="-1"
           jmDriveAfterYellowTime="-1"
           jmDriveAfterRedTime="-1"/>
'''

    text = re.sub(
        r'(<routes[^>]*>)',
        r'\1\n' + car_vtype,
        text,
        count=1,
    )

    path.write_text(text)


def generate_route_variants(periods, route_seeds):
    if not os.path.exists(RANDOM_TRIPS_SCRIPT):
        raise FileNotFoundError(
            f"Could not find randomTrips.py at {RANDOM_TRIPS_SCRIPT}. "
            "Check SUMO_HOME."
        )

    if not os.path.exists(NET_FILE):
        raise FileNotFoundError(f"Missing network file: {NET_FILE}")

    generated_files = []

    for period in periods:
        period_name = safe_period_name(period)

        for seed in route_seeds:
            output_file = os.path.join(
                BASE_DIR,
                f"background_train_all_{period_name}_seed{seed}.rou.xml",
            )

            prefix = f"all_{period_name}_s{seed}_car_"

            cmd = [
                sys.executable,
                RANDOM_TRIPS_SCRIPT,
                "-n", NET_FILE,
                "-r", output_file,
                "--no-validate",
                "-b", "0",
                "-e", str(SIM_END),
                "-p", str(period),
                "--seed", str(seed),
                "--prefix", prefix,
                "-t", 'type="car" departLane="best" departPos="free" departSpeed="max"',
            ]

            print()
            print("Generating route file:")
            print(" ".join(cmd))

            subprocess.run(cmd, check=True)

            patch_car_vtype(output_file)
            generated_files.append(output_file)

            print(f"Generated and patched: {output_file}")

    print()
    print("Finished generating all-intersection training route files.")
    for file in generated_files:
        print(f"  {os.path.basename(file)}")

    return generated_files


# ============================================================
# macOS / XQuartz
# ============================================================

def ensure_xquartz():
    if sys.platform != "darwin":
        return

    os.environ.setdefault("DISPLAY", ":0")

    try:
        subprocess.run(
            ["open", "-a", "XQuartz"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(2)
    except Exception as e:
        print(f"Warning: could not open XQuartz automatically: {e}")


# ============================================================
# Geometry helpers
# ============================================================

def lane_direction_vector(lane_id, incoming):
    shape = traci.lane.getShape(lane_id)

    if len(shape) < 2:
        return None

    if incoming:
        p1 = shape[-2]
        p2 = shape[-1]
    else:
        p1 = shape[0]
        p2 = shape[1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)

    if length == 0:
        return None

    return dx / length, dy / length


def signed_turn_angle(in_vec, out_vec):
    ix, iy = in_vec
    ox, oy = out_vec

    cross = ix * oy - iy * ox
    dot = ix * ox + iy * oy

    return math.degrees(math.atan2(cross, dot))


def classify_approach(in_vec):
    dx, dy = in_vec

    if abs(dy) >= abs(dx):
        return "NB" if dy > 0 else "SB"

    return "EB" if dx > 0 else "WB"


def classify_movement(angle):
    if abs(angle) <= 35:
        return "S"

    if angle > 35:
        return "L"

    return "R"


# ============================================================
# Exit-space / anti-gridlock
# ============================================================

def outgoing_lane_has_space(out_lane_id):
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(out_lane_id)

        if not veh_ids:
            return True

        closest_pos = min(
            traci.vehicle.getLanePosition(veh_id)
            for veh_id in veh_ids
        )

        return closest_pos >= REQUIRED_EXIT_GAP

    except traci.TraCIException:
        return True


def lanes_have_space(out_lanes):
    return all(outgoing_lane_has_space(lane) for lane in out_lanes)


# ============================================================
# Upstream approach-line helpers
# ============================================================

_LANE_PREDECESSOR_CACHE = None


def is_internal_lane(lane_id):
    return not lane_id or lane_id.startswith(":")


def all_normal_lane_ids():
    lanes = []

    for edge_id in traci.edge.getIDList():
        if edge_id.startswith(":"):
            continue

        try:
            lane_count = traci.edge.getLaneNumber(edge_id)
        except traci.TraCIException:
            continue

        for lane_index in range(lane_count):
            lane_id = f"{edge_id}_{lane_index}"

            try:
                traci.lane.getLength(lane_id)
                lanes.append(lane_id)
            except traci.TraCIException:
                continue

    return lanes


def build_lane_predecessor_cache():
    """
    Builds a reverse lane graph:

        lane_id -> set(previous lanes that feed into lane_id)

    This lets the watched-intersection printer trace backward from the
    stop-line lanes and count the full visible line of cars approaching
    the intersection.
    """
    global _LANE_PREDECESSOR_CACHE

    if _LANE_PREDECESSOR_CACHE is not None:
        return _LANE_PREDECESSOR_CACHE

    predecessors = {}

    normal_lanes = all_normal_lane_ids()

    for lane_id in normal_lanes:
        predecessors.setdefault(lane_id, set())

    for lane_id in normal_lanes:
        try:
            links = traci.lane.getLinks(lane_id)
        except traci.TraCIException:
            continue

        for link in links:
            if not link:
                continue

            to_lane = link[0]

            if not to_lane or is_internal_lane(to_lane):
                continue

            predecessors.setdefault(to_lane, set()).add(lane_id)

    _LANE_PREDECESSOR_CACHE = predecessors
    return predecessors


def classify_lane_approach(lane_id):
    try:
        vec = lane_direction_vector(lane_id, incoming=True)

        if vec is None:
            return None

        return classify_approach(vec)

    except traci.TraCIException:
        return None


def trace_upstream_lanes_for_approach(start_lanes, approach, max_distance):
    """
    Starting from direct incoming lanes at the traffic light, walk upstream
    through connected predecessor lanes. Keep lanes that still match the
    same approach direction.

    This is meant for printing/debugging. It gives a better visual queue count
    than only counting the final controlled lanes directly at the stop line.
    """
    from collections import deque

    predecessors = build_lane_predecessor_cache()

    visited = set()
    work = deque()

    for lane_id in start_lanes:
        if is_internal_lane(lane_id):
            continue

        work.append((lane_id, 0.0))

    while work:
        lane_id, distance_so_far = work.popleft()

        if lane_id in visited:
            continue

        if distance_so_far > max_distance:
            continue

        lane_approach = classify_lane_approach(lane_id)

        if lane_approach is not None and lane_approach != approach:
            continue

        visited.add(lane_id)

        try:
            lane_length = traci.lane.getLength(lane_id)
        except traci.TraCIException:
            lane_length = 0.0

        next_distance = distance_so_far + lane_length

        for pred_lane in predecessors.get(lane_id, set()):
            if pred_lane not in visited:
                work.append((pred_lane, next_distance))

    return visited


def build_approach_upstream_cache(movement_in_lanes_cache):
    """
    Builds a cache like:

        {
            "NB": set(lanes upstream of all NB movements),
            "SB": set(...),
            "EB": set(...),
            "WB": set(...),
        }
    """
    approach_cache = {}

    for approach in APPROACHES:
        start_lanes = set()

        for movement in MOVEMENTS:
            label = f"{approach}-{movement}"
            start_lanes.update(movement_in_lanes_cache.get(label, set()))

        approach_cache[approach] = trace_upstream_lanes_for_approach(
            start_lanes=start_lanes,
            approach=approach,
            max_distance=UPSTREAM_APPROACH_DISTANCE,
        )

    return approach_cache


# ============================================================
# Signal state builders
# ============================================================

def all_red_state(length):
    return "r" * length


def build_yellow_state(length, active_indices):
    state = ["r"] * length

    for idx in active_indices:
        if 0 <= idx < length:
            state[idx] = "y"

    return "".join(state)


def build_state_from_movements(
    state_length,
    movement_map,
    phase_rules,
    check_space=True,
):
    state = ["r"] * state_length
    active_indices = set()

    for movement_label, signal_char in phase_rules.items():
        if signal_char == "r":
            continue

        signal_data = movement_map.get(movement_label, {})

        for signal_index, lane_sets in signal_data.items():
            out_lanes = lane_sets["out"]

            if check_space and not lanes_have_space(out_lanes):
                continue

            current = state[signal_index]

            if current == "G":
                continue

            if signal_char == "G":
                state[signal_index] = "G"
            elif signal_char == "g" and current == "r":
                state[signal_index] = "g"

            active_indices.add(signal_index)

    return "".join(state), active_indices


# ============================================================
# Movement classification
# ============================================================

def classify_tls_movements(tls_id):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    state_length = len(traci.trafficlight.getRedYellowGreenState(tls_id))

    movement_map = {
        label: {}
        for label in MOVEMENT_LABELS
    }

    for signal_index, signal_links in enumerate(controlled_links):
        if signal_index >= state_length:
            continue

        for link in signal_links:
            if len(link) < 2:
                continue

            incoming_lane = link[0]
            outgoing_lane = link[1]

            if not incoming_lane or not outgoing_lane:
                continue

            try:
                in_vec = lane_direction_vector(incoming_lane, incoming=True)
                out_vec = lane_direction_vector(outgoing_lane, incoming=False)

                if in_vec is None or out_vec is None:
                    continue

                approach = classify_approach(in_vec)
                angle = signed_turn_angle(in_vec, out_vec)
                movement = classify_movement(angle)
                label = f"{approach}-{movement}"

                if label not in movement_map:
                    continue

                if signal_index not in movement_map[label]:
                    movement_map[label][signal_index] = {
                        "in": set(),
                        "out": set(),
                    }

                movement_map[label][signal_index]["in"].add(incoming_lane)
                movement_map[label][signal_index]["out"].add(outgoing_lane)

            except traci.TraCIException:
                continue

    return state_length, movement_map


def build_safe_phase_plan(movement_map):
    phases = []

    phase_defs = [
        {
            "slot": 0,
            "name": PHASE_SLOT_NAMES[0],
            "core": ["NB-L", "SB-L"],
        },
        {
            "slot": 1,
            "name": PHASE_SLOT_NAMES[1],
            "core": ["NB-S", "SB-S"],
        },
        {
            "slot": 2,
            "name": PHASE_SLOT_NAMES[2],
            "core": ["EB-L", "WB-L"],
        },
        {
            "slot": 3,
            "name": PHASE_SLOT_NAMES[3],
            "core": ["EB-S", "WB-S"],
        },
    ]

    for phase_def in phase_defs:
        existing_core = [
            label
            for label in phase_def["core"]
            if movement_map.get(label)
        ]

        if not existing_core:
            continue

        rules = {}

        for label in phase_def["core"]:
            rules[label] = "G"

        rules.update(ALL_RIGHT_TURNS)

        phases.append({
            "slot": phase_def["slot"],
            "name": phase_def["name"],
            "rules": rules,
            "core_labels": existing_core,
        })

    return phases


def build_lane_caches(movement_map):
    movement_in_lanes_cache = {}
    movement_out_lanes_cache = {}

    for label in MOVEMENT_LABELS:
        in_lanes = set()
        out_lanes = set()

        for lane_sets in movement_map[label].values():
            in_lanes.update(lane_sets["in"])
            out_lanes.update(lane_sets["out"])

        movement_in_lanes_cache[label] = in_lanes
        movement_out_lanes_cache[label] = out_lanes

    all_in_lanes = set()

    for lanes in movement_in_lanes_cache.values():
        all_in_lanes.update(lanes)

    return movement_in_lanes_cache, movement_out_lanes_cache, all_in_lanes


# ============================================================
# Safety verification
# ============================================================

def labels_by_signal_index(tls_id):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    state_length = len(traci.trafficlight.getRedYellowGreenState(tls_id))

    result = {i: set() for i in range(state_length)}

    for signal_index, signal_links in enumerate(controlled_links):
        if signal_index >= state_length:
            continue

        for link in signal_links:
            if len(link) < 2:
                continue

            incoming_lane = link[0]
            outgoing_lane = link[1]

            if not incoming_lane or not outgoing_lane:
                continue

            try:
                in_vec = lane_direction_vector(incoming_lane, incoming=True)
                out_vec = lane_direction_vector(outgoing_lane, incoming=False)

                if in_vec is None or out_vec is None:
                    continue

                approach = classify_approach(in_vec)
                angle = signed_turn_angle(in_vec, out_vec)
                movement = classify_movement(angle)

                result[signal_index].add(f"{approach}-{movement}")

            except traci.TraCIException:
                continue

    return result


def present_right_turn_labels(labels_by_idx):
    labels = set()

    for label_set in labels_by_idx.values():
        for label in label_set:
            if label.endswith("-R"):
                labels.add(label)

    return labels


def verify_controller_safety(tls_id, controller):
    labels_by_idx = labels_by_signal_index(tls_id)
    right_turn_labels = present_right_turn_labels(labels_by_idx)

    messages = []

    for phase in controller["phases"]:
        allowed_labels = set(phase["core_labels"]) | right_turn_labels

        state, _ = build_state_from_movements(
            controller["state_length"],
            controller["movement_map"],
            phase["rules"],
            check_space=False,
        )

        active_labels = set()

        for idx, char in enumerate(state):
            if char not in ("G", "g"):
                continue

            active_labels.update(labels_by_idx.get(idx, set()))

        bad_labels = active_labels - allowed_labels
        missing_core = set(phase["core_labels"]) - active_labels

        if bad_labels:
            messages.append(
                f"{phase['name']}: forbidden movements green: {sorted(bad_labels)}"
            )

        if missing_core:
            messages.append(
                f"{phase['name']}: missing core movements: {sorted(missing_core)}"
            )

    return len(messages) == 0, messages


# ============================================================
# Controller state machine
# ============================================================

def build_controller_for_tls(tls_id, activate=True, require_safe=True):
    try:
        state_length, movement_map = classify_tls_movements(tls_id)
        phases = build_safe_phase_plan(movement_map)
    except traci.TraCIException:
        return None

    if len(phases) < 2:
        return None

    slot_to_pos = {
        phase["slot"]: i
        for i, phase in enumerate(phases)
    }

    movement_in_lanes_cache, movement_out_lanes_cache, all_in_lanes = build_lane_caches(
        movement_map
    )

    approach_upstream_lanes_cache = build_approach_upstream_cache(
        movement_in_lanes_cache
    )

    controller = {
        "tls_id": tls_id,
        "state_length": state_length,
        "movement_map": movement_map,
        "movement_in_lanes_cache": movement_in_lanes_cache,
        "movement_out_lanes_cache": movement_out_lanes_cache,
        "approach_upstream_lanes_cache": approach_upstream_lanes_cache,
        "all_in_lanes": all_in_lanes,
        "phases": phases,
        "slot_to_pos": slot_to_pos,
        "phase_pos": 0,
        "mode": "green",
        "remaining": 0.0,
        "phase_elapsed": 0.0,
        "last_active_indices": set(),
        "next_phase_pos": None,
        "disabled": False,
    }

    if require_safe:
        safe, _ = verify_controller_safety(tls_id, controller)
        if not safe:
            return None

    if activate:
        start_green(controller, phase_pos=0)

    return controller


def start_green(controller, phase_pos=None):
    if phase_pos is not None:
        controller["phase_pos"] = phase_pos

    phase = controller["phases"][controller["phase_pos"]]

    green_state, active_indices = build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
        check_space=True,
    )

    traci.trafficlight.setRedYellowGreenState(
        controller["tls_id"],
        green_state,
    )

    controller["mode"] = "green"
    controller["remaining"] = 0.0
    controller["phase_elapsed"] = 0.0
    controller["last_active_indices"] = active_indices
    controller["next_phase_pos"] = None


def update_green(controller):
    phase = controller["phases"][controller["phase_pos"]]

    green_state, active_indices = build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
        check_space=True,
    )

    traci.trafficlight.setRedYellowGreenState(
        controller["tls_id"],
        green_state,
    )

    controller["last_active_indices"] = active_indices


def start_yellow(controller):
    yellow_state = build_yellow_state(
        controller["state_length"],
        controller["last_active_indices"],
    )

    traci.trafficlight.setRedYellowGreenState(
        controller["tls_id"],
        yellow_state,
    )

    controller["mode"] = "yellow"
    controller["remaining"] = T_YELLOW


def start_all_red(controller):
    traci.trafficlight.setRedYellowGreenState(
        controller["tls_id"],
        all_red_state(controller["state_length"]),
    )

    controller["mode"] = "all_red"
    controller["remaining"] = T_ALL_RED


def request_switch(controller, new_phase_pos):
    if controller["mode"] != "green":
        return False

    if new_phase_pos == controller["phase_pos"]:
        return False

    controller["next_phase_pos"] = new_phase_pos
    start_yellow(controller)
    return True


def update_controller_after_simstep(controller):
    try:
        if controller["mode"] == "green":
            update_green(controller)
            controller["phase_elapsed"] += STEP_LENGTH

        elif controller["mode"] == "yellow":
            controller["remaining"] -= STEP_LENGTH

            if controller["remaining"] <= 0:
                start_all_red(controller)

        elif controller["mode"] == "all_red":
            controller["remaining"] -= STEP_LENGTH

            if controller["remaining"] <= 0:
                next_pos = controller["next_phase_pos"]

                if next_pos is None:
                    next_pos = (
                        controller["phase_pos"] + 1
                    ) % len(controller["phases"])

                start_green(controller, phase_pos=next_pos)

    except traci.TraCIException:
        controller["disabled"] = True


def run_control_steps(seconds, controllers):
    steps = int(round(seconds / STEP_LENGTH))
    arrived_interval = 0

    for _ in range(steps):
        if traci.simulation.getMinExpectedNumber() <= 0:
            return False, arrived_interval

        traci.simulationStep()
        arrived_interval += traci.simulation.getArrivedNumber()

        for controller in controllers:
            if controller.get("disabled"):
                continue

            update_controller_after_simstep(controller)

    return True, arrived_interval


def apply_model_action(controller, action):
    action = int(action)

    if controller.get("disabled"):
        return False

    if controller["mode"] != "green":
        return False

    if (
        controller["phase_elapsed"] >= MAX_GREEN_HOLD
        and len(controller["phases"]) > 1
    ):
        next_pos = (
            controller["phase_pos"] + 1
        ) % len(controller["phases"])

        return request_switch(controller, next_pos)

    if action > 0 and controller["phase_elapsed"] >= MIN_GREEN_BEFORE_SWITCH:
        desired_slot = action - 1

        if desired_slot in controller["slot_to_pos"]:
            desired_pos = controller["slot_to_pos"][desired_slot]
            return request_switch(controller, desired_pos)

    return False


def action_mask_for_controller(controller):
    mask = np.zeros(5, dtype=bool)
    mask[0] = True

    if controller is None or controller.get("disabled"):
        return mask

    if controller["mode"] != "green":
        return mask

    if controller["phase_elapsed"] < MIN_GREEN_BEFORE_SWITCH:
        return mask

    for phase in controller["phases"]:
        mask[phase["slot"] + 1] = True

    return mask


# ============================================================
# Observation / reward / metrics
# ============================================================

def movement_queue_and_wait(controller, movement_label):
    lanes = controller["movement_in_lanes_cache"][movement_label]

    veh_ids = set()

    for lane_id in lanes:
        try:
            veh_ids.update(traci.lane.getLastStepVehicleIDs(lane_id))
        except traci.TraCIException:
            pass

    queue = 0.0
    wait = 0.0

    for veh_id in veh_ids:
        try:
            speed = traci.vehicle.getSpeed(veh_id)

            if speed < QUEUE_SPEED_THRESHOLD:
                queue += 1.0

            wait += traci.vehicle.getWaitingTime(veh_id)

        except traci.TraCIException:
            pass

    return queue, wait


def movement_exists(controller, movement_label):
    return bool(controller["movement_in_lanes_cache"][movement_label])


def movement_is_blocked(controller, movement_label):
    out_lanes = controller["movement_out_lanes_cache"][movement_label]

    if not out_lanes:
        return False

    return not lanes_have_space(out_lanes)


def get_movement_stats(controller):
    stats = {}

    for label in MOVEMENT_LABELS:
        q, w = movement_queue_and_wait(controller, label)

        stats[label] = {
            "queue": q,
            "wait": w,
            "exists": 1.0 if movement_exists(controller, label) else 0.0,
            "blocked": 1.0 if movement_is_blocked(controller, label) else 0.0,
        }

    return stats


def current_core_movements(controller):
    phase = controller["phases"][controller["phase_pos"]]
    return set(phase["core_labels"])


def total_controlled_wait_and_queue(controller):
    stats = get_movement_stats(controller)

    total_wait = sum(item["wait"] for item in stats.values())
    total_queue = sum(item["queue"] for item in stats.values())

    return total_wait, total_queue


def get_observation(controller):
    stats = get_movement_stats(controller)

    obs = []

    for label in MOVEMENT_LABELS:
        obs.append(stats[label]["queue"] / 100.0)
        obs.append(stats[label]["wait"] / 1000.0)
        obs.append(stats[label]["exists"])
        obs.append(stats[label]["blocked"])

    phase_one_hot = [0.0, 0.0, 0.0, 0.0]
    current_phase = controller["phases"][controller["phase_pos"]]
    current_slot = current_phase["slot"]

    phase_one_hot[current_slot] = 1.0
    obs.extend(phase_one_hot)

    valid_phase_slots = [0.0, 0.0, 0.0, 0.0]

    for phase in controller["phases"]:
        valid_phase_slots[phase["slot"]] = 1.0

    obs.extend(valid_phase_slots)

    total_wait = sum(item["wait"] for item in stats.values())
    total_queue = sum(item["queue"] for item in stats.values())

    obs.append(controller["phase_elapsed"] / MAX_GREEN_HOLD)
    obs.append(traci.simulation.getTime() / TRAIN_EPISODE_SECONDS)
    obs.append(total_queue / 100.0)
    obs.append(total_wait / 1000.0)
    obs.append(traci.vehicle.getIDCount() / max(1.0, MAX_NUM_VEHICLES))

    return np.array(obs, dtype=np.float32)


def compute_reward(controller, switched, arrived_interval):
    stats = get_movement_stats(controller)

    total_wait = sum(item["wait"] for item in stats.values())
    total_queue = sum(item["queue"] for item in stats.values())

    existing_waits = [
        item["wait"]
        for item in stats.values()
        if item["exists"] > 0
    ]

    existing_queues = [
        item["queue"]
        for item in stats.values()
        if item["exists"] > 0
    ]

    max_movement_wait = max(existing_waits) if existing_waits else 0.0
    max_movement_queue = max(existing_queues) if existing_queues else 0.0

    active_core = current_core_movements(controller)

    active_core_queue = sum(
        stats[label]["queue"]
        for label in active_core
    )

    existing_non_right = [
        label
        for label in NON_RIGHT_MOVEMENTS
        if stats[label]["exists"] > 0
    ]

    red_core = [
        label
        for label in existing_non_right
        if label not in active_core
    ]

    red_queue = sum(
        stats[label]["queue"]
        for label in red_core
    )

    red_queue_pressure = max(0.0, red_queue - active_core_queue)

    blocked_exit_count = sum(
        1.0
        for label in MOVEMENT_LABELS
        if stats[label]["exists"] > 0 and stats[label]["blocked"] > 0
    )

    reward = 0.0

    reward -= total_wait / 500.0
    reward -= total_queue / 50.0

    reward -= max_movement_wait / 1000.0
    reward -= max_movement_queue / 25.0

    reward -= red_queue_pressure / 100.0
    reward -= blocked_exit_count * 0.5

    reward += arrived_interval * 0.2

    if switched:
        reward -= 0.2

    if controller["phase_elapsed"] > MAX_GREEN_HOLD:
        reward -= 1.0

    return float(reward)


def get_approach_line_metrics(controller):
    """
    Counts the specific upstream line of cars for each approach direction.

    For each of NB/SB/EB/WB, this counts vehicles on lanes leading into the
    watched intersection up to UPSTREAM_APPROACH_DISTANCE meters upstream.

    total_vehicles:
        all vehicles on that upstream approach line

    stopped_queue:
        vehicles with speed < QUEUE_SPEED_THRESHOLD

    slow_or_stopped:
        vehicles with speed < SLOW_SPEED_THRESHOLD

    total_wait:
        accumulated waiting time of vehicles on that approach line
    """
    result = {}

    for approach in APPROACHES:
        lanes = controller.get("approach_upstream_lanes_cache", {}).get(
            approach,
            set(),
        )

        veh_ids = set()

        for lane_id in lanes:
            try:
                veh_ids.update(traci.lane.getLastStepVehicleIDs(lane_id))
            except traci.TraCIException:
                pass

        stopped_queue = 0.0
        slow_or_stopped = 0.0
        total_wait = 0.0

        for veh_id in veh_ids:
            try:
                speed = traci.vehicle.getSpeed(veh_id)

                if speed < QUEUE_SPEED_THRESHOLD:
                    stopped_queue += 1.0

                if speed < SLOW_SPEED_THRESHOLD:
                    slow_or_stopped += 1.0

                total_wait += traci.vehicle.getWaitingTime(veh_id)

            except traci.TraCIException:
                pass

        result[approach] = {
            "lanes": lanes,
            "total_vehicles": float(len(veh_ids)),
            "stopped_queue": stopped_queue,
            "slow_or_stopped": slow_or_stopped,
            "total_wait": total_wait,
        }

    return result


def print_intersection_metrics(controller, last_action=None):
    stats = get_movement_stats(controller)
    phase = controller["phases"][controller["phase_pos"]]
    active_core = current_core_movements(controller)

    total_queue = sum(item["queue"] for item in stats.values())
    total_wait = sum(item["wait"] for item in stats.values())

    existing_items = [
        (label, item)
        for label, item in stats.items()
        if item["exists"] > 0
    ]

    max_queue_label = None
    max_queue_value = 0.0
    max_wait_label = None
    max_wait_value = 0.0

    for label, item in existing_items:
        if item["queue"] > max_queue_value:
            max_queue_label = label
            max_queue_value = item["queue"]

        if item["wait"] > max_wait_value:
            max_wait_label = label
            max_wait_value = item["wait"]

    active_core_queue = sum(
        stats[label]["queue"]
        for label in active_core
        if label in stats
    )

    red_core = [
        label
        for label in NON_RIGHT_MOVEMENTS
        if stats[label]["exists"] > 0 and label not in active_core
    ]

    red_queue = sum(stats[label]["queue"] for label in red_core)
    red_queue_pressure = max(0.0, red_queue - active_core_queue)

    controlled_veh_ids = set()

    for lane_id in controller["all_in_lanes"]:
        try:
            controlled_veh_ids.update(traci.lane.getLastStepVehicleIDs(lane_id))
        except traci.TraCIException:
            pass

    print()
    print("=" * 100)
    print(f"WATCHED INTERSECTION: {controller['tls_id']}")
    print(f"phase: {phase['name']}")
    print(f"mode: {controller['mode']}")
    print(f"phase_elapsed: {controller['phase_elapsed']:.1f}s")
    print(f"last_model_action: {last_action}")
    print(f"vehicles_on_controlled_incoming_lanes: {len(controlled_veh_ids)}")
    print(f"total_queue_near_stop_line: {total_queue:.0f}")
    print(f"total_wait_near_stop_line: {total_wait:.1f}s")
    print(f"max_movement_queue_near_stop_line: {max_queue_label} = {max_queue_value:.0f}")
    print(f"max_movement_wait_near_stop_line: {max_wait_label} = {max_wait_value:.1f}s")
    print(f"active_core_queue: {active_core_queue:.0f}")
    print(f"red_queue: {red_queue:.0f}")
    print(f"red_queue_pressure: {red_queue_pressure:.0f}")

    approach_line_stats = get_approach_line_metrics(controller)

    print()
    print(
        f"upstream approach-line metrics "
        f"(within about {UPSTREAM_APPROACH_DISTANCE:.0f} m of this intersection):"
    )
    print("approach  lanes  vehicles  stopped_queue  slow_or_stopped  total_wait_s")
    print("-" * 78)

    for approach in APPROACHES:
        item = approach_line_stats[approach]

        print(
            f"{approach:8}  "
            f"{len(item['lanes']):>5}  "
            f"{item['total_vehicles']:>8.0f}  "
            f"{item['stopped_queue']:>13.0f}  "
            f"{item['slow_or_stopped']:>15.0f}  "
            f"{item['total_wait']:>12.1f}"
        )

    print()
    print("movement metrics near stop line:")
    print("label   exists  queue  total_wait_s  blocked  active_core")
    print("-" * 70)

    for label in MOVEMENT_LABELS:
        item = stats[label]

        if item["exists"] <= 0 and item["queue"] <= 0 and item["wait"] <= 0:
            continue

        is_active_core = label in active_core

        print(
            f"{label:5}  "
            f"{int(item['exists']):>6}  "
            f"{item['queue']:>5.0f}  "
            f"{item['wait']:>12.1f}  "
            f"{int(item['blocked']):>7}  "
            f"{str(is_active_core):>11}"
        )

    print("=" * 100)


# ============================================================
# Discover usable traffic lights
# ============================================================

def base_sumo_cmd(route_file=None, gui=False, end=10):
    binary = SUMO_GUI_BINARY if gui else SUMO_HEADLESS_BINARY

    cmd = [
        binary,
        "-n", NET_FILE,
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(end),
        *QUIET_SUMO_ARGS,
    ]

    if route_file:
        cmd.extend(["-r", route_file])

    return cmd


def discover_usable_tls(route_file=None, verbose=True, return_skipped=False):
    traci.start(base_sumo_cmd(route_file=route_file, gui=False, end=10))

    usable = []
    skipped = []

    try:
        tls_ids = list(traci.trafficlight.getIDList())

        for tls_id in tls_ids:
            controller = build_controller_for_tls(
                tls_id,
                activate=False,
                require_safe=False,
            )

            if controller is None:
                skipped.append({
                    "id": tls_id,
                    "reason": "unusable: could not classify into at least 2 safe phases",
                    "details": [],
                })
                continue

            safe, messages = verify_controller_safety(tls_id, controller)

            if not safe:
                skipped.append({
                    "id": tls_id,
                    "reason": "unsafe: failed phase-safety verification",
                    "details": messages,
                })
                continue

            usable.append(tls_id)

            if verbose:
                print()
                print(f"TRAINED / usable: {tls_id}")
                for phase in controller["phases"]:
                    print(f"  action {phase['slot'] + 1}: {phase['name']}")

    finally:
        traci.close()

    if verbose:
        print()
        print("=" * 100)
        print("TRAINING INTERSECTION SUMMARY")
        print("=" * 100)

        print()
        print(f"Intersections that WILL be trained: {len(usable)}")
        for tls_id in usable:
            print(f"  TRAINED:     {tls_id}")

        print()
        print(f"Intersections that will NOT be trained: {len(skipped)}")
        for item in skipped:
            print(f"  NOT TRAINED: {item['id']}")
            print(f"               reason: {item['reason']}")

            for detail in item["details"][:3]:
                print(f"               detail: {detail}")

            if len(item["details"]) > 3:
                print(f"               ... {len(item['details']) - 3} more details")

    if return_skipped:
        return usable, skipped

    return usable


def verify_all_tls(route_file=None):
    traci.start(base_sumo_cmd(route_file=route_file, gui=False, end=10))

    passed = []
    failed = []
    unusable = []

    try:
        for tls_id in traci.trafficlight.getIDList():
            controller = build_controller_for_tls(
                tls_id,
                activate=False,
                require_safe=False,
            )

            if controller is None:
                unusable.append(tls_id)
                continue

            safe, messages = verify_controller_safety(tls_id, controller)

            if safe:
                passed.append(tls_id)
            else:
                failed.append((tls_id, messages))

    finally:
        traci.close()

    print()
    print("Verification summary:")
    print(f"  safe usable traffic lights: {len(passed)}")
    print(f"  unsafe traffic lights:      {len(failed)}")
    print(f"  unusable traffic lights:    {len(unusable)}")

    if passed:
        print()
        print("Safe / trained intersections:")
        for tls_id in passed:
            print(f"  TRAINED:     {tls_id}")

    if failed:
        print()
        print("Unsafe / skipped intersections:")

        for tls_id, messages in failed:
            print(f"  NOT TRAINED: {tls_id}")
            for msg in messages[:5]:
                print(f"               detail: {msg}")

            if len(messages) > 5:
                print(f"               ... {len(messages) - 5} more details")

    if unusable:
        print()
        print("Unusable / skipped intersections:")
        for tls_id in unusable:
            print(f"  NOT TRAINED: {tls_id}")
            print("               reason: unusable or fewer than 2 safe phases")

    return passed, failed, unusable


# ============================================================
# Gymnasium environment
# ============================================================

class SharedIntersectionEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tls_ids,
        gui=False,
        randomize_traffic=True,
        route_variants=None,
        max_vehicle_variants=None,
    ):
        if gym is None or spaces is None:
            raise ImportError(
                "Missing dependencies. Run:\n"
                "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
            )

        if not tls_ids:
            raise RuntimeError("No usable traffic lights were found.")

        self.tls_ids = list(tls_ids)
        self.gui = gui
        self.randomize_traffic = randomize_traffic
        self.route_variants = route_variants or discover_background_route_variants()
        self.max_vehicle_variants = max_vehicle_variants or MAX_VEHICLE_VARIANTS

        self.current_route_file = BACKGROUND_ROUTE_FILE
        self.current_max_num_vehicles = MAX_NUM_VEHICLES
        self.current_sumo_seed = 42
        self.current_tls_id = None

        self.started = False
        self.controller = None

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def choose_episode_scenario(self):
        if self.randomize_traffic:
            self.current_route_file = random.choice(self.route_variants)
            self.current_max_num_vehicles = random.choice(self.max_vehicle_variants)
            self.current_sumo_seed = random.randint(SUMO_SEED_MIN, SUMO_SEED_MAX)
        else:
            self.current_route_file = BACKGROUND_ROUTE_FILE
            self.current_max_num_vehicles = MAX_NUM_VEHICLES
            self.current_sumo_seed = 42

        self.current_tls_id = random.choice(self.tls_ids)

    def _sumo_cmd(self):
        binary = SUMO_GUI_BINARY if self.gui else SUMO_HEADLESS_BINARY

        cmd = [
            binary,
            "-n", NET_FILE,
            "-r", self.current_route_file,
            "--start",
            "--step-length", str(STEP_LENGTH),
            "--end", str(TRAIN_EPISODE_SECONDS),
            "--max-num-vehicles", str(self.current_max_num_vehicles),
            "--max-depart-delay", str(MAX_DEPART_DELAY),
            "--time-to-teleport", str(TIME_TO_TELEPORT),
            "--seed", str(self.current_sumo_seed),
            *QUIET_SUMO_ARGS,
        ]

        if TRAIN_WITH_SUMO_LOGS:
            cmd.extend([
                "--log", SUMO_RUN_LOG,
                "--error-log", SUMO_ERROR_LOG,
            ])

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
                f"Training on tls={self.current_tls_id}, "
                f"route={os.path.basename(self.current_route_file)}, "
                f"max_vehicles={self.current_max_num_vehicles}, "
                f"seed={self.current_sumo_seed}"
            )

        traci.start(self._sumo_cmd())
        self.started = True

        self.controller = build_controller_for_tls(
            self.current_tls_id,
            activate=True,
            require_safe=True,
        )

        if self.controller is None:
            raise RuntimeError(
                f"Selected traffic light became unusable: {self.current_tls_id}"
            )

        return get_observation(self.controller), {}

    def action_masks(self):
        return action_mask_for_controller(self.controller)

    def step(self, action):
        switched = apply_model_action(self.controller, int(action))

        alive, arrived_interval = run_control_steps(
            DECISION_INTERVAL,
            [self.controller],
        )

        obs = get_observation(self.controller)
        reward = compute_reward(
            self.controller,
            switched=switched,
            arrived_interval=arrived_interval,
        )

        sim_time = traci.simulation.getTime()

        terminated = sim_time >= TRAIN_EPISODE_SECONDS
        truncated = not alive or traci.simulation.getMinExpectedNumber() <= 0

        phase = self.controller["phases"][self.controller["phase_pos"]]

        info = {
            "tls_id": self.current_tls_id,
            "sim_time": sim_time,
            "phase_name": phase["name"],
            "phase_slot": phase["slot"],
            "mode": self.controller["mode"],
            "switched": switched,
            "arrived_interval": arrived_interval,
            "route": os.path.basename(self.current_route_file),
            "max_num_vehicles": self.current_max_num_vehicles,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.started:
            try:
                traci.close()
            except Exception:
                pass

        self.started = False


# ============================================================
# Training / running
# ============================================================

def train_model(timesteps, model_path, gui=False, randomize_traffic=True, fresh=False):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise ImportError(
            "Missing sb3-contrib. Run:\n"
            "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    route_variants = discover_background_route_variants()

    print()
    print("Discovering usable traffic lights...")

    tls_ids, skipped_tls = discover_usable_tls(
        route_file=BACKGROUND_ROUTE_FILE,
        verbose=False,
        return_skipped=True,
    )

    print()
    print("=" * 100)
    print("INTERSECTIONS INCLUDED IN TRAINING")
    print("=" * 100)
    print(f"Total trained intersections: {len(tls_ids)}")
    for tls_id in tls_ids:
        print(f"  TRAINED:     {tls_id}")

    print()
    print("=" * 100)
    print("INTERSECTIONS NOT INCLUDED IN TRAINING")
    print("=" * 100)
    print(f"Total skipped intersections: {len(skipped_tls)}")
    for item in skipped_tls:
        print(f"  NOT TRAINED: {item['id']}")
        print(f"               reason: {item['reason']}")
        for detail in item["details"][:3]:
            print(f"               detail: {detail}")
        if len(item["details"]) > 3:
            print(f"               ... {len(item['details']) - 3} more details")

    if not tls_ids:
        raise RuntimeError("No usable traffic lights found. Cannot train.")

    print()
    print("Training route variants:")
    for route_file in route_variants:
        print(f"  {os.path.basename(route_file)}")

    env = SharedIntersectionEnv(
        tls_ids=tls_ids,
        gui=gui,
        randomize_traffic=randomize_traffic,
        route_variants=route_variants,
        max_vehicle_variants=MAX_VEHICLE_VARIANTS,
    )

    model_zip_path = model_path
    if not model_zip_path.endswith(".zip"):
        model_zip_path += ".zip"

    resume = (not fresh) and os.path.exists(model_zip_path)

    if resume:
        print()
        print(f"Loading existing model and continuing training: {model_zip_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        print()
        print("Creating new shared all-intersections model.")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
        )

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(model_path)
    env.close()

    print()
    print(f"Saved model to: {model_path}")


def run_all_intersections(model_path, gui=True, watch_tls=None, watch_every=30.0):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise ImportError(
            "Missing sb3-contrib. Run:\n"
            "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    if gui:
        ensure_xquartz()

    binary = SUMO_GUI_BINARY if gui else SUMO_HEADLESS_BINARY

    sumo_cmd = [
        binary,
        "-n", NET_FILE,
        "-r", BACKGROUND_ROUTE_FILE,
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(SIM_END),
        "--max-num-vehicles", str(MAX_NUM_VEHICLES),
        "--max-depart-delay", str(MAX_DEPART_DELAY),
        "--time-to-teleport", str(TIME_TO_TELEPORT),
        *QUIET_SUMO_ARGS,
    ]

    print()
    print("Starting all-intersection model simulation:")
    print(" ".join(sumo_cmd))

    model = MaskablePPO.load(model_path)

    traci.start(sumo_cmd)

    try:
        controllers = []

        for tls_id in traci.trafficlight.getIDList():
            controller = build_controller_for_tls(
                tls_id,
                activate=True,
                require_safe=True,
            )

            if controller is not None:
                controllers.append(controller)

        print()
        print(f"Controlling {len(controllers)} traffic lights.")

        if watch_tls is not None:
            watched_exists = any(c["tls_id"] == watch_tls for c in controllers)
            if watched_exists:
                print(f"Watching detailed metrics for: {watch_tls}")
            else:
                print(f"WARNING: watch TLS is not controlled or not found: {watch_tls}")

        if not controllers:
            raise RuntimeError("No safe usable traffic lights were found.")

        next_print_time = 0.0
        next_watch_time = 0.0
        last_actions = {}

        while traci.simulation.getMinExpectedNumber() > 0:
            for controller in controllers:
                if controller.get("disabled"):
                    continue

                obs = get_observation(controller)
                mask = action_mask_for_controller(controller)

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=mask,
                )

                action = int(action)
                last_actions[controller["tls_id"]] = action

                apply_model_action(controller, action)

            alive, arrived_interval = run_control_steps(
                DECISION_INTERVAL,
                controllers,
            )

            if not alive:
                break

            sim_time = traci.simulation.getTime()

            if sim_time >= next_print_time:
                active = traci.vehicle.getIDCount()

                total_queue = 0.0
                total_wait = 0.0

                for controller in controllers:
                    if controller.get("disabled"):
                        continue

                    wait, queue = total_controlled_wait_and_queue(controller)
                    total_wait += wait
                    total_queue += queue

                print(
                    f"t={sim_time:7.1f}, "
                    f"active={active:5d}, "
                    f"controlled_tls={len(controllers):3d}, "
                    f"total_queue={total_queue:.0f}, "
                    f"total_wait={total_wait:.1f}, "
                    f"arrived={arrived_interval}"
                )

                next_print_time += 30.0

            if watch_tls is not None and sim_time >= next_watch_time:
                watched = None

                for controller in controllers:
                    if controller["tls_id"] == watch_tls:
                        watched = controller
                        break

                if watched is None:
                    print()
                    print(f"WARNING: watched TLS was not found or not controlled: {watch_tls}")
                else:
                    print_intersection_metrics(
                        watched,
                        last_action=last_actions.get(watch_tls),
                    )

                next_watch_time += watch_every

    finally:
        try:
            traci.close()
        except Exception:
            pass

        print()
        print("All-intersection simulation ended.")


def list_tls():
    discover_usable_tls(route_file=BACKGROUND_ROUTE_FILE, verbose=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "generate-routes",
            "list-tls",
            "verify-all",
            "train",
            "run-all",
        ],
        default="list-tls",
    )

    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--model", default=MODEL_FILE)

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--nogui", action="store_true")
    parser.add_argument("--fresh", action="store_true")

    parser.add_argument(
        "--fixed-traffic",
        action="store_true",
        help="Disable randomized route/max-vehicle training scenarios.",
    )

    parser.add_argument(
        "--traffic-periods",
        default=",".join(str(x) for x in DEFAULT_TRAINING_TRAFFIC_PERIODS),
    )

    parser.add_argument(
        "--route-seeds",
        default=",".join(str(x) for x in DEFAULT_TRAINING_ROUTE_SEEDS),
    )

    parser.add_argument(
        "--watch-tls",
        default=None,
        help="Traffic light ID to print detailed per-intersection metrics for during run-all.",
    )

    parser.add_argument(
        "--watch-every",
        type=float,
        default=30.0,
        help="How often to print watched-intersection metrics.",
    )

    args = parser.parse_args()

    if args.mode == "generate-routes":
        periods = parse_float_list(args.traffic_periods)
        route_seeds = parse_int_list(args.route_seeds)
        generate_route_variants(periods, route_seeds)

    elif args.mode == "list-tls":
        list_tls()

    elif args.mode == "verify-all":
        verify_all_tls(route_file=BACKGROUND_ROUTE_FILE)

    elif args.mode == "train":
        train_model(
            timesteps=args.timesteps,
            model_path=args.model,
            gui=args.gui,
            randomize_traffic=not args.fixed_traffic,
            fresh=args.fresh,
        )

    elif args.mode == "run-all":
        run_all_intersections(
            model_path=args.model,
            gui=not args.nogui,
            watch_tls=args.watch_tls,
            watch_every=args.watch_every,
        )


if __name__ == "__main__":
    main()
