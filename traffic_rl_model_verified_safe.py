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
# SUMO / file setup
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
AMBULANCE_ROUTE_FILE = os.path.join(BASE_DIR, "ambulance_random_new.rou.xml")
RANDOM_TRIPS_SCRIPT = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

SUMO_RUN_LOG = os.path.join(BASE_DIR, "sumo_new_run.log")
SUMO_ERROR_LOG = os.path.join(BASE_DIR, "sumo_new_error.log")

TARGET_TLS_ID = "cluster_12179861947_12179861948_12179861949_12185616643_#11more"
REQUIRE_ALL_FOUR_PHASES_FOR_TARGET = True

MODEL_FILE = os.path.join(BASE_DIR, "randomized_four_way_model")

QUIET_SUMO_ARGS = [
    "--no-warnings", "true",
    "--no-step-log", "true",
]


# ============================================================
# Simulation constants
# ============================================================

STEP_LENGTH = 0.5

T_GREEN_STRAIGHT = 30.0
T_GREEN_LEFT = 15.0
T_YELLOW = 4.0
T_ALL_RED = 2.0

DECISION_INTERVAL = 5.0
MIN_GREEN_BEFORE_SWITCH = 8.0
MAX_GREEN_HOLD = 60.0

REQUIRED_EXIT_GAP = 14.0

SIM_END = 7200
TRAIN_EPISODE_SECONDS = 1800

MAX_NUM_VEHICLES = 2000
MAX_DEPART_DELAY = 60
TIME_TO_TELEPORT = 300

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

EXCLUDED_TLS = set()

MOVEMENT_LABELS = [
    "NB-L", "NB-S", "NB-R",
    "SB-L", "SB-S", "SB-R",
    "EB-L", "EB-S", "EB-R",
    "WB-L", "WB-S", "WB-R",
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


# ============================================================
# Training route variant helpers
# ============================================================

def safe_period_name(period):
    return str(period).replace(".", "p")


def parse_float_list(raw):
    values = []

    for part in raw.split(","):
        part = part.strip()

        if not part:
            continue

        values.append(float(part))

    return values


def parse_int_list(raw):
    values = []

    for part in raw.split(","):
        part = part.strip()

        if not part:
            continue

        values.append(int(part))

    return values


def discover_background_route_variants():
    variants = sorted(Path(BASE_DIR).glob("background_train_*.rou.xml"))

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
                f"background_train_{period_name}_seed{seed}.rou.xml",
            )

            prefix = f"train_{period_name}_s{seed}_car_"

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

            print("\nGenerating route file:")
            print(" ".join(cmd))

            subprocess.run(cmd, check=True)

            patch_car_vtype(output_file)
            generated_files.append(output_file)

            print(f"Generated and patched: {output_file}")

    print("\nFinished generating randomized training route files.")
    print("Generated files:")

    for file in generated_files:
        print(f"  {os.path.basename(file)}")

    return generated_files


# ============================================================
# macOS / XQuartz helper
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
# Signal-state builders
# ============================================================

def all_red_state(length):
    return "r" * length


def build_yellow_state(length, active_indices):
    state = ["r"] * length

    for idx in active_indices:
        if 0 <= idx < length:
            state[idx] = "y"

    return "".join(state)


def build_state_from_movements(state_length, movement_map, phase_rules):
    """
    Builds actual traffic-light state from one of the four safe phase templates.

    The model only chooses among four phase templates.
    All right turns are included in every phase as permissive green g.
    Any movement, including right turns, is held red if its outgoing lane does not have enough room.
    """
    state = ["r"] * state_length
    active_indices = set()

    for movement_label, signal_char in phase_rules.items():
        if signal_char == "r":
            continue

        signal_data = movement_map.get(movement_label, {})

        for signal_index, lane_sets in signal_data.items():
            out_lanes = lane_sets["out"]

            if not lanes_have_space(out_lanes):
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


def has_any_movements(movement_map, labels):
    return any(movement_map.get(label) for label in labels)


def build_safe_phase_plan(movement_map):
    """
    The model can only choose these four phase families:

        action 1 = PHASE 1: N/S Protected Lefts
        action 2 = PHASE 2: N/S Straights
        action 3 = PHASE 3: E/W Protected Lefts
        action 4 = PHASE 4: E/W Straights

    action 0 means keep the current phase.

    Corrected safety rule:
        - Left turns are NOT permissive green during straight phases.
        - Straight phases contain only the two opposite straight movements plus all right turns.
        - Protected-left phases contain only the two opposite left movements plus all right turns.
        - Right turns are permissive green in every phase when their exit path has space.
    """
    phases = []

    phase1_rules = {
        # N/S protected lefts only.
        "NB-L": "G",
        "SB-L": "G",
        **ALL_RIGHT_TURNS,
    }

    phase2_rules = {
        # N/S straights only.
        "NB-S": "G",
        "SB-S": "G",
        **ALL_RIGHT_TURNS,
    }

    phase3_rules = {
        # E/W protected lefts only.
        "EB-L": "G",
        "WB-L": "G",
        **ALL_RIGHT_TURNS,
    }

    phase4_rules = {
        # E/W straights only.
        "EB-S": "G",
        "WB-S": "G",
        **ALL_RIGHT_TURNS,
    }

    phase_defs = [
        {
            "slot": 0,
            "name": PHASE_SLOT_NAMES[0],
            "duration": T_GREEN_LEFT,
            "rules": phase1_rules,
            "required_movements": ["NB-L", "SB-L"],
        },
        {
            "slot": 1,
            "name": PHASE_SLOT_NAMES[1],
            "duration": T_GREEN_STRAIGHT,
            "rules": phase2_rules,
            "required_movements": ["NB-S", "SB-S"],
        },
        {
            "slot": 2,
            "name": PHASE_SLOT_NAMES[2],
            "duration": T_GREEN_LEFT,
            "rules": phase3_rules,
            "required_movements": ["EB-L", "WB-L"],
        },
        {
            "slot": 3,
            "name": PHASE_SLOT_NAMES[3],
            "duration": T_GREEN_STRAIGHT,
            "rules": phase4_rules,
            "required_movements": ["EB-S", "WB-S"],
        },
    ]

    for phase in phase_defs:
        if has_any_movements(movement_map, phase["required_movements"]):
            phases.append(phase)

    return phases


def validate_four_way_target(controller, tls_id):
    if not REQUIRE_ALL_FOUR_PHASES_FOR_TARGET:
        return

    if tls_id != TARGET_TLS_ID:
        return

    existing_slots = {phase["slot"] for phase in controller["phases"]}
    required_slots = {0, 1, 2, 3}

    if existing_slots != required_slots:
        raise RuntimeError(
            f"Target intersection {tls_id} was expected to have all four phases, "
            f"but found phase slots {sorted(existing_slots)}. "
            f"Required slots are {sorted(required_slots)}."
        )


# ============================================================
# Controller functions
# ============================================================

def build_controller_for_tls(tls_id, activate=True):
    state_length, movement_map = classify_tls_movements(tls_id)
    phases = build_safe_phase_plan(movement_map)

    if len(phases) < 2:
        return None

    slot_to_pos = {
        phase["slot"]: i
        for i, phase in enumerate(phases)
    }

    controller = {
        "tls_id": tls_id,
        "state_length": state_length,
        "movement_map": movement_map,
        "phases": phases,
        "slot_to_pos": slot_to_pos,
        "phase_pos": 0,
        "mode": "green",
        "remaining": 0.0,
        "phase_elapsed": 0.0,
        "last_active_indices": set(),
    }

    validate_four_way_target(controller, tls_id)

    if activate:
        start_green(tls_id, controller, phase_pos=0)

    return controller


def start_green(tls_id, controller, phase_pos=None):
    if phase_pos is not None:
        controller["phase_pos"] = phase_pos

    phase = controller["phases"][controller["phase_pos"]]

    green_state, active_indices = build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
    )

    traci.trafficlight.setRedYellowGreenState(tls_id, green_state)

    controller["mode"] = "green"
    controller["remaining"] = phase["duration"]
    controller["phase_elapsed"] = 0.0
    controller["last_active_indices"] = active_indices


def update_green(tls_id, controller):
    phase = controller["phases"][controller["phase_pos"]]

    green_state, active_indices = build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
    )

    traci.trafficlight.setRedYellowGreenState(tls_id, green_state)
    controller["last_active_indices"] = active_indices


def start_yellow(tls_id, controller):
    yellow_state = build_yellow_state(
        controller["state_length"],
        controller["last_active_indices"],
    )

    traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)

    controller["mode"] = "yellow"
    controller["remaining"] = T_YELLOW


def start_all_red(tls_id, controller):
    traci.trafficlight.setRedYellowGreenState(
        tls_id,
        all_red_state(controller["state_length"]),
    )

    controller["mode"] = "all_red"
    controller["remaining"] = T_ALL_RED


def run_steps(seconds, tls_id=None, controller=None):
    steps = int(round(seconds / STEP_LENGTH))

    for _ in range(steps):
        if traci.simulation.getMinExpectedNumber() <= 0:
            return False

        traci.simulationStep()

        if tls_id is not None and controller is not None:
            if controller["mode"] == "green":
                update_green(tls_id, controller)
                controller["phase_elapsed"] += STEP_LENGTH

    return True


def switch_to_phase(tls_id, controller, new_phase_pos):
    if new_phase_pos == controller["phase_pos"]:
        return False

    start_yellow(tls_id, controller)
    run_steps(T_YELLOW)

    start_all_red(tls_id, controller)
    run_steps(T_ALL_RED)

    start_green(tls_id, controller, phase_pos=new_phase_pos)
    return True


def cyclic_heuristic_update(tls_id, controller):
    if controller["mode"] == "green":
        update_green(tls_id, controller)
        controller["remaining"] -= STEP_LENGTH
        controller["phase_elapsed"] += STEP_LENGTH

        if controller["remaining"] <= 0:
            start_yellow(tls_id, controller)

    elif controller["mode"] == "yellow":
        controller["remaining"] -= STEP_LENGTH

        if controller["remaining"] <= 0:
            start_all_red(tls_id, controller)

    elif controller["mode"] == "all_red":
        controller["remaining"] -= STEP_LENGTH

        if controller["remaining"] <= 0:
            controller["phase_pos"] = (
                controller["phase_pos"] + 1
            ) % len(controller["phases"])

            start_green(tls_id, controller)


# ============================================================
# Observation / reward
# ============================================================

def movement_in_lanes(movement_map, movement_label):
    lanes = set()

    for lane_sets in movement_map[movement_label].values():
        lanes.update(lane_sets["in"])

    return lanes


def movement_queue_and_wait(movement_map, movement_label):
    lanes = movement_in_lanes(movement_map, movement_label)

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

            if speed < 0.1:
                queue += 1.0

            wait += traci.vehicle.getWaitingTime(veh_id)

        except traci.TraCIException:
            pass

    return queue, wait


def total_controlled_wait_and_queue(controller):
    all_in_lanes = set()

    for label in MOVEMENT_LABELS:
        all_in_lanes.update(movement_in_lanes(controller["movement_map"], label))

    veh_ids = set()

    for lane_id in all_in_lanes:
        try:
            veh_ids.update(traci.lane.getLastStepVehicleIDs(lane_id))
        except traci.TraCIException:
            pass

    total_queue = 0.0
    total_wait = 0.0

    for veh_id in veh_ids:
        try:
            if traci.vehicle.getSpeed(veh_id) < 0.1:
                total_queue += 1.0

            total_wait += traci.vehicle.getWaitingTime(veh_id)

        except traci.TraCIException:
            pass

    return total_wait, total_queue


def get_observation(controller):
    obs = []

    for label in MOVEMENT_LABELS:
        q, w = movement_queue_and_wait(controller["movement_map"], label)

        obs.append(q / 100.0)
        obs.append(w / 1000.0)

    phase_one_hot = [0.0, 0.0, 0.0, 0.0]

    current_phase = controller["phases"][controller["phase_pos"]]
    current_slot = current_phase["slot"]

    phase_one_hot[current_slot] = 1.0

    obs.extend(phase_one_hot)

    obs.append(controller["phase_elapsed"] / MAX_GREEN_HOLD)
    obs.append(traci.simulation.getTime() / SIM_END)

    return np.array(obs, dtype=np.float32)


def compute_reward(controller, switched):
    """
    Rescaled reward.

    This version is intentionally smaller in magnitude than before so PPO's
    value network can learn more easily.

    Watch for:
        explained_variance rising above 0
        value_loss becoming much smaller
        ep_rew_mean becoming less negative over time
    """
    total_wait, total_queue = total_controlled_wait_and_queue(controller)

    reward = 0.0

    # Main traffic objective, scaled down to reduce noisy/huge returns.
    reward -= total_wait / 500.0
    reward -= total_queue / 50.0

    # Mild switch penalty. Switching already costs time due to yellow/all-red.
    if switched:
        reward -= 0.2

    # Mild penalty for letting a phase run too long.
    if controller["phase_elapsed"] > MAX_GREEN_HOLD:
        reward -= 1.0

    return float(reward)


# ============================================================
# Gymnasium environment
# ============================================================

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

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
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
        binary = SUMO_GUI_BINARY if self.gui else SUMO_HEADLESS_BINARY

        route_file = (
            f"{self.current_background_route_file},"
            f"{AMBULANCE_ROUTE_FILE}"
        )

        return [
            binary,
            "-n", NET_FILE,
            "-r", route_file,
            "--start",
            "--step-length", str(STEP_LENGTH),
            "--end", str(TRAIN_EPISODE_SECONDS),
            "--max-num-vehicles", str(self.current_max_num_vehicles),
            "--max-depart-delay", str(MAX_DEPART_DELAY),
            "--time-to-teleport", str(TIME_TO_TELEPORT),
            "--seed", str(self.current_sumo_seed),
            *QUIET_SUMO_ARGS,
            "--log", SUMO_RUN_LOG,
            "--error-log", SUMO_ERROR_LOG,
        ]

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

        print(
            "Training scenario: "
            f"route={os.path.basename(self.current_background_route_file)}, "
            f"max_vehicles={self.current_max_num_vehicles}, "
            f"sumo_seed={self.current_sumo_seed}"
        )

        traci.start(self._sumo_cmd())
        self.started = True
        self.controller = None

        if self.tls_id is None:
            chosen_tls = TARGET_TLS_ID
        else:
            chosen_tls = self.tls_id

        self.controller = build_controller_for_tls(chosen_tls, activate=True)

        if self.controller is None:
            raise RuntimeError(f"Traffic light {chosen_tls} is not usable.")

        return get_observation(self.controller), {}

    def action_masks(self):
        mask = np.zeros(5, dtype=bool)
        mask[0] = True

        if self.controller is None:
            return mask

        for phase in self.controller["phases"]:
            action_index = phase["slot"] + 1
            mask[action_index] = True

        return mask

    def step(self, action):
        action = int(action)
        switched = False

        if action > 0:
            desired_slot = action - 1

            if desired_slot in self.controller["slot_to_pos"]:
                desired_phase_pos = self.controller["slot_to_pos"][desired_slot]

                can_switch = (
                    self.controller["phase_elapsed"] >= MIN_GREEN_BEFORE_SWITCH
                )

                if can_switch and desired_phase_pos != self.controller["phase_pos"]:
                    switched = switch_to_phase(
                        self.controller["tls_id"],
                        self.controller,
                        desired_phase_pos,
                    )

        if self.controller["phase_elapsed"] >= MAX_GREEN_HOLD:
            next_pos = (
                self.controller["phase_pos"] + 1
            ) % len(self.controller["phases"])

            switched = switch_to_phase(
                self.controller["tls_id"],
                self.controller,
                next_pos,
            )

        run_steps(
            DECISION_INTERVAL,
            self.controller["tls_id"],
            self.controller,
        )

        obs = get_observation(self.controller)
        reward = compute_reward(self.controller, switched)

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
# Utility / running modes
# ============================================================

def list_tls():
    traci.start([
        SUMO_HEADLESS_BINARY,
        "-n", NET_FILE,
        "-r", f"{BACKGROUND_ROUTE_FILE},{AMBULANCE_ROUTE_FILE}",
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", "10",
        *QUIET_SUMO_ARGS,
        "--log", SUMO_RUN_LOG,
        "--error-log", SUMO_ERROR_LOG,
    ])

    print("Traffic lights:")

    for tls_id in traci.trafficlight.getIDList():
        try:
            controller = build_controller_for_tls(tls_id, activate=False)
        except RuntimeError as e:
            print(f"{tls_id}: not usable ({e})")
            continue

        if controller is None:
            print(f"{tls_id}: not usable")
            continue

        target_marker = "  <-- TARGET" if tls_id == TARGET_TLS_ID else ""
        print(f"{tls_id}: usable phases/actions:{target_marker}")

        for phase in controller["phases"]:
            action_id = phase["slot"] + 1
            print(f"  action {action_id}: {phase['name']}")

    traci.close()


def train_model(tls_id, timesteps, model_path, gui=False, randomize_traffic=True):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise ImportError(
            "Missing sb3-contrib. Run:\n"
            "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    route_variants = discover_background_route_variants()

    print("\nTraining route variants:")
    for route_file in route_variants:
        print(f"  {os.path.basename(route_file)}")

    print("\nMax vehicle variants:")
    for value in MAX_VEHICLE_VARIANTS:
        print(f"  {value}")

    print(f"\nRandomize traffic during training: {randomize_traffic}")

    env = TrafficSignalEnv(
        tls_id=tls_id,
        gui=gui,
        randomize_traffic=randomize_traffic,
        route_variants=route_variants,
        max_vehicle_variants=MAX_VEHICLE_VARIANTS,
    )

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
    )

    model.save(model_path)
    env.close()

    print(f"Saved model to: {model_path}")


def run_trained_model(tls_id, model_path, gui=True, randomize_traffic=False):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise ImportError(
            "Missing sb3-contrib. Run:\n"
            "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    model = MaskablePPO.load(model_path)

    env = TrafficSignalEnv(
        tls_id=tls_id,
        gui=gui,
        randomize_traffic=randomize_traffic,
    )

    obs, _ = env.reset()

    try:
        while True:
            mask = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"t={info['sim_time']:.1f}, "
                f"tls={info['tls_id']}, "
                f"phase={info['phase_name']}, "
                f"action={int(action)}, "
                f"reward={reward:.2f}, "
                f"route={info['background_route']}, "
                f"max_vehicles={info['max_num_vehicles']}"
            )

            if terminated or truncated:
                break

    finally:
        env.close()


def run_heuristic_controller(gui=True):
    binary = SUMO_GUI_BINARY if gui else SUMO_HEADLESS_BINARY

    if gui:
        ensure_xquartz()

    sumo_cmd = [
        binary,
        "-n", NET_FILE,
        "-r", f"{BACKGROUND_ROUTE_FILE},{AMBULANCE_ROUTE_FILE}",
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(SIM_END),
        "--max-num-vehicles", str(MAX_NUM_VEHICLES),
        "--max-depart-delay", str(MAX_DEPART_DELAY),
        "--time-to-teleport", str(TIME_TO_TELEPORT),
        *QUIET_SUMO_ARGS,
        "--log", SUMO_RUN_LOG,
        "--error-log", SUMO_ERROR_LOG,
    ]

    print("Starting SUMO:")
    print(" ".join(sumo_cmd))

    traci.start(sumo_cmd)

    controllers = {}

    print("\nBuilding safe fixed-phase controllers:")

    for tls_id in traci.trafficlight.getIDList():
        if tls_id in EXCLUDED_TLS:
            print(f"{tls_id}: excluded")
            continue

        try:
            controller = build_controller_for_tls(tls_id, activate=True)
        except RuntimeError as e:
            print(f"{tls_id}: skipped ({e})")
            continue

        if controller is None:
            print(f"{tls_id}: skipped")
            continue

        controllers[tls_id] = controller

        target_marker = "  <-- TARGET" if tls_id == TARGET_TLS_ID else ""
        print(f"{tls_id}: controlled{target_marker}")
        for phase in controller["phases"]:
            print(f"  - {phase['name']}")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            for tls_id, controller in list(controllers.items()):
                try:
                    cyclic_heuristic_update(tls_id, controller)
                except traci.TraCIException as e:
                    print(f"{tls_id}: disabling controller due to error: {e}")
                    controllers.pop(tls_id, None)

    except traci.exceptions.FatalTraCIError as e:
        print("\nSUMO closed the TraCI connection.")
        print(f"Python-side error: {e}")
        print(f"Check SUMO error log: {SUMO_ERROR_LOG}")
        print(f"Check SUMO run log: {SUMO_RUN_LOG}")

    finally:
        try:
            traci.close()
        except Exception:
            pass

        print("\nSimulation ended.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["generate-routes", "list-tls", "heuristic", "train", "run-model"],
        default="heuristic",
    )

    parser.add_argument(
        "--tls",
        default=None,
        help=(
            "Traffic light ID. If omitted, defaults to TARGET_TLS_ID: "
            f"{TARGET_TLS_ID}"
        ),
    )

    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--nogui", action="store_true")
    parser.add_argument("--gui", action="store_true")

    parser.add_argument(
        "--fixed-traffic",
        action="store_true",
        help="Disable randomized route/max-vehicle training scenarios.",
    )

    parser.add_argument(
        "--randomize-traffic",
        action="store_true",
        help="Use randomized route/max-vehicle scenario during run-model.",
    )

    parser.add_argument(
        "--traffic-periods",
        default=",".join(str(x) for x in DEFAULT_TRAINING_TRAFFIC_PERIODS),
        help="Comma-separated randomTrips.py -p values for generate-routes.",
    )

    parser.add_argument(
        "--route-seeds",
        default=",".join(str(x) for x in DEFAULT_TRAINING_ROUTE_SEEDS),
        help="Comma-separated randomTrips.py seeds for generate-routes.",
    )

    args = parser.parse_args()

    if args.mode == "generate-routes":
        periods = parse_float_list(args.traffic_periods)
        route_seeds = parse_int_list(args.route_seeds)
        generate_route_variants(periods, route_seeds)

    elif args.mode == "list-tls":
        list_tls()

    elif args.mode == "heuristic":
        run_heuristic_controller(gui=not args.nogui)

    elif args.mode == "train":
        train_model(
            tls_id=args.tls,
            timesteps=args.timesteps,
            model_path=args.model,
            gui=args.gui,
            randomize_traffic=not args.fixed_traffic,
        )

    elif args.mode == "run-model":
        run_trained_model(
            tls_id=args.tls,
            model_path=args.model,
            gui=not args.nogui,
            randomize_traffic=args.randomize_traffic,
        )


if __name__ == "__main__":
    main()
