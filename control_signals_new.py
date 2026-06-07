import math
import os
import traci

SUMO_BINARY = "sumo-gui"
# SUMO_BINARY = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin/sumo-gui"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NET_FILE = os.path.join(BASE_DIR, "new_map.net.xml")
BACKGROUND_ROUTE_FILE = os.path.join(BASE_DIR, "background_new.rou.xml")
AMBULANCE_ROUTE_FILE = os.path.join(BASE_DIR, "ambulance_random_new.rou.xml")
ROUTE_FILE = f"{BACKGROUND_ROUTE_FILE},{AMBULANCE_ROUTE_FILE}"

SUMO_RUN_LOG = os.path.join(BASE_DIR, "sumo_new_run.log")
SUMO_ERROR_LOG = os.path.join(BASE_DIR, "sumo_new_error.log")

# Simulation timing
STEP_LENGTH = 0.5

# Required signal timing
T_GREEN_STRAIGHT = 30.0
T_GREEN_LEFT = 15.0
T_YELLOW = 4.0
T_ALL_RED = 2.0

# Anti-blocking rule: require space after the intersection
REQUIRED_EXIT_GAP = 14.0

SIM_END = 7200
MAX_NUM_VEHICLES = 1200
MAX_DEPART_DELAY = 60
TIME_TO_TELEPORT = 300

EXCLUDED_TLS = set()


# -----------------------------
# Geometry helpers
# -----------------------------

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
    """
    Classifies incoming movement into NB, SB, EB, WB.

    Vector points toward the intersection:
        NB means vehicle is traveling northbound.
        SB means vehicle is traveling southbound.
        EB means vehicle is traveling eastbound.
        WB means vehicle is traveling westbound.
    """
    dx, dy = in_vec

    if abs(dy) >= abs(dx):
        return "NB" if dy > 0 else "SB"
    else:
        return "EB" if dx > 0 else "WB"


def classify_movement(angle):
    """
    Returns S, L, or R.
    """
    if abs(angle) <= 35:
        return "S"

    if angle > 35:
        return "L"

    return "R"


# -----------------------------
# Exit-space / anti-gridlock
# -----------------------------

def outgoing_lane_has_space(out_lane_id):
    """
    Checks whether there is enough room immediately after the intersection.

    If not, that movement is held red even if its phase would normally be green.
    This helps prevent cars from blocking the box.
    """
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


# -----------------------------
# Signal state builders
# -----------------------------

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
    phase_rules maps movement labels to signal chars:
        "G" = protected green
        "g" = permissive green
        "r" = red

    Movement labels look like:
        NB-L, NB-S, NB-R, SB-L, ...
    """
    state = ["r"] * state_length
    active_indices = set()

    for movement_label, signal_char in phase_rules.items():
        if signal_char == "r":
            continue

        signal_data = movement_map.get(movement_label, {})

        for signal_index, out_lanes in signal_data.items():
            # Do not release vehicles if they cannot fit after the intersection.
            if not lanes_have_space(out_lanes):
                continue

            state[signal_index] = signal_char
            active_indices.add(signal_index)

    return "".join(state), active_indices


# -----------------------------
# Build movement map
# -----------------------------

def classify_tls_movements(tls_id):
    """
    Builds a movement map:
        NB-L, NB-S, NB-R
        SB-L, SB-S, SB-R
        EB-L, EB-S, EB-R
        WB-L, WB-S, WB-R

    Each movement maps to:
        signal_index -> set(outgoing_lanes)
    """
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    state_length = len(traci.trafficlight.getRedYellowGreenState(tls_id))

    movement_map = {}

    for approach in ("NB", "SB", "EB", "WB"):
        for movement in ("L", "S", "R"):
            movement_map[f"{approach}-{movement}"] = {}

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
                movement_map[label].setdefault(signal_index, set()).add(outgoing_lane)

            except traci.TraCIException:
                continue

    return state_length, movement_map


def has_any_movements(movement_map, labels):
    for label in labels:
        if movement_map.get(label):
            return True
    return False


def build_four_phase_plan(movement_map):
    """
    Implements:

    Phase 1: N/S Protected Lefts
        G: NB-L, SB-L
        g: NB-R, SB-R
        r: all others

    Phase 2: N/S Straights
        G: NB-S, SB-S, NB-R, SB-R
        g: NB-L, SB-L
        r: all E/W

    Phase 3: E/W Protected Lefts
        G: EB-L, WB-L
        g: EB-R, WB-R
        r: all others

    Phase 4: E/W Straights
        G: EB-S, WB-S, EB-R, WB-R
        g: EB-L, WB-L
        r: all N/S
    """
    phases = []

    phase1_rules = {
        "NB-L": "G",
        "SB-L": "G",
        "NB-R": "g",
        "SB-R": "g",
    }

    phase2_rules = {
        "NB-S": "G",
        "SB-S": "G",
        "NB-R": "G",
        "SB-R": "G",
        "NB-L": "g",
        "SB-L": "g",
    }

    phase3_rules = {
        "EB-L": "G",
        "WB-L": "G",
        "EB-R": "g",
        "WB-R": "g",
    }

    phase4_rules = {
        "EB-S": "G",
        "WB-S": "G",
        "EB-R": "G",
        "WB-R": "G",
        "EB-L": "g",
        "WB-L": "g",
    }

    # Only include phases that actually have relevant movements.
    if has_any_movements(movement_map, ["NB-L", "SB-L", "NB-R", "SB-R"]):
        phases.append({
            "name": "PHASE 1: N/S Protected Lefts",
            "duration": T_GREEN_LEFT,
            "rules": phase1_rules,
        })

    if has_any_movements(movement_map, ["NB-S", "SB-S", "NB-R", "SB-R", "NB-L", "SB-L"]):
        phases.append({
            "name": "PHASE 2: N/S Straights",
            "duration": T_GREEN_STRAIGHT,
            "rules": phase2_rules,
        })

    if has_any_movements(movement_map, ["EB-L", "WB-L", "EB-R", "WB-R"]):
        phases.append({
            "name": "PHASE 3: E/W Protected Lefts",
            "duration": T_GREEN_LEFT,
            "rules": phase3_rules,
        })

    if has_any_movements(movement_map, ["EB-S", "WB-S", "EB-R", "WB-R", "EB-L", "WB-L"]):
        phases.append({
            "name": "PHASE 4: E/W Straights",
            "duration": T_GREEN_STRAIGHT,
            "rules": phase4_rules,
        })

    return phases


# -----------------------------
# Controller state machine
# -----------------------------

def start_green(tls_id, controller):
    phase = controller["phases"][controller["phase_pos"]]

    green_state, active_indices = build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
    )

    traci.trafficlight.setRedYellowGreenState(tls_id, green_state)

    controller["mode"] = "green"
    controller["remaining"] = phase["duration"]
    controller["last_active_indices"] = active_indices

    print(f"{tls_id}: {phase['name']} GREEN for {phase['duration']}s")


def update_green(tls_id, controller):
    """
    Rebuild green every step so right/permissive movements can become active
    as soon as their outgoing lane becomes clear.
    """
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


def update_controller(tls_id, controller):
    if controller["mode"] == "green":
        update_green(tls_id, controller)
        controller["remaining"] -= STEP_LENGTH

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


# -----------------------------
# Logging
# -----------------------------

def print_recent_log_lines(path, label, n=40):
    print(f"\n--- Last {n} lines of {label} ---")

    if not os.path.exists(path):
        print(f"{path} does not exist.")
        return

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines[-n:]:
        print(line.rstrip())


# -----------------------------
# Main
# -----------------------------

def main():
    sumo_cmd = [
        SUMO_BINARY,
        "-n", NET_FILE,
        "-r", ROUTE_FILE,
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(SIM_END),
        "--max-num-vehicles", str(MAX_NUM_VEHICLES),
        "--max-depart-delay", str(MAX_DEPART_DELAY),
        "--time-to-teleport", str(TIME_TO_TELEPORT),
        "--log", SUMO_RUN_LOG,
        "--error-log", SUMO_ERROR_LOG,
    ]

    print("Starting SUMO:")
    print(" ".join(sumo_cmd))

    traci.start(sumo_cmd)

    controllers = {}

    print("\nBuilding 4-phase NB/SB/EB/WB signal controllers:")

    for tls_id in traci.trafficlight.getIDList():
        if tls_id in EXCLUDED_TLS:
            print(f"{tls_id}: excluded")
            continue

        try:
            state_length, movement_map = classify_tls_movements(tls_id)
            phases = build_four_phase_plan(movement_map)

            if len(phases) < 2:
                print(f"{tls_id}: skipped, not enough usable phases")
                continue

            controllers[tls_id] = {
                "state_length": state_length,
                "movement_map": movement_map,
                "phases": phases,
                "phase_pos": 0,
                "mode": "green",
                "remaining": 0.0,
                "last_active_indices": set(),
            }

            print(f"{tls_id}: controlled phases:")
            for phase in phases:
                print(f"  - {phase['name']}")

            start_green(tls_id, controllers[tls_id])

        except traci.TraCIException as e:
            print(f"{tls_id}: skipped due to TraCI error: {e}")

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            for tls_id, controller in list(controllers.items()):
                try:
                    update_controller(tls_id, controller)
                except traci.TraCIException as e:
                    print(f"{tls_id}: disabling controller due to error: {e}")
                    controllers.pop(tls_id, None)

    except traci.exceptions.FatalTraCIError as e:
        print("\nSUMO closed the TraCI connection.")
        print(f"Python-side error: {e}")
        print_recent_log_lines(SUMO_ERROR_LOG, "sumo_new_error.log")
        print_recent_log_lines(SUMO_RUN_LOG, "sumo_new_run.log")

    finally:
        try:
            traci.close()
        except Exception:
            pass

        print("\nSimulation ended.")
        print(f"SUMO run log: {SUMO_RUN_LOG}")
        print(f"SUMO error log: {SUMO_ERROR_LOG}")


if __name__ == "__main__":
    main()
