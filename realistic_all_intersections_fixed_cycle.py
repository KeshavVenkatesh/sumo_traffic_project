import argparse
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from xml.sax.saxutils import quoteattr

# ============================================================
# SUMO setup
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

import traci

try:
    import sumolib
except Exception:
    sumolib = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMO_BIN_DIR = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin"
SUMO_GUI_BINARY = os.path.join(SUMO_BIN_DIR, "sumo-gui")
SUMO_HEADLESS_BINARY = os.path.join(SUMO_BIN_DIR, "sumo")

if not os.path.exists(SUMO_GUI_BINARY):
    SUMO_GUI_BINARY = "sumo-gui"

if not os.path.exists(SUMO_HEADLESS_BINARY):
    SUMO_HEADLESS_BINARY = "sumo"

NET_FILE = os.path.join(BASE_DIR, "new_map.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "random_drive_dynamic_turns.rou.xml")


# ============================================================
# Constants
# ============================================================

STEP_LENGTH = 0.5
DECISION_INTERVAL = 5.0

T_YELLOW = 4.0
T_ALL_RED = 2.0

REQUIRED_EXIT_GAP = 14.0
QUEUE_SPEED_THRESHOLD = 0.1

MAX_ACTIVE_VEHICLE_CAP = 1500
DEFAULT_SIM_END = 1_000_000_000.0

CAR_LENGTH = 4.8
CAR_WIDTH = 1.8
CAR_MIN_GAP = 2.5

TURN_PROBABILITIES = {
    "S": 0.70,
    "R": 0.175,
    "L": 0.125,
}

MOVEMENT_ORDER = ["S", "R", "L"]

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

PHASE_DEFS = [
    {"slot": 0, "name": PHASE_SLOT_NAMES[0], "core": ["NB-L", "SB-L"]},
    {"slot": 1, "name": PHASE_SLOT_NAMES[1], "core": ["NB-S", "SB-S"]},
    {"slot": 2, "name": PHASE_SLOT_NAMES[2], "core": ["EB-L", "WB-L"]},
    {"slot": 3, "name": PHASE_SLOT_NAMES[3], "core": ["EB-S", "WB-S"]},
]

ALL_RIGHT_TURNS = {
    "NB-R": "g",
    "SB-R": "g",
    "EB-R": "g",
    "WB-R": "g",
}

QUIET_SUMO_ARGS = [
    "--no-warnings", "true",
    "--no-step-log", "true",
]


# ============================================================
# Utility
# ============================================================

def ensure_xquartz():
    if sys.platform != "darwin":
        return

    os.environ.setdefault("DISPLAY", ":0")

    try:
        import subprocess

        subprocess.run(
            ["open", "-a", "XQuartz"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(2)
    except Exception as exc:
        print(f"Warning: could not open XQuartz automatically: {exc}")


def lane_to_edge(lane_id):
    if not lane_id or lane_id.startswith(":"):
        return None

    if "_" not in lane_id:
        return lane_id

    return lane_id.rsplit("_", 1)[0]


def weighted_choice(rng, items, weights):
    total = sum(weights)

    if total <= 0:
        return rng.choice(items)

    r = rng.random() * total
    cumulative = 0.0

    for item, weight in zip(items, weights):
        cumulative += weight

        if r <= cumulative:
            return item

    return items[-1]


# ============================================================
# Edge validity and metadata
# ============================================================

def edge_allows_passenger(edge_id):
    if not edge_id or edge_id.startswith(":"):
        return False

    try:
        lane_count = traci.edge.getLaneNumber(edge_id)

        if lane_count <= 0:
            return False

        for lane_index in range(lane_count):
            lane_id = f"{edge_id}_{lane_index}"

            try:
                allowed = traci.lane.getAllowed(lane_id)
                disallowed = traci.lane.getDisallowed(lane_id)
                length = traci.lane.getLength(lane_id)
            except traci.TraCIException:
                continue

            if length < 15:
                continue

            if not allowed:
                if "passenger" not in disallowed and "private" not in disallowed:
                    return True

            if (
                "passenger" in allowed
                or "private" in allowed
                or "car" in allowed
            ):
                return True

        return False

    except traci.TraCIException:
        return False


def get_valid_passenger_edges():
    edges = []

    for edge_id in traci.edge.getIDList():
        if edge_id.startswith(":"):
            continue

        if edge_allows_passenger(edge_id):
            edges.append(edge_id)

    return sorted(edges)


def normalize_type(edge_type):
    if edge_type is None:
        return ""

    return str(edge_type).lower()


def build_edge_metadata(valid_edges):
    metadata = {}

    for edge_id in valid_edges:
        metadata[edge_id] = {
            "type": "",
            "speed": 0.0,
            "lanes": 1,
            "priority": 0,
            "category": "unknown",
            "base_weight": 1.0,
        }

    if sumolib is not None:
        try:
            net = sumolib.net.readNet(NET_FILE)

            for edge in net.getEdges():
                edge_id = edge.getID()

                if edge_id not in metadata:
                    continue

                try:
                    edge_type = normalize_type(edge.getType())
                except Exception:
                    edge_type = ""

                try:
                    speed = float(edge.getSpeed())
                except Exception:
                    speed = 0.0

                try:
                    lanes = len(edge.getLanes())
                except Exception:
                    lanes = 1

                try:
                    priority = int(edge.getPriority())
                except Exception:
                    priority = 0

                metadata[edge_id].update(
                    {
                        "type": edge_type,
                        "speed": speed,
                        "lanes": lanes,
                        "priority": priority,
                    }
                )

        except Exception:
            print("Warning: sumolib could not read network metadata. Using TraCI fallback.")

    for edge_id in valid_edges:
        item = metadata[edge_id]

        if item["speed"] <= 0.0:
            try:
                lane_count = traci.edge.getLaneNumber(edge_id)
                item["lanes"] = max(1, lane_count)

                speeds = []

                for lane_index in range(lane_count):
                    lane_id = f"{edge_id}_{lane_index}"
                    speeds.append(traci.lane.getMaxSpeed(lane_id))

                if speeds:
                    item["speed"] = max(speeds)

            except traci.TraCIException:
                pass

        road_type = item["type"]
        speed = item["speed"]
        lanes = item["lanes"]
        priority = item["priority"]

        main_keywords = (
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "arterial", "collector"
        )

        local_keywords = (
            "residential", "living_street", "service", "parking",
            "driveway", "track", "alley"
        )

        if any(keyword in road_type for keyword in main_keywords):
            category = "main"
        elif any(keyword in road_type for keyword in local_keywords):
            category = "local"
        elif lanes >= 2 or speed >= 12.0 or priority >= 6:
            category = "main"
        elif speed <= 8.5 and lanes <= 1 and priority <= 3:
            category = "local"
        else:
            category = "connector"

        if category == "main":
            base_weight = 12.0 + 2.0 * lanes + 0.35 * speed + 0.50 * priority
        elif category == "connector":
            base_weight = 3.0 + 0.15 * speed + 0.20 * priority
        else:
            base_weight = 0.20 + 0.02 * speed

        item["category"] = category
        item["base_weight"] = max(0.01, base_weight)

    counts = Counter(item["category"] for item in metadata.values())

    print()
    print("Road classification:")
    print(f"  main roads:      {counts['main']}")
    print(f"  connector roads: {counts['connector']}")
    print(f"  local roads:     {counts['local']}")

    return metadata


def edge_category(edge_id, edge_metadata):
    return edge_metadata.get(edge_id, {}).get("category", "unknown")


def edge_base_weight(edge_id, edge_metadata):
    return edge_metadata.get(edge_id, {}).get("base_weight", 1.0)


def successor_weight(
    current_edge,
    next_edge,
    previous_edge,
    edge_metadata,
    core_edges,
    args,
):
    weight = edge_base_weight(next_edge, edge_metadata)

    current_category = edge_category(current_edge, edge_metadata)
    next_category = edge_category(next_edge, edge_metadata)

    if next_edge == previous_edge:
        weight *= 0.02

    if next_edge not in core_edges:
        weight *= args.non_core_penalty

    if current_category == "main" and next_category == "local":
        weight *= args.local_road_penalty

    if current_category == "connector" and next_category == "local":
        weight *= max(args.local_road_penalty, 0.15)

    if current_category == "local" and next_category == "main":
        weight *= args.leave_local_bonus

    if current_category == "local" and next_category == "local":
        weight *= args.local_to_local_penalty

    return max(weight, 0.001)


# ============================================================
# Route file
# ============================================================

def write_empty_route_file(route_file):
    lines = []
    lines.append("<routes>")

    lines.append(
        f'''    <vType id="global_car"
           vClass="passenger"
           guiShape="passenger"
           length="{CAR_LENGTH}"
           width="{CAR_WIDTH}"
           minGap="{CAR_MIN_GAP}"
           accel="2.6"
           decel="4.5"
           emergencyDecel="9.0"
           maxSpeed="13.9"
           sigma="0.5"
           tau="1.0"
           lcStrategic="35.0"
           lcCooperative="1.0"
           lcSpeedGain="0.03"
           lcKeepRight="0.15"
           lcAssertive="0.45"
           jmIgnoreKeepClearTime="-1"
           jmDriveAfterYellowTime="-1"
           jmDriveAfterRedTime="-1"/>'''
    )

    lines.append("</routes>")

    with open(route_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"Wrote dynamic vehicle-type route file: {route_file}")


# ============================================================
# Geometry / movement classification
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


def classify_movement_by_geometry(in_lane, out_lane):
    try:
        in_vec = lane_direction_vector(in_lane, incoming=True)
        out_vec = lane_direction_vector(out_lane, incoming=False)

        if in_vec is None or out_vec is None:
            return None, None

        approach = classify_approach(in_vec)
        angle = signed_turn_angle(in_vec, out_vec)

        if abs(angle) <= 50:
            movement = "S"
        elif angle > 50:
            movement = "L"
        else:
            movement = "R"

        return approach, movement

    except traci.TraCIException:
        return None, None


def sumo_link_direction_to_movement(direction_code):
    if not direction_code:
        return None

    code = str(direction_code).lower()

    if code.startswith("s"):
        return "S"

    if code.startswith("r"):
        return "R"

    if code.startswith("l"):
        return "L"

    # SUMO uses t for turn-around / U-turn. Do not count it as normal traffic.
    if code.startswith("t"):
        return None

    return None


def get_sumo_link_direction(in_lane, out_lane):
    """
    Prefer SUMO's own lane-link direction over geometric guessing.

    This is the important fix for the previous '0 straight' problem.
    """
    try:
        try:
            links = traci.lane.getLinks(in_lane, extended=True)
        except TypeError:
            links = traci.lane.getLinks(in_lane)

        for link in links:
            if not link:
                continue

            to_lane = link[0]

            if to_lane != out_lane:
                continue

            # Direction is usually a one-character string such as s, r, l, or t.
            for value in link[1:]:
                if isinstance(value, str):
                    movement = sumo_link_direction_to_movement(value)

                    if movement is not None:
                        return movement

        return None

    except traci.TraCIException:
        return None


def classify_connection(in_lane, out_lane):
    movement = get_sumo_link_direction(in_lane, out_lane)

    try:
        in_vec = lane_direction_vector(in_lane, incoming=True)
        approach = classify_approach(in_vec) if in_vec is not None else None
    except traci.TraCIException:
        approach = None

    if movement is None:
        approach2, movement2 = classify_movement_by_geometry(in_lane, out_lane)

        if approach is None:
            approach = approach2

        movement = movement2

    if approach is None or movement is None:
        return None

    return f"{approach}-{movement}"


# ============================================================
# Successor graph
# ============================================================

def build_raw_successor_graph(valid_edges):
    valid_edge_set = set(valid_edges)
    graph = {edge_id: set() for edge_id in valid_edges}

    for edge_id in valid_edges:
        try:
            lane_count = traci.edge.getLaneNumber(edge_id)
        except traci.TraCIException:
            continue

        for lane_index in range(lane_count):
            lane_id = f"{edge_id}_{lane_index}"

            try:
                try:
                    links = traci.lane.getLinks(lane_id, extended=True)
                except TypeError:
                    links = traci.lane.getLinks(lane_id)
            except traci.TraCIException:
                continue

            for link in links:
                if not link:
                    continue

                to_lane = link[0]
                to_edge = lane_to_edge(to_lane)

                if to_edge is None:
                    continue

                if to_edge not in valid_edge_set:
                    continue

                if to_edge == edge_id:
                    continue

                graph[edge_id].add(to_edge)

    return {
        edge_id: sorted(successors)
        for edge_id, successors in graph.items()
        if successors
    }


def strongly_connected_components(graph):
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlinks = {}
    components = []

    def strongconnect(node):
        nonlocal index

        indices[node] = index
        lowlinks[node] = index
        index += 1

        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component = []

            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)

                if item == node:
                    break

            components.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return components


def largest_loop_core(raw_graph):
    components = strongly_connected_components(raw_graph)

    cyclic_components = []

    for component in components:
        if len(component) > 1:
            cyclic_components.append(component)
        elif component and component[0] in raw_graph.get(component[0], []):
            cyclic_components.append(component)

    if not cyclic_components:
        return set(raw_graph)

    largest = max(cyclic_components, key=len)
    return set(largest)


# ============================================================
# Traffic-light movement map
# ============================================================

def classify_tls_movements(tls_id):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    state_length = len(traci.trafficlight.getRedYellowGreenState(tls_id))

    movement_map = {label: {} for label in MOVEMENT_LABELS}

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

            label = classify_connection(incoming_lane, outgoing_lane)

            if label not in movement_map:
                continue

            if signal_index not in movement_map[label]:
                movement_map[label][signal_index] = {
                    "in": set(),
                    "out": set(),
                }

            movement_map[label][signal_index]["in"].add(incoming_lane)
            movement_map[label][signal_index]["out"].add(outgoing_lane)

    return state_length, movement_map


def build_safe_phase_plan(movement_map):
    phases = []

    for phase_def in PHASE_DEFS:
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

            label = classify_connection(link[0], link[1])

            if label is not None:
                result[signal_index].add(label)

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
# Anti-gridlock signal state
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
    return all(outgoing_lane_has_space(lane_id) for lane_id in out_lanes)


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
# Fixed-cycle controller
# ============================================================

def build_controller_for_tls(tls_id, rng, activate=True):
    try:
        state_length, movement_map = classify_tls_movements(tls_id)
        phases = build_safe_phase_plan(movement_map)
    except traci.TraCIException:
        return None

    if len(phases) < 2:
        return None

    movement_in_lanes_cache, movement_out_lanes_cache, all_in_lanes = build_lane_caches(
        movement_map
    )

    controller = {
        "tls_id": tls_id,
        "state_length": state_length,
        "movement_map": movement_map,
        "movement_in_lanes_cache": movement_in_lanes_cache,
        "movement_out_lanes_cache": movement_out_lanes_cache,
        "all_in_lanes": all_in_lanes,
        "phases": phases,
        "phase_pos": 0,
        "mode": "green",
        "remaining": 0.0,
        "phase_elapsed": 0.0,
        "last_active_indices": set(),
        "next_phase_pos": None,
        "disabled": False,
        "green_duration": 30.0,
    }

    safe, _ = verify_controller_safety(tls_id, controller)

    if not safe:
        return None

    controller["phase_pos"] = rng.randrange(len(phases))

    if activate:
        start_green(controller, controller["phase_pos"])

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


def switch_next_fixed_phase(controller):
    if controller.get("disabled"):
        return False

    if controller["mode"] != "green":
        return False

    next_pos = (controller["phase_pos"] + 1) % len(controller["phases"])
    return request_switch(controller, next_pos)


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
                    next_pos = (controller["phase_pos"] + 1) % len(controller["phases"])

                start_green(controller, next_pos)

    except traci.TraCIException:
        controller["disabled"] = True


def build_all_fixed_controllers(rng, args):
    controllers = []
    skipped = []

    for tls_id in traci.trafficlight.getIDList():
        controller = build_controller_for_tls(tls_id, rng=rng, activate=True)

        if controller is None:
            skipped.append(tls_id)
            continue

        jitter = rng.uniform(-args.signal_timing_jitter, args.signal_timing_jitter)
        controller["green_duration"] = max(15.0, args.green_duration * (1.0 + jitter))
        controller["phase_elapsed"] = rng.uniform(0.0, controller["green_duration"])

        controllers.append(controller)

    return controllers, skipped


# ============================================================
# Dynamic turn decision index
# ============================================================

def build_turn_decision_index(controllers, raw_graph):
    """
    incoming_edge -> tls_id -> movement_group -> options

    Uses the full drivable graph, not only the loop-safe core.
    This prevents straight movements from being filtered out.
    """
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    allowed_edges = set(raw_graph)

    for controller in controllers:
        tls_id = controller["tls_id"]

        for label, signal_data in controller["movement_map"].items():
            if "-" not in label:
                continue

            movement_group = label.split("-")[-1]

            if movement_group not in TURN_PROBABILITIES:
                continue

            for lane_sets in signal_data.values():
                for in_lane in lane_sets["in"]:
                    incoming_edge = lane_to_edge(in_lane)

                    if incoming_edge is None or incoming_edge not in allowed_edges:
                        continue

                    for out_lane in lane_sets["out"]:
                        outgoing_edge = lane_to_edge(out_lane)

                        if outgoing_edge is None:
                            continue

                        if outgoing_edge not in allowed_edges:
                            continue

                        if incoming_edge == outgoing_edge:
                            continue

                        index[incoming_edge][tls_id][movement_group].append({
                            "label": label,
                            "movement": movement_group,
                            "incoming_edge": incoming_edge,
                            "outgoing_edge": outgoing_edge,
                        })

    total_edges = len(index)
    movement_option_counts = Counter()
    total_options = 0

    for tls_map in index.values():
        for movement_map in tls_map.values():
            for movement, options in movement_map.items():
                movement_option_counts[movement] += len(options)
                total_options += len(options)

    print()
    print("Dynamic turn-decision index:")
    print(f"  incoming edges with decisions: {total_edges}")
    print(f"  total movement options:        {total_options}")
    print(f"  straight options:              {movement_option_counts['S']}")
    print(f"  right options:                 {movement_option_counts['R']}")
    print(f"  left options:                  {movement_option_counts['L']}")

    if movement_option_counts["S"] == 0:
        print()
        print("WARNING: straight options are still zero.")
        print("That means SUMO did not expose straight lane links for the controlled TLSs.")

    return index


def choose_turn_group(rng, available_groups, turn_counts, strict_split=True):
    groups = [
        group
        for group in MOVEMENT_ORDER
        if group in available_groups and available_groups[group]
    ]

    if not groups:
        return None

    if strict_split:
        total_after = sum(turn_counts.values()) + 1
        deficits = {}

        for group in groups:
            target_count = TURN_PROBABILITIES[group] * total_after
            deficits[group] = max(0.0, target_count - turn_counts[group])

        deficit_total = sum(deficits.values())

        if deficit_total > 0:
            r = rng.random() * deficit_total
            cumulative = 0.0

            for group in groups:
                cumulative += deficits[group]

                if r <= cumulative:
                    return group

    total_weight = sum(TURN_PROBABILITIES[group] for group in groups)
    r = rng.random() * total_weight
    cumulative = 0.0

    for group in groups:
        cumulative += TURN_PROBABILITIES[group]

        if r <= cumulative:
            return group

    return groups[-1]


# ============================================================
# Random-walk route generation
# ============================================================

def choose_uncontrolled_successor(
    current_edge,
    previous_edge,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    args,
):
    successors = raw_graph.get(current_edge, [])

    if not successors:
        return None

    weights = [
        successor_weight(
            current_edge=current_edge,
            next_edge=successor,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        )
        for successor in successors
    ]

    return weighted_choice(rng, successors, weights)


def choose_next_edge(
    current_edge,
    previous_edge,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    turn_counts,
    args,
):
    if current_edge in turn_index:
        tls_map = turn_index[current_edge]

        for tls_id in sorted(tls_map.keys()):
            options_by_group = tls_map[tls_id]

            available = {
                group: options
                for group, options in options_by_group.items()
                if options
            }

            group = choose_turn_group(
                rng=rng,
                available_groups=available,
                turn_counts=turn_counts,
                strict_split=not args.disable_strict_split,
            )

            if group is None:
                continue

            options = list(available[group])
            weighted_options = []
            weights = []

            for option in options:
                outgoing_edge = option["outgoing_edge"]

                if outgoing_edge == previous_edge:
                    continue

                if outgoing_edge not in raw_graph:
                    continue

                weighted_options.append(option)
                weights.append(
                    successor_weight(
                        current_edge=current_edge,
                        next_edge=outgoing_edge,
                        previous_edge=previous_edge,
                        edge_metadata=edge_metadata,
                        core_edges=core_edges,
                        args=args,
                    )
                )

            if not weighted_options:
                continue

            option = weighted_choice(rng, weighted_options, weights)
            turn_counts[group] += 1
            return option["outgoing_edge"]

    return choose_uncontrolled_successor(
        current_edge=current_edge,
        previous_edge=previous_edge,
        raw_graph=raw_graph,
        edge_metadata=edge_metadata,
        core_edges=core_edges,
        rng=rng,
        args=args,
    )


def build_random_walk_route(
    start_edge,
    lookahead_edges,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    turn_counts,
    args,
    previous_edge=None,
):
    route = [start_edge]
    current_edge = start_edge

    for _ in range(lookahead_edges - 1):
        next_edge = choose_next_edge(
            current_edge=current_edge,
            previous_edge=previous_edge,
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
        )

        if next_edge is None:
            break

        if next_edge == current_edge:
            break

        route.append(next_edge)
        previous_edge = current_edge
        current_edge = next_edge

    return route


def recover_vehicle_route(
    veh_id,
    current_edge,
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    args,
):
    core_list = list(core_edges)

    for _ in range(args.recovery_attempts):
        target_edge = weighted_choice(
            rng,
            core_list,
            [edge_base_weight(edge_id, edge_metadata) for edge_id in core_list],
        )

        if target_edge == current_edge:
            continue

        try:
            path = traci.simulation.findRoute(current_edge, target_edge)

            if path is None or not path.edges:
                continue

            path_edges = list(path.edges)

            if path_edges[0] != current_edge:
                continue

            if path_edges[-1] != target_edge:
                continue

            if len(path_edges) < 2:
                continue

            extension = build_random_walk_route(
                start_edge=target_edge,
                lookahead_edges=args.route_lookahead_edges,
                turn_index=turn_index,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                rng=rng,
                turn_counts=turn_counts,
                args=args,
            )

            if len(extension) < 2:
                continue

            new_route = path_edges + extension[1:]

            if len(new_route) < 2:
                continue

            traci.vehicle.setRoute(veh_id, new_route)
            return True

        except traci.TraCIException:
            continue

    return False


def extend_vehicle_route(
    veh_id,
    min_remaining_edges,
    lookahead_edges,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    turn_counts,
    args,
):
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        current_edge = lane_to_edge(lane_id)

        if current_edge is None:
            return False, False

        old_route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)

        if route_index < 0 or route_index >= len(old_route):
            remaining = [current_edge]
        else:
            remaining = old_route[route_index:]

            if not remaining or remaining[0] != current_edge:
                if current_edge in remaining:
                    position = remaining.index(current_edge)
                    remaining = remaining[position:]
                else:
                    remaining = [current_edge]

        if len(remaining) >= min_remaining_edges:
            return False, False

        last_edge = remaining[-1]
        previous_edge = remaining[-2] if len(remaining) >= 2 else None

        if last_edge not in raw_graph:
            recovered = recover_vehicle_route(
                veh_id=veh_id,
                current_edge=current_edge,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                turn_index=turn_index,
                rng=rng,
                turn_counts=turn_counts,
                args=args,
            )

            return recovered, recovered

        extension = build_random_walk_route(
            start_edge=last_edge,
            lookahead_edges=lookahead_edges,
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
            previous_edge=previous_edge,
        )

        if len(extension) < 2:
            recovered = recover_vehicle_route(
                veh_id=veh_id,
                current_edge=current_edge,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                turn_index=turn_index,
                rng=rng,
                turn_counts=turn_counts,
                args=args,
            )

            return recovered, recovered

        new_route = remaining + extension[1:]

        if len(new_route) < 2:
            return False, False

        traci.vehicle.setRoute(veh_id, new_route)
        return True, False

    except traci.TraCIException:
        return False, False


# ============================================================
# Vehicle spawning
# ============================================================

def spawn_vehicle(
    sim_state,
    start_edges,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    turn_counts,
    args,
):
    for _ in range(args.spawn_attempts):
        start_edge = weighted_choice(
            rng,
            start_edges,
            [edge_base_weight(edge_id, edge_metadata) for edge_id in start_edges],
        )

        route_edges = build_random_walk_route(
            start_edge=start_edge,
            lookahead_edges=args.route_lookahead_edges,
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
        )

        if len(route_edges) < 2:
            continue

        veh_id = f"random_car_{sim_state['next_vehicle_id']}"
        route_id = f"random_route_{sim_state['next_route_id']}"

        sim_state["next_vehicle_id"] += 1
        sim_state["next_route_id"] += 1

        try:
            traci.route.add(route_id, route_edges)

            traci.vehicle.add(
                vehID=veh_id,
                routeID=route_id,
                typeID="global_car",
                depart=str(traci.simulation.getTime()),
                departLane="best",
                departPos="random_free",
                departSpeed="max",
            )

            return True

        except traci.TraCIException:
            continue

    return False


def fill_vehicle_population(
    sim_state,
    target_count,
    max_to_spawn,
    start_edges,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    turn_counts,
    args,
):
    active = traci.vehicle.getIDCount()
    need = max(0, target_count - active)
    spawn_count = min(need, max_to_spawn)

    spawned = 0

    for _ in range(spawn_count):
        ok = spawn_vehicle(
            sim_state=sim_state,
            start_edges=start_edges,
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
        )

        if ok:
            spawned += 1

    return spawned


# ============================================================
# Metrics
# ============================================================

def total_controlled_wait_and_queue(controller):
    veh_ids = set()

    for lane_id in controller["all_in_lanes"]:
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

    return wait, queue


def count_active_vehicle_road_categories(edge_metadata):
    counts = Counter()

    for veh_id in traci.vehicle.getIDList():
        try:
            lane_id = traci.vehicle.getLaneID(veh_id)
        except traci.TraCIException:
            continue

        edge_id = lane_to_edge(lane_id)

        if edge_id is None:
            continue

        counts[edge_category(edge_id, edge_metadata)] += 1

    return counts


def print_turn_summary(turn_counts):
    total = turn_counts["S"] + turn_counts["R"] + turn_counts["L"]

    print()
    print("Dynamic planned turn-decision summary")
    print("=" * 76)

    for group, name in [("S", "Straight"), ("R", "Right"), ("L", "Left")]:
        count = turn_counts[group]
        actual = 100.0 * count / total if total else 0.0
        target = 100.0 * TURN_PROBABILITIES[group]

        print(
            f"{name:8}: {count:8d} decisions   "
            f"actual={actual:6.2f}%   target={target:6.2f}%"
        )

    print("=" * 76)


# ============================================================
# Simulation loop
# ============================================================

def run_simulation_steps(
    num_steps,
    controllers,
    start_edges,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    turn_counts,
    sim_state,
    args,
):
    arrived = 0
    spawned_total = 0
    extended_total = 0
    recovered_total = 0

    for _ in range(num_steps):
        if traci.vehicle.getIDCount() < args.target_vehicles:
            spawned_total += fill_vehicle_population(
                sim_state=sim_state,
                target_count=args.target_vehicles,
                max_to_spawn=args.spawn_batch,
                start_edges=start_edges,
                turn_index=turn_index,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                rng=rng,
                turn_counts=turn_counts,
                args=args,
            )

        for veh_id in list(traci.vehicle.getIDList()):
            extended, recovered = extend_vehicle_route(
                veh_id=veh_id,
                min_remaining_edges=args.min_remaining_edges,
                lookahead_edges=args.route_lookahead_edges,
                turn_index=turn_index,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                rng=rng,
                turn_counts=turn_counts,
                args=args,
            )

            if extended:
                extended_total += 1

            if recovered:
                recovered_total += 1

        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()

        for controller in controllers:
            if not controller.get("disabled"):
                update_controller_after_simstep(controller)

    return arrived, spawned_total, extended_total, recovered_total


def run_simulation(args):
    rng = random.Random(args.seed)

    binary = SUMO_GUI_BINARY if args.gui else SUMO_HEADLESS_BINARY

    if args.gui:
        ensure_xquartz()

    sim_end = args.end if args.end is not None else DEFAULT_SIM_END

    sumo_cmd = [
        binary,
        "-n", NET_FILE,
        "-r", args.route_file,
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(sim_end),
        "--max-num-vehicles", str(args.max_vehicles),
        "--max-depart-delay", str(args.max_depart_delay),
        "--time-to-teleport", str(args.time_to_teleport),
        "--ignore-route-errors", "true",
        "--quit-on-end", "false",
        *QUIET_SUMO_ARGS,
    ]

    print()
    print("Starting realistic random-driving fixed-cycle simulation:")
    print(" ".join(sumo_cmd))
    print()

    traci.start(sumo_cmd)

    turn_counts = Counter()

    try:
        print("TraCI connected. Building controllers, metadata, and routing graph...")

        controllers, skipped = build_all_fixed_controllers(rng=rng, args=args)

        valid_edges = get_valid_passenger_edges()
        edge_metadata = build_edge_metadata(valid_edges)
        raw_graph = build_raw_successor_graph(valid_edges)
        core_edges = largest_loop_core(raw_graph)

        print()
        print("Passenger successor graph:")
        print(f"  raw drivable edges with outgoing choices: {len(raw_graph)}")
        print(f"  largest loop-safe core edges:             {len(core_edges)}")
        print(f"  total raw outgoing choices:               {sum(len(v) for v in raw_graph.values())}")

        main_start_edges = [
            edge_id
            for edge_id in core_edges
            if edge_id in raw_graph
            and edge_category(edge_id, edge_metadata) in {"main", "connector"}
        ]

        if len(main_start_edges) < 10:
            main_start_edges = [
                edge_id
                for edge_id in core_edges
                if edge_id in raw_graph
            ]

        if not main_start_edges:
            raise RuntimeError("No valid start edges were found.")

        turn_index = build_turn_decision_index(
            controllers=controllers,
            raw_graph=raw_graph,
        )

        sim_state = {
            "next_vehicle_id": 0,
            "next_route_id": 0,
        }

        initial_spawned = fill_vehicle_population(
            sim_state=sim_state,
            target_count=args.initial_vehicles,
            max_to_spawn=args.initial_vehicles,
            start_edges=main_start_edges,
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
        )

        print()
        print(f"Controlled traffic lights: {len(controllers)}")
        print(f"Skipped traffic lights:    {len(skipped)}")
        print(f"Main/connector start edges:{len(main_start_edges)}")
        print(f"Initially spawned cars:    {initial_spawned}")
        print()
        print("Fixed phase rule:")
        print("  1. N/S protected lefts")
        print("  2. N/S straights")
        print("  3. E/W protected lefts")
        print("  4. E/W straights")
        print()
        print("Dynamic movement target:")
        print("  Straight: 70.0%")
        print("  Right:    17.5%")
        print("  Left:     12.5%")
        print()
        print("Straight-movement fix:")
        print("  Uses SUMO lane-link directions when available.")
        print("  Turn options are built from the full drivable graph, not only the loop-safe core.")
        print()
        print("Realism fixes:")
        print("  Main roads are preferred over residential/service roads.")
        print("  Residential-to-residential cruising is penalized.")
        print("  Cars are recovered away from route endings/dead ends.")
        print("  Signal phases have randomized offsets and slight timing variation.")
        print("  Routes are planned far ahead to reduce last-second lane changes.")
        print()
        print(f"Active vehicle cap: {args.max_vehicles}")
        print(f"Target active cars: {args.target_vehicles}")
        print()
        print("Vehicle scale:")
        print(f"  length = {CAR_LENGTH} m")
        print(f"  width  = {CAR_WIDTH} m")
        print(f"  minGap = {CAR_MIN_GAP} m")
        print()

        decision_steps = max(1, int(round(DECISION_INTERVAL / STEP_LENGTH)))
        next_print_time = 0.0
        total_arrived = 0

        while True:
            sim_time = traci.simulation.getTime()

            if sim_time >= sim_end:
                print(f"Reached simulation end time: t={sim_time:.1f}s")
                break

            for controller in controllers:
                if (
                    not controller.get("disabled")
                    and controller["mode"] == "green"
                    and controller["phase_elapsed"] >= controller["green_duration"]
                ):
                    switch_next_fixed_phase(controller)

            arrived, spawned, extended, recovered = run_simulation_steps(
                num_steps=decision_steps,
                controllers=controllers,
                start_edges=main_start_edges,
                turn_index=turn_index,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                rng=rng,
                turn_counts=turn_counts,
                sim_state=sim_state,
                args=args,
            )

            total_arrived += arrived
            sim_time = traci.simulation.getTime()

            if sim_time >= next_print_time:
                total_queue = 0.0
                total_wait = 0.0

                for controller in controllers:
                    if controller.get("disabled"):
                        continue

                    wait, queue = total_controlled_wait_and_queue(controller)
                    total_wait += wait
                    total_queue += queue

                road_counts = count_active_vehicle_road_categories(edge_metadata)
                active = traci.vehicle.getIDCount()

                local_pct = 100.0 * road_counts["local"] / active if active else 0.0
                main_pct = 100.0 * road_counts["main"] / active if active else 0.0
                connector_pct = 100.0 * road_counts["connector"] / active if active else 0.0

                print(
                    f"t={sim_time:8.1f}, "
                    f"active={active:5d}, "
                    f"expected={traci.simulation.getMinExpectedNumber():5d}, "
                    f"controlled_tls={len(controllers):3d}, "
                    f"queue={total_queue:.0f}, "
                    f"wait={total_wait:.1f}, "
                    f"arrived={arrived}, "
                    f"total_arrived={total_arrived}, "
                    f"spawned={spawned}, "
                    f"extended={extended}, "
                    f"recovered={recovered}"
                )

                print(
                    f"road_mix: main={main_pct:5.1f}%, "
                    f"connector={connector_pct:5.1f}%, "
                    f"local={local_pct:5.1f}%"
                )

                print_turn_summary(turn_counts)

                next_print_time += args.print_every

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")

    except Exception:
        print()
        print("Python error during simulation:")
        traceback.print_exc()

    finally:
        print_turn_summary(turn_counts)

        try:
            traci.close()
        except Exception:
            pass

        print()
        print("Simulation ended.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--route-file", default=ROUTE_FILE)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=MAX_ACTIVE_VEHICLE_CAP,
        help="Hard-capped at 1500.",
    )

    parser.add_argument(
        "--target-vehicles",
        type=int,
        default=1250,
        help="The script tries to keep about this many active cars.",
    )

    parser.add_argument(
        "--initial-vehicles",
        type=int,
        default=400,
        help="Cars spawned immediately at the start.",
    )

    parser.add_argument(
        "--spawn-batch",
        type=int,
        default=25,
        help="Maximum cars to add per simulation step when below target.",
    )

    parser.add_argument(
        "--spawn-attempts",
        type=int,
        default=40,
        help="Attempts to find a valid random route for each spawned car.",
    )

    parser.add_argument(
        "--route-lookahead-edges",
        type=int,
        default=120,
        help="How many edges ahead each random-walk route is planned.",
    )

    parser.add_argument(
        "--min-remaining-edges",
        type=int,
        default=50,
        help="Extend a car's route when fewer than this many edges remain.",
    )

    parser.add_argument(
        "--recovery-attempts",
        type=int,
        default=40,
        help="Attempts to recover a vehicle back into the loop-safe driving core.",
    )

    parser.add_argument(
        "--local-road-penalty",
        type=float,
        default=0.04,
        help="Penalty for choosing local/residential roads from larger roads.",
    )

    parser.add_argument(
        "--local-to-local-penalty",
        type=float,
        default=0.15,
        help="Penalty for continuing from one local road to another local road.",
    )

    parser.add_argument(
        "--leave-local-bonus",
        type=float,
        default=8.0,
        help="Bonus for leaving a local road and returning to a main road.",
    )

    parser.add_argument(
        "--non-core-penalty",
        type=float,
        default=0.20,
        help="Penalty for leaving the loop-safe driving core.",
    )

    parser.add_argument(
        "--signal-timing-jitter",
        type=float,
        default=0.15,
        help="Per-intersection green-duration variation fraction.",
    )

    parser.add_argument("--max-depart-delay", type=int, default=300)

    parser.add_argument(
        "--time-to-teleport",
        type=int,
        default=-1,
        help="Default -1 disables teleporting so cars do not disappear.",
    )

    parser.add_argument("--green-duration", type=float, default=30.0)
    parser.add_argument("--print-every", type=float, default=30.0)

    parser.add_argument(
        "--disable-strict-split",
        action="store_true",
        help=(
            "Use raw 70/17.5/12.5 probabilities without quota correction. "
            "By default, quota correction keeps the observed split closer to target."
        ),
    )

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--run-existing", action="store_true")

    args = parser.parse_args()

    if args.max_vehicles > MAX_ACTIVE_VEHICLE_CAP:
        print(
            f"WARNING: requested --max-vehicles {args.max_vehicles}, "
            f"but the hard cap is {MAX_ACTIVE_VEHICLE_CAP}."
        )
        args.max_vehicles = MAX_ACTIVE_VEHICLE_CAP

    if args.target_vehicles > args.max_vehicles:
        print(
            f"WARNING: requested --target-vehicles {args.target_vehicles}, "
            f"but max vehicles is {args.max_vehicles}. Lowering target."
        )
        args.target_vehicles = args.max_vehicles

    if args.initial_vehicles > args.target_vehicles:
        args.initial_vehicles = args.target_vehicles

    if args.min_remaining_edges >= args.route_lookahead_edges:
        args.min_remaining_edges = max(2, args.route_lookahead_edges // 2)

    if not args.run_existing:
        write_empty_route_file(args.route_file)

    if args.generate_only:
        return

    run_simulation(args)


if __name__ == "__main__":
    main()