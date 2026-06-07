import argparse
import math
import os
import subprocess
import sys
import time

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

# Optional PROJ database setup, prevents proj.db warning spam if available.
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


# ============================================================
# File setup
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
ROUTE_FILE = f"{BACKGROUND_ROUTE_FILE},{AMBULANCE_ROUTE_FILE}"

TARGET_TLS_ID = "cluster_12179861947_12179861948_12179861949_12185616643_#11more"

STEP_LENGTH = 0.5
DEFAULT_END_TIME = 7200

MAX_NUM_VEHICLES = 2000
MAX_DEPART_DELAY = 60
TIME_TO_TELEPORT = 300

REQUIRED_EXIT_GAP = 14.0

QUIET_SUMO_ARGS = [
    "--no-warnings", "true",
    "--no-step-log", "true",
]


# ============================================================
# XQuartz helper
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
    """
    Classifies incoming travel direction into NB/SB/EB/WB.
    The vector points toward the intersection.
    """
    dx, dy = in_vec

    if abs(dy) >= abs(dx):
        return "NB" if dy > 0 else "SB"

    return "EB" if dx > 0 else "WB"


def classify_movement(angle):
    """
    Returns:
        S = straight
        L = left turn
        R = right turn
    """
    if abs(angle) <= 35:
        return "S"

    if angle > 35:
        return "L"

    return "R"


# ============================================================
# Exit-space helpers
# ============================================================

def outgoing_lane_has_space(out_lane_id):
    """
    Checks whether there is enough room immediately after the intersection.

    If not, the movement is temporarily held red to avoid blocking the box.
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


# ============================================================
# Traffic-light link analysis
# ============================================================

def get_signal_links(tls_id):
    """
    Returns one entry per controlled connection.

    SUMO traffic-light states are indexed by signal index.
    One signal index may control multiple lane-to-lane connections.
    """
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    state_length = len(traci.trafficlight.getRedYellowGreenState(tls_id))

    entries = []

    for signal_index, signal_links in enumerate(controlled_links):
        if signal_index >= state_length:
            continue

        for link in signal_links:
            if len(link) < 2:
                continue

            incoming_lane = link[0]
            outgoing_lane = link[1]
            internal_lane = link[2] if len(link) >= 3 else ""

            if not incoming_lane or not outgoing_lane:
                continue

            try:
                in_vec = lane_direction_vector(incoming_lane, incoming=True)
                out_vec = lane_direction_vector(outgoing_lane, incoming=False)

                if in_vec is None or out_vec is None:
                    continue

                angle = signed_turn_angle(in_vec, out_vec)
                approach = classify_approach(in_vec)
                movement = classify_movement(angle)
                label = f"{approach}-{movement}"

                entries.append({
                    "signal_index": signal_index,
                    "label": label,
                    "approach": approach,
                    "movement": movement,
                    "incoming_lane": incoming_lane,
                    "outgoing_lane": outgoing_lane,
                    "internal_lane": internal_lane,
                    "angle": angle,
                })

            except traci.TraCIException:
                continue

    return entries


def print_signal_table(entries):
    print("\nControlled signal-link table:")
    print(
        f"{'idx':>4}  {'move':>5}  {'angle':>8}  "
        f"{'incoming lane':<35}  {'outgoing lane':<35}"
    )
    print("-" * 100)

    for entry in entries:
        print(
            f"{entry['signal_index']:>4}  "
            f"{entry['label']:>5}  "
            f"{entry['angle']:>8.1f}  "
            f"{entry['incoming_lane']:<35}  "
            f"{entry['outgoing_lane']:<35}"
        )


def entries_by_signal_index(entries):
    grouped = {}

    for entry in entries:
        grouped.setdefault(entry["signal_index"], []).append(entry)

    return grouped


def outgoing_lanes_for_entries(entries):
    return {
        entry["outgoing_lane"]
        for entry in entries
        if entry["outgoing_lane"]
    }


def axis_from_movement(movement_label):
    """
    Converts movement labels into road axis.

    NB-S, SB-S, NB-L, SB-L -> NS
    EB-S, WB-S, EB-L, WB-L -> EW
    """
    approach = movement_label.split("-")[0]

    if approach in ("NB", "SB"):
        return "NS"

    if approach in ("EB", "WB"):
        return "EW"

    raise RuntimeError(f"Could not infer axis from movement {movement_label}")


def movement_code_from_turn(turn):
    if turn == "straight":
        return "S"

    if turn == "left":
        return "L"

    raise RuntimeError(f"Unknown turn type: {turn}")


def movement_labels_for_axis_and_turn(axis, turn):
    code = movement_code_from_turn(turn)

    if axis == "NS":
        return {f"NB-{code}", f"SB-{code}"}

    if axis == "EW":
        return {f"EB-{code}", f"WB-{code}"}

    raise RuntimeError(f"Unknown axis: {axis}")


def choose_axis(entries, requested_axis=None, movement_label=None, explicit_index=None):
    """
    Determines whether to force N/S pair or E/W pair.

    Priority:
    1. --axis NS/EW
    2. --signal-index, infer its movement label and axis
    3. --movement, infer axis
    4. fallback to first available straight/left movement
    """
    if requested_axis is not None:
        requested_axis = requested_axis.upper()

        if requested_axis not in ("NS", "EW"):
            raise RuntimeError("--axis must be either NS or EW")

        return requested_axis

    if explicit_index is not None:
        matches = [
            entry for entry in entries
            if entry["signal_index"] == explicit_index
        ]

        if not matches:
            raise RuntimeError(
                f"Signal index {explicit_index} was not found for this TLS."
            )

        inferred_label = matches[0]["label"]
        inferred_axis = axis_from_movement(inferred_label)

        print(
            f"\nSignal index {explicit_index} was selected. "
            f"It corresponds to movement {inferred_label}, so using axis {inferred_axis}."
        )

        return inferred_axis

    if movement_label is not None:
        movement_label = movement_label.upper()

        matches = [
            entry for entry in entries
            if entry["label"] == movement_label
        ]

        if matches:
            return axis_from_movement(movement_label)

    fallback_entries = [
        entry for entry in entries
        if entry["movement"] in ("S", "L")
    ]

    if not fallback_entries:
        raise RuntimeError("No straight or left movement was found for this traffic light.")

    fallback_label = fallback_entries[0]["label"]
    fallback_axis = axis_from_movement(fallback_label)

    print(
        f"\nRequested movement was not found. "
        f"Using first available movement {fallback_label}, axis {fallback_axis}."
    )

    return fallback_axis


def build_forced_state(state_length, entries, selected_axis, selected_turn):
    """
    Builds the traffic-light state for this test.

    Rules:
    1. The two opposite movements for the selected axis and turn get protected green G:
           --axis NS --turn straight -> NB-S and SB-S
           --axis EW --turn straight -> EB-S and WB-S
           --axis NS --turn left     -> NB-L and SB-L
           --axis EW --turn left     -> EB-L and WB-L

    2. All right-turn movements from all approaches get permissive green g
       if their outgoing lanes have space.

    3. Everything else stays red.

    4. If one SUMO signal index controls multiple connections, the whole
       signal index must share the same color. The printed table helps detect
       these cases.
    """
    state = ["r"] * state_length
    grouped = entries_by_signal_index(entries)

    selected_labels = movement_labels_for_axis_and_turn(selected_axis, selected_turn)

    selected_green_indices = []
    right_green_indices = []
    blocked_indices = []

    for signal_index, signal_entries in grouped.items():
        if signal_index >= state_length:
            continue

        selected_entries = [
            entry for entry in signal_entries
            if entry["label"] in selected_labels
        ]

        right_entries = [
            entry for entry in signal_entries
            if entry["movement"] == "R"
        ]

        # Protected green for the selected pair of opposite movements.
        if selected_entries:
            selected_out_lanes = outgoing_lanes_for_entries(selected_entries)

            if lanes_have_space(selected_out_lanes):
                state[signal_index] = "G"
                selected_green_indices.append(signal_index)
            else:
                state[signal_index] = "r"
                blocked_indices.append(signal_index)

            # If selected movement exists on this signal index, it takes priority.
            continue

        # Permissive green for all right turns if their exit lanes have room.
        if right_entries:
            right_out_lanes = outgoing_lanes_for_entries(right_entries)

            if lanes_have_space(right_out_lanes):
                state[signal_index] = "g"
                right_green_indices.append(signal_index)
            else:
                state[signal_index] = "r"
                blocked_indices.append(signal_index)

    return (
        "".join(state),
        sorted(selected_green_indices),
        sorted(right_green_indices),
        sorted(blocked_indices),
    )


def print_forced_state_summary(entries, selected_axis, selected_turn, selected_indices, right_indices):
    selected_labels = movement_labels_for_axis_and_turn(selected_axis, selected_turn)

    print("\nSelected protected movement pair:")
    print(f"  axis = {selected_axis}")
    print(f"  turn = {selected_turn}")
    print(f"  movements = {sorted(selected_labels)}")

    print("\nProtected green signal indices for selected movement pair:")
    print(f"  {sorted(selected_indices)}")

    print("\nPermissive green signal indices for right turns:")
    print(f"  {sorted(right_indices)}")

    print("\nConnections for selected protected movements:")
    for entry in entries:
        if entry["signal_index"] in selected_indices and entry["label"] in selected_labels:
            print(
                f"  idx={entry['signal_index']} | "
                f"{entry['label']} | "
                f"{entry['incoming_lane']} -> {entry['outgoing_lane']} "
                f"(angle={entry['angle']:.1f})"
            )

    print("\nConnections for right turns:")
    for entry in entries:
        if entry["signal_index"] in right_indices and entry["movement"] == "R":
            print(
                f"  idx={entry['signal_index']} | "
                f"{entry['label']} | "
                f"{entry['incoming_lane']} -> {entry['outgoing_lane']} "
                f"(angle={entry['angle']:.1f})"
            )


# ============================================================
# Main simulation
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tls",
        default=TARGET_TLS_ID,
        help="Traffic light ID to test.",
    )

    parser.add_argument(
        "--axis",
        choices=["NS", "EW", "ns", "ew"],
        default=None,
        help=(
            "Axis to keep green. "
            "NS means NB/SB pair. "
            "EW means EB/WB pair."
        ),
    )

    parser.add_argument(
        "--turn",
        choices=["straight", "left"],
        default="straight",
        help=(
            "Which pair to keep protected green: straight or left. "
            "Default is straight."
        ),
    )

    parser.add_argument(
        "--movement",
        default="NB-S",
        help=(
            "Movement used to infer the axis, such as NB-S, SB-S, EB-L, WB-L. "
            "Ignored if --axis is provided."
        ),
    )

    parser.add_argument(
        "--signal-index",
        type=int,
        default=None,
        help=(
            "Optional exact SUMO signal index. The script will infer that index's "
            "movement label and then use its axis."
        ),
    )

    parser.add_argument(
        "--end",
        type=float,
        default=DEFAULT_END_TIME,
        help="Simulation end time.",
    )

    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run headless SUMO instead of sumo-gui.",
    )

    args = parser.parse_args()

    if not args.nogui:
        ensure_xquartz()

    binary = SUMO_HEADLESS_BINARY if args.nogui else SUMO_GUI_BINARY

    sumo_cmd = [
        binary,
        "-n", NET_FILE,
        "-r", ROUTE_FILE,
        "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(args.end),
        "--max-num-vehicles", str(MAX_NUM_VEHICLES),
        "--max-depart-delay", str(MAX_DEPART_DELAY),
        "--time-to-teleport", str(TIME_TO_TELEPORT),
        *QUIET_SUMO_ARGS,
    ]

    print("Starting SUMO:")
    print(" ".join(sumo_cmd))

    traci.start(sumo_cmd)

    try:
        tls_ids = set(traci.trafficlight.getIDList())

        if args.tls not in tls_ids:
            raise RuntimeError(
                f"Traffic light {args.tls} was not found in this network."
            )

        state_length = len(traci.trafficlight.getRedYellowGreenState(args.tls))
        entries = get_signal_links(args.tls)

        if not entries:
            raise RuntimeError(
                f"No controlled signal links found for TLS {args.tls}."
            )

        print(f"\nTesting TLS: {args.tls}")
        print(f"State length: {state_length}")

        print_signal_table(entries)

        selected_axis = choose_axis(
            entries,
            requested_axis=args.axis,
            movement_label=args.movement,
            explicit_index=args.signal_index,
        )

        selected_turn = args.turn

        initial_state, selected_indices, right_indices, blocked_indices = build_forced_state(
            state_length,
            entries,
            selected_axis,
            selected_turn,
        )

        print_forced_state_summary(
            entries,
            selected_axis,
            selected_turn,
            selected_indices,
            right_indices,
        )

        print("\nInitial forced traffic-light state:")
        print(initial_state)

        print(
            "\nMeaning:\n"
            "  G = selected opposite pair, protected green\n"
            "  g = right turns, permissive green if exit path has space\n"
            "  r = all other movements red\n"
        )

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            forced_state, selected_indices, right_indices, blocked_indices = build_forced_state(
                state_length,
                entries,
                selected_axis,
                selected_turn,
            )

            # Keep forcing this test state every simulation step.
            traci.trafficlight.setRedYellowGreenState(
                args.tls,
                forced_state,
            )

            t = traci.simulation.getTime()

            if int(t) % 100 == 0 and abs(t - round(t)) < 1e-9:
                active = traci.vehicle.getIDCount()
                print(
                    f"t={t:.0f}, active vehicles={active}, "
                    f"axis={selected_axis}, "
                    f"turn={selected_turn}, "
                    f"selected_G={selected_indices}, "
                    f"right_g={right_indices}, "
                    f"blocked={blocked_indices}"
                )

    finally:
        try:
            traci.close()
        except Exception:
            pass

        print("\nSimulation ended.")


if __name__ == "__main__":
    main()
