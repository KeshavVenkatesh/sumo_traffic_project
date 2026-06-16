import argparse
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from functools import lru_cache
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

STEP_LENGTH = 1.0
DECISION_INTERVAL = 5.0
SIGNAL_UPDATE_INTERVAL = 2.0

T_YELLOW = 4.0
T_ALL_RED = 2.0

REQUIRED_EXIT_GAP = 14.0
QUEUE_SPEED_THRESHOLD = 0.1

MAX_ACTIVE_VEHICLE_CAP = 750
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

# Runtime lane preference helper. This does not change signal timing or routing.
# It only nudges straight-moving vehicles out of shared right/straight lanes
# when a dedicated straight lane exists on the same incoming edge.
TURN_LANE_CHANGE_DURATION = 12.0
# Do not request lane changes when cars are already queued near the stop line.
# Late forced lane changes are one common cause of the two-sided jams shown in SUMO.
LANE_PREF_MIN_DISTANCE_TO_END = 25.0
LANE_PREF_MIN_SPEED = 0.2
LANE_PREF_MAX_WAITING_TIME = 10.0
LANE_PREF_INTERVAL = 1.0
TURN_LANE_PREFERENCE_INDEX = {}

# Near-intersection movement decision helper.
# Vehicles make their S/R/L choice as they approach an intersection, then the
# route and lane request are aligned with that choice.  This keeps the actual
# approach behavior close to the 70 / 17.5 / 12.5 split without doing expensive
# per-step rerouting.
APPROACH_DECISION_INDEX = {}
APPROACH_TURN_DECISIONS = {}
APPROACH_TURN_COUNTS = Counter()
APPROACH_DECISION_MIN_DISTANCE_TO_END = 35.0
APPROACH_DECISION_MAX_DISTANCE_TO_END = 180.0
APPROACH_LANE_CHANGE_BASE_DISTANCE = 22.0
APPROACH_LANE_CHANGE_DISTANCE_PER_LANE = 32.0
APPROACH_LANE_CHANGE_DURATION = 16.0
APPROACH_DECISION_PRUNE_LIMIT = 10000

# Universal keep-clear / right-of-way safety gate.
# This is enforced at the vehicle level, not by flickering traffic lights.
# Cars approaching any junction, signalized or unsignalized, are held before
# the junction when their next edge or internal junction path is not clear.
KEEP_CLEAR_HELD_VEHICLES = set()
KEEP_CLEAR_HOLD_START_TIME = {}
KEEP_CLEAR_FORCE_RELEASE_UNTIL = {}
KEEP_CLEAR_LOOKAHEAD_DISTANCE = 12.0
KEEP_CLEAR_STOP_BUFFER = 6.0
KEEP_CLEAR_EXIT_GAP = CAR_LENGTH + CAR_MIN_GAP + 1.5
# If the first vehicle on the next edge is already moving at least this fast,
# do not hold the next vehicle just because the gap is momentarily small.
# This prevents slow queue discharge / phantom stops after the front car starts moving.
KEEP_CLEAR_EXIT_MOVING_OK_SPEED = 0.8
KEEP_CLEAR_INTERNAL_QUEUE_SPEED = 0.2
KEEP_CLEAR_INTERNAL_MAX_VEHICLES = 99
# Hard safety valve: custom keep-clear may hold a vehicle briefly, but it must
# never create a permanent/phantom stop. If a car is held longer than this,
# release it and temporarily stop applying custom keep-clear to that car.
KEEP_CLEAR_MAX_HOLD_TIME = 3.0
KEEP_CLEAR_RELEASE_COOLDOWN = 10.0
# If SUMO reports a vehicle as waiting for a long time but our keep-clear test
# no longer justifies holding it, explicitly release its speed override.
STUCK_RELEASE_WAIT_TIME = 12.0
STUCK_RELEASE_SPEED = 0.05

# Extra watchdog for lanes where the custom keep-clear logic previously
# produced phantom stops. These constants must be defined before
# is_phantom_stop_protected_location() is called.
PHANTOM_STOP_PROTECTED_LANES = {
    "-417109192_1",
    "417109221#0_0",
    "-417109221#0_0",
    "417109200_1",
}
PHANTOM_STOP_PROTECTED_EDGES = {
    "-417109192",
    "417109221#0",
    "-417109221#0",
    "417109200",
}
PHANTOM_STOP_PROTECTED_WAIT_TIME = 2.0

# Local circulation loop breaker.
# These two edges were observed to attract vehicles into a small repeating loop:
#   lane -417109221#0_0  -> edge -417109221#0
#   lane  417109200_1    -> edge  417109200
# The routing logic now avoids transitions that keep a vehicle bouncing between
# these roads, and route extension forcibly rewrites routes for cars currently
# on these edges.
LOOP_AVOIDANCE_PROTECTED_LANES = {
    "-417109221#0_0",
    "417109200_1",
}
LOOP_AVOIDANCE_PROTECTED_EDGES = {
    "-417109221#0",
    "417109200",
}

# Hardcoded local-loop kill switch.
# These two roads are not allowed to be ordinary random-cruising choices.
# Vehicles already on them may leave the area, but new random routes, approach
# decisions, spawns, and recoveries will not deliberately send vehicles into
# them. This is stronger than the general anti-cycle weighting above.
HARDCODED_NO_CRUISE_LOOP_LANES = {
    "-417109221#0_0",
    "417109200_1",
}
HARDCODED_NO_CRUISE_LOOP_EDGES = {
    "-417109221#0",
    "417109200",
}

# Unconnected-lane rescue.
# Some SUMO edges have a lane that does not legally connect to the next routed edge.
# Example in this map: lane 417292872_0 cannot continue to edge 518175685;
# only lanes 417292872_1 and 417292872_2 can. Without this rescue, vehicles
# can stop forever at the end of that lane.
UNCONNECTED_LANE_RESCUE_INTERVAL = 1.0
UNCONNECTED_LANE_RESCUE_DURATION = 12.0
UNCONNECTED_LANE_RESCUE_LOOKAHEAD = 240.0
KNOWN_UNCONNECTED_TRAP_LANES = {
    "417292872_0",
}

LOOP_MEMORY_EDGES = 8

# Ambulance settings
AMBULANCE_SPAWN_INTERVAL = 120
AMBULANCE_SPEED          = 16.0
AMBULANCE_COLOR          = (255, 50, 50, 255)

LOOP_RECENT_EDGE_PENALTY = 0.03
LOOP_PROTECTED_TRANSITION_PENALTY = 0.001

# General per-vehicle route-memory loop breaker.  The older fix only handled
# one observed two-road loop.  This prevents the same cycle pattern from
# reappearing elsewhere by remembering the recent edges for each vehicle and
# avoiding choices that would send it back into those edges.
VEHICLE_ROUTE_MEMORY_EDGES = 18
VEHICLE_EDGE_HISTORY = defaultdict(lambda: deque(maxlen=VEHICLE_ROUTE_MEMORY_EDGES))
VEHICLE_LAST_EDGE = {}
SHORT_LOOP_MAX_PERIOD = 5
SHORT_LOOP_MIN_REPEATS = 2
RECENT_EDGE_HARD_AVOID_LOOKBACK = 14

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

    try:
        import subprocess

        subprocess.run(
            ["open", "-a", "XQuartz"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(2)

        result = subprocess.run(
            ["launchctl", "getenv", "DISPLAY"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        display_value = result.stdout.strip()

        if display_value:
            os.environ["DISPLAY"] = display_value
            print(f"Using XQuartz DISPLAY={display_value}")
        elif "DISPLAY" not in os.environ:
            print("WARNING: XQuartz DISPLAY is not set. GUI mode may fail until XQuartz is opened.")

    except Exception as exc:
        print(f"Warning: could not configure XQuartz automatically: {exc}")


def lane_to_edge(lane_id):
    if not lane_id or lane_id.startswith(":"):
        return None

    if "_" not in lane_id:
        return lane_id

    return lane_id.rsplit("_", 1)[0]


def is_phantom_stop_protected_location(lane_id, edge_id=None):
    if lane_id in PHANTOM_STOP_PROTECTED_LANES:
        return True

    if edge_id is None:
        edge_id = lane_to_edge(lane_id)

    return edge_id in PHANTOM_STOP_PROTECTED_EDGES


def is_loop_avoidance_location(lane_id=None, edge_id=None):
    if lane_id in LOOP_AVOIDANCE_PROTECTED_LANES:
        return True

    if edge_id is None and lane_id is not None:
        edge_id = lane_to_edge(lane_id)

    return edge_id in LOOP_AVOIDANCE_PROTECTED_EDGES


def is_hardcoded_no_cruise_loop_location(lane_id=None, edge_id=None):
    if lane_id in HARDCODED_NO_CRUISE_LOOP_LANES:
        return True

    if edge_id is None and lane_id is not None:
        edge_id = lane_to_edge(lane_id)

    return edge_id in HARDCODED_NO_CRUISE_LOOP_EDGES


def remove_hardcoded_loop_region_from_graph(raw_graph):
    """Remove the observed two-road loop from normal random driving.

    This does not delete the roads from SUMO. It only removes them from the
    script's random route graph, so cars are not spawned onto them and route
    planning/approach decisions do not deliberately choose them. If a vehicle is
    already there because of an old route or SUMO routing, recovery sends it out.
    """
    blocked = set(HARDCODED_NO_CRUISE_LOOP_EDGES)
    cleaned = {}
    removed_nodes = 0
    removed_successors = 0

    for edge_id, successors in raw_graph.items():
        if edge_id in blocked:
            removed_nodes += 1
            continue

        filtered = []
        for successor in successors:
            if successor in blocked:
                removed_successors += 1
                continue
            filtered.append(successor)

        if filtered:
            cleaned[edge_id] = filtered

    print()
    print("Hardcoded loop-region rule:")
    print(f"  no-cruise edges:       {sorted(blocked)}")
    print(f"  removed graph nodes:   {removed_nodes}")
    print(f"  removed graph choices: {removed_successors}")

    return cleaned


def route_enters_hardcoded_loop_region(route_edges, allowed_first_edge=None):
    for index, edge_id in enumerate(route_edges):
        if edge_id not in HARDCODED_NO_CRUISE_LOOP_EDGES:
            continue
        if index == 0 and edge_id == allowed_first_edge:
            continue
        return True
    return False


def should_avoid_successor_for_loop(current_edge, next_edge, recent_edges=None):
    if not next_edge:
        return False

    if next_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
        return True

    if (
        current_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
        and next_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
    ):
        return True

    if recent_edges and next_edge in set(recent_edges):
        return True

    return False


def loop_avoidance_weight_multiplier(current_edge, next_edge, recent_edges=None):
    multiplier = 1.0

    if next_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
        multiplier *= 1e-9

    if (
        current_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
        and next_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
    ):
        multiplier *= LOOP_PROTECTED_TRANSITION_PENALTY

    if recent_edges and next_edge in set(recent_edges):
        multiplier *= LOOP_RECENT_EDGE_PENALTY

    return multiplier


def update_vehicle_edge_history(veh_id, edge_id):
    if edge_id is None or edge_id.startswith(":"):
        return tuple(VEHICLE_EDGE_HISTORY.get(veh_id, ()))

    last_edge = VEHICLE_LAST_EDGE.get(veh_id)
    if last_edge != edge_id:
        VEHICLE_EDGE_HISTORY[veh_id].append(edge_id)
        VEHICLE_LAST_EDGE[veh_id] = edge_id

    return tuple(VEHICLE_EDGE_HISTORY[veh_id])


def prune_vehicle_edge_history(active_ids):
    active_ids = set(active_ids)

    for veh_id in list(VEHICLE_EDGE_HISTORY.keys()):
        if veh_id not in active_ids:
            VEHICLE_EDGE_HISTORY.pop(veh_id, None)
            VEHICLE_LAST_EDGE.pop(veh_id, None)


def vehicle_recent_edges(veh_id):
    return tuple(VEHICLE_EDGE_HISTORY.get(veh_id, ()))


def has_repeated_tail_pattern(edges, max_period=SHORT_LOOP_MAX_PERIOD, repeats=SHORT_LOOP_MIN_REPEATS):
    edges = list(edges)
    for period in range(1, max_period + 1):
        needed = period * repeats
        if len(edges) < needed:
            continue
        tail = edges[-period:]
        if all(edges[-period * k:-period * (k - 1) if k > 1 else None] == tail for k in range(1, repeats + 1)):
            return True
    return False


def candidate_edges_after_loop_filter(candidates, get_edge, recent_edges, previous_edge=None):
    """Return loop-safe candidates if possible, otherwise original candidates.

    This is intentionally cheap: it only uses the small per-vehicle recent-edge
    deque and does not call TraCI.
    """
    candidates = list(candidates)
    if not candidates:
        return []

    recent_set = set(tuple(recent_edges or ())[-RECENT_EDGE_HARD_AVOID_LOOKBACK:])
    preferred = []

    for candidate in candidates:
        edge = get_edge(candidate)
        if edge is None:
            continue
        if previous_edge is not None and edge == previous_edge:
            continue
        if should_avoid_successor_for_loop(None, edge, recent_set):
            continue
        preferred.append(candidate)

    return preferred or candidates


@lru_cache(maxsize=20000)
def cached_lane_length(lane_id):
    try:
        return traci.lane.getLength(lane_id)
    except traci.TraCIException:
        return 0.0


@lru_cache(maxsize=20000)
def cached_edge_lanes(edge_id):
    try:
        lane_count = traci.edge.getLaneNumber(edge_id)
    except traci.TraCIException:
        return tuple()

    return tuple(f"{edge_id}_{lane_index}" for lane_index in range(lane_count))


def safe_vehicle_set_speed(veh_id, speed):
    try:
        traci.vehicle.setSpeed(veh_id, speed)
        return True
    except traci.TraCIException:
        return False


def current_sim_time():
    try:
        return traci.simulation.getTime()
    except traci.TraCIException:
        return 0.0


def clear_keep_clear_tracking(veh_id):
    KEEP_CLEAR_HELD_VEHICLES.discard(veh_id)
    KEEP_CLEAR_HOLD_START_TIME.pop(veh_id, None)


def keep_clear_is_in_release_cooldown(veh_id):
    return KEEP_CLEAR_FORCE_RELEASE_UNTIL.get(veh_id, -1.0) > current_sim_time()


def release_keep_clear_vehicle(veh_id):
    # setSpeed(0) persists until reset with setSpeed(-1), so always reset when
    # this vehicle may have been controlled by our custom keep-clear gate.
    was_tracked = veh_id in KEEP_CLEAR_HELD_VEHICLES or veh_id in KEEP_CLEAR_HOLD_START_TIME

    if safe_vehicle_set_speed(veh_id, -1):
        clear_keep_clear_tracking(veh_id)
        return was_tracked

    clear_keep_clear_tracking(veh_id)
    return False


def force_release_keep_clear_vehicle(veh_id):
    now = current_sim_time()
    KEEP_CLEAR_FORCE_RELEASE_UNTIL[veh_id] = now + KEEP_CLEAR_RELEASE_COOLDOWN
    return release_keep_clear_vehicle(veh_id)


def hold_vehicle_before_junction(veh_id):
    now = current_sim_time()

    # If this vehicle was recently released by the watchdog, do not immediately
    # re-hold it. This prevents the "stopped for no reason forever" behavior.
    if KEEP_CLEAR_FORCE_RELEASE_UNTIL.get(veh_id, -1.0) > now:
        safe_vehicle_set_speed(veh_id, -1)
        return False

    hold_start = KEEP_CLEAR_HOLD_START_TIME.get(veh_id)
    if hold_start is None:
        KEEP_CLEAR_HOLD_START_TIME[veh_id] = now
    elif now - hold_start >= KEEP_CLEAR_MAX_HOLD_TIME:
        force_release_keep_clear_vehicle(veh_id)
        return False

    if safe_vehicle_set_speed(veh_id, 0.0):
        KEEP_CLEAR_HELD_VEHICLES.add(veh_id)
        return True

    return False


def planned_next_edge_from_route(veh_id, current_edge):
    try:
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
    except traci.TraCIException:
        return None

    if route_index < 0 or route_index >= len(route):
        return None

    if route[route_index] != current_edge:
        try:
            route_index = route.index(current_edge)
        except ValueError:
            return None

    if route_index + 1 >= len(route):
        return None

    return route[route_index + 1]


def next_edge_has_exit_space(next_edge):
    lanes = cached_edge_lanes(next_edge)

    if not lanes:
        return True

    # A car should be allowed to enter the junction if at least one lane on the
    # next edge has enough immediately usable space. A moving vehicle near the
    # start of the next edge is not treated as a hard blockage; otherwise the
    # queue develops a very slow start-up wave and phantom stops.
    for lane_id in lanes:
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        except traci.TraCIException:
            continue

        if not veh_ids:
            return True

        closest_to_start = None
        closest_speed = 0.0

        for other_id in veh_ids:
            try:
                pos = traci.vehicle.getLanePosition(other_id)
                speed = traci.vehicle.getSpeed(other_id)
            except traci.TraCIException:
                continue

            if closest_to_start is None or pos < closest_to_start:
                closest_to_start = pos
                closest_speed = speed

        if closest_to_start is None:
            return True

        if closest_to_start >= KEEP_CLEAR_EXIT_GAP:
            return True

        # If the closest vehicle is already moving away from the junction,
        # there will be usable space by the time the next car reaches the exit.
        if closest_speed >= KEEP_CLEAR_EXIT_MOVING_OK_SPEED:
            return True

    return False


@lru_cache(maxsize=20000)
def get_lane_links(lane_id):
    """Return SUMO lane-link tuples for a lane.

    The keep-clear code uses this to find the internal/via lane between
    the current lane and the next edge.  Some SUMO/TraCI versions support
    extended=True and some do not, so this helper handles both cases.
    """
    try:
        try:
            return tuple(traci.lane.getLinks(lane_id, extended=True))
        except TypeError:
            return tuple(traci.lane.getLinks(lane_id))
    except traci.TraCIException:
        return tuple()



def lane_has_connection_to_edge(lane_id, target_edge):
    """Return True if this exact lane has a SUMO link to target_edge."""
    if not lane_id or not target_edge:
        return False

    for link in get_lane_links(lane_id):
        if not link:
            continue

        to_lane = link[0]
        if lane_to_edge(to_lane) == target_edge:
            return True

    return False


def lane_outgoing_edges(lane_id, allowed_edges=None):
    """Outgoing edge IDs reachable from this exact lane."""
    result = []
    allowed = set(allowed_edges) if allowed_edges is not None else None

    for link in get_lane_links(lane_id):
        if not link:
            continue

        to_edge = lane_to_edge(link[0])
        if to_edge is None or to_edge.startswith(":"):
            continue
        if allowed is not None and to_edge not in allowed:
            continue
        if to_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
            continue
        if to_edge not in result:
            result.append(to_edge)

    return result


def lanes_on_edge_connecting_to(current_edge, target_edge):
    """Lane IDs on current_edge that can legally connect to target_edge."""
    if not current_edge or not target_edge:
        return []

    result = []
    for lane_id in cached_edge_lanes(current_edge):
        if lane_has_connection_to_edge(lane_id, target_edge):
            result.append(lane_id)

    return result


def choose_best_rescue_lane(candidate_lanes, current_lane_index):
    """Pick a reachable lane that is nearby and not obviously jammed."""
    scored = []

    for lane_id in candidate_lanes:
        lane_index = lane_index_from_lane_id(lane_id)
        if lane_index is None:
            continue

        try:
            halting = traci.lane.getLastStepHaltingNumber(lane_id)
            vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
        except traci.TraCIException:
            halting = 0
            vehicles = 0

        lane_distance = abs(lane_index - current_lane_index) if current_lane_index is not None else 0
        scored.append((halting, vehicles, lane_distance, lane_index, lane_id))

    if not scored:
        return None, None

    scored.sort()
    return scored[0][3], scored[0][4]


def force_route_to_reachable_lane_successor(
    veh_id,
    current_edge,
    current_lane,
    previous_edge,
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    args,
    recent_edges=None,
):
    """Last-resort fallback when the current lane cannot serve the route.

    If the vehicle is already too close to the lane end to change lanes, the
    safest way to avoid a permanent stop is to rewrite the route to an outgoing
    edge that the current lane can actually take.
    """
    outgoing_edges = lane_outgoing_edges(current_lane, allowed_edges=raw_graph)
    if not outgoing_edges:
        return False

    candidates = [edge for edge in outgoing_edges if edge != previous_edge] or outgoing_edges
    weights = [
        successor_weight(
            current_edge=current_edge,
            next_edge=edge,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        ) * loop_avoidance_weight_multiplier(current_edge, edge, recent_edges)
        for edge in candidates
    ]
    next_edge = weighted_choice(rng, candidates, weights)

    continuation_counts = Counter()
    if next_edge in raw_graph:
        continuation = build_random_walk_route(
            start_edge=next_edge,
            lookahead_edges=max(2, args.route_lookahead_edges - 1),
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=continuation_counts,
            args=args,
            previous_edge=current_edge,
            initial_recent_edges=tuple(recent_edges or ()) + (current_edge,),
        )
    else:
        continuation = [next_edge]

    if not continuation or continuation[0] != next_edge:
        continuation = [next_edge]

    new_route = [current_edge] + continuation
    cleaned = []
    for edge_id in new_route:
        if cleaned and cleaned[-1] == edge_id:
            continue
        cleaned.append(edge_id)

    if len(cleaned) < 2:
        return False

    try:
        traci.vehicle.setRoute(veh_id, cleaned)
        safe_vehicle_set_speed(veh_id, -1)
        return True
    except traci.TraCIException:
        return False


def rescue_vehicle_from_unconnected_lane(
    veh_id,
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    args,
    recent_edges=None,
):
    """Prevent vehicles from stopping at the end of a lane with no route link.

    This fixes the 417292872_0 problem generically. If a vehicle's current lane
    cannot reach its next routed edge, it is moved to a lane on the same edge
    that can. If no such lane exists, the route is rewritten to an outgoing edge
    that the current lane can legally take.
    """
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        if not lane_id or lane_id.startswith(":"):
            return False
        current_edge = lane_to_edge(lane_id)
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = cached_lane_length(lane_id)
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
    except traci.TraCIException:
        return False

    if current_edge is None or lane_len <= 0.0:
        return False

    distance_to_end = lane_len - lane_pos
    if distance_to_end > UNCONNECTED_LANE_RESCUE_LOOKAHEAD and lane_id not in KNOWN_UNCONNECTED_TRAP_LANES:
        return False

    next_edge = planned_next_edge_from_route(veh_id, current_edge)
    if next_edge is None or next_edge.startswith(":"):
        return False

    # Nothing to fix if this exact lane can already continue to the routed edge.
    if lane_has_connection_to_edge(lane_id, next_edge):
        return False

    current_lane_index = lane_index_from_lane_id(lane_id)
    candidate_lanes = lanes_on_edge_connecting_to(current_edge, next_edge)
    target_lane_index, _target_lane = choose_best_rescue_lane(candidate_lanes, current_lane_index)

    if target_lane_index is not None and target_lane_index != current_lane_index:
        try:
            # Clear any previous custom stop, then push the vehicle into a lane
            # that actually has the required connection.
            release_keep_clear_vehicle(veh_id)
            safe_vehicle_set_speed(veh_id, -1)
            traci.vehicle.setLaneChangeMode(veh_id, 1621)
            traci.vehicle.changeLane(veh_id, target_lane_index, UNCONNECTED_LANE_RESCUE_DURATION)
            return True
        except traci.TraCIException:
            pass

    previous_edge = route[route_index - 1] if route_index > 0 and route_index < len(route) else None
    return force_route_to_reachable_lane_successor(
        veh_id=veh_id,
        current_edge=current_edge,
        current_lane=lane_id,
        previous_edge=previous_edge,
        raw_graph=raw_graph,
        edge_metadata=edge_metadata,
        core_edges=core_edges,
        turn_index=turn_index,
        rng=rng,
        turn_counts=turn_counts,
        args=args,
        recent_edges=recent_edges,
    )


def rescue_unconnected_lanes_for_all_vehicles(
    active_ids,
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    args,
):
    rescued = 0
    for veh_id in active_ids:
        if rescue_vehicle_from_unconnected_lane(
            veh_id=veh_id,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            turn_index=turn_index,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
            recent_edges=vehicle_recent_edges(veh_id),
        ):
            rescued += 1
    return rescued


def internal_lanes_for_link(current_lane, next_edge):
    internal_lanes = []

    for link in get_lane_links(current_lane):
        if not link:
            continue

        to_lane = link[0]
        to_edge = lane_to_edge(to_lane)

        if to_edge != next_edge:
            continue

        # Extended TraCI lane links usually expose the internal/via lane at
        # index 4: (toLane, hasPrio, isOpen, hasFoe, viaLane, state, dir, len).
        if len(link) > 4 and isinstance(link[4], str) and link[4]:
            internal_lanes.append(link[4])

    return internal_lanes


def internal_junction_path_is_clear(current_lane, next_edge):
    internal_lanes = internal_lanes_for_link(current_lane, next_edge)

    if not internal_lanes:
        return True

    for internal_lane in internal_lanes:
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(internal_lane)
        except traci.TraCIException:
            continue

        # Do not block the next car merely because another car is moving through
        # the internal junction lane. That created very slow start-up waves.
        # Only hold vehicles when the internal path contains a stopped/near-stopped
        # vehicle or is truly packed.
        if len(veh_ids) > KEEP_CLEAR_INTERNAL_MAX_VEHICLES:
            return False

        for other_id in veh_ids:
            try:
                if traci.vehicle.getSpeed(other_id) <= KEEP_CLEAR_INTERNAL_QUEUE_SPEED:
                    return False
            except traci.TraCIException:
                continue

    return True


def should_hold_for_keep_clear(veh_id):
    if keep_clear_is_in_release_cooldown(veh_id):
        return False

    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
    except traci.TraCIException:
        return False

    current_edge = lane_to_edge(lane_id)

    # Do not intervene once the vehicle is already inside an internal junction
    # lane. The whole point is to stop it before this point.
    if current_edge is None or lane_id.startswith(":") or current_edge.startswith(":"):
        return False

    lane_len = cached_lane_length(lane_id)
    if lane_len <= 0:
        return False

    try:
        lane_pos = traci.vehicle.getLanePosition(veh_id)
    except traci.TraCIException:
        return False

    distance_to_junction = lane_len - lane_pos

    # On a few short/odd connector lanes, the custom setSpeed(0) keep-clear
    # override can create a phantom stop even when SUMO's own model would let
    # the car proceed safely. Do not allow a custom hold to persist there.
    if is_phantom_stop_protected_location(lane_id, current_edge):
        try:
            if traci.vehicle.getWaitingTime(veh_id) >= PHANTOM_STOP_PROTECTED_WAIT_TIME:
                force_release_keep_clear_vehicle(veh_id)
                return False
        except traci.TraCIException:
            return False

    # Only the lead vehicle near the junction should be controlled. Cars behind
    # it should simply follow normally; holding every car in the queue makes the
    # queue restart painfully slowly.
    try:
        leader = traci.vehicle.getLeader(veh_id, KEEP_CLEAR_LOOKAHEAD_DISTANCE)
    except traci.TraCIException:
        leader = None

    if leader is not None:
        _, leader_gap = leader
        if leader_gap < max(2.0, distance_to_junction - KEEP_CLEAR_STOP_BUFFER):
            return False

    # Only check cars that are close enough to actually enter the next junction.
    if distance_to_junction > KEEP_CLEAR_LOOKAHEAD_DISTANCE:
        return False

    next_edge = planned_next_edge_from_route(veh_id, current_edge)

    if next_edge is None or next_edge.startswith(":"):
        return False

    if not next_edge_has_exit_space(next_edge):
        return True

    if not internal_junction_path_is_clear(lane_id, next_edge):
        return True

    return False


def release_if_unjustifiably_stuck(veh_id):
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        edge_id = lane_to_edge(lane_id)
        speed = traci.vehicle.getSpeed(veh_id)
        waiting_time = traci.vehicle.getWaitingTime(veh_id)
    except traci.TraCIException:
        return False

    if speed > STUCK_RELEASE_SPEED:
        return False

    # Strong lane-specific watchdog for the locations where phantom stops were
    # observed. This only clears our speed override; it does not change the
    # traffic light or disable SUMO's own collision/right-of-way rules.
    if is_phantom_stop_protected_location(lane_id, edge_id) and waiting_time >= PHANTOM_STOP_PROTECTED_WAIT_TIME:
        return force_release_keep_clear_vehicle(veh_id)

    if waiting_time < STUCK_RELEASE_WAIT_TIME:
        return False

    # General watchdog: never let a custom setSpeed(0) hold become permanent.
    # After this release, the cooldown prevents the same car from being held
    # again immediately, so it can resume normal SUMO behavior.
    return force_release_keep_clear_vehicle(veh_id)


def apply_keep_clear_and_right_of_way_to_vehicle(veh_id):
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
    except traci.TraCIException:
        clear_keep_clear_tracking(veh_id)
        return False

    if lane_id.startswith(":"):
        clear_keep_clear_tracking(veh_id)
        return False

    # Run the stuck watchdog before applying a new hold. This prevents the
    # reported lanes from being re-held forever on every simulation step.
    if release_if_unjustifiably_stuck(veh_id):
        return False

    if should_hold_for_keep_clear(veh_id):
        return hold_vehicle_before_junction(veh_id)

    release_keep_clear_vehicle(veh_id)
    return False


def apply_keep_clear_and_right_of_way_to_all_vehicles():
    held_count = 0
    active_ids = set(traci.vehicle.getIDList())

    # Release/delete stale held IDs first.
    for veh_id in list(KEEP_CLEAR_HELD_VEHICLES):
        if veh_id not in active_ids:
            clear_keep_clear_tracking(veh_id)

    for veh_id in list(KEEP_CLEAR_HOLD_START_TIME):
        if veh_id not in active_ids:
            clear_keep_clear_tracking(veh_id)

    now = current_sim_time()
    for veh_id, until in list(KEEP_CLEAR_FORCE_RELEASE_UNTIL.items()):
        if veh_id not in active_ids or until <= now:
            KEEP_CLEAR_FORCE_RELEASE_UNTIL.pop(veh_id, None)

    for veh_id in active_ids:
        if apply_keep_clear_and_right_of_way_to_vehicle(veh_id):
            held_count += 1

    return held_count


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
            "x": 0.0,
            "y": 0.0,
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

                try:
                    shape = edge.getShape()
                    if shape:
                        x = sum(point[0] for point in shape) / len(shape)
                        y = sum(point[1] for point in shape) / len(shape)
                    else:
                        x = y = 0.0
                except Exception:
                    x = y = 0.0

                metadata[edge_id].update(
                    {
                        "type": edge_type,
                        "speed": speed,
                        "lanes": lanes,
                        "priority": priority,
                        "x": x,
                        "y": y,
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
                points = []

                for lane_index in range(lane_count):
                    lane_id = f"{edge_id}_{lane_index}"
                    speeds.append(traci.lane.getMaxSpeed(lane_id))
                    points.extend(traci.lane.getShape(lane_id))

                if speeds:
                    item["speed"] = max(speeds)

                if points and item.get("x", 0.0) == 0.0 and item.get("y", 0.0) == 0.0:
                    item["x"] = sum(point[0] for point in points) / len(points)
                    item["y"] = sum(point[1] for point in points) / len(points)

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


def edge_xy(edge_id, edge_metadata):
    item = edge_metadata.get(edge_id, {})
    return float(item.get("x", 0.0)), float(item.get("y", 0.0))


def build_spawn_zones(start_edges, edge_metadata, grid_size):
    """Split start edges into geographic zones so spawning covers the whole map.

    This prevents all cars from being created on whichever arterial happens to
    have the highest edge weight. Spawning later cycles through these zones in
    round-robin order.
    """
    if not start_edges:
        return []

    grid_size = max(1, int(grid_size))

    xs = [edge_xy(edge_id, edge_metadata)[0] for edge_id in start_edges]
    ys = [edge_xy(edge_id, edge_metadata)[1] for edge_id in start_edges]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    zones = defaultdict(list)

    for edge_id in start_edges:
        x, y = edge_xy(edge_id, edge_metadata)
        gx = min(grid_size - 1, max(0, int((x - min_x) / width * grid_size)))
        gy = min(grid_size - 1, max(0, int((y - min_y) / height * grid_size)))
        zones[(gx, gy)].append(edge_id)

    zone_list = [
        {"id": zone_id, "edges": sorted(edges)}
        for zone_id, edges in sorted(zones.items())
        if edges
    ]

    print()
    print("Spawn zones:")
    print(f"  zones used:      {len(zone_list)}")
    print(f"  grid size:       {grid_size} x {grid_size}")
    print(f"  candidate edges: {len(start_edges)}")

    return zone_list


def choose_spawn_edge(sim_state, start_edges_or_zones, edge_metadata, rng):
    """Choose spawn edge with zone balancing if zones are provided.

    start_edges_or_zones may be either:
      - a flat list of edge IDs, or
      - a list of {id, edges} zone dictionaries.
    """
    if not start_edges_or_zones:
        return None

    first = start_edges_or_zones[0]

    if isinstance(first, dict):
        zone_index = sim_state.get("next_spawn_zone_index", 0) % len(start_edges_or_zones)
        sim_state["next_spawn_zone_index"] = zone_index + 1
        zone = start_edges_or_zones[zone_index]
        edges = zone["edges"]
    else:
        edges = start_edges_or_zones

    return weighted_choice(
        rng,
        edges,
        [edge_base_weight(edge_id, edge_metadata) for edge_id in edges],
    )


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

    # Avoid route choices that keep vehicles circulating between the known
    # two-road loop around -417109221#0 and 417109200.
    weight *= loop_avoidance_weight_multiplier(current_edge, next_edge)

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
        '''    <vType id="ambulance"
           vClass="emergency"
           guiShape="emergency"
           length="15.0"
           width="5.0"
           minGap="0.5"
           accel="3.5"
           decel="6.0"
           emergencyDecel="9.0"
           maxSpeed="22.2"
           sigma="0.1"
           tau="0.5"
           color="255,0,0"
           jmIgnoreFoeSpeed="100"
           jmIgnoreFoeProb="1.0"
           jmIgnoreKeepClearTime="-1"
           jmDriveAfterYellowTime="100"
           jmDriveAfterRedTime="100"/>'''
    )

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
           lcStrategic="80.0"
           lcCooperative="1.0"
           lcSpeedGain="0.03"
           lcKeepRight="0.0"
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

        # Actual tuple layout in SUMO 1.26:
        #   (toLane, hasPriority, isOpen, hasFoe, viaLane, state, direction, length)
        #     [0]      [1]          [2]    [3]      [4]     [5]      [6]       [7]
        #
        # Match on outgoing *edge* rather than exact lane — trafficlight.getControlledLinks
        # and lane.getLinks sometimes disagree on which lane index represents the connection,
        # causing exact-lane matches to fail even when the direction is clearly available.
        out_edge = out_lane.rsplit("_", 1)[0] if "_" in out_lane else out_lane

        for link in links:
            if not link or len(link) < 7:
                continue

            to_lane = link[0]
            to_edge = to_lane.rsplit("_", 1)[0] if "_" in to_lane else to_lane

            if to_edge != out_edge:
                continue

            movement = sumo_link_direction_to_movement(link[6])
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
        "last_signal_update": -1e9,
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
        check_space=False,
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
    controller["last_signal_update"] = traci.simulation.getTime()


def update_green(controller):
    phase = controller["phases"][controller["phase_pos"]]

    green_state, active_indices = build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
        check_space=False,
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
        sim_time = traci.simulation.getTime()

        if controller["mode"] == "green":
            # Do not resend the exact same green state every simulation step.
            # This saves many TraCI calls and also preserves the no-flicker behavior:
            # the signal stays green for the phase, while vehicles obey SUMO keep-clear logic.
            if sim_time - controller.get("last_signal_update", -1e9) >= SIGNAL_UPDATE_INTERVAL:
                update_green(controller)
                controller["last_signal_update"] = sim_time

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

                    if incoming_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
                        continue

                    for out_lane in lane_sets["out"]:
                        outgoing_edge = lane_to_edge(out_lane)

                        if outgoing_edge is None:
                            continue

                        if outgoing_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
                            continue

                        # For straight movements, allow outgoing edges that are
                        # network boundary edges (not in raw_graph because they
                        # have no further successors). Filtering them out was
                        # the main cause of straight options being missing from
                        # the index.
                        if movement_group != "S" and outgoing_edge not in allowed_edges:
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


def lane_index_from_lane_id(lane_id):
    """Return SUMO lane index from a lane id like edge_2."""
    if not lane_id or "_" not in lane_id:
        return None

    try:
        return int(lane_id.rsplit("_", 1)[1])
    except ValueError:
        return None


def build_turn_lane_preference_index(controllers):
    """
    Build per-incoming-edge lane-choice information for controlled intersections.

    Purpose:
      If a straight-going car is sitting in a lane that can also turn right,
      and the same incoming edge has a dedicated straight-only lane, nudge the
      straight-going car into the straight-only lane. This keeps right-turning
      cars from being trapped behind straight cars waiting for the straight phase.

    This does NOT change the traffic-light logic or the car's route.
    """
    index = defaultdict(lambda: {
        "edge_to_movement": {},
        "edge_to_lanes": defaultdict(lambda: defaultdict(set)),
        "movement_to_lanes": defaultdict(set),
        "lane_to_movements": defaultdict(set),
    })

    for controller in controllers:
        tls_id = controller["tls_id"]

        try:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        except traci.TraCIException:
            continue

        for signal_links in controlled_links:
            for link in signal_links:
                if len(link) < 2:
                    continue

                in_lane = link[0]
                out_lane = link[1]

                if not in_lane or not out_lane:
                    continue

                incoming_edge = lane_to_edge(in_lane)
                outgoing_edge = lane_to_edge(out_lane)

                if incoming_edge is None or outgoing_edge is None:
                    continue

                label = classify_connection(in_lane, out_lane)

                if not label or "-" not in label:
                    continue

                movement = label.split("-")[-1]

                if movement not in TURN_PROBABILITIES:
                    continue

                item = index[incoming_edge]

                # If SUMO exposes multiple lane links to the same outgoing edge,
                # prefer S when any straight connection exists; otherwise keep the
                # first normal movement seen. This avoids misclassifying shared
                # through connections as only right/left.
                if outgoing_edge not in item["edge_to_movement"]:
                    item["edge_to_movement"][outgoing_edge] = movement
                elif movement == "S":
                    item["edge_to_movement"][outgoing_edge] = "S"

                item["edge_to_lanes"][outgoing_edge][movement].add(in_lane)
                item["movement_to_lanes"][movement].add(in_lane)
                item["lane_to_movements"][in_lane].add(movement)

    incoming_edges = len(index)
    straight_only_edges = 0
    shared_right_straight_edges = 0

    for item in index.values():
        lane_to_movements = item["lane_to_movements"]
        has_straight_only = any(movements == {"S"} for movements in lane_to_movements.values())
        has_shared_right_straight = any({"S", "R"}.issubset(movements) for movements in lane_to_movements.values())

        if has_straight_only:
            straight_only_edges += 1
        if has_shared_right_straight:
            shared_right_straight_edges += 1

    print()
    print("Turn-lane preference index:")
    print(f"  controlled incoming edges indexed:        {incoming_edges}")
    print(f"  edges with straight-only lane options:    {straight_only_edges}")
    print(f"  edges with shared right/straight lanes:   {shared_right_straight_edges}")

    return index


def build_approach_decision_index(raw_graph):
    """Precompute all lane/link choices for near-intersection decisions.

    This is broader than the traffic-light-only index: it covers every drivable
    edge with SUMO lane links, including unsignalized intersections.  Building
    it once keeps the simulation fast; runtime only does dictionary lookups.
    """
    index = {}
    raw_edge_set = set(raw_graph)

    for incoming_edge, successors in raw_graph.items():
        successor_set = set(successors)
        item = {
            "edge_to_movement": {},
            "edge_to_lanes": defaultdict(lambda: defaultdict(set)),
            "movement_to_lanes": defaultdict(set),
            "lane_to_movements": defaultdict(set),
            "movement_to_edges": defaultdict(set),
        }

        for in_lane in cached_edge_lanes(incoming_edge):
            for link in get_lane_links(in_lane):
                if not link:
                    continue

                out_lane = link[0]
                outgoing_edge = lane_to_edge(out_lane)

                if outgoing_edge is None:
                    continue

                if outgoing_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
                    continue

                if outgoing_edge not in successor_set and outgoing_edge not in raw_edge_set:
                    continue

                if outgoing_edge == incoming_edge:
                    continue

                movement = None
                if len(link) > 6:
                    movement = sumo_link_direction_to_movement(link[6])

                if movement is None:
                    _, movement = classify_movement_by_geometry(in_lane, out_lane)

                if movement not in TURN_PROBABILITIES:
                    continue

                # If multiple links point to the same outgoing edge, prefer S
                # when any straight link exists.
                if outgoing_edge not in item["edge_to_movement"]:
                    item["edge_to_movement"][outgoing_edge] = movement
                elif movement == "S":
                    item["edge_to_movement"][outgoing_edge] = "S"

                item["edge_to_lanes"][outgoing_edge][movement].add(in_lane)
                item["movement_to_lanes"][movement].add(in_lane)
                item["lane_to_movements"][in_lane].add(movement)
                item["movement_to_edges"][movement].add(outgoing_edge)

        if any(item["movement_to_edges"].values()):
            index[incoming_edge] = item

    movement_edge_counts = Counter()
    straight_only_edges = 0
    shared_right_straight_edges = 0

    for item in index.values():
        for movement, edges in item["movement_to_edges"].items():
            movement_edge_counts[movement] += len(edges)

        lane_to_movements = item["lane_to_movements"]
        if any(movements == {"S"} for movements in lane_to_movements.values()):
            straight_only_edges += 1
        if any({"S", "R"}.issubset(movements) for movements in lane_to_movements.values()):
            shared_right_straight_edges += 1

    print()
    print("Approach decision index:")
    print(f"  incoming edges indexed:              {len(index)}")
    print(f"  straight outgoing choices:           {movement_edge_counts['S']}")
    print(f"  right-turn outgoing choices:         {movement_edge_counts['R']}")
    print(f"  left-turn outgoing choices:          {movement_edge_counts['L']}")
    print(f"  edges with straight-only lanes:      {straight_only_edges}")
    print(f"  edges with shared right/straight:    {shared_right_straight_edges}")

    return index


def target_lanes_for_movement(lane_info, outgoing_edge, movement):
    """Return preferred lane ids for a planned movement."""
    edge_to_lanes = lane_info["edge_to_lanes"]
    movement_to_lanes = lane_info["movement_to_lanes"]
    lane_to_movements = lane_info["lane_to_movements"]

    specific_lanes = set(edge_to_lanes.get(outgoing_edge, {}).get(movement, set()))
    movement_lanes = set(movement_to_lanes.get(movement, set()))
    candidates = specific_lanes or movement_lanes

    if not candidates:
        return []

    def movements_for(lane_id):
        return set(lane_to_movements.get(lane_id, set()))

    def avoid_rightmost_when_possible(lanes):
        # In SUMO's right-hand traffic convention, lane index 0 is usually the
        # rightmost lane.  For straight traffic, avoid that lane whenever another
        # straight-capable lane exists, so right turns are not trapped behind
        # through traffic.
        if len(lanes) <= 1:
            return list(lanes)

        non_rightmost = [
            lane for lane in lanes
            if (lane_index_from_lane_id(lane) is not None and lane_index_from_lane_id(lane) > 0)
        ]
        return non_rightmost or list(lanes)

    if movement == "S":
        # Best case: true straight-only lanes. This is the main fix.
        straight_only = [lane for lane in candidates if movements_for(lane) == {"S"}]
        if straight_only:
            return sorted(avoid_rightmost_when_possible(straight_only))

        # Next best: lanes that allow straight but do not also allow right.
        no_right = [lane for lane in candidates if "S" in movements_for(lane) and "R" not in movements_for(lane)]
        if no_right:
            return sorted(avoid_rightmost_when_possible(no_right))

        return sorted(avoid_rightmost_when_possible(candidates))

    if movement == "R":
        right_only = [lane for lane in candidates if movements_for(lane) == {"R"}]
        if right_only:
            return sorted(right_only)
        return sorted(candidates)

    if movement == "L":
        left_only = [lane for lane in candidates if movements_for(lane) == {"L"}]
        if left_only:
            return sorted(left_only)
        return sorted(candidates)

    return sorted(candidates)


def choose_best_target_lane(candidate_lanes, current_lane_index):
    """Pick a target lane that is nearby and not obviously packed."""
    scored = []

    for lane_id in candidate_lanes:
        lane_index = lane_index_from_lane_id(lane_id)

        if lane_index is None:
            continue

        try:
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            halting = traci.lane.getLastStepHaltingNumber(lane_id)
        except traci.TraCIException:
            veh_count = 0
            halting = 0

        distance = abs(lane_index - current_lane_index) if current_lane_index is not None else 0
        scored.append((halting, veh_count, distance, lane_index, lane_id))

    if not scored:
        return None

    scored.sort()
    return scored[0][3]


def planned_next_edge_for_vehicle(veh_id, current_edge):
    try:
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
    except traci.TraCIException:
        return None

    if route_index < 0 or route_index >= len(route):
        return None

    if route[route_index] != current_edge:
        try:
            route_index = route.index(current_edge)
        except ValueError:
            return None

    if route_index + 1 >= len(route):
        return None

    return route[route_index + 1]


def required_lane_change_distance(current_lane_index, target_lane_index):
    if current_lane_index is None or target_lane_index is None:
        return APPROACH_LANE_CHANGE_BASE_DISTANCE

    lane_delta = abs(target_lane_index - current_lane_index)
    return APPROACH_LANE_CHANGE_BASE_DISTANCE + APPROACH_LANE_CHANGE_DISTANCE_PER_LANE * lane_delta


def reachable_preferred_lanes(lane_info, outgoing_edge, movement, current_lane, distance_to_end):
    current_lane_index = lane_index_from_lane_id(current_lane)
    preferred = target_lanes_for_movement(lane_info, outgoing_edge, movement)
    reachable = []

    for lane_id in preferred:
        target_lane_index = lane_index_from_lane_id(lane_id)
        required_distance = required_lane_change_distance(current_lane_index, target_lane_index)

        if target_lane_index == current_lane_index or distance_to_end >= required_distance:
            reachable.append(lane_id)

    return reachable


def approach_lane_preference_weight(lane_info, lane_id, movement):
    lane_movements = set(lane_info["lane_to_movements"].get(lane_id, set()))
    weight = 1.0

    if lane_movements == {movement}:
        weight *= 8.0
    elif movement == "S" and "S" in lane_movements and "R" not in lane_movements:
        weight *= 4.0
    elif movement in lane_movements:
        weight *= 1.5

    # Strongly discourage straight cars from using the rightmost lane when any
    # other straight-capable lane was available.
    if movement == "S" and lane_index_from_lane_id(lane_id) == 0:
        weight *= 0.05

    return weight


def build_approach_options_for_vehicle(lane_info, current_lane, distance_to_end):
    options_by_group = defaultdict(list)

    for movement in MOVEMENT_ORDER:
        for outgoing_edge in sorted(lane_info["movement_to_edges"].get(movement, set())):
            candidate_lanes = reachable_preferred_lanes(
                lane_info=lane_info,
                outgoing_edge=outgoing_edge,
                movement=movement,
                current_lane=current_lane,
                distance_to_end=distance_to_end,
            )

            if not candidate_lanes:
                continue

            current_lane_index = lane_index_from_lane_id(current_lane)
            target_lane_index = choose_best_target_lane(candidate_lanes, current_lane_index)

            if target_lane_index is None:
                continue

            target_lane = None
            for lane_id in candidate_lanes:
                if lane_index_from_lane_id(lane_id) == target_lane_index:
                    target_lane = lane_id
                    break

            if target_lane is None:
                continue

            options_by_group[movement].append({
                "movement": movement,
                "outgoing_edge": outgoing_edge,
                "target_lane": target_lane,
                "target_lane_index": target_lane_index,
            })

    return options_by_group


def set_vehicle_route_for_approach_decision(
    veh_id,
    current_edge,
    outgoing_edge,
    previous_edge,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    args,
    recent_edges=None,
):
    try:
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
    except traci.TraCIException:
        return False

    if route_index < 0 or route_index >= len(route):
        route_prefix = [current_edge]
    else:
        route_prefix = [current_edge]

    continuation_counts = Counter()

    if outgoing_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
        return False

    if outgoing_edge in raw_graph:
        continuation = build_random_walk_route(
            start_edge=outgoing_edge,
            lookahead_edges=max(2, args.route_lookahead_edges - 1),
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            turn_counts=continuation_counts,
            args=args,
            previous_edge=current_edge,
            initial_recent_edges=tuple(recent_edges or ()) + (current_edge,),
        )
    else:
        continuation = [outgoing_edge]

    if not continuation or continuation[0] != outgoing_edge:
        continuation = [outgoing_edge]

    new_route = route_prefix + continuation

    # Remove accidental adjacent duplicates.
    cleaned = []
    for edge_id in new_route:
        if cleaned and cleaned[-1] == edge_id:
            continue
        cleaned.append(edge_id)

    if len(cleaned) < 2:
        return False

    try:
        traci.vehicle.setRoute(veh_id, cleaned)
        return True
    except traci.TraCIException:
        return False


def enforce_approach_target_lane(veh_id, target_lane_index, distance_to_end):
    try:
        current_lane = traci.vehicle.getLaneID(veh_id)
    except traci.TraCIException:
        return False

    current_lane_index = lane_index_from_lane_id(current_lane)

    if target_lane_index is None or target_lane_index == current_lane_index:
        return False

    if distance_to_end < required_lane_change_distance(current_lane_index, target_lane_index):
        return False

    try:
        traci.vehicle.changeLane(veh_id, target_lane_index, APPROACH_LANE_CHANGE_DURATION)
        return True
    except traci.TraCIException:
        return False


def apply_approach_turn_decision_to_vehicle(
    veh_id,
    turn_index,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    args,
    recent_edges=None,
):
    if not APPROACH_DECISION_INDEX:
        return False

    recent_edges = tuple(recent_edges or ())

    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = traci.lane.getLength(lane_id)
    except traci.TraCIException:
        return False

    if lane_id.startswith(":"):
        return False

    current_edge = lane_to_edge(lane_id)
    lane_info = APPROACH_DECISION_INDEX.get(current_edge)

    if current_edge is None or lane_info is None:
        return False

    distance_to_end = lane_len - lane_pos
    key = (veh_id, current_edge)

    existing = APPROACH_TURN_DECISIONS.get(key)
    if existing is not None:
        return enforce_approach_target_lane(
            veh_id=veh_id,
            target_lane_index=existing.get("target_lane_index"),
            distance_to_end=distance_to_end,
        )

    if distance_to_end > APPROACH_DECISION_MAX_DISTANCE_TO_END:
        return False

    if distance_to_end < APPROACH_DECISION_MIN_DISTANCE_TO_END:
        # Too late to safely impose a new movement/lane choice. SUMO should keep
        # following the existing route instead of making a dangerous last-second
        # lane change.
        return False

    options_by_group = build_approach_options_for_vehicle(
        lane_info=lane_info,
        current_lane=lane_id,
        distance_to_end=distance_to_end,
    )

    if not options_by_group:
        return False

    loop_safe_options_by_group = {}
    for group, options in options_by_group.items():
        filtered = candidate_edges_after_loop_filter(
            candidates=options,
            get_edge=lambda option: option.get("outgoing_edge"),
            recent_edges=recent_edges,
            previous_edge=None,
        )
        if filtered:
            loop_safe_options_by_group[group] = filtered

    decision_options_by_group = loop_safe_options_by_group or options_by_group

    movement = choose_turn_group(
        rng=rng,
        available_groups=decision_options_by_group,
        turn_counts=APPROACH_TURN_COUNTS,
        strict_split=not args.disable_strict_split,
    )

    if movement is None:
        return False

    try:
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
        previous_edge = route[route_index - 1] if route_index > 0 else None
    except traci.TraCIException:
        previous_edge = None

    weighted_options = []
    weights = []

    for option in decision_options_by_group[movement]:
        outgoing_edge = option["outgoing_edge"]

        if outgoing_edge == previous_edge:
            continue

        if outgoing_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
            continue

        base = successor_weight(
            current_edge=current_edge,
            next_edge=outgoing_edge,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        )
        lane_weight = approach_lane_preference_weight(
            lane_info=lane_info,
            lane_id=option["target_lane"],
            movement=movement,
        )
        weighted_options.append(option)
        weights.append(
            base
            * lane_weight
            * loop_avoidance_weight_multiplier(current_edge, outgoing_edge, recent_edges)
        )

    if not weighted_options:
        weighted_options = list(decision_options_by_group[movement])
        weights = [1.0] * len(weighted_options)

    option = weighted_choice(rng, weighted_options, weights)
    outgoing_edge = option["outgoing_edge"]

    route_ok = set_vehicle_route_for_approach_decision(
        veh_id=veh_id,
        current_edge=current_edge,
        outgoing_edge=outgoing_edge,
        previous_edge=previous_edge,
        turn_index=turn_index,
        raw_graph=raw_graph,
        edge_metadata=edge_metadata,
        core_edges=core_edges,
        rng=rng,
        args=args,
        recent_edges=recent_edges,
    )

    if not route_ok:
        return False

    APPROACH_TURN_COUNTS[movement] += 1
    APPROACH_TURN_DECISIONS[key] = {
        "movement": movement,
        "outgoing_edge": outgoing_edge,
        "target_lane": option["target_lane"],
        "target_lane_index": option["target_lane_index"],
        "time": current_sim_time(),
    }

    enforce_approach_target_lane(
        veh_id=veh_id,
        target_lane_index=option["target_lane_index"],
        distance_to_end=distance_to_end,
    )

    return True


def prune_approach_turn_decisions(active_ids):
    if len(APPROACH_TURN_DECISIONS) < APPROACH_DECISION_PRUNE_LIMIT:
        return

    active_ids = set(active_ids)
    for key in list(APPROACH_TURN_DECISIONS):
        veh_id, _ = key
        if veh_id not in active_ids:
            APPROACH_TURN_DECISIONS.pop(key, None)


def apply_turn_lane_preference_to_vehicle(veh_id):
    """
    Nudge vehicles into better approach lanes without changing routes/signals.

    Most important case:
      - planned movement is straight
      - current lane is shared right+straight
      - another lane on this same incoming edge is straight-only

    Then we request a lane change to the straight-only lane, so right-turning
    vehicles are less likely to be blocked by straight-through vehicles.
    """
    if not TURN_LANE_PREFERENCE_INDEX:
        return False

    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        speed = traci.vehicle.getSpeed(veh_id)
        waiting_time = traci.vehicle.getWaitingTime(veh_id)
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = traci.lane.getLength(lane_id)
    except traci.TraCIException:
        return False

    # Only do lane preference while the car is still moving and far enough
    # upstream. If the vehicle is already queued near the intersection, forcing
    # a lane change can block both lanes and create the deadlock seen in the GUI.
    distance_to_end = lane_len - lane_pos
    if speed < LANE_PREF_MIN_SPEED:
        return False
    if waiting_time > LANE_PREF_MAX_WAITING_TIME:
        return False
    if distance_to_end < LANE_PREF_MIN_DISTANCE_TO_END:
        return False

    current_edge = lane_to_edge(lane_id)

    if current_edge is None:
        return False

    lane_info = TURN_LANE_PREFERENCE_INDEX.get(current_edge)

    if not lane_info:
        return False

    outgoing_edge = planned_next_edge_for_vehicle(veh_id, current_edge)

    if outgoing_edge is None:
        return False

    movement = lane_info["edge_to_movement"].get(outgoing_edge)

    if movement not in TURN_PROBABILITIES:
        return False

    lane_to_movements = lane_info["lane_to_movements"]
    current_movements = set(lane_to_movements.get(lane_id, set()))
    preferred_lanes = target_lanes_for_movement(lane_info, outgoing_edge, movement)

    if not preferred_lanes:
        return False

    # If the current lane is already one of the preferred lanes, do nothing.
    if lane_id in preferred_lanes:
        return False

    # Avoid unnecessary lane changes. For straight vehicles, only force the
    # change when a better straight-only/no-right lane exists. For right/left,
    # only force if the current lane does not support that movement.
    if movement == "S":
        current_is_shared_right_straight = "S" in current_movements and "R" in current_movements
        current_supports_straight = "S" in current_movements
        preferred_has_true_straight_lane = any(
            set(lane_to_movements.get(lane, set())) == {"S"}
            for lane in preferred_lanes
        )

        if current_supports_straight and not current_is_shared_right_straight:
            return False

        if current_supports_straight and not preferred_has_true_straight_lane:
            return False

    elif movement in {"R", "L"}:
        if movement in current_movements:
            return False

    current_lane_index = lane_index_from_lane_id(lane_id)
    target_lane_index = choose_best_target_lane(preferred_lanes, current_lane_index)

    if target_lane_index is None or target_lane_index == current_lane_index:
        return False

    try:
        traci.vehicle.changeLane(veh_id, target_lane_index, TURN_LANE_CHANGE_DURATION)
        return True
    except traci.TraCIException:
        return False


def apply_turn_lane_preference_to_all_vehicles():
    changed = 0

    if not TURN_LANE_PREFERENCE_INDEX:
        return changed

    for veh_id in list(traci.vehicle.getIDList()):
        if apply_turn_lane_preference_to_vehicle(veh_id):
            changed += 1

    return changed


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

@lru_cache(maxsize=None)
def classify_edge_successor_movement(current_edge, next_edge):
    """
    Classify next_edge as S/L/R relative to current_edge.

    Uses SUMO's lane link direction (link[6]) which is reliable and fast.
    Iterates all lanes of current_edge and checks every link to find one
    whose toLane belongs to next_edge.
    """
    try:
        lane_count = traci.edge.getLaneNumber(current_edge)
    except traci.TraCIException:
        return None

    for lane_index in range(lane_count):
        in_lane = f"{current_edge}_{lane_index}"
        try:
            try:
                links = traci.lane.getLinks(in_lane, extended=True)
            except TypeError:
                links = traci.lane.getLinks(in_lane)
        except traci.TraCIException:
            continue

        for link in links:
            if not link or len(link) < 7:
                continue
            to_lane = link[0]
            # toLane belongs to next_edge if its edge prefix matches
            to_edge = to_lane.rsplit("_", 1)[0] if "_" in to_lane else to_lane
            if to_edge != next_edge:
                continue
            movement = sumo_link_direction_to_movement(link[6])
            if movement is not None:
                return movement

    return None


def choose_uncontrolled_successor(
    current_edge,
    previous_edge,
    raw_graph,
    edge_metadata,
    core_edges,
    rng,
    args,
    turn_counts=None,
    recent_edges=None,
):
    successors = raw_graph.get(current_edge, [])

    if not successors:
        return None

    recent_edges = tuple(recent_edges or ())
    non_backtrack_successors = [s for s in successors if s != previous_edge]
    preferred_successors = [
        s
        for s in non_backtrack_successors
        if not should_avoid_successor_for_loop(current_edge, s, recent_edges)
    ]

    # Use loop-safe candidates when possible, but never strand the route if the
    # network only exposes one successor.
    successors = preferred_successors or non_backtrack_successors or list(successors)

    # Group successors by turn movement (S/L/R) using geometry.
    groups = {"S": [], "L": [], "R": []}
    unclassified = []

    for successor in successors:
        if successor == previous_edge:
            continue
        movement = classify_edge_successor_movement(current_edge, successor)
        if movement in groups:
            groups[movement].append(successor)
        else:
            unclassified.append(successor)

    # Build available dict — only groups with options.
    available = {g: edges for g, edges in groups.items() if edges}

    if available and turn_counts is not None:
        # Pick a movement group honouring TURN_PROBABILITIES.
     
            
        group = choose_turn_group(
            rng=rng,
            available_groups=available,
            turn_counts=turn_counts,
            strict_split=True,
        )
        
        if group is not None:
            candidates = available[group]
            weights = [
                successor_weight(
                    current_edge=current_edge,
                    next_edge=s,
                    previous_edge=previous_edge,
                    edge_metadata=edge_metadata,
                    core_edges=core_edges,
                    args=args,
                ) * loop_avoidance_weight_multiplier(current_edge, s, recent_edges)
                for s in candidates
            ]
            chosen = weighted_choice(rng, candidates, weights)
            turn_counts[group] += 1
            return chosen
    
    # Fallback: weight all successors (including unclassified) normally.
    candidates = list(successors)
    weights = [
        successor_weight(
            current_edge=current_edge,
            next_edge=s,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        ) * loop_avoidance_weight_multiplier(current_edge, s, recent_edges)
        for s in candidates
    ]
    return weighted_choice(rng, candidates, weights)


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
    recent_edges=None,
):
    recent_edges = tuple(recent_edges or ())

    if current_edge in turn_index:
        tls_map = turn_index[current_edge]

        for tls_id in sorted(tls_map.keys()):
            options_by_group = tls_map[tls_id]

            available = {
                group: list(options)
                for group, options in options_by_group.items()
                if options
            }

            if not available:
                continue

            # First remove choices that immediately send the vehicle back into
            # its own recent path.  If any loop-safe movement groups remain,
            # choose the S/R/L group from those groups.  This is important: the
            # older version chose S/R/L first and only filtered inside that one
            # group, so it could still choose a cyclic straight/right movement
            # even when a non-cyclic left/right/straight option existed.
            loop_safe_available = {}
            for group, options in available.items():
                filtered = candidate_edges_after_loop_filter(
                    candidates=options,
                    get_edge=lambda option: option.get("outgoing_edge"),
                    recent_edges=recent_edges,
                    previous_edge=previous_edge,
                )
                if filtered:
                    loop_safe_available[group] = filtered

            decision_pool = loop_safe_available or available

            group = choose_turn_group(
                rng=rng,
                available_groups=decision_pool,
                turn_counts=turn_counts,
                strict_split=not args.disable_strict_split,
            )

            if group is None:
                continue

            options = list(decision_pool[group])
            weighted_options = []
            weights = []

            for option in options:
                outgoing_edge = option["outgoing_edge"]

                if outgoing_edge == previous_edge and len(options) > 1:
                    continue

                if outgoing_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
                    continue

                # Allow straight movements onto boundary/dead-end edges that
                # aren't in raw_graph — the vehicle will be recovered from
                # there by the normal recovery mechanism.
                if option["movement"] != "S" and outgoing_edge not in raw_graph:
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
                    ) * loop_avoidance_weight_multiplier(current_edge, outgoing_edge, recent_edges)
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
        turn_counts=turn_counts,
        recent_edges=recent_edges,
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
    initial_recent_edges=None,
):
    route = [start_edge]
    current_edge = start_edge
    initial_recent_edges = tuple(initial_recent_edges or ())

    for _ in range(lookahead_edges - 1):
        recent_edges = tuple((list(initial_recent_edges) + route)[-LOOP_MEMORY_EDGES:])
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
            recent_edges=recent_edges,
        )

        if next_edge is None:
            break

        if next_edge == current_edge:
            break

        # Do not knowingly create a repeated tail pattern such as
        # A -> B -> A -> B or A -> B -> C -> A -> B -> C.  If this happens,
        # stop the planned extension here; the next extension/recovery will
        # choose a different target instead of committing to the loop.
        tentative = route + [next_edge]
        if has_repeated_tail_pattern(list(initial_recent_edges) + tentative):
            break

        # Commit be3e31a behavior: do not append a dead-end edge to the tail
        # of a random-walk route unless it is the very first step. A route whose
        # tail is already unextendable causes unnecessary recovery calls and can
        # make cars pause at the end of an edge while the script repairs it.
        if next_edge not in raw_graph and len(route) > 1:
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
    recent_edges=None,
):
    core_list = [edge for edge in core_edges if edge not in LOOP_AVOIDANCE_PROTECTED_EDGES]
    if not core_list:
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

            if route_enters_hardcoded_loop_region(
                path_edges,
                allowed_first_edge=current_edge if current_edge in HARDCODED_NO_CRUISE_LOOP_EDGES else None,
            ):
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
                initial_recent_edges=tuple(recent_edges or ()) + tuple(path_edges),
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
    recent_edges=None,
):
    try:
        recent_edges = tuple(recent_edges or ())
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

        force_loop_reroute = (
            current_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
            or any(edge in LOOP_AVOIDANCE_PROTECTED_EDGES for edge in remaining[:3])
            or has_repeated_tail_pattern(recent_edges)
        )

        if len(remaining) >= min_remaining_edges and not force_loop_reroute:
            return False, False

        if force_loop_reroute:
            # Rewrite the route now instead of letting the vehicle follow a small
            # local cycle repeatedly. The new random-walk route will avoid recent
            # edges and strongly avoid the protected two-road loop.
            remaining = [current_edge]

        # Commit be3e31a behavior: trim dead-end edges off the tail of the
        # remaining route before extending it. This prevents a route from ending
        # at an edge that cannot be extended and avoids repeated recovery calls.
        while len(remaining) > 1 and remaining[-1] not in raw_graph:
            remaining = remaining[:-1]

        last_edge = remaining[-1]
        previous_edge = remaining[-2] if len(remaining) >= 2 else None

        if last_edge not in raw_graph:
            # Even current_edge has no successors, so full recovery is needed.
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
                recent_edges=recent_edges,
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
            initial_recent_edges=recent_edges,
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
                recent_edges=recent_edges,
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
        start_edge = choose_spawn_edge(
            sim_state=sim_state,
            start_edges_or_zones=start_edges,
            edge_metadata=edge_metadata,
            rng=rng,
        )

        if start_edge is None:
            continue

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


def print_turn_summary(turn_counts, title="Dynamic planned turn-decision summary"):
    total = turn_counts["S"] + turn_counts["R"] + turn_counts["L"]

    print()
    print(title)
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


# ---------------------------------------------------------------------------
# Ambulance management
# ---------------------------------------------------------------------------

def spawn_ambulance(sim_state, raw_graph, rng):
    edge_list = [e for e in raw_graph if not e.startswith(":")]
    if len(edge_list) < 2:
        return None

    # Cache edge positions in sim_state so we only do 1374 TraCI calls once
    if "edge_positions" not in sim_state:
        positions = {}
        for edge_id in edge_list:
            try:
                pos = traci.lane.getShape(f"{edge_id}_0")
                if pos:
                    x = sum(p[0] for p in pos) / len(pos)
                    y = sum(p[1] for p in pos) / len(pos)
                    positions[edge_id] = (x, y)
            except traci.TraCIException:
                continue
        sim_state["edge_positions"] = positions
        print(f"  [ambulance] cached positions for {len(positions)} edges", flush=True)

    edge_positions = sim_state["edge_positions"]
    positioned_edges = [e for e in edge_list if e in edge_positions]
    if len(positioned_edges) < 2:
        return None

    # Try to find an origin/dest pair that are far apart (>500m)
    for attempt in range(100):
        origin = rng.choice(positioned_edges)
        dest   = rng.choice(positioned_edges)
        if origin == dest:
            continue

        ox, oy = edge_positions[origin]
        dx, dy = edge_positions[dest]
        dist = ((ox - dx)**2 + (oy - dy)**2) ** 0.5

        # Require at least 500m apart so route is long enough to see
        if dist < 500:
            continue

        try:
            path = traci.simulation.findRoute(origin, dest, vType="ambulance")
            if not path or not path.edges or len(path.edges) < 5:
                continue

            amb_id   = f"ambulance_{sim_state['next_vehicle_id']}"
            route_id = f"ambulance_route_{sim_state['next_route_id']}"
            sim_state["next_vehicle_id"] += 1
            sim_state["next_route_id"]   += 1

            traci.route.add(route_id, list(path.edges))
            traci.vehicle.add(
                vehID=amb_id,
                routeID=route_id,
                typeID="ambulance",
                depart=str(traci.simulation.getTime()),
                departLane="best",
                departPos="random_free",
                departSpeed="0",
            )
            traci.vehicle.setColor(amb_id, AMBULANCE_COLOR)
            # Limit speed so it's visible (~40 km/h)
            traci.vehicle.setMaxSpeed(amb_id, 11.0)

            # Add large blue POIs at origin and destination so they're
            # visible even when zoomed all the way out
            ox, oy = edge_positions[origin]
            dx, dy = edge_positions[dest]
            poi_a = f"{amb_id}_A"
            poi_b = f"{amb_id}_B"
            try:
                traci.poi.add(poi_a, ox, oy, color=(0, 0, 255, 255),
                              poiType="ambulance_origin", layer=10, imgWidth=80, imgHeight=80)
                traci.poi.add(poi_b, dx, dy, color=(0, 200, 255, 255),
                              poiType="ambulance_dest",   layer=10, imgWidth=80, imgHeight=80)
            except traci.TraCIException:
                pass

            sim_state.setdefault("active_ambulances", {})[amb_id] = {
                "origin": origin, "dest": dest,
                "route_len": len(path.edges),
                "poi_a": poi_a, "poi_b": poi_b,
            }
            return amb_id
        except traci.TraCIException:
            continue

    return None


def update_ambulances(sim_state, raw_graph, rng, sim_time, args):
    active     = sim_state.setdefault("active_ambulances", {})
    next_spawn = sim_state.setdefault("next_ambulance_spawn", 0.0)
    if sim_time >= next_spawn:
        amb_id = spawn_ambulance(sim_state, raw_graph, rng)
        if amb_id:
            info = active[amb_id]
            print(f"  [ambulance] {amb_id}: {info['origin']} -> {info['dest']} ({info.get('route_len','?')} edges)", flush=True)
        sim_state["next_ambulance_spawn"] = sim_time + args.ambulance_interval
    current_ids = set(traci.vehicle.getIDList())
    for amb_id in list(active.keys()):
        if amb_id not in current_ids:
            print(f"  [ambulance] {amb_id} arrived", flush=True)
            info = active.pop(amb_id, {})
            # Remove POIs
            for poi_key in ("poi_a", "poi_b"):
                poi_id = info.get(poi_key)
                if poi_id:
                    try:
                        traci.poi.remove(poi_id)
                    except traci.TraCIException:
                        pass
            continue
        try:
            traci.vehicle.getSpeed(amb_id)
        except traci.TraCIException:
            active.pop(amb_id, None)


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

    # Expensive Python/TraCI work happens once per DECISION_INTERVAL, not once per SUMO step.
    # This keeps the same broad multi-region behavior while avoiding a full vehicle scan
    # every 0.5/1.0 simulation seconds.
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

    active_vehicle_ids = list(traci.vehicle.getIDList())
    prune_vehicle_edge_history(active_vehicle_ids)

    for veh_id in active_vehicle_ids:
        try:
            current_lane_for_history = traci.vehicle.getLaneID(veh_id)
            current_edge_for_history = lane_to_edge(current_lane_for_history)
        except traci.TraCIException:
            current_edge_for_history = None

        recent_edges = update_vehicle_edge_history(veh_id, current_edge_for_history)

        # Rescue vehicles from lanes that cannot legally reach their next routed edge.
        # This specifically prevents lane 417292872_0 from trapping vehicles at its end.
        if rescue_vehicle_from_unconnected_lane(
            veh_id=veh_id,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            turn_index=turn_index,
            rng=rng,
            turn_counts=turn_counts,
            args=args,
            recent_edges=recent_edges,
        ):
            recovered_total += 1

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
            recent_edges=recent_edges,
        )

        if extended:
            extended_total += 1

        if recovered:
            recovered_total += 1

        # Make the actual S/R/L movement decision as the vehicle approaches
        # the next intersection, then request a safe lane change toward the
        # best lane for that movement.
        apply_approach_turn_decision_to_vehicle(
            veh_id=veh_id,
            turn_index=turn_index,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            rng=rng,
            args=args,
            recent_edges=recent_edges,
        )

        # Keep straight-through vehicles out of shared right/straight lanes
        # whenever a dedicated straight/no-right lane exists on the same approach.
        apply_turn_lane_preference_to_vehicle(veh_id)

    prune_approach_turn_decisions(traci.vehicle.getIDList())

    for _ in range(num_steps):
        sim_time = traci.simulation.getTime()

        # Rescue lane/route mismatches frequently, but not with a full expensive
        # route-extension pass. This catches cars before they reach the end of
        # lanes like 417292872_0 that cannot serve their next routed edge.
        if sim_time >= sim_state.get("next_unconnected_lane_rescue_time", 0.0):
            recovered_total += rescue_unconnected_lanes_for_all_vehicles(
                active_ids=list(traci.vehicle.getIDList()),
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                turn_index=turn_index,
                rng=rng,
                turn_counts=turn_counts,
                args=args,
            )
            sim_state["next_unconnected_lane_rescue_time"] = sim_time + UNCONNECTED_LANE_RESCUE_INTERVAL

        # Lane preference must run more often than route extension. Otherwise a
        # straight vehicle can enter a shared right/straight lane and remain
        # there until it is too late to move.
        if sim_time >= sim_state.get("next_lane_pref_time", 0.0):
            apply_turn_lane_preference_to_all_vehicles()
            sim_state["next_lane_pref_time"] = sim_time + LANE_PREF_INTERVAL

        # Vehicle-level keep-clear / right-of-way gate.
        # Lights stay green according to the fixed cycle; cars decide whether
        # they have enough downstream space to enter the junction. This applies
        # to both signalized and unsignalized intersections.
        apply_keep_clear_and_right_of_way_to_all_vehicles()

        update_ambulances(sim_state, raw_graph, rng, traci.simulation.getTime(), args)

        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()

        for controller in controllers:
            if not controller.get("disabled"):
                update_controller_after_simstep(controller)

    return arrived, spawned_total, extended_total, recovered_total


def resolve_run_seed(user_seed):
    if user_seed is not None:
        return int(user_seed)

    seed = (
        time.time_ns()
        ^ (os.getpid() << 16)
        ^ random.SystemRandom().randrange(1, 2_147_483_647)
    ) % 2_147_483_647
    return seed or 1


def run_simulation(args):
    actual_seed = resolve_run_seed(args.seed)
    args.seed = actual_seed
    rng = random.Random(actual_seed)

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
        "--seed", str(args.seed),
        "--max-num-vehicles", str(args.max_vehicles),
        "--max-depart-delay", str(args.max_depart_delay),
        "--time-to-teleport", str(args.time_to_teleport),
        "--ignore-route-errors", "true",
        "--quit-on-end", "false",
        *QUIET_SUMO_ARGS,
    ]

    print()
    print("Starting realistic random-driving fixed-cycle simulation:")
    print(f"Random seed for this run: {args.seed}")
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
        raw_graph = remove_hardcoded_loop_region_from_graph(raw_graph)

        # IMPORTANT: do not restrict routing/spawning to only largest_loop_core(raw_graph).
        # On this map, that can be a small local subregion, which causes every car
        # to pile up in one place. Use the full drivable graph as the routing core
        # so vehicles can be spawned and recovered across the whole map.
        largest_core_edges = largest_loop_core(raw_graph)
        core_edges = set(raw_graph)

        print()
        print("Passenger successor graph:")
        print(f"  raw drivable edges with outgoing choices: {len(raw_graph)}")
        print(f"  largest loop-safe core edges:             {len(largest_core_edges)}")
        print(f"  global routing core edges used:           {len(core_edges)}")
        print(f"  total raw outgoing choices:               {sum(len(v) for v in raw_graph.values())}")

        # Spawn from the full drivable graph, then balance by geographic zones.
        # This avoids the previous regression where cars appeared only in one
        # small high-weight region of the map. Road weights still make major
        # roads more likely inside each zone.
        main_start_edge_candidates = list(raw_graph.keys())

        if not main_start_edge_candidates:
            raise RuntimeError("No valid start edges were found.")

        main_start_edges = build_spawn_zones(
            start_edges=main_start_edge_candidates,
            edge_metadata=edge_metadata,
            grid_size=args.spawn_grid_size,
        )

        if not main_start_edges:
            main_start_edges = main_start_edge_candidates

        turn_index = build_turn_decision_index(
            controllers=controllers,
            raw_graph=raw_graph,
        )

        global TURN_LANE_PREFERENCE_INDEX, APPROACH_DECISION_INDEX
        APPROACH_DECISION_INDEX = build_approach_decision_index(raw_graph)
        # Use the broader all-intersection approach index for lane preference too,
        # not only the traffic-light-controlled incoming edges.
        TURN_LANE_PREFERENCE_INDEX = APPROACH_DECISION_INDEX

        sim_state = {
            "next_vehicle_id": 0,
            "next_route_id": 0,
            "next_spawn_zone_index": 0,
            "next_lane_pref_time": 0.0,
            "next_unconnected_lane_rescue_time": 0.0,
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
        print(f"Main/connector start candidates: {len(main_start_edge_candidates)}")
        print(f"Spawn zones used:                {len(main_start_edges) if main_start_edges and isinstance(main_start_edges[0], dict) else 0}")
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
        print("  Decisions are now made near each intersection when a safe lane change is still possible.")
        print("  Straight vehicles prefer straight-only/no-right lanes and avoid the rightmost lane when possible.")
        print()
        print("Straight-movement fix:")
        print("  Uses SUMO lane-link directions when available.")
        print("  Turn options are built from the full drivable graph, not only the loop-safe core.")
        print()
        print("Realism fixes:")
        print("  Main roads are preferred over residential/service roads.")
        print("  Residential-to-residential cruising is penalized.")
        print("  Cars are spawned across geographic zones covering the full map.")
        print("  Cars are recovered to the full drivable graph, not one small core.")
        print("  The observed two-road local loop is hard-blocked from normal random routing.")
        print("  Unconnected-lane rescue prevents lane-end stops such as 417292872_0.")
        print("  Signal phases have randomized offsets and slight timing variation.")
        print("  Routes are planned far ahead to reduce last-second lane changes.")
        print("  Per-vehicle route memory prevents repeated cycle-like paths.")
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
                if sum(APPROACH_TURN_COUNTS.values()) > 0:
                    print_turn_summary(APPROACH_TURN_COUNTS, "Actual near-intersection S/R/L decisions")

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
        if sum(APPROACH_TURN_COUNTS.values()) > 0:
            print_turn_summary(APPROACH_TURN_COUNTS, "Actual near-intersection S/R/L decisions")

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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed. Omit this for a fresh randomized run each time. "
            "Pass a fixed value, e.g. --seed 7, to reproduce an exact scenario."
        ),
    )

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=MAX_ACTIVE_VEHICLE_CAP,
        help="Hard-capped at 750.",
    )

    parser.add_argument(
        "--target-vehicles",
        type=int,
        default=650,
        help="The script tries to keep about this many active cars.",
    )

    parser.add_argument(
        "--initial-vehicles",
        type=int,
        default=200,
        help="Cars spawned immediately at the start.",
    )

    parser.add_argument(
        "--spawn-batch",
        type=int,
        default=12,
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
        default=60,
        help="How many edges ahead each random-walk route is planned.",
    )

    parser.add_argument(
        "--min-remaining-edges",
        type=int,
        default=15,
        help="Extend a car's route when fewer than this many edges remain.",
    )

    parser.add_argument(
        "--recovery-attempts",
        type=int,
        default=20,
        help="Attempts to recover a vehicle back into the loop-safe driving core.",
    )

    parser.add_argument(
        "--spawn-grid-size",
        type=int,
        default=6,
        help="Geographic grid size used to spread vehicle spawning across the map.",
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
        default=1.0,
        help="Penalty for leaving the loop-safe driving core. Default 1.0 avoids pulling all cars into one region.",
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
        default=180,
        help="Last-resort gridlock breaker. Use -1 to disable teleporting entirely.",
    )

    parser.add_argument("--green-duration", type=float, default=30.0)
    parser.add_argument("--ambulance-interval", type=float, default=float(AMBULANCE_SPAWN_INTERVAL), help="Seconds between ambulance spawns. Default 120.")
    parser.add_argument(
        "--max-consecutive-straight",
        type=int,
        default=4,
        help="Accepted for compatibility with earlier anti-cycle versions; route-memory loop prevention remains active.",
    )
    parser.add_argument("--print-every", type=float, default=60.0)

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