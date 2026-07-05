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

# SUMO on macOS may start but then immediately close the TraCI connection if
# PROJ cannot find proj.db.  SUMO_HOME usually points at .../share/sumo, while
# proj.db is usually one directory over in .../share/proj.  Do not rely on an
# inherited empty/broken PROJ_DATA value; repair it before importing/running SUMO.
SUMO_SHARE_DIR = os.path.dirname(os.environ["SUMO_HOME"])
SUMO_ROOT_DIR = os.path.dirname(SUMO_SHARE_DIR)


def _proj_dir_is_valid(path):
    return bool(path) and os.path.exists(os.path.join(path, "proj.db"))


def _repair_proj_environment():
    current = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if _proj_dir_is_valid(current):
        os.environ["PROJ_DATA"] = current
        os.environ["PROJ_LIB"] = current
        return

    candidates = (
        os.path.join(SUMO_SHARE_DIR, "proj"),
        os.path.join(SUMO_ROOT_DIR, "share", "proj"),
        "/opt/homebrew/share/proj",
        "/usr/local/share/proj",
        os.path.join(os.environ["SUMO_HOME"], "proj"),
    )

    for proj_candidate in candidates:
        if _proj_dir_is_valid(proj_candidate):
            os.environ["PROJ_DATA"] = proj_candidate
            os.environ["PROJ_LIB"] = proj_candidate
            return


def _repair_fontconfig_environment():
    current = os.environ.get("FONTCONFIG_FILE")
    if current and os.path.exists(current):
        return

    candidates = (
        "/opt/X11/etc/fonts/fonts.conf",
        "/opt/homebrew/etc/fonts/fonts.conf",
        "/usr/local/etc/fonts/fonts.conf",
        os.path.join(SUMO_ROOT_DIR, "etc", "fonts", "fonts.conf"),
    )

    for fontconfig_file in candidates:
        if os.path.exists(fontconfig_file):
            os.environ["FONTCONFIG_FILE"] = fontconfig_file
            os.environ.setdefault("FONTCONFIG_PATH", os.path.dirname(fontconfig_file))
            return


_repair_proj_environment()
_repair_fontconfig_environment()

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

MAX_ACTIVE_VEHICLE_CAP = 3750
DEFAULT_SIM_END = 1_000_000_000.0

CAR_LENGTH = 4.8
CAR_WIDTH = 1.8
CAR_MIN_GAP = 2.5

# Ambulance / emergency-vehicle helper.
# Ambulances are allowed to proceed through red/yellow lights, but they should
# still use SUMO's collision avoidance and this script's keep-clear checks.
# Therefore, do not give them foe-ignore settings, do not forcibly move them
# onto occupied lanes, and do not apply an artificial low max-speed cap.
AMBULANCE_SPAWN_INTERVAL = 120.0
AMBULANCE_COLOR = (255, 50, 50, 255)
AMBULANCE_MIN_EUCLIDEAN_DISTANCE = 1500.0
AMBULANCE_MIN_ROUTE_DISTANCE = 1800.0
AMBULANCE_MIN_ROUTE_EDGES = 20
AMBULANCE_ROUTE_ATTEMPTS = 100
AMBULANCE_DEPART_LANE = "free"
AMBULANCE_DEPART_POS = "random_free"
AMBULANCE_POI_RADIUS = 250.0

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

# Mid-road lane-balancing helper.
# The existing lane-preference logic only fixes approach-lane choice near
# intersections. On long multi-lane roads, SUMO can still let most cars settle
# into one lane. This helper gently moves cruising vehicles from an overloaded
# lane to a nearby underused lane, but only when the target lane can still serve
# the vehicle's next routed edge and the vehicle is far enough from the next
# junction. This avoids unsafe last-second lane changes.
LANE_BALANCE_INTERVAL = 2.0
LANE_BALANCE_CHANGE_DURATION = 8.0
LANE_BALANCE_MIN_EDGE_LANES = 2
LANE_BALANCE_MIN_DISTANCE_FROM_START = 30.0
LANE_BALANCE_MIN_DISTANCE_TO_END = 90.0
LANE_BALANCE_MIN_SPEED = 1.0
LANE_BALANCE_MAX_WAITING_TIME = 3.0
LANE_BALANCE_MIN_IMBALANCE = 3
LANE_BALANCE_MAX_LANE_DELTA = 1
LANE_BALANCE_MIN_TIME_BETWEEN_CHANGES = 12.0
LANE_BALANCE_LAST_CHANGE = {}

# Intersection approach lane-change lock.
# Vehicles may still prepare for turns upstream, but once they are this close
# to the end of any ordinary road segment, neither this script nor SUMO's own
# lane-change model is allowed to move them into another lane. This prevents
# last-second turn-lane changes right before intersections. Signalized
# approaches are included and may use an equal-or-larger lock distance.
INTERSECTION_NO_LANE_CHANGE_DISTANCE = 100.0
INTERSECTION_LANE_PREP_DISTANCE = 320.0
TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE = 100.0
TRAFFIC_LIGHT_LANE_PREP_DISTANCE = 320.0
TRAFFIC_LIGHT_LOCKED_LANE_CHANGE_MODE = 0
TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE = 1621
TRAFFIC_LIGHT_APPROACH_LANES = set()
TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES = set()


# Origin-destination routing helper.
# The original route generator used a controlled random walk and explicitly
# chose S/R/L movement groups.  OD mode instead chooses a realistic origin and a
# far-away destination, asks SUMO for the fastest legal route, and lets the
# resulting route naturally determine whether each intersection movement is
# straight, right, or left.  Local roads are mostly used as trip endpoints or
# access roads, while the middle of most trips is encouraged to stay on faster
# main/connector roads.
ROUTING_MODE_OD = "od"
ROUTING_MODE_RANDOM_WALK = "random-walk"
OD_BOUNDARY_MARGIN_FRACTION = 0.13
OD_MIN_EUCLIDEAN_DISTANCE = 900.0
OD_MIN_ROUTE_DISTANCE = 1200.0
OD_MIN_ZONE_SEPARATION = 2
OD_ROUTE_ATTEMPTS = 120
OD_MAX_LOCAL_MIDDLE_FRACTION = 0.35
OD_LOCAL_MIDDLE_TRIM_EDGES = 2
OD_THROUGH_TRIP_PROBABILITY = 0.72
OD_ACCESS_TRIP_PROBABILITY = 0.23
OD_LONG_LOCAL_TRIP_PROBABILITY = 0.05
OD_RANDOM_WALK_FALLBACK = True
OD_MIN_EDGE_LENGTH = 20.0
OD_DEPART_LANE = "free"

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

# Unjustified-stop watchdog.
# A vehicle should not remain stopped in the middle of a normal road unless
# there is a clear reason: a leader in front, an upcoming junction/traffic
# light, a route/lane mismatch being repaired, or the end of its route. This
# watchdog releases stale speed/lane-change overrides and repairs routes for
# vehicles that are stopped on free road segments for no valid reason.
UNJUSTIFIED_STOP_WATCHDOG_ENABLED = True
UNJUSTIFIED_STOP_CHECK_INTERVAL = 1.0
UNJUSTIFIED_STOP_SPEED = 0.20
UNJUSTIFIED_STOP_MIN_TIME = 3.0
UNJUSTIFIED_STOP_ACTION_COOLDOWN = 6.0
UNJUSTIFIED_STOP_LEADER_DISTANCE = 24.0
UNJUSTIFIED_STOP_JUNCTION_DISTANCE = 28.0
UNJUSTIFIED_STOP_ROUTE_END_DISTANCE = 18.0
UNJUSTIFIED_STOP_START_DISTANCE = 10.0
UNJUSTIFIED_STOP_NUDGE_SPEED = 2.5
UNJUSTIFIED_STOP_NUDGE_DURATION = 1.0
UNJUSTIFIED_STOP_ROUTE_REPAIR_ATTEMPTS = 30
UNJUSTIFIED_STOP_TRACKING = {}
UNJUSTIFIED_STOP_LAST_ACTION = {}

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


def is_ambulance(veh_id):
    return str(veh_id).startswith("ambulance_")


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
    """recent_edges may be any iterable. Callers that invoke this repeatedly
    for many candidate edges against the *same* recent_edges should convert it
    to a set once beforehand and pass that set in, to avoid rebuilding the
    same set on every call (this function will reuse a set as-is without
    copying)."""
    if not next_edge:
        return False

    if next_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
        return True

    if (
        current_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
        and next_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
    ):
        return True

    if recent_edges:
        recent_set = recent_edges if isinstance(recent_edges, (set, frozenset)) else set(recent_edges)
        if next_edge in recent_set:
            return True

    return False


def loop_avoidance_weight_multiplier(current_edge, next_edge, recent_edges=None):
    """See should_avoid_successor_for_loop docstring re: recent_edges type."""
    multiplier = 1.0

    if next_edge in HARDCODED_NO_CRUISE_LOOP_EDGES:
        multiplier *= 1e-9

    if (
        current_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
        and next_edge in LOOP_AVOIDANCE_PROTECTED_EDGES
    ):
        multiplier *= LOOP_PROTECTED_TRANSITION_PENALTY

    if recent_edges:
        recent_set = recent_edges if isinstance(recent_edges, (set, frozenset)) else set(recent_edges)
        if next_edge in recent_set:
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


def rebuild_traffic_light_approach_lanes(controllers):
    """Cache all lanes on edges that feed a controlled traffic light.

    The controller's all_in_lanes set gives the exact signalized incoming
    lanes. We expand that to all lanes on the same incoming edge so a vehicle
    cannot make a last-second lane change from one parallel approach lane into
    another right at the signal.
    """
    global TRAFFIC_LIGHT_APPROACH_LANES

    approach_lanes = set()

    for controller in controllers:
        for lane_id in controller.get("all_in_lanes", set()):
            if not lane_id or lane_id.startswith(":"):
                continue

            approach_lanes.add(lane_id)
            edge_id = lane_to_edge(lane_id)

            if edge_id is not None:
                approach_lanes.update(cached_edge_lanes(edge_id))

    TRAFFIC_LIGHT_APPROACH_LANES = approach_lanes

    print()
    print("Intersection lane-change lock:")
    print(f"  no lane changes within: {INTERSECTION_NO_LANE_CHANGE_DISTANCE:.0f} m of any intersection approach")
    print(f"  signalized approaches use at least: {TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE:.0f} m")
    print(f"  route/lane preparation begins up to: {TRAFFIC_LIGHT_LANE_PREP_DISTANCE:.0f} m upstream")
    print(f"  signalized approach lanes cached:   {len(TRAFFIC_LIGHT_APPROACH_LANES)}")


def traffic_light_no_lane_change_distance_for_lane(lane_id):
    """Distance from lane end where lane changes are forbidden.

    Despite the historical function name, this now applies to every normal
    intersection approach. Traffic-light approaches are a subset and may use a
    larger lock distance. Internal junction lanes are excluded.
    """
    if not lane_id or lane_id.startswith(":"):
        return 0.0

    edge_id = lane_to_edge(lane_id)
    if edge_id is None or edge_id.startswith(":"):
        return 0.0

    distance = INTERSECTION_NO_LANE_CHANGE_DISTANCE

    if lane_id in TRAFFIC_LIGHT_APPROACH_LANES:
        distance = max(distance, TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE)

    return max(0.0, distance)


def inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end=None):
    lock_distance = traffic_light_no_lane_change_distance_for_lane(lane_id)
    if lock_distance <= 0.0:
        return False

    if distance_to_end is None:
        return False

    return distance_to_end <= lock_distance


def cleanup_traffic_light_lane_change_locks(active_ids):
    active_ids = set(active_ids)
    for veh_id in list(TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES):
        if veh_id not in active_ids:
            TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)


def apply_traffic_light_lane_change_lock_to_vehicle(veh_id):
    """Disable autonomous lane changes close to intersections.

    This is separate from the script's explicit changeLane() guards. Without
    this, SUMO's internal lane-change model can still decide to merge at the
    last moment even when our own helpers stop requesting lane changes.
    """
    try:
        lane_id = traci.vehicle.getLaneID(veh_id)

        if not lane_id or lane_id.startswith(":"):
            if veh_id in TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES:
                traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
                TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
            return False

        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = cached_lane_length(lane_id)
    except traci.TraCIException:
        TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
        return False

    distance_to_end = lane_len - lane_pos

    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        try:
            traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_LOCKED_LANE_CHANGE_MODE)
            TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.add(veh_id)
            return True
        except traci.TraCIException:
            return False

    if veh_id in TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES:
        try:
            traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        except traci.TraCIException:
            pass
        TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)

    return False


def apply_traffic_light_lane_change_lock_to_all_vehicles():
    active_ids = list(traci.vehicle.getIDList())
    cleanup_traffic_light_lane_change_locks(active_ids)

    locked = 0
    for veh_id in active_ids:
        if apply_traffic_light_lane_change_lock_to_vehicle(veh_id):
            locked += 1

    return locked


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
        # Fallback path only: SUMO's route_index didn't match current_edge
        # (can happen briefly after a route rewrite). This is an O(n) scan
        # over the vehicle's route, but it's unavoidable here since `route`
        # is fetched fresh from TraCI each call and isn't a structure we can
        # index into persistently. Should be rare in steady state.
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

    recent_edges_set = set(recent_edges or ())
    candidates = [edge for edge in outgoing_edges if edge != previous_edge] or outgoing_edges
    weights = [
        successor_weight(
            current_edge=current_edge,
            next_edge=edge,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        ) * loop_avoidance_weight_multiplier(current_edge, edge, recent_edges_set)
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
    if is_ambulance(veh_id):
        return False

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

    # Never force a lane change inside the traffic-light no-change zone. If this
    # lane cannot serve the route that close to the signal, rewrite the route
    # to an outgoing edge reachable from the current lane instead.
    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
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
    if is_ambulance(veh_id):
        return False

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


def cleanup_unjustified_stop_tracking(active_ids):
    active_ids = set(active_ids)
    for veh_id in list(UNJUSTIFIED_STOP_TRACKING):
        if veh_id not in active_ids:
            UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
    for veh_id in list(UNJUSTIFIED_STOP_LAST_ACTION):
        if veh_id not in active_ids:
            UNJUSTIFIED_STOP_LAST_ACTION.pop(veh_id, None)


def vehicle_has_close_leader(veh_id, lookahead=UNJUSTIFIED_STOP_LEADER_DISTANCE):
    try:
        leader = traci.vehicle.getLeader(veh_id, lookahead)
    except traci.TraCIException:
        return False

    if leader is None:
        return False

    try:
        _leader_id, gap = leader
        return gap <= lookahead
    except Exception:
        return True


def vehicle_route_end_is_near(veh_id, current_edge, distance_to_end):
    try:
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
    except traci.TraCIException:
        return False

    if route_index < 0 or route_index >= len(route):
        return False

    # If the current edge is the final route edge and the car is close to the
    # end of it, stopping can be legitimate because the trip is ending.
    if route_index == len(route) - 1 and route[route_index] == current_edge:
        return distance_to_end <= UNJUSTIFIED_STOP_ROUTE_END_DISTANCE

    return False


def has_valid_reason_to_be_stopped(veh_id, lane_id, current_edge, lane_pos, distance_to_end):
    if veh_id in KEEP_CLEAR_HELD_VEHICLES:
        return True

    if not lane_id or lane_id.startswith(":"):
        return True

    # Newly inserted vehicles can briefly be motionless while SUMO finds a free
    # gap. Do not immediately kick them at the lane start.
    if lane_pos <= UNJUSTIFIED_STOP_START_DISTANCE:
        return True

    # A car behind another car is allowed to stop. This covers normal queues,
    # including long queues that extend far upstream from a red light.
    if vehicle_has_close_leader(veh_id):
        return True

    # The first vehicle approaching a signalized stop line can legitimately
    # stop inside the signal approach zone. For unsignalized junctions, use a
    # smaller generic junction buffer.
    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        return True

    if distance_to_end <= UNJUSTIFIED_STOP_JUNCTION_DISTANCE:
        return True

    if vehicle_route_end_is_near(veh_id, current_edge, distance_to_end):
        return True

    try:
        if traci.vehicle.getStopState(veh_id) != 0:
            return True
    except traci.TraCIException:
        pass

    return False


def build_od_recovery_route_from_current_edge(current_edge, sim_state, edge_metadata, rng, args):
    context = sim_state.get("od_context")
    if not context or current_edge not in context.get("all", []):
        return None

    destination_pool = context.get("boundary") or context.get("main_like") or context.get("all") or []
    if not destination_pool:
        return None

    far_destinations = [
        edge_id
        for edge_id in destination_pool
        if edge_id != current_edge
        and edge_distance(current_edge, edge_id, edge_metadata) >= max(300.0, args.od_min_euclidean_distance * 0.5)
    ]
    if not far_destinations:
        far_destinations = [edge_id for edge_id in context.get("all", []) if edge_id != current_edge]

    for _ in range(UNJUSTIFIED_STOP_ROUTE_REPAIR_ATTEMPTS):
        destination = weighted_edge_choice(rng, far_destinations, edge_metadata)
        if destination is None:
            return None

        try:
            path = fastest_sumo_route(current_edge, destination)
        except traci.TraCIException:
            continue

        route_edges = list(getattr(path, "edges", []) or [])
        if len(route_edges) < 2:
            continue
        if route_edges[0] != current_edge:
            continue
        if route_enters_hardcoded_loop_region(route_edges):
            continue
        if route_distance(route_edges, edge_metadata) < max(300.0, args.od_min_route_distance * 0.4):
            continue

        return route_edges

    return None


def repair_unjustified_stop_route(
    veh_id,
    lane_id,
    current_edge,
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    sim_state,
    args,
):
    next_edge = planned_next_edge_from_route(veh_id, current_edge)

    # First fix the common concrete cause: this exact lane cannot reach the
    # vehicle's next routed edge. This is the same type of failure as the known
    # 417292872_0 trap, but handled generically.
    if next_edge is not None and not next_edge.startswith(":"):
        if not lane_has_connection_to_edge(lane_id, next_edge):
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
                return True

    # If the route has effectively ended in the middle of the map, give this
    # vehicle a new fastest OD route from its current edge. This keeps OD mode
    # realistic while avoiding mid-road dead ends.
    if use_od_routing(args):
        route_edges = build_od_recovery_route_from_current_edge(
            current_edge=current_edge,
            sim_state=sim_state,
            edge_metadata=edge_metadata,
            rng=rng,
            args=args,
        )
        if route_edges:
            try:
                traci.vehicle.setRoute(veh_id, route_edges)
                return True
            except traci.TraCIException:
                pass

    # Last resort: use the existing recovery helper. This is only used for
    # abnormal stopped vehicles, not normal OD trip generation.
    return recover_vehicle_route(
        veh_id=veh_id,
        current_edge=current_edge,
        raw_graph=raw_graph,
        edge_metadata=edge_metadata,
        core_edges=core_edges,
        turn_index=turn_index,
        rng=rng,
        turn_counts=turn_counts,
        args=args,
        recent_edges=vehicle_recent_edges(veh_id),
    )


def nudge_unjustifiably_stopped_vehicle(veh_id):
    try:
        max_speed = traci.vehicle.getMaxSpeed(veh_id)
        target_speed = min(max_speed, UNJUSTIFIED_STOP_NUDGE_SPEED)
        if target_speed <= 0:
            target_speed = UNJUSTIFIED_STOP_NUDGE_SPEED
        traci.vehicle.slowDown(veh_id, target_speed, UNJUSTIFIED_STOP_NUDGE_DURATION)
        return True
    except (traci.TraCIException, AttributeError):
        # Fall back to clearing any forced stop/speed override. Avoid a
        # persistent positive setSpeed() command because that can be unsafe.
        return safe_vehicle_set_speed(veh_id, -1)


def apply_unjustified_stop_watchdog_to_vehicle(
    veh_id,
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    sim_state,
    args,
):
    if is_ambulance(veh_id):
        return False

    if not getattr(args, "unjustified_stop_watchdog", UNJUSTIFIED_STOP_WATCHDOG_ENABLED):
        return False

    now = current_sim_time()

    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        if not lane_id or lane_id.startswith(":"):
            UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
            return False

        current_edge = lane_to_edge(lane_id)
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = cached_lane_length(lane_id)
        speed = traci.vehicle.getSpeed(veh_id)
    except traci.TraCIException:
        UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
        return False

    if current_edge is None or lane_len <= 0.0:
        UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
        return False

    distance_to_end = lane_len - lane_pos

    stop_speed = getattr(args, "unjustified_stop_speed", UNJUSTIFIED_STOP_SPEED)
    min_time = getattr(args, "unjustified_stop_min_time", UNJUSTIFIED_STOP_MIN_TIME)

    if speed > stop_speed:
        UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
        return False

    if has_valid_reason_to_be_stopped(veh_id, lane_id, current_edge, lane_pos, distance_to_end):
        UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
        return False

    first_seen = UNJUSTIFIED_STOP_TRACKING.setdefault(veh_id, now)
    if now - first_seen < min_time:
        return False

    if now - UNJUSTIFIED_STOP_LAST_ACTION.get(veh_id, -1e9) < UNJUSTIFIED_STOP_ACTION_COOLDOWN:
        return False

    # Clear anything that our script may have done first: stale keep-clear
    # speed holds, old lane-change locks, or a previous forced speed.
    release_keep_clear_vehicle(veh_id)
    try:
        if not inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
            traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
            TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
    except traci.TraCIException:
        pass

    repaired = repair_unjustified_stop_route(
        veh_id=veh_id,
        lane_id=lane_id,
        current_edge=current_edge,
        raw_graph=raw_graph,
        edge_metadata=edge_metadata,
        core_edges=core_edges,
        turn_index=turn_index,
        rng=rng,
        turn_counts=turn_counts,
        sim_state=sim_state,
        args=args,
    )

    if not repaired:
        nudge_unjustifiably_stopped_vehicle(veh_id)

    UNJUSTIFIED_STOP_LAST_ACTION[veh_id] = now
    UNJUSTIFIED_STOP_TRACKING.pop(veh_id, None)
    return True


def apply_unjustified_stop_watchdog_to_all_vehicles(
    raw_graph,
    edge_metadata,
    core_edges,
    turn_index,
    rng,
    turn_counts,
    sim_state,
    args,
):
    active_ids = list(traci.vehicle.getIDList())
    cleanup_unjustified_stop_tracking(active_ids)

    fixed = 0
    for veh_id in active_ids:
        if apply_unjustified_stop_watchdog_to_vehicle(
            veh_id=veh_id,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            turn_index=turn_index,
            rng=rng,
            turn_counts=turn_counts,
            sim_state=sim_state,
            args=args,
        ):
            fixed += 1

    return fixed


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
            "length": 0.0,
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
                    length = float(edge.getLength())
                except Exception:
                    length = 0.0

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
                        "length": length,
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
                lengths = []
                points = []

                for lane_index in range(lane_count):
                    lane_id = f"{edge_id}_{lane_index}"
                    speeds.append(traci.lane.getMaxSpeed(lane_id))
                    lengths.append(traci.lane.getLength(lane_id))
                    points.extend(traci.lane.getShape(lane_id))

                if speeds:
                    item["speed"] = max(speeds)

                if lengths and item.get("length", 0.0) <= 0.0:
                    item["length"] = max(lengths)

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


def edge_length(edge_id, edge_metadata):
    item = edge_metadata.get(edge_id, {})
    length = float(item.get("length", 0.0) or 0.0)
    if length > 0.0:
        return length

    lanes = cached_edge_lanes(edge_id)
    if not lanes:
        return 0.0

    lengths = [cached_lane_length(lane_id) for lane_id in lanes]
    lengths = [length for length in lengths if length > 0.0]
    return max(lengths) if lengths else 0.0


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
# Origin-destination route generation
# ============================================================

def use_od_routing(args):
    return getattr(args, "routing_mode", ROUTING_MODE_OD) == ROUTING_MODE_OD


def route_distance(route_edges, edge_metadata):
    return sum(edge_length(edge_id, edge_metadata) for edge_id in route_edges)


def route_travel_time_estimate(route_edges, edge_metadata):
    total = 0.0
    for edge_id in route_edges:
        length = edge_length(edge_id, edge_metadata)
        speed = max(0.1, float(edge_metadata.get(edge_id, {}).get("speed", 0.0) or 0.0))
        total += length / speed
    return total


def edge_distance(edge_a, edge_b, edge_metadata):
    ax, ay = edge_xy(edge_a, edge_metadata)
    bx, by = edge_xy(edge_b, edge_metadata)
    return math.hypot(ax - bx, ay - by)


def build_od_zones(edges, edge_metadata, grid_size):
    if not edges:
        return [], {}, (0.0, 0.0, 1.0, 1.0)

    grid_size = max(1, int(grid_size))
    xs = [edge_xy(edge_id, edge_metadata)[0] for edge_id in edges]
    ys = [edge_xy(edge_id, edge_metadata)[1] for edge_id in edges]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    raw_zones = defaultdict(list)
    edge_to_zone = {}

    for edge_id in edges:
        x, y = edge_xy(edge_id, edge_metadata)
        gx = min(grid_size - 1, max(0, int((x - min_x) / width * grid_size)))
        gy = min(grid_size - 1, max(0, int((y - min_y) / height * grid_size)))
        zone_id = (gx, gy)
        raw_zones[zone_id].append(edge_id)
        edge_to_zone[edge_id] = zone_id

    zones = []
    for zone_id, zone_edges in sorted(raw_zones.items()):
        zones.append({"id": zone_id, "edges": sorted(zone_edges)})

    return zones, edge_to_zone, (min_x, min_y, max_x, max_y)


def zone_separation(edge_a, edge_b, edge_to_zone):
    za = edge_to_zone.get(edge_a)
    zb = edge_to_zone.get(edge_b)
    if za is None or zb is None:
        return 0
    return abs(za[0] - zb[0]) + abs(za[1] - zb[1])


def weighted_edge_choice(rng, edges, edge_metadata):
    edges = list(edges)
    if not edges:
        return None

    weights = []
    for edge_id in edges:
        item = edge_metadata.get(edge_id, {})
        category = item.get("category", "unknown")
        weight = edge_base_weight(edge_id, edge_metadata)

        # Prefer useful driving corridors for OD endpoints, but still allow
        # local roads as trip origins/destinations in access-trip cases.
        if category == "main":
            weight *= 2.5
        elif category == "connector":
            weight *= 1.4
        elif category == "local":
            weight *= 0.45

        weights.append(max(0.001, weight))

    return weighted_choice(rng, edges, weights)


def choose_zone_balanced_edge(sim_state, pool, context, edge_metadata, rng):
    pool = set(pool)
    zones = [zone for zone in context["zones"] if pool.intersection(zone["edges"])]

    if not zones:
        return weighted_edge_choice(rng, pool, edge_metadata)

    start = sim_state.get("next_od_origin_zone_index", 0)
    zone = zones[start % len(zones)]
    sim_state["next_od_origin_zone_index"] = start + 1
    zone_edges = sorted(pool.intersection(zone["edges"]))
    return weighted_edge_choice(rng, zone_edges, edge_metadata)


def is_boundary_edge(edge_id, context, edge_metadata, margin_fraction):
    min_x, min_y, max_x, max_y = context["bounds"]
    x, y = edge_xy(edge_id, edge_metadata)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    margin_x = width * margin_fraction
    margin_y = height * margin_fraction

    return (
        x <= min_x + margin_x
        or x >= max_x - margin_x
        or y <= min_y + margin_y
        or y >= max_y - margin_y
    )


def build_od_context(valid_edges, raw_graph, edge_metadata, args):
    candidates = []
    for edge_id in valid_edges:
        if edge_id in HARDCODED_NO_CRUISE_LOOP_EDGES:
            continue
        if edge_id.startswith(":"):
            continue
        if edge_length(edge_id, edge_metadata) < args.od_min_edge_length:
            continue
        candidates.append(edge_id)

    zones, edge_to_zone, bounds = build_od_zones(
        candidates,
        edge_metadata,
        args.spawn_grid_size,
    )

    context = {
        "all": sorted(candidates),
        "raw_graph_edges": set(raw_graph),
        "zones": zones,
        "edge_to_zone": edge_to_zone,
        "bounds": bounds,
    }

    main_edges = []
    connector_edges = []
    local_edges = []

    for edge_id in candidates:
        category = edge_category(edge_id, edge_metadata)
        if category == "main":
            main_edges.append(edge_id)
        elif category == "connector":
            connector_edges.append(edge_id)
        elif category == "local":
            local_edges.append(edge_id)

    main_like = sorted(set(main_edges) | set(connector_edges))
    boundary_edges = [
        edge_id
        for edge_id in main_like or candidates
        if is_boundary_edge(edge_id, context, edge_metadata, args.od_boundary_margin_fraction)
    ]

    context.update({
        "main": sorted(main_edges),
        "connector": sorted(connector_edges),
        "local": sorted(local_edges),
        "main_like": main_like,
        "boundary": sorted(boundary_edges) or main_like or sorted(candidates),
    })

    print()
    print("Origin-destination routing:")
    print("  routing mode:               fastest OD routes")
    print(f"  candidate OD edges:          {len(context['all'])}")
    print(f"  boundary/main-like edges:    {len(context['boundary'])}")
    print(f"  main edges:                  {len(context['main'])}")
    print(f"  connector edges:             {len(context['connector'])}")
    print(f"  local endpoint edges:        {len(context['local'])}")
    print(f"  minimum route distance:      {args.od_min_route_distance:.0f} m")
    print(f"  minimum Euclidean distance:  {args.od_min_euclidean_distance:.0f} m")
    print(f"  minimum zone separation:     {args.od_min_zone_separation}")
    print("  S/R/L movements are no longer directly forced in OD mode.")

    return context


def choose_od_trip_type(rng, args):
    through = max(0.0, args.od_through_trip_probability)
    access = max(0.0, args.od_access_trip_probability)
    local = max(0.0, args.od_long_local_trip_probability)
    total = through + access + local

    if total <= 0.0:
        return "through"

    r = rng.random() * total
    if r < through:
        return "through"
    if r < through + access:
        return "access"
    return "long_local"


def od_pair_far_enough(origin, destination, context, edge_metadata, args):
    if not origin or not destination or origin == destination:
        return False

    if edge_distance(origin, destination, edge_metadata) < args.od_min_euclidean_distance:
        return False

    if zone_separation(origin, destination, context["edge_to_zone"]) < args.od_min_zone_separation:
        return False

    return True


def choose_od_pair(sim_state, context, edge_metadata, rng, args):
    trip_type = choose_od_trip_type(rng, args)

    if trip_type == "through":
        origin_pool = context["boundary"] or context["main_like"] or context["all"]
        destination_pool = context["boundary"] or context["main_like"] or context["all"]

    elif trip_type == "access":
        local_pool = context["local"] or context["connector"] or context["all"]
        main_pool = context["boundary"] or context["main_like"] or context["all"]
        if rng.random() < 0.5:
            origin_pool = local_pool
            destination_pool = main_pool
            trip_type = "local_to_far"
        else:
            origin_pool = main_pool
            destination_pool = local_pool
            trip_type = "far_to_local"

    else:
        # Long local trips are kept rare, but when they happen they still must
        # cross a meaningful portion of the map. This represents car trips that
        # begin or end in neighborhoods, not tiny blocks that people would walk.
        local_pool = context["local"] or context["connector"] or context["all"]
        origin_pool = local_pool
        destination_pool = local_pool
        trip_type = "long_local"

    origin = choose_zone_balanced_edge(sim_state, origin_pool, context, edge_metadata, rng)
    if origin is None:
        return None, None, trip_type

    far_destinations = [
        edge_id
        for edge_id in destination_pool
        if od_pair_far_enough(origin, edge_id, context, edge_metadata, args)
    ]

    if not far_destinations:
        far_destinations = [edge_id for edge_id in context["all"] if edge_id != origin]

    destination = weighted_edge_choice(rng, far_destinations, edge_metadata)
    return origin, destination, trip_type


def fastest_sumo_route(origin_edge, destination_edge):
    try:
        return traci.simulation.findRoute(
            origin_edge,
            destination_edge,
            "global_car",
            current_sim_time(),
        )
    except TypeError:
        try:
            return traci.simulation.findRoute(origin_edge, destination_edge, "global_car")
        except TypeError:
            return traci.simulation.findRoute(origin_edge, destination_edge)


def route_local_middle_fraction(route_edges, edge_metadata, trim_edges):
    if not route_edges:
        return 1.0

    total_distance = route_distance(route_edges, edge_metadata)
    if total_distance <= 0.0:
        return 1.0

    start = min(len(route_edges), max(0, trim_edges))
    end = max(start, len(route_edges) - max(0, trim_edges))
    middle_edges = route_edges[start:end]

    local_distance = 0.0
    for edge_id in middle_edges:
        if edge_category(edge_id, edge_metadata) == "local":
            local_distance += edge_length(edge_id, edge_metadata)

    return local_distance / total_distance


def od_route_is_reasonable(route_edges, origin, destination, trip_type, context, edge_metadata, args):
    if not route_edges or len(route_edges) < 2:
        return False
    if route_edges[0] != origin:
        return False
    if route_edges[-1] != destination:
        return False
    if route_enters_hardcoded_loop_region(route_edges):
        return False

    distance = route_distance(route_edges, edge_metadata)
    if distance < args.od_min_route_distance:
        return False

    local_middle_fraction = route_local_middle_fraction(
        route_edges,
        edge_metadata,
        args.od_local_middle_trim_edges,
    )

    # Through traffic should not cut through neighborhoods. Access trips may
    # touch local roads at the beginning/end, but the route middle should still
    # be mostly main/connector roads.
    max_fraction = args.od_max_local_middle_fraction
    if trip_type == "through":
        max_fraction = min(max_fraction, 0.25)

    if local_middle_fraction > max_fraction:
        return False

    return True


def build_od_route(sim_state, context, raw_graph, edge_metadata, rng, args):
    for _ in range(args.od_route_attempts):
        origin, destination, trip_type = choose_od_pair(
            sim_state=sim_state,
            context=context,
            edge_metadata=edge_metadata,
            rng=rng,
            args=args,
        )

        if origin is None or destination is None:
            continue

        try:
            path = fastest_sumo_route(origin, destination)
        except traci.TraCIException:
            continue

        route_edges = list(getattr(path, "edges", []) or [])
        if not od_route_is_reasonable(
            route_edges=route_edges,
            origin=origin,
            destination=destination,
            trip_type=trip_type,
            context=context,
            edge_metadata=edge_metadata,
            args=args,
        ):
            continue

        return route_edges, {
            "trip_type": trip_type,
            "origin": origin,
            "destination": destination,
            "distance": route_distance(route_edges, edge_metadata),
            "time": float(getattr(path, "travelTime", route_travel_time_estimate(route_edges, edge_metadata)) or 0.0),
        }

    return None, None


def count_route_movements(route_edges):
    counts = Counter()
    for current_edge, next_edge in zip(route_edges, route_edges[1:]):
        movement = classify_edge_successor_movement(current_edge, next_edge)
        if movement in TURN_PROBABILITIES:
            counts[movement] += 1
    return counts


def print_od_summary(sim_state):
    counts = sim_state.get("od_trip_counts")
    if not counts:
        return

    total = sum(counts.values())
    if total <= 0:
        return

    print()
    print("OD trip summary")
    print("=" * 76)
    for key in sorted(counts):
        pct = 100.0 * counts[key] / total
        print(f"{key:16}: {counts[key]:8d} trips   share={pct:6.2f}%")

    movement_counts = sim_state.get("od_movement_counts", Counter())
    movement_total = movement_counts["S"] + movement_counts["R"] + movement_counts["L"]
    if movement_total > 0:
        print("-" * 76)
        for group, name in [("S", "Straight"), ("R", "Right"), ("L", "Left")]:
            count = movement_counts[group]
            pct = 100.0 * count / movement_total
            print(f"{name:8}: {count:8d} routed movements   share={pct:6.2f}%")

    print(f"route_failures    : {sim_state.get('od_route_failures', 0):8d}")
    print("=" * 76)


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
           length="7.0"
           width="2.4"
           minGap="1.0"
           accel="3.5"
           decel="6.0"
           emergencyDecel="9.0"
           sigma="0.2"
           tau="0.6"
           color="255,50,50"
           speedFactor="1.35"
           lcStrategic="100.0"
           lcCooperative="1.0"
           lcSpeedGain="0.10"
           lcKeepRight="0.0"
           lcAssertive="0.60"
           jmIgnoreFoeProb="0.0"
           jmIgnoreJunctionFoeProb="0.0"
           jmTimegapMinor="1.0"
           jmAdvance="0"
           jmExtraGap="2.5"
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


def right_turn_signal_indices(controller, check_space=True):
    """Return signal indices that can safely remain permissive-green for right turns.

    A signal index is treated as a right-turn-only index only when the same
    signal index is not also used by a straight/left movement.  If SUMO shares
    one signal index between a right turn and another movement, the non-right
    movement wins and the signal is not forced to permissive green.
    """
    movement_map = controller["movement_map"]
    right_indices = set()
    non_right_indices = set()

    for movement_label, signal_data in movement_map.items():
        for signal_index, lane_sets in signal_data.items():
            if movement_label in ALL_RIGHT_TURNS:
                if check_space and not lanes_have_space(lane_sets["out"]):
                    continue
                right_indices.add(signal_index)
            else:
                non_right_indices.add(signal_index)

    return right_indices - non_right_indices


def state_with_permissive_right_turns(controller, base_char="r"):
    """Return a signal state with right-turn-only lanes held at permissive green."""
    state = [base_char] * controller["state_length"]
    active_indices = set()

    for signal_index in right_turn_signal_indices(controller, check_space=True):
        if 0 <= signal_index < controller["state_length"]:
            state[signal_index] = "g"
            active_indices.add(signal_index)

    return "".join(state), active_indices


def all_red_with_permissive_right_turns_state(controller):
    """Return a clearance state where right turns stay permissive green.

    During the short clearance period between phases, protected through/left
    movements remain red, but right-turn-only lanes stay lower-case permissive
    green (g).  The outgoing-lane space check is kept here, and the normal
    vehicle-level keep-clear/right-of-way gate still runs every simulation step,
    so right-turning cars are not allowed to enter a blocked junction/exit.
    """
    return state_with_permissive_right_turns(controller, base_char="r")


def build_yellow_state(length, active_indices, permissive_green_indices=None):
    state = ["r"] * length
    permissive_green_indices = set(permissive_green_indices or ())

    for idx in active_indices:
        if 0 <= idx < length and idx not in permissive_green_indices:
            state[idx] = "y"

    for idx in permissive_green_indices:
        if 0 <= idx < length:
            state[idx] = "g"

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
    # Right-turn-only signal indices should not turn yellow during the
    # transition. They remain permissive green while protected straight/left
    # movements change to yellow. Vehicles still obey the keep-clear and
    # right-of-way checks before actually entering the junction.
    permissive_right_indices = right_turn_signal_indices(controller, check_space=True)

    yellow_state = build_yellow_state(
        controller["state_length"],
        controller["last_active_indices"],
        permissive_green_indices=permissive_right_indices,
    )

    traci.trafficlight.setRedYellowGreenState(
        controller["tls_id"],
        yellow_state,
    )

    controller["mode"] = "yellow"
    controller["remaining"] = T_YELLOW
    controller["last_active_indices"] = permissive_right_indices
    controller["last_signal_update"] = traci.simulation.getTime()


def start_all_red(controller):
    clearance_state, active_indices = all_red_with_permissive_right_turns_state(controller)

    traci.trafficlight.setRedYellowGreenState(
        controller["tls_id"],
        clearance_state,
    )

    controller["mode"] = "all_red"
    controller["remaining"] = T_ALL_RED
    controller["last_active_indices"] = active_indices
    controller["last_signal_update"] = traci.simulation.getTime()


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
    # Identical logic to planned_next_edge_from_route — kept as a thin alias
    # so both call sites share one implementation instead of two copies that
    # could silently drift apart.
    return planned_next_edge_from_route(veh_id, current_edge)


def required_lane_change_distance(current_lane_index, target_lane_index):
    if current_lane_index is None or target_lane_index is None:
        return APPROACH_LANE_CHANGE_BASE_DISTANCE

    lane_delta = abs(target_lane_index - current_lane_index)
    return APPROACH_LANE_CHANGE_BASE_DISTANCE + APPROACH_LANE_CHANGE_DISTANCE_PER_LANE * lane_delta


def reachable_preferred_lanes(
    lane_info,
    outgoing_edge,
    movement,
    current_lane,
    distance_to_end,
    reserved_no_change_distance=0.0,
):
    current_lane_index = lane_index_from_lane_id(current_lane)
    preferred = target_lanes_for_movement(lane_info, outgoing_edge, movement)
    reachable = []

    usable_distance = max(0.0, distance_to_end - reserved_no_change_distance)

    for lane_id in preferred:
        target_lane_index = lane_index_from_lane_id(lane_id)
        required_distance = required_lane_change_distance(current_lane_index, target_lane_index)

        if target_lane_index == current_lane_index or usable_distance >= required_distance:
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


def build_approach_options_for_vehicle(
    lane_info,
    current_lane,
    distance_to_end,
    reserved_no_change_distance=0.0,
):
    options_by_group = defaultdict(list)

    for movement in MOVEMENT_ORDER:
        for outgoing_edge in sorted(lane_info["movement_to_edges"].get(movement, set())):
            candidate_lanes = reachable_preferred_lanes(
                lane_info=lane_info,
                outgoing_edge=outgoing_edge,
                movement=movement,
                current_lane=current_lane,
                distance_to_end=distance_to_end,
                reserved_no_change_distance=reserved_no_change_distance,
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

    no_change_buffer = traffic_light_no_lane_change_distance_for_lane(current_lane)
    required_distance = required_lane_change_distance(current_lane_index, target_lane_index)

    # If this is a signalized approach, the lane change must be started early
    # enough to complete before the protected no-change zone begins.
    if distance_to_end <= no_change_buffer + required_distance:
        return False

    try:
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
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
    no_change_buffer = traffic_light_no_lane_change_distance_for_lane(lane_id)

    existing = APPROACH_TURN_DECISIONS.get(key)
    if existing is not None:
        return enforce_approach_target_lane(
            veh_id=veh_id,
            target_lane_index=existing.get("target_lane_index"),
            distance_to_end=distance_to_end,
        )

    # On signalized approaches, make the S/R/L and lane decision farther
    # upstream, then stop making new lane-change decisions inside the protected
    # no-change zone.
    max_decision_distance = APPROACH_DECISION_MAX_DISTANCE_TO_END
    if no_change_buffer > 0.0:
        max_decision_distance = max(max_decision_distance, TRAFFIC_LIGHT_LANE_PREP_DISTANCE)

    if distance_to_end > max_decision_distance:
        return False

    if no_change_buffer > 0.0 and distance_to_end <= no_change_buffer:
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
        reserved_no_change_distance=no_change_buffer,
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
    recent_edges_set = set(recent_edges or ())

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
            * loop_avoidance_weight_multiplier(current_edge, outgoing_edge, recent_edges_set)
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
    if is_ambulance(veh_id):
        return False

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
    no_change_buffer = traffic_light_no_lane_change_distance_for_lane(lane_id)

    if speed < LANE_PREF_MIN_SPEED:
        return False
    if waiting_time > LANE_PREF_MAX_WAITING_TIME:
        return False
    if distance_to_end < LANE_PREF_MIN_DISTANCE_TO_END:
        return False
    if no_change_buffer > 0.0 and distance_to_end <= no_change_buffer:
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

    required_distance = required_lane_change_distance(current_lane_index, target_lane_index)
    if no_change_buffer > 0.0 and distance_to_end <= no_change_buffer + required_distance:
        return False

    try:
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
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


def cleanup_lane_balance_tracking(active_ids):
    """Remove stale lane-balance cooldown entries for vehicles that left SUMO."""
    active_ids = set(active_ids)
    for veh_id in list(LANE_BALANCE_LAST_CHANGE):
        if veh_id not in active_ids:
            LANE_BALANCE_LAST_CHANGE.pop(veh_id, None)


def lane_balance_candidate_lanes(current_edge, next_edge):
    """Lanes on current_edge that can legally serve the vehicle's next edge."""
    if not current_edge or not next_edge:
        return []

    candidates = lanes_on_edge_connecting_to(current_edge, next_edge)
    if len(candidates) >= LANE_BALANCE_MIN_EDGE_LANES:
        return candidates

    # If SUMO does not expose a clean next-edge lane-link set, fall back to all
    # lanes on the edge. This keeps the helper useful on ordinary multi-lane
    # road sections, but the main path above is stricter and safer.
    lanes = list(cached_edge_lanes(current_edge))
    if len(lanes) >= LANE_BALANCE_MIN_EDGE_LANES:
        return lanes

    return []


def choose_lane_balance_target(current_lane, candidate_lanes):
    """Choose a nearby underused lane, or None if lanes are already balanced."""
    current_lane_index = lane_index_from_lane_id(current_lane)
    if current_lane_index is None:
        return None

    lane_stats = []
    current_count = None

    for lane_id in candidate_lanes:
        lane_index = lane_index_from_lane_id(lane_id)
        if lane_index is None:
            continue

        try:
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            halting_count = traci.lane.getLastStepHaltingNumber(lane_id)
        except traci.TraCIException:
            continue

        if lane_id == current_lane:
            current_count = vehicle_count

        lane_stats.append((vehicle_count, halting_count, abs(lane_index - current_lane_index), lane_index, lane_id))

    if current_count is None or len(lane_stats) < LANE_BALANCE_MIN_EDGE_LANES:
        return None

    lane_stats.sort()
    best_count, best_halting, lane_delta, target_lane_index, target_lane = lane_stats[0]

    # Do not churn vehicles when the imbalance is small. This is intentionally
    # conservative: it fixes obvious one-lane pileups without making the road
    # look like cars are constantly weaving.
    if current_count - best_count < LANE_BALANCE_MIN_IMBALANCE:
        return None

    if lane_delta == 0 or lane_delta > LANE_BALANCE_MAX_LANE_DELTA:
        return None

    return target_lane_index


def apply_lane_balancing_to_vehicle(veh_id):
    """Gently spread cruising traffic across usable lanes on multi-lane roads."""
    if is_ambulance(veh_id):
        return False

    now = current_sim_time()

    if now - LANE_BALANCE_LAST_CHANGE.get(veh_id, -1e9) < LANE_BALANCE_MIN_TIME_BETWEEN_CHANGES:
        return False

    if veh_id in KEEP_CLEAR_HELD_VEHICLES:
        return False

    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        if not lane_id or lane_id.startswith(":"):
            return False

        current_edge = lane_to_edge(lane_id)
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = cached_lane_length(lane_id)
        speed = traci.vehicle.getSpeed(veh_id)
        waiting_time = traci.vehicle.getWaitingTime(veh_id)
    except traci.TraCIException:
        return False

    if current_edge is None or lane_len <= 0.0:
        return False

    distance_to_end = lane_len - lane_pos

    # Only balance free-flowing mid-road traffic. Near intersections, the
    # approach-decision and turn-lane-preference logic should remain in charge.
    no_change_buffer = traffic_light_no_lane_change_distance_for_lane(lane_id)

    if lane_pos < LANE_BALANCE_MIN_DISTANCE_FROM_START:
        return False
    if distance_to_end < LANE_BALANCE_MIN_DISTANCE_TO_END:
        return False
    if no_change_buffer > 0.0 and distance_to_end <= no_change_buffer + LANE_BALANCE_MIN_DISTANCE_TO_END:
        return False
    if speed < LANE_BALANCE_MIN_SPEED:
        return False
    if waiting_time > LANE_BALANCE_MAX_WAITING_TIME:
        return False

    try:
        route = list(traci.vehicle.getRoute(veh_id))
        route_index = traci.vehicle.getRouteIndex(veh_id)
    except traci.TraCIException:
        return False

    next_edge = planned_next_edge_for_vehicle(veh_id, current_edge)
    if next_edge is None or next_edge.startswith(":"):
        return False

    candidate_lanes = lane_balance_candidate_lanes(current_edge, next_edge)
    if lane_id not in candidate_lanes or len(candidate_lanes) < LANE_BALANCE_MIN_EDGE_LANES:
        return False

    target_lane_index = choose_lane_balance_target(lane_id, candidate_lanes)
    if target_lane_index is None:
        return False

    try:
        # Mode 1621 preserves SUMO safety/collision checks while allowing a
        # script-requested lane change. It is already used by the unconnected
        # lane rescue in this file, so this keeps behavior consistent.
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        traci.vehicle.changeLane(veh_id, target_lane_index, LANE_BALANCE_CHANGE_DURATION)
        LANE_BALANCE_LAST_CHANGE[veh_id] = now
        return True
    except traci.TraCIException:
        return False


def apply_lane_balancing_to_all_vehicles():
    changed = 0
    active_ids = list(traci.vehicle.getIDList())
    cleanup_lane_balance_tracking(active_ids)

    for veh_id in active_ids:
        if apply_lane_balancing_to_vehicle(veh_id):
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
    # Convert once; should_avoid_successor_for_loop / loop_avoidance_weight_multiplier
    # are called once per candidate successor below, and would otherwise rebuild
    # this same set on every single call.
    recent_edges_set = set(recent_edges)
    non_backtrack_successors = [s for s in successors if s != previous_edge]
    preferred_successors = [
        s
        for s in non_backtrack_successors
        if not should_avoid_successor_for_loop(current_edge, s, recent_edges_set)
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
                ) * loop_avoidance_weight_multiplier(current_edge, s, recent_edges_set)
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
        ) * loop_avoidance_weight_multiplier(current_edge, s, recent_edges_set)
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
    # Convert once here; this function calls loop-avoidance helpers across
    # nested loops (per TLS group, then per candidate option) that would
    # otherwise each rebuild this same set from the tuple repeatedly.
    recent_edges_set = set(recent_edges)

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
                    ) * loop_avoidance_weight_multiplier(current_edge, outgoing_edge, recent_edges_set)
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

        if use_od_routing(args) and sim_state.get("od_context") is not None:
            route_edges, od_info = build_od_route(
                sim_state=sim_state,
                context=sim_state["od_context"],
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                rng=rng,
                args=args,
            )

            if not route_edges:
                sim_state["od_route_failures"] = sim_state.get("od_route_failures", 0) + 1
                if not args.od_random_walk_fallback:
                    continue

                # Last-resort fallback only. Normal OD mode should use fastest
                # SUMO routes, not forced S/R/L random-walk routing.
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
                od_info = {"trip_type": "random_walk_fallback"}

            if od_info is not None:
                sim_state.setdefault("od_trip_counts", Counter())[od_info.get("trip_type", "unknown")] += 1
                sim_state.setdefault("od_movement_counts", Counter()).update(count_route_movements(route_edges))

        else:
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
                departLane=getattr(args, "depart_lane", OD_DEPART_LANE),
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
# Ambulance management
# ============================================================

def ambulance_route(origin_edge, destination_edge):
    try:
        return traci.simulation.findRoute(
            origin_edge,
            destination_edge,
            "ambulance",
            current_sim_time(),
        )
    except TypeError:
        try:
            return traci.simulation.findRoute(origin_edge, destination_edge, "ambulance")
        except TypeError:
            return traci.simulation.findRoute(origin_edge, destination_edge)


def make_circle_polygon(cx, cy, radius, points=24):
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / points),
            cy + radius * math.sin(2 * math.pi * i / points),
        )
        for i in range(points)
    ]


def choose_ambulance_origin_destination(raw_graph, edge_metadata, rng, args):
    candidates = [
        edge_id
        for edge_id in raw_graph
        if edge_id and not edge_id.startswith(":") and edge_length(edge_id, edge_metadata) >= OD_MIN_EDGE_LENGTH
    ]

    if len(candidates) < 2:
        return None, None

    weights = [max(0.001, edge_base_weight(edge_id, edge_metadata)) for edge_id in candidates]

    for _ in range(max(1, int(args.ambulance_route_attempts))):
        origin = weighted_choice(rng, candidates, weights)
        destination = weighted_choice(rng, candidates, weights)

        if origin == destination:
            continue

        if edge_distance(origin, destination, edge_metadata) < args.ambulance_min_euclidean_distance:
            continue

        return origin, destination

    return None, None


def spawn_ambulance(sim_state, raw_graph, edge_metadata, rng, args):
    if getattr(args, "disable_ambulances", False):
        return None

    for _ in range(max(1, int(args.ambulance_route_attempts))):
        origin, destination = choose_ambulance_origin_destination(
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            rng=rng,
            args=args,
        )

        if origin is None or destination is None:
            return None

        try:
            path = ambulance_route(origin, destination)
        except traci.TraCIException:
            continue

        route_edges = list(getattr(path, "edges", []) or [])
        if len(route_edges) < args.ambulance_min_route_edges:
            continue
        if route_edges[0] != origin or route_edges[-1] != destination:
            continue
        if route_distance(route_edges, edge_metadata) < args.ambulance_min_route_distance:
            continue
        if route_enters_hardcoded_loop_region(route_edges):
            continue

        amb_id = f"ambulance_{sim_state['next_vehicle_id']}"
        route_id = f"ambulance_route_{sim_state['next_route_id']}"
        sim_state["next_vehicle_id"] += 1
        sim_state["next_route_id"] += 1

        try:
            traci.route.add(route_id, route_edges)
            traci.vehicle.add(
                vehID=amb_id,
                routeID=route_id,
                typeID="ambulance",
                depart=str(current_sim_time()),
                departLane=args.ambulance_depart_lane,
                departPos=args.ambulance_depart_pos,
                departSpeed="max",
            )
            traci.vehicle.setColor(amb_id, AMBULANCE_COLOR)

            # Important: do not call moveTo() here. moveTo() can forcibly place
            # the ambulance onto an occupied lane, which looks like phasing
            # through normal cars. Let SUMO perform a normal safe departure.
            # Also do not call setMaxSpeed(); the vType has no artificial low
            # maxSpeed cap, while normal car-following and lane constraints still
            # prevent impossible overlap with other vehicles.
            try:
                traci.vehicle.setParameter(amb_id, "time-to-teleport", "-1")
            except traci.TraCIException:
                pass

            ox, oy = edge_xy(origin, edge_metadata)
            dx, dy = edge_xy(destination, edge_metadata)
            poi_a = f"{amb_id}_A"
            poi_b = f"{amb_id}_B"

            try:
                traci.polygon.add(
                    poi_a,
                    make_circle_polygon(ox, oy, args.ambulance_poi_radius),
                    color=(0, 255, 0, 230),
                    fill=True,
                    layer=100,
                )
                traci.polygon.add(
                    poi_b,
                    make_circle_polygon(dx, dy, args.ambulance_poi_radius),
                    color=(255, 0, 0, 230),
                    fill=True,
                    layer=100,
                )
            except Exception:
                poi_a = None
                poi_b = None

            sim_state.setdefault("active_ambulances", {})[amb_id] = {
                "origin": origin,
                "destination": destination,
                "route_len": len(route_edges),
                "route_distance": route_distance(route_edges, edge_metadata),
                "poi_a": poi_a,
                "poi_b": poi_b,
            }
            return amb_id

        except traci.TraCIException:
            continue

    return None


def update_ambulances(sim_state, raw_graph, edge_metadata, rng, sim_time, args):
    active = sim_state.setdefault("active_ambulances", {})

    if getattr(args, "disable_ambulances", False):
        return 0

    spawned = 0
    next_spawn = sim_state.setdefault("next_ambulance_spawn", 0.0)
    if args.ambulance_interval > 0.0 and sim_time >= next_spawn:
        amb_id = spawn_ambulance(
            sim_state=sim_state,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            rng=rng,
            args=args,
        )
        if amb_id:
            spawned += 1
            info = active[amb_id]
            print(
                f"  [ambulance] {amb_id}: {info['origin']} -> {info['destination']} "
                f"({info.get('route_len', '?')} edges, {info.get('route_distance', 0.0):.0f} m)",
                flush=True,
            )
        sim_state["next_ambulance_spawn"] = sim_time + args.ambulance_interval

    current_ids = set(traci.vehicle.getIDList())
    for amb_id in list(active.keys()):
        if amb_id not in current_ids:
            info = active.pop(amb_id, {})
            print(f"  [ambulance] {amb_id} arrived or left the simulation", flush=True)
            for poi_key in ("poi_a", "poi_b"):
                poi_id = info.get(poi_key)
                if poi_id:
                    try:
                        traci.polygon.remove(poi_id)
                    except Exception:
                        pass
            continue

        try:
            # Do not setMaxSpeed() and do not setSpeed() here. Repeated speed
            # overrides can defeat the keep-clear gate. The ambulance may pass
            # red lights through the vType junction parameters, but it still
            # must obey car-following and downstream-space checks.
            traci.vehicle.setParameter(amb_id, "time-to-teleport", "-1")

            if getattr(args, "ambulance_debug", False):
                route = traci.vehicle.getRoute(amb_id)
                route_idx = traci.vehicle.getRouteIndex(amb_id)
                speed = traci.vehicle.getSpeed(amb_id)
                edge = traci.vehicle.getRoadID(amb_id)
                print(
                    f"  [ambulance] {amb_id} edge={edge} idx={route_idx}/{len(route)} speed={speed:.1f}",
                    flush=True,
                )
        except traci.TraCIException:
            active.pop(amb_id, None)

    return spawned

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
    apply_traffic_light_lane_change_lock_to_all_vehicles()

    for veh_id in active_vehicle_ids:
        # Ambulances keep their own emergency OD routes. Do not rewrite their
        # routes or force lane changes with the normal passenger-car helpers.
        if is_ambulance(veh_id):
            continue

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

        if not use_od_routing(args):
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

            # Random-walk mode still chooses S/R/L near intersections. OD mode
            # does not: it preserves the fastest origin-destination route so
            # turn ratios emerge naturally from the chosen trip endpoints.
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

        # Enforce the signalized-intersection no-lane-change zone before any
        # lane-preference or lane-balancing helpers run. This also suppresses
        # SUMO's own autonomous last-second lane changes near traffic lights.
        apply_traffic_light_lane_change_lock_to_all_vehicles()

        # Lane preference must run more often than route extension. Otherwise a
        # straight vehicle can enter a shared right/straight lane and remain
        # there until it is too late to move.
        if sim_time >= sim_state.get("next_lane_pref_time", 0.0):
            apply_turn_lane_preference_to_all_vehicles()
            sim_state["next_lane_pref_time"] = sim_time + LANE_PREF_INTERVAL

        # Mid-road lane balancing spreads free-flowing traffic across parallel
        # lanes before it reaches the approach-decision zone. It is intentionally
        # conservative so it does not cause last-second weaving near junctions.
        if sim_time >= sim_state.get("next_lane_balance_time", 0.0):
            apply_lane_balancing_to_all_vehicles()
            sim_state["next_lane_balance_time"] = sim_time + LANE_BALANCE_INTERVAL

        # Spawn and track emergency vehicles before the keep-clear gate runs,
        # so a newly spawned ambulance is still checked for downstream space
        # before the next simulation step.
        update_ambulances(
            sim_state=sim_state,
            raw_graph=raw_graph,
            edge_metadata=edge_metadata,
            rng=rng,
            sim_time=sim_time,
            args=args,
        )

        # Vehicle-level keep-clear / right-of-way gate.
        # Lights stay green according to the fixed cycle; cars decide whether
        # they have enough downstream space to enter the junction. This applies
        # to both signalized and unsignalized intersections.
        apply_keep_clear_and_right_of_way_to_all_vehicles()

        # Strong anti-phantom-stop watchdog. A car is allowed to stop only when
        # it has a leader ahead, is legitimately at an intersection/traffic
        # light, is at the end of its route, or is being actively rescued.
        # Otherwise, stale speed holds and route/lane problems are repaired.
        if sim_time >= sim_state.get("next_unjustified_stop_check_time", 0.0):
            apply_unjustified_stop_watchdog_to_all_vehicles(
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                core_edges=core_edges,
                turn_index=turn_index,
                rng=rng,
                turn_counts=turn_counts,
                sim_state=sim_state,
                args=args,
            )
            sim_state["next_unjustified_stop_check_time"] = sim_time + getattr(
                args, "unjustified_stop_check_interval", UNJUSTIFIED_STOP_CHECK_INTERVAL
            )

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
        rebuild_traffic_light_approach_lanes(controllers)

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

        od_context = None
        if use_od_routing(args):
            od_context = build_od_context(
                valid_edges=valid_edges,
                raw_graph=raw_graph,
                edge_metadata=edge_metadata,
                args=args,
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
            "next_od_origin_zone_index": 0,
            "next_lane_pref_time": 0.0,
            "next_lane_balance_time": 0.0,
            "next_unjustified_stop_check_time": 0.0,
            "next_unconnected_lane_rescue_time": 0.0,
            "next_ambulance_spawn": 0.0,
            "active_ambulances": {},
            "od_context": od_context,
            "od_trip_counts": Counter(),
            "od_movement_counts": Counter(),
            "od_route_failures": 0,
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
        print("  Clearance interval keeps only permissive right turns green when exit space is available.")
        print()
        if use_od_routing(args):
            print("Origin-destination movement model:")
            print("  Vehicles choose a start edge and a far-away destination edge.")
            print("  SUMO computes a fastest legal route using road speeds/travel time.")
            print("  Straight/right/left movements emerge from the OD routes instead of being forced.")
            print("  Local roads are mainly endpoints/access roads, not random cruising roads.")
            print("  Lane preference still helps vehicles prepare for the next routed movement early.")
        else:
            print("Dynamic movement target:")
            print("  Straight: 70.0%")
            print("  Right:    17.5%")
            print("  Left:     12.5%")
            print("  Decisions are made near each intersection when a safe lane change is still possible.")
            print("  Straight vehicles prefer straight-only/no-right lanes and avoid the rightmost lane when possible.")
        print()
        if args.disable_ambulances:
            print("Ambulance model: disabled")
        else:
            print("Ambulance model:")
            print(f"  spawn interval: {args.ambulance_interval:.1f} s")
            print("  no artificial low max-speed cap is applied by this script")
            print("  ambulances may proceed through red/yellow lights when safe")
            print("  collision/foe ignoring is disabled; no forced moveTo() placement is used")
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
        print("  No lane changes are allowed in the last 100 m before an intersection by default.")
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

                if use_od_routing(args):
                    print_od_summary(sim_state)
                else:
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
        if use_od_routing(args):
            try:
                print_od_summary(locals().get("sim_state", {}))
            except Exception:
                pass
        else:
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
        default=3250,
        help="The script tries to keep about this many active cars.",
    )

    parser.add_argument(
        "--initial-vehicles",
        type=int,
        default=1000,
        help="Cars spawned immediately at the start.",
    )

    parser.add_argument(
        "--spawn-batch",
        type=int,
        default=60,
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
        "--routing-mode",
        choices=[ROUTING_MODE_OD, ROUTING_MODE_RANDOM_WALK],
        default=ROUTING_MODE_OD,
        help=(
            "Route-generation mode. OD mode chooses far-away origin/destination "
            "pairs and uses SUMO fastest routes; random-walk preserves the older "
            "forced S/R/L movement generator."
        ),
    )
    parser.add_argument(
        "--od-route-attempts",
        type=int,
        default=OD_ROUTE_ATTEMPTS,
        help="Attempts to find a realistic fastest OD route for each spawned vehicle.",
    )
    parser.add_argument("--od-boundary-margin-fraction", type=float, default=OD_BOUNDARY_MARGIN_FRACTION)
    parser.add_argument("--od-min-euclidean-distance", type=float, default=OD_MIN_EUCLIDEAN_DISTANCE)
    parser.add_argument("--od-min-route-distance", type=float, default=OD_MIN_ROUTE_DISTANCE)
    parser.add_argument("--od-min-zone-separation", type=int, default=OD_MIN_ZONE_SEPARATION)
    parser.add_argument("--od-max-local-middle-fraction", type=float, default=OD_MAX_LOCAL_MIDDLE_FRACTION)
    parser.add_argument("--od-local-middle-trim-edges", type=int, default=OD_LOCAL_MIDDLE_TRIM_EDGES)
    parser.add_argument("--od-through-trip-probability", type=float, default=OD_THROUGH_TRIP_PROBABILITY)
    parser.add_argument("--od-access-trip-probability", type=float, default=OD_ACCESS_TRIP_PROBABILITY)
    parser.add_argument("--od-long-local-trip-probability", type=float, default=OD_LONG_LOCAL_TRIP_PROBABILITY)
    parser.add_argument("--od-min-edge-length", type=float, default=OD_MIN_EDGE_LENGTH)
    parser.add_argument(
        "--od-no-random-walk-fallback",
        action="store_true",
        help="Disable the emergency random-walk fallback when no OD route can be found.",
    )
    parser.add_argument(
        "--depart-lane",
        default=OD_DEPART_LANE,
        help="SUMO departLane value for spawned vehicles. 'free' spreads starts better than always using 'best'.",
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

    parser.add_argument(
        "--tls-no-lane-change-distance",
        type=float,
        default=TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE,
        help="Distance before a traffic light where lane changes are completely disabled.",
    )

    parser.add_argument(
        "--tls-lane-prep-distance",
        type=float,
        default=TRAFFIC_LIGHT_LANE_PREP_DISTANCE,
        help="Distance before a traffic light where route/lane decisions start being made earlier.",
    )

    parser.add_argument(
        "--intersection-no-lane-change-distance",
        type=float,
        default=INTERSECTION_NO_LANE_CHANGE_DISTANCE,
        help="Distance before any intersection where lane changes are completely disabled.",
    )

    parser.add_argument(
        "--intersection-lane-prep-distance",
        type=float,
        default=INTERSECTION_LANE_PREP_DISTANCE,
        help="Distance before any intersection where vehicles should already begin preparing for the correct turn lane.",
    )

    parser.add_argument("--max-depart-delay", type=int, default=300)

    parser.add_argument(
        "--time-to-teleport",
        type=int,
        default=180,
        help="Last-resort gridlock breaker. Use -1 to disable teleporting entirely.",
    )

    parser.add_argument(
        "--disable-unjustified-stop-watchdog",
        dest="unjustified_stop_watchdog",
        action="store_false",
        help="Disable the watchdog that repairs cars stopped in the middle of roads for no valid reason.",
    )
    parser.set_defaults(unjustified_stop_watchdog=UNJUSTIFIED_STOP_WATCHDOG_ENABLED)
    parser.add_argument(
        "--unjustified-stop-check-interval",
        type=float,
        default=UNJUSTIFIED_STOP_CHECK_INTERVAL,
        help="Seconds between scans for cars stopped on free road segments.",
    )
    parser.add_argument(
        "--unjustified-stop-speed",
        type=float,
        default=UNJUSTIFIED_STOP_SPEED,
        help="Speed below which a car is considered stopped/near-stopped by the watchdog.",
    )
    parser.add_argument(
        "--unjustified-stop-min-time",
        type=float,
        default=UNJUSTIFIED_STOP_MIN_TIME,
        help="How long a car may remain stopped without a valid reason before repair is attempted.",
    )

    parser.add_argument("--green-duration", type=float, default=30.0)
    parser.add_argument("--ambulance-interval", type=float, default=AMBULANCE_SPAWN_INTERVAL)
    parser.add_argument("--disable-ambulances", action="store_true")
    parser.add_argument("--ambulance-min-euclidean-distance", type=float, default=AMBULANCE_MIN_EUCLIDEAN_DISTANCE)
    parser.add_argument("--ambulance-min-route-distance", type=float, default=AMBULANCE_MIN_ROUTE_DISTANCE)
    parser.add_argument("--ambulance-min-route-edges", type=int, default=AMBULANCE_MIN_ROUTE_EDGES)
    parser.add_argument("--ambulance-route-attempts", type=int, default=AMBULANCE_ROUTE_ATTEMPTS)
    parser.add_argument("--ambulance-depart-lane", default=AMBULANCE_DEPART_LANE)
    parser.add_argument("--ambulance-depart-pos", default=AMBULANCE_DEPART_POS)
    parser.add_argument("--ambulance-poi-radius", type=float, default=AMBULANCE_POI_RADIUS)
    parser.add_argument("--ambulance-debug", action="store_true")
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

    globals()["INTERSECTION_NO_LANE_CHANGE_DISTANCE"] = max(
        0.0,
        float(args.intersection_no_lane_change_distance),
    )
    globals()["INTERSECTION_LANE_PREP_DISTANCE"] = max(
        globals()["INTERSECTION_NO_LANE_CHANGE_DISTANCE"],
        float(args.intersection_lane_prep_distance),
    )
    globals()["TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE"] = max(
        globals()["INTERSECTION_NO_LANE_CHANGE_DISTANCE"],
        float(args.tls_no_lane_change_distance),
    )
    globals()["TRAFFIC_LIGHT_LANE_PREP_DISTANCE"] = max(
        globals()["TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE"],
        globals()["INTERSECTION_LANE_PREP_DISTANCE"],
        float(args.tls_lane_prep_distance),
    )

    args.od_boundary_margin_fraction = min(0.45, max(0.01, float(args.od_boundary_margin_fraction)))
    args.od_min_euclidean_distance = max(0.0, float(args.od_min_euclidean_distance))
    args.od_min_route_distance = max(0.0, float(args.od_min_route_distance))
    args.od_min_zone_separation = max(0, int(args.od_min_zone_separation))
    args.od_route_attempts = max(1, int(args.od_route_attempts))
    args.od_max_local_middle_fraction = min(1.0, max(0.0, float(args.od_max_local_middle_fraction)))
    args.od_local_middle_trim_edges = max(0, int(args.od_local_middle_trim_edges))
    args.od_min_edge_length = max(0.0, float(args.od_min_edge_length))
    args.od_random_walk_fallback = not bool(args.od_no_random_walk_fallback)
    args.ambulance_interval = max(0.0, float(args.ambulance_interval))
    args.ambulance_min_euclidean_distance = max(0.0, float(args.ambulance_min_euclidean_distance))
    args.ambulance_min_route_distance = max(0.0, float(args.ambulance_min_route_distance))
    args.ambulance_min_route_edges = max(2, int(args.ambulance_min_route_edges))
    args.ambulance_route_attempts = max(1, int(args.ambulance_route_attempts))
    args.ambulance_poi_radius = max(1.0, float(args.ambulance_poi_radius))

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
