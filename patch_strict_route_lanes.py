#!/usr/bin/env python3
"""
Patch the realistic SUMO simulation so vehicles commit to route-compatible
turn/straight lanes much earlier and lane balancing cannot move them away from
those lanes near intersections.

Run from ~/Downloads/sumo_traffic_project:
    python3 patch_strict_route_lanes.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

SIM_FILE = Path("realistic_all_intersections_fixed_cycle.py")
COMPARE_FILE = Path("compare_fixed_vs_single_vs_all_model_realistic.py")


def replace_line_constant(text: str, name: str, value: str) -> str:
    pattern = rf"^{re.escape(name)}\s*=\s*.*$"
    repl = f"{name} = {value}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        raise RuntimeError(f"Could not find constant {name}")
    return new_text


def replace_function(text: str, name: str, next_name: str, new_func: str) -> str:
    start_pat = f"\ndef {name}("
    end_pat = f"\ndef {next_name}("
    start = text.find(start_pat)
    if start == -1:
        if text.startswith(f"def {name}("):
            start = 0
        else:
            raise RuntimeError(f"Could not find function {name}")
    else:
        start += 1
    end = text.find(end_pat, start + 1)
    if end == -1:
        raise RuntimeError(f"Could not find function following {name}: {next_name}")
    end += 1
    return text[:start] + new_func.rstrip() + "\n\n" + text[end:]


STRICT_HELPERS = r'''
def route_lane_prep_distance_for_lane(lane_id):
    """Distance upstream where vehicles start committing to route-compatible lanes."""
    if not lane_id or lane_id.startswith(":"):
        return 0.0

    edge_id = lane_to_edge(lane_id)
    if edge_id is None or edge_id.startswith(":"):
        return 0.0

    distance = INTERSECTION_LANE_PREP_DISTANCE
    if lane_id in TRAFFIC_LIGHT_APPROACH_LANES:
        distance = max(distance, TRAFFIC_LIGHT_LANE_PREP_DISTANCE)

    return max(traffic_light_no_lane_change_distance_for_lane(lane_id), distance)


def cleanup_route_lane_commitment_locks(active_ids):
    """Remove stale per-vehicle lane-commitment locks after cars leave SUMO."""
    active_ids = set(active_ids)
    for veh_id in list(ROUTE_LANE_COMMITTED_VEHICLES):
        if veh_id not in active_ids:
            ROUTE_LANE_COMMITTED_VEHICLES.discard(veh_id)
            ROUTE_LANE_COMMITTED_EDGE.pop(veh_id, None)


def release_route_lane_commitment(veh_id):
    """Release a route-lane lock unless the vehicle is also in the hard no-change zone."""
    if veh_id in ROUTE_LANE_COMMITTED_VEHICLES:
        try:
            if veh_id not in TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES:
                traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        except traci.TraCIException:
            pass
    ROUTE_LANE_COMMITTED_VEHICLES.discard(veh_id)
    ROUTE_LANE_COMMITTED_EDGE.pop(veh_id, None)


def set_route_lane_commitment_lock(veh_id, current_edge):
    """Stop SUMO's autonomous lane changes after the car reaches its route lane."""
    try:
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_LOCKED_LANE_CHANGE_MODE)
        ROUTE_LANE_COMMITTED_VEHICLES.add(veh_id)
        ROUTE_LANE_COMMITTED_EDGE[veh_id] = current_edge
        return True
    except traci.TraCIException:
        ROUTE_LANE_COMMITTED_VEHICLES.discard(veh_id)
        ROUTE_LANE_COMMITTED_EDGE.pop(veh_id, None)
        return False
'''

STRICT_APPLY_TURN_LANE = r'''
def apply_turn_lane_preference_to_vehicle(veh_id):
    """Commit each passenger car to a lane that matches its actual routed next edge.

    Older logic only nudged straight cars out of shared right/straight lanes.
    This stricter version uses the car's planned next edge, determines whether
    that movement is straight/right/left, and then moves the car into a lane that
    can serve that exact movement. Once the car is close enough and already in a
    preferred lane, SUMO's autonomous lane changes are disabled so the vehicle
    stays committed instead of drifting into a turn lane and then going straight.
    """
    if is_ambulance(veh_id):
        return False

    if not TURN_LANE_PREFERENCE_INDEX:
        return False

    try:
        lane_id = traci.vehicle.getLaneID(veh_id)
        if not lane_id or lane_id.startswith(":"):
            release_route_lane_commitment(veh_id)
            return False

        speed = traci.vehicle.getSpeed(veh_id)
        waiting_time = traci.vehicle.getWaitingTime(veh_id)
        lane_pos = traci.vehicle.getLanePosition(veh_id)
        lane_len = traci.vehicle.getLaneLength(veh_id)
    except traci.TraCIException:
        release_route_lane_commitment(veh_id)
        return False

    distance_to_end = lane_len - lane_pos
    current_edge = lane_to_edge(lane_id)
    if current_edge is None:
        release_route_lane_commitment(veh_id)
        return False

    if ROUTE_LANE_COMMITTED_EDGE.get(veh_id) not in (None, current_edge):
        release_route_lane_commitment(veh_id)

    lane_info = TURN_LANE_PREFERENCE_INDEX.get(current_edge)
    if not lane_info:
        release_route_lane_commitment(veh_id)
        return False

    next_edge = planned_next_edge_for_vehicle(veh_id, current_edge)
    if next_edge is None or next_edge.startswith(":"):
        release_route_lane_commitment(veh_id)
        return False

    movement = lane_info["edge_to_movement"].get(next_edge)
    if movement not in TURN_PROBABILITIES:
        release_route_lane_commitment(veh_id)
        return False

    preferred_lanes = target_lanes_for_movement(lane_info, next_edge, movement)
    if not preferred_lanes:
        release_route_lane_commitment(veh_id)
        return False

    no_change_buffer = traffic_light_no_lane_change_distance_for_lane(lane_id)
    prep_distance = route_lane_prep_distance_for_lane(lane_id)

    if prep_distance > 0.0 and distance_to_end > prep_distance:
        release_route_lane_commitment(veh_id)
        return False

    current_lane_index = lane_index_from_lane_id(lane_id)
    current_is_preferred = lane_id in preferred_lanes

    if current_is_preferred:
        hold_distance = max(
            no_change_buffer,
            min(route_lane_prep_distance_for_lane(lane_id), ROUTE_LANE_COMMITMENT_HOLD_DISTANCE),
        )
        if distance_to_end <= hold_distance:
            return set_route_lane_commitment_lock(veh_id, current_edge)
        release_route_lane_commitment(veh_id)
        return False

    target_lane_index = choose_best_target_lane(preferred_lanes, current_lane_index)
    if target_lane_index is None or target_lane_index == current_lane_index:
        return False

    required_distance = required_lane_change_distance(current_lane_index, target_lane_index)

    if no_change_buffer > 0.0 and distance_to_end <= no_change_buffer + required_distance:
        return False
    if distance_to_end < LANE_PREF_MIN_DISTANCE_TO_END:
        return False
    if speed < LANE_PREF_MIN_SPEED:
        return False
    if waiting_time > LANE_PREF_MAX_WAITING_TIME:
        return False

    try:
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        traci.vehicle.changeLane(veh_id, target_lane_index, TURN_LANE_CHANGE_DURATION)
        return True
    except traci.TraCIException:
        return False
'''

STRICT_APPLY_ALL = r'''
def apply_turn_lane_preference_to_all_vehicles():
    changed = 0

    if not TURN_LANE_PREFERENCE_INDEX:
        return changed

    active_ids = list(traci.vehicle.getIDList())
    cleanup_route_lane_commitment_locks(active_ids)

    for veh_id in active_ids:
        if apply_turn_lane_preference_to_vehicle(veh_id):
            changed += 1

    return changed
'''

STRICT_LANE_BALANCE_CANDIDATES = r'''
def lane_balance_candidate_lanes(current_edge, next_edge):
    """Only balance among lanes that can legally serve the vehicle's next edge."""
    if not current_edge or not next_edge:
        return []

    candidates = lanes_on_edge_connecting_to(current_edge, next_edge)
    if len(candidates) >= LANE_BALANCE_MIN_EDGE_LANES:
        return candidates

    # Strict route-lane mode: do not fall back to all lanes. That fallback can
    # make traffic look smoother, but it can also move cars into lanes that do
    # not match their next routed turn/straight movement.
    return []
'''

STRICT_LANE_BALANCE_VEHICLE = r'''
def apply_lane_balancing_to_vehicle(veh_id):
    """Gently spread cruising traffic, but never inside the route-lane prep zone."""
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

    # Once the route-lane preparation zone begins, the route-specific lane
    # commitment helper is in charge. Lane balancing must not move vehicles away
    # from lanes needed for the next turn/straight movement.
    prep_distance = route_lane_prep_distance_for_lane(lane_id)
    if prep_distance > 0.0 and distance_to_end <= prep_distance:
        return False

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
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        traci.vehicle.changeLane(veh_id, target_lane_index, LANE_BALANCE_CHANGE_DURATION)
        LANE_BALANCE_LAST_CHANGE[veh_id] = now
        return True
    except traci.TraCIException:
        return False
'''


def patch_sim_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    constants = {
        "INTERSECTION_NO_LANE_CHANGE_DISTANCE": "140.0",
        "INTERSECTION_LANE_PREP_DISTANCE": "650.0",
        "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE": "160.0",
        "TRAFFIC_LIGHT_LANE_PREP_DISTANCE": "750.0",
        "TURN_LANE_CHANGE_DURATION": "18.0",
        "APPROACH_LANE_CHANGE_DURATION": "20.0",
        "APPROACH_LANE_CHANGE_BASE_DISTANCE": "35.0",
        "APPROACH_LANE_CHANGE_DISTANCE_PER_LANE": "55.0",
        "OD_DEPART_LANE": '"best"',
    }
    for name, value in constants.items():
        text = replace_line_constant(text, name, value)

    if "ROUTE_LANE_COMMITMENT_HOLD_DISTANCE" not in text:
        marker = "TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES = set()\n"
        insert = (
            marker
            + "# Route-lane commitment helper. Once a vehicle reaches a lane that matches\n"
            + "# its routed next movement, keep it there through the final approach.\n"
            + "ROUTE_LANE_COMMITMENT_HOLD_DISTANCE = 550.0\n"
            + "ROUTE_LANE_COMMITTED_VEHICLES = set()\n"
            + "ROUTE_LANE_COMMITTED_EDGE = {}\n"
        )
        if marker not in text:
            raise RuntimeError("Could not find lane-change locked vehicles marker")
        text = text.replace(marker, insert, 1)
    else:
        text = replace_line_constant(text, "ROUTE_LANE_COMMITMENT_HOLD_DISTANCE", "550.0")

    if "def route_lane_prep_distance_for_lane(" not in text:
        marker = "\ndef apply_traffic_light_lane_change_lock_to_vehicle(veh_id):"
        if marker not in text:
            raise RuntimeError("Could not find apply_traffic_light_lane_change_lock_to_vehicle marker")
        text = text.replace(marker, "\n" + STRICT_HELPERS.strip() + "\n\n" + marker.lstrip("\n"), 1)

    text = replace_function(text, "apply_turn_lane_preference_to_vehicle", "apply_turn_lane_preference_to_all_vehicles", STRICT_APPLY_TURN_LANE)
    text = replace_function(text, "apply_turn_lane_preference_to_all_vehicles", "cleanup_lane_balance_tracking", STRICT_APPLY_ALL)
    text = replace_function(text, "lane_balance_candidate_lanes", "choose_lane_balance_target", STRICT_LANE_BALANCE_CANDIDATES)
    text = replace_function(text, "apply_lane_balancing_to_vehicle", "apply_lane_balancing_to_all_vehicles", STRICT_LANE_BALANCE_VEHICLE)

    path.write_text(text, encoding="utf-8")


def patch_compare_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if '"ROUTE_LANE_COMMITTED_VEHICLES"' not in text:
        text = text.replace(
            '        "TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES",\n',
            '        "TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES",\n'
            '        "ROUTE_LANE_COMMITTED_VEHICLES",\n'
            '        "ROUTE_LANE_COMMITTED_EDGE",\n',
            1,
        )
    path.write_text(text, encoding="utf-8")


def syntax_check(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)


def main() -> None:
    if not SIM_FILE.exists():
        raise SystemExit(f"Run this from your SUMO project folder. Missing {SIM_FILE}")

    sim_backup = SIM_FILE.with_suffix(".before_strict_route_lanes.py")
    shutil.copy2(SIM_FILE, sim_backup)
    print(f"Backed up {SIM_FILE} -> {sim_backup}")

    compare_backup = None
    if COMPARE_FILE.exists():
        compare_backup = COMPARE_FILE.with_suffix(".before_strict_route_lanes.py")
        shutil.copy2(COMPARE_FILE, compare_backup)
        print(f"Backed up {COMPARE_FILE} -> {compare_backup}")

    try:
        patch_sim_file(SIM_FILE)
        patch_compare_file(COMPARE_FILE)
        syntax_check(SIM_FILE)
        if COMPARE_FILE.exists():
            syntax_check(COMPARE_FILE)
    except Exception:
        shutil.copy2(sim_backup, SIM_FILE)
        if compare_backup is not None:
            shutil.copy2(compare_backup, COMPARE_FILE)
        print("Patch failed; restored backups.", file=sys.stderr)
        raise

    print("\nStrict route-lane commitment patch applied.")
    print("Key changes:")
    print("  - cars spawn with departLane='best'")
    print("  - route/lane preparation starts 650-750m upstream")
    print("  - cars commit to route-compatible lanes and stay there near intersections")
    print("  - lane balancing no longer moves cars inside the approach-prep zone")


if __name__ == "__main__":
    main()
