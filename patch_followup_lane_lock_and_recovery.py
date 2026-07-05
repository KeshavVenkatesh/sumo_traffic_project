#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import sys

p = Path("realistic_all_intersections_fixed_cycle.py")
if not p.exists():
    sys.exit("Run this from ~/Downloads/sumo_traffic_project; realistic_all_intersections_fixed_cycle.py not found.")

backup = Path("realistic_all_intersections_fixed_cycle_before_followup_lane_lock_and_recovery.py")
if not backup.exists():
    shutil.copy2(p, backup)
    print(f"Backed up {p} -> {backup}")
else:
    print(f"Backup already exists: {backup}")

s = p.read_text()
changed = []

# ---------------------------------------------------------------------------
# 1) Route-lane commitment should mark intent, not permanently hard-lock cars.
# ---------------------------------------------------------------------------
new_set_commit = '''def set_route_lane_commitment_lock(veh_id, current_edge):
    """Mark a vehicle as route-lane committed without hard-locking it forever.

    The earlier strict patch used laneChangeMode=0.  That kept cars disciplined,
    but it also let boxed-in vehicles freeze near signals.  We still remember
    that the car is committed to this edge/lane choice, but we leave normal SUMO
    lane changing available so the later rescue/watchdog code can recover.
    """
    try:
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        ROUTE_LANE_COMMITTED_VEHICLES.add(veh_id)
        ROUTE_LANE_COMMITTED_EDGE[veh_id] = current_edge
        return True
    except traci.TraCIException:
        ROUTE_LANE_COMMITTED_VEHICLES.discard(veh_id)
        ROUTE_LANE_COMMITTED_EDGE.pop(veh_id, None)
        return False

'''

s2, n = re.subn(
    r'def set_route_lane_commitment_lock\(veh_id, current_edge\):\n.*?\ndef apply_traffic_light_lane_change_lock_to_vehicle\(veh_id\):',
    new_set_commit + 'def apply_traffic_light_lane_change_lock_to_vehicle(veh_id):',
    s,
    count=1,
    flags=re.S,
)
if n:
    s = s2
    changed.append("softened route-lane commitment lock")
else:
    print("Warning: could not replace set_route_lane_commitment_lock()")

# ---------------------------------------------------------------------------
# 2) Final approach lock: do not hard-lock a vehicle if its current lane cannot
#    legally reach its next routed edge. That is exactly when it needs rescue.
# ---------------------------------------------------------------------------
new_tls_lock = '''def apply_traffic_light_lane_change_lock_to_vehicle(veh_id):
    """Disable last-second lane changes only when the current lane is route-compatible.

    If the current lane cannot reach the vehicle's planned next edge, locking
    lane changes would trap the car.  In that case, release stale holds/locks and
    let the unconnected-lane rescue or unjustified-stop watchdog reroute it.
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
        current_edge = lane_to_edge(lane_id)
    except traci.TraCIException:
        TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
        return False

    distance_to_end = lane_len - lane_pos

    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        next_edge = planned_next_edge_from_route(veh_id, current_edge) if current_edge else None
        if next_edge is not None and not next_edge.startswith(":") and not lane_has_connection_to_edge(lane_id, next_edge):
            # Wrong lane at the end: do NOT lock it.  This vehicle needs the
            # recovery path, not stricter lane discipline.
            try:
                traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
            except traci.TraCIException:
                pass
            TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
            release_route_lane_commitment(veh_id)
            release_keep_clear_vehicle(veh_id)
            return False

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

'''

s2, n = re.subn(
    r'def apply_traffic_light_lane_change_lock_to_vehicle\(veh_id\):\n.*?\ndef apply_traffic_light_lane_change_lock_to_all_vehicles\(\):',
    new_tls_lock + 'def apply_traffic_light_lane_change_lock_to_all_vehicles():',
    s,
    count=1,
    flags=re.S,
)
if n:
    s = s2
    changed.append("made traffic-light final lock route-compatible")
else:
    print("Warning: could not replace apply_traffic_light_lane_change_lock_to_vehicle()")

# ---------------------------------------------------------------------------
# 3) If a stopped car is in a lane that cannot reach its next edge, do not treat
#    being near an intersection as a valid reason to remain stopped forever.
# ---------------------------------------------------------------------------
needle = '''    # A car behind another car is allowed to stop. This covers normal queues,
    # including long queues that extend far upstream from a red light.
    if vehicle_has_close_leader(veh_id):
        return True

    # The first vehicle approaching a signalized stop line can legitimately
'''
insert = '''    # If this lane cannot reach the vehicle's planned next edge, the vehicle is
    # not legitimately stopped just because it is near an intersection.  It must
    # be rescued/rerouted rather than allowed to freeze at a green light.
    next_edge = planned_next_edge_from_route(veh_id, current_edge)
    if next_edge is not None and not next_edge.startswith(":"):
        if not lane_has_connection_to_edge(lane_id, next_edge):
            return False

    # A car behind another car is allowed to stop. This covers normal queues,
    # including long queues that extend far upstream from a red light.
    if vehicle_has_close_leader(veh_id):
        return True

    # The first vehicle approaching a signalized stop line can legitimately
'''
if needle in s and "not legitimately stopped just because it is near an intersection" not in s:
    s = s.replace(needle, insert, 1)
    changed.append("made wrong-lane stopped cars eligible for watchdog rescue")
elif "not legitimately stopped just because it is near an intersection" in s:
    print("Stopped-car watchdog patch already present")
else:
    print("Warning: could not insert wrong-lane stopped-car watchdog check")

# ---------------------------------------------------------------------------
# 4) Emergency reroute should prefer straight/right exits over left exits when
#    alternatives exist, which reduces cropped-boundary left-turn artifacts.
# ---------------------------------------------------------------------------
helper = r'''

def movement_for_lane_successor_edge(current_lane, outgoing_edge):
    """Classify the movement from current_lane to outgoing_edge as S/R/L if possible."""
    for link in get_lane_links(current_lane):
        if not link:
            continue
        to_lane = link[0]
        if lane_to_edge(to_lane) != outgoing_edge:
            continue
        movement = None
        if len(link) > 6:
            movement = sumo_link_direction_to_movement(link[6])
        if movement is None:
            _, movement = classify_movement_by_geometry(current_lane, to_lane)
        if movement in TURN_PROBABILITIES:
            return movement
    return None


def emergency_successor_bias(current_lane, outgoing_edge):
    """Prefer straight/right recovery over left when a car is being rescued."""
    movement = movement_for_lane_successor_edge(current_lane, outgoing_edge)
    if movement == "S":
        return 8.0
    if movement == "R":
        return 4.0
    if movement == "L":
        return 0.45
    return 1.0
'''
if "def emergency_successor_bias(" not in s:
    marker = "\ndef force_route_to_reachable_lane_successor(\n"
    if marker in s:
        s = s.replace(marker, helper + marker, 1)
        changed.append("added emergency straight/right successor bias helper")
    else:
        print("Warning: could not insert emergency successor helper")

old_weights = '''    weights = [
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
'''
new_weights = '''    weights = [
        successor_weight(
            current_edge=current_edge,
            next_edge=edge,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        )
        * loop_avoidance_weight_multiplier(current_edge, edge, recent_edges_set)
        * emergency_successor_bias(current_lane, edge)
        for edge in candidates
    ]
'''
if old_weights in s:
    s = s.replace(old_weights, new_weights, 1)
    changed.append("biased emergency reroutes away from left-turn-only artifacts")
elif "emergency_successor_bias(current_lane, edge)" in s:
    print("Emergency successor weighting already present")
else:
    print("Warning: could not patch emergency recovery weights")

# Ensure route reroute clears stale locks/holds before setting the new route.
old_set_route = '''    try:
        traci.vehicle.setRoute(veh_id, cleaned)
        safe_vehicle_set_speed(veh_id, -1)
        return True
    except traci.TraCIException:
        return False
'''
new_set_route = '''    try:
        release_keep_clear_vehicle(veh_id)
        release_route_lane_commitment(veh_id)
        TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
        traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
        traci.vehicle.setRoute(veh_id, cleaned)
        safe_vehicle_set_speed(veh_id, -1)
        return True
    except traci.TraCIException:
        return False
'''
if old_set_route in s:
    s = s.replace(old_set_route, new_set_route, 1)
    changed.append("cleared stale locks during emergency reroute")
elif "release_route_lane_commitment(veh_id)" in s and "traci.vehicle.setRoute(veh_id, cleaned)" in s:
    print("Emergency reroute lock clearing appears already present")
else:
    print("Warning: could not add stale-lock clearing to emergency reroute")

# ---------------------------------------------------------------------------
# 5) Soften the final no-change zone a bit, but keep early lane prep.
# ---------------------------------------------------------------------------
replacements = {
    "INTERSECTION_NO_LANE_CHANGE_DISTANCE = 140.0": "INTERSECTION_NO_LANE_CHANGE_DISTANCE = 80.0",
    "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE = 160.0": "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE = 90.0",
}
for a, b in replacements.items():
    if a in s:
        s = s.replace(a, b, 1)
        changed.append(f"changed {a} -> {b}")

p.write_text(s)
print("\nApplied follow-up patches:")
for item in changed:
    print(f"  - {item}")
if not changed:
    print("  (No changes made; file may already be patched.)")
print("\nNow run: python3 -m py_compile realistic_all_intersections_fixed_cycle.py")
