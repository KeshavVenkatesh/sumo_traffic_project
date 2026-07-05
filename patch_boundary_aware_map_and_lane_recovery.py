from pathlib import Path
import shutil

src = Path('realistic_all_intersections_fixed_cycle.py')
backup = Path('realistic_all_intersections_fixed_cycle_before_boundary_aware_lane_recovery.py')
if not src.exists():
    raise SystemExit('realistic_all_intersections_fixed_cycle.py not found in current folder')
if not backup.exists():
    shutil.copy2(src, backup)
    print(f'Backed up {src} -> {backup}')
s = src.read_text()

# 1) Add boundary/left-bias constants after OD_DEPART_LANE.
anchor = 'OD_DEPART_LANE = "best"\n'
insert = '''OD_DEPART_LANE = "best"

# Boundary-aware OD demand shaping.
# In a finite OSM crop, some boundary approaches have missing straight/right
# continuations because the real road continues outside the downloaded map.
# If OD demand repeatedly routes through those clipped approaches, SUMO can make
# almost every vehicle turn left simply because that is the only in-map path.
# These filters reject OD routes that would create unrealistic left-turn
# dominance while still allowing left turns when they are genuinely part of a
# reasonable route.
OD_ROUTE_MIN_MOVEMENTS_FOR_LEFT_FILTER = 4
OD_ROUTE_MAX_LEFT_SHARE = 0.48
OD_APPROACH_LEFT_BIAS_MIN_SAMPLES = 5
OD_APPROACH_MAX_LEFT_SHARE = 0.58
OD_LEFT_BIAS_RELAX_AFTER_ATTEMPTS_FRACTION = 0.80
OD_DESTINATION_ZONE_BALANCE = True

# Recovery reroutes should prefer clearing the junction straight/right when
# possible. This prevents emergency route repair from creating extra left-turn
# bias on clipped map edges.
RECOVERY_STRAIGHT_WEIGHT_MULTIPLIER = 2.25
RECOVERY_RIGHT_WEIGHT_MULTIPLIER = 1.35
RECOVERY_LEFT_WEIGHT_MULTIPLIER = 0.35
'''
if 'OD_ROUTE_MAX_LEFT_SHARE' not in s:
    if anchor not in s:
        raise SystemExit('Could not find OD_DEPART_LANE anchor')
    s = s.replace(anchor, insert, 1)

# 2) Add destination zone balancing helper after choose_zone_balanced_edge.
marker = '\ndef successor_weight(\n'
helper = r'''

def choose_zone_balanced_destination_edge(sim_state, pool, context, edge_metadata, rng):
    """Choose OD destinations by geographic zone, not just road weight.

    Origin selection was already zone-balanced, but destination selection used a
    global weighted choice. On a clipped map this can overuse one boundary or a
    few high-weight edges, which then makes the same boundary intersections show
    unrealistic one-direction turn patterns. This function keeps destinations
    distributed across the map.
    """
    pool = set(pool)
    if not pool:
        return None

    zones = [zone for zone in context.get("zones", []) if pool.intersection(zone.get("edges", []))]
    if not zones:
        return weighted_edge_choice(rng, pool, edge_metadata)

    start = sim_state.get("next_od_destination_zone_index", 0)
    # Try each zone in round-robin order until one has a candidate.
    for offset in range(len(zones)):
        zone = zones[(start + offset) % len(zones)]
        zone_edges = sorted(pool.intersection(zone.get("edges", [])))
        if zone_edges:
            sim_state["next_od_destination_zone_index"] = start + offset + 1
            return weighted_edge_choice(rng, zone_edges, edge_metadata)

    return weighted_edge_choice(rng, pool, edge_metadata)
'''
if 'def choose_zone_balanced_destination_edge' not in s:
    if marker not in s:
        raise SystemExit('Could not find successor_weight marker')
    s = s.replace(marker, helper + marker, 1)

# 3) Use destination zone balancing in choose_od_pair.
old = '    destination = weighted_edge_choice(rng, far_destinations, edge_metadata)\n    return origin, destination, trip_type\n'
new = '''    if OD_DESTINATION_ZONE_BALANCE:
        destination = choose_zone_balanced_destination_edge(
            sim_state=sim_state,
            pool=far_destinations,
            context=context,
            edge_metadata=edge_metadata,
            rng=rng,
        )
    else:
        destination = weighted_edge_choice(rng, far_destinations, edge_metadata)
    return origin, destination, trip_type
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'choose_zone_balanced_destination_edge(' not in s:
    raise SystemExit('Could not patch destination selection')

# 4) Add route movement/left-bias filter helpers before build_od_route.
marker = '\ndef build_od_route(sim_state, context, raw_graph, edge_metadata, rng, args):\n'
helpers = r'''

def route_movement_entries(route_edges):
    """Return (incoming_edge, outgoing_edge, movement) for route transitions."""
    entries = []
    for current_edge, next_edge in zip(route_edges, route_edges[1:]):
        movement = classify_edge_successor_movement(current_edge, next_edge)
        if movement in TURN_PROBABILITIES:
            entries.append((current_edge, next_edge, movement))
    return entries


def route_has_excessive_left_bias(route_edges, sim_state, args):
    """Reject routes that would create obvious left-turn artifacts.

    This is not a hard ban on left turns. It only filters routes when either the
    whole route is left-heavy or one incoming edge is already becoming a nearly
    all-left approach. The goal is to avoid artifacts caused by clipped map
    boundaries, where straight/right continuations may simply not exist in the
    downloaded OSM crop.
    """
    entries = route_movement_entries(route_edges)
    if not entries:
        return False

    counts = Counter(movement for _cur, _nxt, movement in entries)
    total = counts["S"] + counts["R"] + counts["L"]
    if total >= OD_ROUTE_MIN_MOVEMENTS_FOR_LEFT_FILTER:
        if counts["L"] / max(1, total) > OD_ROUTE_MAX_LEFT_SHARE:
            return True

    approach_counts = sim_state.setdefault("od_approach_movement_counts", defaultdict(Counter))
    for current_edge, _next_edge, movement in entries:
        if movement != "L":
            continue

        existing = approach_counts[current_edge]
        seen = existing["S"] + existing["R"] + existing["L"]
        if seen < OD_APPROACH_LEFT_BIAS_MIN_SAMPLES:
            continue

        projected_left_share = (existing["L"] + 1) / max(1, seen + 1)
        if projected_left_share > OD_APPROACH_MAX_LEFT_SHARE:
            return True

    return False


def record_od_route_approach_movements(sim_state, route_edges):
    """Record per-approach movement counts after accepting/spawning a route."""
    approach_counts = sim_state.setdefault("od_approach_movement_counts", defaultdict(Counter))
    for current_edge, _next_edge, movement in route_movement_entries(route_edges):
        approach_counts[current_edge][movement] += 1


def recovery_movement_multiplier(current_edge, next_edge):
    movement = classify_edge_successor_movement(current_edge, next_edge)
    if movement == "S":
        return RECOVERY_STRAIGHT_WEIGHT_MULTIPLIER
    if movement == "R":
        return RECOVERY_RIGHT_WEIGHT_MULTIPLIER
    if movement == "L":
        return RECOVERY_LEFT_WEIGHT_MULTIPLIER
    return 1.0
'''
if 'def route_has_excessive_left_bias' not in s:
    if marker not in s:
        raise SystemExit('Could not find build_od_route marker')
    s = s.replace(marker, helpers + marker, 1)

# 5) Patch build_od_route loop and filter.
s = s.replace('def build_od_route(sim_state, context, raw_graph, edge_metadata, rng, args):\n    for _ in range(args.od_route_attempts):',
'''def build_od_route(sim_state, context, raw_graph, edge_metadata, rng, args):
    relax_after = max(1, int(args.od_route_attempts * OD_LEFT_BIAS_RELAX_AFTER_ATTEMPTS_FRACTION))
    for attempt in range(args.od_route_attempts):''', 1)
old = '''        if not od_route_is_reasonable(
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
'''
new = '''        if not od_route_is_reasonable(
            route_edges=route_edges,
            origin=origin,
            destination=destination,
            trip_type=trip_type,
            context=context,
            edge_metadata=edge_metadata,
            args=args,
        ):
            continue

        # Prefer another OD pair if this route would over-concentrate left
        # turns. Late in the attempt budget we relax this so spawning does not
        # fail completely on very small or highly constrained maps.
        if attempt < relax_after and route_has_excessive_left_bias(route_edges, sim_state, args):
            continue

        return route_edges, {
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'route_has_excessive_left_bias(route_edges' not in s:
    raise SystemExit('Could not patch build_od_route filter')

# 6) Record per-approach movements when a vehicle is actually spawned.
old = '''            if od_info is not None:
                sim_state.setdefault("od_trip_counts", Counter())[od_info.get("trip_type", "unknown")] += 1
                sim_state.setdefault("od_movement_counts", Counter()).update(count_route_movements(route_edges))
'''
new = '''            if od_info is not None:
                sim_state.setdefault("od_trip_counts", Counter())[od_info.get("trip_type", "unknown")] += 1
                sim_state.setdefault("od_movement_counts", Counter()).update(count_route_movements(route_edges))
                record_od_route_approach_movements(sim_state, route_edges)
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'record_od_route_approach_movements(sim_state, route_edges)' not in s:
    raise SystemExit('Could not patch spawn recording')

# 7) Add next_od_destination_zone_index and movement count state init if present.
s = s.replace('''            "next_od_origin_zone_index": 0,
            "next_lane_pref_time": 0.0,''',
'''            "next_od_origin_zone_index": 0,
            "next_od_destination_zone_index": 0,
            "next_lane_pref_time": 0.0,''')
s = s.replace('''            "od_movement_counts": Counter(),
            "od_route_failures": 0,''',
'''            "od_movement_counts": Counter(),
            "od_approach_movement_counts": defaultdict(Counter),
            "od_route_failures": 0,''')

# 8) Make recovery route successor prefer straight/right over left.
old = '''    weights = [
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
new = '''    weights = [
        successor_weight(
            current_edge=current_edge,
            next_edge=edge,
            previous_edge=previous_edge,
            edge_metadata=edge_metadata,
            core_edges=core_edges,
            args=args,
        )
        * loop_avoidance_weight_multiplier(current_edge, edge, recent_edges_set)
        * recovery_movement_multiplier(current_edge, edge)
        for edge in candidates
    ]
'''
if old in s:
    s = s.replace(old, new, 1)
elif '* recovery_movement_multiplier(current_edge, edge)' not in s:
    print('Warning: did not patch recovery weights; pattern not found')

# 9) Soften route commitment lock.
old = '''def set_route_lane_commitment_lock(veh_id, current_edge):
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
new = '''def set_route_lane_commitment_lock(veh_id, current_edge):
    """Mark the car as route-lane committed without hard-freezing it.

    A full laneChangeMode=0 lock made boxed-in cars unable to recover.  The
    preference helper still keeps vehicles in route-compatible lanes, while SUMO
    and the emergency watchdog retain enough freedom to unblock traffic.
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
if old in s:
    s = s.replace(old, new, 1)
else:
    print('Warning: route commitment lock pattern not found')

# 10) Add lane-link/current-green helper before has_valid_reason_to_be_stopped.
marker = '\ndef has_valid_reason_to_be_stopped(veh_id, lane_id, current_edge, lane_pos, distance_to_end):\n'
helper = r'''

def lane_link_is_open_to_edge(lane_id, target_edge):
    """True when SUMO currently exposes an open/green-ish link to target_edge."""
    if not lane_id or not target_edge:
        return False

    for link in get_lane_links(lane_id):
        if not link:
            continue
        to_lane = link[0]
        if lane_to_edge(to_lane) != target_edge:
            continue

        # Extended layout usually: (toLane, hasPrio, isOpen, hasFoe, viaLane,
        # state, direction, length). Be permissive across SUMO versions.
        is_open = False
        if len(link) > 2:
            try:
                is_open = bool(link[2])
            except Exception:
                is_open = False
        state = str(link[5]).lower() if len(link) > 5 else ""
        if is_open or state in {"g", "m", "o"}:
            return True

    return False


def vehicle_has_clear_open_path(veh_id, lane_id, current_edge, distance_to_end):
    """A stopped lead car should not be considered validly stopped if it has
    a green/open link and clear downstream space.
    """
    next_edge = planned_next_edge_from_route(veh_id, current_edge)
    if next_edge is None or next_edge.startswith(":"):
        return False

    if not lane_has_connection_to_edge(lane_id, next_edge):
        return False

    if not lane_link_is_open_to_edge(lane_id, next_edge):
        return False

    if vehicle_has_close_leader(veh_id, max(6.0, min(UNJUSTIFIED_STOP_LEADER_DISTANCE, distance_to_end + 4.0))):
        return False

    if not next_edge_has_exit_space(next_edge):
        return False

    if not internal_junction_path_is_clear(lane_id, next_edge):
        return False

    return True
'''
if 'def vehicle_has_clear_open_path' not in s:
    if marker not in s:
        raise SystemExit('Could not find has_valid_reason marker')
    s = s.replace(marker, helper + marker, 1)

# 11) Modify has_valid_reason_to_be_stopped to allow watchdog for mismatch/open green.
old = '''    # The first vehicle approaching a signalized stop line can legitimately
    # stop inside the signal approach zone. For unsignalized junctions, use a
    # smaller generic junction buffer.
    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        return True

    if distance_to_end <= UNJUSTIFIED_STOP_JUNCTION_DISTANCE:
        return True
'''
new = '''    next_edge = planned_next_edge_from_route(veh_id, current_edge)

    # Wrong-lane route mismatch is not a valid reason to freeze. Let the
    # watchdog route/lane rescue handle it.
    if next_edge is not None and not next_edge.startswith(":"):
        if not lane_has_connection_to_edge(lane_id, next_edge):
            return False

    # If the lead car has an open/green link and downstream space, it should not
    # stay stopped merely because it is near an intersection.
    if vehicle_has_clear_open_path(veh_id, lane_id, current_edge, distance_to_end):
        return False

    # The first vehicle approaching a red/blocked stop line can legitimately
    # stop inside the signal approach zone. For unsignalized junctions, use a
    # smaller generic junction buffer.
    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        return True

    if distance_to_end <= UNJUSTIFIED_STOP_JUNCTION_DISTANCE:
        return True
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'vehicle_has_clear_open_path(veh_id' not in s:
    raise SystemExit('Could not patch has_valid_reason')

# 12) In the traffic-light lane-change lock, don't hard-lock a vehicle in a lane that cannot reach route.
old = '''    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        try:
            traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_LOCKED_LANE_CHANGE_MODE)
            TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.add(veh_id)
            return True
        except traci.TraCIException:
            return False
'''
new = '''    if inside_traffic_light_no_lane_change_zone(lane_id, distance_to_end):
        current_edge = lane_to_edge(lane_id)
        next_edge = planned_next_edge_from_route(veh_id, current_edge) if current_edge else None
        if next_edge is not None and not next_edge.startswith(":"):
            if not lane_has_connection_to_edge(lane_id, next_edge):
                # Wrong lane near the stop line: leave lane changing/recovery
                # enabled so rescue_vehicle_from_unconnected_lane can rewrite
                # the route or make a safe lane change instead of freezing.
                try:
                    traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_NORMAL_LANE_CHANGE_MODE)
                except traci.TraCIException:
                    pass
                TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.discard(veh_id)
                return False
        try:
            traci.vehicle.setLaneChangeMode(veh_id, TRAFFIC_LIGHT_LOCKED_LANE_CHANGE_MODE)
            TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES.add(veh_id)
            return True
        except traci.TraCIException:
            return False
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'Wrong lane near the stop line' not in s:
    print('Warning: did not patch traffic light lock; pattern not found')

# 13) Add recovery tracking cleanup if emergency state not needed (not applicable); ensure compare reset clears approach counts if existing reset list in compare.

src.write_text(s)
print(f'Patched {src} with boundary-aware OD balancing and lane recovery fixes.')
