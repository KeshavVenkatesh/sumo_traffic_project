import argparse
import os
import random
from collections import defaultdict
from xml.sax.saxutils import quoteattr

import traci
import traffic_rl_all_intersections as m


DEFAULT_TLS = "cluster_12179861947_12179861948_12179861949_12185616643_#11more"

APPROACHES = ["NB", "SB", "EB", "WB"]
MOVEMENTS = ["L", "S", "R"]

TURN_PROPORTIONS = {
    "L": 0.15,
    "S": 0.70,
    "R": 0.15,
}


def lane_to_edge(lane_id):
    if not lane_id:
        return None

    if lane_id.startswith(":"):
        return None

    if "_" not in lane_id:
        return lane_id

    return lane_id.rsplit("_", 1)[0]


def edge_allows_passenger(edge_id):
    if not edge_id or edge_id.startswith(":"):
        return False

    try:
        lane_count = traci.edge.getLaneNumber(edge_id)

        if lane_count <= 0:
            return False

        lane_id = f"{edge_id}_0"
        allowed = traci.lane.getAllowed(lane_id)
        disallowed = traci.lane.getDisallowed(lane_id)

        if not allowed:
            return "passenger" not in disallowed and "private" not in disallowed

        return (
            "passenger" in allowed
            or "private" in allowed
            or "car" in allowed
        )

    except traci.TraCIException:
        return False


def build_movement_edge_pairs(tls_id):
    """
    Builds edge pairs for movements through the target intersection.

    Example:
        {
            "NB-S": [(incoming_edge, outgoing_edge), ...],
            "NB-L": [...],
            ...
        }
    """
    movement_pairs = {
        f"{approach}-{movement}": set()
        for approach in APPROACHES
        for movement in MOVEMENTS
    }

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for signal_links in controlled_links:
        for link in signal_links:
            if len(link) < 2:
                continue

            incoming_lane = link[0]
            outgoing_lane = link[1]

            if not incoming_lane or not outgoing_lane:
                continue

            in_edge = lane_to_edge(incoming_lane)
            out_edge = lane_to_edge(outgoing_lane)

            if not in_edge or not out_edge:
                continue

            try:
                in_vec = m.lane_direction_vector(incoming_lane, incoming=True)
                out_vec = m.lane_direction_vector(outgoing_lane, incoming=False)

                if in_vec is None or out_vec is None:
                    continue

                approach = m.classify_approach(in_vec)
                angle = m.signed_turn_angle(in_vec, out_vec)
                movement = m.classify_movement(angle)
                label = f"{approach}-{movement}"

                if label in movement_pairs and in_edge != out_edge:
                    movement_pairs[label].add((in_edge, out_edge))

            except traci.TraCIException:
                continue

    return {
        label: sorted(pairs)
        for label, pairs in movement_pairs.items()
    }


def get_far_destination_edges(excluded_edges):
    edges = []

    for edge_id in traci.edge.getIDList():
        if edge_id.startswith(":"):
            continue

        if edge_id in excluded_edges:
            continue

        if not edge_allows_passenger(edge_id):
            continue

        try:
            lane_id = f"{edge_id}_0"
            length = traci.lane.getLength(lane_id)

            if length < 25:
                continue

            edges.append(edge_id)

        except traci.TraCIException:
            continue

    return sorted(edges)


def inspect_network(tls_id):
    """
    Starts SUMO briefly to inspect the target TLS and network edges.
    """
    sumo_cmd = [
        m.SUMO_HEADLESS_BINARY,
        "-n", m.NET_FILE,
        "--start",
        "--step-length", str(m.STEP_LENGTH),
        "--end", "1",
        *m.QUIET_SUMO_ARGS,
    ]

    traci.start(sumo_cmd)

    try:
        movement_pairs = build_movement_edge_pairs(tls_id)

        intersection_edges = set()

        for pairs in movement_pairs.values():
            for in_edge, out_edge in pairs:
                intersection_edges.add(in_edge)
                intersection_edges.add(out_edge)

        destination_edges = get_far_destination_edges(intersection_edges)

        if not destination_edges:
            raise RuntimeError("No valid far destination edges were found.")

        return movement_pairs, destination_edges

    finally:
        traci.close()


def choose_far_destination(rng, destination_edges, forbidden_edges):
    for _ in range(100):
        edge = rng.choice(destination_edges)

        if edge not in forbidden_edges:
            return edge

    return rng.choice(destination_edges)


def period_for_turn(base_approach_period, movement):
    """
    Converts a desired 70/15/15 split into flow periods.

    Smaller period = more cars.

    If base_approach_period means one total car every X seconds from an approach,
    then:
        straight period = X / 0.70
        left period     = X / 0.15
        right period    = X / 0.15
    """
    proportion = TURN_PROPORTIONS[movement]
    return base_approach_period / proportion


def write_realistic_route_file(
    output_file,
    movement_pairs,
    destination_edges,
    base_approach_period,
    begin,
    end,
    seed,
):
    rng = random.Random(seed)

    lines = []
    lines.append("<routes>")

    lines.append(
        '''    <vType id="realistic_car"
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
           jmDriveAfterRedTime="-1"/>'''
    )

    flow_count = 0

    print()
    print("Generated realistic turn-pattern flows:")
    print("Target proportions per approach: 70% straight, 15% left, 15% right")
    print("-" * 110)

    for approach in APPROACHES:
        for movement in MOVEMENTS:
            label = f"{approach}-{movement}"
            pairs = movement_pairs.get(label, [])

            if not pairs:
                continue

            movement_period = period_for_turn(base_approach_period, movement)

            # If SUMO has multiple equivalent lane/edge pairs for the same movement,
            # split the demand across those pairs.
            pair_period = movement_period * len(pairs)

            for pair_index, (from_edge, via_edge) in enumerate(pairs):
                destination = choose_far_destination(
                    rng,
                    destination_edges,
                    forbidden_edges={from_edge, via_edge},
                )

                flow_id = f"realistic_{label}_{pair_index}"

                lines.append(
                    "    "
                    f"<flow id={quoteattr(flow_id)} "
                    f"type={quoteattr('realistic_car')} "
                    f"begin={quoteattr(str(begin))} "
                    f"end={quoteattr(str(end))} "
                    f"period={quoteattr(f'{pair_period:.3f}')} "
                    f"from={quoteattr(from_edge)} "
                    f"to={quoteattr(destination)} "
                    f"via={quoteattr(via_edge)} "
                    f"departLane={quoteattr('best')} "
                    f"departPos={quoteattr('free')} "
                    f"departSpeed={quoteattr('max')}/>"
                )

                print(
                    f"{label:5} "
                    f"target_share={TURN_PROPORTIONS[movement] * 100:5.1f}% "
                    f"period={pair_period:8.3f}s | "
                    f"{from_edge} -> via {via_edge} -> {destination}"
                )

                flow_count += 1

    lines.append("</routes>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("-" * 110)
    print(f"Wrote {flow_count} flows to: {output_file}")


def build_transition_map(tls_id):
    """
    Maps actual lane transitions through the watched intersection:

        (incoming_lane, outgoing_lane) -> movement label
    """
    transition_map = {}

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for signal_links in controlled_links:
        for link in signal_links:
            if len(link) < 2:
                continue

            in_lane = link[0]
            out_lane = link[1]

            if not in_lane or not out_lane:
                continue

            try:
                in_vec = m.lane_direction_vector(in_lane, incoming=True)
                out_vec = m.lane_direction_vector(out_lane, incoming=False)

                if in_vec is None or out_vec is None:
                    continue

                approach = m.classify_approach(in_vec)
                angle = m.signed_turn_angle(in_vec, out_vec)
                movement = m.classify_movement(angle)
                label = f"{approach}-{movement}"

                transition_map[(in_lane, out_lane)] = label

            except traci.TraCIException:
                pass

    return transition_map


def print_results(tls_id, movement_counts):
    total = sum(movement_counts.values())

    print()
    print(f"Turn movement counts for TLS: {tls_id}")
    print("=" * 70)

    for label in m.MOVEMENT_LABELS:
        count = movement_counts[label]
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"{label:5}: {count:6d} vehicles   {pct:6.2f}%")

    print("=" * 70)
    print(f"TOTAL: {total} vehicles")

    straight = sum(
        movement_counts[label]
        for label in m.MOVEMENT_LABELS
        if label.endswith("-S")
    )

    left = sum(
        movement_counts[label]
        for label in m.MOVEMENT_LABELS
        if label.endswith("-L")
    )

    right = sum(
        movement_counts[label]
        for label in m.MOVEMENT_LABELS
        if label.endswith("-R")
    )

    print()
    print("Grouped totals:")

    straight_pct = 100.0 * straight / total if total else 0.0
    left_pct = 100.0 * left / total if total else 0.0
    right_pct = 100.0 * right / total if total else 0.0

    print(f"Straight: {straight:6d} vehicles   {straight_pct:6.2f}%")
    print(f"Left:     {left:6d} vehicles   {left_pct:6.2f}%")
    print(f"Right:    {right:6d} vehicles   {right_pct:6.2f}%")


def run_simulation(args):
    binary = m.SUMO_GUI_BINARY if args.gui else m.SUMO_HEADLESS_BINARY

    if args.gui:
        m.ensure_xquartz()

    route_files = [args.route_file]

    if args.include_background:
        route_files.insert(0, m.BACKGROUND_ROUTE_FILE)

    routes = ",".join(route_files)

    sumo_cmd = [
        binary,
        "-n", m.NET_FILE,
        "-r", routes,
        "--start",
        "--step-length", str(m.STEP_LENGTH),
        "--end", str(args.end),
        "--max-num-vehicles", str(args.max_vehicles),
        "--max-depart-delay", str(m.MAX_DEPART_DELAY),
        "--time-to-teleport", str(m.TIME_TO_TELEPORT),
        "--ignore-route-errors", "true",
        *m.QUIET_SUMO_ARGS,
    ]

    print()
    print("Starting SUMO:")
    print(" ".join(sumo_cmd))
    print()
    print(f"Simulation stops at t = {args.end:.1f}s")
    print(f"Printing time every {args.print_every_steps} simulation timesteps.")
    print()

    traci.start(sumo_cmd)

    try:
        transition_map = build_transition_map(args.tls)

        if not transition_map:
            raise RuntimeError(
                f"No controlled lane transitions found for TLS: {args.tls}"
            )

        incoming_lanes = {
            in_lane
            for in_lane, _ in transition_map.keys()
        }

        movement_counts = defaultdict(int)
        last_controlled_in_lane = {}
        counted_vehicle_crossings = set()

        step_count = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()

            if sim_time >= args.end:
                print()
                print(f"Reached stop time: t = {sim_time:.1f}s")
                break

            traci.simulationStep()
            step_count += 1

            sim_time = traci.simulation.getTime()

            if step_count % args.print_every_steps == 0:
                print(
                    f"simulation_step={step_count}, "
                    f"simulation_time={sim_time:.1f}s, "
                    f"active_vehicles={traci.vehicle.getIDCount()}"
                )

            for veh_id in traci.vehicle.getIDList():
                try:
                    lane_id = traci.vehicle.getLaneID(veh_id)
                except traci.TraCIException:
                    continue

                if lane_id in incoming_lanes:
                    last_controlled_in_lane[veh_id] = lane_id
                    continue

                if lane_id.startswith(":"):
                    continue

                if veh_id not in last_controlled_in_lane:
                    continue

                in_lane = last_controlled_in_lane[veh_id]
                pair = (in_lane, lane_id)

                if pair not in transition_map:
                    continue

                crossing_key = (veh_id, in_lane, lane_id)

                if crossing_key not in counted_vehicle_crossings:
                    label = transition_map[pair]
                    movement_counts[label] += 1
                    counted_vehicle_crossings.add(crossing_key)

                del last_controlled_in_lane[veh_id]

        print_results(args.tls, movement_counts)

    finally:
        traci.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tls",
        default=DEFAULT_TLS,
        help="Traffic light ID to generate/check realistic turn movements for.",
    )

    parser.add_argument(
        "--route-file",
        default="realistic_turn_pattern.rou.xml",
        help="Generated route file.",
    )

    parser.add_argument(
        "--base-approach-period",
        type=float,
        default=2.0,
        help=(
            "Average total traffic rate per approach. "
            "Smaller = more cars. Example: 2.0 means about one car every "
            "2 seconds per approach before applying 70/15/15 split."
        ),
    )

    parser.add_argument(
        "--begin",
        type=float,
        default=0.0,
        help="Flow begin time.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=3600.0,
        help="Stop simulation after this many simulation seconds.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for far destination selection.",
    )

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=3000,
        help="Maximum active vehicles.",
    )

    parser.add_argument(
        "--print-every-steps",
        type=int,
        default=100,
        help="Print simulation time every this many SUMO timesteps.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI.",
    )

    parser.add_argument(
        "--include-background",
        action="store_true",
        help=(
            "Also include the old randomTrips background route. "
            "Leave this off if you want the 70/15/15 pattern to dominate."
        ),
    )

    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate the route file; do not run SUMO.",
    )

    args = parser.parse_args()

    print(f"Inspecting target traffic light: {args.tls}")

    movement_pairs, destination_edges = inspect_network(args.tls)

    write_realistic_route_file(
        output_file=args.route_file,
        movement_pairs=movement_pairs,
        destination_edges=destination_edges,
        base_approach_period=args.base_approach_period,
        begin=args.begin,
        end=args.end,
        seed=args.seed,
    )

    if args.generate_only:
        return

    run_simulation(args)


if __name__ == "__main__":
    main()
