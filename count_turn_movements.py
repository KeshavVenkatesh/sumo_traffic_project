import argparse
from collections import defaultdict

import traci
import traffic_rl_all_intersections as m


def build_transition_map(tls_id):
    """
    Maps actual lane transitions through the intersection:

        (incoming_lane, outgoing_lane) -> movement label

    Example:
        ("417109161#0_2", "732311662#1_1") -> "NB-L"
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tls",
        default="cluster_12179861947_12179861948_12179861949_12185616643_#11more",
        help="Traffic light ID to count turn movements for.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=3600.0,
        help="Stop after this many simulation seconds.",
    )

    parser.add_argument(
        "--print-every-steps",
        type=int,
        default=100,
        help="Print simulation time every this many simulation timesteps.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI.",
    )

    args = parser.parse_args()

    binary = m.SUMO_GUI_BINARY if args.gui else m.SUMO_HEADLESS_BINARY

    if args.gui:
        m.ensure_xquartz()

    sumo_cmd = [
        binary,
        "-n", m.NET_FILE,
        "-r", m.BACKGROUND_ROUTE_FILE,
        "--start",
        "--step-length", str(m.STEP_LENGTH),
        "--end", str(args.end),
        "--max-num-vehicles", str(m.MAX_NUM_VEHICLES),
        "--max-depart-delay", str(m.MAX_DEPART_DELAY),
        "--time-to-teleport", str(m.TIME_TO_TELEPORT),
        *m.QUIET_SUMO_ARGS,
    ]

    print("Starting SUMO:")
    print(" ".join(sumo_cmd))
    print()
    print(f"Counting turn movements for: {args.tls}")
    print(f"Simulation will stop at t = {args.end:.1f} seconds.")
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

        # For each vehicle, remember the last controlled incoming lane it was on.
        last_controlled_in_lane = {}

        # Prevent counting the same vehicle crossing more than once.
        counted_vehicle_crossings = set()

        step_count = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()

            if sim_time >= args.end:
                print()
                print(f"Reached stop time: t = {sim_time:.1f} seconds.")
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

                # Vehicle is on one of the incoming lanes of the watched TLS.
                if lane_id in incoming_lanes:
                    last_controlled_in_lane[veh_id] = lane_id
                    continue

                # Ignore internal junction lanes.
                if lane_id.startswith(":"):
                    continue

                # Vehicle has not recently been on a watched incoming lane.
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


if __name__ == "__main__":
    main()
