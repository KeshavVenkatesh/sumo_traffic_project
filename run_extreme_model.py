import argparse
import os
import sys
from xml.sax.saxutils import quoteattr

import numpy as np

import traffic_rl_model as m
import traci


APPROACHES = ["NB", "SB", "EB", "WB"]
MOVEMENTS = ["S", "L", "R"]

# Straight gets most of the traffic, left/right get somewhat less.
MOVEMENT_PERIOD_MULTIPLIER = {
    "S": 1.0,
    "L": 1.6,
    "R": 1.8,
}


def lane_to_edge(lane_id):
    """
    SUMO lane ids look like:
        edge_id_0
        edge_id_1
        417109161#0_2

    The edge id is everything before the final underscore.
    """
    if not lane_id:
        return None

    if lane_id.startswith(":"):
        return None

    if "_" not in lane_id:
        return lane_id

    return lane_id.rsplit("_", 1)[0]


def get_target_movement_map(tls_id):
    """
    Starts SUMO briefly so we can inspect the trained intersection's lane links.
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
        state_length, movement_map = m.classify_tls_movements(tls_id)
        return state_length, movement_map
    finally:
        traci.close()


def edge_pairs_for_movement(movement_map, movement_label):
    """
    Returns edge pairs for a movement such as EB-S or NB-L:
        [(incoming_edge, outgoing_edge), ...]
    """
    pairs = set()

    for signal_index, lane_sets in movement_map.get(movement_label, {}).items():
        for in_lane in lane_sets["in"]:
            for out_lane in lane_sets["out"]:
                in_edge = lane_to_edge(in_lane)
                out_edge = lane_to_edge(out_lane)

                if in_edge and out_edge and in_edge != out_edge:
                    pairs.add((in_edge, out_edge))

    return sorted(pairs)


def parse_approaches(raw):
    approaches = []

    for part in raw.split(","):
        part = part.strip().upper()

        if not part:
            continue

        if part not in APPROACHES:
            raise ValueError(
                f"Invalid approach {part}. Use one or more of: NB, SB, EB, WB"
            )

        approaches.append(part)

    if not approaches:
        raise ValueError("At least one heavy approach is required.")

    return set(approaches)


def write_extreme_route_file(
    output_file,
    movement_map,
    heavy_approaches,
    heavy_period,
    light_period,
    begin,
    end,
):
    """
    Creates a custom asymmetric traffic file.

    Smaller period = more vehicles.
    Example:
        heavy_period = 0.7 means roughly one vehicle every 0.7 seconds.
        light_period = 8.0 means roughly one vehicle every 8 seconds.
    """
    lines = []
    lines.append("<routes>")

    lines.append(
        '''    <vType id="car"
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

    print("\nGenerated extreme traffic flows:")

    for approach in APPROACHES:
        for movement in MOVEMENTS:
            label = f"{approach}-{movement}"
            pairs = edge_pairs_for_movement(movement_map, label)

            if not pairs:
                continue

            base_period = heavy_period if approach in heavy_approaches else light_period
            period = base_period * MOVEMENT_PERIOD_MULTIPLIER[movement]

            for pair_index, (from_edge, to_edge) in enumerate(pairs):
                flow_id = f"extreme_{label}_{pair_index}"

                lines.append(
                    "    "
                    f"<flow id={quoteattr(flow_id)} "
                    f"type={quoteattr('car')} "
                    f"begin={quoteattr(str(begin))} "
                    f"end={quoteattr(str(end))} "
                    f"period={quoteattr(f'{period:.3f}')} "
                    f"from={quoteattr(from_edge)} "
                    f"to={quoteattr(to_edge)} "
                    f"departLane={quoteattr('best')} "
                    f"departPos={quoteattr('free')} "
                    f"departSpeed={quoteattr('max')}/>"
                )

                intensity = "HEAVY" if approach in heavy_approaches else "light"
                print(
                    f"  {intensity:5} {label:4} "
                    f"period={period:.3f}s | {from_edge} -> {to_edge}"
                )

                flow_count += 1

    lines.append("</routes>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {flow_count} flows to: {output_file}")


def action_mask_for_controller(controller):
    mask = np.zeros(5, dtype=bool)
    mask[0] = True

    for phase in controller["phases"]:
        mask[phase["slot"] + 1] = True

    return mask


def run_model_on_extreme_route(args):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        raise ImportError(
            "Missing sb3-contrib. Run:\n"
            "python3 -m pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    if not args.nogui:
        m.ensure_xquartz()

    binary = m.SUMO_HEADLESS_BINARY if args.nogui else m.SUMO_GUI_BINARY

    route_file = f"{args.route_file},{m.AMBULANCE_ROUTE_FILE}"

    sumo_cmd = [
        binary,
        "-n", m.NET_FILE,
        "-r", route_file,
        "--start",
        "--step-length", str(m.STEP_LENGTH),
        "--end", str(args.end),
        "--max-num-vehicles", str(args.max_vehicles),
        "--max-depart-delay", str(m.MAX_DEPART_DELAY),
        "--time-to-teleport", str(m.TIME_TO_TELEPORT),
        *m.QUIET_SUMO_ARGS,
    ]

    print("\nStarting extreme-traffic simulation:")
    print(" ".join(sumo_cmd))

    model = MaskablePPO.load(args.model)

    traci.start(sumo_cmd)

    try:
        controller = m.build_controller_for_tls(args.tls, activate=True)

        if controller is None:
            raise RuntimeError(f"Traffic light {args.tls} is not usable.")

        next_print_time = 0.0

        while traci.simulation.getMinExpectedNumber() > 0:
            obs = m.get_observation(controller)
            mask = action_mask_for_controller(controller)

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )

            action = int(action)
            switched = False

            if action > 0:
                desired_slot = action - 1

                if desired_slot in controller["slot_to_pos"]:
                    desired_phase_pos = controller["slot_to_pos"][desired_slot]

                    can_switch = (
                        controller["phase_elapsed"] >= m.MIN_GREEN_BEFORE_SWITCH
                    )

                    if can_switch and desired_phase_pos != controller["phase_pos"]:
                        switched = m.switch_to_phase(
                            controller["tls_id"],
                            controller,
                            desired_phase_pos,
                        )

            if controller["phase_elapsed"] >= m.MAX_GREEN_HOLD:
                next_pos = (
                    controller["phase_pos"] + 1
                ) % len(controller["phases"])

                switched = m.switch_to_phase(
                    controller["tls_id"],
                    controller,
                    next_pos,
                )

            m.run_steps(
                m.DECISION_INTERVAL,
                controller["tls_id"],
                controller,
            )

            sim_time = traci.simulation.getTime()

            if sim_time >= next_print_time:
                current_phase = controller["phases"][controller["phase_pos"]]
                total_wait, total_queue = m.total_controlled_wait_and_queue(controller)
                reward = m.compute_reward(controller, switched)

                print(
                    f"t={sim_time:7.1f}, "
                    f"active={traci.vehicle.getIDCount():4d}, "
                    f"phase={current_phase['name']}, "
                    f"action={action}, "
                    f"queue={total_queue:.0f}, "
                    f"wait={total_wait:.1f}, "
                    f"reward={reward:.2f}"
                )

                next_print_time += args.print_every

    finally:
        try:
            traci.close()
        except Exception:
            pass

        print("\nExtreme-traffic simulation ended.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tls",
        default=m.TARGET_TLS_ID,
        help="Traffic light ID to stress-test.",
    )

    parser.add_argument(
        "--model",
        default="randomized_four_way_model_no_left_conflict_old_scheme",
        help="Trained model path without or with .zip.",
    )

    parser.add_argument(
        "--heavy-approaches",
        default="EB",
        help="Comma-separated heavy approaches. Example: EB or EB,WB or NB.",
    )

    parser.add_argument(
        "--heavy-period",
        type=float,
        default=0.7,
        help="Vehicle period for heavy approaches. Smaller = more traffic.",
    )

    parser.add_argument(
        "--light-period",
        type=float,
        default=8.0,
        help="Vehicle period for light approaches. Larger = less traffic.",
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
        help="Simulation end time and flow end time.",
    )

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=3000,
        help="Maximum active vehicles allowed in SUMO.",
    )

    parser.add_argument(
        "--route-file",
        default="extreme_asymmetric.rou.xml",
        help="Output route file to generate and then run.",
    )

    parser.add_argument(
        "--print-every",
        type=float,
        default=30.0,
        help="How often to print simulation stats.",
    )

    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run without GUI.",
    )

    args = parser.parse_args()

    heavy_approaches = parse_approaches(args.heavy_approaches)

    print(f"Inspecting TLS movement map for: {args.tls}")
    _, movement_map = get_target_movement_map(args.tls)

    write_extreme_route_file(
        output_file=args.route_file,
        movement_map=movement_map,
        heavy_approaches=heavy_approaches,
        heavy_period=args.heavy_period,
        light_period=args.light_period,
        begin=args.begin,
        end=args.end,
    )

    run_model_on_extreme_route(args)


if __name__ == "__main__":
    main()
