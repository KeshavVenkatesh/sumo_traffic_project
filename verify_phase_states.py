import argparse
import os
import sys

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
import traffic_rl_model as m


# ============================================================
# Signal-index analysis
# ============================================================

def labels_by_signal_index(tls_id):
    """
    Returns:
        {
            signal_index: {
                "labels": {"NB-S", "NB-R", ...},
                "links": [
                    {
                        "label": "NB-S",
                        "incoming_lane": "...",
                        "outgoing_lane": "...",
                        "angle": ...
                    },
                    ...
                ]
            }
        }

    This is important because SUMO traffic-light state strings are indexed by
    signal index, not by individual lane. One signal index may control multiple
    lane-to-lane connections.
    """
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    state_length = len(traci.trafficlight.getRedYellowGreenState(tls_id))

    result = {
        i: {
            "labels": set(),
            "links": [],
        }
        for i in range(state_length)
    }

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

            try:
                in_vec = m.lane_direction_vector(incoming_lane, incoming=True)
                out_vec = m.lane_direction_vector(outgoing_lane, incoming=False)

                if in_vec is None or out_vec is None:
                    continue

                approach = m.classify_approach(in_vec)
                angle = m.signed_turn_angle(in_vec, out_vec)
                movement = m.classify_movement(angle)
                label = f"{approach}-{movement}"

                result[signal_index]["labels"].add(label)
                result[signal_index]["links"].append({
                    "label": label,
                    "incoming_lane": incoming_lane,
                    "outgoing_lane": outgoing_lane,
                    "angle": angle,
                })

            except traci.TraCIException:
                pass

    return result


def present_right_turn_labels(labels_by_idx):
    labels = set()

    for data in labels_by_idx.values():
        for label in data["labels"]:
            if label.endswith("-R"):
                labels.add(label)

    return labels


def print_signal_index_table(labels_by_idx):
    print("\nSignal-index movement table:")
    print("-" * 100)

    for idx, data in labels_by_idx.items():
        labels = sorted(data["labels"])

        if not labels:
            continue

        print(f"index {idx:>2}: {labels}")

        for link in data["links"]:
            print(
                f"          {link['label']:>5} | "
                f"{link['incoming_lane']} -> {link['outgoing_lane']} "
                f"(angle={link['angle']:.1f})"
            )


# ============================================================
# Phase verification
# ============================================================

def expected_allowed_labels_by_slot(all_right_turns):
    """
    These are the ONLY movement labels allowed in each phase.

    Right turns are allowed in every phase.
    Left turns are not allowed during straight phases.
    Straights are not allowed during protected-left phases.
    """
    return {
        0: {"NB-L", "SB-L"} | all_right_turns,
        1: {"NB-S", "SB-S"} | all_right_turns,
        2: {"EB-L", "WB-L"} | all_right_turns,
        3: {"EB-S", "WB-S"} | all_right_turns,
    }


def expected_required_core_labels_by_slot():
    """
    These are the non-right-turn movements that must be active in each phase.
    """
    return {
        0: {"NB-L", "SB-L"},
        1: {"NB-S", "SB-S"},
        2: {"EB-L", "WB-L"},
        3: {"EB-S", "WB-S"},
    }


def verify_phase(controller, phase, labels_by_idx, allowed_by_slot, required_core_by_slot):
    state, active_indices = m.build_state_from_movements(
        controller["state_length"],
        controller["movement_map"],
        phase["rules"],
    )

    slot = phase["slot"]
    allowed_labels = allowed_by_slot[slot]
    required_core_labels = required_core_by_slot[slot]

    actual_active_labels = set()
    violations = []
    active_right_turns = set()

    for idx, char in enumerate(state):
        if char not in ("G", "g"):
            continue

        labels = labels_by_idx.get(idx, {}).get("labels", set())
        actual_active_labels.update(labels)

        for label in labels:
            if label.endswith("-R"):
                active_right_turns.add(label)

        bad_labels = labels - allowed_labels

        if bad_labels:
            violations.append({
                "index": idx,
                "light": char,
                "all_labels": labels,
                "bad_labels": bad_labels,
                "links": labels_by_idx.get(idx, {}).get("links", []),
            })

    missing_core = required_core_labels - actual_active_labels

    return {
        "state": state,
        "active_indices": active_indices,
        "actual_active_labels": actual_active_labels,
        "allowed_labels": allowed_labels,
        "required_core_labels": required_core_labels,
        "active_right_turns": active_right_turns,
        "missing_core": missing_core,
        "violations": violations,
    }


def print_phase_result(phase, result, all_right_turns):
    print("\n" + "=" * 100)
    print(phase["name"])
    print(f"State string: {result['state']}")
    print(f"Active movement labels: {sorted(result['actual_active_labels'])}")
    print(f"Allowed movement labels: {sorted(result['allowed_labels'])}")

    missing_right_turns = all_right_turns - result["active_right_turns"]

    failed = False

    if result["missing_core"]:
        failed = True
        print("\nMISSING REQUIRED CORE MOVEMENTS:")
        for label in sorted(result["missing_core"]):
            print(f"  {label}")

    if missing_right_turns:
        failed = True
        print("\nMISSING RIGHT TURNS:")
        for label in sorted(missing_right_turns):
            print(f"  {label}")

    if result["violations"]:
        failed = True
        print("\nFORBIDDEN MOVEMENTS TURNED GREEN:")
        for violation in result["violations"]:
            print(
                f"  index={violation['index']}, "
                f"light={violation['light']}, "
                f"all_labels={sorted(violation['all_labels'])}, "
                f"bad_labels={sorted(violation['bad_labels'])}"
            )

            for link in violation["links"]:
                print(
                    f"      {link['label']:>5} | "
                    f"{link['incoming_lane']} -> {link['outgoing_lane']} "
                    f"(angle={link['angle']:.1f})"
                )

    if failed:
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")

    return not failed


def verify_controller(tls_id, print_table=False):
    controller = m.build_controller_for_tls(tls_id, activate=False)

    if controller is None:
        raise RuntimeError(f"{tls_id} is not usable.")

    labels_by_idx = labels_by_signal_index(tls_id)

    if print_table:
        print_signal_index_table(labels_by_idx)

    all_right_turns = present_right_turn_labels(labels_by_idx)

    allowed_by_slot = expected_allowed_labels_by_slot(all_right_turns)
    required_core_by_slot = expected_required_core_labels_by_slot()

    print(f"\nChecking target TLS: {tls_id}")
    print(f"State length: {controller['state_length']}")
    print(f"Detected right-turn labels: {sorted(all_right_turns)}")

    expected_slots = {0, 1, 2, 3}
    actual_slots = {phase["slot"] for phase in controller["phases"]}

    all_passed = True

    if actual_slots != expected_slots:
        all_passed = False
        print("\nPHASE SLOT ERROR:")
        print(f"  Expected slots: {sorted(expected_slots)}")
        print(f"  Actual slots:   {sorted(actual_slots)}")

    for phase in controller["phases"]:
        result = verify_phase(
            controller,
            phase,
            labels_by_idx,
            allowed_by_slot,
            required_core_by_slot,
        )

        phase_passed = print_phase_result(
            phase,
            result,
            all_right_turns,
        )

        if not phase_passed:
            all_passed = False

    print("\n" + "=" * 100)

    if all_passed:
        print("FINAL RESULT: PASS")
        print("The four phase templates are safe according to the SUMO signal-index mapping.")
        print("It is reasonable to retrain.")
    else:
        print("FINAL RESULT: FAIL")
        print("Do NOT retrain yet.")
        print("At least one phase allows a forbidden movement or is missing a required movement.")

    return all_passed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tls",
        default=m.TARGET_TLS_ID,
        help="Traffic light ID to verify.",
    )

    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the full signal-index to movement-label table.",
    )

    args = parser.parse_args()

    # Important:
    # We want to verify the phase templates, not temporary downstream blockage.
    # So force all exit-space checks to pass during verification.
    m.lanes_have_space = lambda out_lanes: True

    sumo_cmd = [
        m.SUMO_HEADLESS_BINARY,
        "-n", m.NET_FILE,
        "-r", f"{m.BACKGROUND_ROUTE_FILE},{m.AMBULANCE_ROUTE_FILE}",
        "--start",
        "--step-length", str(m.STEP_LENGTH),
        "--end", "10",
        *m.QUIET_SUMO_ARGS,
        "--log", m.SUMO_RUN_LOG,
        "--error-log", m.SUMO_ERROR_LOG,
    ]

    traci.start(sumo_cmd)

    try:
        verify_controller(args.tls, print_table=args.print_table)

    finally:
        traci.close()


if __name__ == "__main__":
    main()
