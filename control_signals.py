import os
import traci

# Use full path if "sumo-gui" ever fails.
SUMO_BINARY = "sumo-gui"
# SUMO_BINARY = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin/sumo-gui"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NET_FILE = os.path.join(BASE_DIR, "map.net.xml")
BACKGROUND_ROUTE_FILE = os.path.join(BASE_DIR, "background.rou.xml")
AMBULANCE_ROUTE_FILE = os.path.join(BASE_DIR, "ambulance_random.rou.xml")
ROUTE_FILE = f"{BACKGROUND_ROUTE_FILE},{AMBULANCE_ROUTE_FILE}"

SUMO_RUN_LOG = os.path.join(BASE_DIR, "sumo_run.log")
SUMO_ERROR_LOG = os.path.join(BASE_DIR, "sumo_error.log")

# Leave this problematic intersection alone.
EXCLUDED_TLS = {
    "1080494809"
}

MIN_GREEN = 6
MAX_GREEN = 60
DEFAULT_GREEN = 20

YELLOW_TIME = 5
CLEARANCE_TIME = 3

PRIORITY_POWER = 3.5
LOW_TRAFFIC_THRESHOLD = 2
LOW_TRAFFIC_WEIGHT = 0.01

AMBULANCE_LOOKAHEAD = 180.0
AMBULANCE_PRIORITY_BONUS = 50.0

# Anti-gridlock: require space after the intersection before giving green.
REQUIRED_EXIT_GAP = 18.0
SPACE_WAIT_TIME = 1

MAX_NUM_VEHICLES = 1200
MAX_DEPART_DELAY = 60
TIME_TO_TELEPORT = 300
SIM_END = 7200


def get_green_phases(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = logic.phases

    green_phases = []

    for i, phase in enumerate(phases):
        state = phase.state
        has_green = "G" in state or "g" in state
        has_yellow = "y" in state or "Y" in state

        if has_green and not has_yellow:
            green_phases.append(i)

    return green_phases


def has_yellow_phase(tls_id):
    phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases

    for phase in phases:
        if "y" in phase.state or "Y" in phase.state:
            return True

    return False


def get_queue_for_phase(tls_id, phase_index):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = logic.phases

    if phase_index >= len(phases):
        return 0

    state = phases[phase_index].state
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)

    lanes_to_count = set()

    for i, signal_char in enumerate(state):
        if i >= len(controlled_lanes):
            continue

        if signal_char in ("G", "g"):
            lanes_to_count.add(controlled_lanes[i])

    total_queue = 0

    for lane_id in lanes_to_count:
        try:
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        except traci.TraCIException:
            pass

    return total_queue


def outgoing_lane_has_space(out_lane_id, required_gap=REQUIRED_EXIT_GAP):
    """
    Checks whether there is enough empty space immediately after the intersection.

    Lane position is measured from the start of the outgoing lane.
    The vehicle closest to the intersection exit has the smallest position.
    """
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(out_lane_id)

        if not veh_ids:
            return True

        closest_pos = min(
            traci.vehicle.getLanePosition(veh_id)
            for veh_id in veh_ids
        )

        return closest_pos >= required_gap

    except traci.TraCIException:
        return True


def phase_has_exit_space(tls_id, phase_index):
    """
    A green phase is only allowed if all outgoing lanes for green movements
    have enough room for vehicles to fully clear the intersection.
    """
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = logic.phases

    if phase_index >= len(phases):
        return False

    state = phases[phase_index].state
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    for i, signal_char in enumerate(state):
        if signal_char not in ("G", "g"):
            continue

        if i >= len(controlled_links):
            continue

        for link in controlled_links[i]:
            # Usually: (incomingLane, outgoingLane, internalLane)
            if len(link) < 2:
                continue

            outgoing_lane = link[1]

            if not outgoing_lane_has_space(outgoing_lane):
                return False

    return True


def find_next_green_with_exit_space(tls_id, data):
    """
    Starting from the current green position, find the next green phase whose
    outgoing lanes have enough space.
    """
    green_phases = data["green_phases"]
    n = len(green_phases)
    start = data["green_pos"]

    for offset in range(n):
        pos = (start + offset) % n
        phase_index = green_phases[pos]

        if phase_has_exit_space(tls_id, phase_index):
            data["green_pos"] = pos
            return phase_index

    return None


def get_ambulance_bonus_for_phase(tls_id, phase_index):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = logic.phases

    if phase_index >= len(phases):
        return 0.0

    state = phases[phase_index].state
    bonus = 0.0

    for veh_id in traci.vehicle.getIDList():
        try:
            is_ambulance = (
                veh_id.startswith("ambulance_")
                or traci.vehicle.getTypeID(veh_id) == "ambulance"
            )

            if not is_ambulance:
                continue

            next_tls_list = traci.vehicle.getNextTLS(veh_id)

            for next_tls in next_tls_list:
                next_tls_id = next_tls[0]
                tls_link_index = next_tls[1]
                distance = next_tls[2]

                if next_tls_id != tls_id:
                    continue

                if distance > AMBULANCE_LOOKAHEAD:
                    continue

                if tls_link_index >= len(state):
                    continue

                if state[tls_link_index] in ("G", "g"):
                    closeness = 1.0 + (AMBULANCE_LOOKAHEAD - distance) / AMBULANCE_LOOKAHEAD
                    bonus += AMBULANCE_PRIORITY_BONUS * closeness

        except traci.TraCIException:
            continue

    return bonus


def phase_weight(tls_id, phase_index):
    # Critical "do not block the box" rule.
    # If cars cannot fit after the intersection, this phase gets zero priority.
    if not phase_has_exit_space(tls_id, phase_index):
        return 0.0

    queue = get_queue_for_phase(tls_id, phase_index)
    ambulance_bonus = get_ambulance_bonus_for_phase(tls_id, phase_index)

    if queue <= LOW_TRAFFIC_THRESHOLD and ambulance_bonus == 0:
        return LOW_TRAFFIC_WEIGHT

    return (queue + ambulance_bonus) ** PRIORITY_POWER


def choose_green_duration(tls_id, selected_phase, green_phases):
    weights = [phase_weight(tls_id, phase) for phase in green_phases]
    total_weight = sum(weights)

    if total_weight <= 0:
        return SPACE_WAIT_TIME

    if total_weight <= LOW_TRAFFIC_WEIGHT * len(weights):
        return DEFAULT_GREEN

    selected_index = green_phases.index(selected_phase)
    selected_weight = weights[selected_index]

    duration = MIN_GREEN + (MAX_GREEN - MIN_GREEN) * selected_weight / total_weight
    return max(MIN_GREEN, min(MAX_GREEN, int(duration)))


def yellow_phase_after(tls_id, green_phase_index):
    phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
    candidate = green_phase_index + 1

    if candidate < len(phases):
        state = phases[candidate].state
        if "y" in state or "Y" in state:
            return candidate

    return None


def set_green(tls_id, data):
    phase_index = find_next_green_with_exit_space(tls_id, data)

    if phase_index is None:
        # No green movement has safe exit space.
        # Keep the current transition/wait state and re-check shortly.
        data["mode"] = "clearance_wait"
        data["remaining"] = SPACE_WAIT_TIME

        print(f"{tls_id}: no green phase has enough exit space; waiting")
        return

    green_phases = data["green_phases"]
    duration = choose_green_duration(tls_id, phase_index, green_phases)

    traci.trafficlight.setPhase(tls_id, phase_index)
    traci.trafficlight.setPhaseDuration(tls_id, duration)

    data["mode"] = "green"
    data["current_green_phase"] = phase_index
    data["remaining"] = duration

    queue = get_queue_for_phase(tls_id, phase_index)
    ambulance_bonus = get_ambulance_bonus_for_phase(tls_id, phase_index)

    print(
        f"{tls_id}: GREEN phase={phase_index}, "
        f"duration={duration}, queue={queue}, "
        f"ambulance_bonus={ambulance_bonus:.1f}, "
        f"exit_space_ok=True"
    )


def set_yellow_or_clearance(tls_id, data):
    yellow_phase = yellow_phase_after(tls_id, data["current_green_phase"])

    if yellow_phase is not None:
        traci.trafficlight.setPhase(tls_id, yellow_phase)
        traci.trafficlight.setPhaseDuration(tls_id, YELLOW_TIME + CLEARANCE_TIME)

        data["mode"] = "yellow_clearance"
        data["remaining"] = YELLOW_TIME + CLEARANCE_TIME
    else:
        # Should be rare. We skip controlling TLS with no yellow phase during setup.
        data["mode"] = "clearance_wait"
        data["remaining"] = CLEARANCE_TIME


def advance_tls_controller(tls_id, data):
    data["remaining"] -= 1

    if data["mode"] == "green":
        if data["remaining"] <= 0:
            set_yellow_or_clearance(tls_id, data)

    elif data["mode"] in ("yellow_clearance", "clearance_wait"):
        if data["remaining"] <= 0:
            data["green_pos"] = (data["green_pos"] + 1) % len(data["green_phases"])
            set_green(tls_id, data)


def print_recent_log_lines(path, label, n=40):
    print(f"\n--- Last {n} lines of {label} ---")

    if not os.path.exists(path):
        print(f"{path} does not exist.")
        return

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines[-n:]:
            print(line.rstrip())

    except Exception as e:
        print(f"Could not read {path}: {e}")


def main():
    sumo_cmd = [
        SUMO_BINARY,
        "-n", NET_FILE,
        "-r", ROUTE_FILE,
        "--start",
        "--end", str(SIM_END),
        "--max-num-vehicles", str(MAX_NUM_VEHICLES),
        "--max-depart-delay", str(MAX_DEPART_DELAY),
        "--time-to-teleport", str(TIME_TO_TELEPORT),
        "--log", SUMO_RUN_LOG,
        "--error-log", SUMO_ERROR_LOG,
    ]

    print("Starting SUMO with command:")
    print(" ".join(sumo_cmd))

    traci.start(sumo_cmd)

    tls_ids = list(traci.trafficlight.getIDList())
    controllers = {}

    print("\nTraffic light controllers:")

    for tls_id in tls_ids:
        if tls_id in EXCLUDED_TLS:
            print(f"{tls_id}: EXCLUDED, using SUMO default signal timing")
            continue

        green_phases = get_green_phases(tls_id)

        if len(green_phases) < 2:
            print(f"{tls_id}: skipped, not enough green phases")
            continue

        if not has_yellow_phase(tls_id):
            print(f"{tls_id}: skipped, no yellow phase found")
            continue

        controllers[tls_id] = {
            "green_phases": green_phases,
            "green_pos": 0,
            "mode": "green",
            "current_green_phase": green_phases[0],
            "remaining": DEFAULT_GREEN,
        }

        print(f"{tls_id}: controlled, green phases = {green_phases}")
        set_green(tls_id, controllers[tls_id])

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            for tls_id, data in list(controllers.items()):
                try:
                    advance_tls_controller(tls_id, data)
                except traci.TraCIException as e:
                    print(f"Non-fatal TraCI error while controlling TLS {tls_id}: {e}")
                    print(f"Disabling controller for {tls_id}.")
                    controllers.pop(tls_id, None)

    except traci.exceptions.FatalTraCIError as e:
        print("\nSUMO closed the TraCI connection.")
        print("This usually means SUMO crashed, exited, or hit a simulation error.")
        print("Check the logs below.")
        print(f"Python-side error: {e}")

        print_recent_log_lines(SUMO_ERROR_LOG, "sumo_error.log")
        print_recent_log_lines(SUMO_RUN_LOG, "sumo_run.log")

    finally:
        try:
            traci.close()
        except Exception:
            pass

        print("\nSimulation ended or connection closed.")
        print(f"SUMO run log: {SUMO_RUN_LOG}")
        print(f"SUMO error log: {SUMO_ERROR_LOG}")


if __name__ == "__main__":
    main()
