#!/usr/bin/env python3
"""
Run the original/native traffic-light programs stored in the SUMO .net.xml map,
while still using this project's dynamic OD vehicle generation and safety logic.

This file is intentionally separate from compare_fixed_vs_single_vs_all_model_realistic.py.
It does NOT load the RL model, does NOT build generated safe traffic-light phases,
and does NOT call traci.trafficlight.setRedYellowGreenState().

Usage example:
    python3 run_native_sumo_signals.py \
      --gui \
      --episode-seconds 3600 \
      --eval-steps 10000 \
      --compare-seeds 42 \
      --max-vehicle-center 1000 \
      --target-vehicle-center 1000 \
      --initial-vehicle-center 250 \
      --spawn-batch-center 12 \
      --metrics-interval 10 \
      --eval-print-every 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import realistic_all_intersections_fixed_cycle as sim
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import realistic_all_intersections_fixed_cycle.py.\n"
        "Put this file in the same folder as realistic_all_intersections_fixed_cycle.py."
    ) from exc


# -----------------------------------------------------------------------------
# Small helpers copied/simplified from the comparison runner.
# -----------------------------------------------------------------------------


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw).split(','):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [42]


def safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def runtime_interval_steps(args: argparse.Namespace, attr: str, default_seconds: float) -> int:
    seconds = float(getattr(args, attr, default_seconds) or default_seconds)
    return max(1, int(round(seconds / float(getattr(sim, "STEP_LENGTH", 1.0)))))


def reset_sim_globals() -> None:
    """Clear mutable module-level state before each SUMO episode."""
    for name in (
        "KEEP_CLEAR_HELD_VEHICLES",
        "KEEP_CLEAR_HOLD_START_TIME",
        "KEEP_CLEAR_FORCE_RELEASE_UNTIL",
        "TRAFFIC_LIGHT_LANE_CHANGE_LOCKED_VEHICLES",
        "ROUTE_LANE_COMMITTED_VEHICLES",
        "ROUTE_LANE_COMMITTED_EDGE",
        "EMERGENCY_ROUTE_REROUTE_LAST",
        "APPROACH_TURN_DECISIONS",
        "APPROACH_TURN_COUNTS",
        "APPROACH_TURN_COUNTS_BY_EDGE",
        "VEHICLE_EDGE_HISTORY",
        "VEHICLE_LAST_EDGE",
        "LANE_BALANCE_LAST_CHANGE",
        "UNJUSTIFIED_STOP_TRACKING",
        "UNJUSTIFIED_STOP_LAST_ACTION",
    ):
        obj = getattr(sim, name, None)
        if hasattr(obj, "clear"):
            obj.clear()

    if hasattr(sim, "TURN_LANE_PREFERENCE_INDEX"):
        sim.TURN_LANE_PREFERENCE_INDEX = {}
    if hasattr(sim, "APPROACH_DECISION_INDEX"):
        sim.APPROACH_DECISION_INDEX = {}
    if hasattr(sim, "TRAFFIC_LIGHT_APPROACH_LANES"):
        sim.TRAFFIC_LIGHT_APPROACH_LANES = set()


def apply_sim_distance_globals(args: argparse.Namespace) -> None:
    """Match the same global-distance updates the main simulation CLI performs."""
    if hasattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE"):
        sim.INTERSECTION_NO_LANE_CHANGE_DISTANCE = max(0.0, float(args.intersection_no_lane_change_distance))
    if hasattr(sim, "INTERSECTION_LANE_PREP_DISTANCE"):
        sim.INTERSECTION_LANE_PREP_DISTANCE = max(
            getattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE", 0.0),
            float(args.intersection_lane_prep_distance),
        )
    if hasattr(sim, "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE"):
        sim.TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE = max(
            getattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE", 0.0),
            float(args.tls_no_lane_change_distance),
        )
    if hasattr(sim, "TRAFFIC_LIGHT_LANE_PREP_DISTANCE"):
        sim.TRAFFIC_LIGHT_LANE_PREP_DISTANCE = max(
            getattr(sim, "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE", 0.0),
            getattr(sim, "INTERSECTION_LANE_PREP_DISTANCE", 0.0),
            float(args.tls_lane_prep_distance),
        )


def make_sim_args(user_args: argparse.Namespace, route_file: str, seed: int) -> SimpleNamespace:
    """Build the args namespace expected by realistic_all_intersections_fixed_cycle.py."""
    args = SimpleNamespace()
    args.route_file = route_file
    args.end = float(user_args.episode_seconds)
    args.seed = int(seed)
    args.gui = bool(user_args.gui)

    args.max_vehicles = min(int(user_args.max_vehicle_center), int(getattr(sim, "MAX_ACTIVE_VEHICLE_CAP", user_args.max_vehicle_center)))
    args.target_vehicles = min(int(user_args.target_vehicle_center), args.max_vehicles)
    args.initial_vehicles = min(int(user_args.initial_vehicle_center), args.target_vehicles)
    args.spawn_batch = int(user_args.spawn_batch_center)
    args.spawn_attempts = int(user_args.spawn_attempts)

    args.route_lookahead_edges = int(user_args.route_lookahead_edges)
    args.min_remaining_edges = int(user_args.min_remaining_edges)
    args.recovery_attempts = int(user_args.recovery_attempts)
    args.spawn_grid_size = int(user_args.spawn_grid_size)

    args.local_road_penalty = float(user_args.local_road_penalty)
    args.local_to_local_penalty = float(user_args.local_to_local_penalty)
    args.leave_local_bonus = float(user_args.leave_local_bonus)
    args.non_core_penalty = float(user_args.non_core_penalty)
    args.signal_timing_jitter = float(user_args.signal_timing_jitter)
    args.max_depart_delay = int(user_args.max_depart_delay)
    args.time_to_teleport = int(user_args.time_to_teleport)
    args.green_duration = float(user_args.green_duration_center)
    args.max_consecutive_straight = int(user_args.max_consecutive_straight)
    args.print_every = float(user_args.print_every)
    args.disable_strict_split = bool(user_args.disable_strict_split)
    args.generate_only = False
    args.run_existing = False

    args.routing_mode = str(user_args.routing_mode)
    args.od_route_attempts = int(user_args.od_route_attempts)
    args.od_boundary_margin_fraction = float(user_args.od_boundary_margin_fraction)
    args.od_min_euclidean_distance = float(user_args.od_min_euclidean_distance)
    args.od_min_route_distance = float(user_args.od_min_route_distance)
    args.od_min_zone_separation = int(user_args.od_min_zone_separation)
    args.od_max_local_middle_fraction = float(user_args.od_max_local_middle_fraction)
    args.od_local_middle_trim_edges = int(user_args.od_local_middle_trim_edges)
    args.od_through_trip_probability = float(user_args.od_through_trip_probability)
    args.od_access_trip_probability = float(user_args.od_access_trip_probability)
    args.od_long_local_trip_probability = float(user_args.od_long_local_trip_probability)
    args.od_min_edge_length = float(user_args.od_min_edge_length)
    args.od_random_walk_fallback = bool(user_args.od_random_walk_fallback)
    args.od_no_random_walk_fallback = not args.od_random_walk_fallback
    args.depart_lane = str(user_args.depart_lane)

    args.intersection_no_lane_change_distance = float(user_args.intersection_no_lane_change_distance)
    args.intersection_lane_prep_distance = float(user_args.intersection_lane_prep_distance)
    args.tls_no_lane_change_distance = float(user_args.tls_no_lane_change_distance)
    args.tls_lane_prep_distance = float(user_args.tls_lane_prep_distance)

    args.unjustified_stop_watchdog = bool(user_args.unjustified_stop_watchdog)
    args.unjustified_stop_check_interval = float(user_args.unjustified_stop_check_interval)
    args.unjustified_stop_speed = float(user_args.unjustified_stop_speed)
    args.unjustified_stop_min_time = float(user_args.unjustified_stop_min_time)

    args.disable_ambulances = bool(user_args.disable_ambulances)
    args.ambulance_interval = float(user_args.ambulance_interval)
    args.ambulance_min_euclidean_distance = float(user_args.ambulance_min_euclidean_distance)
    args.ambulance_min_route_distance = float(user_args.ambulance_min_route_distance)
    args.ambulance_min_route_edges = int(user_args.ambulance_min_route_edges)
    args.ambulance_route_attempts = int(user_args.ambulance_route_attempts)
    args.ambulance_depart_lane = str(user_args.ambulance_depart_lane)
    args.ambulance_depart_pos = str(user_args.ambulance_depart_pos)
    args.ambulance_poi_radius = float(user_args.ambulance_poi_radius)
    args.ambulance_debug = bool(user_args.ambulance_debug)

    return args


def network_wait_queue_speed() -> tuple[float, float, float]:
    total_wait = 0.0
    total_queue = 0.0
    total_speed = 0.0
    count = 0
    for veh_id in sim.traci.vehicle.getIDList():
        try:
            if str(veh_id).startswith("ambulance_"):
                continue
            speed = sim.traci.vehicle.getSpeed(veh_id)
            wait = sim.traci.vehicle.getWaitingTime(veh_id)
        except sim.traci.TraCIException:
            continue
        count += 1
        total_speed += speed
        total_wait += wait
        if speed < sim.QUEUE_SPEED_THRESHOLD:
            total_queue += 1.0
    avg_speed = total_speed / count if count else 0.0
    return total_wait, total_queue, avg_speed


def collect_sample(samples: list[dict[str, float]], info: dict[str, Any]) -> None:
    samples.append({
        "sim_time": float(info.get("sim_time", 0.0) or 0.0),
        "active_vehicles": float(info.get("active_vehicles", 0) or 0),
        "global_queue": float(info.get("global_queue", 0.0) or 0.0),
        "global_wait": float(info.get("global_wait", 0.0) or 0.0),
        "avg_speed": float(info.get("avg_speed", 0.0) or 0.0),
        "arrived": float(info.get("arrived", 0) or 0),
        "spawned": float(info.get("spawned", 0) or 0),
        "extended": float(info.get("extended", 0) or 0),
        "recovered": float(info.get("recovered", 0) or 0),
        "total_arrived": float(info.get("total_arrived", 0) or 0),
        "ambulance_count": float(info.get("ambulance_count", 0) or 0),
    })


def summarize_samples(label: str, seed: int, samples: list[dict[str, float]]) -> dict[str, float | int | str]:
    if not samples:
        return {"controller": label, "seed": seed, "sim_seconds": 0.0, "samples": 0}

    def col(name: str) -> list[float]:
        return [float(s.get(name, 0.0)) for s in samples]

    return {
        "controller": label,
        "seed": seed,
        "sim_seconds": col("sim_time")[-1],
        "samples": len(samples),
        "total_arrived": int(col("total_arrived")[-1]),
        "arrived_this_window_sum": int(sum(col("arrived"))),
        "spawned_total": int(sum(col("spawned"))),
        "recovered_total": int(sum(col("recovered"))),
        "extended_total": int(sum(col("extended"))),
        "mean_active_vehicles": safe_mean(col("active_vehicles")),
        "mean_avg_speed_mps": safe_mean(col("avg_speed")),
        "mean_global_queue": safe_mean(col("global_queue")),
        "max_global_queue": safe_max(col("global_queue")),
        "mean_global_wait": safe_mean(col("global_wait")),
        "max_global_wait": safe_max(col("global_wait")),
        "mean_ambulance_count": safe_mean(col("ambulance_count")),
    }


class NativeSignalEpisode:
    """One SUMO run that leaves native .net.xml traffic-light programs untouched."""

    def __init__(self, user_args: argparse.Namespace, seed: int, env_rank: int = 0):
        self.user_args = user_args
        self.seed = int(seed)
        self.env_rank = int(env_rank)
        self.route_file = os.path.join(
            sim.BASE_DIR,
            f"native_sumo_signals_{os.getpid()}_{self.env_rank}.rou.xml",
        )
        self.args = make_sim_args(user_args, self.route_file, self.seed)
        self.rng = random.Random(self.seed)
        self.turn_counts: Counter[str] = Counter()
        self.controllers: list[dict[str, Any]] = []
        self.native_tls_ids: list[str] = []
        self.sim_state: dict[str, Any] = {}
        self.main_start_edges: Any = None
        self.turn_index: Any = None
        self.raw_graph: dict[str, list[str]] = {}
        self.edge_metadata: dict[str, Any] = {}
        self.core_edges: set[str] = set()
        self.total_arrived = 0
        self.started = False

    def _sumo_cmd(self) -> list[str]:
        binary = sim.SUMO_GUI_BINARY if self.user_args.gui else sim.SUMO_HEADLESS_BINARY
        return [
            binary,
            "-n", sim.NET_FILE,
            "-r", self.route_file,
            "--start",
            "--step-length", str(sim.STEP_LENGTH),
            "--end", str(float(self.user_args.episode_seconds)),
            "--seed", str(self.seed),
            "--max-num-vehicles", str(self.args.max_vehicles),
            "--max-depart-delay", str(self.args.max_depart_delay),
            "--time-to-teleport", str(self.args.time_to_teleport),
            "--ignore-route-errors", "true",
            "--quit-on-end", "false",
            *sim.QUIET_SUMO_ARGS,
        ]

    def reset(self) -> dict[str, Any]:
        reset_sim_globals()
        apply_sim_distance_globals(self.args)
        sim.write_empty_route_file(self.route_file)

        if self.user_args.gui:
            sim.ensure_xquartz()

        sim.traci.start(self._sumo_cmd())
        self.started = True

        # This is the key line: do not call sim.build_all_fixed_controllers().
        # SUMO will therefore keep and advance the original <tlLogic> programs
        # embedded in the map's .net.xml file.
        self.controllers = []
        self.native_tls_ids = list(sim.traci.trafficlight.getIDList())
        sim.rebuild_traffic_light_approach_lanes(self.controllers)

        valid_edges = sim.get_valid_passenger_edges()
        self.edge_metadata = sim.build_edge_metadata(valid_edges)
        self.raw_graph = sim.build_raw_successor_graph(valid_edges)
        self.raw_graph = sim.remove_hardcoded_loop_region_from_graph(self.raw_graph)
        self.core_edges = set(self.raw_graph)

        start_candidates = list(self.raw_graph.keys())
        if not start_candidates:
            raise RuntimeError("No valid start edges were found.")

        self.main_start_edges = sim.build_spawn_zones(
            start_edges=start_candidates,
            edge_metadata=self.edge_metadata,
            grid_size=self.args.spawn_grid_size,
        ) or start_candidates

        self.turn_index = sim.build_turn_decision_index(
            controllers=self.controllers,
            raw_graph=self.raw_graph,
        )

        od_context = None
        if sim.use_od_routing(self.args):
            od_context = sim.build_od_context(
                valid_edges=valid_edges,
                raw_graph=self.raw_graph,
                edge_metadata=self.edge_metadata,
                args=self.args,
            )

        sim.APPROACH_DECISION_INDEX = sim.build_approach_decision_index(self.raw_graph)
        sim.TURN_LANE_PREFERENCE_INDEX = sim.APPROACH_DECISION_INDEX

        self.sim_state = {
            "next_vehicle_id": 0,
            "next_route_id": 0,
            "next_spawn_zone_index": 0,
            "next_od_origin_zone_index": 0,
            "next_lane_pref_time": 0.0,
            "next_lane_balance_time": 0.0,
            "next_unconnected_lane_rescue_time": 0.0,
            "next_unjustified_stop_check_time": 0.0,
            "next_ambulance_spawn": float("inf") if self.args.disable_ambulances else 0.0,
            "active_ambulances": {},
            "od_context": od_context,
            "od_trip_counts": Counter(),
            "od_movement_counts": Counter(),
            "od_route_failures": 0,
        }

        sim.fill_vehicle_population(
            sim_state=self.sim_state,
            target_count=self.args.initial_vehicles,
            max_to_spawn=self.args.initial_vehicles,
            start_edges=self.main_start_edges,
            turn_index=self.turn_index,
            raw_graph=self.raw_graph,
            edge_metadata=self.edge_metadata,
            core_edges=self.core_edges,
            rng=self.rng,
            turn_counts=self.turn_counts,
            args=self.args,
        )

        self.total_arrived = 0
        return self.info()

    def close(self) -> None:
        if self.started:
            try:
                sim.traci.close(False)
            except Exception:
                pass
        self.started = False

    def info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "total_arrived": self.total_arrived,
            "ambulance_count": len(self.sim_state.get("active_ambulances", {})),
        }
        if self.started:
            try:
                info.update(
                    sim_time=sim.traci.simulation.getTime(),
                    active_vehicles=sim.traci.vehicle.getIDCount(),
                )
            except Exception:
                pass
        info.update(extra)
        return info


def run_native_episode(user_args: argparse.Namespace, seed: int) -> dict[str, float | int | str]:
    episode = NativeSignalEpisode(user_args, seed=seed, env_rank=0)
    samples: list[dict[str, float]] = []

    try:
        episode.reset()
        decision_steps = max(1, int(round(sim.DECISION_INTERVAL / sim.STEP_LENGTH)))
        metrics_period_steps = runtime_interval_steps(user_args, "metrics_interval", 10.0)

        print()
        print("=" * 88)
        print(f"Native SUMO signal run, seed={seed}")
        print("=" * 88)
        print(f"network file:           {sim.NET_FILE}")
        print(f"native traffic lights:  {len(episode.native_tls_ids)}")
        print(f"max active vehicles:    {episode.args.max_vehicles}")
        print(f"target active vehicles: {episode.args.target_vehicles}")
        print("signal control:         ORIGINAL .net.xml <tlLogic> programs")
        print("generated safe phases:  OFF")
        print("RL model:               OFF")
        print("=" * 88)
        print()

        last_global_wait = 0.0
        last_global_queue = 0.0
        last_avg_speed = 0.0
        last_info = episode.info(
            spawned=0,
            extended=0,
            recovered=0,
            arrived=0,
            global_wait=0.0,
            global_queue=0.0,
            avg_speed=0.0,
        )

        for step in range(1, int(user_args.eval_steps) + 1):
            # No controller loop here. SUMO advances the native traffic-light
            # programs automatically during simulationStep().
            arrived, spawned, extended, recovered = sim.run_simulation_steps(
                num_steps=decision_steps,
                controllers=episode.controllers,
                start_edges=episode.main_start_edges,
                turn_index=episode.turn_index,
                raw_graph=episode.raw_graph,
                edge_metadata=episode.edge_metadata,
                core_edges=episode.core_edges,
                rng=episode.rng,
                turn_counts=episode.turn_counts,
                sim_state=episode.sim_state,
                args=episode.args,
            )
            episode.total_arrived += arrived

            do_metrics = (
                step == 1
                or step % metrics_period_steps == 0
                or step % int(user_args.eval_print_every) == 0
            )
            if do_metrics:
                last_global_wait, last_global_queue, last_avg_speed = network_wait_queue_speed()
                last_info = episode.info(
                    spawned=spawned,
                    extended=extended,
                    recovered=recovered,
                    arrived=arrived,
                    global_wait=last_global_wait,
                    global_queue=last_global_queue,
                    avg_speed=last_avg_speed,
                )
                collect_sample(samples, last_info)
            else:
                last_info = episode.info(
                    spawned=spawned,
                    extended=extended,
                    recovered=recovered,
                    arrived=arrived,
                    global_wait=last_global_wait,
                    global_queue=last_global_queue,
                    avg_speed=last_avg_speed,
                )

            if step % int(user_args.eval_print_every) == 0:
                print(
                    f"[native seed {seed}] step={step:6d}, "
                    f"t={float(last_info.get('sim_time', 0.0)):8.1f}, "
                    f"active={int(last_info.get('active_vehicles', 0)):4d}, "
                    f"gq={float(last_info.get('global_queue', 0.0)):7.1f}, "
                    f"gw={float(last_info.get('global_wait', 0.0)):9.1f}, "
                    f"arrived={int(last_info.get('total_arrived', 0)):5d}"
                )

            if float(last_info.get("sim_time", 0.0)) >= float(user_args.episode_seconds):
                break

    finally:
        episode.close()

    row = summarize_samples("native_sumo", seed, samples)
    print()
    print("Summary")
    print("-" * 88)
    for key, value in row.items():
        print(f"{key:24}: {value}")
    print()
    return row


def write_results(prefix: str, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return

    json_path = Path(f"{prefix}.json")
    csv_path = Path(f"{prefix}.csv")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dynamic traffic with the original/native SUMO traffic-light programs from the map."
    )

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episode-seconds", type=int, default=3600)
    parser.add_argument("--eval-steps", type=int, default=10000)
    parser.add_argument("--compare-seeds", type=str, default="42")
    parser.add_argument("--eval-print-every", type=int, default=50)
    parser.add_argument("--metrics-interval", type=float, default=10.0)
    parser.add_argument("--output-prefix", type=str, default="native_sumo_signal_results")

    parser.add_argument("--max-vehicle-center", type=int, default=int(getattr(sim, "MAX_ACTIVE_VEHICLE_CAP", 1000)))
    parser.add_argument("--target-vehicle-center", type=int, default=650)
    parser.add_argument("--initial-vehicle-center", type=int, default=200)
    parser.add_argument("--spawn-batch-center", type=int, default=12)
    parser.add_argument("--spawn-attempts", type=int, default=8)

    parser.add_argument("--route-lookahead-edges", type=int, default=60)
    parser.add_argument("--min-remaining-edges", type=int, default=15)
    parser.add_argument("--recovery-attempts", type=int, default=20)
    parser.add_argument("--spawn-grid-size", type=int, default=6)
    parser.add_argument("--green-duration-center", type=float, default=30.0, help="Unused for native signals, kept for compatibility.")
    parser.add_argument("--signal-timing-jitter", type=float, default=0.15, help="Unused for native signals, kept for compatibility.")
    parser.add_argument("--max-depart-delay", type=int, default=300)
    parser.add_argument("--time-to-teleport", type=int, default=180)
    parser.add_argument("--max-consecutive-straight", type=int, default=4)
    parser.add_argument("--print-every", type=float, default=60.0)
    parser.add_argument("--disable-strict-split", action="store_true")

    parser.add_argument("--local-road-penalty", type=float, default=0.04)
    parser.add_argument("--local-to-local-penalty", type=float, default=0.15)
    parser.add_argument("--leave-local-bonus", type=float, default=8.0)
    parser.add_argument("--non-core-penalty", type=float, default=1.0)

    parser.add_argument("--routing-mode", choices=[getattr(sim, "ROUTING_MODE_OD", "od"), getattr(sim, "ROUTING_MODE_RANDOM_WALK", "random-walk")], default=getattr(sim, "ROUTING_MODE_OD", "od"))
    parser.add_argument("--od-route-attempts", type=int, default=int(getattr(sim, "OD_ROUTE_ATTEMPTS", 15)))
    parser.add_argument("--od-boundary-margin-fraction", type=float, default=float(getattr(sim, "OD_BOUNDARY_MARGIN_FRACTION", 0.13)))
    parser.add_argument("--od-min-euclidean-distance", type=float, default=float(getattr(sim, "OD_MIN_EUCLIDEAN_DISTANCE", 900.0)))
    parser.add_argument("--od-min-route-distance", type=float, default=float(getattr(sim, "OD_MIN_ROUTE_DISTANCE", 1200.0)))
    parser.add_argument("--od-min-zone-separation", type=int, default=int(getattr(sim, "OD_MIN_ZONE_SEPARATION", 2)))
    parser.add_argument("--od-max-local-middle-fraction", type=float, default=float(getattr(sim, "OD_MAX_LOCAL_MIDDLE_FRACTION", 0.35)))
    parser.add_argument("--od-local-middle-trim-edges", type=int, default=int(getattr(sim, "OD_LOCAL_MIDDLE_TRIM_EDGES", 2)))
    parser.add_argument("--od-through-trip-probability", type=float, default=float(getattr(sim, "OD_THROUGH_TRIP_PROBABILITY", 0.72)))
    parser.add_argument("--od-access-trip-probability", type=float, default=float(getattr(sim, "OD_ACCESS_TRIP_PROBABILITY", 0.23)))
    parser.add_argument("--od-long-local-trip-probability", type=float, default=float(getattr(sim, "OD_LONG_LOCAL_TRIP_PROBABILITY", 0.05)))
    parser.add_argument("--od-min-edge-length", type=float, default=float(getattr(sim, "OD_MIN_EDGE_LENGTH", 20.0)))
    parser.add_argument("--od-random-walk-fallback", action="store_true", default=bool(getattr(sim, "OD_RANDOM_WALK_FALLBACK", False)))
    parser.add_argument("--depart-lane", type=str, default=str(getattr(sim, "OD_DEPART_LANE", "best")))

    parser.add_argument("--intersection-no-lane-change-distance", type=float, default=float(getattr(sim, "INTERSECTION_NO_LANE_CHANGE_DISTANCE", 75.0)))
    parser.add_argument("--intersection-lane-prep-distance", type=float, default=float(getattr(sim, "INTERSECTION_LANE_PREP_DISTANCE", 550.0)))
    parser.add_argument("--tls-no-lane-change-distance", type=float, default=float(getattr(sim, "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE", 90.0)))
    parser.add_argument("--tls-lane-prep-distance", type=float, default=float(getattr(sim, "TRAFFIC_LIGHT_LANE_PREP_DISTANCE", 650.0)))

    parser.add_argument("--unjustified-stop-watchdog", action="store_true", default=bool(getattr(sim, "UNJUSTIFIED_STOP_WATCHDOG_ENABLED", True)))
    parser.add_argument("--disable-unjustified-stop-watchdog", action="store_false", dest="unjustified_stop_watchdog")
    parser.add_argument("--unjustified-stop-check-interval", type=float, default=float(getattr(sim, "UNJUSTIFIED_STOP_CHECK_INTERVAL", 1.0)))
    parser.add_argument("--unjustified-stop-speed", type=float, default=float(getattr(sim, "UNJUSTIFIED_STOP_SPEED", 0.20)))
    parser.add_argument("--unjustified-stop-min-time", type=float, default=float(getattr(sim, "UNJUSTIFIED_STOP_MIN_TIME", 3.0)))

    parser.add_argument("--disable-ambulances", action="store_true")
    parser.add_argument("--ambulance-interval", type=float, default=float(getattr(sim, "AMBULANCE_SPAWN_INTERVAL", 120.0)))
    parser.add_argument("--ambulance-min-euclidean-distance", type=float, default=float(getattr(sim, "AMBULANCE_MIN_EUCLIDEAN_DISTANCE", 1500.0)))
    parser.add_argument("--ambulance-min-route-distance", type=float, default=float(getattr(sim, "AMBULANCE_MIN_ROUTE_DISTANCE", 1800.0)))
    parser.add_argument("--ambulance-min-route-edges", type=int, default=int(getattr(sim, "AMBULANCE_MIN_ROUTE_EDGES", 20)))
    parser.add_argument("--ambulance-route-attempts", type=int, default=int(getattr(sim, "AMBULANCE_ROUTE_ATTEMPTS", 100)))
    parser.add_argument("--ambulance-depart-lane", type=str, default=str(getattr(sim, "AMBULANCE_DEPART_LANE", "free")))
    parser.add_argument("--ambulance-depart-pos", type=str, default=str(getattr(sim, "AMBULANCE_DEPART_POS", "random_free")))
    parser.add_argument("--ambulance-poi-radius", type=float, default=float(getattr(sim, "AMBULANCE_POI_RADIUS", 250.0)))
    parser.add_argument("--ambulance-debug", action="store_true")

    # Compatibility no-ops so you can paste old compare commands and remove only
    # the model-specific intent mentally. These arguments are ignored here.
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--vecnormalize-path", type=str, default=None)
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-single-model", action="store_true")
    parser.add_argument("--skip-all-model", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unsupported arguments: {' '.join(unknown)}")

    seeds = parse_seed_list(args.compare_seeds)
    rows = []
    start_wall = time.time()
    for seed in seeds:
        rows.append(run_native_episode(args, seed))
    write_results(args.output_prefix, rows)
    print(f"Total wall time: {time.time() - start_wall:.1f}s")


if __name__ == "__main__":
    main()
