#!/usr/bin/env python3
"""
Compare the trained copied all-TLS model against the original/native SUMO
traffic-light programs stored in the map's .net.xml file.

This file is intentionally separate from compare_fixed_vs_single_vs_all_model_realistic.py.
It reuses that file's all-model runner and scenario setup, but adds a native-SUMO
baseline that does NOT build generated safe traffic-light controllers and does
NOT call traci.trafficlight.setRedYellowGreenState().

Native baseline:
    - vehicle generation: same dynamic OD/spawn logic as the project
    - traffic lights: original .net.xml <tlLogic> programs advanced by SUMO
    - RL model: off
    - generated safe phases: off

All-model:
    - vehicle generation: same dynamic OD/spawn logic
    - traffic lights: generated safe phases from the project
    - RL model: copied independently to every compatible TLS

Example:
    python3 compare_native_sumo_vs_all_model.py \
      --episode-seconds 3600 \
      --eval-steps 10000 \
      --compare-seeds 42,43,44 \
      --max-vehicle-center 1500 \
      --target-vehicle-center 1500 \
      --initial-vehicle-center 300 \
      --spawn-batch-center 20 \
      --model-path models/traffic_signal_maskable_ppo_fast_proxy_strong \
      --vecnormalize-path models/traffic_signal_maskable_ppo_fast_proxy_strong_vecnormalize.pkl \
      --model-update-period 10 \
      --metrics-interval 10 \
      --eval-print-every 50 \
      --stats-csv native_vs_model_1500.csv \
      --stats-json native_vs_model_1500.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import compare_fixed_vs_single_vs_all_model_realistic as cmp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import compare_fixed_vs_single_vs_all_model_realistic.py.\n"
        "Put this file in the same folder as your current compare file."
    ) from exc

sim = cmp.sim


class NativeSignalEpisode:
    """One SUMO run that leaves native .net.xml traffic-light programs untouched."""

    def __init__(
        self,
        scenario: cmp.TrafficScenario,
        seed: int,
        args: argparse.Namespace,
        gui: bool = False,
        env_rank: int = 0,
    ):
        self.scenario = scenario
        self.seed = int(seed)
        self.outer_args = args
        self.gui = bool(gui)
        self.env_rank = int(env_rank)
        self.route_file = os.path.join(
            sim.BASE_DIR,
            f"native_sumo_vs_model_{os.getpid()}_{self.env_rank}.rou.xml",
        )
        self.args = cmp.make_sim_args(
            scenario=scenario,
            route_file=self.route_file,
            episode_seconds=int(args.episode_seconds),
            gui=bool(gui),
        )
        self.rng = random.Random(self.scenario.seed)
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
        binary = sim.SUMO_GUI_BINARY if self.gui else sim.SUMO_HEADLESS_BINARY
        return [
            binary,
            "-n", sim.NET_FILE,
            "-r", self.route_file,
            "--start",
            "--step-length", str(sim.STEP_LENGTH),
            "--end", str(float(self.outer_args.episode_seconds)),
            "--seed", str(self.scenario.seed),
            "--max-num-vehicles", str(self.scenario.max_vehicles),
            "--max-depart-delay", str(self.scenario.max_depart_delay),
            "--time-to-teleport", str(self.scenario.time_to_teleport),
            "--ignore-route-errors", "true",
            "--quit-on-end", "false",
            *sim.QUIET_SUMO_ARGS,
        ]

    def reset(self) -> dict[str, Any]:
        cmp.reset_sim_globals()
        cmp.apply_sim_distance_globals(self.scenario)
        sim.write_empty_route_file(self.route_file)

        if self.gui:
            sim.ensure_xquartz()

        sim.traci.start(self._sumo_cmd())
        self.started = True

        # KEY DIFFERENCE FROM FIXED/MODEL RUNS:
        # Do not call sim.build_all_fixed_controllers().  That function replaces
        # the network's original <tlLogic> behavior with this project's generated
        # safe phases.  Leaving controllers empty lets SUMO advance the native
        # signal programs from the .net.xml file.
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
            # Keep this consistent with the current compare runner's anchorless
            # baseline: normal traffic only, no ambulance events.
            "next_ambulance_spawn": float("inf"),
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
            "target_queue": 0.0,
            "target_wait": 0.0,
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


def collect_metric_sample(
    samples: list[dict[str, float]],
    episode: NativeSignalEpisode,
    spawned_window: int,
    extended_window: int,
    recovered_window: int,
    arrived_window: int,
) -> dict[str, Any]:
    global_wait, global_queue, avg_speed = cmp.network_wait_queue_speed()
    info = episode.info(
        spawned=spawned_window,
        extended=extended_window,
        recovered=recovered_window,
        arrived=arrived_window,
        global_wait=global_wait,
        global_queue=global_queue,
        avg_speed=avg_speed,
    )
    cmp.collect_sample(samples, info, reward=0.0)
    return info


def run_native_signal_episode(args: argparse.Namespace, scenario: cmp.TrafficScenario, seed: int) -> dict[str, float | int | str]:
    """Run native SUMO map timing with equal-time metric samples."""
    episode = NativeSignalEpisode(
        scenario=scenario,
        seed=seed,
        args=args,
        gui=bool(args.gui),
        env_rank=0,
    )
    samples: list[dict[str, float]] = []

    try:
        episode.reset()

        # Use 1-second chunks so native and all-model runs have the same outer
        # simulated-time loop. SUMO still advances native signal programs itself.
        decision_steps = 1
        metrics_interval = max(float(sim.STEP_LENGTH), float(getattr(args, "metrics_interval", 10.0)))
        print_interval = max(float(sim.STEP_LENGTH), float(getattr(args, "eval_print_every", 20)))

        print()
        print("=" * 92)
        print(f"[native seed {seed}] ORIGINAL SUMO .net.xml traffic-light programs")
        print("=" * 92)
        print(f"network file:           {sim.NET_FILE}")
        print(f"native traffic lights:  {len(episode.native_tls_ids)}")
        print(f"max active vehicles:    {episode.args.max_vehicles}")
        print(f"target active vehicles: {episode.args.target_vehicles}")
        print("generated safe phases:  OFF")
        print("RL model:               OFF")
        print(f"metrics every:          {metrics_interval:g} sim seconds")
        print("=" * 92)
        print()

        last_info: dict[str, Any] = collect_metric_sample(
            samples=samples,
            episode=episode,
            spawned_window=0,
            extended_window=0,
            recovered_window=0,
            arrived_window=0,
        )
        next_metrics_time = metrics_interval
        next_print_time = print_interval
        window_spawned = 0
        window_extended = 0
        window_recovered = 0
        window_arrived = 0

        for step in range(1, int(args.eval_steps) + 1):
            # No controller loop here. SUMO automatically advances the native
            # traffic-light programs during simulationStep().
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
            episode.total_arrived += int(arrived)
            window_arrived += int(arrived)
            window_spawned += int(spawned)
            window_extended += int(extended)
            window_recovered += int(recovered)

            cheap_info = episode.info(
                spawned=window_spawned,
                extended=window_extended,
                recovered=window_recovered,
                arrived=window_arrived,
                global_wait=float(last_info.get("global_wait", 0.0) or 0.0),
                global_queue=float(last_info.get("global_queue", 0.0) or 0.0),
                avg_speed=float(last_info.get("avg_speed", 0.0) or 0.0),
            )
            sim_time = float(cheap_info.get("sim_time", 0.0) or 0.0)

            if sim_time + 1e-9 >= next_metrics_time or sim_time >= float(args.episode_seconds):
                last_info = collect_metric_sample(
                    samples=samples,
                    episode=episode,
                    spawned_window=window_spawned,
                    extended_window=window_extended,
                    recovered_window=window_recovered,
                    arrived_window=window_arrived,
                )
                window_spawned = 0
                window_extended = 0
                window_recovered = 0
                window_arrived = 0
                while next_metrics_time <= sim_time + 1e-9:
                    next_metrics_time += metrics_interval
            else:
                last_info = cheap_info

            if sim_time + 1e-9 >= next_print_time:
                print(
                    f"[native seed {seed}] step={step:6d}, "
                    f"t={sim_time:8.1f}, "
                    f"active={int(last_info.get('active_vehicles', 0)):4d}, "
                    f"gq={float(last_info.get('global_queue', 0.0)):7.1f}, "
                    f"gw={float(last_info.get('global_wait', 0.0)):9.1f}, "
                    f"arrived={int(last_info.get('total_arrived', 0)):5d}"
                )
                while next_print_time <= sim_time + 1e-9:
                    next_print_time += print_interval

            if sim_time >= float(args.episode_seconds):
                break

    finally:
        episode.close()

    return cmp.summarize_samples("native_sumo", seed, samples)


def aggregate_by_controller(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return cmp.aggregate_by_controller(rows)


def print_native_vs_model_table(rows: list[dict[str, Any]]) -> None:
    agg = {row["controller"]: row for row in aggregate_by_controller(rows)}
    native = agg.get("native_sumo")
    model = agg.get("all_model")
    fixed = agg.get("fixed_cycle")

    if not native or not model:
        print("Not enough data to print native-vs-model comparison table.")
        return

    metrics = [
        ("total_arrived", "higher"),
        ("mean_avg_speed_mps", "higher"),
        ("mean_global_queue", "lower"),
        ("max_global_queue", "lower"),
        ("mean_global_wait", "lower"),
        ("max_global_wait", "lower"),
        ("recovered_total", "lower"),
    ]

    print("\n" + "=" * 112)
    print("NATIVE SUMO MAP TIMING VS COPIED ALL-TLS MODEL")
    print("=" * 112)

    if fixed:
        print(
            f"{'metric':28s} {'native_sumo':>14s} {'fixed_cycle':>14s} {'fixed imp.':>12s} "
            f"{'all_model':>14s} {'model imp.':>12s}"
        )
        print("-" * 112)
        for metric, direction in metrics:
            native_value = float(native.get(metric, 0.0) or 0.0)
            fixed_value = float(fixed.get(metric, 0.0) or 0.0)
            model_value = float(model.get(metric, 0.0) or 0.0)
            fixed_pct = cmp.improvement_percent(fixed_value, native_value, higher_is_better=(direction == "higher"))
            model_pct = cmp.improvement_percent(model_value, native_value, higher_is_better=(direction == "higher"))
            print(
                f"{metric:28s} {native_value:14.3f} {fixed_value:14.3f} {fixed_pct:11.2f}% "
                f"{model_value:14.3f} {model_pct:11.2f}%"
            )
    else:
        print(f"{'metric':28s} {'native_sumo':>14s} {'all_model':>14s} {'model imp.':>12s}")
        print("-" * 112)
        for metric, direction in metrics:
            native_value = float(native.get(metric, 0.0) or 0.0)
            model_value = float(model.get(metric, 0.0) or 0.0)
            model_pct = cmp.improvement_percent(model_value, native_value, higher_is_better=(direction == "higher"))
            print(f"{metric:28s} {native_value:14.3f} {model_value:14.3f} {model_pct:11.2f}%")

    print("=" * 112)
    print("Positive improvement means the controller beat native SUMO timing for that metric.")
    print("For queue/wait/recovery metrics, lower is better. For arrived/speed, higher is better.")
    print("native_sumo uses the original <tlLogic> programs from the .net.xml map.")
    print("all_model uses the learned policy copied independently to every compatible TLS.\n")


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        return

    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    csv_path = Path(args.stats_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote per-run CSV stats: {csv_path}")

    if args.stats_json:
        json_path = Path(args.stats_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runs": rows,
            "aggregate": aggregate_by_controller(rows),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON stats: {json_path}")


def compare_native_to_model(args: argparse.Namespace) -> None:
    seeds = cmp.parse_seed_list(args.compare_seeds)
    rows: list[dict[str, Any]] = []

    if args.gui and len(seeds) > 1:
        print("WARNING: --gui with multiple seeds will open/close SUMO GUI repeatedly and will be slow.")

    for seed in seeds:
        print("\n" + "#" * 92)
        print(f"Comparing native SUMO timing vs all-model, seed {seed}")
        print("#" * 92)

        # Same exact scenario object for native, fixed optional, and all-model.
        scenario = cmp.build_fixed_scenario(seed=seed, args=args)

        if not args.skip_native:
            rows.append(run_native_signal_episode(args, scenario, seed))

        if args.include_fixed_cycle:
            rows.append(cmp.run_fixed_cycle_episode(args, scenario, seed))

        if not args.skip_all_model:
            rows.append(cmp.run_all_model_episode(args, scenario, seed))

    print_native_vs_model_table(rows)
    write_outputs(rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original/native SUMO map timing against copied all-TLS RL control."
    )
    parser.add_argument("--episode-seconds", type=int, default=3600)
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--eval-print-every", type=int, default=20)
    parser.add_argument("--model-path", default=cmp.MODEL_DEFAULT)
    parser.add_argument("--vecnormalize-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-seeds", default="42")
    parser.add_argument("--stats-csv", default="native_vs_model_stats.csv")
    parser.add_argument("--stats-json", default="native_vs_model_stats.json")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI. For statistics, headless mode is much faster.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-update-period", type=float, default=10.0,
                        help="Seconds between policy updates for each traffic light in all-model mode. Larger is faster.")
    parser.add_argument("--metrics-interval", type=float, default=10.0,
                        help="Seconds between expensive global metric scans. Larger is faster.")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-all-model", action="store_true")
    parser.add_argument("--include-fixed-cycle", action="store_true",
                        help="Also include the generated fixed-cycle controller as a third baseline.")
    parser.add_argument("--print-scenarios", action="store_true")

    parser.add_argument("--max-vehicle-center", type=int, default=cmp.SIM_CENTER_MAX_VEHICLES)
    parser.add_argument("--target-vehicle-center", type=int, default=cmp.SIM_CENTER_TARGET_VEHICLES)
    parser.add_argument("--initial-vehicle-center", type=int, default=cmp.SIM_CENTER_INITIAL_VEHICLES)
    parser.add_argument("--spawn-batch-center", type=int, default=cmp.SIM_CENTER_SPAWN_BATCH)
    parser.add_argument("--green-duration-center", type=float, default=cmp.SIM_CENTER_GREEN_DURATION)
    parser.add_argument("--density-spread", type=float, default=0.0)
    parser.add_argument("--initial-spread", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_vehicle_center = min(int(args.max_vehicle_center), cmp.SIM_CENTER_MAX_VEHICLES)
    args.target_vehicle_center = min(int(args.target_vehicle_center), args.max_vehicle_center)
    args.initial_vehicle_center = min(int(args.initial_vehicle_center), args.target_vehicle_center)
    args.spawn_batch_center = max(1, int(args.spawn_batch_center))
    args.model_update_period = max(float(sim.STEP_LENGTH), float(getattr(args, "model_update_period", 10.0)))
    args.metrics_interval = max(float(sim.STEP_LENGTH), float(getattr(args, "metrics_interval", 10.0)))
    args.density_spread = min(0.60, max(0.0, float(args.density_spread)))
    args.initial_spread = min(1.50, max(0.0, float(args.initial_spread)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        cmp.torch.manual_seed(args.seed)
    except Exception:
        pass

    compare_native_to_model(args)


if __name__ == "__main__":
    main()
