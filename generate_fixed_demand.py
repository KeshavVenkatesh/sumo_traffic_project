#!/usr/bin/env python3
"""Generate controller-independent SUMO routes for paired evaluation.

The existing evaluator maintains a target *active* population. A faster
controller therefore completes more trips and causes more vehicles to be
spawned, so equal seeds do not imply equal demand. This utility creates the
departure schedule and routes before either controller runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from train_map_agnostic_multimap import parse_csv, passenger_lane_km


ROOT = Path(__file__).resolve().parent


def find_random_trips(explicit: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools" / "randomTrips.py")
    executable = shutil.which("randomTrips.py")
    if executable:
        candidates.append(Path(executable))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find randomTrips.py. Set SUMO_HOME or pass --random-trips."
    )


def count_routes(path: Path) -> int:
    count = 0
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag in {"vehicle", "trip", "flow"}:
            count += 1
        element.clear()
    return count


def generate_one(
    random_trips: Path,
    net_file: Path,
    output: Path,
    seed: int,
    episode_seconds: float,
    period: float,
    min_distance: float,
    fringe_factor: float,
    force: bool,
) -> dict[str, object]:
    if output.exists() and not force:
        scheduled_records = count_routes(output)
        if scheduled_records <= 0:
            raise RuntimeError(
                f"Existing demand route contains no scheduled records: {output}. "
                "Regenerate with --force or increase demand/duration."
            )
        return {
            "seed": seed,
            "net_file": str(net_file),
            "route_file": str(output),
            "period_seconds": period,
            "scheduled_records": scheduled_records,
            "reused": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    trip_file = output.with_name(output.name.removesuffix(".rou.xml") + ".trips.xml")
    command = [
        sys.executable,
        str(random_trips),
        "-n",
        str(net_file),
        "-o",
        str(trip_file),
        "--route-file",
        str(output),
        "--seed",
        str(seed),
        "--begin",
        "0",
        "--end",
        str(float(episode_seconds)),
        "--period",
        str(float(period)),
        "--vehicle-class",
        "passenger",
        "--fringe-factor",
        str(float(fringe_factor)),
        "--min-distance",
        str(float(min_distance)),
        "--trip-attributes",
        'departLane="best" departSpeed="max"',
        "--prefix",
        f"fixed_seed{seed}_",
        "--validate",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not output.is_file():
        raise RuntimeError(f"randomTrips.py did not create {output}")
    scheduled_records = count_routes(output)
    if scheduled_records <= 0:
        raise RuntimeError(
            f"No valid routes were generated for seed {seed}. Increase "
            "--episode-seconds or --trips-per-lane-km-hour, or reduce --min-distance."
        )
    return {
        "seed": seed,
        "net_file": str(net_file),
        "route_file": str(output),
        "period_seconds": period,
        "scheduled_records": scheduled_records,
        "reused": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--episode-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--trips-per-lane-km-hour",
        type=float,
        default=12.0,
        help="Map-normalized demand intensity used to derive randomTrips --period.",
    )
    parser.add_argument("--min-distance", type=float, default=300.0)
    parser.add_argument("--fringe-factor", type=float, default=3.0)
    parser.add_argument("--random-trips", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    net_file = Path(args.net_file).expanduser()
    if not net_file.is_absolute():
        net_file = ROOT / net_file
    net_file = net_file.resolve()
    if not net_file.is_file():
        raise FileNotFoundError(net_file)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    if not seeds:
        parser.error("--seeds cannot be empty")
    if args.episode_seconds <= 0 or args.trips_per_lane_km_hour <= 0:
        parser.error("episode duration and demand intensity must be positive")

    lane_km = passenger_lane_km(net_file)
    departures_per_second = (
        args.trips_per_lane_km_hour * lane_km / 3600.0
    )
    period = max(0.05, 1.0 / max(1e-9, departures_per_second))
    random_trips = find_random_trips(args.random_trips)
    output_dir = Path(args.output_dir).expanduser().resolve()

    records = []
    for seed in seeds:
        output = output_dir / f"seed_{seed}.rou.xml"
        records.append(
            generate_one(
                random_trips=random_trips,
                net_file=net_file,
                output=output,
                seed=seed,
                episode_seconds=args.episode_seconds,
                period=period,
                min_distance=args.min_distance,
                fringe_factor=args.fringe_factor,
                force=args.force,
            )
        )

    payload = {
        "schema_version": 1,
        "net_file": str(net_file),
        "passenger_lane_km": lane_km,
        "trips_per_lane_km_hour": args.trips_per_lane_km_hour,
        "episode_seconds": args.episode_seconds,
        "period_seconds": period,
        "routes": records,
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
