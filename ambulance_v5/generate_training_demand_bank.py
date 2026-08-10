#!/usr/bin/env python3
"""Pre-generate randomized route banks so PPO never computes OD routes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fixed_demand import sha256_file
from generate_fixed_demand import find_random_trips, generate_one
from train_map_agnostic_multimap import load_maps, parse_csv, passenger_lane_km


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--maps",
        default="",
        help="Optional comma-separated .net.xml files in addition to the manifest.",
    )
    parser.add_argument("--splits", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("training_demand_bank"))
    parser.add_argument("--rates", default="4,8,12")
    parser.add_argument("--seeds", default="101,102")
    parser.add_argument("--episode-seconds", type=float, default=7200.0)
    parser.add_argument("--min-distance", type=float, default=300.0)
    parser.add_argument("--fringe-factor", type=float, default=3.0)
    parser.add_argument("--random-trips", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.splits = set(parse_csv(args.splits))
    rates = [float(value) for value in parse_csv(args.rates)]
    seeds = [int(value) for value in parse_csv(args.seeds)]
    if not rates or min(rates) <= 0:
        parser.error("--rates must contain positive values")
    if not seeds:
        parser.error("--seeds cannot be empty")

    maps = load_maps(args)
    manifest = args.output_dir / "manifest.json"
    network_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in maps
    }
    requested_identity = {
        "episode_seconds": float(args.episode_seconds),
        "rates": rates,
        "seeds": seeds,
        "maps": sorted(str(path.resolve()) for path in maps),
        "network_hashes": sorted(network_hashes.items()),
    }
    previous_route_hashes: dict[str, str] = {}
    if manifest.exists() and not args.force:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        for record in previous.get("routes", []):
            route_path = Path(str(record["route_file"])).expanduser()
            if not route_path.is_absolute():
                route_path = manifest.parent / route_path
            previous_route_hashes[str(route_path.resolve())] = str(
                record.get("route_sha256", "")
            )
        previous_identity = {
            "episode_seconds": float(
                previous.get("episode_seconds", 0.0) or 0.0
            ),
            "rates": [float(value) for value in previous.get("rates", [])],
            "seeds": [int(value) for value in previous.get("seeds", [])],
            "maps": sorted(
                {
                    str(
                        (
                            Path(record["net_file"])
                            if Path(record["net_file"]).is_absolute()
                            else manifest.parent
                            / Path(record["net_file"])
                        ).resolve()
                    )
                    for record in previous.get("routes", [])
                }
            ),
            "network_hashes": sorted(
                {
                    (
                        str(
                            (
                                Path(record["net_file"])
                                if Path(record["net_file"]).is_absolute()
                                else manifest.parent
                                / Path(record["net_file"])
                            ).resolve()
                        ),
                        str(record.get("network_sha256", "")),
                    )
                    for record in previous.get("routes", [])
                }
            ),
        }
        if previous_identity != requested_identity:
            raise RuntimeError(
                f"{manifest} was generated with different maps/rates/seeds/"
                "duration. Use a new --output-dir or pass --force."
            )
    random_trips = find_random_trips(args.random_trips)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for net_file in maps:
        lane_km = passenger_lane_km(net_file)
        network_hash = network_hashes[str(net_file.resolve())]
        map_dir = args.output_dir / net_file.stem.replace(".net", "")
        for rate in rates:
            period = max(0.05, 3600.0 / max(1e-9, rate * lane_km))
            rate_tag = str(rate).replace(".", "p")
            for seed in seeds:
                output = map_dir / f"rate_{rate_tag}" / f"seed_{seed}.rou.xml"
                jobs.append(
                    {
                        "random_trips": random_trips,
                        "net_file": net_file,
                        "output": output,
                        "seed": seed,
                        "episode_seconds": args.episode_seconds,
                        "period": period,
                        "min_distance": args.min_distance,
                        "fringe_factor": args.fringe_factor,
                        "force": args.force,
                        "rate": rate,
                        "network_sha256": network_hash,
                    }
                )

    if not args.force:
        for job in jobs:
            output = Path(job["output"]).resolve()
            expected_hash = previous_route_hashes.get(str(output))
            if output.is_file() and expected_hash:
                if sha256_file(output) != expected_hash:
                    raise RuntimeError(
                        "Existing fixed-demand route changed after its "
                        f"manifest was written: {output}. Use a new "
                        "--output-dir or pass --force."
                    )

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                generate_one,
                random_trips=job["random_trips"],
                net_file=job["net_file"],
                output=job["output"],
                seed=job["seed"],
                episode_seconds=job["episode_seconds"],
                period=job["period"],
                min_distance=job["min_distance"],
                fringe_factor=job["fringe_factor"],
                force=job["force"],
            ): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            record = future.result()
            record["trips_per_lane_km_hour"] = job["rate"]
            record["network_sha256"] = job[
                "network_sha256"
            ]
            record["route_sha256"] = sha256_file(
                Path(str(record["route_file"]))
            )
            records.append(record)
            print(f"[{completed}/{len(jobs)}] {record['route_file']}", flush=True)

    records.sort(key=lambda record: (record["net_file"], record["period_seconds"], record["seed"]))
    payload = {
        "schema_version": 2,
        "generator": "generate_training_demand_bank.py",
        "episode_seconds": args.episode_seconds,
        "rates": rates,
        "seeds": seeds,
        "routes": records,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
