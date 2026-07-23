#!/usr/bin/env python3
"""Generate immutable, checksummed SUMO route banks with randomTrips.py.

Unlike the legacy target-population evaluator, every controller consuming this
bank receives identical scheduled departures and routes.  Congestion may delay
an actual insertion, which is itself measured as source backlog; it never causes
the evaluator to invent replacement demand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fixed_demand import (
    FIXED_DEMAND_SCHEMA_VERSION,
    count_scheduled_vehicles,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent


def parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def parse_scenarios(raw: str) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    names: set[str] = set()
    for item in parse_csv(raw):
        if "=" not in item:
            raise ValueError(f"Scenario must be NAME=PERIOD_SECONDS, got {item!r}")
        name, period_raw = item.split("=", 1)
        name = name.strip()
        period = float(period_raw)
        if not name or name in names or period <= 0.0:
            raise ValueError(f"Invalid or duplicate scenario {item!r}")
        names.add(name)
        values.append((name, period))
    if not values:
        raise ValueError("At least one demand scenario is required")
    return values


def find_random_trips(explicit: str = "") -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py")
    found = shutil.which("randomTrips.py")
    if found:
        candidates.append(Path(found))
    spec = importlib.util.find_spec("sumolib")
    if spec and spec.origin:
        package = Path(spec.origin).resolve().parent
        candidates.extend(
            [
                package.parent / "tools" / "randomTrips.py",
                package.parent / "randomTrips.py",
            ]
        )
    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry)
        candidates.extend(
            [
                base / "sumo" / "tools" / "randomTrips.py",
                base / "tools" / "randomTrips.py",
            ]
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find SUMO randomTrips.py. Set SUMO_HOME or pass "
        f"--random-trips. Searched:\n  {searched}"
    )


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload if isinstance(payload, list) else payload.get("maps", []))


def load_maps(args: argparse.Namespace) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    if args.map_manifest:
        manifest_path = Path(args.map_manifest).expanduser().resolve()
        for index, record in enumerate(_manifest_records(manifest_path)):
            if str(record.get("split", "train")) not in args.splits:
                continue
            raw_path = record.get("net_file") or record.get("path")
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            map_id = str(
                record.get("map_id")
                or record.get("name")
                or record.get("id")
                or f"map_{index:03d}"
            )
            records.append((map_id, path.resolve()))
    for raw_path in parse_csv(args.maps):
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        records.append((path.stem.replace(".net", ""), path))

    result: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for map_id, path in records:
        if not path.exists():
            raise FileNotFoundError(path)
        if map_id in seen_ids and path not in seen_paths:
            raise ValueError(f"Duplicate map_id {map_id!r} for different network files")
        if path in seen_paths:
            continue
        seen_ids.add(map_id)
        seen_paths.add(path)
        result.append((map_id, path))
    if not result:
        raise RuntimeError("No maps selected; pass --map-manifest or --maps")
    return result


def portable_path(path: Path, manifest_dir: Path) -> str:
    try:
        return os.path.relpath(path, manifest_dir)
    except ValueError:
        return str(path)


def ensure_global_car_type(route_file: Path) -> None:
    """Add the vehicle type used by this project's routing/recovery helpers."""

    tree = ET.parse(route_file)
    root = tree.getroot()
    changed = False
    for tag in ("vehicle", "trip", "flow"):
        for element in root.findall(tag):
            if element.get("type") != "global_car":
                element.set("type", "global_car")
                changed = True
    for element in root.findall("vType"):
        if element.get("id") == "global_car":
            break
    else:
        vehicle_type = ET.Element(
            "vType",
            {
                "id": "global_car",
                "vClass": "passenger",
                "guiShape": "passenger",
                "length": "4.6",
                "minGap": "2.5",
                "accel": "2.6",
                "decel": "4.5",
                "emergencyDecel": "9.0",
                "maxSpeed": "13.9",
                "sigma": "0.5",
                "tau": "1.0",
            },
        )
        root.insert(0, vehicle_type)
        changed = True
    if not changed:
        return
    try:
        ET.indent(tree, space="    ")
    except AttributeError:  # pragma: no cover - Python < 3.9
        pass
    tree.write(route_file, encoding="utf-8", xml_declaration=True)


def generate(args: argparse.Namespace) -> Path:
    random_trips = find_random_trips(args.random_trips)
    maps = load_maps(args)
    scenarios = parse_scenarios(args.scenarios)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    if not seeds:
        raise ValueError("At least one seed is required")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    print(f"randomTrips.py: {random_trips}")
    print(f"maps:           {len(maps)}")
    print(f"scenarios:      {scenarios}")
    print(f"seeds:          {seeds}")

    for map_id, net_file in maps:
        network_hash = sha256_file(net_file)
        map_dir = output_dir / map_id
        map_dir.mkdir(parents=True, exist_ok=True)
        for scenario, period in scenarios:
            for seed in seeds:
                route_file = map_dir / f"{scenario}_seed{seed}.rou.xml"
                command = [
                    sys.executable,
                    str(random_trips),
                    "-n",
                    str(net_file),
                    "-r",
                    str(route_file),
                    "-b",
                    str(float(args.begin_seconds)),
                    "-e",
                    str(float(args.end_seconds)),
                    "-p",
                    str(float(period)),
                    "--seed",
                    str(seed),
                    "--vehicle-class",
                    "passenger",
                    "--prefix",
                    f"fd_{map_id}_{scenario}_{seed}_",
                    "--min-distance",
                    str(float(args.min_distance)),
                    "--fringe-factor",
                    str(float(args.fringe_factor)),
                    "--trip-attributes",
                    'departLane="best" departSpeed="max"',
                    "--remove-loops",
                    "--validate",
                ]
                print(" ".join(shlex.quote(value) for value in command), flush=True)
                if args.dry_run:
                    continue
                if route_file.exists() and not args.overwrite:
                    print(f"Reusing existing route file: {route_file}")
                else:
                    subprocess.run(command, cwd=ROOT, check=True)
                ensure_global_car_type(route_file)
                scheduled = count_scheduled_vehicles(route_file)
                if scheduled <= 0:
                    raise RuntimeError(f"randomTrips generated no vehicles: {route_file}")
                records.append(
                    {
                        "map_id": map_id,
                        "net_file": portable_path(net_file, output_dir),
                        "network_sha256": network_hash,
                        "scenario": scenario,
                        "seed": seed,
                        "begin_seconds": float(args.begin_seconds),
                        "end_seconds": float(args.end_seconds),
                        "period_seconds": float(period),
                        "scheduled_vehicles": scheduled,
                        "route_file": portable_path(route_file, output_dir),
                        "route_sha256": sha256_file(route_file),
                    }
                )

    manifest = output_dir / "manifest.json"
    if args.dry_run:
        print(f"Dry run: would write {manifest}")
        return manifest
    payload = {
        "schema_version": FIXED_DEMAND_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(random_trips),
        "records": records,
    }
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    print(f"Wrote {manifest} with {len(records)} immutable demand records")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-manifest", default="")
    parser.add_argument("--maps", default="")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--output-dir", default="fixed_demand_bank_v1")
    parser.add_argument("--seeds", default="1001,1002,1003")
    parser.add_argument(
        "--scenarios",
        default="light=4.0,medium=2.0,heavy=1.0,oversaturated=0.5",
        help="Comma-separated NAME=SECONDS_BETWEEN_DEPARTURES",
    )
    parser.add_argument("--begin-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=1200.0)
    parser.add_argument("--min-distance", type=float, default=800.0)
    parser.add_argument("--fringe-factor", type=float, default=5.0)
    parser.add_argument("--random-trips", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.splits = set(parse_csv(args.splits))
    if args.end_seconds <= args.begin_seconds:
        parser.error("--end-seconds must be greater than --begin-seconds")
    if args.min_distance < 0.0:
        parser.error("--min-distance cannot be negative")
    return args


if __name__ == "__main__":
    generate(parse_args())
