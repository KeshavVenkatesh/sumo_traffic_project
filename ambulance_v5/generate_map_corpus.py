#!/usr/bin/env python3
"""Download varied OSM road regions through Overpass and convert them to SUMO.

The generated OSM/net files are data artifacts and are intentionally ignored by
git.  ``manifest.json`` records exact bounding boxes and train/validation/test
splits so an experiment remains reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

EXCLUDED_HIGHWAYS = (
    "footway|path|steps|cycleway|bridleway|construction|proposed|raceway|"
    "corridor|elevator|platform"
)


def make_query(bbox: list[float]) -> str:
    south, west, north, east = bbox
    return f"""[out:xml][timeout:240];
(
  way[\"highway\"][\"highway\"!~\"{EXCLUDED_HIGHWAYS}\"]({south},{west},{north},{east});
);
(._;>;);
out meta;
"""


def subarea_bbox(center_lat: float, center_lon: float, half_height: float, half_width: float):
    return [
        round(center_lat - half_height, 7),
        round(center_lon - half_width, 7),
        round(center_lat + half_height, 7),
        round(center_lon + half_width, 7),
    ]


def expand_regions(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    maps: list[dict[str, Any]] = []
    for region in config["regions"]:
        count = max(1, int(region.get("subareas", 1)))
        center_lat, center_lon = map(float, region["center"])
        jitter_lat, jitter_lon = map(float, region.get("jitter", [0.0, 0.0]))
        half_height, half_width = map(float, region.get("half_size", [0.018, 0.022]))
        size_jitter = float(region.get("half_size_jitter", 0.0))
        if not 0.0 <= size_jitter < 1.0:
            raise ValueError(
                f"half_size_jitter for {region['name']!r} must be in [0, 1)"
            )
        for index in range(count):
            lat = center_lat + rng.uniform(-jitter_lat, jitter_lat)
            lon = center_lon + rng.uniform(-jitter_lon, jitter_lon)
            # Vary north/south and east/west extent independently. This avoids
            # teaching the policy that every domain is the same rectangular
            # crop while keeping every generated bounding box reproducible.
            height = half_height * rng.uniform(1.0 - size_jitter, 1.0 + size_jitter)
            width = half_width * rng.uniform(1.0 - size_jitter, 1.0 + size_jitter)
            suffix = f"_{index + 1}" if count > 1 else ""
            maps.append(
                {
                    "name": f"{region['name']}{suffix}",
                    "split": region.get("split", "train"),
                    "bbox": subarea_bbox(lat, lon, height, width),
                }
            )
    return maps


def download_overpass(query: str, output: Path, endpoints: list[str], retries: int) -> str:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    errors: list[str] = []
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "User-Agent": "sumo-traffic-map-generalization/2.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = response.read()
            if not payload.lstrip().startswith((b"<?xml", b"<osm")):
                raise RuntimeError("Overpass response was not OSM XML")
            output.write_bytes(payload)
            return endpoint
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            errors.append(f"{endpoint}: {exc}")
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError("Overpass download failed:\n" + "\n".join(errors))


def count_osm_signals(osm_file: Path) -> int:
    count = 0
    for _event, element in ET.iterparse(osm_file, events=("end",)):
        if element.tag == "node" and any(
            tag.get("k") == "highway" and tag.get("v") == "traffic_signals"
            for tag in element.findall("tag")
        ):
            count += 1
        # Do not clear child <tag> elements before their parent <node> has been
        # inspected; doing so erases k/v attributes and makes every count zero.
        if element.tag in {"node", "way", "relation"}:
            element.clear()
    return count


def net_tls_stats(net_file: Path) -> tuple[int, int, int]:
    total = 0
    usable = 0
    max_candidates = 0
    for _event, element in ET.iterparse(net_file, events=("end",)):
        if element.tag == "tlLogic":
            total += 1
            candidates: list[str] = []
            for phase in element.findall("phase"):
                state = str(phase.get("state", ""))
                if any(char in state for char in "yYu"):
                    continue
                if not any(char in state for char in "Ggs"):
                    continue
                if state not in candidates:
                    candidates.append(state)
            if len(candidates) >= 2:
                usable += 1
            max_candidates = max(max_candidates, len(candidates))
            element.clear()
    return total, usable, max_candidates


def convert_with_netconvert(osm_file: Path, net_file: Path, netconvert: str) -> None:
    command = [
        netconvert,
        "--osm-files",
        str(osm_file),
        "--output-file",
        str(net_file),
        "--geometry.remove",
        "true",
        "--roundabouts.guess",
        "true",
        "--ramps.guess",
        "true",
        "--junctions.join",
        "true",
        "--tls.guess",
        "true",
        "--tls.guess-signals",
        "true",
        "--tls.join",
        "true",
        "--no-turnarounds",
        "true",
        "--remove-edges.isolated",
        "true",
        "--ignore-errors",
        "false",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("map_corpus_regions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_map_corpus"))
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--netconvert", default="netconvert")
    parser.add_argument("--overpass-endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--min-osm-signals", type=int, default=4)
    parser.add_argument("--min-sumo-tls", type=int, default=2)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    planned = expand_regions(config, args.seed)
    endpoints = [x.strip() for x in args.overpass_endpoints.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": args.seed,
        "generator": "generate_map_corpus.py",
        "overpass_turbo_ui": "https://overpass-turbo.eu/",
        "maps": [],
    }

    for index, record in enumerate(planned, start=1):
        map_dir = args.output_dir / record["name"]
        map_dir.mkdir(parents=True, exist_ok=True)
        osm_file = map_dir / f"{record['name']}.osm"
        net_file = map_dir / f"{record['name']}.net.xml"
        query_file = map_dir / f"{record['name']}.overpassql"
        query = make_query(record["bbox"])
        query_file.write_text(query, encoding="utf-8")

        print(f"\n[{index}/{len(planned)}] {record['name']} ({record['split']})")
        endpoint = None
        if not args.skip_download and (args.overwrite or not osm_file.exists()):
            endpoint = download_overpass(query, osm_file, endpoints, args.retries)
        if not osm_file.exists():
            print("  skipped: OSM file absent")
            continue

        osm_signals = count_osm_signals(osm_file)
        if osm_signals < args.min_osm_signals:
            print(f"  rejected: only {osm_signals} OSM signal nodes")
            continue

        if args.overwrite or not net_file.exists():
            convert_with_netconvert(osm_file, net_file, args.netconvert)
        tls_count, usable_tls_count, max_phase_candidates = net_tls_stats(net_file)
        if usable_tls_count < args.min_sumo_tls:
            print(
                f"  rejected: only {usable_tls_count} SUMO TLS have at least "
                "two stable native green candidates"
            )
            continue

        manifest["maps"].append(
            {
                **record,
                "osm_file": str(osm_file.resolve()),
                "net_file": str(net_file.resolve()),
                "query_file": str(query_file.resolve()),
                "overpass_endpoint": endpoint,
                "osm_traffic_signal_nodes": osm_signals,
                "sumo_tl_logic_count": tls_count,
                "usable_variable_phase_tls_count": usable_tls_count,
                "max_stable_phase_candidates": max_phase_candidates,
            }
        )
        print(
            f"  accepted: {osm_signals} OSM signals, {tls_count} SUMO TLS, "
            f"{usable_tls_count} with variable green candidates"
        )

    manifest_file = args.output_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "ODbL_ATTRIBUTION.txt").write_text(
        "Map data copyright OpenStreetMap contributors, available under the ODbL.\n"
        "https://www.openstreetmap.org/copyright\n",
        encoding="utf-8",
    )
    print(f"\nAccepted {len(manifest['maps'])}/{len(planned)} maps")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
