#!/usr/bin/env python3
"""Validate and freeze the leakage-free schema-v4 map split protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from generate_map_corpus import expand_regions


ALLOWED_SPLITS = {"train", "validation", "test"}


class ProtocolError(RuntimeError):
    """Raised when a planned or generated corpus violates the split contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    south, west, north, east = map(float, bbox)
    return (0.5 * (south + north), 0.5 * (west + east))


def bboxes_overlap(left: list[float], right: list[float]) -> bool:
    left_south, left_west, left_north, left_east = map(float, left)
    right_south, right_west, right_north, right_east = map(float, right)
    return not (
        left_north < right_south
        or right_north < left_south
        or left_east < right_west
        or right_east < left_west
    )


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    term = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(term)))


def _duplicates(values: list[str]) -> list[str]:
    return sorted(name for name, count in Counter(values).items() if count > 1)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"{path} must contain a JSON object")
    return payload


def verify_training_protocol_lock(
    lock_value: str,
    manifest_value: str,
    selected_splits: set[str],
) -> dict[str, Any] | None:
    """Verify that training uses only the train split of the frozen manifest."""
    if not lock_value:
        return None
    if selected_splits != {"train"}:
        raise ProtocolError(
            "Locked schema-v4 training requires --splits train exactly"
        )
    lock_path = Path(lock_value).expanduser().resolve()
    manifest_path = Path(manifest_value).expanduser().resolve()
    lock = _load_json(lock_path)
    manifest = _load_json(manifest_path)
    if lock.get("status") != "valid":
        raise ProtocolError(f"Split protocol lock is not valid: {lock_path}")
    generated = lock.get("generated_manifest")
    if not isinstance(generated, dict) or not generated.get("sha256"):
        raise ProtocolError(
            "Training requires a post-generation protocol lock created with "
            "validate_map_split_protocol.py --manifest"
        )
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != str(generated["sha256"]):
        raise ProtocolError("Training manifest does not match the protocol lock")
    expected_by_split = {
        "train": set(lock.get("new_corpus", {}).get("train_maps", [])),
        "validation": set(
            lock.get("new_corpus", {}).get("validation_maps", [])
        ),
        "test": set(lock.get("new_corpus", {}).get("final_test_maps", [])),
    }
    actual_by_split = {
        split: {
            str(record.get("name"))
            for record in manifest.get("maps", [])
            if str(record.get("split")) == split
        }
        for split in ALLOWED_SPLITS
    }
    if actual_by_split != expected_by_split:
        raise ProtocolError(
            "Training manifest split membership differs from the protocol lock"
        )
    return {
        "path": str(lock_path),
        "sha256": file_sha256(lock_path),
        "manifest_sha256": manifest_sha,
        "training_maps": sorted(expected_by_split["train"]),
        "validation_maps": sorted(expected_by_split["validation"]),
        "excluded_final_test_maps": sorted(expected_by_split["test"]),
    }


def build_protocol_lock(
    new_config_path: Path,
    old_config_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    new_config_path = new_config_path.resolve()
    old_config_path = old_config_path.resolve()
    new_config = _load_json(new_config_path)
    old_config = _load_json(old_config_path)
    protocol = new_config.get("protocol")
    if not isinstance(protocol, dict):
        raise ProtocolError(f"{new_config_path} has no protocol object")

    errors: list[str] = []
    new_seed = int(protocol.get("seed", -1))
    old_seed = int(protocol.get("old_schema_v3_seed", -1))
    minimum_distance_km = float(
        protocol.get("minimum_test_center_distance_km", 0.0)
    )
    expected_counts = {
        str(split): int(count)
        for split, count in protocol.get("expected_split_counts", {}).items()
    }
    if set(expected_counts) != ALLOWED_SPLITS:
        errors.append(
            "expected_split_counts must contain exactly train, validation, and test"
        )
    if any(count <= 0 for count in expected_counts.values()):
        errors.append("every expected split count must be positive")
    if new_seed < 0 or old_seed < 0:
        errors.append("both new and historical corpus seeds must be nonnegative")
    if minimum_distance_km <= 0.0:
        errors.append("minimum_test_center_distance_km must be positive")
    configured_old_name = str(protocol.get("old_schema_v3_config", ""))
    if configured_old_name and Path(configured_old_name).name != old_config_path.name:
        errors.append(
            "old config argument does not match protocol.old_schema_v3_config: "
            f"{old_config_path.name!r} != {configured_old_name!r}"
        )

    new_regions = new_config.get("regions", [])
    old_regions = old_config.get("regions", [])
    if not isinstance(new_regions, list) or not isinstance(old_regions, list):
        errors.append("both configs must contain a regions array")
        new_regions = []
        old_regions = []

    new_region_names = [str(item.get("name", "")) for item in new_regions]
    old_region_names = [str(item.get("name", "")) for item in old_regions]
    duplicate_regions = _duplicates(new_region_names)
    if duplicate_regions:
        errors.append(f"duplicate new region names: {', '.join(duplicate_regions)}")
    reused_regions = sorted(set(new_region_names) & set(old_region_names))
    if reused_regions:
        errors.append(
            "new corpus reuses historical region names: " + ", ".join(reused_regions)
        )

    if errors:
        raise ProtocolError("Map split protocol failed:\n- " + "\n- ".join(errors))

    new_maps = expand_regions(new_config, new_seed)
    old_maps = expand_regions(old_config, old_seed)
    duplicate_maps = _duplicates([str(item["name"]) for item in new_maps])
    if duplicate_maps:
        errors.append(f"duplicate expanded map names: {', '.join(duplicate_maps)}")

    actual_counts = dict(Counter(str(item["split"]) for item in new_maps))
    unknown_splits = sorted(set(actual_counts) - ALLOWED_SPLITS)
    if unknown_splits:
        errors.append(f"unknown new-corpus splits: {', '.join(unknown_splits)}")
    if actual_counts != expected_counts:
        errors.append(
            f"planned split counts {actual_counts} do not match {expected_counts}"
        )

    # Even differently named regions must not overlap across development splits.
    for index, left in enumerate(new_maps):
        for right in new_maps[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            if bboxes_overlap(left["bbox"], right["bbox"]):
                errors.append(
                    "cross-split bounding boxes overlap: "
                    f"{left['name']} ({left['split']}) and "
                    f"{right['name']} ({right['split']})"
                )

    test_maps = [item for item in new_maps if item["split"] == "test"]
    development_maps = [item for item in new_maps if item["split"] != "test"]
    exclusion_maps = [
        ("new development", item) for item in development_maps
    ] + [("historical corpus", item) for item in old_maps]
    minimum_observed_distance_km = math.inf
    closest_pair: dict[str, Any] | None = None
    for test_map in test_maps:
        test_center = bbox_center(test_map["bbox"])
        for source, excluded_map in exclusion_maps:
            if bboxes_overlap(test_map["bbox"], excluded_map["bbox"]):
                errors.append(
                    f"final-test map {test_map['name']} overlaps {source} map "
                    f"{excluded_map['name']}"
                )
            distance = haversine_km(
                test_center, bbox_center(excluded_map["bbox"])
            )
            if distance < minimum_observed_distance_km:
                minimum_observed_distance_km = distance
                closest_pair = {
                    "test_map": test_map["name"],
                    "excluded_source": source,
                    "excluded_map": excluded_map["name"],
                    "center_distance_km": round(distance, 3),
                }
            if distance < minimum_distance_km:
                errors.append(
                    f"final-test map {test_map['name']} is only {distance:.1f} km "
                    f"from {source} map {excluded_map['name']} "
                    f"(minimum {minimum_distance_km:.1f} km)"
                )

    manifest_summary: dict[str, Any] | None = None
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        manifest = _load_json(manifest_path)
        if int(manifest.get("seed", -1)) != new_seed:
            errors.append(
                f"manifest seed {manifest.get('seed')!r} does not match {new_seed}"
            )
        records = manifest.get("maps", [])
        if not isinstance(records, list):
            errors.append("manifest maps must be an array")
            records = []
        planned_by_name = {str(item["name"]): item for item in new_maps}
        accepted_names = [str(item.get("name", "")) for item in records]
        duplicate_accepted = _duplicates(accepted_names)
        if duplicate_accepted:
            errors.append(
                "manifest contains duplicate maps: " + ", ".join(duplicate_accepted)
            )
        missing = sorted(set(planned_by_name) - set(accepted_names))
        unexpected = sorted(set(accepted_names) - set(planned_by_name))
        if missing:
            errors.append(
                "generation did not accept every frozen map: " + ", ".join(missing)
            )
        if unexpected:
            errors.append("manifest has unplanned maps: " + ", ".join(unexpected))
        for record in records:
            name = str(record.get("name", ""))
            planned = planned_by_name.get(name)
            if planned is None:
                continue
            if str(record.get("split")) != str(planned["split"]):
                errors.append(f"manifest split changed for {name}")
            if [float(value) for value in record.get("bbox", [])] != [
                float(value) for value in planned["bbox"]
            ]:
                errors.append(f"manifest bounding box changed for {name}")
            net_file = Path(str(record.get("net_file", "")))
            if not net_file.is_file():
                errors.append(
                    f"manifest network file is missing for {name}: {net_file}"
                )
        accepted_counts = dict(
            Counter(str(item.get("split")) for item in records)
        )
        if accepted_counts != expected_counts:
            errors.append(
                "accepted split counts "
                f"{accepted_counts} do not match {expected_counts}"
            )
        manifest_summary = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "accepted_split_counts": accepted_counts,
        }

    if errors:
        raise ProtocolError("Map split protocol failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": 1,
        "status": "valid",
        "corpus_id": str(protocol.get("corpus_id", "")),
        "new_corpus": {
            "config": str(new_config_path),
            "config_sha256": file_sha256(new_config_path),
            "seed": new_seed,
            "split_counts": actual_counts,
            "train_maps": [
                item["name"] for item in new_maps if item["split"] == "train"
            ],
            "validation_maps": [
                item["name"] for item in new_maps if item["split"] == "validation"
            ],
            "final_test_maps": [item["name"] for item in test_maps],
        },
        "historical_exclusion": {
            "config": str(old_config_path),
            "config_sha256": file_sha256(old_config_path),
            "seed": old_seed,
            "excluded_map_count": len(old_maps),
            "excluded_maps": [item["name"] for item in old_maps],
        },
        "checks": {
            "region_names_disjoint": True,
            "cross_split_bounding_boxes_disjoint": True,
            "minimum_required_test_center_distance_km": minimum_distance_km,
            "minimum_observed_test_center_distance_km": round(
                minimum_observed_distance_km, 3
            ),
            "closest_test_exclusion_pair": closest_pair,
        },
        "generated_manifest": manifest_summary,
        "final_test_policy": str(protocol.get("policy", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-config", type=Path, default=Path("map_corpus_regions_v4.json")
    )
    parser.add_argument(
        "--old-config", type=Path, default=Path("map_corpus_regions.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional generated manifest; requires every frozen map to be accepted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("detector_v4_split_protocol_lock.json"),
    )
    args = parser.parse_args()
    try:
        payload = build_protocol_lock(
            args.new_config, args.old_config, args.manifest
        )
    except (OSError, ValueError, json.JSONDecodeError, ProtocolError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts = payload["new_corpus"]["split_counts"]
    print(
        "VALID leakage-free map protocol: "
        f"train={counts['train']}, validation={counts['validation']}, "
        f"test={counts['test']}"
    )
    print(
        "Closest final-test/excluded center distance: "
        f"{payload['checks']['minimum_observed_test_center_distance_km']:.1f} km"
    )
    print(f"Protocol lock: {args.output}")


if __name__ == "__main__":
    main()
