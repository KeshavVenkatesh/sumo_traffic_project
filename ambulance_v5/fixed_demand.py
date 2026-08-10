#!/usr/bin/env python3
"""Small, dependency-free helpers for immutable fixed-demand route banks."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path


FIXED_DEMAND_SCHEMA_VERSION = 1
FIXED_DEMAND_VTYPE_ID = "fixed_demand_passenger"
REQUIRED_PASSENGER_VTYPE_ATTRIBUTES = {
    "vClass": "passenger",
    "jmIgnoreKeepClearTime": "-1",
    "jmDriveAfterYellowTime": "-1",
    "jmDriveAfterRedTime": "-1",
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_scheduled_vehicles(path: str | Path) -> int:
    """Count concrete scheduled departures in a SUMO route file.

    Fixed-demand comparisons require explicit ``vehicle`` or ``trip`` records.
    A ``flow`` is deliberately rejected because its realized vehicle count can
    depend on runtime insertion conditions and therefore is not an immutable
    paired demand schedule.
    """

    count = 0
    for _event, element in ET.iterparse(Path(path), events=("end",)):
        if element.tag in {"vehicle", "trip"}:
            count += 1
        elif element.tag == "flow":
            raise ValueError(
                f"Fixed-demand route bank cannot contain <flow>: {path}"
            )
        element.clear()
    return count


def fixed_demand_vehicle_type_is_safe(path: str | Path) -> bool:
    """Return whether every scheduled vehicle uses the audited car vType."""

    vehicle_type_safe = False
    scheduled_total = 0
    scheduled_types_safe = True
    for _event, element in ET.iterparse(Path(path), events=("end",)):
        if (
            element.tag == "vType"
            and element.get("id") == FIXED_DEMAND_VTYPE_ID
        ):
            vehicle_type_safe = all(
                element.get(name) == value
                for name, value in (
                    REQUIRED_PASSENGER_VTYPE_ATTRIBUTES.items()
                )
            )
        elif element.tag in {"vehicle", "trip"}:
            scheduled_total += 1
            scheduled_types_safe = (
                scheduled_types_safe
                and element.get("type") == FIXED_DEMAND_VTYPE_ID
            )
        element.clear()
    return (
        vehicle_type_safe
        and scheduled_total > 0
        and scheduled_types_safe
    )


def enforce_fixed_demand_vehicle_type(path: str | Path) -> None:
    """Atomically bind all fixed departures to the audited passenger vType."""

    route_path = Path(path)
    tree = ET.parse(route_path)
    root = tree.getroot()
    scheduled = root.findall("vehicle") + root.findall("trip")
    if not scheduled:
        raise ValueError(
            f"Fixed-demand route contains no vehicle or trip: {route_path}"
        )
    vehicle_type = next(
        (
            element
            for element in root.findall("vType")
            if element.get("id") == FIXED_DEMAND_VTYPE_ID
        ),
        None,
    )
    if vehicle_type is None:
        vehicle_type = ET.Element("vType", {"id": FIXED_DEMAND_VTYPE_ID})
        root.insert(0, vehicle_type)
    for name, value in REQUIRED_PASSENGER_VTYPE_ATTRIBUTES.items():
        vehicle_type.set(name, value)
    for element in scheduled:
        element.set("type", FIXED_DEMAND_VTYPE_ID)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="    ")
    temporary = route_path.with_suffix(route_path.suffix + ".tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(route_path)
    if not fixed_demand_vehicle_type_is_safe(route_path):
        raise RuntimeError(
            f"Could not enforce fixed-demand vehicle type in {route_path}"
        )


__all__ = (
    "FIXED_DEMAND_SCHEMA_VERSION",
    "FIXED_DEMAND_VTYPE_ID",
    "REQUIRED_PASSENGER_VTYPE_ATTRIBUTES",
    "count_scheduled_vehicles",
    "enforce_fixed_demand_vehicle_type",
    "fixed_demand_vehicle_type_is_safe",
    "sha256_file",
)
