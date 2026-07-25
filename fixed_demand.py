#!/usr/bin/env python3
"""Small, dependency-free helpers for immutable fixed-demand route banks."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path


FIXED_DEMAND_SCHEMA_VERSION = 1


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


__all__ = (
    "FIXED_DEMAND_SCHEMA_VERSION",
    "count_scheduled_vehicles",
    "sha256_file",
)
