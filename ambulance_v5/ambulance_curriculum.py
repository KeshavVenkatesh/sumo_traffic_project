#!/usr/bin/env python3
"""Dependency-free curriculum helpers for ambulance training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def curriculum_demand_routes(
    records: Sequence[Mapping[str, Any]], progress: float
) -> list[str]:
    """Select low/medium demand early and medium/high demand late.

    Unknown-intensity records remain eligible at every stage so older demand
    manifests do not silently lose coverage.
    """

    known = sorted(
        {
            float(record["intensity"])
            for record in records
            if record.get("intensity") is not None
        }
    )
    if len(known) <= 1 or 0.25 <= float(progress) < 0.65:
        selected = records
    elif float(progress) < 0.25:
        cutoff = known[min(1, len(known) - 1)]
        selected = [
            record
            for record in records
            if record.get("intensity") is None
            or float(record["intensity"]) <= cutoff
        ]
    else:
        cutoff = known[max(0, len(known) - 2)]
        selected = [
            record
            for record in records
            if record.get("intensity") is None
            or float(record["intensity"]) >= cutoff
        ]
    return [
        str(record["route_file"])
        for record in (selected or records)
    ]


__all__ = ["curriculum_demand_routes"]
