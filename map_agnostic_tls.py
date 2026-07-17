#!/usr/bin/env python3
"""Map-agnostic traffic-signal representation, masking, and reward utilities.

The old controller exposed twelve compass-labelled movements and four fixed
phase slots.  That makes action 2 mean something different when lane ordering
or phase topology changes.  This module instead builds a set of physical
movements (incoming edge -> outgoing edge), analytically normalizes their live
traffic state, and represents every safe phase by the movements it serves.

The code deliberately has no PyTorch/SB3 dependency.  It can be used by the
Gym environment, batched all-TLS evaluation, diagnostics, and unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


# Padded only for batching.  Candidate scoring is weight-shared, so a padded
# position has no learned semantic meaning.  Increase these before training if
# validate_controller() reports a larger intersection.
MAX_MOVEMENTS = 160
MAX_PHASES = 16

MOVEMENT_FEATURE_NAMES = (
    "queue_density",
    "vehicle_density",
    "speed_ratio",
    "mean_wait_log",
    "max_wait_log",
    "arrival_rate_ratio",
    "queue_trend",
    "eta_0_5_density",
    "eta_5_15_density",
    "eta_15_30_density",
    "downstream_occupancy",
    "downstream_speed_ratio",
    "downstream_space",
    "normalized_pressure",
    "blocked_exit_ratio",
    "currently_green",
    "time_since_service",
    "turn_left",
    "turn_straight",
    "turn_right",
    "turn_other",
    "incoming_lane_count",
    "outgoing_lane_count",
    "speed_limit_ratio",
)
MOVEMENT_FEATURE_DIM = len(MOVEMENT_FEATURE_NAMES)

PHASE_FEATURE_NAMES = (
    "is_current",
    "green_elapsed",
    "mean_queue_density",
    "mean_pressure",
    "mean_downstream_space",
    "max_starvation",
    "movement_fraction",
    "mean_wait",
)
PHASE_FEATURE_DIM = len(PHASE_FEATURE_NAMES)

GLOBAL_FEATURE_NAMES = (
    "green_elapsed",
    "minimum_green_progress",
    "mean_queue_density",
    "mean_vehicle_density",
    "mean_speed_ratio",
    "mean_downstream_occupancy",
    "max_starvation",
    "movement_fraction",
)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

OBSERVATION_KEYS = (
    "movements",
    "movement_mask",
    "movement_adjacency",
    "phase_membership",
    "phase_features",
    "global_features",
)

DETECTION_DISTANCE_METERS = 180.0
DOWNSTREAM_DISTANCE_METERS = 120.0
VEHICLE_STORAGE_METERS = 7.5
WAIT_LOG_REFERENCE_SECONDS = 300.0
STARVATION_REFERENCE_SECONDS = 90.0
SATURATION_FLOW_PER_LANE = 0.50  # vehicles / second; configurable approximation
SPEED_LIMIT_REFERENCE_MPS = 22.22  # about 50 mph
MAX_REFERENCE_LANES = 6.0
BLOCKED_OCCUPANCY = 0.90

DEFAULT_MIN_GREEN = 6.0
DEFAULT_MAX_GREEN = 55.0


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def clip_signed(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / max(1e-9, float(denominator))


def log_ratio(value: float, reference: float) -> float:
    return clamp01(math.log1p(max(0.0, value)) / math.log1p(max(1.0, reference)))


def observation_space():
    """Return the Gymnasium Dict space without importing Gym at module import."""

    try:
        from gymnasium import spaces
    except ImportError as exc:  # pragma: no cover - exercised on training hosts
        raise ImportError("Install gymnasium to construct the RL observation space.") from exc

    return spaces.Dict(
        {
            "movements": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(MAX_MOVEMENTS, MOVEMENT_FEATURE_DIM),
                dtype=np.float32,
            ),
            "movement_mask": spaces.Box(
                low=0.0, high=1.0, shape=(MAX_MOVEMENTS,), dtype=np.float32
            ),
            "movement_adjacency": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(MAX_MOVEMENTS, MAX_MOVEMENTS),
                dtype=np.float32,
            ),
            "phase_membership": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(MAX_PHASES, MAX_MOVEMENTS),
                dtype=np.float32,
            ),
            "phase_features": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(MAX_PHASES, PHASE_FEATURE_DIM),
                dtype=np.float32,
            ),
            "global_features": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(GLOBAL_FEATURE_DIM,),
                dtype=np.float32,
            ),
        }
    )


def empty_observation() -> dict[str, np.ndarray]:
    return {
        "movements": np.zeros((MAX_MOVEMENTS, MOVEMENT_FEATURE_DIM), dtype=np.float32),
        "movement_mask": np.zeros((MAX_MOVEMENTS,), dtype=np.float32),
        "movement_adjacency": np.eye(MAX_MOVEMENTS, dtype=np.float32),
        "phase_membership": np.zeros((MAX_PHASES, MAX_MOVEMENTS), dtype=np.float32),
        "phase_features": np.zeros((MAX_PHASES, PHASE_FEATURE_DIM), dtype=np.float32),
        "global_features": np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32),
    }


def stack_observations(observations: Sequence[Mapping[str, np.ndarray]]):
    """Batch Dict observations while retaining support for legacy arrays."""

    if not observations:
        raise ValueError("Cannot stack an empty observation list.")
    first = observations[0]
    if isinstance(first, Mapping):
        return {
            key: np.stack([np.asarray(obs[key]) for obs in observations], axis=0)
            for key in first
        }
    return np.stack(observations, axis=0)


@dataclass(frozen=True)
class MovementSpec:
    incoming_edge: str
    outgoing_edge: str
    turn: str
    incoming_lanes: tuple[str, ...]
    outgoing_lanes: tuple[str, ...]
    signal_indices: tuple[int, ...]


@dataclass(frozen=True)
class IntersectionTopology:
    tls_id: str
    movements: tuple[MovementSpec, ...]
    # Candidate index -> original controller phase position.
    phase_positions: tuple[int, ...]
    phase_members: tuple[tuple[int, ...], ...]
    # Aligned with phase_members: 1.0 protected green, 0.5 permissive/stop green.
    phase_weights: tuple[tuple[float, ...], ...]
    adjacency: np.ndarray = field(repr=False, compare=False)


@dataclass
class ObservationSnapshot:
    observation: dict[str, np.ndarray]
    sim_time: float
    vehicle_ids: frozenset[str]
    mean_queue_density: float
    mean_vehicle_density: float
    mean_wait: float
    mean_speed_ratio: float
    mean_downstream_occupancy: float
    spillback: float
    max_starvation: float
    served_pressure: float
    total_incoming_lanes: int
    phase_downstream_occupancy: np.ndarray


@dataclass(frozen=True)
class RewardWeights:
    discharged: float = 2.0
    served_pressure: float = 0.65
    queue_improvement: float = 1.0
    queue_level: float = 1.0
    waiting: float = 0.35
    spillback: float = 1.50
    starvation: float = 0.40
    switch: float = 0.03
    forced_switch: float = 0.25


class MapTrafficSnapshot:
    """One map-wide cache of TraCI values reused by every TLS observation.

    The single-TLS implementation queried the same lane and vehicle repeatedly
    for different movements. In all-TLS training that would multiply socket
    calls by the number of intersections. This cache performs each relevant
    query at most once per policy decision while keeping topology/static lane
    properties across decisions.
    """

    def __init__(self, traci_module: Any, simulation_module: Any):
        self.traci = traci_module
        self.sim = simulation_module
        self.lane_lengths: dict[str, float] = {}
        self.lane_speed_limits: dict[str, float] = {}
        self.lane_vehicle_ids: dict[str, tuple[str, ...]] = {}
        self.vehicle_positions: dict[str, float] = {}
        self.vehicle_speeds: dict[str, float] = {}
        self.vehicle_waits: dict[str, float] = {}
        self.vehicle_next_edges: dict[str, str | None] = {}
        self.outgoing_has_space: dict[str, bool] = {}
        self.sim_time = 0.0

    @staticmethod
    def _safe(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:
            return default

    def refresh(self, adapters: Sequence["MapAgnosticTLSAdapter"]) -> None:
        incoming_lanes: set[str] = set()
        outgoing_lanes: set[str] = set()
        for adapter in adapters:
            for movement in adapter.topology.movements:
                incoming_lanes.update(movement.incoming_lanes)
                outgoing_lanes.update(movement.outgoing_lanes)
        all_lanes = incoming_lanes | outgoing_lanes

        for lane_id in all_lanes:
            if lane_id not in self.lane_lengths:
                self.lane_lengths[lane_id] = max(
                    1.0,
                    float(
                        self._safe(
                            lambda lane=lane_id: self.traci.lane.getLength(lane),
                            1.0,
                        )
                    ),
                )
            if lane_id not in self.lane_speed_limits:
                self.lane_speed_limits[lane_id] = max(
                    0.1,
                    float(
                        self._safe(
                            lambda lane=lane_id: self.traci.lane.getMaxSpeed(lane),
                            13.9,
                        )
                    ),
                )
            self.lane_vehicle_ids[lane_id] = tuple(
                str(vehicle_id)
                for vehicle_id in self._safe(
                    lambda lane=lane_id: self.traci.lane.getLastStepVehicleIDs(lane),
                    (),
                )
            )

        all_vehicle_ids = {
            vehicle_id
            for lane_id in all_lanes
            for vehicle_id in self.lane_vehicle_ids.get(lane_id, ())
        }
        route_vehicle_ids = {
            vehicle_id
            for lane_id in incoming_lanes
            for vehicle_id in self.lane_vehicle_ids.get(lane_id, ())
        }
        self.vehicle_positions = {
            vehicle_id: float(
                self._safe(
                    lambda vehicle=vehicle_id: self.traci.vehicle.getLanePosition(vehicle),
                    0.0,
                )
            )
            for vehicle_id in all_vehicle_ids
        }
        self.vehicle_speeds = {
            vehicle_id: max(
                0.0,
                float(
                    self._safe(
                        lambda vehicle=vehicle_id: self.traci.vehicle.getSpeed(vehicle),
                        0.0,
                    )
                ),
            )
            for vehicle_id in all_vehicle_ids
        }
        self.vehicle_waits = {
            vehicle_id: max(
                0.0,
                float(
                    self._safe(
                        lambda vehicle=vehicle_id: self.traci.vehicle.getWaitingTime(vehicle),
                        0.0,
                    )
                ),
            )
            for vehicle_id in all_vehicle_ids
        }
        self.vehicle_next_edges = {}
        for vehicle_id in route_vehicle_ids:
            route = tuple(
                self._safe(
                    lambda vehicle=vehicle_id: self.traci.vehicle.getRoute(vehicle),
                    (),
                )
            )
            route_index = int(
                self._safe(
                    lambda vehicle=vehicle_id: self.traci.vehicle.getRouteIndex(vehicle),
                    -1,
                )
            )
            next_edge = (
                str(route[route_index + 1])
                if route_index >= 0 and route_index + 1 < len(route)
                else None
            )
            self.vehicle_next_edges[vehicle_id] = next_edge

        has_space = getattr(self.sim, "outgoing_lane_has_space", None)
        self.outgoing_has_space = {}
        if callable(has_space):
            for lane_id in outgoing_lanes:
                self.outgoing_has_space[lane_id] = bool(
                    self._safe(lambda lane=lane_id: has_space(lane), True)
                )
        self.sim_time = float(
            self._safe(lambda: self.traci.simulation.getTime(), self.sim_time)
        )


def normalized_reward(
    previous: ObservationSnapshot | None,
    current: ObservationSnapshot,
    local_cleared: int,
    decision_seconds: float,
    switched: bool,
    forced: bool = False,
    weights: RewardWeights = RewardWeights(),
) -> tuple[float, dict[str, float]]:
    """Capacity-normalized local reward; every term remains O(1) across maps."""

    discharge_capacity = max(
        1.0,
        SATURATION_FLOW_PER_LANE
        * max(1, current.total_incoming_lanes)
        * max(1.0, float(decision_seconds)),
    )
    discharge_ratio = clamp01(safe_div(local_cleared, discharge_capacity))
    queue_improvement = 0.0
    if previous is not None:
        queue_improvement = clip_signed(
            previous.mean_queue_density - current.mean_queue_density
        )

    components = {
        "discharged": weights.discharged * discharge_ratio,
        "served_pressure": weights.served_pressure * current.served_pressure,
        "queue_improvement": weights.queue_improvement * queue_improvement,
        "queue_level": -weights.queue_level * current.mean_queue_density,
        "waiting": -weights.waiting * current.mean_wait,
        "spillback": -weights.spillback * current.spillback,
        "starvation": -weights.starvation * current.max_starvation,
        "switch": -weights.switch if switched else 0.0,
        "forced_switch": -weights.forced_switch if forced else 0.0,
    }
    return float(sum(components.values())), components


class MapAgnosticTLSAdapter:
    """Build invariant observations and action mappings for one SUMO TLS."""

    def __init__(
        self,
        controller: dict[str, Any],
        traci_module: Any,
        simulation_module: Any,
        snapshot_cache: MapTrafficSnapshot | None = None,
        detection_distance: float = DETECTION_DISTANCE_METERS,
        downstream_distance: float = DOWNSTREAM_DISTANCE_METERS,
    ):
        self.controller = controller
        self.traci = traci_module
        self.sim = simulation_module
        self.snapshot_cache = snapshot_cache
        self.detection_distance = float(detection_distance)
        self.downstream_distance = float(downstream_distance)
        self.topology = self._build_topology()

        self.previous_snapshot: ObservationSnapshot | None = None
        self.last_snapshot: ObservationSnapshot | None = None
        self._last_time: float | None = None
        self._previous_ids: list[set[str]] = [set() for _ in self.topology.movements]
        self._previous_queue_density = np.zeros(len(self.topology.movements), dtype=np.float32)
        self._last_served = np.zeros(len(self.topology.movements), dtype=np.float64)

    @property
    def tls_id(self) -> str:
        return str(self.controller.get("tls_id", ""))

    @property
    def phase_count(self) -> int:
        return len(self.topology.phase_positions)

    def reset_history(self) -> None:
        self.previous_snapshot = None
        self.last_snapshot = None
        self._last_time = None
        self._previous_ids = [set() for _ in self.topology.movements]
        self._previous_queue_density = np.zeros(len(self.topology.movements), dtype=np.float32)
        self._last_served = np.zeros(len(self.topology.movements), dtype=np.float64)

    def _safe(self, fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:
            return default

    def _edge_id(self, lane_id: str) -> str:
        value = self._safe(lambda: self.traci.lane.getEdgeID(lane_id), None)
        if value:
            return str(value)
        return lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id

    def _turn_type(self, incoming_lane: str, outgoing_lane: str) -> str:
        value = None
        getter = getattr(self.sim, "get_sumo_link_direction", None)
        if callable(getter):
            value = self._safe(lambda: getter(incoming_lane, outgoing_lane), None)

        if value is None:
            classifier = getattr(self.sim, "classify_connection", None)
            if callable(classifier):
                label = self._safe(lambda: classifier(incoming_lane, outgoing_lane), None)
                if isinstance(label, str) and "-" in label:
                    value = label.rsplit("-", 1)[-1]
                elif isinstance(label, str):
                    value = label

        value = str(value or "O").upper()
        return value if value in {"L", "S", "R"} else "O"

    def _phase_signal_weights(self, phase: Mapping[str, Any]) -> dict[int, float]:
        state = phase.get("state")
        if isinstance(state, str):
            return {
                i: (1.0 if char == "G" else 0.5)
                for i, char in enumerate(state)
                if char in {"G", "g", "s"}
            }

        green: dict[int, float] = {}
        rules = phase.get("rules", {})
        movement_map = self.controller.get("movement_map", {})
        for label, char in rules.items():
            if char not in {"G", "g"}:
                continue
            for signal_index in movement_map.get(label, {}):
                index = int(signal_index)
                green[index] = max(green.get(index, 0.0), 1.0 if char == "G" else 0.5)
        return green

    def _build_topology(self) -> IntersectionTopology:
        controlled_links = self._safe(
            lambda: self.traci.trafficlight.getControlledLinks(self.tls_id), ()
        )
        grouped: dict[tuple[str, str, str], dict[str, set[Any]]] = {}

        for signal_index, signal_links in enumerate(controlled_links or ()):
            for link in signal_links or ():
                if len(link) < 2 or not link[0] or not link[1]:
                    continue
                incoming_lane, outgoing_lane = str(link[0]), str(link[1])
                key = (
                    self._edge_id(incoming_lane),
                    self._edge_id(outgoing_lane),
                    self._turn_type(incoming_lane, outgoing_lane),
                )
                entry = grouped.setdefault(
                    key, {"incoming": set(), "outgoing": set(), "signals": set()}
                )
                entry["incoming"].add(incoming_lane)
                entry["outgoing"].add(outgoing_lane)
                entry["signals"].add(int(signal_index))

        movements = tuple(
            MovementSpec(
                incoming_edge=key[0],
                outgoing_edge=key[1],
                turn=key[2],
                incoming_lanes=tuple(sorted(value["incoming"])),
                outgoing_lanes=tuple(sorted(value["outgoing"])),
                signal_indices=tuple(sorted(int(x) for x in value["signals"])),
            )
            for key, value in sorted(grouped.items(), key=lambda item: item[0])
        )

        if not movements:
            raise ValueError(f"TLS {self.tls_id!r} has no usable controlled movements.")
        if len(movements) > MAX_MOVEMENTS:
            raise ValueError(
                f"TLS {self.tls_id!r} has {len(movements)} movements, exceeding "
                f"MAX_MOVEMENTS={MAX_MOVEMENTS}. Increase the constant and retrain."
            )

        phase_positions: list[int] = []
        phase_members: list[tuple[int, ...]] = []
        phase_weights: list[tuple[float, ...]] = []
        for phase_pos, phase in enumerate(self.controller.get("phases", ())):
            signal_weights = self._phase_signal_weights(phase)
            members = tuple(
                i
                for i, movement in enumerate(movements)
                if any(index in signal_weights for index in movement.signal_indices)
            )
            if members:
                phase_positions.append(phase_pos)
                phase_members.append(members)
                phase_weights.append(
                    tuple(
                        max(
                            (signal_weights.get(index, 0.0) for index in movements[i].signal_indices),
                            default=0.0,
                        )
                        for i in members
                    )
                )

        if not phase_positions:
            raise ValueError(f"TLS {self.tls_id!r} has no non-empty safe phase candidates.")
        if len(phase_positions) > MAX_PHASES:
            raise ValueError(
                f"TLS {self.tls_id!r} has {len(phase_positions)} phases, exceeding "
                f"MAX_PHASES={MAX_PHASES}. Increase the constant and retrain."
            )

        adjacency = np.eye(len(movements), dtype=np.float32)
        co_phase = [set() for _ in movements]
        for members in phase_members:
            for i in members:
                co_phase[i].update(members)

        for i, left in enumerate(movements):
            for j, right in enumerate(movements):
                if (
                    left.incoming_edge == right.incoming_edge
                    or left.outgoing_edge == right.outgoing_edge
                    or j in co_phase[i]
                ):
                    adjacency[i, j] = 1.0

        return IntersectionTopology(
            tls_id=self.tls_id,
            movements=movements,
            phase_positions=tuple(phase_positions),
            phase_members=tuple(phase_members),
            phase_weights=tuple(phase_weights),
            adjacency=adjacency,
        )

    def _lane_length(self, lane_id: str) -> float:
        if self.snapshot_cache is not None and lane_id in self.snapshot_cache.lane_lengths:
            return self.snapshot_cache.lane_lengths[lane_id]
        return max(1.0, float(self._safe(lambda: self.traci.lane.getLength(lane_id), 1.0)))

    def _lane_speed_limit(self, lane_id: str) -> float:
        if (
            self.snapshot_cache is not None
            and lane_id in self.snapshot_cache.lane_speed_limits
        ):
            return self.snapshot_cache.lane_speed_limits[lane_id]
        return max(0.1, float(self._safe(lambda: self.traci.lane.getMaxSpeed(lane_id), 13.9)))

    def _lane_ids_in_window(self, lane_id: str, incoming: bool) -> list[str]:
        if self.snapshot_cache is not None:
            ids = list(self.snapshot_cache.lane_vehicle_ids.get(lane_id, ()))
        else:
            ids = list(self._safe(lambda: self.traci.lane.getLastStepVehicleIDs(lane_id), ()))
        length = self._lane_length(lane_id)
        window = self.detection_distance if incoming else self.downstream_distance
        selected: list[str] = []
        for vehicle_id in ids:
            if self.snapshot_cache is not None:
                pos = float(self.snapshot_cache.vehicle_positions.get(vehicle_id, 0.0))
            else:
                pos = float(self._safe(lambda v=vehicle_id: self.traci.vehicle.getLanePosition(v), 0.0))
            if incoming:
                if length - pos <= window + 1e-6:
                    selected.append(str(vehicle_id))
            elif pos <= window + 1e-6:
                selected.append(str(vehicle_id))
        return selected

    def _storage_capacity(self, lanes: Iterable[str], incoming: bool) -> float:
        window = self.detection_distance if incoming else self.downstream_distance
        return max(
            1.0,
            sum(min(self._lane_length(lane_id), window) / VEHICLE_STORAGE_METERS for lane_id in lanes),
        )

    def _vehicle_next_edge(self, vehicle_id: str) -> str | None:
        if self.snapshot_cache is not None:
            return self.snapshot_cache.vehicle_next_edges.get(vehicle_id)
        route = self._safe(lambda: tuple(self.traci.vehicle.getRoute(vehicle_id)), ())
        route_index = int(self._safe(lambda: self.traci.vehicle.getRouteIndex(vehicle_id), -1))
        if route_index >= 0 and route_index + 1 < len(route):
            return str(route[route_index + 1])
        return None

    def _vehicle_speed(self, vehicle_id: str) -> float:
        if self.snapshot_cache is not None:
            return max(0.0, float(self.snapshot_cache.vehicle_speeds.get(vehicle_id, 0.0)))
        return max(
            0.0,
            float(self._safe(lambda: self.traci.vehicle.getSpeed(vehicle_id), 0.0)),
        )

    def _vehicle_wait(self, vehicle_id: str) -> float:
        if self.snapshot_cache is not None:
            return max(0.0, float(self.snapshot_cache.vehicle_waits.get(vehicle_id, 0.0)))
        return max(
            0.0,
            float(self._safe(lambda: self.traci.vehicle.getWaitingTime(vehicle_id), 0.0)),
        )

    def _movement_features(
        self,
        movement_index: int,
        movement: MovementSpec,
        sim_time: float,
        dt: float,
        currently_green: bool,
        update_history: bool,
    ) -> tuple[np.ndarray, set[str], dict[str, float]]:
        in_vehicle_ids: set[str] = set()
        out_vehicle_ids: set[str] = set()
        for lane_id in movement.incoming_lanes:
            in_vehicle_ids.update(self._lane_ids_in_window(lane_id, incoming=True))
        # Shared approach lanes can support several turns.  Attribute vehicles
        # to their planned outgoing edge so an intersection with more possible
        # turns is not automatically assigned a larger queue.  If route data is
        # temporarily unavailable, retain the vehicle rather than hiding demand.
        in_vehicle_ids = {
            vehicle_id
            for vehicle_id in in_vehicle_ids
            if (next_edge := self._vehicle_next_edge(vehicle_id)) is None
            or next_edge == movement.outgoing_edge
        }
        for lane_id in movement.outgoing_lanes:
            out_vehicle_ids.update(self._lane_ids_in_window(lane_id, incoming=False))

        speed_cache: dict[str, float] = {}
        wait_cache: dict[str, float] = {}
        for vehicle_id in in_vehicle_ids | out_vehicle_ids:
            speed_cache[vehicle_id] = self._vehicle_speed(vehicle_id)
            wait_cache[vehicle_id] = self._vehicle_wait(vehicle_id)

        in_capacity = self._storage_capacity(movement.incoming_lanes, incoming=True)
        out_capacity = self._storage_capacity(movement.outgoing_lanes, incoming=False)
        queue = sum(1 for vehicle_id in in_vehicle_ids if speed_cache.get(vehicle_id, 0.0) < 0.1)
        queue_density = clamp01(safe_div(queue, in_capacity))
        vehicle_density = clamp01(safe_div(len(in_vehicle_ids), in_capacity))

        mean_speed_limit = safe_div(
            sum(self._lane_speed_limit(lane_id) for lane_id in movement.incoming_lanes),
            max(1, len(movement.incoming_lanes)),
        )
        mean_speed = safe_div(
            sum(speed_cache.get(vehicle_id, 0.0) for vehicle_id in in_vehicle_ids),
            max(1, len(in_vehicle_ids)),
        )
        speed_ratio = (
            clamp01(safe_div(mean_speed, mean_speed_limit)) if in_vehicle_ids else 1.0
        )

        waits = [wait_cache.get(vehicle_id, 0.0) for vehicle_id in in_vehicle_ids]
        mean_wait = safe_div(sum(waits), max(1, len(waits)))
        max_wait = max(waits, default=0.0)

        new_arrivals = (
            0
            if self._last_time is None
            else len(in_vehicle_ids - self._previous_ids[movement_index])
        )
        arrival_capacity = max(
            1e-6,
            SATURATION_FLOW_PER_LANE * max(1, len(movement.incoming_lanes)) * max(1.0, dt),
        )
        arrival_ratio = clamp01(safe_div(new_arrivals, arrival_capacity))
        queue_trend = 0.0
        if self._last_time is not None:
            queue_trend = clip_signed(
                safe_div(
                    queue_density - float(self._previous_queue_density[movement_index]),
                    max(1.0, dt),
                )
                * 10.0
            )

        eta_counts = [0.0, 0.0, 0.0]
        for lane_id in movement.incoming_lanes:
            length = self._lane_length(lane_id)
            for vehicle_id in self._lane_ids_in_window(lane_id, incoming=True):
                if vehicle_id not in in_vehicle_ids:
                    continue
                if self.snapshot_cache is not None:
                    pos = float(
                        self.snapshot_cache.vehicle_positions.get(vehicle_id, 0.0)
                    )
                else:
                    pos = float(self._safe(lambda v=vehicle_id: self.traci.vehicle.getLanePosition(v), 0.0))
                distance = max(0.0, length - pos)
                eta = distance / max(0.5, speed_cache.get(vehicle_id, 0.0))
                if eta < 5.0:
                    eta_counts[0] += 1.0
                elif eta < 15.0:
                    eta_counts[1] += 1.0
                elif eta < 30.0:
                    eta_counts[2] += 1.0

        downstream_occupancy = clamp01(safe_div(len(out_vehicle_ids), out_capacity))
        out_speed_limit = safe_div(
            sum(self._lane_speed_limit(lane_id) for lane_id in movement.outgoing_lanes),
            max(1, len(movement.outgoing_lanes)),
        )
        out_speed = safe_div(
            sum(speed_cache.get(vehicle_id, 0.0) for vehicle_id in out_vehicle_ids),
            max(1, len(out_vehicle_ids)),
        )
        downstream_speed_ratio = (
            clamp01(safe_div(out_speed, out_speed_limit)) if out_vehicle_ids else 1.0
        )
        downstream_space = clamp01(1.0 - downstream_occupancy)

        blocked_lanes = 0
        has_space = getattr(self.sim, "outgoing_lane_has_space", None)
        for lane_id in movement.outgoing_lanes:
            blocked = False
            if self.snapshot_cache is not None:
                blocked = not self.snapshot_cache.outgoing_has_space.get(lane_id, True)
            elif callable(has_space):
                blocked = not bool(self._safe(lambda lane=lane_id: has_space(lane), True))
            if blocked:
                blocked_lanes += 1
        blocked_ratio = max(
            downstream_occupancy >= BLOCKED_OCCUPANCY,
            safe_div(blocked_lanes, max(1, len(movement.outgoing_lanes))),
        )
        blocked_ratio = clamp01(float(blocked_ratio))

        pressure = clip_signed(queue_density - downstream_occupancy)
        if currently_green and update_history:
            self._last_served[movement_index] = sim_time
        time_since_service = 0.0 if currently_green else clamp01(
            safe_div(sim_time - float(self._last_served[movement_index]), STARVATION_REFERENCE_SECONDS)
        )

        turn = movement.turn
        in_lane_norm = clamp01(
            math.log1p(len(movement.incoming_lanes)) / math.log1p(MAX_REFERENCE_LANES)
        )
        out_lane_norm = clamp01(
            math.log1p(len(movement.outgoing_lanes)) / math.log1p(MAX_REFERENCE_LANES)
        )
        speed_limit_ratio = clamp01(mean_speed_limit / SPEED_LIMIT_REFERENCE_MPS)

        features = np.asarray(
            [
                queue_density,
                vehicle_density,
                speed_ratio,
                log_ratio(mean_wait, WAIT_LOG_REFERENCE_SECONDS),
                log_ratio(max_wait, WAIT_LOG_REFERENCE_SECONDS),
                arrival_ratio,
                queue_trend,
                clamp01(safe_div(eta_counts[0], in_capacity)),
                clamp01(safe_div(eta_counts[1], in_capacity)),
                clamp01(safe_div(eta_counts[2], in_capacity)),
                downstream_occupancy,
                downstream_speed_ratio,
                downstream_space,
                pressure,
                blocked_ratio,
                float(currently_green),
                time_since_service,
                float(turn == "L"),
                float(turn == "S"),
                float(turn == "R"),
                float(turn not in {"L", "S", "R"}),
                in_lane_norm,
                out_lane_norm,
                speed_limit_ratio,
            ],
            dtype=np.float32,
        )

        if update_history:
            self._previous_ids[movement_index] = set(in_vehicle_ids)
            self._previous_queue_density[movement_index] = queue_density

        details = {
            "queue_density": queue_density,
            "vehicle_density": vehicle_density,
            "speed_ratio": speed_ratio,
            "mean_wait": log_ratio(mean_wait, WAIT_LOG_REFERENCE_SECONDS),
            "downstream_occupancy": downstream_occupancy,
            "downstream_space": downstream_space,
            "pressure": pressure,
            "starvation": time_since_service,
            "blocked": blocked_ratio,
        }
        return features, in_vehicle_ids, details

    def _current_candidate(self) -> int | None:
        original_pos = int(self.controller.get("phase_pos", -1))
        try:
            return self.topology.phase_positions.index(original_pos)
        except ValueError:
            return None

    def observe(self, update_history: bool = True) -> ObservationSnapshot:
        sim_time = float(self._safe(lambda: self.traci.simulation.getTime(), 0.0))
        dt = max(1.0, sim_time - self._last_time) if self._last_time is not None else 1.0
        current_candidate = self._current_candidate()
        current_members = (
            set(self.topology.phase_members[current_candidate])
            if current_candidate is not None and self.controller.get("mode") == "green"
            else set()
        )

        observation = empty_observation()
        movement_details: list[dict[str, float]] = []
        all_vehicle_ids: set[str] = set()

        for index, movement in enumerate(self.topology.movements):
            features, vehicle_ids, details = self._movement_features(
                index,
                movement,
                sim_time=sim_time,
                dt=dt,
                currently_green=index in current_members,
                update_history=update_history,
            )
            observation["movements"][index] = features
            observation["movement_mask"][index] = 1.0
            movement_details.append(details)
            all_vehicle_ids.update(vehicle_ids)

        n_movements = len(self.topology.movements)
        observation["movement_adjacency"][:n_movements, :n_movements] = self.topology.adjacency

        elapsed = max(0.0, float(self.controller.get("phase_elapsed", 0.0)))
        phase_downstream = np.zeros(self.phase_count, dtype=np.float32)
        phase_pressure_values = np.zeros(self.phase_count, dtype=np.float32)

        for candidate, members_tuple in enumerate(self.topology.phase_members):
            members = list(members_tuple)
            service_weights = list(self.topology.phase_weights[candidate])
            observation["phase_membership"][candidate, members] = np.asarray(
                service_weights, dtype=np.float32
            )
            details = [movement_details[i] for i in members]
            service_total = max(1e-6, sum(service_weights))
            mean_queue = safe_div(
                sum(w * x["queue_density"] for w, x in zip(service_weights, details)),
                service_total,
            )
            mean_pressure = safe_div(
                sum(w * x["pressure"] for w, x in zip(service_weights, details)),
                service_total,
            )
            mean_space = safe_div(
                sum(w * x["downstream_space"] for w, x in zip(service_weights, details)),
                service_total,
            )
            max_starvation = max((x["starvation"] for x in details), default=0.0)
            mean_wait = safe_div(
                sum(w * x["mean_wait"] for w, x in zip(service_weights, details)),
                service_total,
            )
            is_current = float(candidate == current_candidate)

            observation["phase_features"][candidate] = np.asarray(
                [
                    is_current,
                    clamp01(elapsed / DEFAULT_MAX_GREEN) if is_current else 0.0,
                    clamp01(mean_queue),
                    clip_signed(mean_pressure),
                    clamp01(mean_space),
                    clamp01(max_starvation),
                    clamp01(len(members) / max(1, n_movements)),
                    clamp01(mean_wait),
                ],
                dtype=np.float32,
            )
            # One blocked exit is enough to create spillback even if the other
            # movements in the phase are empty.
            phase_downstream[candidate] = max(
                (max(x["downstream_occupancy"], x["blocked"]) for x in details),
                default=0.0,
            )
            phase_pressure_values[candidate] = max(0.0, mean_pressure)

        mean_queue = safe_div(sum(x["queue_density"] for x in movement_details), n_movements)
        mean_density = safe_div(sum(x["vehicle_density"] for x in movement_details), n_movements)
        mean_speed = safe_div(sum(x["speed_ratio"] for x in movement_details), n_movements)
        mean_downstream = safe_div(
            sum(x["downstream_occupancy"] for x in movement_details), n_movements
        )
        mean_wait = safe_div(sum(x["mean_wait"] for x in movement_details), n_movements)
        max_starvation = max((x["starvation"] for x in movement_details), default=0.0)
        spillback = safe_div(
            sum(
                clamp01((x["downstream_occupancy"] - 0.85) / 0.15) ** 2
                for x in movement_details
            ),
            n_movements,
        )
        served_pressure = (
            float(phase_pressure_values[current_candidate])
            if current_candidate is not None
            else 0.0
        )

        observation["global_features"] = np.asarray(
            [
                clamp01(elapsed / DEFAULT_MAX_GREEN),
                clamp01(elapsed / DEFAULT_MIN_GREEN),
                clamp01(mean_queue),
                clamp01(mean_density),
                clamp01(mean_speed),
                clamp01(mean_downstream),
                clamp01(max_starvation),
                clamp01(n_movements / MAX_MOVEMENTS),
            ],
            dtype=np.float32,
        )

        snapshot = ObservationSnapshot(
            observation=observation,
            sim_time=sim_time,
            vehicle_ids=frozenset(all_vehicle_ids),
            mean_queue_density=clamp01(mean_queue),
            mean_vehicle_density=clamp01(mean_density),
            mean_wait=clamp01(mean_wait),
            mean_speed_ratio=clamp01(mean_speed),
            mean_downstream_occupancy=clamp01(mean_downstream),
            spillback=clamp01(spillback),
            max_starvation=clamp01(max_starvation),
            served_pressure=clamp01(served_pressure),
            total_incoming_lanes=len(
                {lane for movement in self.topology.movements for lane in movement.incoming_lanes}
            ),
            phase_downstream_occupancy=phase_downstream,
        )

        if update_history:
            self.previous_snapshot = self.last_snapshot
            self.last_snapshot = snapshot
            self._last_time = sim_time
        return snapshot

    def action_mask(
        self,
        min_green: float = DEFAULT_MIN_GREEN,
        max_green: float = DEFAULT_MAX_GREEN,
        block_threshold: float = BLOCKED_OCCUPANCY,
    ) -> np.ndarray:
        """Return 0=hold, 1..N=candidate phase, remaining padded=False."""

        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        if self.controller.get("disabled") or self.controller.get("mode") != "green":
            mask[0] = True
            return mask

        elapsed = max(0.0, float(self.controller.get("phase_elapsed", 0.0)))
        if elapsed < min_green:
            mask[0] = True
            return mask

        if elapsed < max_green:
            mask[0] = True

        current = self._current_candidate()
        downstream = (
            self.last_snapshot.phase_downstream_occupancy
            if self.last_snapshot is not None
            else np.zeros(self.phase_count, dtype=np.float32)
        )
        for candidate in range(self.phase_count):
            if candidate == current:
                continue
            if candidate < len(downstream) and downstream[candidate] >= block_threshold:
                continue
            mask[candidate + 1] = True

        # At the hard maximum a spillback mask must never deadlock the signal.
        if elapsed >= max_green and not mask[1:].any():
            alternatives = [i for i in range(self.phase_count) if i != current]
            if alternatives:
                best = min(
                    alternatives,
                    key=lambda i: float(downstream[i]) if i < len(downstream) else 0.0,
                )
                mask[best + 1] = True

        if not mask.any():
            mask[0] = True
        return mask

    def action_to_phase_position(self, action: int) -> int | None:
        candidate = int(action) - 1
        if candidate < 0 or candidate >= self.phase_count:
            return None
        return int(self.topology.phase_positions[candidate])

    def validate_controller(self) -> dict[str, Any]:
        return {
            "tls_id": self.tls_id,
            "movements": len(self.topology.movements),
            "phases": self.phase_count,
            "incoming_edges": len({m.incoming_edge for m in self.topology.movements}),
            "outgoing_edges": len({m.outgoing_edge for m in self.topology.movements}),
        }


def adapter_for_controller(
    controller: dict[str, Any],
    traci_module: Any,
    simulation_module: Any,
    snapshot_cache: MapTrafficSnapshot | None = None,
) -> MapAgnosticTLSAdapter:
    adapter = controller.get("_map_agnostic_adapter")
    if not isinstance(adapter, MapAgnosticTLSAdapter):
        adapter = MapAgnosticTLSAdapter(
            controller,
            traci_module,
            simulation_module,
            snapshot_cache=snapshot_cache,
        )
        controller["_map_agnostic_adapter"] = adapter
    elif snapshot_cache is not None:
        adapter.snapshot_cache = snapshot_cache
    return adapter
