#!/usr/bin/env python3
"""Detector-limited, map-agnostic traffic-signal observations.

Schema v4 deliberately separates the simulator's internal state from the
information exposed to the policy.  SUMO vehicle positions are used only to
emulate physical detector zones and aggregate pulses; routes, destinations,
vehicle waiting times, and per-vehicle ETAs are never policy inputs.

Each graph node is an incoming detector lane, not an oracle movement.  A
shared through/right lane therefore stays ambiguous.  The policy receives
rolling detector measurements, deployable queue/delay estimates, controller
state, and static phase membership.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from map_agnostic_tls import (
    BLOCKED_OCCUPANCY,
    DEFAULT_MAX_GREEN,
    DEFAULT_MIN_GREEN,
    MAX_MOVEMENTS,
    ObservationSnapshot,
    RewardWeights,
    SATURATION_FLOW_PER_LANE,
    STARVATION_REFERENCE_SECONDS,
    VEHICLE_STORAGE_METERS,
    clamp01,
    clip_signed,
    normalized_reward,
    safe_div,
)


SCHEMA_VERSION = 4

# Detector schema v4 needs a larger catalog than historical schema v3 because
# the frozen corpus contains joined native controllers with more than sixteen
# distinct stable green states. Keep this v4-only so existing schema-v3
# checkpoints retain their original Discrete(17) action space.
MAX_PHASES = 32

# Feature dimensions retain the proven permutation-equivariant graph
# architecture. Schema v4 uses its own padded phase dimension and checkpoint
# metadata, so v3 and v4 checkpoints must never be interchanged.
DETECTOR_FEATURE_NAMES = (
    "stopbar_presence",
    "stopbar_occupancy_short",
    "stopbar_occupancy_60s",
    "advance_presence",
    "advance_occupancy_short",
    "advance_occupancy_60s",
    "arrival_rate_short",
    "arrival_rate_60s",
    "departure_rate_short",
    "departure_rate_60s",
    "estimated_queue",
    "estimated_queue_trend",
    "estimated_delay",
    "speed_ratio",
    "downstream_occupancy",
    "estimated_pressure",
    "detector_call_duration",
    "currently_green",
    "time_since_service",
    "turn_left",
    "turn_straight",
    "turn_right",
    "speed_available",
    "sensor_health",
)
MOVEMENT_FEATURE_NAMES = DETECTOR_FEATURE_NAMES
MOVEMENT_FEATURE_DIM = len(MOVEMENT_FEATURE_NAMES)

PHASE_FEATURE_NAMES = (
    "is_current",
    "green_elapsed",
    "mean_estimated_queue",
    "mean_estimated_pressure",
    "mean_downstream_space",
    "max_detector_call_duration",
    "detector_coverage",
    "mean_arrival_rate",
)
PHASE_FEATURE_DIM = len(PHASE_FEATURE_NAMES)

GLOBAL_FEATURE_NAMES = (
    "green_elapsed",
    "minimum_green_progress",
    "mean_stopbar_occupancy",
    "mean_estimated_queue",
    "mean_arrival_rate",
    "mean_downstream_occupancy",
    "max_time_since_service",
    "detector_coverage",
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


def observation_space():
    """Return the Gymnasium space without importing Gym at module import."""

    try:
        from gymnasium import spaces
    except ImportError as exc:  # pragma: no cover - training hosts provide it
        raise ImportError("Install gymnasium to construct the observation space.") from exc
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
    if not observations:
        raise ValueError("Cannot stack an empty observation list.")
    return {
        key: np.stack([np.asarray(observation[key]) for observation in observations])
        for key in observations[0]
    }


def _environment_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class DetectorSensorConfig:
    """Physical sensor layout and sim-to-real corruption parameters."""

    profile: str = "mixed"
    nominal_decision_seconds: float = 10.0
    stopbar_zone_meters: float = 12.0
    advance_distance_meters: float = 80.0
    advance_zone_meters: float = 10.0
    downstream_zone_meters: float = 30.0
    history_seconds: float = 60.0
    observation_noise_std: float = 0.02
    calibration_jitter: float = 0.05
    transient_dropout_probability: float = 0.03
    stuck_detector_probability: float = 0.01
    max_latency_decisions: int = 1
    mixed_speed_probability: float = 0.50
    mixed_downstream_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.profile not in {"loops", "camera", "mixed"}:
            raise ValueError("profile must be one of: loops, camera, mixed")
        positive = (
            self.nominal_decision_seconds,
            self.stopbar_zone_meters,
            self.advance_distance_meters,
            self.advance_zone_meters,
            self.downstream_zone_meters,
            self.history_seconds,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Detector distances, history, and cadence must be positive.")
        probabilities = (
            self.transient_dropout_probability,
            self.stuck_detector_probability,
            self.mixed_speed_probability,
            self.mixed_downstream_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Detector probabilities must be in [0, 1].")
        if self.observation_noise_std < 0.0 or self.calibration_jitter < 0.0:
            raise ValueError("Noise and calibration jitter must be nonnegative.")
        if self.max_latency_decisions < 0:
            raise ValueError("max_latency_decisions must be nonnegative.")

    def signature(self) -> tuple[Any, ...]:
        return tuple(getattr(self, item.name) for item in self.__dataclass_fields__.values())


def sensor_config_from_environment(training: bool = False) -> DetectorSensorConfig:
    """Build the evaluation configuration used by command-line comparators."""

    return DetectorSensorConfig(
        profile=os.environ.get("DETECTOR_SENSOR_PROFILE", "mixed"),
        nominal_decision_seconds=_environment_float("DETECTOR_DECISION_SECONDS", 10.0),
        observation_noise_std=_environment_float(
            "DETECTOR_NOISE_STD", 0.02 if training else 0.0
        ),
        calibration_jitter=_environment_float(
            "DETECTOR_CALIBRATION_JITTER", 0.05 if training else 0.0
        ),
        transient_dropout_probability=_environment_float(
            "DETECTOR_DROPOUT_PROB", 0.03 if training else 0.0
        ),
        stuck_detector_probability=_environment_float(
            "DETECTOR_STUCK_PROB", 0.01 if training else 0.0
        ),
        max_latency_decisions=_environment_int(
            "DETECTOR_MAX_LATENCY_DECISIONS", 1 if training else 0
        ),
    )


@dataclass(frozen=True)
class DetectorGroupSpec:
    incoming_edge: str
    outgoing_edge: str
    turn: str
    turns: tuple[str, ...]
    incoming_lanes: tuple[str, ...]
    outgoing_lanes: tuple[str, ...]
    signal_indices: tuple[int, ...]


@dataclass(frozen=True)
class DetectorIntersectionTopology:
    tls_id: str
    movements: tuple[DetectorGroupSpec, ...]
    phase_positions: tuple[int, ...]
    phase_members: tuple[tuple[int, ...], ...]
    phase_weights: tuple[tuple[float, ...], ...]
    adjacency: np.ndarray = field(repr=False, compare=False)


@dataclass
class DetectorLayout:
    speed_available: bool
    downstream_available: bool
    stopbar_fault: str
    advance_fault: str
    calibration_scale: float
    calibration_bias: float
    latency_decisions: int


@dataclass(frozen=True)
class DetectorReading:
    timestamp: float
    stopbar_presence: float
    stopbar_occupancy: float
    advance_presence: float
    advance_occupancy: float
    arrivals: float
    departures: float
    speed_ratio: float
    downstream_occupancy: float
    speed_available: float
    downstream_available: float
    sensor_health: float


class DetectorTrafficSnapshot:
    """Map-wide cache used only to emulate aggregate roadside sensors.

    This class intentionally has no route, destination, or waiting-time cache.
    Vehicle identifiers and positions are an implementation detail of the
    virtual detector, never part of the policy observation.
    """

    def __init__(self, traci_module: Any):
        self.traci = traci_module
        self.lane_lengths: dict[str, float] = {}
        self.lane_speed_limits: dict[str, float] = {}
        self.lane_vehicle_ids: dict[str, tuple[str, ...]] = {}
        self.vehicle_positions: dict[str, float] = {}
        self.vehicle_speeds: dict[str, float] = {}
        self.sim_time = 0.0

    @staticmethod
    def _safe(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:
            return default

    def refresh(self, adapters: Sequence["DetectorRealisticTLSAdapter"]) -> None:
        lanes: set[str] = set()
        for adapter in adapters:
            for group in adapter.topology.movements:
                lanes.update(group.incoming_lanes)
                lanes.update(group.outgoing_lanes)

        for lane_id in lanes:
            if lane_id not in self.lane_lengths:
                self.lane_lengths[lane_id] = max(
                    1.0,
                    float(self._safe(lambda lane=lane_id: self.traci.lane.getLength(lane), 1.0)),
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
            for lane_id in lanes
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
        self.sim_time = float(
            self._safe(lambda: self.traci.simulation.getTime(), self.sim_time)
        )


class DetectorRealisticTLSAdapter:
    """Build one local detector-limited observation and legal action mask."""

    def __init__(
        self,
        controller: dict[str, Any],
        traci_module: Any,
        simulation_module: Any,
        snapshot_cache: DetectorTrafficSnapshot | None = None,
        sensor_config: DetectorSensorConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.controller = controller
        self.traci = traci_module
        self.sim = simulation_module
        self.snapshot_cache = snapshot_cache
        self.sensor_config = sensor_config or sensor_config_from_environment(False)
        self.rng = rng or np.random.default_rng(0)
        self.topology = self._build_topology()
        self.previous_snapshot: ObservationSnapshot | None = None
        self.last_snapshot: ObservationSnapshot | None = None
        self.reset_history()

    @property
    def tls_id(self) -> str:
        return str(self.controller.get("tls_id", ""))

    @property
    def phase_count(self) -> int:
        return len(self.topology.phase_positions)

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
                if isinstance(label, str):
                    value = label.rsplit("-", 1)[-1]
        value = str(value or "O").upper()
        return value if value in {"L", "S", "R"} else "O"

    def _phase_signal_weights(self, phase: Mapping[str, Any]) -> dict[int, float]:
        state = phase.get("state")
        if isinstance(state, str):
            return {
                index: (1.0 if char == "G" else 0.5)
                for index, char in enumerate(state)
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

    def _build_topology(self) -> DetectorIntersectionTopology:
        controlled_links = self._safe(
            lambda: self.traci.trafficlight.getControlledLinks(self.tls_id), ()
        )
        by_lane: dict[str, dict[str, set[Any]]] = {}
        for signal_index, signal_links in enumerate(controlled_links or ()):
            for link in signal_links or ():
                if len(link) < 2 or not link[0] or not link[1]:
                    continue
                incoming_lane, outgoing_lane = str(link[0]), str(link[1])
                entry = by_lane.setdefault(
                    incoming_lane,
                    {"outgoing": set(), "outgoing_edges": set(), "signals": set(), "turns": set()},
                )
                entry["outgoing"].add(outgoing_lane)
                entry["outgoing_edges"].add(self._edge_id(outgoing_lane))
                entry["signals"].add(int(signal_index))
                entry["turns"].add(self._turn_type(incoming_lane, outgoing_lane))

        groups = tuple(
            DetectorGroupSpec(
                incoming_edge=self._edge_id(incoming_lane),
                outgoing_edge="|".join(sorted(str(x) for x in value["outgoing_edges"])),
                turn=(next(iter(value["turns"])) if len(value["turns"]) == 1 else "O"),
                turns=tuple(sorted(str(x) for x in value["turns"])),
                incoming_lanes=(incoming_lane,),
                outgoing_lanes=tuple(sorted(str(x) for x in value["outgoing"])),
                signal_indices=tuple(sorted(int(x) for x in value["signals"])),
            )
            for incoming_lane, value in sorted(by_lane.items())
        )
        if not groups:
            raise ValueError(f"TLS {self.tls_id!r} has no detectorizable incoming lanes.")
        if len(groups) > MAX_MOVEMENTS:
            raise ValueError(
                f"TLS {self.tls_id!r} has {len(groups)} detector groups, exceeding "
                f"MAX_MOVEMENTS={MAX_MOVEMENTS}."
            )

        phase_positions: list[int] = []
        phase_members: list[tuple[int, ...]] = []
        phase_weights: list[tuple[float, ...]] = []
        for phase_position, phase in enumerate(self.controller.get("phases", ())):
            signal_weights = self._phase_signal_weights(phase)
            members = tuple(
                index
                for index, group in enumerate(groups)
                if any(signal in signal_weights for signal in group.signal_indices)
            )
            if not members:
                continue
            phase_positions.append(phase_position)
            phase_members.append(members)
            phase_weights.append(
                tuple(
                    max(
                        (
                            signal_weights.get(signal, 0.0)
                            for signal in groups[index].signal_indices
                        ),
                        default=0.0,
                    )
                    for index in members
                )
            )
        if len(phase_positions) < 2:
            raise ValueError(f"TLS {self.tls_id!r} has fewer than two usable phases.")
        if len(phase_positions) > MAX_PHASES:
            raise ValueError(
                f"TLS {self.tls_id!r} has {len(phase_positions)} phases, exceeding "
                f"MAX_PHASES={MAX_PHASES}."
            )

        adjacency = np.eye(len(groups), dtype=np.float32)
        co_phase = [set() for _ in groups]
        for members in phase_members:
            for index in members:
                co_phase[index].update(members)
        for left_index, left in enumerate(groups):
            for right_index, right in enumerate(groups):
                if (
                    left.incoming_edge == right.incoming_edge
                    or set(left.outgoing_lanes) & set(right.outgoing_lanes)
                    or right_index in co_phase[left_index]
                ):
                    adjacency[left_index, right_index] = 1.0

        return DetectorIntersectionTopology(
            tls_id=self.tls_id,
            movements=groups,
            phase_positions=tuple(phase_positions),
            phase_members=tuple(phase_members),
            phase_weights=tuple(phase_weights),
            adjacency=adjacency,
        )

    def _sample_fault(self) -> str:
        if self.rng.random() >= self.sensor_config.stuck_detector_probability:
            return "none"
        return "on" if self.rng.random() < 0.5 else "off"

    def _sample_layout(self) -> DetectorLayout:
        profile = self.sensor_config.profile
        if profile == "camera":
            speed_available = downstream_available = True
        elif profile == "loops":
            speed_available = downstream_available = False
        else:
            speed_available = bool(
                self.rng.random() < self.sensor_config.mixed_speed_probability
            )
            downstream_available = bool(
                self.rng.random() < self.sensor_config.mixed_downstream_probability
            )
        scale = float(
            self.rng.lognormal(0.0, self.sensor_config.calibration_jitter)
            if self.sensor_config.calibration_jitter > 0.0
            else 1.0
        )
        bias = float(
            self.rng.normal(0.0, self.sensor_config.calibration_jitter * 0.10)
            if self.sensor_config.calibration_jitter > 0.0
            else 0.0
        )
        latency = (
            int(self.rng.integers(0, self.sensor_config.max_latency_decisions + 1))
            if self.sensor_config.max_latency_decisions > 0
            else 0
        )
        return DetectorLayout(
            speed_available=speed_available,
            downstream_available=downstream_available,
            stopbar_fault=self._sample_fault(),
            advance_fault=self._sample_fault(),
            calibration_scale=scale,
            calibration_bias=bias,
            latency_decisions=latency,
        )

    def reset_history(self) -> None:
        count = len(self.topology.movements)
        maximum_history = max(
            2,
            int(
                math.ceil(
                    self.sensor_config.history_seconds
                    / self.sensor_config.nominal_decision_seconds
                )
            )
            + self.sensor_config.max_latency_decisions
            + 2,
        )
        self._last_time: float | None = None
        self._previous_distances: list[dict[str, float]] = [dict() for _ in range(count)]
        self._previous_near_stop: list[set[str]] = [set() for _ in range(count)]
        self._queue_estimates = np.zeros(count, dtype=np.float64)
        self._previous_queue_norm = np.zeros(count, dtype=np.float64)
        self._call_duration = np.zeros(count, dtype=np.float64)
        self._last_served = np.zeros(count, dtype=np.float64)
        self._histories: list[deque[DetectorReading]] = [
            deque(maxlen=maximum_history) for _ in range(count)
        ]
        self._latency_buffers: list[deque[DetectorReading]] = [
            deque(maxlen=maximum_history) for _ in range(count)
        ]
        self._layouts = [self._sample_layout() for _ in range(count)]
        # Aggregate pulse count retained for the detector-derived training
        # reward. Raw vehicle identifiers never leave this adapter.
        self.last_detected_departures = 0.0
        self.previous_snapshot = None
        self.last_snapshot = None

    def _lane_length(self, lane_id: str) -> float:
        if self.snapshot_cache is not None and lane_id in self.snapshot_cache.lane_lengths:
            return self.snapshot_cache.lane_lengths[lane_id]
        return max(1.0, float(self._safe(lambda: self.traci.lane.getLength(lane_id), 1.0)))

    def _lane_speed_limit(self, lane_id: str) -> float:
        if self.snapshot_cache is not None and lane_id in self.snapshot_cache.lane_speed_limits:
            return self.snapshot_cache.lane_speed_limits[lane_id]
        return max(0.1, float(self._safe(lambda: self.traci.lane.getMaxSpeed(lane_id), 13.9)))

    def _lane_vehicle_ids(self, lane_id: str) -> tuple[str, ...]:
        if self.snapshot_cache is not None:
            return self.snapshot_cache.lane_vehicle_ids.get(lane_id, ())
        return tuple(
            str(value)
            for value in self._safe(lambda: self.traci.lane.getLastStepVehicleIDs(lane_id), ())
        )

    def _position(self, vehicle_id: str) -> float:
        if self.snapshot_cache is not None:
            return float(self.snapshot_cache.vehicle_positions.get(vehicle_id, 0.0))
        return float(self._safe(lambda: self.traci.vehicle.getLanePosition(vehicle_id), 0.0))

    def _speed(self, vehicle_id: str) -> float:
        if self.snapshot_cache is not None:
            return max(0.0, float(self.snapshot_cache.vehicle_speeds.get(vehicle_id, 0.0)))
        return max(0.0, float(self._safe(lambda: self.traci.vehicle.getSpeed(vehicle_id), 0.0)))

    @staticmethod
    def _apply_fault(value: float, fault: str) -> float:
        if fault == "on":
            return 1.0
        if fault == "off":
            return 0.0
        return value

    def _raw_reading(
        self,
        index: int,
        group: DetectorGroupSpec,
        sim_time: float,
        update_history: bool,
    ) -> tuple[DetectorReading, dict[str, float]]:
        lane_id = group.incoming_lanes[0]
        lane_length = self._lane_length(lane_id)
        advance_line = max(
            2.0,
            min(self.sensor_config.advance_distance_meters, 0.80 * lane_length),
        )
        half_advance_zone = 0.5 * self.sensor_config.advance_zone_meters
        ids = set(self._lane_vehicle_ids(lane_id))
        distances = {
            vehicle_id: max(0.0, lane_length - self._position(vehicle_id))
            for vehicle_id in ids
        }
        stop_ids = {
            vehicle_id
            for vehicle_id, distance in distances.items()
            if distance <= self.sensor_config.stopbar_zone_meters
        }
        advance_ids = {
            vehicle_id
            for vehicle_id, distance in distances.items()
            if abs(distance - advance_line) <= half_advance_zone
        }
        stop_capacity = max(
            1.0, self.sensor_config.stopbar_zone_meters / VEHICLE_STORAGE_METERS
        )
        advance_capacity = max(
            1.0, self.sensor_config.advance_zone_meters / VEHICLE_STORAGE_METERS
        )
        stop_occupancy = clamp01(len(stop_ids) / stop_capacity)
        advance_occupancy = clamp01(len(advance_ids) / advance_capacity)

        previous = self._previous_distances[index]
        arrivals = 0
        if self._last_time is not None:
            for vehicle_id, distance in distances.items():
                old_distance = previous.get(vehicle_id)
                if old_distance is None:
                    arrivals += int(distance <= advance_line)
                else:
                    arrivals += int(old_distance > advance_line >= distance)
        departures = 0
        if self._last_time is not None:
            departures = len(self._previous_near_stop[index] - ids)

        layout = self._layouts[index]
        speed_ratio = -1.0
        if layout.speed_available:
            observed_ids = advance_ids or stop_ids
            if observed_ids:
                speed_ratio = clamp01(
                    safe_div(
                        sum(self._speed(vehicle_id) for vehicle_id in observed_ids),
                        len(observed_ids) * self._lane_speed_limit(lane_id),
                    )
                )
            else:
                speed_ratio = 1.0

        downstream_occupancy = -1.0
        if layout.downstream_available:
            downstream_ids: set[str] = set()
            downstream_capacity = 0.0
            for outgoing_lane in group.outgoing_lanes:
                outgoing_length = self._lane_length(outgoing_lane)
                zone = min(outgoing_length, self.sensor_config.downstream_zone_meters)
                downstream_capacity += zone / VEHICLE_STORAGE_METERS
                downstream_ids.update(
                    vehicle_id
                    for vehicle_id in self._lane_vehicle_ids(outgoing_lane)
                    if self._position(vehicle_id) <= zone
                )
            downstream_occupancy = clamp01(
                safe_div(len(downstream_ids), max(1.0, downstream_capacity))
            )

        # The sensor model corrupts only aggregate readings.  It never exposes
        # the vehicle IDs used by this virtual-instrument implementation.
        stop_occupancy = self._apply_fault(stop_occupancy, layout.stopbar_fault)
        advance_occupancy = self._apply_fault(
            advance_occupancy, layout.advance_fault
        )
        stop_presence = float(stop_occupancy > 0.0)
        advance_presence = float(advance_occupancy > 0.0)
        health = float(
            layout.stopbar_fault == "none" and layout.advance_fault == "none"
        )
        transient_dropout = bool(
            update_history
            and self.rng.random() < self.sensor_config.transient_dropout_probability
        )
        if transient_dropout:
            stop_presence = stop_occupancy = 0.0
            advance_presence = advance_occupancy = 0.0
            arrivals = departures = 0
            speed_ratio = -1.0
            downstream_occupancy = -1.0
            health = 0.0

        if health > 0.0:
            for name, value in (
                ("stop", stop_occupancy),
                ("advance", advance_occupancy),
            ):
                adjusted = value * layout.calibration_scale + layout.calibration_bias
                if self.sensor_config.observation_noise_std > 0.0 and update_history:
                    adjusted += float(
                        self.rng.normal(0.0, self.sensor_config.observation_noise_std)
                    )
                adjusted = clamp01(adjusted)
                if name == "stop":
                    stop_occupancy = adjusted
                    stop_presence = float(adjusted > 0.01)
                else:
                    advance_occupancy = adjusted
                    advance_presence = float(adjusted > 0.01)

        reading = DetectorReading(
            timestamp=sim_time,
            stopbar_presence=stop_presence,
            stopbar_occupancy=stop_occupancy,
            advance_presence=advance_presence,
            advance_occupancy=advance_occupancy,
            arrivals=float(arrivals),
            departures=float(departures),
            speed_ratio=float(speed_ratio),
            downstream_occupancy=float(downstream_occupancy),
            speed_available=float(speed_ratio >= 0.0),
            downstream_available=float(downstream_occupancy >= 0.0),
            sensor_health=health,
        )
        if update_history:
            self._previous_distances[index] = distances
            self._previous_near_stop[index] = {
                vehicle_id
                for vehicle_id, distance in distances.items()
                if distance <= max(30.0, self.sensor_config.stopbar_zone_meters)
            }
        return reading, {"advance_line": advance_line, "lane_length": lane_length}

    def _delayed_reading(
        self, index: int, reading: DetectorReading, update_history: bool
    ) -> DetectorReading:
        if not update_history:
            return reading
        buffer = self._latency_buffers[index]
        buffer.append(reading)
        delay = self._layouts[index].latency_decisions
        return list(buffer)[max(0, len(buffer) - 1 - delay)]

    @staticmethod
    def _history_window(
        history: deque[DetectorReading], sim_time: float, seconds: float
    ) -> list[DetectorReading]:
        values = [item for item in history if sim_time - item.timestamp <= seconds + 1e-6]
        return values or ([history[-1]] if history else [])

    def _current_candidate(self) -> int | None:
        original_position = int(self.controller.get("phase_pos", -1))
        try:
            return self.topology.phase_positions.index(original_position)
        except ValueError:
            return None

    def observe(self, update_history: bool = True) -> ObservationSnapshot:
        sim_time = float(self._safe(lambda: self.traci.simulation.getTime(), 0.0))
        dt = (
            max(1.0, sim_time - self._last_time)
            if self._last_time is not None
            else self.sensor_config.nominal_decision_seconds
        )
        current_candidate = self._current_candidate()
        current_members = (
            set(self.topology.phase_members[current_candidate])
            if current_candidate is not None and self.controller.get("mode") == "green"
            else set()
        )
        observation = empty_observation()
        details: list[dict[str, float]] = []
        detected_departures = 0.0

        for index, group in enumerate(self.topology.movements):
            raw, geometry = self._raw_reading(
                index, group, sim_time, update_history
            )
            reading = self._delayed_reading(index, raw, update_history)
            detected_departures += max(0.0, reading.departures)
            history = self._histories[index]
            if update_history:
                history.append(reading)
            short = self._history_window(
                history, sim_time, max(dt, self.sensor_config.nominal_decision_seconds)
            )
            medium = self._history_window(
                history, sim_time, self.sensor_config.history_seconds
            )
            short_seconds = max(dt, self.sensor_config.nominal_decision_seconds)
            medium_seconds = max(
                short_seconds,
                (medium[-1].timestamp - medium[0].timestamp + dt) if medium else dt,
            )
            mean = lambda values: safe_div(sum(values), max(1, len(values)))
            stop_short = mean([item.stopbar_occupancy for item in short])
            stop_medium = mean([item.stopbar_occupancy for item in medium])
            advance_short = mean([item.advance_occupancy for item in short])
            advance_medium = mean([item.advance_occupancy for item in medium])
            arrivals_short = sum(item.arrivals for item in short)
            arrivals_medium = sum(item.arrivals for item in medium)
            departures_short = sum(item.departures for item in short)
            departures_medium = sum(item.departures for item in medium)
            arrival_rate_short = clamp01(
                safe_div(
                    arrivals_short,
                    SATURATION_FLOW_PER_LANE * short_seconds,
                )
            )
            arrival_rate_medium = clamp01(
                safe_div(
                    arrivals_medium,
                    SATURATION_FLOW_PER_LANE * medium_seconds,
                )
            )
            departure_rate_short = clamp01(
                safe_div(
                    departures_short,
                    SATURATION_FLOW_PER_LANE * short_seconds,
                )
            )
            departure_rate_medium = clamp01(
                safe_div(
                    departures_medium,
                    SATURATION_FLOW_PER_LANE * medium_seconds,
                )
            )

            storage_capacity = max(
                1.0,
                min(geometry["lane_length"], geometry["advance_line"])
                / VEHICLE_STORAGE_METERS,
            )
            previous_queue = float(self._queue_estimates[index])
            queue = max(0.0, previous_queue + reading.arrivals - reading.departures)
            if reading.stopbar_presence > 0.0:
                queue = max(queue, 1.0)
            # Persistent occupation at both loops indicates a queue reaching
            # the advance detector.  This is intentionally conservative and
            # uses only detector history, not vehicle positions.
            if stop_medium >= 0.80 and advance_medium >= 0.80:
                queue = max(queue, geometry["advance_line"] / VEHICLE_STORAGE_METERS)
            queue = min(storage_capacity, queue)
            queue_norm = clamp01(queue / storage_capacity)
            queue_trend = clip_signed(
                (queue_norm - float(self._previous_queue_norm[index]))
                * self.sensor_config.nominal_decision_seconds
                / max(1.0, dt)
            )

            currently_green = index in current_members
            demand_call = reading.stopbar_presence > 0.0 or queue > 0.25
            call_duration = (
                float(self._call_duration[index]) + dt if demand_call else 0.0
            )
            if currently_green:
                call_duration = max(0.0, call_duration - 2.0 * dt)
                if update_history:
                    self._last_served[index] = sim_time
            time_since_service = (
                0.0
                if currently_green
                else clamp01(
                    (sim_time - float(self._last_served[index]))
                    / STARVATION_REFERENCE_SECONDS
                )
            )
            delay_proxy = clamp01(
                queue_norm
                * min(1.0, call_duration / STARVATION_REFERENCE_SECONDS)
            )
            downstream = reading.downstream_occupancy
            pressure = clip_signed(
                queue_norm - (downstream if downstream >= 0.0 else 0.0)
            )
            turns = set(group.turns)
            features = np.asarray(
                [
                    reading.stopbar_presence,
                    stop_short,
                    stop_medium,
                    reading.advance_presence,
                    advance_short,
                    advance_medium,
                    arrival_rate_short,
                    arrival_rate_medium,
                    departure_rate_short,
                    departure_rate_medium,
                    queue_norm,
                    queue_trend,
                    delay_proxy,
                    reading.speed_ratio,
                    downstream,
                    pressure,
                    clamp01(call_duration / STARVATION_REFERENCE_SECONDS),
                    float(currently_green),
                    time_since_service,
                    float("L" in turns),
                    float("S" in turns),
                    float("R" in turns),
                    reading.speed_available,
                    reading.sensor_health,
                ],
                dtype=np.float32,
            )
            observation["movements"][index] = np.clip(features, -1.0, 1.0)
            observation["movement_mask"][index] = 1.0
            details.append(
                {
                    "stopbar": stop_short,
                    "queue": queue_norm,
                    "arrival": arrival_rate_medium,
                    "delay": delay_proxy,
                    "speed": reading.speed_ratio,
                    "downstream": downstream,
                    "pressure": pressure,
                    "call": clamp01(call_duration / STARVATION_REFERENCE_SECONDS),
                    "starvation": time_since_service,
                    "health": reading.sensor_health,
                }
            )
            if update_history:
                self._queue_estimates[index] = queue
                self._previous_queue_norm[index] = queue_norm
                self._call_duration[index] = call_duration

        group_count = len(self.topology.movements)
        observation["movement_adjacency"][:group_count, :group_count] = (
            self.topology.adjacency
        )
        elapsed = max(0.0, float(self.controller.get("phase_elapsed", 0.0)))
        phase_downstream = np.full(self.phase_count, -1.0, dtype=np.float32)
        phase_pressures = np.zeros(self.phase_count, dtype=np.float32)

        for candidate, member_tuple in enumerate(self.topology.phase_members):
            members = list(member_tuple)
            weights = list(self.topology.phase_weights[candidate])
            observation["phase_membership"][candidate, members] = np.asarray(
                weights, dtype=np.float32
            )
            member_details = [details[index] for index in members]
            weight_total = max(1e-6, sum(weights))
            weighted = lambda key: safe_div(
                sum(weight * item[key] for weight, item in zip(weights, member_details)),
                weight_total,
            )
            queue = weighted("queue")
            pressure = weighted("pressure")
            available_downstream = [
                item["downstream"] for item in member_details if item["downstream"] >= 0.0
            ]
            downstream = (
                safe_div(sum(available_downstream), len(available_downstream))
                if available_downstream
                else -1.0
            )
            downstream_space = 1.0 - downstream if downstream >= 0.0 else -1.0
            max_call = max((item["call"] for item in member_details), default=0.0)
            coverage = weighted("health")
            arrival = weighted("arrival")
            is_current = float(candidate == current_candidate)
            observation["phase_features"][candidate] = np.asarray(
                [
                    is_current,
                    clamp01(elapsed / DEFAULT_MAX_GREEN) if is_current else 0.0,
                    queue,
                    pressure,
                    downstream_space,
                    max_call,
                    coverage,
                    arrival,
                ],
                dtype=np.float32,
            )
            phase_downstream[candidate] = downstream
            phase_pressures[candidate] = max(0.0, pressure)

        available_speeds = [item["speed"] for item in details if item["speed"] >= 0.0]
        available_downstream = [
            item["downstream"] for item in details if item["downstream"] >= 0.0
        ]
        mean_stopbar = safe_div(sum(item["stopbar"] for item in details), group_count)
        mean_queue = safe_div(sum(item["queue"] for item in details), group_count)
        mean_arrival = safe_div(sum(item["arrival"] for item in details), group_count)
        mean_delay = safe_div(sum(item["delay"] for item in details), group_count)
        mean_speed = (
            safe_div(sum(available_speeds), len(available_speeds))
            if available_speeds
            else 1.0
        )
        mean_downstream = (
            safe_div(sum(available_downstream), len(available_downstream))
            if available_downstream
            else 0.0
        )
        downstream_feature = mean_downstream if available_downstream else -1.0
        max_starvation = max((item["starvation"] for item in details), default=0.0)
        coverage = safe_div(sum(item["health"] for item in details), group_count)
        spillback = (
            safe_div(
                sum(clamp01((value - 0.85) / 0.15) ** 2 for value in available_downstream),
                len(available_downstream),
            )
            if available_downstream
            else 0.0
        )
        served_pressure = (
            float(phase_pressures[current_candidate])
            if current_candidate is not None
            else 0.0
        )
        observation["global_features"] = np.asarray(
            [
                clamp01(elapsed / DEFAULT_MAX_GREEN),
                clamp01(elapsed / DEFAULT_MIN_GREEN),
                mean_stopbar,
                mean_queue,
                mean_arrival,
                downstream_feature,
                max_starvation,
                coverage,
            ],
            dtype=np.float32,
        )

        snapshot = ObservationSnapshot(
            observation=observation,
            sim_time=sim_time,
            # The shared reward container has this legacy field, but schema v4
            # deliberately leaves it empty. Training uses aggregate departure
            # pulses instead of per-vehicle identity differences.
            vehicle_ids=frozenset(),
            mean_queue_density=clamp01(mean_queue),
            mean_vehicle_density=clamp01(mean_stopbar),
            mean_wait=clamp01(mean_delay),
            mean_speed_ratio=clamp01(mean_speed),
            mean_downstream_occupancy=clamp01(mean_downstream),
            spillback=clamp01(spillback),
            max_starvation=clamp01(max_starvation),
            served_pressure=clamp01(served_pressure),
            total_incoming_lanes=group_count,
            phase_downstream_occupancy=phase_downstream,
        )
        if update_history:
            self.previous_snapshot = self.last_snapshot
            self.last_snapshot = snapshot
            self.last_detected_departures = detected_departures
            self._last_time = sim_time
        return snapshot

    def action_mask(
        self,
        min_green: float = DEFAULT_MIN_GREEN,
        max_green: float = DEFAULT_MAX_GREEN,
        block_threshold: float = BLOCKED_OCCUPANCY,
    ) -> np.ndarray:
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
            else np.full(self.phase_count, -1.0, dtype=np.float32)
        )
        for candidate in range(self.phase_count):
            if candidate == current:
                continue
            # A negative value means no downstream detector is installed.  It
            # must not be silently replaced by oracle SUMO occupancy.
            if candidate < len(downstream) and downstream[candidate] >= block_threshold:
                continue
            mask[candidate + 1] = True
        if elapsed >= max_green and not mask[1:].any():
            alternatives = [index for index in range(self.phase_count) if index != current]
            if alternatives:
                best = min(
                    alternatives,
                    key=lambda index: (
                        float(downstream[index])
                        if index < len(downstream) and downstream[index] >= 0.0
                        else 0.0
                    ),
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
            "incoming_edges": len({item.incoming_edge for item in self.topology.movements}),
            "outgoing_edges": len({item.outgoing_edge for item in self.topology.movements}),
            "sensor_profile": self.sensor_config.profile,
        }


def adapter_for_controller(
    controller: dict[str, Any],
    traci_module: Any,
    simulation_module: Any,
    snapshot_cache: DetectorTrafficSnapshot | None = None,
    sensor_config: DetectorSensorConfig | None = None,
    rng: np.random.Generator | None = None,
) -> DetectorRealisticTLSAdapter:
    config = sensor_config or sensor_config_from_environment(False)
    adapter = controller.get("_detector_realistic_adapter")
    signature = controller.get("_detector_realistic_config_signature")
    if not isinstance(adapter, DetectorRealisticTLSAdapter) or signature != config.signature():
        adapter = DetectorRealisticTLSAdapter(
            controller,
            traci_module,
            simulation_module,
            snapshot_cache=snapshot_cache,
            sensor_config=config,
            rng=rng,
        )
        controller["_detector_realistic_adapter"] = adapter
        controller["_detector_realistic_config_signature"] = config.signature()
    elif snapshot_cache is not None:
        adapter.snapshot_cache = snapshot_cache
    return adapter


__all__ = [
    "DETECTOR_FEATURE_NAMES",
    "GLOBAL_FEATURE_DIM",
    "GLOBAL_FEATURE_NAMES",
    "MAX_MOVEMENTS",
    "MAX_PHASES",
    "MOVEMENT_FEATURE_DIM",
    "MOVEMENT_FEATURE_NAMES",
    "PHASE_FEATURE_DIM",
    "PHASE_FEATURE_NAMES",
    "DetectorRealisticTLSAdapter",
    "DetectorSensorConfig",
    "DetectorTrafficSnapshot",
    "RewardWeights",
    "adapter_for_controller",
    "empty_observation",
    "normalized_reward",
    "observation_space",
    "sensor_config_from_environment",
    "stack_observations",
]
