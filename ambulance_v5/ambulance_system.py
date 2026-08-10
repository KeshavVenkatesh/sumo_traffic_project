#!/usr/bin/env python3
"""Traffic-aware ambulance routing and exact per-step lifecycle accounting.

This module is intentionally independent of Gymnasium, SB3, and the learned
signal policy.  It owns the emergency-vehicle part of an experiment:

* deterministic origin/destination schedules that are identical across
  controller ablations;
* free-flow reference routes and travel times from SUMO ``findRoute``;
* optional congestion-aware routing with conservative reroute hysteresis;
* route-to-TLS/movement indexing for coordinated signal control;
* departed, arrived, teleported, collision, pending, and censored states from
  the exact SUMO microstep where each event occurs;
* dense, incremental travel progress, time loss, and stopped-time accounting.

The ambulance type is copied from an ordinary safe SUMO vehicle type and then
given emergency dimensions/performance.  It is not equipped with SUMO's
blue-light device, is not forcibly moved between lanes, and inherits ordinary
red-light and collision-avoidance behavior.  Signal priority therefore comes
from the controller rather than hidden vehicle teleportation or red running.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


AMBULANCE_ID_PREFIX = "ambulance_v5_"
AMBULANCE_TYPE_ID = "ambulance_v5"
TERMINAL_STATUSES = {
    "arrived",
    "teleported",
    "collided",
    "insertion_failed",
    "removed",
    "censored",
}


def _safe_call(fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _finite_positive(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) and number > 0.0 else float(fallback)


@dataclass(frozen=True)
class AmbulanceSystemConfig:
    """Routing, spawning, and measurement settings for one SUMO episode."""

    routing_mode: str = "traffic_aware"
    step_length_seconds: float = 1.0
    first_spawn_seconds: float = 60.0
    spawn_interval_seconds: float = 180.0
    spawn_jitter_seconds: float = 30.0
    max_ambulances: int = 12
    max_active_ambulances: int = 2
    planned_active_duration_factor: float = 1.5
    min_euclidean_distance: float = 1_200.0
    min_route_distance: float = 1_500.0
    min_route_edges: int = 12
    min_route_tls: int = 2
    route_attempts_per_ambulance: int = 120
    min_endpoint_edge_length: float = 20.0
    reroute_interval_seconds: float = 12.0
    reroute_jitter_seconds: float = 2.0
    reroute_min_savings_seconds: float = 8.0
    reroute_min_savings_fraction: float = 0.05
    no_reroute_within_tls_meters: float = 100.0
    pending_timeout_seconds: float = 120.0
    last_spawn_buffer_seconds: float = 300.0
    upcoming_tls_horizon: int = 3
    ambulance_type_id: str = AMBULANCE_TYPE_ID
    ambulance_id_prefix: str = AMBULANCE_ID_PREFIX
    depart_lane: str = "best"
    depart_position: str = "free"
    fail_on_schedule_shortfall: bool = True

    def __post_init__(self) -> None:
        if self.routing_mode not in {"free_flow", "traffic_aware"}:
            raise ValueError("routing_mode must be 'free_flow' or 'traffic_aware'")
        if self.step_length_seconds <= 0.0:
            raise ValueError("step_length_seconds must be positive")
        if self.first_spawn_seconds + 1e-9 < self.step_length_seconds:
            raise ValueError(
                "first_spawn_seconds must leave at least one SUMO step "
                "for boundary-safe prequeuing"
            )
        if self.spawn_interval_seconds <= 0.0:
            raise ValueError("spawn_interval_seconds must be positive")
        if self.spawn_jitter_seconds < 0.0:
            raise ValueError("spawn_jitter_seconds cannot be negative")
        if self.max_ambulances <= 0 or self.max_active_ambulances <= 0:
            raise ValueError("ambulance limits must be positive")
        if self.planned_active_duration_factor < 1.0:
            raise ValueError(
                "planned_active_duration_factor must be at least 1"
            )
        if (
            self.min_euclidean_distance < 0.0
            or self.min_route_distance < 0.0
            or self.min_endpoint_edge_length < 0.0
        ):
            raise ValueError("route distance thresholds cannot be negative")
        if self.min_route_tls < 1 or self.min_route_edges < 2:
            raise ValueError("route minima must be positive")
        if self.route_attempts_per_ambulance <= 0:
            raise ValueError("route_attempts_per_ambulance must be positive")
        if self.reroute_interval_seconds <= 0.0:
            raise ValueError("reroute_interval_seconds must be positive")
        if (
            self.reroute_jitter_seconds < 0.0
            or self.reroute_min_savings_seconds < 0.0
            or self.reroute_min_savings_fraction < 0.0
            or self.no_reroute_within_tls_meters < 0.0
        ):
            raise ValueError("reroute thresholds cannot be negative")
        if self.pending_timeout_seconds <= 0.0:
            raise ValueError("pending_timeout_seconds must be positive")
        if self.last_spawn_buffer_seconds < 0.0:
            raise ValueError("last_spawn_buffer_seconds cannot be negative")
        if self.upcoming_tls_horizon <= 0:
            raise ValueError("upcoming_tls_horizon must be positive")


@dataclass(frozen=True)
class RouteTLS:
    tls_id: str
    movement_index: int
    route_transition_index: int
    protected_candidate_actions: tuple[int, ...]
    permissive_candidate_actions: tuple[int, ...]

    @property
    def candidate_actions(self) -> tuple[int, ...]:
        return self.protected_candidate_actions + tuple(
            action
            for action in self.permissive_candidate_actions
            if action not in self.protected_candidate_actions
        )


class RouteTLSIndex:
    """Map route edge transitions and SUMO link indices to schema-v3 movements."""

    def __init__(self, adapters: Sequence[Any]):
        self.adapters_by_tls = {str(adapter.tls_id): adapter for adapter in adapters}
        self.by_transition: dict[tuple[str, str], list[tuple[str, int]]] = {}
        self.by_link_index: dict[tuple[str, int], int] = {}

        for adapter in adapters:
            tls_id = str(adapter.tls_id)
            for movement_index, movement in enumerate(adapter.topology.movements):
                key = (str(movement.incoming_edge), str(movement.outgoing_edge))
                self.by_transition.setdefault(key, []).append(
                    (tls_id, int(movement_index))
                )
                for link_index in movement.signal_indices:
                    self.by_link_index[(tls_id, int(link_index))] = int(
                        movement_index
                    )

    def movement_for_link(self, tls_id: str, link_index: int) -> int | None:
        return self.by_link_index.get((str(tls_id), int(link_index)))

    def route_tls(self, edges: Sequence[str]) -> tuple[RouteTLS, ...]:
        result: list[RouteTLS] = []
        previous: tuple[str, int] | None = None
        for transition_index, pair in enumerate(zip(edges, edges[1:])):
            for tls_id, movement_index in self.by_transition.get(
                (str(pair[0]), str(pair[1])), ()
            ):
                identity = (tls_id, movement_index)
                if identity == previous:
                    continue
                adapter = self.adapters_by_tls[tls_id]
                protected: list[int] = []
                permissive: list[int] = []
                for candidate, members in enumerate(adapter.topology.phase_members):
                    if movement_index not in members:
                        continue
                    member_position = members.index(movement_index)
                    weight = float(
                        adapter.topology.phase_weights[candidate][member_position]
                    )
                    action = candidate + 1
                    if weight >= 0.99:
                        protected.append(action)
                    else:
                        permissive.append(action)
                result.append(
                    RouteTLS(
                        tls_id=tls_id,
                        movement_index=movement_index,
                        route_transition_index=transition_index,
                        protected_candidate_actions=tuple(protected),
                        permissive_candidate_actions=tuple(permissive),
                    )
                )
                previous = identity
        return tuple(result)


@dataclass(frozen=True)
class AmbulanceScheduleEntry:
    sequence: int
    spawn_time: float
    origin_edge: str
    destination_edge: str
    free_flow_edges: tuple[str, ...]
    free_flow_time: float
    free_flow_distance: float
    route_tls: tuple[RouteTLS, ...]

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "spawn_time": round(float(self.spawn_time), 6),
            "origin_edge": self.origin_edge,
            "destination_edge": self.destination_edge,
            "free_flow_edges": list(self.free_flow_edges),
            "free_flow_time": round(float(self.free_flow_time), 6),
        }


def schedule_fingerprint(schedule: Sequence[AmbulanceScheduleEntry]) -> str:
    payload = [entry.fingerprint_payload() for entry in schedule]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class AmbulanceRecord:
    ambulance_id: str
    schedule: AmbulanceScheduleEntry
    requested_departure: float
    status: str = "pending"
    actual_departure: float | None = None
    end_time: float | None = None
    failure_reason: str | None = None
    route_edges: tuple[str, ...] = ()
    route_tls: tuple[RouteTLS, ...] = ()
    route_mode: str = "free_flow"
    route_updates: int = 0
    reroute_checks: int = 0
    last_reroute_time: float = float("-inf")
    next_reroute_time: float = float("inf")
    last_seen_time: float | None = None
    last_speed: float = 0.0
    last_remaining_distance: float | None = None
    last_time_loss: float = 0.0
    distance_driven: float = 0.0
    stopped_seconds: float = 0.0
    time_loss: float = 0.0
    next_tls: tuple[tuple[str, int, float, str], ...] = ()
    previous_first_tls: str | None = None
    cleared_tls: list[str] = field(default_factory=list)

    @property
    def destination_edge(self) -> str:
        return self.schedule.destination_edge

    @property
    def completed(self) -> bool:
        return self.status == "arrived"

    @property
    def failed(self) -> bool:
        return self.status in {
            "teleported",
            "collided",
            "insertion_failed",
            "removed",
        }


@dataclass
class AmbulanceDecisionDelta:
    progress_meters: dict[str, float] = field(default_factory=dict)
    time_loss_seconds: dict[str, float] = field(default_factory=dict)
    stopped_seconds: dict[str, float] = field(default_factory=dict)
    cleared_tls: list[tuple[str, str]] = field(default_factory=list)
    departed: list[str] = field(default_factory=list)
    arrived: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    censored: list[str] = field(default_factory=list)
    rerouted: list[str] = field(default_factory=list)
    collisions: set[str] = field(default_factory=set)
    teleports: set[str] = field(default_factory=set)


class AmbulanceSystem:
    """Own emergency demand, routing, lifecycle events, and dense progress."""

    def __init__(
        self,
        traci_module: Any,
        simulation_module: Any,
        adapters: Sequence[Any],
        raw_graph: Mapping[str, Sequence[str]],
        edge_metadata: Mapping[str, Any],
        sim_state: dict[str, Any],
        episode_seconds: float,
        schedule_seed: int,
        config: AmbulanceSystemConfig = AmbulanceSystemConfig(),
    ):
        self.traci = traci_module
        self.sim = simulation_module
        self.adapters = list(adapters)
        self.raw_graph = raw_graph
        self.edge_metadata = edge_metadata
        self.sim_state = sim_state
        self.episode_seconds = float(episode_seconds)
        self.config = config
        self.schedule_seed = int(schedule_seed)
        self.rng = random.Random(self.schedule_seed)
        self.route_index = RouteTLSIndex(self.adapters)
        self.records: dict[str, AmbulanceRecord] = {}
        # ``findRoute`` needs the requested vType to exist, so the safe
        # passenger-derived ambulance type must be installed before the
        # deterministic O/D schedule is constructed.
        self._ensure_vehicle_type()
        self.schedule = self._build_schedule()
        self.schedule_hash = schedule_fingerprint(self.schedule)
        self._schedule_cursor = 0
        self._route_counter = 0
        self._decision_delta = AmbulanceDecisionDelta()
        self.last_decision_delta = AmbulanceDecisionDelta()
        self.all_collision_vehicle_ids: set[str] = set()
        self.all_teleported_vehicle_ids: set[str] = set()

        # The legacy emergency spawner remains disabled.  These hooks execute
        # inside each one-second SUMO step and are removed with sim_state when
        # the episode is reset.
        before_hooks = self.sim_state.setdefault(
            "before_simulation_step_hooks", []
        )
        if callable(before_hooks):
            before_hooks = [before_hooks]
            self.sim_state["before_simulation_step_hooks"] = before_hooks
        before_hooks.append(self.before_simulation_step)
        after_hooks = self.sim_state.setdefault(
            "after_simulation_step_hooks", []
        )
        if callable(after_hooks):
            after_hooks = [after_hooks]
            self.sim_state["after_simulation_step_hooks"] = after_hooks
        after_hooks.append(self.after_simulation_step)
        self.sim_state["active_ambulances"] = {}

    def begin_decision(self) -> None:
        self._decision_delta = AmbulanceDecisionDelta()

    def end_decision(self) -> AmbulanceDecisionDelta:
        self.last_decision_delta = self._decision_delta
        return self.last_decision_delta

    def _routing_constant(self, name: str, fallback: int) -> int:
        constants = getattr(self.traci, "constants", None)
        return int(getattr(constants, name, fallback))

    def _find_route(
        self,
        origin: str,
        destination: str,
        aggregated: bool,
        depart: float | None = None,
    ) -> Any:
        routing_mode = self._routing_constant(
            "ROUTING_MODE_AGGREGATED" if aggregated else "ROUTING_MODE_DEFAULT",
            1 if aggregated else 0,
        )
        depart_time = (
            float(depart)
            if depart is not None
            else float(_safe_call(lambda: self.traci.simulation.getTime(), 0.0))
        )
        finder = self.traci.simulation.findRoute
        try:
            return finder(
                origin,
                destination,
                self.config.ambulance_type_id,
                depart_time,
                routing_mode,
            )
        except TypeError:
            try:
                return finder(
                    origin,
                    destination,
                    self.config.ambulance_type_id,
                    depart_time,
                )
            except TypeError:
                return finder(
                    origin,
                    destination,
                    self.config.ambulance_type_id,
                )

    def _edge_length(self, edge_id: str) -> float:
        helper = getattr(self.sim, "edge_length", None)
        if callable(helper):
            return _finite_positive(
                _safe_call(
                    lambda: helper(edge_id, self.edge_metadata),
                    0.0,
                ),
                1.0,
            )
        lane_count = int(
            _safe_call(lambda: self.traci.edge.getLaneNumber(edge_id), 1)
        )
        for lane_index in range(max(1, lane_count)):
            value = _safe_call(
                lambda i=lane_index: self.traci.lane.getLength(
                    f"{edge_id}_{i}"
                ),
                0.0,
            )
            if float(value or 0.0) > 0.0:
                return float(value)
        return 1.0

    def _route_distance(self, edges: Sequence[str]) -> float:
        helper = getattr(self.sim, "route_distance", None)
        if callable(helper):
            value = _safe_call(
                lambda: helper(edges, self.edge_metadata),
                None,
            )
            if value is not None:
                return max(0.0, float(value))
        return float(sum(self._edge_length(str(edge)) for edge in edges))

    def _edge_xy(self, edge_id: str) -> tuple[float, float] | None:
        helper = getattr(self.sim, "edge_xy", None)
        if callable(helper):
            value = _safe_call(
                lambda: helper(edge_id, self.edge_metadata),
                None,
            )
            if value is not None and len(value) >= 2:
                return float(value[0]), float(value[1])
        return None

    def _euclidean_distance(self, first: str, second: str) -> float:
        left = self._edge_xy(first)
        right = self._edge_xy(second)
        if left is None or right is None:
            return float("inf")
        return math.hypot(left[0] - right[0], left[1] - right[1])

    def _candidate_edges(self) -> list[str]:
        return sorted(
            str(edge_id)
            for edge_id in self.raw_graph
            if edge_id
            and not str(edge_id).startswith(":")
            and self._edge_length(str(edge_id))
            >= self.config.min_endpoint_edge_length
        )

    def _build_schedule(self) -> tuple[AmbulanceScheduleEntry, ...]:
        candidates = self._candidate_edges()
        if len(candidates) < 2:
            raise RuntimeError("The map has fewer than two ambulance endpoints")
        weights = [max(1.0, self._edge_length(edge)) for edge in candidates]

        entries: list[AmbulanceScheduleEntry] = []
        planned_end_times: list[float] = []
        spawn_time = max(0.0, float(self.config.first_spawn_seconds))
        buffered_horizon = self.episode_seconds - max(
            self.config.last_spawn_buffer_seconds,
            0.25 * self.config.spawn_interval_seconds,
        )
        latest_useful_spawn = (
            max(spawn_time, buffered_horizon)
            if spawn_time <= self.episode_seconds
            else buffered_horizon
        )
        while (
            len(entries) < self.config.max_ambulances
            and spawn_time <= latest_useful_spawn + 1e-9
        ):
            planned_end_times = [
                end_time
                for end_time in planned_end_times
                if end_time > spawn_time + 1e-9
            ]
            if (
                len(planned_end_times)
                >= self.config.max_active_ambulances
            ):
                # Concurrency is constrained while the schedule is built,
                # using only controller-independent free-flow durations. It
                # must never be enforced later using controller-dependent
                # arrival times.
                spawn_time = min(planned_end_times)
                continue

            selected: AmbulanceScheduleEntry | None = None
            for _attempt in range(
                self.config.route_attempts_per_ambulance
            ):
                origin = self.rng.choices(
                    candidates, weights=weights, k=1
                )[0]
                destination = self.rng.choices(
                    candidates, weights=weights, k=1
                )[0]
                if origin == destination:
                    continue
                if (
                    self._euclidean_distance(origin, destination)
                    < self.config.min_euclidean_distance
                ):
                    continue
                path = _safe_call(
                    lambda: self._find_route(
                        origin,
                        destination,
                        aggregated=False,
                        depart=spawn_time,
                    ),
                    None,
                )
                edges = tuple(
                    str(edge)
                    for edge in getattr(path, "edges", ()) or ()
                )
                if (
                    len(edges) < self.config.min_route_edges
                    or edges[0] != origin
                    or edges[-1] != destination
                ):
                    continue
                distance = self._route_distance(edges)
                if distance < self.config.min_route_distance:
                    continue
                route_tls = self.route_index.route_tls(edges)
                if len(route_tls) < self.config.min_route_tls:
                    continue
                free_flow_time = _finite_positive(
                    getattr(path, "travelTime", None),
                    distance / 13.9,
                )
                selected = AmbulanceScheduleEntry(
                    sequence=len(entries),
                    spawn_time=float(spawn_time),
                    origin_edge=origin,
                    destination_edge=destination,
                    free_flow_edges=edges,
                    free_flow_time=free_flow_time,
                    free_flow_distance=distance,
                    route_tls=route_tls,
                )
                break

            if selected is None:
                if self.config.fail_on_schedule_shortfall:
                    raise RuntimeError(
                        "Could not construct ambulance schedule entry "
                        f"{len(entries)} after "
                        f"{self.config.route_attempts_per_ambulance} "
                        "route attempts. Reduce the route-distance/TLS "
                        "minimums only if this map is genuinely too small."
                    )
                break

            entries.append(selected)
            planned_end_times.append(
                spawn_time
                + self.config.planned_active_duration_factor
                * selected.free_flow_time
            )
            jitter = self.rng.uniform(
                -self.config.spawn_jitter_seconds,
                self.config.spawn_jitter_seconds,
            )
            spawn_time += max(
                1.0,
                self.config.spawn_interval_seconds + jitter,
            )

        if self.config.fail_on_schedule_shortfall and not entries:
            raise RuntimeError(
                "Could not construct any ambulance schedule entries."
            )
        return tuple(entries)

    def _ensure_vehicle_type(self) -> None:
        api = self.traci.vehicletype
        type_ids = {str(value) for value in api.getIDList()}
        target = self.config.ambulance_type_id
        if target not in type_ids:
            source = next(
                (
                    candidate
                    for candidate in (
                        "global_car",
                        "DEFAULT_VEHTYPE",
                    )
                    if candidate in type_ids
                ),
                None,
            )
            if source is None:
                source = next(
                    (
                        value
                        for value in sorted(type_ids)
                        if value.endswith("__passenger")
                    ),
                    None,
                )
            if source is None:
                raise RuntimeError(
                    "No safe passenger vehicle type is available to copy for "
                    "the ambulance."
                )
            api.copy(source, target)

        settings = (
            # Keep passenger permissions so an ambulance can use every edge
            # used by the controller-independent passenger demand.  The
            # emergency appearance/performance is independent of vClass, and
            # no blue-light device is installed.
            ("setVehicleClass", "passenger"),
            ("setShapeClass", "emergency"),
            ("setLength", 5.95),
            ("setWidth", 2.04),
            ("setMinGap", 1.0),
            ("setAccel", 3.5),
            ("setDecel", 6.0),
            ("setEmergencyDecel", 9.0),
            # Tau must not be below the one-second simulation step.
            ("setTau", 1.0),
            ("setSpeedFactor", 1.35),
            ("setMaxSpeed", 33.33),
            ("setColor", (255, 50, 50, 255)),
        )
        for method_name, value in settings:
            method = getattr(api, method_name, None)
            if callable(method):
                method(target, value)

    def _initial_route(
        self, entry: AmbulanceScheduleEntry, sim_time: float
    ) -> tuple[tuple[str, ...], tuple[RouteTLS, ...], str]:
        if self.config.routing_mode != "traffic_aware":
            return entry.free_flow_edges, entry.route_tls, "free_flow"
        path = _safe_call(
            lambda: self._find_route(
                entry.origin_edge,
                entry.destination_edge,
                aggregated=True,
                depart=sim_time,
            ),
            None,
        )
        edges = tuple(str(edge) for edge in getattr(path, "edges", ()) or ())
        route_tls = self.route_index.route_tls(edges) if edges else ()
        if (
            len(edges) >= 2
            and edges[0] == entry.origin_edge
            and edges[-1] == entry.destination_edge
            and len(route_tls) >= self.config.min_route_tls
        ):
            return edges, route_tls, "traffic_aware"
        return entry.free_flow_edges, entry.route_tls, "free_flow_fallback"

    def _spawn(self, entry: AmbulanceScheduleEntry, sim_time: float) -> None:
        ambulance_id = f"{self.config.ambulance_id_prefix}{entry.sequence}"
        route_id = (
            f"{self.config.ambulance_id_prefix}route_{entry.sequence}_"
            f"{self._route_counter}"
        )
        self._route_counter += 1
        # The request is queued one SUMO microstep early so an ambulance whose
        # scheduled departure equals a policy boundary is already observable
        # when that policy decision is made.  Its actual departure time stays
        # exogenous and equal to ``entry.spawn_time``.
        edges, route_tls, route_mode = self._initial_route(
            entry, entry.spawn_time
        )
        record = AmbulanceRecord(
            ambulance_id=ambulance_id,
            schedule=entry,
            requested_departure=entry.spawn_time,
            route_edges=edges,
            route_tls=route_tls,
            route_mode=route_mode,
            next_reroute_time=(
                entry.spawn_time
                + self._next_reroute_delay(entry.sequence, 0)
                if self.config.routing_mode == "traffic_aware"
                else float("inf")
            ),
        )
        self.records[ambulance_id] = record
        try:
            self.traci.route.add(route_id, list(edges))
            self.traci.vehicle.add(
                vehID=ambulance_id,
                routeID=route_id,
                typeID=self.config.ambulance_type_id,
                depart=str(float(entry.spawn_time)),
                departLane=self.config.depart_lane,
                departPos=self.config.depart_position,
                departSpeed="max",
            )
        except Exception as exc:
            record.status = "insertion_failed"
            record.failure_reason = f"vehicle_add_failed:{type(exc).__name__}"
            record.end_time = sim_time
            self._decision_delta.failed.append(
                (ambulance_id, record.failure_reason)
            )
        else:
            # Appearance is cosmetic.  A SUMO build that does not allow a
            # pending vehicle's color to be changed must not turn a successful
            # insertion request into an ``insertion_failed`` record.
            set_color = getattr(self.traci.vehicle, "setColor", None)
            if callable(set_color):
                _safe_call(
                    lambda: set_color(
                        ambulance_id, (255, 50, 50, 255)
                    ),
                    None,
                )
        self._sync_legacy_active_metadata()

    def _next_reroute_delay(
        self, sequence: int, check_index: int
    ) -> float:
        # Per-ambulance deterministic jitter prevents a faster controller from
        # changing later ambulances' reroute clocks merely by completing an
        # earlier trip before it consumes another shared RNG draw.
        digest = hashlib.sha256(
            f"{self.schedule_seed}|{int(sequence)}|{int(check_index)}".encode(
                "utf-8"
            )
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        jitter = self.config.reroute_jitter_seconds * (2.0 * unit - 1.0)
        return max(
            1.0,
            self.config.reroute_interval_seconds + jitter,
        )

    def before_simulation_step(
        self,
        sim_state: dict[str, Any],
        sim_time: float,
        args: Any,
    ) -> None:
        del sim_state, args
        while (
            self._schedule_cursor < len(self.schedule)
            and self.schedule[self._schedule_cursor].spawn_time
            <= sim_time + self.config.step_length_seconds + 1e-9
        ):
            entry = self.schedule[self._schedule_cursor]
            self._schedule_cursor += 1
            self._spawn(entry, sim_time)

        for record in list(self.records.values()):
            if (
                record.status == "active"
                and sim_time + 1e-9 >= record.next_reroute_time
            ):
                self._maybe_reroute(record, sim_time)

    def _simulation_ids(self, method_name: str) -> set[str]:
        method = getattr(self.traci.simulation, method_name, None)
        if not callable(method):
            return set()
        return {
            str(value)
            for value in _safe_call(method, ())
        }

    def after_simulation_step(
        self,
        sim_state: dict[str, Any],
        sim_time: float,
        args: Any,
    ) -> None:
        del sim_state, args
        departed = self._simulation_ids("getDepartedIDList")
        arrived = self._simulation_ids("getArrivedIDList")
        teleported = self._simulation_ids("getStartingTeleportIDList")
        collided = self._simulation_ids("getCollidingVehiclesIDList")
        current_ids = {
            str(value)
            for value in _safe_call(self.traci.vehicle.getIDList, ())
        }
        self.all_collision_vehicle_ids.update(collided)
        self.all_teleported_vehicle_ids.update(teleported)
        self._decision_delta.collisions.update(collided)
        self._decision_delta.teleports.update(teleported)

        for ambulance_id, record in self.records.items():
            if record.status in TERMINAL_STATUSES:
                continue
            if ambulance_id in departed and record.status == "pending":
                record.status = "active"
                departure = _safe_call(
                    lambda aid=ambulance_id: self.traci.vehicle.getDeparture(aid),
                    sim_time,
                )
                record.actual_departure = (
                    float(departure)
                    if float(departure) >= 0.0
                    else float(sim_time)
                )
                record.last_seen_time = float(sim_time)
                self._decision_delta.departed.append(ambulance_id)

            if ambulance_id in collided:
                self._finish_failure(record, "collided", sim_time)
                continue
            if ambulance_id in teleported:
                self._finish_failure(record, "teleported", sim_time)
                continue
            if ambulance_id in arrived:
                if (
                    record.previous_first_tls is not None
                    and record.previous_first_tls not in record.cleared_tls
                ):
                    record.cleared_tls.append(record.previous_first_tls)
                    self._decision_delta.cleared_tls.append(
                        (ambulance_id, record.previous_first_tls)
                    )
                record.status = "arrived"
                record.end_time = float(sim_time)
                self._decision_delta.arrived.append(ambulance_id)
                continue

            if record.status == "pending":
                if (
                    sim_time - record.requested_departure
                    > self.config.pending_timeout_seconds
                ):
                    self._finish_failure(
                        record, "insertion_failed", sim_time
                    )
                continue

            if record.status == "active" and ambulance_id not in current_ids:
                self._finish_failure(record, "removed", sim_time)
                continue
            if record.status == "active":
                self._sample_active_record(record, sim_time)

        self._sync_legacy_active_metadata()

    def _finish_failure(
        self, record: AmbulanceRecord, status: str, sim_time: float
    ) -> None:
        record.status = status
        record.failure_reason = status
        record.end_time = float(sim_time)
        self._decision_delta.failed.append(
            (record.ambulance_id, status)
        )

    def _remaining_route_distance(self, ambulance_id: str) -> float:
        route = tuple(
            str(value)
            for value in _safe_call(
                lambda: self.traci.vehicle.getRoute(ambulance_id),
                (),
            )
        )
        route_index = int(
            _safe_call(
                lambda: self.traci.vehicle.getRouteIndex(ambulance_id),
                -1,
            )
        )
        if route_index < 0 or route_index >= len(route):
            return float("nan")
        lane_id = str(
            _safe_call(
                lambda: self.traci.vehicle.getLaneID(ambulance_id),
                "",
            )
        )
        lane_position = max(
            0.0,
            float(
                _safe_call(
                    lambda: self.traci.vehicle.getLanePosition(ambulance_id),
                    0.0,
                )
            ),
        )
        current_length = self._edge_length(route[route_index])
        if lane_id and not lane_id.startswith(":"):
            current_length = _finite_positive(
                _safe_call(lambda: self.traci.lane.getLength(lane_id), 0.0),
                current_length,
            )
        return max(0.0, current_length - lane_position) + sum(
            self._edge_length(edge) for edge in route[route_index + 1 :]
        )

    def _sample_active_record(
        self, record: AmbulanceRecord, sim_time: float
    ) -> None:
        ambulance_id = record.ambulance_id
        elapsed = (
            max(0.0, sim_time - record.last_seen_time)
            if record.last_seen_time is not None
            else 0.0
        )
        speed = max(
            0.0,
            float(
                _safe_call(
                    lambda: self.traci.vehicle.getSpeed(ambulance_id),
                    0.0,
                )
            ),
        )
        remaining = self._remaining_route_distance(ambulance_id)
        time_loss = max(
            0.0,
            float(
                _safe_call(
                    lambda: self.traci.vehicle.getTimeLoss(ambulance_id),
                    record.last_time_loss,
                )
            ),
        )
        distance_driven = max(
            record.distance_driven,
            float(
                _safe_call(
                    lambda: self.traci.vehicle.getDistance(ambulance_id),
                    record.distance_driven,
                )
            ),
        )
        next_tls = tuple(
            (
                str(item[0]),
                int(item[1]),
                max(0.0, float(item[2])),
                str(item[3]),
            )
            for item in _safe_call(
                lambda: self.traci.vehicle.getNextTLS(ambulance_id),
                (),
            )
            if len(item) >= 4
        )

        # Physical odometer progress is invariant to rerouting. A decrease in
        # remaining route length could otherwise reward the signal policy for
        # a route shortcut chosen independently by the deterministic router.
        progress = max(0.0, distance_driven - record.distance_driven)
        self._decision_delta.progress_meters[ambulance_id] = (
            self._decision_delta.progress_meters.get(ambulance_id, 0.0)
            + progress
        )
        loss_delta = max(0.0, time_loss - record.last_time_loss)
        self._decision_delta.time_loss_seconds[ambulance_id] = (
            self._decision_delta.time_loss_seconds.get(ambulance_id, 0.0)
            + loss_delta
        )
        if speed < 1.0 and elapsed > 0.0:
            record.stopped_seconds += elapsed
            self._decision_delta.stopped_seconds[ambulance_id] = (
                self._decision_delta.stopped_seconds.get(ambulance_id, 0.0)
                + elapsed
            )

        first_tls = next_tls[0][0] if next_tls else None
        if (
            record.previous_first_tls is not None
            and first_tls != record.previous_first_tls
            and record.previous_first_tls not in record.cleared_tls
        ):
            record.cleared_tls.append(record.previous_first_tls)
            self._decision_delta.cleared_tls.append(
                (ambulance_id, record.previous_first_tls)
            )
        record.previous_first_tls = first_tls
        record.next_tls = next_tls
        record.last_seen_time = float(sim_time)
        record.last_speed = speed
        record.last_remaining_distance = (
            remaining if math.isfinite(remaining) else None
        )
        record.last_time_loss = time_loss
        record.time_loss = max(record.time_loss, time_loss)
        record.distance_driven = distance_driven

    def _edge_travel_time(self, edge_id: str) -> float:
        travel_time = _safe_call(
            lambda: self.traci.edge.getTraveltime(edge_id),
            None,
        )
        if travel_time is not None and float(travel_time) > 0.0:
            return float(travel_time)
        lane_count = max(
            1,
            int(
                _safe_call(
                    lambda: self.traci.edge.getLaneNumber(edge_id),
                    1,
                )
            ),
        )
        speeds = [
            _finite_positive(
                _safe_call(
                    lambda index=index: self.traci.lane.getMaxSpeed(
                        f"{edge_id}_{index}"
                    ),
                    13.9,
                ),
                13.9,
            )
            for index in range(lane_count)
        ]
        return self._edge_length(edge_id) / max(0.1, max(speeds))

    def _current_route_eta(
        self, ambulance_id: str, route: Sequence[str], route_index: int
    ) -> float:
        if route_index < 0 or route_index >= len(route):
            return float("inf")
        eta = sum(self._edge_travel_time(edge) for edge in route[route_index:])
        current_edge_length = self._edge_length(route[route_index])
        lane_position = max(
            0.0,
            float(
                _safe_call(
                    lambda: self.traci.vehicle.getLanePosition(ambulance_id),
                    0.0,
                )
            ),
        )
        remaining_fraction = max(
            0.0,
            min(1.0, 1.0 - lane_position / max(1.0, current_edge_length)),
        )
        eta -= self._edge_travel_time(route[route_index]) * (
            1.0 - remaining_fraction
        )
        return max(0.0, eta)

    def _maybe_reroute(
        self, record: AmbulanceRecord, sim_time: float
    ) -> None:
        record.reroute_checks += 1
        record.next_reroute_time = (
            sim_time
            + self._next_reroute_delay(
                record.schedule.sequence, record.reroute_checks
            )
        )
        nearest_tls_distance = (
            float(record.next_tls[0][2])
            if record.next_tls
            else float("inf")
        )
        if (
            nearest_tls_distance
            < self.config.no_reroute_within_tls_meters
        ):
            return
        ambulance_id = record.ambulance_id
        current_edge = str(
            _safe_call(
                lambda: self.traci.vehicle.getRoadID(ambulance_id),
                "",
            )
        )
        if not current_edge or current_edge.startswith(":"):
            return
        current_route = tuple(
            str(value)
            for value in _safe_call(
                lambda: self.traci.vehicle.getRoute(ambulance_id),
                (),
            )
        )
        current_index = int(
            _safe_call(
                lambda: self.traci.vehicle.getRouteIndex(ambulance_id),
                -1,
            )
        )
        current_eta = self._current_route_eta(
            ambulance_id, current_route, current_index
        )
        path = _safe_call(
            lambda: self._find_route(
                current_edge,
                record.destination_edge,
                aggregated=True,
                depart=sim_time,
            ),
            None,
        )
        candidate = tuple(
            str(edge) for edge in getattr(path, "edges", ()) or ()
        )
        if (
            len(candidate) < 2
            or candidate[0] != current_edge
            or candidate[-1] != record.destination_edge
            or candidate == current_route[current_index:]
        ):
            return
        candidate_tls = self.route_index.route_tls(candidate)
        if len(candidate_tls) < 1:
            return
        # Compare both alternatives with the same live edge-travel-time
        # estimator.  Mixing a smoothed Stage.travelTime for the candidate
        # with last-step edge times for the current route can invent savings.
        candidate_eta = self._current_route_eta(
            ambulance_id,
            candidate,
            0,
        )
        savings = current_eta - candidate_eta
        required = max(
            self.config.reroute_min_savings_seconds,
            self.config.reroute_min_savings_fraction * current_eta,
        )
        if not math.isfinite(savings) or savings + 1e-9 < required:
            return
        try:
            self.traci.vehicle.setRoute(ambulance_id, list(candidate))
        except Exception:
            return
        record.route_edges = candidate
        record.route_tls = candidate_tls
        record.route_mode = "traffic_aware_rerouted"
        record.route_updates += 1
        record.last_reroute_time = float(sim_time)
        # A changed next-TLS identity after rerouting is not evidence that the
        # old junction was physically cleared.
        record.previous_first_tls = None
        self._decision_delta.rerouted.append(ambulance_id)

    def _sync_legacy_active_metadata(self) -> None:
        active: dict[str, dict[str, Any]] = {}
        for ambulance_id, record in self.records.items():
            if record.status not in {"pending", "active"}:
                continue
            active[ambulance_id] = {
                "origin": record.schedule.origin_edge,
                "destination": record.schedule.destination_edge,
                "route_len": len(record.route_edges),
                "route_distance": self._route_distance(record.route_edges),
                "free_flow_time": record.schedule.free_flow_time,
                "status": record.status,
                "route_mode": record.route_mode,
            }
        self.sim_state["active_ambulances"] = active

    def active_records(self) -> list[AmbulanceRecord]:
        return [
            record
            for record in self.records.values()
            if record.status == "active"
        ]

    def finish_episode(self, sim_time: float | None = None) -> tuple[str, ...]:
        end_time = (
            float(sim_time)
            if sim_time is not None
            else float(
                _safe_call(lambda: self.traci.simulation.getTime(), 0.0)
            )
        )
        newly_censored: list[str] = []
        for record in self.records.values():
            if record.status not in TERMINAL_STATUSES:
                record.status = "censored"
                record.end_time = end_time
                newly_censored.append(record.ambulance_id)
        # Any scheduled entry not requested before the measured horizon must
        # remain visible in completion statistics.
        for entry in self.schedule:
            ambulance_id = f"{self.config.ambulance_id_prefix}{entry.sequence}"
            if ambulance_id in self.records:
                continue
            self.records[ambulance_id] = AmbulanceRecord(
                ambulance_id=ambulance_id,
                schedule=entry,
                requested_departure=entry.spawn_time,
                status="censored",
                end_time=end_time,
                route_edges=entry.free_flow_edges,
                route_tls=entry.route_tls,
                route_mode="not_spawned_before_horizon",
            )
            newly_censored.append(ambulance_id)
        self._decision_delta.censored.extend(newly_censored)
        self._sync_legacy_active_metadata()
        return tuple(newly_censored)

    def summary(self) -> dict[str, Any]:
        records = list(self.records.values())
        departed = [
            record
            for record in records
            if record.actual_departure is not None
        ]
        arrived = [record for record in records if record.status == "arrived"]
        failed = [record for record in records if record.failed]
        censored = [record for record in records if record.status == "censored"]
        completed_trips = [
            (
                record,
                max(
                    0.0,
                    float(record.end_time)
                    - float(record.actual_departure),
                ),
            )
            for record in arrived
            if record.end_time is not None
            and record.actual_departure is not None
        ]
        trip_times = [trip_time for _record, trip_time in completed_trips]
        response_times = [
            max(
                0.0,
                float(record.end_time)
                - float(record.requested_departure),
            )
            for record, _trip_time in completed_trips
        ]
        departure_delays = [
            max(
                0.0,
                float(record.actual_departure)
                - float(record.requested_departure),
            )
            for record, _trip_time in completed_trips
        ]
        time_losses = [record.time_loss for record in departed]
        stopped = [record.stopped_seconds for record in departed]
        ratios = [
            response_time
            / max(1e-9, record.schedule.free_flow_time)
            for (record, _trip_time), response_time in zip(
                completed_trips, response_times
            )
        ]
        return {
            "schedule_sha256": self.schedule_hash,
            "scheduled_total": len(self.schedule),
            "spawned_total": len(records),
            "departed_total": len(departed),
            "arrived_total": len(arrived),
            "failed_total": len(failed),
            "censored_total": len(censored),
            "completion_rate": len(arrived) / max(1, len(records)),
            "departed_completion_rate": (
                len(arrived) / max(1, len(departed))
            ),
            "mean_trip_time_s": (
                sum(trip_times) / len(trip_times) if trip_times else 0.0
            ),
            "p95_trip_time_s": _percentile(trip_times, 95.0),
            "mean_response_time_s": (
                sum(response_times) / len(response_times)
                if response_times
                else 0.0
            ),
            "p95_response_time_s": _percentile(
                response_times, 95.0
            ),
            "mean_departure_delay_s": (
                sum(departure_delays) / len(departure_delays)
                if departure_delays
                else 0.0
            ),
            "mean_free_flow_ratio": (
                sum(ratios) / len(ratios) if ratios else 0.0
            ),
            "mean_time_loss_s": (
                sum(time_losses) / len(time_losses)
                if time_losses
                else 0.0
            ),
            "mean_stopped_seconds": (
                sum(stopped) / len(stopped) if stopped else 0.0
            ),
            "route_updates_total": sum(
                record.route_updates for record in records
            ),
            "collision_vehicle_ids": sorted(self.all_collision_vehicle_ids),
            "teleported_vehicle_ids": sorted(
                self.all_teleported_vehicle_ids
            ),
            "records": [
                {
                    "ambulance_id": record.ambulance_id,
                    "status": record.status,
                    "requested_departure": record.requested_departure,
                    "actual_departure": record.actual_departure,
                    "end_time": record.end_time,
                    "origin": record.schedule.origin_edge,
                    "destination": record.schedule.destination_edge,
                    "free_flow_time": record.schedule.free_flow_time,
                    "route_mode": record.route_mode,
                    "route_updates": record.route_updates,
                    "reroute_checks": record.reroute_checks,
                    "time_loss": record.time_loss,
                    "stopped_seconds": record.stopped_seconds,
                    "distance_driven": record.distance_driven,
                    "cleared_tls": list(record.cleared_tls),
                }
                for record in records
            ],
            "config": asdict(self.config),
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def is_ambulance_vehicle(vehicle_id: str) -> bool:
    return str(vehicle_id).startswith(AMBULANCE_ID_PREFIX)


__all__ = [
    "AMBULANCE_ID_PREFIX",
    "AMBULANCE_TYPE_ID",
    "AmbulanceDecisionDelta",
    "AmbulanceRecord",
    "AmbulanceScheduleEntry",
    "AmbulanceSystem",
    "AmbulanceSystemConfig",
    "RouteTLS",
    "RouteTLSIndex",
    "is_ambulance_vehicle",
    "schedule_fingerprint",
]
