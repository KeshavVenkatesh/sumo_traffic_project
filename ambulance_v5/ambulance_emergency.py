#!/usr/bin/env python3
"""Ambulance-aware observations, green-corridor logic, and override network.

The normal schema-v3 traffic checkpoint remains frozen and receives its
original observation dictionary.  This module builds a separate emergency
side-channel and a small permutation-equivariant residual network.  With no
relevant ambulance, the final action is forced to the frozen base action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ambulance_system import AmbulanceDecisionDelta, AmbulanceSystem
from map_agnostic_tls import (
    GLOBAL_FEATURE_DIM,
    MAX_MOVEMENTS,
    MAX_PHASES,
    MOVEMENT_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    clamp01,
)


EMERGENCY_MOVEMENT_FEATURE_NAMES = (
    "ambulance_count",
    "distance_closeness",
    "eta_closeness",
    "is_next_signal",
    "route_order_priority",
    "recent_time_loss",
    "recently_stopped",
    "protected_service_available",
)
EMERGENCY_MOVEMENT_FEATURE_DIM = len(EMERGENCY_MOVEMENT_FEATURE_NAMES)

EMERGENCY_PHASE_FEATURE_NAMES = (
    "serves_ambulance",
    "protected_service",
    "eta_closeness",
    "ambulance_load",
    "is_current",
    "downstream_space",
    "recovery_active",
)
EMERGENCY_PHASE_FEATURE_DIM = len(EMERGENCY_PHASE_FEATURE_NAMES)

EMERGENCY_GLOBAL_FEATURE_NAMES = (
    "has_relevant_ambulance",
    "nearest_eta_closeness",
    "active_ambulance_fraction",
    "next_signal_ambulance_fraction",
    "recent_time_loss",
    "recent_stopped_fraction",
    "recovery_active",
    "preemption_budget_remaining",
)
EMERGENCY_GLOBAL_FEATURE_DIM = len(EMERGENCY_GLOBAL_FEATURE_NAMES)
MIN_EMERGENCY_DOWNSTREAM_SPACE = 0.08
CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS = 8.0


@dataclass(frozen=True)
class EmergencyObservationConfig:
    relevance_distance_meters: float = 650.0
    relevance_eta_seconds: float = 60.0
    eta_floor_speed_mps: float = 5.0
    max_active_reference: float = 3.0
    route_horizon: int = 3


@dataclass
class CorridorTLSState:
    mode: str = "normal"
    tracked_ambulances: set[str] = field(default_factory=set)
    preemption_started: float | None = None
    recovery_until: float = float("-inf")
    last_update_time: float | None = None
    preemption_seconds: float = 0.0


@dataclass(frozen=True)
class EmergencyTLSContext:
    tls_id: str
    observation: dict[str, np.ndarray]
    relevant_ambulances: tuple[str, ...]
    ambulance_priorities: Mapping[str, float]
    target_actions: tuple[int, ...]
    protected_target_actions: tuple[int, ...]
    nearest_eta_seconds: float
    next_signal_present: bool
    recovery_active: bool
    corridor_mode: str = "normal"
    authority_available: bool = False

    @property
    def relevant(self) -> bool:
        return bool(self.relevant_ambulances)

    @property
    def active_for_training(self) -> bool:
        return self.recovery_active or self.corridor_mode in {
            "prepare",
            "serve",
        }

    @property
    def override_allowed(self) -> bool:
        """Hard authority budget with a close-ambulance safety exception."""

        if self.recovery_active:
            return True
        if (
            not self.relevant
            or self.corridor_mode not in {"prepare", "serve"}
        ):
            return False
        return (
            self.authority_available
            or self.nearest_eta_seconds
            <= CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS
        )


def empty_emergency_observation() -> dict[str, np.ndarray]:
    return {
        "emergency_movements": np.zeros(
            (MAX_MOVEMENTS, EMERGENCY_MOVEMENT_FEATURE_DIM),
            dtype=np.float32,
        ),
        "emergency_phase_features": np.zeros(
            (MAX_PHASES, EMERGENCY_PHASE_FEATURE_DIM),
            dtype=np.float32,
        ),
        "emergency_global_features": np.zeros(
            (EMERGENCY_GLOBAL_FEATURE_DIM,),
            dtype=np.float32,
        ),
    }


def stack_emergency_observations(
    observations: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("Cannot stack empty emergency observations")
    return {
        key: np.stack(
            [np.asarray(observation[key]) for observation in observations],
            axis=0,
        )
        for key in observations[0]
    }


class RollingGreenCorridor:
    """Maintain prepare/serve/recovery state for every traffic light."""

    def __init__(
        self,
        recovery_seconds: float = 30.0,
        max_preemption_seconds: float = 45.0,
        clearance_buffer_seconds: float = 3.0,
        prepare_eta_seconds: float = 25.0,
        serve_eta_seconds: float = 12.0,
    ):
        self.recovery_seconds = max(0.0, float(recovery_seconds))
        self.max_preemption_seconds = max(
            1.0, float(max_preemption_seconds)
        )
        self.clearance_buffer_seconds = max(
            0.0, float(clearance_buffer_seconds)
        )
        self.prepare_eta_seconds = max(
            1.0, float(prepare_eta_seconds)
        )
        self.serve_eta_seconds = max(0.0, float(serve_eta_seconds))
        if self.serve_eta_seconds > self.prepare_eta_seconds:
            raise ValueError(
                "serve_eta_seconds cannot exceed prepare_eta_seconds"
            )
        self.states: dict[str, CorridorTLSState] = {}

    def state(self, tls_id: str) -> CorridorTLSState:
        return self.states.setdefault(str(tls_id), CorridorTLSState())

    def update_cleared(
        self, delta: AmbulanceDecisionDelta, sim_time: float
    ) -> None:
        for ambulance_id, tls_id in delta.cleared_tls:
            state = self.state(tls_id)
            state.tracked_ambulances.discard(ambulance_id)
            state.mode = "recovery"
            state.recovery_until = max(
                state.recovery_until,
                float(sim_time)
                + self.clearance_buffer_seconds
                + self.recovery_seconds,
            )

    def observe_context(
        self, context: EmergencyTLSContext, sim_time: float
    ) -> CorridorTLSState:
        state = self.state(context.tls_id)
        now = float(sim_time)
        if state.last_update_time is not None and state.mode in {
            "prepare",
            "serve",
        }:
            state.preemption_seconds += max(
                0.0, now - state.last_update_time
            )
        state.last_update_time = now

        in_preemption_window = (
            context.relevant
            and context.nearest_eta_seconds
            <= self.prepare_eta_seconds
        )
        if in_preemption_window:
            state.tracked_ambulances.update(context.relevant_ambulances)
            state.mode = (
                "serve"
                if context.next_signal_present
                and context.nearest_eta_seconds
                <= self.serve_eta_seconds
                else "prepare"
            )
            if state.preemption_started is None:
                state.preemption_started = now
        elif state.mode in {"prepare", "serve"}:
            state.mode = "recovery"
            state.recovery_until = max(
                state.recovery_until,
                now + self.recovery_seconds,
            )
            state.tracked_ambulances.clear()
        elif state.mode == "recovery" and now >= state.recovery_until:
            state.mode = "normal"
            state.preemption_started = None
            state.preemption_seconds = 0.0
        return state

    def recovery_active(self, tls_id: str, sim_time: float) -> bool:
        state = self.state(tls_id)
        return (
            state.mode == "recovery"
            and float(sim_time) < state.recovery_until
        )

    def budget_remaining(self, tls_id: str) -> float:
        state = self.state(tls_id)
        return clamp01(
            1.0
            - state.preemption_seconds
            / max(1.0, self.max_preemption_seconds)
        )

    def can_afford_preemption(
        self, tls_id: str, decision_seconds: float
    ) -> bool:
        """Return whether another full fixed-cadence action fits the budget."""

        state = self.state(tls_id)
        remaining = (
            self.max_preemption_seconds - state.preemption_seconds
        )
        return remaining + 1e-9 >= max(0.0, float(decision_seconds))

    @staticmethod
    def _legal_fallback(
        base_action: int, action_mask: np.ndarray
    ) -> int:
        base_action = int(base_action)
        if (
            0 <= base_action < len(action_mask)
            and bool(action_mask[base_action])
        ):
            return base_action
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("Emergency controller received an empty action mask")
        return int(legal[0])

    def teacher_action(
        self,
        context: EmergencyTLSContext,
        base_action: int,
        action_mask: np.ndarray,
        sim_time: float,
    ) -> int:
        """Deterministic safe preemption baseline used for ablation/imitation."""

        action_mask = np.asarray(action_mask, dtype=bool)
        state = self.observe_context(context, sim_time)
        fallback = self._legal_fallback(base_action, action_mask)
        if state.mode not in {"prepare", "serve"}:
            return fallback

        # The synchronized context reserves a complete fixed-cadence action
        # inside the disturbance budget. Continued service after that is only
        # available for an ambulance already close to the stop line.
        if not context.override_allowed:
            return fallback

        target_actions = tuple(
            dict.fromkeys(
                context.protected_target_actions
                + context.target_actions
            )
        )
        phase_features = context.observation[
            "emergency_phase_features"
        ]
        candidates: list[tuple[float, int]] = []
        for action in target_actions:
            if not (
                0 < action < len(action_mask)
                and action_mask[action]
            ):
                continue
            features = phase_features[action - 1]
            # Prefer protected service, then the phase serving the closest and
            # largest ambulance demand.  Current-phase and downstream-space
            # terms break close ties without relying on arbitrary action IDs.
            score = (
                4.0 * float(features[1])
                + 2.0 * float(features[2])
                + 1.5 * float(features[3])
                + 0.35 * float(features[4])
                + 0.50 * float(features[5])
                + 0.25 * float(features[0])
            )
            candidates.append((score, int(action)))
        if candidates:
            return max(candidates, key=lambda item: (item[0], -item[1]))[1]

        # If minimum-green/clearance timing temporarily prevents the required
        # phase, safely hold when possible and try again next decision.
        if action_mask[0]:
            return 0
        return fallback


class EmergencyFeatureBuilder:
    def __init__(
        self,
        system: AmbulanceSystem,
        adapters: Sequence[Any],
        corridor: RollingGreenCorridor,
        config: EmergencyObservationConfig = EmergencyObservationConfig(),
    ):
        self.system = system
        self.adapters = list(adapters)
        self.corridor = corridor
        self.config = config
        self.adapter_by_tls = {
            str(adapter.tls_id): adapter for adapter in self.adapters
        }

    def _speed_eta(self, distance: float, speed: float) -> float:
        return max(0.0, float(distance)) / max(
            self.config.eta_floor_speed_mps,
            float(speed),
        )

    def build(self, sim_time: float) -> list[EmergencyTLSContext]:
        contexts: dict[str, dict[str, Any]] = {}
        for adapter in self.adapters:
            tls_id = str(adapter.tls_id)
            recovery = self.corridor.recovery_active(tls_id, sim_time)
            contexts[tls_id] = {
                "adapter": adapter,
                "observation": empty_emergency_observation(),
                "relevant": set(),
                "priorities": {},
                "target_actions": set(),
                "protected_actions": set(),
                "nearest_eta": float("inf"),
                "next_present": False,
                "recovery": recovery,
            }

        delta = self.system.last_decision_delta
        for record in self.system.active_records():
            ambulance_id = record.ambulance_id
            recent_loss = clamp01(
                delta.time_loss_seconds.get(ambulance_id, 0.0) / 10.0
            )
            recent_stop = clamp01(
                delta.stopped_seconds.get(ambulance_id, 0.0) / 10.0
            )
            for order, item in enumerate(
                record.next_tls[: self.config.route_horizon]
            ):
                tls_id, link_index, distance, _state = item
                context = contexts.get(str(tls_id))
                if context is None:
                    continue
                eta = self._speed_eta(distance, record.last_speed)
                if (
                    distance > self.config.relevance_distance_meters
                    or eta > self.config.relevance_eta_seconds
                ):
                    continue
                adapter = context["adapter"]
                movement_index = self.system.route_index.movement_for_link(
                    tls_id, link_index
                )
                if movement_index is None or movement_index >= len(
                    adapter.topology.movements
                ):
                    continue

                priority = 1.0 / float(order + 1)
                context["relevant"].add(ambulance_id)
                context["priorities"][ambulance_id] = max(
                    priority,
                    context["priorities"].get(ambulance_id, 0.0),
                )
                context["nearest_eta"] = min(
                    context["nearest_eta"], eta
                )
                context["next_present"] = (
                    context["next_present"] or order == 0
                )

                movement = context["observation"][
                    "emergency_movements"
                ][movement_index]
                movement[0] = min(
                    1.0,
                    movement[0]
                    + 1.0 / self.config.max_active_reference,
                )
                movement[1] = max(
                    movement[1],
                    clamp01(
                        1.0
                        - distance
                        / self.config.relevance_distance_meters
                    ),
                )
                movement[2] = max(
                    movement[2],
                    clamp01(
                        1.0
                        - eta / self.config.relevance_eta_seconds
                    ),
                )
                movement[3] = max(movement[3], float(order == 0))
                movement[4] = max(movement[4], priority)
                movement[5] = max(movement[5], recent_loss)
                movement[6] = max(movement[6], recent_stop)

                for candidate, members in enumerate(
                    adapter.topology.phase_members
                ):
                    if movement_index not in members:
                        continue
                    member_position = members.index(movement_index)
                    strength = float(
                        adapter.topology.phase_weights[candidate][
                            member_position
                        ]
                    )
                    action = candidate + 1
                    context["target_actions"].add(action)
                    if strength >= 0.99:
                        context["protected_actions"].add(action)
                        movement[7] = 1.0

        output: list[EmergencyTLSContext] = []
        for adapter in self.adapters:
            tls_id = str(adapter.tls_id)
            item = contexts[tls_id]
            observation = item["observation"]
            base_observation = (
                adapter.last_snapshot.observation
                if adapter.last_snapshot is not None
                else None
            )
            current_candidate = None
            if base_observation is not None:
                current_flags = base_observation["phase_features"][:, 0]
                current_indices = np.flatnonzero(current_flags > 0.5)
                if len(current_indices):
                    current_candidate = int(current_indices[0])

            emergency_movements = observation["emergency_movements"]
            for candidate, members in enumerate(
                adapter.topology.phase_members
            ):
                member_values = emergency_movements[list(members)]
                if len(member_values) == 0:
                    continue
                weights = np.asarray(
                    adapter.topology.phase_weights[candidate],
                    dtype=np.float32,
                )
                demand = member_values[:, 0]
                relevant = demand > 0.0
                if relevant.any():
                    protected = float(
                        np.max(
                            weights[relevant]
                            * member_values[relevant, 7]
                        )
                    )
                    serves = float(np.max(member_values[relevant, 4]))
                    eta_closeness = float(
                        np.max(member_values[relevant, 2])
                    )
                    load = clamp01(float(np.sum(demand[relevant])))
                else:
                    protected = 0.0
                    serves = 0.0
                    eta_closeness = 0.0
                    load = 0.0
                downstream_space = (
                    float(base_observation["phase_features"][candidate, 4])
                    if base_observation is not None
                    else 1.0
                )
                observation["emergency_phase_features"][
                    candidate
                ] = np.asarray(
                    [
                        serves,
                        protected,
                        eta_closeness,
                        load,
                        float(candidate == current_candidate),
                        clamp01(downstream_space),
                        float(item["recovery"]),
                    ],
                    dtype=np.float32,
                )

            relevant_ids = tuple(sorted(item["relevant"]))
            nearest_eta = float(item["nearest_eta"])
            active_records = self.system.active_records()
            total_recent_loss = sum(
                delta.time_loss_seconds.get(record.ambulance_id, 0.0)
                for record in active_records
            )
            total_recent_stopped = sum(
                delta.stopped_seconds.get(record.ambulance_id, 0.0)
                for record in active_records
            )
            observation["emergency_global_features"] = np.asarray(
                [
                    float(bool(relevant_ids)),
                    (
                        clamp01(
                            1.0
                            - nearest_eta
                            / self.config.relevance_eta_seconds
                        )
                        if math.isfinite(nearest_eta)
                        else 0.0
                    ),
                    clamp01(
                        len(active_records)
                        / self.config.max_active_reference
                    ),
                    clamp01(
                        len(relevant_ids)
                        / self.config.max_active_reference
                    ),
                    clamp01(total_recent_loss / 10.0),
                    clamp01(total_recent_stopped / 10.0),
                    float(item["recovery"]),
                    self.corridor.budget_remaining(tls_id),
                ],
                dtype=np.float32,
            )
            output.append(
                EmergencyTLSContext(
                    tls_id=tls_id,
                    observation=observation,
                    relevant_ambulances=relevant_ids,
                    ambulance_priorities=dict(item["priorities"]),
                    target_actions=tuple(sorted(item["target_actions"])),
                    protected_target_actions=tuple(
                        sorted(item["protected_actions"])
                    ),
                    nearest_eta_seconds=nearest_eta,
                    next_signal_present=bool(item["next_present"]),
                    recovery_active=bool(item["recovery"]),
                    corridor_mode=self.corridor.state(tls_id).mode,
                    authority_available=False,
                )
            )
        return output


class EmergencyGraphBlock(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)

    def forward(
        self,
        values: torch.Tensor,
        adjacency: torch.Tensor,
        movement_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = movement_mask > 0.5
        pair_mask = (
            (adjacency > 0.5)
            & valid.unsqueeze(1)
            & valid.unsqueeze(2)
        )
        eye = torch.eye(
            values.shape[1],
            dtype=torch.bool,
            device=values.device,
        ).unsqueeze(0)
        pair_mask = pair_mask | (eye & ~valid.unsqueeze(2))
        scores = torch.matmul(
            self.query(values),
            self.key(values).transpose(-1, -2),
        ) / math.sqrt(float(values.shape[-1]))
        scores = scores.masked_fill(~pair_mask, -1e8)
        messages = torch.matmul(
            torch.softmax(scores, dim=-1),
            self.value(values),
        )
        return self.norm(values + self.output(messages)) * valid.unsqueeze(
            -1
        ).to(values.dtype)


class EmergencyOverrideNetwork(nn.Module):
    """Small trainable residual around a frozen schema-v3 base policy."""

    def __init__(
        self,
        embed_dim: int = 96,
        graph_layers: int = 1,
        residual_bound: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.graph_layers_count = int(graph_layers)
        self.residual_bound = float(residual_bound)
        movement_input = (
            MOVEMENT_FEATURE_DIM + EMERGENCY_MOVEMENT_FEATURE_DIM
        )
        self.movement_encoder = nn.Sequential(
            nn.Linear(movement_input, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
        )
        self.graph_blocks = nn.ModuleList(
            [
                EmergencyGraphBlock(self.embed_dim)
                for _ in range(self.graph_layers_count)
            ]
        )
        phase_input = (
            self.embed_dim
            + PHASE_FEATURE_DIM
            + EMERGENCY_PHASE_FEATURE_DIM
            + 1
        )
        global_input = GLOBAL_FEATURE_DIM + EMERGENCY_GLOBAL_FEATURE_DIM
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
        )
        self.phase_scorer = nn.Sequential(
            nn.Linear(phase_input + self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.hold_scorer = nn.Sequential(
            nn.Linear(2 * self.embed_dim + 1, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )
        nn.init.zeros_(self.phase_scorer[-1].weight)
        nn.init.zeros_(self.phase_scorer[-1].bias)
        nn.init.zeros_(self.hold_scorer[-1].weight)
        nn.init.zeros_(self.hold_scorer[-1].bias)

    def forward(
        self,
        base_observation: Mapping[str, torch.Tensor],
        emergency_observation: Mapping[str, torch.Tensor],
        base_logits: torch.Tensor,
        authority: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        movements = base_observation["movements"].float()
        movement_mask = base_observation["movement_mask"].float()
        adjacency = base_observation["movement_adjacency"].float()
        membership = base_observation["phase_membership"].float()
        phase_features = base_observation["phase_features"].float()
        global_features = base_observation["global_features"].float()
        emergency_movements = emergency_observation[
            "emergency_movements"
        ].float()
        emergency_phases = emergency_observation[
            "emergency_phase_features"
        ].float()
        emergency_global = emergency_observation[
            "emergency_global_features"
        ].float()

        active_movements = max(
            1,
            int(
                movement_mask.sum(dim=1).max().detach().cpu().item()
            ),
        )
        movements = movements[:, :active_movements]
        emergency_movements = emergency_movements[:, :active_movements]
        movement_mask = movement_mask[:, :active_movements]
        adjacency = adjacency[
            :, :active_movements, :active_movements
        ]
        membership = membership[:, :, :active_movements]
        encoded = self.movement_encoder(
            torch.cat([movements, emergency_movements], dim=-1)
        )
        for block in self.graph_blocks:
            encoded = block(encoded, adjacency, movement_mask)

        valid_membership = membership * movement_mask.unsqueeze(1)
        denominator = valid_membership.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        phase_embedding = torch.bmm(
            valid_membership, encoded
        ) / denominator
        global_embedding = self.global_encoder(
            torch.cat([global_features, emergency_global], dim=-1)
        )

        base_action = base_logits.argmax(dim=-1)
        base_phase_flag = torch.zeros(
            (
                base_logits.shape[0],
                MAX_PHASES,
                1,
            ),
            dtype=base_logits.dtype,
            device=base_logits.device,
        )
        phase_base = base_action - 1
        valid_phase_base = (phase_base >= 0) & (phase_base < MAX_PHASES)
        if valid_phase_base.any():
            rows = torch.arange(
                base_logits.shape[0], device=base_logits.device
            )[valid_phase_base]
            base_phase_flag[
                rows, phase_base[valid_phase_base], 0
            ] = 1.0

        expanded_global = global_embedding.unsqueeze(1).expand(
            -1, MAX_PHASES, -1
        )
        phase_residual = self.phase_scorer(
            torch.cat(
                [
                    phase_embedding,
                    phase_features,
                    emergency_phases,
                    base_phase_flag,
                    expanded_global,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        current_weights = phase_features[..., 0:1].clamp(0.0, 1.0)
        current_denominator = current_weights.sum(dim=1).clamp_min(1.0)
        current_embedding = (
            phase_embedding * current_weights
        ).sum(dim=1) / current_denominator
        base_hold_flag = (base_action == 0).to(
            base_logits.dtype
        ).unsqueeze(-1)
        hold_residual = self.hold_scorer(
            torch.cat(
                [current_embedding, global_embedding, base_hold_flag],
                dim=-1,
            )
        )
        residual = torch.cat(
            [hold_residual, phase_residual], dim=-1
        )
        bounded = torch.tanh(residual) * self.residual_bound
        combined = base_logits + float(authority) * bounded

        pooled = (
            encoded * movement_mask.unsqueeze(-1)
        ).sum(dim=1) / movement_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        value = self.value_head(
            torch.cat([pooled, global_embedding], dim=-1)
        )
        return combined, value, bounded


@dataclass(frozen=True)
class EmergencyRewardWeights:
    progress: float = 1.25
    time_loss: float = 1.75
    stopped: float = 1.25
    tls_cleared: float = 0.35
    arrived: float = 0.75
    failure: float = 4.0
    censored: float = 4.0
    ordinary_traffic: float = 0.30
    ordinary_damage: float = 0.65
    override: float = 0.025
    collision: float = 6.0
    teleport: float = 5.0


def emergency_rewards(
    system: AmbulanceSystem,
    delta: AmbulanceDecisionDelta,
    contexts: Sequence[EmergencyTLSContext],
    traffic_rewards: np.ndarray,
    decision_seconds: float,
    base_actions: np.ndarray,
    final_actions: np.ndarray,
    weights: EmergencyRewardWeights = EmergencyRewardWeights(),
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Allocate dense ambulance credit to only the affected corridor TLS."""

    count = len(contexts)
    rewards = np.zeros(count, dtype=np.float32)
    components = [dict() for _ in contexts]
    by_tls = {
        context.tls_id: index for index, context in enumerate(contexts)
    }
    ambulance_contexts: dict[str, list[tuple[int, float]]] = {}
    for index, context in enumerate(contexts):
        for ambulance_id, priority in context.ambulance_priorities.items():
            ambulance_contexts.setdefault(ambulance_id, []).append(
                (index, float(priority))
            )

    def add(index: int, name: str, value: float) -> None:
        rewards[index] += float(value)
        components[index][name] = (
            components[index].get(name, 0.0) + float(value)
        )

    for ambulance_id, allocations in ambulance_contexts.items():
        total_priority = max(
            1e-9, sum(priority for _index, priority in allocations)
        )
        progress = clamp01(
            delta.progress_meters.get(ambulance_id, 0.0)
            / max(1.0, 20.0 * float(decision_seconds))
        )
        time_loss = clamp01(
            delta.time_loss_seconds.get(ambulance_id, 0.0)
            / max(1.0, float(decision_seconds))
        )
        stopped = clamp01(
            delta.stopped_seconds.get(ambulance_id, 0.0)
            / max(1.0, float(decision_seconds))
        )
        for index, priority in allocations:
            share = priority / total_priority
            add(index, "ambulance_progress", weights.progress * progress * share)
            add(index, "ambulance_time_loss", -weights.time_loss * time_loss * share)
            add(index, "ambulance_stopped", -weights.stopped * stopped * share)

    for ambulance_id, tls_id in delta.cleared_tls:
        index = by_tls.get(tls_id)
        if index is not None:
            add(index, "tls_cleared", weights.tls_cleared)

    for ambulance_id in delta.arrived:
        record = system.records.get(ambulance_id)
        if record is None:
            continue
        tls_ids = list(
            dict.fromkeys(route_tls.tls_id for route_tls in record.route_tls)
        )
        if not tls_ids:
            continue
        travel_time = (
            max(
                0.0,
                float(record.end_time)
                - float(record.requested_departure),
            )
            if record.end_time is not None
            else record.schedule.free_flow_time
        )
        efficiency = clamp01(
            record.schedule.free_flow_time / max(
                record.schedule.free_flow_time, travel_time, 1.0
            )
        )
        for tls_id in tls_ids:
            index = by_tls.get(tls_id)
            if index is not None:
                add(
                    index,
                    "ambulance_arrived",
                    weights.arrived * efficiency / len(tls_ids),
                )

    for ambulance_id, _reason in delta.failed:
        record = system.records.get(ambulance_id)
        tls_ids = (
            list(
                dict.fromkeys(
                    route_tls.tls_id for route_tls in record.route_tls
                )
            )
            if record is not None
            else []
        )
        targets = [
            by_tls[tls_id]
            for tls_id in tls_ids
            if tls_id in by_tls
        ] or [
            index
            for index, context in enumerate(contexts)
            if context.active_for_training
        ]
        for index in targets:
            add(
                index,
                "ambulance_failure",
                -weights.failure / max(1, len(targets)),
            )

    for ambulance_id in delta.censored:
        record = system.records.get(ambulance_id)
        tls_ids = (
            list(
                dict.fromkeys(
                    route_tls.tls_id for route_tls in record.route_tls
                )
            )
            if record is not None
            else []
        )
        targets = [
            by_tls[tls_id]
            for tls_id in tls_ids
            if tls_id in by_tls
        ] or [
            index
            for index, context in enumerate(contexts)
            if context.active_for_training
        ]
        for index in targets:
            add(
                index,
                "ambulance_censored",
                -weights.censored / max(1, len(targets)),
            )

    safety_targets = [
        index
        for index, context in enumerate(contexts)
        if context.active_for_training
    ]
    if delta.collisions:
        for index in safety_targets:
            add(
                index,
                "collision",
                -weights.collision / max(1, len(safety_targets)),
            )
    if delta.teleports:
        for index in safety_targets:
            add(
                index,
                "teleport",
                -weights.teleport / max(1, len(safety_targets)),
            )

    for index, context in enumerate(contexts):
        if not context.active_for_training:
            continue
        traffic = float(traffic_rewards[index])
        add(index, "ordinary_traffic", weights.ordinary_traffic * traffic)
        add(
            index,
            "ordinary_damage",
            -weights.ordinary_damage * max(0.0, -traffic),
        )
        if int(final_actions[index]) != int(base_actions[index]):
            add(index, "override", -weights.override)

    return rewards, components


__all__ = [
    "EMERGENCY_GLOBAL_FEATURE_DIM",
    "EMERGENCY_GLOBAL_FEATURE_NAMES",
    "CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS",
    "EMERGENCY_MOVEMENT_FEATURE_DIM",
    "EMERGENCY_MOVEMENT_FEATURE_NAMES",
    "EMERGENCY_PHASE_FEATURE_DIM",
    "EMERGENCY_PHASE_FEATURE_NAMES",
    "MIN_EMERGENCY_DOWNSTREAM_SPACE",
    "EmergencyFeatureBuilder",
    "EmergencyObservationConfig",
    "EmergencyOverrideNetwork",
    "EmergencyRewardWeights",
    "EmergencyTLSContext",
    "RollingGreenCorridor",
    "emergency_rewards",
    "empty_emergency_observation",
    "stack_emergency_observations",
]
