#!/usr/bin/env python3
"""Normalized coordinated MaxPressure and bounded residual action selection.

This module is intentionally independent of SUMO, PyTorch, Gymnasium, and SB3.
It consumes the map-agnostic observation dictionary from ``map_agnostic_tls``
and returns action scores for ``hold`` plus the padded candidate phases.

The controller is safe by construction:

* the physical CMPP score is always present;
* a learned policy may add only a bounded residual;
* uncertainty or out-of-distribution input reduces learned authority to zero;
* actions that are materially worse under CMPP are removed by a regret guard;
* the environment's timing/conflict/spillback action mask remains authoritative.

With ``residual_authority=0`` the returned action is exactly the deterministic
CMPP action.  This invariant is covered by unit tests and is the foundation for
gradually granting a learned policy more control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from map_agnostic_tls import (
    MAX_PHASES,
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_NAMES,
)


_MOVEMENT_INDEX = {name: index for index, name in enumerate(MOVEMENT_FEATURE_NAMES)}
_PHASE_INDEX = {name: index for index, name in enumerate(PHASE_FEATURE_NAMES)}


@dataclass(frozen=True)
class CMPPConfig:
    """Weights for normalized coordinated MaxPressure-plus-penalty scoring.

    All inputs are already dimensionless and bounded by the observation
    adapter.  The defaults deliberately emphasize pressure and downstream
    storage while giving starvation enough weight to prevent indefinite delay.
    They are baseline parameters to validate across maps, not learned weights.
    """

    pressure_weight: float = 1.25
    queue_weight: float = 0.45
    wait_weight: float = 0.30
    starvation_weight: float = 0.65
    near_platoon_weight: float = 0.20
    downstream_occupancy_weight: float = 0.85
    blocked_exit_weight: float = 2.25
    switch_penalty: float = 0.08
    hold_bias: float = 0.025
    starvation_threshold: float = 0.80
    starvation_override_bonus: float = 0.45

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not 0.0 <= self.starvation_threshold <= 1.0:
            raise ValueError("starvation_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CMPPConfig":
        if value is None:
            return cls()
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown CMPP configuration keys: {unknown}")
        return cls(**{key: float(item) for key, item in value.items()})


def _as_float_array(observation: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in observation:
        raise KeyError(f"Observation is missing required key {key!r}")
    value = np.asarray(observation[key], dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Observation key {key!r} contains NaN or infinity")
    return value


def _movement_column(movements: np.ndarray, name: str) -> np.ndarray:
    index = _MOVEMENT_INDEX[name]
    if movements.ndim != 2 or movements.shape[1] <= index:
        raise ValueError(
            f"movements has shape {movements.shape}; feature {name!r} requires "
            f"at least {index + 1} columns"
        )
    return movements[:, index]


def _phase_column(phase_features: np.ndarray, name: str) -> np.ndarray:
    index = _PHASE_INDEX[name]
    if phase_features.ndim != 2 or phase_features.shape[1] <= index:
        raise ValueError(
            f"phase_features has shape {phase_features.shape}; feature {name!r} "
            f"requires at least {index + 1} columns"
        )
    return phase_features[:, index]


def normalized_cmpp_scores(
    observation: Mapping[str, Any],
    action_mask: Sequence[bool] | np.ndarray | None = None,
    config: CMPPConfig = CMPPConfig(),
) -> np.ndarray:
    """Score hold and every phase using bounded physical traffic quantities.

    ``score[0]`` is hold and ``score[k + 1]`` is candidate phase ``k``.  Padded
    or externally unsafe actions receive ``-inf``.  Phase aggregation is a
    service-strength-weighted mean, so an intersection with more encoded
    movements does not automatically receive larger logits.
    """

    movements = _as_float_array(observation, "movements")
    movement_mask = _as_float_array(observation, "movement_mask").reshape(-1)
    membership = _as_float_array(observation, "phase_membership")
    phase_features = _as_float_array(observation, "phase_features")

    if movements.shape[0] != movement_mask.shape[0]:
        raise ValueError("movement_mask length does not match movements")
    if membership.ndim != 2 or membership.shape[1] != movements.shape[0]:
        raise ValueError("phase_membership shape does not match movements")
    if phase_features.ndim != 2 or phase_features.shape[0] != membership.shape[0]:
        raise ValueError("phase_features shape does not match phase_membership")

    valid_movement = np.clip(movement_mask, 0.0, 1.0)
    pressure = _movement_column(movements, "normalized_pressure")
    queue = _movement_column(movements, "queue_density")
    wait = _movement_column(movements, "mean_wait_log")
    starvation = _movement_column(movements, "time_since_service")
    eta_near = (
        _movement_column(movements, "eta_0_5_density")
        + 0.50 * _movement_column(movements, "eta_5_15_density")
        + 0.20 * _movement_column(movements, "eta_15_30_density")
    )
    downstream = _movement_column(movements, "downstream_occupancy")
    blocked = _movement_column(movements, "blocked_exit_ratio")

    movement_score = (
        config.pressure_weight * pressure
        + config.queue_weight * queue
        + config.wait_weight * wait
        + config.starvation_weight * starvation
        + config.near_platoon_weight * eta_near
        - config.downstream_occupancy_weight * downstream
        - config.blocked_exit_weight * blocked
    )
    movement_score *= valid_movement

    service = np.clip(membership, 0.0, 1.0) * valid_movement[None, :]
    service_total = service.sum(axis=1)
    valid_phase = service_total > 1e-9
    phase_scores = np.full(membership.shape[0], -np.inf, dtype=np.float64)
    if np.any(valid_phase):
        phase_scores[valid_phase] = (
            service[valid_phase] @ movement_score
        ) / service_total[valid_phase]

    # A heavily starved served movement receives a bounded emergency bonus.
    starved = np.clip(
        (starvation - config.starvation_threshold)
        / max(1e-9, 1.0 - config.starvation_threshold),
        0.0,
        1.0,
    )
    if np.any(valid_phase):
        phase_starvation = np.zeros_like(phase_scores)
        phase_starvation[valid_phase] = np.max(
            np.where(service[valid_phase] > 0.0, starved[None, :], 0.0),
            axis=1,
        )
        phase_scores[valid_phase] += (
            config.starvation_override_bonus * phase_starvation[valid_phase]
        )

    current_flags = np.clip(_phase_column(phase_features, "is_current"), 0.0, 1.0)
    current_index: int | None = None
    if np.any(current_flags > 0.5):
        current_index = int(np.argmax(current_flags))

    for candidate in np.flatnonzero(valid_phase):
        if current_index is None or int(candidate) != current_index:
            phase_scores[candidate] -= config.switch_penalty

    scores = np.full(MAX_PHASES + 1, -np.inf, dtype=np.float64)
    candidate_count = min(MAX_PHASES, len(phase_scores))
    scores[1 : candidate_count + 1] = phase_scores[:candidate_count]
    if current_index is not None and current_index < len(phase_scores):
        scores[0] = phase_scores[current_index] + config.hold_bias
    else:
        scores[0] = 0.0

    if action_mask is not None:
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.shape != scores.shape:
            raise ValueError(
                f"action_mask has shape {mask.shape}, expected {scores.shape}"
            )
        scores[~mask] = -np.inf
    if not np.any(np.isfinite(scores)):
        raise ValueError("No finite CMPP action remains after applying the action mask")
    return scores


@dataclass(frozen=True)
class SafeResidualDecision:
    action: int
    baseline_action: int
    effective_authority: float
    used_fallback: bool
    fallback_reason: str | None
    baseline_scores: np.ndarray
    combined_scores: np.ndarray
    allowed_actions: np.ndarray


class SafeResidualController:
    """Combine CMPP with a bounded learned score correction."""

    def __init__(
        self,
        config: CMPPConfig = CMPPConfig(),
        residual_authority: float = 0.0,
        residual_bound: float = 1.0,
        max_baseline_regret: float = 0.20,
        uncertainty_threshold: float = 0.35,
    ):
        self.config = config
        self.residual_authority = float(residual_authority)
        self.residual_bound = float(residual_bound)
        self.max_baseline_regret = float(max_baseline_regret)
        self.uncertainty_threshold = float(uncertainty_threshold)
        if not 0.0 <= self.residual_authority <= 1.0:
            raise ValueError("residual_authority must be in [0, 1]")
        if self.residual_bound < 0.0:
            raise ValueError("residual_bound must be non-negative")
        if self.max_baseline_regret < 0.0:
            raise ValueError("max_baseline_regret must be non-negative")
        if self.uncertainty_threshold <= 0.0:
            raise ValueError("uncertainty_threshold must be positive")

    def select_action(
        self,
        observation: Mapping[str, Any],
        action_mask: Sequence[bool] | np.ndarray,
        residual_scores: Sequence[float] | np.ndarray | None = None,
        *,
        uncertainty: float = 0.0,
        in_distribution: bool = True,
    ) -> SafeResidualDecision:
        baseline = normalized_cmpp_scores(observation, action_mask, self.config)
        baseline_action = int(np.argmax(baseline))
        valid = np.isfinite(baseline)
        best_baseline = float(baseline[baseline_action])
        allowed = valid & (baseline >= best_baseline - self.max_baseline_regret)

        fallback_reason: str | None = None
        uncertainty = max(0.0, float(uncertainty))
        if not in_distribution:
            effective_authority = 0.0
            fallback_reason = "out_of_distribution"
        elif uncertainty >= self.uncertainty_threshold:
            effective_authority = 0.0
            fallback_reason = "high_uncertainty"
        else:
            attenuation = 1.0 - uncertainty / self.uncertainty_threshold
            effective_authority = self.residual_authority * max(0.0, attenuation)

        if residual_scores is None:
            residual = np.zeros_like(baseline)
        else:
            residual = np.asarray(residual_scores, dtype=np.float64).reshape(-1)
            if residual.shape != baseline.shape:
                raise ValueError(
                    f"residual_scores has shape {residual.shape}, expected {baseline.shape}"
                )
            if not np.all(np.isfinite(residual)):
                fallback_reason = fallback_reason or "non_finite_residual"
                effective_authority = 0.0
                residual = np.zeros_like(baseline)

        bounded_residual = np.tanh(residual) * self.residual_bound
        combined = baseline + effective_authority * bounded_residual
        combined[~allowed] = -np.inf
        action = int(np.argmax(combined))

        # This is redundant with the regret mask, but it makes the invariant
        # explicit and protects future score-combination changes.
        if not allowed[action]:
            action = baseline_action
            fallback_reason = fallback_reason or "baseline_regret_guard"
        used_fallback = bool(fallback_reason is not None or effective_authority == 0.0)
        return SafeResidualDecision(
            action=action,
            baseline_action=baseline_action,
            effective_authority=float(effective_authority),
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            baseline_scores=baseline.copy(),
            combined_scores=combined.copy(),
            allowed_actions=allowed.copy(),
        )


__all__ = [
    "CMPPConfig",
    "SafeResidualController",
    "SafeResidualDecision",
    "normalized_cmpp_scores",
]
