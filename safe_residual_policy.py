#!/usr/bin/env python3
"""MaskablePPO policy that learns only a bounded residual over CMPP.

The neural graph encoder is reused from ``map_agnostic_policy``.  Its actor
heads are initialized to exactly zero and interpreted as residual scores.  The
physical CMPP logits are calculated directly from the observation and are
always added back before sampling or evaluating an action.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch as th
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from map_agnostic_policy import MapAgnosticMaskablePolicy
from map_agnostic_tls import (
    GLOBAL_FEATURE_DIM,
    MOVEMENT_FEATURE_DIM,
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_DIM,
    PHASE_FEATURE_NAMES,
)
from safe_residual_controller import CMPPConfig


_MOVEMENT_INDEX = {name: index for index, name in enumerate(MOVEMENT_FEATURE_NAMES)}
_PHASE_INDEX = {name: index for index, name in enumerate(PHASE_FEATURE_NAMES)}


def normalized_cmpp_scores_tensor(
    obs: Mapping[str, th.Tensor],
    config: CMPPConfig,
) -> th.Tensor:
    """Batched differentiable-form CMPP scores (the baseline has no weights)."""

    movements = obs["movements"].float()
    movement_mask = obs["movement_mask"].float().clamp(0.0, 1.0)
    membership = obs["phase_membership"].float().clamp(0.0, 1.0)
    phase_features = obs["phase_features"].float()

    def movement(name: str) -> th.Tensor:
        return movements[..., _MOVEMENT_INDEX[name]]

    pressure = movement("normalized_pressure")
    queue = movement("queue_density")
    wait = movement("mean_wait_log")
    starvation = movement("time_since_service")
    eta_near = (
        movement("eta_0_5_density")
        + 0.50 * movement("eta_5_15_density")
        + 0.20 * movement("eta_15_30_density")
    )
    downstream = movement("downstream_occupancy")
    blocked = movement("blocked_exit_ratio")

    movement_scores = (
        config.pressure_weight * pressure
        + config.queue_weight * queue
        + config.wait_weight * wait
        + config.starvation_weight * starvation
        + config.near_platoon_weight * eta_near
        - config.downstream_occupancy_weight * downstream
        - config.blocked_exit_weight * blocked
    ) * movement_mask

    service = membership * movement_mask.unsqueeze(1)
    service_total = service.sum(dim=-1)
    valid_phase = service_total > 1e-9
    phase_scores = (service * movement_scores.unsqueeze(1)).sum(dim=-1)
    phase_scores = phase_scores / service_total.clamp_min(1e-9)

    starved = (
        (starvation - config.starvation_threshold)
        / max(1e-9, 1.0 - config.starvation_threshold)
    ).clamp(0.0, 1.0)
    phase_starvation = th.where(
        service > 0.0,
        starved.unsqueeze(1),
        th.zeros_like(service),
    ).max(dim=-1).values
    phase_scores = phase_scores + config.starvation_override_bonus * phase_starvation

    current = phase_features[..., _PHASE_INDEX["is_current"]].clamp(0.0, 1.0)
    phase_scores = phase_scores - config.switch_penalty * (1.0 - current)
    phase_scores = phase_scores.masked_fill(~valid_phase, -1e8)

    current_exists = current.sum(dim=-1, keepdim=True) > 0.5
    hold = (phase_scores * current).sum(dim=-1, keepdim=True) + config.hold_bias
    hold = th.where(current_exists, hold, th.zeros_like(hold))
    return th.cat([hold, phase_scores], dim=-1)


class PermutationEquivariantResidualAdapter(nn.Module):
    """Tiny map-specific scorer with no movement/phase positional weights."""

    def __init__(self, hidden_dim: int = 24):
        super().__init__()
        phase_input = MOVEMENT_FEATURE_DIM + PHASE_FEATURE_DIM + GLOBAL_FEATURE_DIM
        hold_input = MOVEMENT_FEATURE_DIM + GLOBAL_FEATURE_DIM
        self.phase_scorer = nn.Sequential(
            nn.Linear(phase_input, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.hold_scorer = nn.Sequential(
            nn.Linear(hold_input, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Adding a new adapter cannot change behavior until it is trained.
        nn.init.zeros_(self.phase_scorer[-1].weight)
        nn.init.zeros_(self.phase_scorer[-1].bias)
        nn.init.zeros_(self.hold_scorer[-1].weight)
        nn.init.zeros_(self.hold_scorer[-1].bias)

    def forward(self, obs: Mapping[str, th.Tensor]) -> th.Tensor:
        movements = obs["movements"].float()
        movement_mask = obs["movement_mask"].float().clamp(0.0, 1.0)
        membership = obs["phase_membership"].float().clamp(0.0, 1.0)
        phase_features = obs["phase_features"].float()
        global_features = obs["global_features"].float()

        service = membership * movement_mask.unsqueeze(1)
        denominator = service.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        phase_movement = th.bmm(service, movements) / denominator
        phase_global = global_features.unsqueeze(1).expand(
            -1, phase_features.shape[1], -1
        )
        phase_residual = self.phase_scorer(
            th.cat([phase_movement, phase_features, phase_global], dim=-1)
        ).squeeze(-1)

        current = phase_features[..., _PHASE_INDEX["is_current"]].clamp(0.0, 1.0)
        current_denominator = current.sum(dim=-1, keepdim=True).clamp_min(1.0)
        current_movement = (
            phase_movement * current.unsqueeze(-1)
        ).sum(dim=1) / current_denominator
        hold_residual = self.hold_scorer(
            th.cat([current_movement, global_features], dim=-1)
        )
        return th.cat([hold_residual, phase_residual], dim=-1)


class SafeResidualMapAgnosticPolicy(MapAgnosticMaskablePolicy):
    """Shared movement-GNN actor constrained around normalized CMPP."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        *,
        residual_authority: float = 0.20,
        residual_bound: float = 1.0,
        max_baseline_regret: float = 0.20,
        cmpp_config: Mapping[str, float] | None = None,
        adapter_names: Sequence[str] = (),
        adapter_dim: int = 24,
        active_adapter: str | None = None,
        **kwargs: Any,
    ):
        self.residual_authority = float(residual_authority)
        self.residual_bound = float(residual_bound)
        self.max_baseline_regret = float(max_baseline_regret)
        self.cmpp_config = CMPPConfig.from_mapping(cmpp_config)
        self.adapter_names = tuple(str(name) for name in adapter_names)
        if len(set(self.adapter_names)) != len(self.adapter_names):
            raise ValueError("adapter_names must be unique")
        if any(not name or "." in name for name in self.adapter_names):
            raise ValueError("Adapter names must be non-empty and cannot contain '.'")
        self.adapter_dim = int(adapter_dim)
        self.active_adapter = str(active_adapter) if active_adapter else None
        if self.active_adapter is not None and self.active_adapter not in self.adapter_names:
            raise ValueError("active_adapter must be listed in adapter_names")
        if not 0.0 <= self.residual_authority <= 1.0:
            raise ValueError("residual_authority must be in [0, 1]")
        if self.residual_bound < 0.0:
            raise ValueError("residual_bound must be non-negative")
        if self.max_baseline_regret < 0.0:
            raise ValueError("max_baseline_regret must be non-negative")
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # A fresh checkpoint exactly reproduces CMPP.  PPO must learn a useful
        # correction before the residual can change a decision.
        actor_heads = (
            self.map_network.phase_scorer[-1],
            self.map_network.hold_scorer[-1],
        )
        for layer in actor_heads:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.map_adapters = nn.ModuleDict(
            {
                name: PermutationEquivariantResidualAdapter(self.adapter_dim)
                for name in self.adapter_names
            }
        )
        # The parent constructed its optimizer before these modules existed.
        self._rebuild_optimizer(lr_schedule(1))

    def _rebuild_optimizer(self, learning_rate: float | None = None) -> None:
        if learning_rate is None:
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable policy parameters remain")
        self.optimizer = self.optimizer_class(
            parameters,
            lr=float(learning_rate),
            **self.optimizer_kwargs,
        )

    def add_adapter(self, name: str) -> None:
        name = str(name)
        if not name or "." in name:
            raise ValueError("Adapter name must be non-empty and cannot contain '.'")
        if name in self.map_adapters:
            return
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.map_adapters[name] = PermutationEquivariantResidualAdapter(
            self.adapter_dim
        ).to(self.device)
        self.adapter_names = (*self.adapter_names, name)
        self._rebuild_optimizer(learning_rate)

    def set_active_adapter(self, name: str | None) -> None:
        if name is not None and name not in self.map_adapters:
            raise KeyError(
                f"Unknown adapter {name!r}; available={list(self.map_adapters.keys())}"
            )
        self.active_adapter = name

    def set_residual_authority(self, authority: float) -> None:
        """Update learned authority without rebuilding the policy/optimizer.

        Curriculum drivers may call this between rollout batches.  Keeping the
        value in the policy constructor state also ensures subsequent SB3 saves
        record the currently deployed authority.
        """

        authority = float(authority)
        if not 0.0 <= authority <= 1.0:
            raise ValueError("residual authority must be in [0, 1]")
        self.residual_authority = authority

    def freeze_for_adapter(self, name: str) -> None:
        self.set_active_adapter(name)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.map_adapters[name].parameters():
            parameter.requires_grad_(True)
        self._rebuild_optimizer()

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self._rebuild_optimizer()

    def safe_logits(
        self,
        obs: Mapping[str, th.Tensor],
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        residual_logits, values = self.map_network(obs)
        if self.active_adapter is not None:
            residual_logits = residual_logits + self.map_adapters[
                self.active_adapter
            ](obs)
        baseline_logits = normalized_cmpp_scores_tensor(obs, self.cmpp_config)

        if action_masks is None:
            external_valid = baseline_logits > -1e7
        else:
            external_valid = th.as_tensor(
                action_masks, dtype=th.bool, device=baseline_logits.device
            )
            if external_valid.ndim == 1:
                external_valid = external_valid.unsqueeze(0)
            external_valid = external_valid & (baseline_logits > -1e7)

        masked_baseline = baseline_logits.masked_fill(~external_valid, -1e8)
        best_baseline = masked_baseline.max(dim=-1, keepdim=True).values
        regret_valid = masked_baseline >= best_baseline - self.max_baseline_regret
        safe_valid = external_valid & regret_valid

        bounded_residual = th.tanh(residual_logits) * self.residual_bound
        combined = baseline_logits + self.residual_authority * bounded_residual
        combined = combined.masked_fill(~safe_valid, -1e8)
        return combined, values, baseline_logits

    def _distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> tuple[MaskableDistribution, th.Tensor]:
        logits, values, _baseline = self.safe_logits(obs, action_masks)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, values

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            residual_authority=self.residual_authority,
            residual_bound=self.residual_bound,
            max_baseline_regret=self.max_baseline_regret,
            cmpp_config=self.cmpp_config.to_dict(),
            adapter_names=list(self.adapter_names),
            adapter_dim=self.adapter_dim,
            active_adapter=self.active_adapter,
        )
        return data


__all__ = [
    "SafeResidualMapAgnosticPolicy",
    "normalized_cmpp_scores_tensor",
]
