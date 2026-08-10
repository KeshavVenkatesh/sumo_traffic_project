#!/usr/bin/env python3
"""Permutation-equivariant movement GNN and shared candidate-phase policy."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from map_agnostic_tls import (
    GLOBAL_FEATURE_DIM,
    MAX_MOVEMENTS,
    MAX_PHASES,
    MOVEMENT_FEATURE_DIM,
    PHASE_FEATURE_DIM,
)


class UnusedDictExtractor(BaseFeaturesExtractor):
    """SB3 requires an extractor, but the policy consumes the Dict directly."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        batch = observations["global_features"].shape[0]
        return th.zeros((batch, 1), device=observations["global_features"].device)


class GraphAttentionBlock(nn.Module):
    """Small graph-attention layer with a per-sample topology mask."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: th.Tensor,
        adjacency: th.Tensor,
        movement_mask: th.Tensor,
    ) -> th.Tensor:
        valid = movement_mask > 0.5
        pair_mask = (adjacency > 0.5) & valid.unsqueeze(1) & valid.unsqueeze(2)

        # A padded row otherwise has no valid softmax entries.  Its self edge is
        # temporarily enabled, then its output is zeroed by valid below.
        eye = th.eye(x.shape[1], dtype=th.bool, device=x.device).unsqueeze(0)
        pair_mask = pair_mask | (eye & ~valid.unsqueeze(2))

        scores = th.matmul(self.query(x), self.key(x).transpose(-1, -2))
        scores = scores / np.sqrt(float(x.shape[-1]))
        scores = scores.masked_fill(~pair_mask, -1e8)
        weights = th.softmax(scores, dim=-1)
        message = th.matmul(weights, self.value(x))
        x = self.norm1(x + self.output(message))
        x = self.norm2(x + self.ff(x))
        return x * valid.unsqueeze(-1).to(x.dtype)


class MovementGraphNetwork(nn.Module):
    def __init__(self, embed_dim: int = 128, graph_layers: int = 2):
        super().__init__()
        self.movement_encoder = nn.Sequential(
            nn.Linear(MOVEMENT_FEATURE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.graph_blocks = nn.ModuleList(
            [GraphAttentionBlock(embed_dim) for _ in range(graph_layers)]
        )
        self.phase_feature_encoder = nn.Sequential(
            nn.Linear(PHASE_FEATURE_DIM, embed_dim // 2),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        phase_input_dim = embed_dim + embed_dim // 2 + embed_dim
        self.phase_scorer = nn.Sequential(
            nn.Linear(phase_input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.hold_token = nn.Parameter(th.zeros(embed_dim))
        self.hold_scorer = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, obs: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        movements = obs["movements"].float()
        movement_mask = obs["movement_mask"].float()
        adjacency = obs["movement_adjacency"].float()
        membership = obs["phase_membership"].float()
        phase_features = obs["phase_features"].float()
        global_features = obs["global_features"].float()

        # Adapters pack real movements before padded slots. Cropping the batch
        # to its largest real intersection is mathematically identical because
        # padded embeddings/memberships are masked to zero, but changes dense
        # attention cost from MAX_MOVEMENTS² to actual_batch_max².
        active_movements = max(
            1,
            int(movement_mask.sum(dim=1).max().detach().cpu().item()),
        )
        movements = movements[:, :active_movements]
        movement_mask = movement_mask[:, :active_movements]
        adjacency = adjacency[:, :active_movements, :active_movements]
        membership = membership[:, :, :active_movements]

        encoded = self.movement_encoder(movements)
        for block in self.graph_blocks:
            encoded = block(encoded, adjacency, movement_mask)

        valid_membership = membership * movement_mask.unsqueeze(1)
        phase_denominator = valid_membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
        phase_embedding = th.bmm(valid_membership, encoded) / phase_denominator

        global_embedding = self.global_encoder(global_features)
        phase_static = self.phase_feature_encoder(phase_features)
        expanded_global = global_embedding.unsqueeze(1).expand(-1, MAX_PHASES, -1)
        phase_input = th.cat([phase_embedding, phase_static, expanded_global], dim=-1)
        phase_logits = self.phase_scorer(phase_input).squeeze(-1)

        current_weight = phase_features[..., 0:1].clamp(0.0, 1.0)
        current_denominator = current_weight.sum(dim=1).clamp_min(1.0)
        current_embedding = (phase_embedding * current_weight).sum(dim=1) / current_denominator
        no_current = current_weight.sum(dim=1) <= 0.0
        current_embedding = th.where(
            no_current,
            self.hold_token.unsqueeze(0).expand_as(current_embedding),
            current_embedding,
        )
        hold_logit = self.hold_scorer(
            th.cat([current_embedding, global_embedding], dim=-1)
        )
        logits = th.cat([hold_logit, phase_logits], dim=-1)

        movement_denominator = movement_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_movements = (
            encoded * movement_mask.unsqueeze(-1)
        ).sum(dim=1) / movement_denominator
        value = self.value_head(th.cat([pooled_movements, global_embedding], dim=-1))
        return logits, value


class MapAgnosticMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose phase logits come from one shared scorer.

    Padded phase position is never fed into a position-specific output layer.
    Permuting movements and phase candidates therefore permutes the relevant
    logits rather than changing their physical meaning.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        embed_dim: int = 128,
        graph_layers: int = 2,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("MapAgnosticMaskablePolicy requires a Gymnasium Dict observation space.")
        if not isinstance(action_space, spaces.Discrete) or int(action_space.n) != MAX_PHASES + 1:
            raise TypeError(
                f"Expected Discrete({MAX_PHASES + 1}) action space (hold + padded phase candidates)."
            )
        self.embed_dim = int(embed_dim)
        self.graph_layers = int(graph_layers)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[],
            activation_fn=nn.Tanh,
            ortho_init=False,
            features_extractor_class=UnusedDictExtractor,
            features_extractor_kwargs=None,
            share_features_extractor=True,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self.map_network = MovementGraphNetwork(
            embed_dim=self.embed_dim,
            graph_layers=self.graph_layers,
        )
        # Keep conventional attribute names for SB3 inspection utilities.
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.map_network.apply(partial(self.init_weights, gain=np.sqrt(2.0)))
        nn.init.normal_(self.map_network.phase_scorer[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.map_network.phase_scorer[-1].bias)
        nn.init.normal_(self.map_network.hold_scorer[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.map_network.hold_scorer[-1].bias)
        nn.init.normal_(self.map_network.value_head[-1].weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.map_network.value_head[-1].bias)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> tuple[MaskableDistribution, th.Tensor]:
        logits, values = self.map_network(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, values

    def forward(
        self,
        obs: dict[str, th.Tensor],
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution, values = self._distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(
        self,
        obs: dict[str, th.Tensor],
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        distribution, values = self._distribution(obs, action_masks)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | None = None,
    ) -> MaskableDistribution:
        distribution, _values = self._distribution(obs, action_masks)
        return distribution

    def predict_values(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        _logits, values = self.map_network(obs)
        return values

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(embed_dim=self.embed_dim, graph_layers=self.graph_layers)
        return data
