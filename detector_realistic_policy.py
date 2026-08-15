#!/usr/bin/env python3
"""Shared graph policy for detector-realistic schema v4 observations."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from map_agnostic_policy import MapAgnosticMaskablePolicy, MovementGraphNetwork


class DetectorGraphNetwork(MovementGraphNetwork):
    """The proven phase-scoring GNN applied to detector-history features.

    Schema v4 intentionally keeps the 24/8/8 tensor dimensions from schema v3.
    Temporal information is encoded through rolling 10/60-second detector
    measurements and conservation-based state estimates, while this network
    retains the map- and permutation-equivariant spatial inductive bias.
    """


class DetectorRealisticMaskablePolicy(MapAgnosticMaskablePolicy):
    """Maskable SB3 policy with a schema-v4 checkpoint identity."""

    def _build(self, lr_schedule: Schedule) -> None:
        self.map_network = DetectorGraphNetwork(
            embed_dim=self.embed_dim,
            graph_layers=self.graph_layers,
        )
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


__all__ = ["DetectorGraphNetwork", "DetectorRealisticMaskablePolicy"]
