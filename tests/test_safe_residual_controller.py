from __future__ import annotations

import numpy as np
import pytest

from map_agnostic_tls import (
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_NAMES,
    empty_observation,
)
from safe_residual_controller import SafeResidualController, normalized_cmpp_scores


def observation():
    obs = empty_observation()
    queue = MOVEMENT_FEATURE_NAMES.index("queue_density")
    pressure = MOVEMENT_FEATURE_NAMES.index("normalized_pressure")
    current = PHASE_FEATURE_NAMES.index("is_current")
    obs["movement_mask"][:2] = 1.0
    obs["phase_membership"][0, 0] = 1.0
    obs["phase_membership"][1, 1] = 1.0
    obs["phase_features"][0, current] = 1.0
    obs["movements"][0, [queue, pressure]] = (0.2, 0.2)
    obs["movements"][1, [queue, pressure]] = (0.9, 0.9)
    return obs


def mask():
    value = np.zeros(17, dtype=bool)
    value[:3] = True
    return value


def test_zero_authority_exactly_uses_deterministic_baseline():
    obs = observation()
    expected = int(np.argmax(normalized_cmpp_scores(obs, mask())))
    decision = SafeResidualController(residual_authority=0.0).select_action(
        obs, mask(), residual_scores=np.linspace(-100, 100, 17)
    )
    assert decision.action == expected == decision.baseline_action


def test_external_mask_remains_authoritative():
    obs = observation()
    allowed = mask()
    allowed[2] = False
    decision = SafeResidualController(residual_authority=1.0).select_action(
        obs, allowed, residual_scores=np.full(17, 1_000.0)
    )
    assert allowed[decision.action]
    assert not decision.allowed_actions[2]


def test_uncertainty_and_ood_remove_residual_authority():
    controller = SafeResidualController(residual_authority=0.8)
    for kwargs in ({"uncertainty": 1.0}, {"in_distribution": False}):
        decision = controller.select_action(
            observation(), mask(), residual_scores=np.ones(17), **kwargs
        )
        assert decision.effective_authority == 0.0
        assert decision.action == decision.baseline_action


def test_invalid_all_false_mask_fails_closed():
    with pytest.raises(ValueError, match="No finite CMPP action"):
        normalized_cmpp_scores(observation(), np.zeros(17, dtype=bool))
