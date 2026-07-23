from __future__ import annotations

import pytest

from checkpoint_contract import (
    CheckpointCompatibilityError,
    CheckpointContract,
    load_checkpoint_contract,
    write_checkpoint_contract,
)


def contract():
    return CheckpointContract.create(
        decision_seconds=10.0,
        step_length_seconds=1.0,
        minimum_green_seconds=6.0,
        maximum_green_seconds=55.0,
        residual_authority=0.2,
        residual_bound=1.0,
        max_baseline_regret=0.2,
    )


def test_contract_round_trip_and_runtime_validation(tmp_path):
    model = tmp_path / "controller.zip"
    path = write_checkpoint_contract(model, contract())
    assert path.name == "controller_contract.json"
    loaded = load_checkpoint_contract(model)
    loaded.validate_runtime(
        decision_seconds=10.0,
        step_length_seconds=1.0,
        minimum_green_seconds=6.0,
        maximum_green_seconds=55.0,
    )


def test_contract_rejects_runtime_cadence_mismatch():
    with pytest.raises(CheckpointCompatibilityError, match="decision_seconds"):
        contract().validate_runtime(
            decision_seconds=5.0,
            step_length_seconds=1.0,
            minimum_green_seconds=6.0,
            maximum_green_seconds=55.0,
        )
