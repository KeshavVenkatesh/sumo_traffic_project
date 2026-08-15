#!/usr/bin/env python3
"""Evaluate detector-realistic schema v4 against native SUMO timing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import compare_fixed_vs_single_vs_all_model_realistic as cmp
from detector_realistic_tls import (
    MAX_PHASES,
    DetectorTrafficSnapshot,
    adapter_for_controller,
    sensor_config_from_environment,
)
from traffic_rl_detector_realistic_env import DetectorRealisticPolicyShapeEnv


def decision_seconds_from_argv(argv: list[str]) -> float:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-update-period",
        type=float,
        default=float(os.environ.get("DETECTOR_DECISION_SECONDS", "10")),
    )
    parsed, _unknown = parser.parse_known_args(argv)
    if parsed.model_update_period <= 0.0:
        raise ValueError("--model-update-period must be positive")
    return float(parsed.model_update_period)


SENSOR_CONFIG = replace(
    sensor_config_from_environment(training=False),
    nominal_decision_seconds=decision_seconds_from_argv(sys.argv[1:]),
)


def detector_metadata_path(path: str | Path) -> Path:
    value = Path(path)
    base = value.with_suffix("") if value.suffix == ".zip" else value
    return base.parent / f"{base.name}_detector_realistic.json"


def verify_detector_checkpoint(path: str | Path) -> dict:
    metadata_file = detector_metadata_path(path)
    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"Missing schema-v4 checkpoint metadata: {metadata_file}. "
            "Use a checkpoint exported by train_detector_realistic_multiagent.py."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != 4:
        raise ValueError(
            f"Refusing non-v4 checkpoint metadata in {metadata_file}: "
            f"schema_version={metadata.get('schema_version')!r}."
        )
    expected_policy = (
        "detector_realistic_policy.DetectorRealisticMaskablePolicy"
    )
    if metadata.get("policy_class") != expected_policy:
        raise ValueError(
            f"Refusing incompatible policy in {metadata_file}: "
            f"{metadata.get('policy_class')!r}."
        )
    trained_decision_seconds = float(metadata.get("decision_seconds", 0.0))
    if trained_decision_seconds <= 0.0:
        raise ValueError(f"Invalid decision_seconds in {metadata_file}.")
    if not np.isclose(
        trained_decision_seconds,
        SENSOR_CONFIG.nominal_decision_seconds,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Detector decision cadence differs from training: "
            f"trained={trained_decision_seconds:g}s, "
            f"evaluation={SENSOR_CONFIG.nominal_decision_seconds:g}s."
        )
    return metadata


def model_path_from_argv(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", default=cmp.MODEL_DEFAULT)
    parsed, _unknown = parser.parse_known_args(argv)
    return str(parsed.model_path)


def detector_realistic_observation(controller, episode_seconds):
    del episode_seconds  # all normalization is local/analytical
    adapter = adapter_for_controller(
        controller,
        cmp.sim.traci,
        cmp.sim,
        sensor_config=SENSOR_CONFIG,
    )
    return adapter.observe(update_history=True).observation


_evaluation_snapshot_cache = DetectorTrafficSnapshot(cmp.sim.traci)


def detector_realistic_observation_batch(controllers, episode_seconds):
    del episode_seconds
    adapters = [
        adapter_for_controller(
            controller,
            cmp.sim.traci,
            cmp.sim,
            snapshot_cache=_evaluation_snapshot_cache,
            sensor_config=SENSOR_CONFIG,
            rng=np.random.default_rng(index + 17),
        )
        for index, controller in enumerate(controllers)
    ]
    _evaluation_snapshot_cache.refresh(adapters)
    return [
        adapter.observe(update_history=True).observation
        for adapter in adapters
    ]


def detector_realistic_action_mask(controller):
    adapter = adapter_for_controller(
        controller, cmp.sim.traci, cmp.sim, sensor_config=SENSOR_CONFIG
    )
    if adapter.last_snapshot is None:
        adapter.observe(update_history=True)
    return adapter.action_mask(
        min_green=cmp.MIN_GREEN_BEFORE_SWITCH,
        max_green=cmp.HARD_MAX_GREEN,
    )


def apply_detector_realistic_action(controller, action):
    adapter = adapter_for_controller(
        controller, cmp.sim.traci, cmp.sim, sensor_config=SENSOR_CONFIG
    )
    mask = detector_realistic_action_mask(controller)
    action = int(action)
    invalid = action < 0 or action >= len(mask) or not mask[action]
    controller["_last_model_action_was_invalid"] = bool(invalid)
    if invalid:
        action = 0 if mask[0] else int(np.flatnonzero(mask)[0])

    elapsed = float(controller.get("phase_elapsed", 0.0))
    if action == 0:
        if (
            controller.get("mode") == "green"
            and elapsed >= cmp.HARD_MAX_GREEN
            and not mask[1:].any()
        ):
            return bool(cmp.sim.switch_next_fixed_phase(controller)), True
        return False, False

    phase_pos = adapter.action_to_phase_position(action)
    if (
        phase_pos is None
        or controller.get("mode") != "green"
        or elapsed < cmp.MIN_GREEN_BEFORE_SWITCH
    ):
        return False, False
    return bool(cmp.sim.request_switch(controller, phase_pos)), False


# Patch only the policy-facing seams.  Traffic generation, native timing, and
# metric collection remain exactly the shared comparison implementation.
cmp.PolicyShapeEnv = DetectorRealisticPolicyShapeEnv
cmp.get_observation = detector_realistic_observation
cmp.get_observation_batch = detector_realistic_observation_batch
cmp.valid_action_mask_for_controller = detector_realistic_action_mask
cmp.apply_model_action_to_controller = apply_detector_realistic_action
cmp.MODEL_DEFAULT = "models/detector_realistic_multiagent_v4"

# Schema v4 uses bounded physical ratios and must never load map-specific
# VecNormalize statistics from a legacy checkpoint with a similar file name.
cmp.find_vecnormalize_path = lambda model_path, explicit_path=None: None

print(
    "[detector-realistic compare] schema=4, "
    f"profile={SENSOR_CONFIG.profile}, no route/wait/ETA inputs"
)
print(
    f"[detector-realistic compare] action size={MAX_PHASES + 1} "
    "(hold + padded candidates)"
)

import compare_native_sumo_vs_all_model as native  # noqa: E402  (after patching cmp)


if __name__ == "__main__":
    verified = verify_detector_checkpoint(model_path_from_argv(sys.argv[1:]))
    print(
        "[detector-realistic compare] verified checkpoint "
        f"schema={verified['schema_version']} policy={verified['policy_class']}"
    )
    native.main()
