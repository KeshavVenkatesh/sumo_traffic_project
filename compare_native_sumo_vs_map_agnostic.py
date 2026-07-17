#!/usr/bin/env python3
"""Evaluate a schema-v3 movement-GNN policy against native SUMO timing."""

from __future__ import annotations

import numpy as np

import compare_fixed_vs_single_vs_all_model_realistic as cmp
from map_agnostic_tls import (
    MAX_PHASES,
    MapTrafficSnapshot,
    adapter_for_controller,
)
from traffic_rl_map_agnostic_env import MapAgnosticPolicyShapeEnv


def map_agnostic_observation(controller, episode_seconds):
    del episode_seconds  # all normalization is local/analytical
    adapter = adapter_for_controller(controller, cmp.sim.traci, cmp.sim)
    return adapter.observe(update_history=True).observation


_evaluation_snapshot_cache = MapTrafficSnapshot(cmp.sim.traci, cmp.sim)


def map_agnostic_observation_batch(controllers, episode_seconds):
    del episode_seconds
    adapters = [
        adapter_for_controller(
            controller,
            cmp.sim.traci,
            cmp.sim,
            snapshot_cache=_evaluation_snapshot_cache,
        )
        for controller in controllers
    ]
    _evaluation_snapshot_cache.refresh(adapters)
    return [
        adapter.observe(update_history=True).observation
        for adapter in adapters
    ]


def map_agnostic_action_mask(controller):
    adapter = adapter_for_controller(controller, cmp.sim.traci, cmp.sim)
    if adapter.last_snapshot is None:
        adapter.observe(update_history=True)
    return adapter.action_mask(
        min_green=cmp.MIN_GREEN_BEFORE_SWITCH,
        max_green=cmp.HARD_MAX_GREEN,
    )


def apply_map_agnostic_action(controller, action):
    adapter = adapter_for_controller(controller, cmp.sim.traci, cmp.sim)
    mask = map_agnostic_action_mask(controller)
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
cmp.PolicyShapeEnv = MapAgnosticPolicyShapeEnv
cmp.get_observation = map_agnostic_observation
cmp.get_observation_batch = map_agnostic_observation_batch
cmp.valid_action_mask_for_controller = map_agnostic_action_mask
cmp.apply_model_action_to_controller = apply_map_agnostic_action
cmp.MODEL_DEFAULT = "models/traffic_signal_map_agnostic_v3"

# Schema v2 uses bounded physical ratios and must never load map-specific
# VecNormalize statistics from a legacy checkpoint with a similar file name.
cmp.find_vecnormalize_path = lambda model_path, explicit_path=None: None

print("[map-agnostic compare] schema=3, analytical normalization, movement/phase candidates")
print(f"[map-agnostic compare] action size={MAX_PHASES + 1} (hold + padded candidates)")

import compare_native_sumo_vs_all_model as native  # noqa: E402  (after patching cmp)


if __name__ == "__main__":
    native.main()
