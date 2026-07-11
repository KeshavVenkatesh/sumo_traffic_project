#!/usr/bin/env python3
from __future__ import annotations

import importlib
import numpy as np

from gymnasium import spaces

import compare_fixed_vs_single_vs_all_model_realistic as cmp
import compare_native_sumo_vs_all_model as native


POLICY_ENV_MODULE = "traffic_rl_model_general_proxy"
policy_env = importlib.import_module(POLICY_ENV_MODULE)

OBS_SIZE = int(getattr(policy_env, "OBSERVATION_SIZE", 46))


class GeneralPolicyShapeEnv(cmp.gym.Env):
    """Dummy env only for loading VecNormalize/model shapes."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((OBS_SIZE,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((OBS_SIZE,), dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(5, dtype=bool)


def ensure_general_controller(controller):
    if controller is None:
        return controller

    if "slot_to_pos" not in controller:
        controller["slot_to_pos"] = {
            int(phase.get("slot", i)): i
            for i, phase in enumerate(controller.get("phases", []))
        }

    if "movement_out_lanes_cache" not in controller or "all_out_lanes" not in controller:
        movement_map = controller.get("movement_map")
        if movement_map is not None:
            in_cache, out_cache, all_in, all_out = policy_env.build_lane_caches_general(
                movement_map
            )
            controller.setdefault("movement_in_lanes_cache", in_cache)
            controller["movement_out_lanes_cache"] = out_cache
            controller["all_in_lanes"] = set(controller.get("all_in_lanes", set())) | set(all_in)
            controller["all_out_lanes"] = set(controller.get("all_out_lanes", set())) | set(all_out)

    return controller


def general_get_observation(controller, episode_seconds):
    ensure_general_controller(controller)
    return policy_env.get_observation(controller).astype(np.float32)


def general_valid_action_mask_for_controller(controller):
    ensure_general_controller(controller)

    mask = np.zeros(5, dtype=bool)

    if controller.get("disabled") or controller.get("mode") != "green":
        mask[0] = True
        return mask

    elapsed = float(controller.get("phase_elapsed", 0.0))
    min_green = float(getattr(policy_env, "MIN_GREEN_BEFORE_SWITCH", cmp.MIN_GREEN_BEFORE_SWITCH))
    max_green = float(getattr(policy_env, "MAX_GREEN_HOLD", cmp.HARD_MAX_GREEN))

    if elapsed < min_green:
        mask[0] = True
        return mask

    if elapsed < max_green:
        mask[0] = True

    current_pos = controller.get("phase_pos")

    lanes = set(controller.get("all_in_lanes", set())) | set(controller.get("all_out_lanes", set()))
    try:
        snapshot = policy_env.lane_vehicle_snapshot(lanes)
    except Exception:
        snapshot = None

    for phase in controller.get("phases", []):
        slot = int(phase.get("slot", -1))
        if slot < 0 or slot >= 4:
            continue

        phase_pos = controller.get("slot_to_pos", {}).get(slot)
        if phase_pos is None or phase_pos == current_pos:
            continue

        # General-model safety mask: avoid serving blocked downstream exits.
        try:
            if policy_env.phase_is_downstream_blocked(controller, phase, snapshot=snapshot):
                continue
        except Exception:
            pass

        mask[slot + 1] = True

    # If downstream blocking masked every switch, fall back to the old legal phase mask.
    # This prevents deadlock at max-green.
    if not mask.any() or (not mask[1:].any() and elapsed >= max_green):
        if elapsed < max_green:
            mask[0] = True

        for phase in controller.get("phases", []):
            slot = int(phase.get("slot", -1))
            if slot < 0 or slot >= 4:
                continue

            phase_pos = controller.get("slot_to_pos", {}).get(slot)
            if phase_pos is None or phase_pos == current_pos:
                continue

            mask[slot + 1] = True

    if not mask.any():
        mask[0] = True

    return mask


# Patch the comparison runner for the generalized policy.
cmp.PolicyShapeEnv = GeneralPolicyShapeEnv
cmp.get_observation = general_get_observation
cmp.valid_action_mask_for_controller = general_valid_action_mask_for_controller

# Match the force-switch timing to the training env if available.
if hasattr(policy_env, "MIN_GREEN_BEFORE_SWITCH"):
    cmp.MIN_GREEN_BEFORE_SWITCH = float(policy_env.MIN_GREEN_BEFORE_SWITCH)
if hasattr(policy_env, "MAX_GREEN_HOLD"):
    cmp.HARD_MAX_GREEN = float(policy_env.MAX_GREEN_HOLD)

print(f"[general compare] policy env module = {POLICY_ENV_MODULE}")
print(f"[general compare] observation size = {OBS_SIZE}")

if __name__ == "__main__":
    native.main()
