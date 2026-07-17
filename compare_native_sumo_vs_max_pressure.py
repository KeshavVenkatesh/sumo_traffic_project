#!/usr/bin/env python3
"""Evaluate a normalized MaxPressure/starvation heuristic on every TLS."""

from __future__ import annotations

import numpy as np

import compare_native_sumo_vs_map_agnostic as schema


cmp = schema.cmp


class IdentityNormalizer:
    def normalize_obs(self, observation):
        return observation

    def close(self):
        return None


class NormalizedMaxPressureModel:
    def predict(
        self,
        observations,
        deterministic=True,
        action_masks=None,
    ):
        del deterministic
        phase = np.asarray(observations["phase_features"], dtype=np.float32)
        global_features = np.asarray(
            observations["global_features"], dtype=np.float32
        )
        if phase.ndim == 2:
            phase = phase[None, ...]
            global_features = global_features[None, ...]
        batch = phase.shape[0]
        masks = (
            np.asarray(action_masks, dtype=bool)
            if action_masks is not None
            else np.ones((batch, schema.MAX_PHASES + 1), dtype=bool)
        )
        # Queue and positive pressure favor service; downstream space and
        # starvation prevent spillback and indefinite neglect.
        phase_scores = (
            1.00 * phase[..., 2]
            + 1.50 * phase[..., 3]
            + 0.60 * phase[..., 4]
            + 0.80 * phase[..., 5]
            + 0.20 * phase[..., 7]
        )
        scores = np.full(
            (batch, schema.MAX_PHASES + 1), -1e9, dtype=np.float32
        )
        scores[:, 1:] = phase_scores
        current = phase[..., 0]
        current_score = (phase_scores * current).sum(axis=1)
        # Holding avoids unnecessary clearance loss early in a useful green.
        minimum_green_progress = global_features[:, 1]
        scores[:, 0] = current_score + 0.15 * (
            1.0 - minimum_green_progress
        )
        scores[~masks] = -1e9
        actions = np.argmax(scores, axis=1).astype(np.int64)
        return actions, None


def load_heuristic(args, seed, label):
    del args, seed, label
    return NormalizedMaxPressureModel(), IdentityNormalizer()


cmp.load_model_with_vecnormalize = load_heuristic
cmp.ALL_MODEL_CONTROLLER_LABEL = "max_pressure"


if __name__ == "__main__":
    schema.native.main()
