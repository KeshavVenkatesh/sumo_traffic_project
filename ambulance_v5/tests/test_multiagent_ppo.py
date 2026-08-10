from __future__ import annotations

import math
import types
import unittest

import numpy as np

try:
    import torch

    from map_agnostic_multiagent_worker import normalized_max_pressure_actions
    from map_agnostic_tls import MAX_PHASES, empty_observation
    from train_map_agnostic_multiagent import SharedPPOTrainer, flatten_rollouts

    HAS_RL = True
except ImportError:
    HAS_RL = False


@unittest.skipUnless(HAS_RL, "PyTorch/Gym/SB3 dependencies are not installed")
class MultiAgentPPOTests(unittest.TestCase):
    def observations(self):
        result = []
        for index in range(2):
            observation = empty_observation()
            observation["movement_mask"][:2] = 1.0
            observation["phase_membership"][0, 0] = 1.0
            observation["phase_membership"][1, 1] = 1.0
            observation["phase_features"][0, 0] = float(index == 0)
            observation["phase_features"][1, 0] = float(index == 1)
            observation["phase_features"][0, 2] = 0.2
            observation["phase_features"][1, 2] = 0.8
            result.append(observation)
        return result

    def rollout(self):
        observations = self.observations()
        time_steps = 3
        agents = 2
        dynamic = {
            key: np.stack(
                [np.stack([obs[key] for obs in observations])] * time_steps
            )
            for key in ("movements", "phase_features", "global_features")
        }
        static = {
            key: np.stack([obs[key] for obs in observations])
            for key in (
                "movement_mask",
                "movement_adjacency",
                "phase_membership",
            )
        }
        action_masks = np.zeros(
            (time_steps, agents, MAX_PHASES + 1), dtype=np.uint8
        )
        action_masks[..., :3] = 1
        return {
            "dynamic": dynamic,
            "static": static,
            "action_masks": action_masks,
            "actions": np.ones((time_steps, agents), dtype=np.int16),
            "old_log_probs": np.full(
                (time_steps, agents), -math.log(3.0), dtype=np.float32
            ),
            "old_values": np.zeros((time_steps, agents), dtype=np.float32),
            "advantages": np.ones((time_steps, agents), dtype=np.float32),
            "returns": np.ones((time_steps, agents), dtype=np.float32),
            "teacher_actions": np.full(
                (time_steps, agents), 2, dtype=np.int16
            ),
            "agent_weights": np.asarray([0.5, 0.5], dtype=np.float32),
        }

    def test_teacher_never_selects_a_masked_action(self):
        observations = self.observations()
        masks = np.zeros((2, MAX_PHASES + 1), dtype=bool)
        masks[0, [0, 1]] = True
        masks[1, [0, 2]] = True
        actions = normalized_max_pressure_actions(observations, masks)
        self.assertTrue(all(masks[i, action] for i, action in enumerate(actions)))

    def test_flatten_and_weighted_ppo_update(self):
        rollout = self.rollout()
        flat = flatten_rollouts([rollout])
        self.assertEqual(flat["actions"].shape, (6,))
        self.assertEqual(flat["movements"].shape[0], 6)
        self.assertAlmostEqual(float(flat["sample_weights"].sum()), 3.0)

        args = types.SimpleNamespace(
            embed_dim=16,
            graph_layers=1,
            learning_rate=1e-3,
            final_learning_rate=1e-4,
            ppo_epochs=1,
            minibatch_size=4,
            clip_range=0.2,
            value_clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            teacher_coef=0.1,
            teacher_decay_fraction=0.25,
            max_grad_norm=0.5,
            target_kl=0.0,
        )
        trainer = SharedPPOTrainer(args, torch.device("cpu"))
        metrics = trainer.update([rollout], total_planned_updates=4)
        self.assertEqual(trainer.completed_updates, 1)
        self.assertEqual(trainer.agent_transitions, 6)
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
