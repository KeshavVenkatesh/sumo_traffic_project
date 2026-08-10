from __future__ import annotations

import unittest

try:
    import gymnasium  # noqa: F401
    import torch
    from gymnasium import spaces

    from map_agnostic_policy import MapAgnosticMaskablePolicy
    from map_agnostic_tls import MAX_PHASES, empty_observation, observation_space

    HAVE_RL = True
except ImportError:
    HAVE_RL = False


@unittest.skipUnless(HAVE_RL, "PyTorch/Gym/SB3 dependencies are not installed")
class PolicyEquivarianceTests(unittest.TestCase):
    def make_policy_and_observation(self):
        policy = MapAgnosticMaskablePolicy(
            observation_space(),
            spaces.Discrete(MAX_PHASES + 1),
            lr_schedule=lambda _: 1e-3,
            embed_dim=32,
            graph_layers=1,
        )
        obs = empty_observation()
        obs["movement_mask"][:3] = 1.0
        obs["movements"][:3, 0] = [0.2, 0.7, 0.4]
        obs["movement_adjacency"][:3, :3] = 1.0
        obs["phase_membership"][0, [0, 2]] = 1.0
        obs["phase_membership"][1, [1]] = 1.0
        obs["phase_features"][0, 0] = 1.0
        return policy, obs

    def test_permuting_movements_does_not_change_candidate_logits(self):
        policy, obs = self.make_policy_and_observation()

        def tensorize(value):
            return {key: torch.as_tensor(array[None]).float() for key, array in value.items()}

        with torch.no_grad():
            logits1, _ = policy.map_network(tensorize(obs))

        perm = [2, 0, 1]
        moved = {key: value.copy() for key, value in obs.items()}
        moved["movements"][:3] = obs["movements"][perm]
        moved["movement_mask"][:3] = obs["movement_mask"][perm]
        moved["movement_adjacency"][:3, :3] = obs["movement_adjacency"][perm][:, perm]
        moved["phase_membership"][:, :3] = obs["phase_membership"][:, perm]
        with torch.no_grad():
            logits2, _ = policy.map_network(tensorize(moved))

        torch.testing.assert_close(logits1, logits2, rtol=1e-5, atol=1e-6)

    def test_permuting_phase_candidates_only_permutes_phase_logits(self):
        policy, obs = self.make_policy_and_observation()

        def tensorize(value):
            return {key: torch.as_tensor(array[None]).float() for key, array in value.items()}

        with torch.no_grad():
            logits1, value1 = policy.map_network(tensorize(obs))

        swapped = {key: value.copy() for key, value in obs.items()}
        swapped["phase_membership"][[0, 1]] = obs["phase_membership"][[1, 0]]
        swapped["phase_features"][[0, 1]] = obs["phase_features"][[1, 0]]
        with torch.no_grad():
            logits2, value2 = policy.map_network(tensorize(swapped))

        # Action zero is hold; candidate logits start at one.
        torch.testing.assert_close(logits1[:, 0], logits2[:, 0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits1[:, 1], logits2[:, 2], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits1[:, 2], logits2[:, 1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(value1, value2, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
