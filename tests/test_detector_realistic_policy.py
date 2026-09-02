from __future__ import annotations

import unittest

try:
    import gymnasium  # noqa: F401
    import torch
    from gymnasium import spaces

    from detector_realistic_policy import DetectorRealisticMaskablePolicy
    from detector_realistic_tls import MAX_PHASES, empty_observation, observation_space
    from map_agnostic_tls import MAX_PHASES as SCHEMA_V3_MAX_PHASES

    HAVE_RL = True
except ImportError:
    HAVE_RL = False


@unittest.skipUnless(HAVE_RL, "PyTorch/Gym/SB3 dependencies are not installed")
class DetectorPolicyEquivarianceTests(unittest.TestCase):
    def test_detector_phase_cap_is_independent_of_schema_v3(self):
        self.assertEqual(SCHEMA_V3_MAX_PHASES, 16)
        self.assertEqual(MAX_PHASES, 32)

    def test_lane_group_permutation_does_not_change_phase_scores(self):
        policy = DetectorRealisticMaskablePolicy(
            observation_space(),
            spaces.Discrete(MAX_PHASES + 1),
            lr_schedule=lambda _: 1e-3,
            embed_dim=32,
            graph_layers=1,
        )
        observation = empty_observation()
        observation["movement_mask"][:3] = 1.0
        observation["movements"][:3, 10] = [0.2, 0.7, 0.4]
        observation["movement_adjacency"][:3, :3] = 1.0
        observation["phase_membership"][0, [0, 2]] = 1.0
        observation["phase_membership"][1, [1]] = 1.0
        observation["phase_features"][0, 0] = 1.0

        def tensorize(value):
            return {
                key: torch.as_tensor(array[None]).float()
                for key, array in value.items()
            }

        with torch.no_grad():
            first, _ = policy.map_network(tensorize(observation))
        self.assertEqual(first.shape, (1, MAX_PHASES + 1))
        permutation = [2, 0, 1]
        moved = {key: value.copy() for key, value in observation.items()}
        moved["movements"][:3] = observation["movements"][permutation]
        moved["movement_mask"][:3] = observation["movement_mask"][permutation]
        moved["movement_adjacency"][:3, :3] = observation[
            "movement_adjacency"
        ][permutation][:, permutation]
        moved["phase_membership"][:, :3] = observation["phase_membership"][
            :, permutation
        ]
        with torch.no_grad():
            second, _ = policy.map_network(tensorize(moved))
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
