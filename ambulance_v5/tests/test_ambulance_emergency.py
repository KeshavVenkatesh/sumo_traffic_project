from __future__ import annotations

import math
import unittest

import numpy as np

torch = None
try:
    import torch

    from ambulance_emergency import (
        EmergencyOverrideNetwork,
        EmergencyTLSContext,
        RollingGreenCorridor,
        empty_emergency_observation,
    )
    from ambulance_system import AMBULANCE_ID_PREFIX
    from ambulance_multiagent_worker import (
        EmergencyAllTLSEpisode,
        _policy_actions,
    )
    from map_agnostic_tls import MAX_PHASES, empty_observation

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class CorridorTests(unittest.TestCase):
    @staticmethod
    def context(
        *,
        relevant=True,
        nearest_eta=5.0,
        recovery=False,
        authority=True,
    ):
        return EmergencyTLSContext(
            tls_id="tls",
            observation=empty_emergency_observation(),
            relevant_ambulances=(
                (f"{AMBULANCE_ID_PREFIX}0",) if relevant else ()
            ),
            ambulance_priorities=(
                {f"{AMBULANCE_ID_PREFIX}0": 1.0}
                if relevant
                else {}
            ),
            target_actions=((2,) if relevant else ()),
            protected_target_actions=((2,) if relevant else ()),
            nearest_eta_seconds=nearest_eta,
            next_signal_present=relevant,
            recovery_active=recovery,
            corridor_mode=("serve" if relevant else "normal"),
            authority_available=authority,
        )

    def test_no_relevant_ambulance_returns_exact_base_action(self):
        corridor = RollingGreenCorridor()
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[[0, 1, 2]] = True
        action = corridor.teacher_action(
            self.context(relevant=False),
            base_action=1,
            action_mask=mask,
            sim_time=10.0,
        )
        self.assertEqual(action, 1)

    def test_teacher_prefers_legal_protected_ambulance_phase(self):
        corridor = RollingGreenCorridor()
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[[0, 1, 2]] = True
        action = corridor.teacher_action(
            self.context(),
            base_action=1,
            action_mask=mask,
            sim_time=10.0,
        )
        self.assertEqual(action, 2)

    def test_teacher_waits_until_the_preparation_window(self):
        corridor = RollingGreenCorridor(prepare_eta_seconds=25.0)
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[[0, 1, 2]] = True
        action = corridor.teacher_action(
            self.context(nearest_eta=40.0),
            base_action=1,
            action_mask=mask,
            sim_time=10.0,
        )
        self.assertEqual(action, 1)

    def test_teacher_holds_safely_until_target_phase_is_legal(self):
        corridor = RollingGreenCorridor()
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[[0, 1]] = True
        action = corridor.teacher_action(
            self.context(),
            base_action=1,
            action_mask=mask,
            sim_time=10.0,
        )
        self.assertEqual(action, 0)

    def test_teacher_scores_conflicting_ambulance_phases_by_urgency(self):
        corridor = RollingGreenCorridor()
        observation = empty_emergency_observation()
        observation["emergency_phase_features"][0, [0, 1, 2, 3, 5]] = [
            1.0,
            1.0,
            0.2,
            0.3,
            1.0,
        ]
        observation["emergency_phase_features"][1, [0, 1, 2, 3, 5]] = [
            1.0,
            1.0,
            0.9,
            0.8,
            1.0,
        ]
        ambulance_id = f"{AMBULANCE_ID_PREFIX}0"
        context = EmergencyTLSContext(
            tls_id="tls",
            observation=observation,
            relevant_ambulances=(ambulance_id,),
            ambulance_priorities={ambulance_id: 1.0},
            target_actions=(1, 2),
            protected_target_actions=(1, 2),
            nearest_eta_seconds=5.0,
            next_signal_present=True,
            recovery_active=False,
            corridor_mode="serve",
            authority_available=True,
        )
        mask = np.zeros(MAX_PHASES + 1, dtype=bool)
        mask[[0, 1, 2]] = True
        self.assertEqual(
            corridor.teacher_action(context, 1, mask, 10.0),
            2,
        )

    def test_override_budget_is_hard_except_near_stop_line(self):
        far = self.context(nearest_eta=20.0, authority=False)
        far.observation["emergency_global_features"][7] = 0.0
        self.assertFalse(far.override_allowed)
        near = self.context(nearest_eta=5.0, authority=False)
        near.observation["emergency_global_features"][7] = 0.0
        self.assertTrue(near.override_allowed)

    def test_budget_reserves_a_complete_decision_interval(self):
        corridor = RollingGreenCorridor(max_preemption_seconds=45.0)
        corridor.state("tls").preemption_seconds = 40.0
        self.assertFalse(
            corridor.can_afford_preemption("tls", 10.0)
        )
        self.assertTrue(
            corridor.can_afford_preemption("tls", 5.0)
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class EmergencyNetworkTests(unittest.TestCase):
    def test_zero_initialized_override_starts_at_exact_base_logits(self):
        network = EmergencyOverrideNetwork(
            embed_dim=16,
            graph_layers=1,
            residual_bound=4.0,
        )
        observation = empty_observation()
        observation["movement_mask"][:2] = 1.0
        observation["movement_adjacency"][:2, :2] = 1.0
        observation["phase_membership"][0, 0] = 1.0
        observation["phase_membership"][1, 1] = 1.0
        batch = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in observation.items()
        }
        emergency = {
            key: torch.as_tensor(value).unsqueeze(0)
            for key, value in empty_emergency_observation().items()
        }
        base_logits = torch.linspace(
            -1.0, 1.0, MAX_PHASES + 1
        ).unsqueeze(0)
        combined, value, residual = network(
            batch,
            emergency,
            base_logits,
            authority=1.0,
        )
        self.assertTrue(torch.equal(combined, base_logits))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertTrue(
            all(math.isfinite(float(item)) for item in value.flatten())
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class EmergencyOnlyExitSpaceTests(unittest.TestCase):
    class BaseNetwork(torch.nn.Module if HAS_TORCH else object):
        def forward(self, observations):
            batch = observations["movements"].shape[0]
            logits = torch.full(
                (batch, MAX_PHASES + 1),
                -5.0,
                device=observations["movements"].device,
            )
            logits[:, 2] = 10.0
            return logits, torch.zeros(
                (batch, 1), device=logits.device
            )

    class OverrideNetwork(torch.nn.Module if HAS_TORCH else object):
        def forward(
            self,
            base_observation,
            emergency_observation,
            base_logits,
            authority,
        ):
            combined = base_logits.clone()
            combined[:, 1] = 100.0
            return (
                combined,
                torch.zeros(
                    (len(combined), 1), device=combined.device
                ),
                torch.zeros_like(combined),
            )

    def policy(self, override_safe):
        masks = np.zeros((1, MAX_PHASES + 1), dtype=bool)
        masks[0, [0, 1, 2]] = True
        return _policy_actions(
            self.BaseNetwork(),
            self.OverrideNetwork(),
            [empty_observation()],
            [empty_emergency_observation()],
            masks,
            override_safe,
            np.asarray([True]),
            1.0,
            torch.device("cpu"),
            deterministic=True,
        )

    def test_unsafe_override_cannot_erase_frozen_base_action(self):
        override_safe = np.zeros(
            (1, MAX_PHASES + 1), dtype=bool
        )
        override_safe[0, 0] = True
        policy = self.policy(override_safe)
        self.assertEqual(policy["base_actions"].tolist(), [2])
        self.assertEqual(policy["actions"].tolist(), [2])
        self.assertTrue(policy["effective_masks"][0, 2])

    def test_safe_override_can_replace_frozen_base_action(self):
        override_safe = np.zeros(
            (1, MAX_PHASES + 1), dtype=bool
        )
        override_safe[0, [0, 1]] = True
        policy = self.policy(override_safe)
        self.assertEqual(policy["base_actions"].tolist(), [2])
        self.assertEqual(policy["actions"].tolist(), [1])

    def test_exact_gap_mask_is_computed_only_for_active_tls(self):
        class Adapter:
            tls_id = "tls"

            def __init__(self):
                self.checked = []

            @staticmethod
            def action_to_phase_position(action):
                return int(action) - 1

            def phase_position_has_exit_space(self, phase_pos, gap):
                self.checked.append((phase_pos, gap))
                return phase_pos == 1

        adapter = Adapter()
        episode = EmergencyAllTLSEpisode.__new__(
            EmergencyAllTLSEpisode
        )
        episode.adapters = [adapter]
        episode._emergency_required_exit_gap_meters = 18.0
        base_masks = np.zeros(
            (1, MAX_PHASES + 1), dtype=bool
        )
        base_masks[0, [0, 1, 2]] = True

        result = episode.emergency_override_exit_space_masks(
            base_masks, np.asarray([True])
        )
        self.assertEqual(
            np.flatnonzero(result[0]).tolist(), [0, 2]
        )
        self.assertEqual(adapter.checked, [(0, 18.0), (1, 18.0)])

        adapter.checked.clear()
        episode.emergency_override_exit_space_masks(
            base_masks, np.asarray([False])
        )
        self.assertEqual(adapter.checked, [])

    def test_override_switch_stores_frozen_base_phase_for_clearance(self):
        class Adapter:
            tls_id = "tls"

            def __init__(self):
                self.controller = {
                    "mode": "green",
                    "phase_pos": 0,
                    "phase_elapsed": 10.0,
                }

            @staticmethod
            def action_mask(**kwargs):
                mask = np.zeros(MAX_PHASES + 1, dtype=bool)
                mask[[0, 1, 2]] = True
                return mask

            @staticmethod
            def action_to_phase_position(action):
                return int(action) - 1

            @staticmethod
            def phase_position_has_exit_space(phase_pos, gap):
                return phase_pos == 1 and gap == 18.0

        class Legacy:
            MIN_GREEN_BEFORE_SWITCH = 6.0
            HARD_MAX_GREEN = 55.0

        class Simulation:
            @staticmethod
            def request_switch(controller, phase_pos):
                controller["next_phase_pos"] = phase_pos
                controller["mode"] = "yellow"
                return True

            @staticmethod
            def switch_next_fixed_phase(controller):
                raise AssertionError("hard-max fallback was not expected")

        adapter = Adapter()
        episode = EmergencyAllTLSEpisode.__new__(
            EmergencyAllTLSEpisode
        )
        episode._emergency_required_exit_gap_meters = 18.0
        episode._decision_base_actions = {"tls": 0}
        episode.config = {
            "strict_exit_space": False,
            "required_exit_gap_meters": 18.0,
            "allow_unsafe_hard_max_fallback": True,
        }
        episode.legacy = Legacy()
        episode.sim = Simulation()

        switched, hard_max = episode._apply_action(adapter, 2)

        self.assertTrue(switched)
        self.assertFalse(hard_max)
        self.assertEqual(adapter.controller["next_phase_pos"], 1)
        self.assertEqual(
            adapter.controller["emergency_override_phase_pos"], 1
        )
        self.assertEqual(
            adapter.controller["emergency_base_fallback_phase_pos"], 0
        )
        self.assertTrue(
            adapter.controller["emergency_green_activation_guard"](1)
        )
        self.assertEqual(
            adapter.controller["emergency_override_switch_requests"], 1
        )


if __name__ == "__main__":
    unittest.main()
