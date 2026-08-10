from __future__ import annotations

import importlib
import random
import sys
import types
import unittest


if "traci" not in sys.modules:
    traci_stub = types.ModuleType("traci")

    class TraCIException(Exception):
        pass

    traci_stub.TraCIException = TraCIException
    sys.modules["traci"] = traci_stub

sim = importlib.import_module("realistic_all_intersections_fixed_cycle")


class NativePhase:
    def __init__(self, state, duration=10.0):
        self.state = state
        self.duration = duration


class NativeLogic:
    programID = "0"
    phases = [
        NativePhase("GGrr"),
        NativePhase("yyrr"),
        NativePhase("rrrr"),
        NativePhase("rrGG"),
        NativePhase("rryy"),
        NativePhase("GGrr"),
    ]


class FakeTrafficLight:
    def __init__(self):
        self.states = []

    def getProgram(self, tls_id):
        return "0"

    def setRedYellowGreenState(self, tls_id, state):
        self.states.append(state)


class FakeSimulation:
    def getTime(self):
        return 0.0


class FakeTraci:
    def __init__(self):
        self.trafficlight = FakeTrafficLight()
        self.simulation = FakeSimulation()


class NativePhaseCatalogTests(unittest.TestCase):
    def test_filters_transitions_deduplicates_and_clears_safely(self):
        old_traci = sim.traci
        old_logic = sim._get_tls_program_logics_for_debug
        old_classifier = sim.classify_tls_movements
        fake = FakeTraci()
        try:
            sim.traci = fake
            sim._get_tls_program_logics_for_debug = lambda tls_id: [NativeLogic()]
            sim.classify_tls_movements = lambda tls_id: (
                4,
                {label: {} for label in sim.MOVEMENT_LABELS},
            )
            controller = sim.build_map_agnostic_controller_for_tls(
                "tls", random.Random(1), activate=True
            )
            self.assertIsNotNone(controller)
            self.assertEqual(len(controller["phases"]), 2)
            self.assertEqual(controller["phase_catalog"], "native_stable_greens")

            sim.start_yellow(controller)
            self.assertLessEqual(set(fake.trafficlight.states[-1]), set("yr"))
            sim.start_all_red(controller)
            self.assertEqual(fake.trafficlight.states[-1], "rrrr")
        finally:
            sim.traci = old_traci
            sim._get_tls_program_logics_for_debug = old_logic
            sim.classify_tls_movements = old_classifier

    def test_green_activation_rechecks_exit_space_after_clearance(self):
        old_traci = sim.traci
        fake = FakeTraci()
        try:
            sim.traci = fake
            allowed = {"value": False}
            controller = {
                "tls_id": "tls",
                "state_length": 4,
                "movement_map": {},
                "phases": [
                    {"state": "GGrr"},
                    {"state": "rrGG"},
                ],
                "phase_pos": 0,
                "mode": "all_red",
                "remaining": sim.STEP_LENGTH,
                "phase_elapsed": 0.0,
                "last_active_indices": set(),
                "next_phase_pos": 1,
                "disabled": False,
                "last_signal_update": 0.0,
                "phase_catalog": "native_stable_greens",
                "green_activation_guard": (
                    lambda phase_position: (
                        phase_position == 1 and allowed["value"]
                    )
                ),
            }
            sim.update_controller_after_simstep(controller)
            self.assertEqual(controller["mode"], "all_red")
            self.assertEqual(controller["phase_pos"], 0)
            self.assertEqual(
                controller["blocked_green_activation_seconds"],
                sim.STEP_LENGTH,
            )

            allowed["value"] = True
            sim.update_controller_after_simstep(controller)
            self.assertEqual(controller["mode"], "green")
            self.assertEqual(controller["phase_pos"], 1)
            self.assertEqual(fake.trafficlight.states[-1], "rrGG")
        finally:
            sim.traci = old_traci


    def test_blocked_requested_green_uses_guarded_safe_alternative(self):
        old_traci = sim.traci
        fake = FakeTraci()
        try:
            sim.traci = fake
            checked_positions = []

            def activation_guard(phase_position):
                checked_positions.append(phase_position)
                return phase_position == 0

            controller = {
                "tls_id": "tls",
                "state_length": 4,
                "movement_map": {},
                "phases": [
                    {"state": "GGrr"},
                    {"state": "rrGG"},
                ],
                "phase_pos": 0,
                "mode": "all_red",
                "remaining": sim.STEP_LENGTH,
                "phase_elapsed": 0.0,
                "last_active_indices": set(),
                "next_phase_pos": 1,
                "disabled": False,
                "last_signal_update": 0.0,
                "phase_catalog": "native_stable_greens",
                "green_activation_guard": activation_guard,
            }

            sim.update_controller_after_simstep(controller)

            self.assertEqual(checked_positions, [1, 0])
            self.assertEqual(controller["mode"], "green")
            self.assertEqual(controller["phase_pos"], 0)
            self.assertEqual(fake.trafficlight.states[-1], "GGrr")
            self.assertEqual(
                controller.get("guarded_fallback_green_activations", 0),
                1,
            )
        finally:
            sim.traci = old_traci

    def test_blocked_emergency_green_resumes_stored_base_phase(self):
        old_traci = sim.traci
        fake = FakeTraci()
        try:
            sim.traci = fake
            checked_positions = []

            def emergency_guard(phase_position):
                checked_positions.append(phase_position)
                return False

            controller = {
                "tls_id": "tls",
                "state_length": 4,
                "movement_map": {},
                "phases": [
                    {"state": "GGrr"},
                    {"state": "rrGG"},
                ],
                "phase_pos": 0,
                "mode": "all_red",
                "remaining": sim.STEP_LENGTH,
                "phase_elapsed": 0.0,
                "last_active_indices": set(),
                "next_phase_pos": 1,
                "disabled": False,
                "last_signal_update": 0.0,
                "phase_catalog": "native_stable_greens",
                "emergency_green_activation_guard": emergency_guard,
                "emergency_override_phase_pos": 1,
                "emergency_base_fallback_phase_pos": 0,
            }

            sim.update_controller_after_simstep(controller)

            self.assertEqual(checked_positions, [1])
            self.assertEqual(controller["mode"], "green")
            self.assertEqual(controller["phase_pos"], 0)
            self.assertEqual(fake.trafficlight.states[-1], "GGrr")
            self.assertEqual(
                controller.get(
                    "emergency_base_fallback_green_activations", 0
                ),
                1,
            )
            self.assertEqual(
                controller.get("blocked_green_activation_seconds", 0.0),
                0.0,
            )
            self.assertNotIn(
                "emergency_green_activation_guard", controller
            )
            self.assertNotIn(
                "emergency_base_fallback_phase_pos", controller
            )
        finally:
            sim.traci = old_traci


if __name__ == "__main__":
    unittest.main()
