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


if __name__ == "__main__":
    unittest.main()

