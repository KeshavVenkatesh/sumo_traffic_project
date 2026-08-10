from __future__ import annotations

import unittest

import numpy as np

from ambulance_system import AMBULANCE_ID_PREFIX
from map_agnostic_tls import (
    DEFAULT_REQUIRED_EXIT_GAP_METERS,
    MAX_PHASES,
    MapAgnosticTLSAdapter,
    MapTrafficSnapshot,
    normalized_reward,
)


class FakeLaneAPI:
    def __init__(self, world):
        self.world = world
        self.calls = {}

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def getEdgeID(self, lane):
        return lane.split("_")[0]

    def getLength(self, lane):
        self._count("getLength")
        return 100.0

    def getMaxSpeed(self, lane):
        self._count("getMaxSpeed")
        return 10.0

    def getLastStepVehicleIDs(self, lane):
        self._count("getLastStepVehicleIDs")
        return tuple(self.world.lane_vehicles.get(lane, ()))


class FakeVehicleAPI:
    def __init__(self, world):
        self.world = world
        self.calls = {}

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def getLanePosition(self, vehicle):
        self._count("getLanePosition")
        return self.world.positions[vehicle]

    def getSpeed(self, vehicle):
        self._count("getSpeed")
        return self.world.speeds[vehicle]

    def getWaitingTime(self, vehicle):
        self._count("getWaitingTime")
        return self.world.waits[vehicle]

    def getRoute(self, vehicle):
        self._count("getRoute")
        return ()

    def getRouteIndex(self, vehicle):
        self._count("getRouteIndex")
        return -1


class FakeTrafficLightAPI:
    def getControlledLinks(self, tls_id):
        assert tls_id == "tls"
        return (
            (("north_0", "southout_0", ""),),
            (("south_0", "northout_0", ""),),
            (("east_0", "westout_0", ""),),
            (("west_0", "eastout_0", ""),),
        )


class FakeSimulationAPI:
    def __init__(self, world):
        self.world = world

    def getTime(self):
        return self.world.time


class FakeWorld:
    def __init__(self):
        self.time = 10.0
        self.lane_vehicles = {
            "north_0": ["n1", "n2"],
            "south_0": ["s1"],
            "east_0": [],
            "west_0": [],
            "southout_0": [],
            "northout_0": [],
            "westout_0": [],
            "eastout_0": [],
        }
        self.positions = {"n1": 95.0, "n2": 80.0, "s1": 90.0}
        self.speeds = {"n1": 0.0, "n2": 5.0, "s1": 0.0}
        self.waits = {"n1": 20.0, "n2": 0.0, "s1": 5.0}
        self.lane = FakeLaneAPI(self)
        self.vehicle = FakeVehicleAPI(self)
        self.trafficlight = FakeTrafficLightAPI()
        self.simulation = FakeSimulationAPI(self)


class FakeSimulationModule:
    @staticmethod
    def get_sumo_link_direction(in_lane, out_lane):
        del in_lane, out_lane
        return "S"

    @staticmethod
    def outgoing_lane_has_space(lane):
        del lane
        return True


def controller():
    return {
        "tls_id": "tls",
        "phase_pos": 0,
        "phase_elapsed": 10.0,
        "mode": "green",
        "disabled": False,
        "phases": [
            {"name": "north-south", "state": "GGrr"},
            {"name": "east-west", "state": "rrGG"},
        ],
    }


class MapAgnosticAdapterTests(unittest.TestCase):
    def setUp(self):
        self.world = FakeWorld()
        self.controller = controller()
        self.adapter = MapAgnosticTLSAdapter(
            self.controller, self.world, FakeSimulationModule
        )

    def test_builds_physical_movements_and_variable_phase_membership(self):
        self.assertEqual(len(self.adapter.topology.movements), 4)
        self.assertEqual(self.adapter.phase_count, 2)
        phase_edges = [
            {self.adapter.topology.movements[i].incoming_edge for i in members}
            for members in self.adapter.topology.phase_members
        ]
        self.assertEqual(phase_edges[0], {"north", "south"})
        self.assertEqual(phase_edges[1], {"east", "west"})

    def test_phase_encoding_distinguishes_protected_and_permissive_green(self):
        mixed = controller()
        mixed["phases"][0]["state"] = "Ggrr"
        adapter = MapAgnosticTLSAdapter(mixed, self.world, FakeSimulationModule)
        self.assertEqual(set(adapter.topology.phase_weights[0]), {0.5, 1.0})
        snapshot = adapter.observe()
        encoded = snapshot.observation["phase_membership"][0]
        self.assertIn(0.5, encoded)
        self.assertIn(1.0, encoded)

    def test_observation_is_bounded_and_padding_is_masked(self):
        snapshot = self.adapter.observe()
        obs = snapshot.observation
        self.assertEqual(int(obs["movement_mask"].sum()), 4)
        self.assertEqual(int(obs["phase_membership"].sum()), 4)
        self.assertTrue(np.all(np.isfinite(obs["movements"])))
        self.assertTrue(np.all(obs["movements"] <= 1.0))
        self.assertTrue(np.all(obs["movements"] >= -1.0))

    def test_action_indices_are_candidate_relative_not_named_slots(self):
        mask = self.adapter.action_mask(min_green=6.0, max_green=55.0)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])  # current candidate
        self.assertTrue(mask[2])
        self.assertFalse(mask[3:].any())
        self.assertEqual(self.adapter.action_to_phase_position(2), 1)

    def test_minimum_and_maximum_green_are_enforced(self):
        self.controller["phase_elapsed"] = 2.0
        self.assertEqual(np.flatnonzero(self.adapter.action_mask()).tolist(), [0])
        self.controller["phase_elapsed"] = 60.0
        mask = self.adapter.action_mask()
        self.assertFalse(mask[0])
        self.assertTrue(mask[2])

    def test_strict_exit_gap_masks_blocked_phase_and_holds_at_hard_max(self):
        self.world.lane_vehicles["westout_0"] = ["blocked"]
        self.world.positions["blocked"] = (
            DEFAULT_REQUIRED_EXIT_GAP_METERS - 0.1
        )
        self.world.speeds["blocked"] = 0.0
        self.world.waits["blocked"] = 0.0
        self.controller["phase_elapsed"] = 60.0
        mask = self.adapter.action_mask(
            require_exit_space=True,
            required_exit_gap_meters=(
                DEFAULT_REQUIRED_EXIT_GAP_METERS
            ),
            allow_unsafe_hard_max_fallback=False,
        )
        self.assertEqual(np.flatnonzero(mask).tolist(), [0])

    def test_strict_exit_gap_allows_phase_at_exact_required_distance(self):
        self.world.lane_vehicles["westout_0"] = ["clear"]
        self.world.positions["clear"] = (
            DEFAULT_REQUIRED_EXIT_GAP_METERS
        )
        self.world.speeds["clear"] = 0.0
        self.world.waits["clear"] = 0.0
        mask = self.adapter.action_mask(
            require_exit_space=True,
            required_exit_gap_meters=(
                DEFAULT_REQUIRED_EXIT_GAP_METERS
            ),
            allow_unsafe_hard_max_fallback=False,
        )
        self.assertTrue(mask[2])

    def test_reward_has_no_raw_map_size_term(self):
        first = self.adapter.observe()
        self.world.time += 5.0
        self.world.lane_vehicles["north_0"] = ["n2"]
        second = self.adapter.observe()
        reward, components = normalized_reward(
            previous=first,
            current=second,
            local_cleared=1,
            decision_seconds=5.0,
            switched=False,
        )
        self.assertTrue(np.isfinite(reward))
        self.assertIn("queue_level", components)
        self.assertNotIn("global_vehicle_count", components)

    def test_action_mask_has_fixed_batch_shape_only(self):
        self.assertEqual(self.adapter.action_mask().shape, (MAX_PHASES + 1,))

    def test_map_snapshot_reuses_dynamic_queries_across_adapters(self):
        second = MapAgnosticTLSAdapter(
            controller(), self.world, FakeSimulationModule
        )
        cache = MapTrafficSnapshot(self.world, FakeSimulationModule)
        self.adapter.snapshot_cache = cache
        second.snapshot_cache = cache
        cache.refresh([self.adapter, second])
        self.adapter.observe()
        second.observe()

        # Eight unique lanes and three unique vehicles are queried once for
        # the map snapshot, even though both adapters use the same topology.
        self.assertEqual(
            self.world.lane.calls["getLastStepVehicleIDs"], 8
        )
        self.assertEqual(self.world.vehicle.calls["getSpeed"], 3)
        self.assertEqual(self.world.vehicle.calls["getWaitingTime"], 3)

        cache.refresh([self.adapter, second])
        self.assertEqual(self.world.lane.calls["getLength"], 8)
        self.assertEqual(self.world.lane.calls["getMaxSpeed"], 8)
        self.assertEqual(
            self.world.lane.calls["getLastStepVehicleIDs"], 16
        )

    def test_map_snapshot_can_exclude_ambulances_from_ordinary_state(self):
        ambulance_id = f"{AMBULANCE_ID_PREFIX}0"
        self.world.lane_vehicles["north_0"].append(ambulance_id)
        cache = MapTrafficSnapshot(
            self.world,
            FakeSimulationModule,
            exclude_vehicle=lambda vehicle_id: vehicle_id.startswith(
                AMBULANCE_ID_PREFIX
            ),
        )
        self.adapter.snapshot_cache = cache
        cache.refresh([self.adapter])
        snapshot = self.adapter.observe()
        self.assertNotIn(ambulance_id, snapshot.vehicle_ids)


if __name__ == "__main__":
    unittest.main()
