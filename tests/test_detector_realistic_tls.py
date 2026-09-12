from __future__ import annotations

import unittest

import numpy as np

from detector_realistic_tls import (
    DETECTOR_FEATURE_NAMES,
    MAX_PHASES,
    DetectorRealisticTLSAdapter,
    DetectorSensorConfig,
    DetectorTrafficSnapshot,
)


class FakeLaneAPI:
    def __init__(self, world):
        self.world = world

    def getEdgeID(self, lane):
        return lane.split("_")[0]

    def getLength(self, lane):
        return 100.0

    def getMaxSpeed(self, lane):
        return 10.0

    def getLastStepVehicleIDs(self, lane):
        return tuple(self.world.lane_vehicles.get(lane, ()))


class FakeVehicleAPI:
    def __init__(self, world):
        self.world = world
        self.calls = []

    def getLanePosition(self, vehicle):
        self.calls.append("getLanePosition")
        return self.world.positions[vehicle]

    def getSpeed(self, vehicle):
        self.calls.append("getSpeed")
        return self.world.speeds[vehicle]

    def getRoute(self, vehicle):
        raise AssertionError("Detector-realistic observations must not query routes")

    def getRouteIndex(self, vehicle):
        raise AssertionError("Detector-realistic observations must not query routes")

    def getWaitingTime(self, vehicle):
        raise AssertionError("Detector-realistic observations must not query waiting time")


class FakeTrafficLightAPI:
    def getControlledLinks(self, tls_id):
        assert tls_id == "tls"
        # The first two signal links use one shared incoming lane.  A real loop
        # cannot split that lane's demand by intended destination.
        return (
            (("north_0", "southout_0", ""),),
            (("north_0", "eastout_0", ""),),
            (("east_0", "westout_0", ""),),
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
            "north_0": ["near_stop", "advance"],
            "east_0": [],
            "southout_0": [],
            "eastout_0": [],
            "westout_0": [],
        }
        self.positions = {"near_stop": 95.0, "advance": 15.0}
        self.speeds = {"near_stop": 0.0, "advance": 5.0}
        self.lane = FakeLaneAPI(self)
        self.vehicle = FakeVehicleAPI(self)
        self.trafficlight = FakeTrafficLightAPI()
        self.simulation = FakeSimulationAPI(self)


class FakeSimulationModule:
    @staticmethod
    def get_sumo_link_direction(incoming, outgoing):
        del incoming
        if outgoing.startswith("eastout"):
            return "L"
        return "S"


def controller():
    return {
        "tls_id": "tls",
        "phase_pos": 0,
        "phase_elapsed": 10.0,
        "mode": "green",
        "disabled": False,
        "phases": [
            {"name": "north", "state": "GGr"},
            {"name": "east", "state": "rrG"},
        ],
    }


def clean_config(profile="loops"):
    return DetectorSensorConfig(
        profile=profile,
        observation_noise_std=0.0,
        calibration_jitter=0.0,
        transient_dropout_probability=0.0,
        stuck_detector_probability=0.0,
        max_latency_decisions=0,
    )


class DetectorRealisticAdapterTests(unittest.TestCase):
    def setUp(self):
        self.world = FakeWorld()
        self.controller = controller()
        self.cache = DetectorTrafficSnapshot(self.world)
        self.adapter = DetectorRealisticTLSAdapter(
            self.controller,
            self.world,
            FakeSimulationModule,
            snapshot_cache=self.cache,
            sensor_config=clean_config("loops"),
            rng=np.random.default_rng(7),
        )
        self.cache.refresh([self.adapter])

    def test_shared_lane_is_one_detector_group(self):
        self.assertEqual(len(self.adapter.topology.movements), 2)
        north = next(
            item
            for item in self.adapter.topology.movements
            if item.incoming_lanes == ("north_0",)
        )
        self.assertEqual(set(north.turns), {"L", "S"})
        self.assertEqual(set(north.signal_indices), {0, 1})

    def test_loop_profile_exposes_no_oracle_speed_or_downstream_state(self):
        snapshot = self.adapter.observe()
        north_index = next(
            index
            for index, item in enumerate(self.adapter.topology.movements)
            if item.incoming_lanes == ("north_0",)
        )
        features = snapshot.observation["movements"][north_index]
        speed = DETECTOR_FEATURE_NAMES.index("speed_ratio")
        downstream = DETECTOR_FEATURE_NAMES.index("downstream_occupancy")
        speed_available = DETECTOR_FEATURE_NAMES.index("speed_available")
        self.assertEqual(features[speed], -1.0)
        self.assertEqual(features[downstream], -1.0)
        self.assertEqual(features[speed_available], 0.0)

    def test_observation_never_queries_route_or_waiting_time(self):
        self.adapter.observe()
        self.assertNotIn("getRoute", self.world.vehicle.calls)
        self.assertNotIn("getRouteIndex", self.world.vehicle.calls)
        self.assertNotIn("getWaitingTime", self.world.vehicle.calls)

    def test_crossing_pulses_update_rolling_arrival_and_queue_estimates(self):
        self.adapter.observe()
        self.world.time += 10.0
        self.world.lane_vehicles["north_0"] = ["advance"]
        self.world.positions["advance"] = 25.0  # distance 85 -> 75, crosses loop at 80
        self.cache.refresh([self.adapter])
        snapshot = self.adapter.observe()
        north_index = next(
            index
            for index, item in enumerate(self.adapter.topology.movements)
            if item.incoming_lanes == ("north_0",)
        )
        features = snapshot.observation["movements"][north_index]
        self.assertGreater(
            features[DETECTOR_FEATURE_NAMES.index("arrival_rate_short")], 0.0
        )
        self.assertGreater(self.adapter.last_detected_departures, 0.0)
        self.assertEqual(snapshot.vehicle_ids, frozenset())
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(features >= -1.0))
        self.assertTrue(np.all(features <= 1.0))

    def test_camera_profile_can_expose_speed_and_downstream_occupancy(self):
        camera = DetectorRealisticTLSAdapter(
            controller(),
            self.world,
            FakeSimulationModule,
            sensor_config=clean_config("camera"),
            rng=np.random.default_rng(1),
        )
        snapshot = camera.observe()
        real = snapshot.observation["movements"][
            snapshot.observation["movement_mask"].astype(bool)
        ]
        self.assertTrue(
            np.all(real[:, DETECTOR_FEATURE_NAMES.index("speed_available")] == 1.0)
        )
        self.assertTrue(
            np.all(real[:, DETECTOR_FEATURE_NAMES.index("downstream_occupancy")] >= 0.0)
        )

    def test_legal_action_mask_keeps_fixed_padded_shape(self):
        self.adapter.observe()
        mask = self.adapter.action_mask(min_green=6.0, max_green=55.0)
        self.assertEqual(mask.shape, (MAX_PHASES + 1,))
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])


if __name__ == "__main__":
    unittest.main()
