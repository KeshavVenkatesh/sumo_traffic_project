from __future__ import annotations

import types
import unittest

from ambulance_system import (
    AMBULANCE_ID_PREFIX,
    AMBULANCE_TYPE_ID,
    AmbulanceSystem,
    AmbulanceSystemConfig,
    RouteTLSIndex,
    schedule_fingerprint,
)


def adapter():
    movements = (
        types.SimpleNamespace(
            incoming_edge="e0",
            outgoing_edge="e1",
            signal_indices=(0,),
        ),
        types.SimpleNamespace(
            incoming_edge="e1",
            outgoing_edge="e0",
            signal_indices=(1,),
        ),
    )
    topology = types.SimpleNamespace(
        movements=movements,
        phase_members=((0,), (1,)),
        phase_weights=((1.0,), (0.5,)),
    )
    return types.SimpleNamespace(tls_id="tls", topology=topology)


class FakeVehicleTypeAPI:
    def __init__(self):
        self.types = {"global_car"}
        self.settings = []

    def getIDList(self):
        return tuple(self.types)

    def copy(self, source, target):
        assert source == "global_car"
        self.types.add(target)

    def __getattr__(self, name):
        if not name.startswith("set"):
            raise AttributeError(name)

        def setter(type_id, value):
            self.settings.append((name, type_id, value))

        return setter


class FakeRouteAPI:
    def __init__(self):
        self.routes = {}

    def add(self, route_id, edges):
        self.routes[route_id] = tuple(edges)


class FakeVehicleAPI:
    def __init__(self, route_api):
        self.route_api = route_api
        self.routes = {}
        self.ids = set()
        self.added = {}
        self.raise_on_color = False

    def add(self, **kwargs):
        self.added[kwargs["vehID"]] = dict(kwargs)
        self.routes[kwargs["vehID"]] = self.route_api.routes[
            kwargs["routeID"]
        ]

    def setColor(self, vehicle_id, color):
        del vehicle_id, color
        if self.raise_on_color:
            raise RuntimeError("cosmetic color failure")

    def getIDList(self):
        return tuple(self.ids)

    def getDeparture(self, vehicle_id):
        return float(self.added[vehicle_id]["depart"])

    def getRoute(self, vehicle_id):
        return self.routes[vehicle_id]

    def getRouteIndex(self, vehicle_id):
        del vehicle_id
        return 0

    def getLaneID(self, vehicle_id):
        return f"{self.routes[vehicle_id][0]}_0"

    def getLanePosition(self, vehicle_id):
        del vehicle_id
        return 10.0

    def getSpeed(self, vehicle_id):
        del vehicle_id
        return 10.0

    def getTimeLoss(self, vehicle_id):
        del vehicle_id
        return 0.5

    def getDistance(self, vehicle_id):
        del vehicle_id
        return 10.0

    def getNextTLS(self, vehicle_id):
        route = self.routes[vehicle_id]
        link_index = 0 if route[0] == "e0" else 1
        return (("tls", link_index, 50.0, "r"),)

    def getRoadID(self, vehicle_id):
        return self.routes[vehicle_id][0]

    def setRoute(self, vehicle_id, edges):
        self.routes[vehicle_id] = tuple(edges)


class FakeSimulationAPI:
    def __init__(self, vehicle_types):
        self.vehicle_types = vehicle_types
        self.time = 0.0
        self.departed = set()
        self.arrived = set()
        self.teleported = set()
        self.collided = set()

    def getTime(self):
        return self.time

    def findRoute(
        self,
        origin,
        destination,
        type_id,
        depart=0.0,
        routingMode=0,
    ):
        del depart, routingMode
        # Regression check: AmbulanceSystem must create its vType before
        # asking SUMO to find a route with it.
        assert type_id in self.vehicle_types.types
        return types.SimpleNamespace(
            edges=(origin, destination),
            travelTime=20.0,
        )

    def getDepartedIDList(self):
        return tuple(self.departed)

    def getArrivedIDList(self):
        return tuple(self.arrived)

    def getStartingTeleportIDList(self):
        return tuple(self.teleported)

    def getCollidingVehiclesIDList(self):
        return tuple(self.collided)


class FakeEdgeAPI:
    @staticmethod
    def getLaneNumber(edge_id):
        del edge_id
        return 1

    @staticmethod
    def getTraveltime(edge_id):
        del edge_id
        return 10.0


class FakeLaneAPI:
    @staticmethod
    def getLength(lane_id):
        del lane_id
        return 100.0

    @staticmethod
    def getMaxSpeed(lane_id):
        del lane_id
        return 10.0


class FakeTraCI:
    def __init__(self):
        self.vehicletype = FakeVehicleTypeAPI()
        self.route = FakeRouteAPI()
        self.vehicle = FakeVehicleAPI(self.route)
        self.simulation = FakeSimulationAPI(self.vehicletype)
        self.edge = FakeEdgeAPI()
        self.lane = FakeLaneAPI()
        self.constants = types.SimpleNamespace(
            ROUTING_MODE_DEFAULT=0,
            ROUTING_MODE_AGGREGATED=1,
        )


class FakeSimulationModule:
    @staticmethod
    def edge_length(edge_id, edge_metadata):
        return float(edge_metadata[edge_id]["length"])

    @staticmethod
    def route_distance(edges, edge_metadata):
        return sum(
            float(edge_metadata[edge]["length"]) for edge in edges
        )

    @staticmethod
    def edge_xy(edge_id, edge_metadata):
        item = edge_metadata[edge_id]
        return float(item["x"]), float(item["y"])


def make_system(
    *,
    max_ambulances=1,
    max_active=1,
    interval=100.0,
    last_spawn_buffer=300.0,
    first_spawn=1.0,
):
    traci = FakeTraCI()
    config = AmbulanceSystemConfig(
        routing_mode="traffic_aware",
        first_spawn_seconds=first_spawn,
        spawn_interval_seconds=interval,
        spawn_jitter_seconds=0.0,
        max_ambulances=max_ambulances,
        max_active_ambulances=max_active,
        min_euclidean_distance=0.0,
        min_route_distance=1.0,
        min_route_edges=2,
        min_route_tls=1,
        route_attempts_per_ambulance=40,
        reroute_jitter_seconds=0.0,
        last_spawn_buffer_seconds=last_spawn_buffer,
    )
    system = AmbulanceSystem(
        traci_module=traci,
        simulation_module=FakeSimulationModule,
        adapters=[adapter()],
        raw_graph={"e0": ("e1",), "e1": ("e0",)},
        edge_metadata={
            "e0": {"length": 100.0, "x": 0.0, "y": 0.0},
            "e1": {"length": 100.0, "x": 100.0, "y": 0.0},
        },
        sim_state={},
        episode_seconds=100.0,
        schedule_seed=17,
        config=config,
    )
    return system, traci


class RouteIndexTests(unittest.TestCase):
    def test_maps_link_indices_and_protected_then_permissive_actions(self):
        index = RouteTLSIndex([adapter()])
        self.assertEqual(index.movement_for_link("tls", 0), 0)
        route_tls = index.route_tls(("e0", "e1"))
        self.assertEqual(len(route_tls), 1)
        self.assertEqual(route_tls[0].protected_candidate_actions, (1,))
        self.assertEqual(route_tls[0].candidate_actions, (1,))

        reverse = index.route_tls(("e1", "e0"))
        self.assertEqual(reverse[0].protected_candidate_actions, ())
        self.assertEqual(reverse[0].permissive_candidate_actions, (2,))

    def test_schedule_fingerprint_is_stable_and_sensitive(self):
        first, _traci = make_system()
        second, _traci = make_system()
        self.assertEqual(first.schedule_hash, second.schedule_hash)
        self.assertEqual(
            first.schedule_hash,
            schedule_fingerprint(first.schedule),
        )


class AmbulanceLifecycleTests(unittest.TestCase):
    def _depart(self, system, traci):
        system.begin_decision()
        system.before_simulation_step({}, 0.0, None)
        ambulance_id = f"{AMBULANCE_ID_PREFIX}0"
        traci.vehicle.ids.add(ambulance_id)
        traci.simulation.departed = {ambulance_id}
        traci.simulation.time = 1.0
        system.after_simulation_step({}, 1.0, None)
        traci.simulation.departed.clear()
        return ambulance_id

    def test_vehicle_type_exists_before_route_planning_and_is_passenger(self):
        _system, traci = make_system()
        self.assertIn(AMBULANCE_TYPE_ID, traci.vehicletype.types)
        self.assertIn(
            (
                "setVehicleClass",
                AMBULANCE_TYPE_ID,
                "passenger",
            ),
            traci.vehicletype.settings,
        )

    def test_episode_start_spawn_is_rejected_without_prequeue_step(self):
        with self.assertRaisesRegex(ValueError, "prequeuing"):
            AmbulanceSystemConfig(first_spawn_seconds=0.0)

    def test_spawn_is_prequeued_one_step_early_with_fixed_departure(self):
        system, traci = make_system(first_spawn=10.0)
        ambulance_id = f"{AMBULANCE_ID_PREFIX}0"
        system.begin_decision()
        system.before_simulation_step({}, 8.0, None)
        self.assertNotIn(ambulance_id, system.records)

        system.before_simulation_step({}, 9.0, None)
        self.assertEqual(system.records[ambulance_id].status, "pending")
        self.assertEqual(
            traci.vehicle.added[ambulance_id]["depart"], "10.0"
        )

        traci.vehicle.ids.add(ambulance_id)
        traci.simulation.departed = {ambulance_id}
        traci.simulation.time = 10.0
        system.after_simulation_step({}, 10.0, None)
        self.assertEqual(system.records[ambulance_id].status, "active")
        self.assertEqual(
            system.records[ambulance_id].actual_departure, 10.0
        )
        self.assertTrue(system.records[ambulance_id].next_tls)

    def test_cosmetic_color_failure_does_not_invalidate_insertion(self):
        system, traci = make_system()
        traci.vehicle.raise_on_color = True
        system.begin_decision()
        system.before_simulation_step({}, 0.0, None)
        record = system.records[f"{AMBULANCE_ID_PREFIX}0"]
        self.assertEqual(record.status, "pending")
        self.assertEqual(system.end_decision().failed, [])

    def test_disappearance_without_arrival_is_failure_not_success(self):
        system, traci = make_system()
        ambulance_id = self._depart(system, traci)
        traci.vehicle.ids.clear()
        traci.simulation.time = 2.0
        system.after_simulation_step({}, 2.0, None)
        self.assertEqual(system.records[ambulance_id].status, "removed")
        self.assertEqual(system.summary()["arrived_total"], 0)

    def test_only_arrived_event_counts_as_success(self):
        system, traci = make_system()
        ambulance_id = self._depart(system, traci)
        traci.vehicle.ids.clear()
        traci.simulation.arrived = {ambulance_id}
        traci.simulation.time = 2.0
        system.after_simulation_step({}, 2.0, None)
        self.assertEqual(system.records[ambulance_id].status, "arrived")
        self.assertEqual(system.summary()["arrived_total"], 1)

    def test_starting_teleport_is_a_failure_and_never_an_arrival(self):
        system, traci = make_system()
        ambulance_id = self._depart(system, traci)
        traci.simulation.teleported = {ambulance_id}
        traci.simulation.time = 2.0
        system.after_simulation_step({}, 2.0, None)
        summary = system.summary()
        self.assertEqual(
            system.records[ambulance_id].status, "teleported"
        )
        self.assertEqual(summary["failed_total"], 1)
        self.assertEqual(summary["arrived_total"], 0)
        self.assertEqual(
            summary["teleported_vehicle_ids"], [ambulance_id]
        )

    def test_rerouting_is_suppressed_near_the_next_signal(self):
        system, traci = make_system()
        ambulance_id = self._depart(system, traci)
        record = system.records[ambulance_id]
        original_route = record.route_edges
        self.assertLess(
            record.next_tls[0][2],
            system.config.no_reroute_within_tls_meters,
        )
        system._maybe_reroute(record, 20.0)
        self.assertEqual(record.route_updates, 0)
        self.assertEqual(record.route_edges, original_route)

    def test_unspawned_schedule_entries_are_censored_not_hidden(self):
        system, traci = make_system(
            max_ambulances=2,
            max_active=1,
            interval=10.0,
            last_spawn_buffer=0.0,
        )
        self._depart(system, traci)
        system.before_simulation_step({}, 20.0, None)
        self.assertEqual(len(system.records), 1)
        system.finish_episode(100.0)
        summary = system.summary()
        self.assertEqual(summary["scheduled_total"], 2)
        self.assertEqual(summary["spawned_total"], 2)
        self.assertEqual(summary["censored_total"], 2)
        self.assertEqual(
            sum(
                record["route_mode"]
                == "not_spawned_before_horizon"
                for record in summary["records"]
            ),
            1,
        )
        self.assertEqual(summary["completion_rate"], 0.0)
        self.assertEqual(
            set(system.end_decision().censored),
            {
                f"{AMBULANCE_ID_PREFIX}0",
                f"{AMBULANCE_ID_PREFIX}1",
            },
        )

    def test_due_spawn_is_not_delayed_by_controller_dependent_arrival(self):
        system, traci = make_system(
            max_ambulances=2,
            max_active=1,
            interval=10.0,
            last_spawn_buffer=0.0,
        )
        self._depart(system, traci)
        second_spawn = system.schedule[1].spawn_time
        system.before_simulation_step({}, second_spawn, None)
        self.assertEqual(len(system.records), 2)
        self.assertEqual(
            system.records[f"{AMBULANCE_ID_PREFIX}1"].requested_departure,
            second_spawn,
        )

    def test_reroute_jitter_is_per_ambulance_not_shared_rng_state(self):
        first, _traci = make_system()
        expected = first._next_reroute_delay(1, 0)
        for check_index in range(20):
            first._next_reroute_delay(0, check_index)
        self.assertEqual(first._next_reroute_delay(1, 0), expected)

        second, _traci = make_system()
        self.assertEqual(
            second._next_reroute_delay(1, 0), expected
        )


if __name__ == "__main__":
    unittest.main()
