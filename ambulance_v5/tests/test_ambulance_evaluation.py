from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from evaluate_ambulance_system import (
    ABLATIONS,
    build_summary,
    load_paired_demand,
    verify_pairing,
)
from fixed_demand import enforce_fixed_demand_vehicle_type, sha256_file


def evaluation_record(
    ablation: str,
    *,
    trip_time: float,
    ordinary_delay: float,
    ordinary_arrived: int,
):
    ambulance_record = {
        "status": "arrived",
        "requested_departure": 0.0,
        "actual_departure": 0.0,
        "end_time": trip_time,
        "time_loss": max(0.0, trip_time - 50.0),
        "stopped_seconds": 2.0,
    }
    return {
        "scenario_key": "map|medium|1001|routehash",
        "map_id": "map",
        "net_file": "/tmp/map.net.xml",
        "scenario": "medium",
        "seed": 1001,
        "route_file": "/tmp/seed_1001.rou.xml",
        "route_sha256": "routehash",
        "network_sha256": "networkhash",
        "scheduled_ordinary_vehicles": 100,
        "ablation": ablation,
        "schedule_sha256": "schedulehash",
        "ambulance": {
            "records": [ambulance_record],
            "mean_trip_time_s": trip_time,
            "mean_response_time_s": trip_time,
            "collision_vehicle_ids": [],
            "teleported_vehicle_ids": [],
        },
        "ordinary_traffic": {
            "arrived_total": ordinary_arrived,
            "departed_total": 100,
            "mean_time_loss_s": ordinary_delay,
            "mean_queue_vehicles": 5.0,
            "mean_speed_mps": 9.0,
        },
        "recovery": {
            "mean_seconds": 12.0,
            "completed_events": 1,
            "unrecovered_events": 0,
        },
    }


class EvaluationMetricTests(unittest.TestCase):
    def records(self):
        values = {
            "free_flow_route_base_signals": (100.0, 10.0, 100),
            "traffic_aware_route_base_signals": (90.0, 10.0, 100),
            "free_flow_route_learned_signals": (80.0, 10.2, 100),
            "traffic_aware_route_learned_signals": (70.0, 10.4, 99),
            "traffic_aware_route_deterministic_preemption": (
                75.0,
                10.3,
                99,
            ),
        }
        return [
            evaluation_record(
                name,
                trip_time=values[name][0],
                ordinary_delay=values[name][1],
                ordinary_arrived=values[name][2],
            )
            for name in (item["name"] for item in ABLATIONS)
        ]

    def test_summary_separates_routing_signal_and_combined_gains(self):
        records = self.records()
        verify_pairing(records)
        summary = build_summary(
            records,
            types.SimpleNamespace(
                ordinary_delay_budget_percent=5.0,
                throughput_budget_percent=2.0,
            ),
        )
        self.assertTrue(summary["eligible"])
        self.assertAlmostEqual(
            summary["routing_only_gain_percent"], 10.0
        )
        self.assertAlmostEqual(
            summary["ambulance_gain_percent"],
            100.0 * (90.0 - 70.0) / 90.0,
        )
        self.assertAlmostEqual(
            summary[
                "combined_routing_and_signal_gain_percent"
            ],
            30.0,
        )
        self.assertAlmostEqual(
            summary["ordinary_delay_change_percent"], 4.0
        )
        self.assertAlmostEqual(
            summary["throughput_change_percent"], -1.0
        )

    def test_pairing_rejects_a_different_ambulance_schedule(self):
        records = self.records()
        records[-1]["schedule_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            verify_pairing(records)

    def test_checkpoint_is_not_promoted_below_deterministic_preemption(self):
        records = self.records()
        deterministic = next(
            record
            for record in records
            if record["ablation"]
            == "traffic_aware_route_deterministic_preemption"
        )
        deterministic_record = deterministic["ambulance"]["records"][0]
        deterministic_record["end_time"] = 65.0
        deterministic["ambulance"]["mean_trip_time_s"] = 65.0
        deterministic["ambulance"]["mean_response_time_s"] = 65.0
        summary = build_summary(
            records,
            types.SimpleNamespace(
                ordinary_delay_budget_percent=5.0,
                throughput_budget_percent=2.0,
            ),
        )
        self.assertFalse(summary["eligible"])
        self.assertFalse(
            summary["gates"][
                "deterministic_preemption_not_better"
            ]
        )


class DemandManifestTests(unittest.TestCase):
    def test_rejects_an_unchecksummed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = root / "map.net.xml"
            net.write_text("<net/>", encoding="utf-8")
            route = root / "demand.rou.xml"
            route.write_text("<routes/>", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                (
                    '{"schema_version": 1, "episode_seconds": 1200, '
                    '"routes": [{"net_file": "map.net.xml", '
                    '"route_file": "demand.rou.xml", "seed": 1001}]}'
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_paired_demand(
                    manifest,
                    [net],
                    {1001},
                    1200.0,
                )

    def test_loads_exact_requested_seed_and_rejects_bad_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = root / "map.net.xml"
            net.write_text("<net/>", encoding="utf-8")
            route = root / "demand.rou.xml"
            route.write_text(
                '<routes><vehicle id="v0" depart="0"/></routes>',
                encoding="utf-8",
            )
            enforce_fixed_demand_vehicle_type(route)
            manifest = root / "manifest.json"
            network_hash = sha256_file(net)
            route_hash = sha256_file(route)
            manifest.write_text(
                (
                    '{"schema_version": 2, '
                    '"episode_seconds": 1200, "routes": ['
                    '{"net_file": "map.net.xml", '
                    '"route_file": "demand.rou.xml", '
                    f'"network_sha256": "{network_hash}", '
                    f'"route_sha256": "{route_hash}", '
                    '"seed": 1001, "scheduled_records": 1}]}'
                ),
                encoding="utf-8",
            )
            records = load_paired_demand(
                manifest,
                [net],
                {1001},
                1200.0,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].scheduled_vehicles, 1)
            self.assertEqual(len(records[0].route_sha256), 64)

            manifest.write_text(
                (
                    '{"schema_version": 2, '
                    '"episode_seconds": 1200, "routes": ['
                    '{"net_file": "map.net.xml", '
                    '"route_file": "demand.rou.xml", '
                    f'"network_sha256": "{network_hash}", '
                    '"route_sha256": "bad", '
                    '"seed": 1001, "scheduled_records": 1}]}'
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_paired_demand(
                    manifest,
                    [net],
                    {1001},
                    1200.0,
                )


if __name__ == "__main__":
    unittest.main()
