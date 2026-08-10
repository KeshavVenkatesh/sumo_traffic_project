import unittest
from types import SimpleNamespace

import numpy as np

from ambulance_multiagent_worker import (
    EmergencyAllTLSEpisode,
    _ClearanceDeltaView,
)


class RecoveryAccountingTests(unittest.TestCase):
    @staticmethod
    def episode():
        episode = object.__new__(EmergencyAllTLSEpisode)
        episode.adapters = [SimpleNamespace(tls_id="tls-a")]
        episode._ordinary_queue_baselines = {}
        episode._intervention_queue_baselines = {}
        episode._pending_recoveries = {}
        episode._completed_recovery_seconds = []
        return episode

    def test_base_action_creates_no_recovery_event(self):
        episode = self.episode()
        episode._remember_interventions(
            np.asarray([1]),
            np.asarray([1]),
            {"tls-a": 7.0},
        )

        clearances = episode._intervened_clearances(
            [("ambulance-0", "tls-a")]
        )
        episode._start_recoveries(clearances, 10.0)

        self.assertEqual(clearances, [])
        self.assertEqual(episode._pending_recoveries, {})
        self.assertEqual(
            episode._recovery_summary(),
            {
                "mean_seconds": 0.0,
                "p95_seconds": 0.0,
                "completed_events": 0,
                "unrecovered_events": 0,
            },
        )

    def test_divergence_uses_pre_intervention_local_baseline(self):
        episode = self.episode()
        episode._ordinary_queue_baselines["tls-a"] = 4.0

        episode._remember_interventions(
            np.asarray([2]),
            np.asarray([1]),
            {"tls-a": 9.0},
        )

        self.assertEqual(
            episode._intervention_queue_baselines,
            {"tls-a": 4.0},
        )

        clearances = episode._intervened_clearances(
            [("ambulance-0", "tls-a")]
        )
        episode._start_recoveries(clearances, 10.0)

        self.assertEqual(
            episode._pending_recoveries,
            {"tls-a": {"start": 10.0, "threshold": 5.0}},
        )

    def test_overlapping_clearances_are_one_tls_event(self):
        episode = self.episode()
        episode._intervention_queue_baselines["tls-a"] = 4.0

        clearances = episode._intervened_clearances(
            [
                ("ambulance-0", "tls-a"),
                ("ambulance-1", "tls-a"),
            ]
        )
        self.assertEqual(clearances, [("ambulance-0", "tls-a")])

        episode._start_recoveries(clearances, 10.0)
        episode._intervention_queue_baselines["tls-a"] = 8.0
        episode._start_recoveries(
            [("ambulance-1", "tls-a")],
            20.0,
        )

        self.assertEqual(len(episode._pending_recoveries), 1)
        self.assertEqual(
            episode._pending_recoveries["tls-a"],
            {"start": 10.0, "threshold": 9.0},
        )

    def test_recovery_uses_only_the_matching_tls_queue(self):
        episode = self.episode()
        episode._pending_recoveries = {
            "tls-a": {"start": 10.0, "threshold": 5.0}
        }

        episode._advance_recoveries(
            {"tls-a": 8.0, "tls-b": 0.0},
            20.0,
        )
        self.assertIn("tls-a", episode._pending_recoveries)

        episode._advance_recoveries({"tls-a": 5.0}, 30.0)
        self.assertEqual(episode._pending_recoveries, {})
        self.assertEqual(
            episode._completed_recovery_seconds,
            [20.0],
        )

    def test_clearance_view_forwards_non_clearance_fields(self):
        source = SimpleNamespace(
            cleared_tls=[("ambulance-0", "tls-a")],
            arrived=["ambulance-0"],
        )
        view = _ClearanceDeltaView(source, [])

        self.assertEqual(view.cleared_tls, [])
        self.assertEqual(view.arrived, ["ambulance-0"])


if __name__ == "__main__":
    unittest.main()
