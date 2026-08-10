from __future__ import annotations

import random
import unittest

from train_map_agnostic_multimap import (
    choose_topology_balanced,
    topology_bucket,
)


class MultiMapSamplingTests(unittest.TestCase):
    def test_topology_bucket_is_coarse_and_map_size_independent(self):
        small = {
            "incoming_edges": 3,
            "outgoing_edges": 3,
            "movements": 6,
            "phases": 3,
        }
        large_lanes_same_shape = dict(small, tls_id="other")
        self.assertEqual(topology_bucket(small), topology_bucket(large_lanes_same_shape))

    def test_balanced_sampler_reaches_rare_bucket_before_repeating_common(self):
        common = [
            {
                "tls_id": f"four_{index}",
                "incoming_edges": 4,
                "outgoing_edges": 4,
                "movements": 8,
                "phases": 4,
            }
            for index in range(12)
        ]
        rare = {
            "tls_id": "rare_three_way",
            "incoming_edges": 3,
            "outgoing_edges": 3,
            "movements": 5,
            "phases": 2,
        }
        selected = choose_topology_balanced(common + [rare], 2, random.Random(7))
        self.assertIn("rare_three_way", {record["tls_id"] for record in selected})

    def test_balanced_sampler_is_reproducible(self):
        records = [
            {
                "tls_id": str(index),
                "incoming_edges": 3 + index % 2,
                "outgoing_edges": 3 + index % 2,
                "movements": 5 + index,
                "phases": 2 + index % 3,
            }
            for index in range(8)
        ]
        left = choose_topology_balanced(records, 12, random.Random(99))
        right = choose_topology_balanced(records, 12, random.Random(99))
        self.assertEqual(
            [record["tls_id"] for record in left],
            [record["tls_id"] for record in right],
        )


if __name__ == "__main__":
    unittest.main()
