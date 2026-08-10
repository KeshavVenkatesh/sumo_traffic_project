import unittest

from ambulance_curriculum import curriculum_demand_routes


class AmbulanceCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"route_file": "light.rou.xml", "intensity": 4.0},
            {"route_file": "medium.rou.xml", "intensity": 8.0},
            {"route_file": "heavy.rou.xml", "intensity": 12.0},
            {"route_file": "legacy.rou.xml", "intensity": None},
        ]

    def test_uses_light_and_medium_demand_first(self):
        self.assertEqual(
            curriculum_demand_routes(self.records, 0.10),
            [
                "light.rou.xml",
                "medium.rou.xml",
                "legacy.rou.xml",
            ],
        )

    def test_uses_every_demand_level_in_the_middle(self):
        self.assertEqual(
            curriculum_demand_routes(self.records, 0.40),
            [record["route_file"] for record in self.records],
        )

    def test_uses_medium_and_heavy_demand_late(self):
        self.assertEqual(
            curriculum_demand_routes(self.records, 0.90),
            [
                "medium.rou.xml",
                "heavy.rou.xml",
                "legacy.rou.xml",
            ],
        )


if __name__ == "__main__":
    unittest.main()
