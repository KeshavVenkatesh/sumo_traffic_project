import unittest

from analyze_comprehensive_evaluation import comparisons_for_controllers


class ComprehensiveFourWayContractTests(unittest.TestCase):
    def test_schema_v3_adds_direct_paired_comparisons(self):
        comparisons = set(
            comparisons_for_controllers(
                {"native_sumo", "max_pressure", "all_model", "schema_v3"}
            )
        )

        self.assertIn(
            ("all_model", "schema_v3", "detector_v4_vs_schema_v3"),
            comparisons,
        )
        self.assertIn(
            ("schema_v3", "max_pressure", "schema_v3_vs_max_pressure"),
            comparisons,
        )
        self.assertIn(
            ("schema_v3", "native_sumo", "schema_v3_vs_native"),
            comparisons,
        )


if __name__ == "__main__":
    unittest.main()
