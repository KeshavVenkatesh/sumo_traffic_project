import json
import tempfile
import unittest
from pathlib import Path

from analyze_comprehensive_evaluation import comparisons_for_controllers
from launch_comprehensive_evaluation import (
    file_sha256,
    verify_split_protocol_lock,
)


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

    def test_final_evaluation_is_bound_to_locked_test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "seed": 7,
                        "maps": [
                            {
                                "name": "held_out",
                                "split": "test",
                                "net_file": str(root / "held_out.net.xml"),
                            },
                            {
                                "name": "development",
                                "split": "train",
                                "net_file": str(root / "development.net.xml"),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            lock = root / "lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "status": "valid",
                        "new_corpus": {"final_test_maps": ["held_out"]},
                        "generated_manifest": {
                            "sha256": file_sha256(manifest)
                        },
                    }
                ),
                encoding="utf-8",
            )
            verified = verify_split_protocol_lock(
                str(lock),
                str(manifest),
                {"test"},
                {"held_out": root / "held_out.net.xml"},
            )
            self.assertEqual(verified["final_test_maps"], ["held_out"])

            with self.assertRaisesRegex(RuntimeError, "only frozen final-test"):
                verify_split_protocol_lock(
                    str(lock),
                    str(manifest),
                    {"test"},
                    {
                        "held_out": root / "held_out.net.xml",
                        "legacy": root / "legacy.net.xml",
                    },
                )


if __name__ == "__main__":
    unittest.main()
