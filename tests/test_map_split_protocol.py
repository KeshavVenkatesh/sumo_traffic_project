from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_map_split_protocol import (
    ProtocolError,
    build_protocol_lock,
    file_sha256,
    verify_training_protocol_lock,
)


ROOT = Path(__file__).resolve().parents[1]


class MapSplitProtocolTests(unittest.TestCase):
    def test_repository_v4_protocol_is_new_balanced_and_leakage_free(self):
        payload = build_protocol_lock(
            ROOT / "map_corpus_regions_v4.json",
            ROOT / "map_corpus_regions.json",
        )
        self.assertEqual(payload["status"], "valid")
        self.assertEqual(
            payload["new_corpus"]["split_counts"],
            {"train": 32, "validation": 8, "test": 8},
        )
        self.assertEqual(len(payload["new_corpus"]["final_test_maps"]), 8)
        self.assertTrue(payload["checks"]["region_names_disjoint"])
        self.assertGreaterEqual(
            payload["checks"]["minimum_observed_test_center_distance_km"],
            payload["checks"]["minimum_required_test_center_distance_km"],
        )

    def test_geographically_relabelled_old_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_config = {
                "regions": [
                    {
                        "name": "historical",
                        "center": [40.0, -75.0],
                        "half_size": [0.01, 0.01],
                        "split": "train",
                    }
                ]
            }
            new_config = {
                "protocol": {
                    "corpus_id": "bad",
                    "seed": 2,
                    "old_schema_v3_config": "old.json",
                    "old_schema_v3_seed": 1,
                    "minimum_test_center_distance_km": 75.0,
                    "expected_split_counts": {
                        "train": 1,
                        "validation": 1,
                        "test": 1,
                    },
                },
                "regions": [
                    {
                        "name": "new_train",
                        "center": [10.0, 10.0],
                        "half_size": [0.01, 0.01],
                        "split": "train",
                    },
                    {
                        "name": "new_validation",
                        "center": [20.0, 20.0],
                        "half_size": [0.01, 0.01],
                        "split": "validation",
                    },
                    {
                        "name": "renamed_test",
                        "center": [40.0, -75.0],
                        "half_size": [0.01, 0.01],
                        "split": "test",
                    },
                ],
            }
            old_path = root / "old.json"
            new_path = root / "new.json"
            old_path.write_text(json.dumps(old_config), encoding="utf-8")
            new_path.write_text(json.dumps(new_config), encoding="utf-8")

            with self.assertRaisesRegex(
                ProtocolError, "final-test map renamed_test"
            ):
                build_protocol_lock(new_path, old_path)

    def test_post_generation_check_rejects_missing_frozen_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_config = {
                "protocol": {
                    "corpus_id": "missing",
                    "seed": 2,
                    "old_schema_v3_config": "old.json",
                    "old_schema_v3_seed": 1,
                    "minimum_test_center_distance_km": 75.0,
                    "expected_split_counts": {
                        "train": 1,
                        "validation": 1,
                        "test": 1,
                    },
                },
                "regions": [
                    {"name": "a", "center": [0.0, 0.0], "split": "train"},
                    {
                        "name": "b",
                        "center": [10.0, 10.0],
                        "split": "validation",
                    },
                    {"name": "c", "center": [20.0, 20.0], "split": "test"},
                ],
            }
            old_config = {
                "regions": [
                    {
                        "name": "old",
                        "center": [-20.0, -20.0],
                        "split": "train",
                    }
                ]
            }
            new_path = root / "new.json"
            old_path = root / "old.json"
            manifest_path = root / "manifest.json"
            new_path.write_text(json.dumps(new_config), encoding="utf-8")
            old_path.write_text(json.dumps(old_config), encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"seed": 2, "maps": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ProtocolError, "did not accept every frozen map"
            ):
                build_protocol_lock(new_path, old_path, manifest_path)

    def test_training_is_restricted_to_locked_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "maps": [
                            {"name": "train_map", "split": "train"},
                            {"name": "validation_map", "split": "validation"},
                            {"name": "test_map", "split": "test"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "status": "valid",
                        "new_corpus": {
                            "train_maps": ["train_map"],
                            "validation_maps": ["validation_map"],
                            "final_test_maps": ["test_map"],
                        },
                        "generated_manifest": {
                            "sha256": file_sha256(manifest_path)
                        },
                    }
                ),
                encoding="utf-8",
            )

            verified = verify_training_protocol_lock(
                str(lock_path), str(manifest_path), {"train"}
            )
            self.assertEqual(verified["training_maps"], ["train_map"])
            self.assertEqual(
                verified["excluded_final_test_maps"], ["test_map"]
            )
            with self.assertRaisesRegex(ProtocolError, "requires --splits train"):
                verify_training_protocol_lock(
                    str(lock_path), str(manifest_path), {"test"}
                )


if __name__ == "__main__":
    unittest.main()
