from __future__ import annotations

import ast
import unittest
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "detector_realistic_multiagent_worker.py"
)


def method_source(class_name: str, method_name: str) -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == method_name:
                        return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"Could not find {class_name}.{method_name}")


class DetectorWorkerInformationBoundaryTests(unittest.TestCase):
    def test_episode_start_does_not_construct_the_oracle_adapter(self):
        start = method_source("DetectorRealisticAllTLSEpisode", "_start_episode")
        self.assertNotIn("super()._start_episode", start)
        self.assertNotIn("MapTrafficSnapshot", start)
        self.assertNotIn("getRoute", start)
        self.assertNotIn("getWaitingTime", start)
        self.assertIn("DetectorTrafficSnapshot", start)

    def test_training_reward_uses_aggregate_detector_departures(self):
        step = method_source("DetectorRealisticAllTLSEpisode", "step")
        self.assertIn("last_detected_departures", step)
        self.assertNotIn("vehicle_ids", step)


if __name__ == "__main__":
    unittest.main()
