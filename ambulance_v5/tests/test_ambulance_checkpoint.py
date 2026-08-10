from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from ambulance_checkpoint import (
        AmbulanceCheckpointCompatibilityError,
        AmbulanceCheckpointContract,
        load_emergency_checkpoint,
        save_emergency_checkpoint,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class AmbulanceCheckpointTests(unittest.TestCase):
    def make_contract(self, base_checkpoint):
        return AmbulanceCheckpointContract.create(
            base_checkpoint=base_checkpoint,
            decision_seconds=10.0,
            step_length_seconds=1.0,
            minimum_green_seconds=6.0,
            maximum_green_seconds=55.0,
            emergency_embed_dim=16,
            emergency_graph_layers=1,
            residual_bound=4.0,
            authority=1.0,
            ambulance_system={
                "routing_mode": "traffic_aware",
                "step_length_seconds": 1.0,
                "min_route_tls": 2,
                "time_to_teleport": -1,
                "spawn_queue_lead_seconds": 1.0,
                "terminal_censor_penalty": True,
                "ordinary_delay_metric": (
                    "mean_time_loss_all_departed_s"
                ),
                "blue_light_device": False,
                "obeys_signals": True,
                "reroute_jitter_stream": "per_ambulance_sha256",
            },
            emergency_observation={
                "relevance_distance_meters": 650.0,
                "route_horizon": 3,
            },
            corridor={
                "recovery_seconds": 30.0,
                "max_preemption_seconds": 45.0,
                "min_emergency_downstream_space": 0.08,
                "budget_close_eta_exception_seconds": 8.0,
                "strict_exit_space": True,
                "required_exit_gap_meters": 18.0,
                "allow_unsafe_hard_max_fallback": False,
                "recovery_teacher": "normalized_max_pressure",
            },
        )

    def test_round_trip_binds_base_checkpoint_and_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.zip"
            base.write_bytes(b"frozen base")
            contract = self.make_contract(base)
            save_emergency_checkpoint(
                root / "emergency",
                state_dict={"weight": torch.ones(2)},
                contract=contract,
            )
            payload, loaded = load_emergency_checkpoint(
                root / "emergency",
                base_checkpoint=base,
                decision_seconds=10.0,
                step_length_seconds=1.0,
                minimum_green_seconds=6.0,
                maximum_green_seconds=55.0,
            )
            self.assertEqual(loaded.schema_version, 5)
            self.assertTrue(
                torch.equal(
                    payload["state_dict"]["weight"],
                    torch.ones(2),
                )
            )

    def test_rejects_a_different_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.zip"
            base.write_bytes(b"frozen base")
            other = root / "other.zip"
            other.write_bytes(b"different base")
            contract = self.make_contract(base)
            save_emergency_checkpoint(
                root / "emergency",
                state_dict={"weight": torch.ones(1)},
                contract=contract,
            )
            with self.assertRaises(
                AmbulanceCheckpointCompatibilityError
            ):
                load_emergency_checkpoint(
                    root / "emergency",
                    base_checkpoint=other,
                    decision_seconds=10.0,
                    step_length_seconds=1.0,
                    minimum_green_seconds=6.0,
                    maximum_green_seconds=55.0,
                )

    def test_rejects_payload_contract_shape_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.zip"
            base.write_bytes(b"frozen base")
            contract = self.make_contract(base)
            model_path, _contract_path = save_emergency_checkpoint(
                root / "emergency",
                state_dict={"weight": torch.ones(1)},
                contract=contract,
            )
            payload = torch.load(model_path)
            payload["emergency_embed_dim"] = 99
            torch.save(payload, model_path)
            with self.assertRaises(
                AmbulanceCheckpointCompatibilityError
            ):
                load_emergency_checkpoint(
                    root / "emergency",
                    base_checkpoint=base,
                    decision_seconds=10.0,
                    step_length_seconds=1.0,
                    minimum_green_seconds=6.0,
                    maximum_green_seconds=55.0,
                )

    def test_rejects_a_tampered_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.zip"
            base.write_bytes(b"frozen base")
            contract = self.make_contract(base)
            _model_path, contract_path = save_emergency_checkpoint(
                root / "emergency",
                state_dict={"weight": torch.ones(1)},
                contract=contract,
            )
            payload = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
            payload["corridor"]["recovery_seconds"] = 999.0
            contract_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaises(
                AmbulanceCheckpointCompatibilityError
            ):
                load_emergency_checkpoint(
                    root / "emergency",
                    base_checkpoint=base,
                    decision_seconds=10.0,
                    step_length_seconds=1.0,
                    minimum_green_seconds=6.0,
                    maximum_green_seconds=55.0,
                )


if __name__ == "__main__":
    unittest.main()
