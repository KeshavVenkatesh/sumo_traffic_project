#!/usr/bin/env python3
"""Strict checkpoint contract for the schema-v5 ambulance override."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ambulance_emergency import (
    CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS,
    EMERGENCY_GLOBAL_FEATURE_NAMES,
    EMERGENCY_MOVEMENT_FEATURE_NAMES,
    EMERGENCY_PHASE_FEATURE_NAMES,
    MIN_EMERGENCY_DOWNSTREAM_SPACE,
)
from map_agnostic_tls import (
    DEFAULT_REQUIRED_EXIT_GAP_METERS,
    GLOBAL_FEATURE_NAMES,
    MAX_MOVEMENTS,
    MAX_PHASES,
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_NAMES,
)


AMBULANCE_SCHEMA_VERSION = 5


class AmbulanceCheckpointCompatibilityError(RuntimeError):
    pass


def _base(path: str | Path) -> Path:
    value = Path(path)
    return value.with_suffix("") if value.suffix in {".pt", ".zip"} else value


def emergency_model_path(path: str | Path) -> Path:
    return _base(path).with_suffix(".pt")


def emergency_contract_path(path: str | Path) -> Path:
    base = _base(path)
    return base.parent / f"{base.name}_contract.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_sha256(contract: "AmbulanceCheckpointContract") -> str:
    encoded = json.dumps(
        contract.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AmbulanceCheckpointContract:
    schema_version: int
    controller_family: str
    base_checkpoint_sha256: str
    decision_seconds: float
    step_length_seconds: float
    minimum_green_seconds: float
    maximum_green_seconds: float
    max_movements: int
    max_phases: int
    base_movement_features: tuple[str, ...]
    base_phase_features: tuple[str, ...]
    base_global_features: tuple[str, ...]
    emergency_movement_features: tuple[str, ...]
    emergency_phase_features: tuple[str, ...]
    emergency_global_features: tuple[str, ...]
    emergency_embed_dim: int
    emergency_graph_layers: int
    residual_bound: float
    authority: float
    ambulance_system: Mapping[str, Any]
    emergency_observation: Mapping[str, Any]
    corridor: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        base_checkpoint: str | Path,
        decision_seconds: float,
        step_length_seconds: float,
        minimum_green_seconds: float,
        maximum_green_seconds: float,
        emergency_embed_dim: int,
        emergency_graph_layers: int,
        residual_bound: float,
        authority: float,
        ambulance_system: Mapping[str, Any],
        emergency_observation: Mapping[str, Any],
        corridor: Mapping[str, Any],
    ) -> "AmbulanceCheckpointContract":
        return cls(
            schema_version=AMBULANCE_SCHEMA_VERSION,
            controller_family="frozen_schema_v3_plus_emergency_residual",
            base_checkpoint_sha256=sha256_file(base_checkpoint),
            decision_seconds=float(decision_seconds),
            step_length_seconds=float(step_length_seconds),
            minimum_green_seconds=float(minimum_green_seconds),
            maximum_green_seconds=float(maximum_green_seconds),
            max_movements=MAX_MOVEMENTS,
            max_phases=MAX_PHASES,
            base_movement_features=tuple(MOVEMENT_FEATURE_NAMES),
            base_phase_features=tuple(PHASE_FEATURE_NAMES),
            base_global_features=tuple(GLOBAL_FEATURE_NAMES),
            emergency_movement_features=tuple(
                EMERGENCY_MOVEMENT_FEATURE_NAMES
            ),
            emergency_phase_features=tuple(
                EMERGENCY_PHASE_FEATURE_NAMES
            ),
            emergency_global_features=tuple(
                EMERGENCY_GLOBAL_FEATURE_NAMES
            ),
            emergency_embed_dim=int(emergency_embed_dim),
            emergency_graph_layers=int(emergency_graph_layers),
            residual_bound=float(residual_bound),
            authority=float(authority),
            ambulance_system=dict(ambulance_system),
            emergency_observation=dict(emergency_observation),
            corridor=dict(corridor),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "base_movement_features",
            "base_phase_features",
            "base_global_features",
            "emergency_movement_features",
            "emergency_phase_features",
            "emergency_global_features",
        ):
            payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AmbulanceCheckpointContract":
        values = dict(payload)
        for key in (
            "base_movement_features",
            "base_phase_features",
            "base_global_features",
            "emergency_movement_features",
            "emergency_phase_features",
            "emergency_global_features",
        ):
            values[key] = tuple(str(value) for value in values[key])
        return cls(**values)

    def validate(
        self,
        *,
        base_checkpoint: str | Path,
        decision_seconds: float,
        step_length_seconds: float,
        minimum_green_seconds: float,
        maximum_green_seconds: float,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != AMBULANCE_SCHEMA_VERSION:
            errors.append(
                f"schema={self.schema_version}, expected={AMBULANCE_SCHEMA_VERSION}"
            )
        if self.controller_family != (
            "frozen_schema_v3_plus_emergency_residual"
        ):
            errors.append("controller family differs")
        if self.base_checkpoint_sha256 != sha256_file(base_checkpoint):
            errors.append("frozen base checkpoint hash differs")
        expected_features = (
            (self.max_movements, MAX_MOVEMENTS, "max_movements"),
            (self.max_phases, MAX_PHASES, "max_phases"),
            (
                self.base_movement_features,
                tuple(MOVEMENT_FEATURE_NAMES),
                "base movement feature order",
            ),
            (
                self.base_phase_features,
                tuple(PHASE_FEATURE_NAMES),
                "base phase feature order",
            ),
            (
                self.base_global_features,
                tuple(GLOBAL_FEATURE_NAMES),
                "base global feature order",
            ),
            (
                self.emergency_movement_features,
                tuple(EMERGENCY_MOVEMENT_FEATURE_NAMES),
                "emergency movement feature order",
            ),
            (
                self.emergency_phase_features,
                tuple(EMERGENCY_PHASE_FEATURE_NAMES),
                "emergency phase feature order",
            ),
            (
                self.emergency_global_features,
                tuple(EMERGENCY_GLOBAL_FEATURE_NAMES),
                "emergency global feature order",
            ),
        )
        for actual, expected, name in expected_features:
            if actual != expected:
                errors.append(f"{name} differs")
        cadence = (
            (self.decision_seconds, float(decision_seconds), "decision_seconds"),
            (
                self.step_length_seconds,
                float(step_length_seconds),
                "step_length_seconds",
            ),
            (
                self.minimum_green_seconds,
                float(minimum_green_seconds),
                "minimum_green_seconds",
            ),
            (
                self.maximum_green_seconds,
                float(maximum_green_seconds),
                "maximum_green_seconds",
            ),
        )
        for saved, runtime, name in cadence:
            if abs(float(saved) - runtime) > 1e-9:
                errors.append(
                    f"{name}: checkpoint={saved:g}, runtime={runtime:g}"
                )
        if not self.ambulance_system:
            errors.append("ambulance routing contract is empty")
        if not self.emergency_observation:
            errors.append("emergency observation contract is empty")
        if not self.corridor:
            errors.append("corridor contract is empty")
        safety_contract = (
            (
                self.corridor.get(
                    "min_emergency_downstream_space"
                ),
                MIN_EMERGENCY_DOWNSTREAM_SPACE,
                "minimum emergency downstream space",
            ),
            (
                self.corridor.get(
                    "budget_close_eta_exception_seconds"
                ),
                CLOSE_AMBULANCE_BUDGET_EXCEPTION_SECONDS,
                "close-ambulance budget exception",
            ),
            (
                self.corridor.get("required_exit_gap_meters"),
                DEFAULT_REQUIRED_EXIT_GAP_METERS,
                "required exit gap",
            ),
        )
        for saved, expected, name in safety_contract:
            try:
                matches = (
                    math.isfinite(float(saved))
                    and abs(float(saved) - expected) <= 1e-9
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(f"{name} differs")
        if self.corridor.get("strict_exit_space") is not True:
            errors.append("strict exit-space shield is not enabled")
        if (
            self.corridor.get("allow_unsafe_hard_max_fallback")
            is not False
        ):
            errors.append("unsafe hard-max fallback is not disabled")
        try:
            time_to_teleport = int(
                self.ambulance_system.get("time_to_teleport", 0)
            )
        except (TypeError, ValueError):
            time_to_teleport = 0
        if time_to_teleport != -1:
            errors.append("SUMO time-to-teleport must be disabled")
        try:
            routing_step = float(
                self.ambulance_system.get(
                    "step_length_seconds", float("nan")
                )
            )
        except (TypeError, ValueError):
            routing_step = float("nan")
        if (
            not math.isfinite(routing_step)
            or abs(routing_step - self.step_length_seconds) > 1e-9
        ):
            errors.append("ambulance routing step length differs")
        try:
            queue_lead = float(
                self.ambulance_system.get(
                    "spawn_queue_lead_seconds", float("nan")
                )
            )
        except (TypeError, ValueError):
            queue_lead = float("nan")
        if (
            not math.isfinite(queue_lead)
            or abs(queue_lead - self.step_length_seconds) > 1e-9
        ):
            errors.append("spawn queue lead must equal one SUMO step")
        if (
            self.ambulance_system.get("terminal_censor_penalty")
            is not True
        ):
            errors.append("terminal censor penalty is not enabled")
        if (
            self.ambulance_system.get("ordinary_delay_metric")
            != "mean_time_loss_all_departed_s"
        ):
            errors.append("ordinary delay metric is not censor-resistant")
        if self.ambulance_system.get("blue_light_device") is not False:
            errors.append("SUMO blue-light device must remain disabled")
        if self.ambulance_system.get("obeys_signals") is not True:
            errors.append("ambulance signal compliance differs")
        if self.ambulance_system.get("reroute_jitter_stream") != (
            "per_ambulance_sha256"
        ):
            errors.append("reroute jitter stream differs")
        if self.corridor.get("recovery_teacher") != (
            "normalized_max_pressure"
        ):
            errors.append("recovery teacher differs")
        if errors:
            raise AmbulanceCheckpointCompatibilityError(
                "Incompatible ambulance checkpoint: " + "; ".join(errors)
            )


def save_emergency_checkpoint(
    path: str | Path,
    *,
    state_dict: Mapping[str, Any],
    contract: AmbulanceCheckpointContract,
    training_state: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    model_path = emergency_model_path(path)
    contract_path = emergency_contract_path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    contract_temporary = contract_path.with_suffix(
        contract_path.suffix + ".tmp"
    )
    torch.save(
        {
            "schema_version": AMBULANCE_SCHEMA_VERSION,
            "contract_sha256": _contract_sha256(contract),
            "state_dict": dict(state_dict),
            "emergency_embed_dim": contract.emergency_embed_dim,
            "emergency_graph_layers": contract.emergency_graph_layers,
            "residual_bound": contract.residual_bound,
            "authority": contract.authority,
            "training_state": dict(training_state or {}),
        },
        model_temporary,
    )
    contract_temporary.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_temporary.replace(model_path)
    contract_temporary.replace(contract_path)
    return model_path, contract_path


def load_emergency_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | Path,
    decision_seconds: float,
    step_length_seconds: float,
    minimum_green_seconds: float,
    maximum_green_seconds: float,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], AmbulanceCheckpointContract]:
    model_path = emergency_model_path(path)
    contract_path = emergency_contract_path(path)
    if not model_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError(
            f"Missing emergency checkpoint or contract: {model_path}, "
            f"{contract_path}"
        )
    contract = AmbulanceCheckpointContract.from_dict(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    contract.validate(
        base_checkpoint=base_checkpoint,
        decision_seconds=decision_seconds,
        step_length_seconds=step_length_seconds,
        minimum_green_seconds=minimum_green_seconds,
        maximum_green_seconds=maximum_green_seconds,
    )
    payload = torch.load(model_path, map_location=device)
    if int(payload.get("schema_version", -1)) != AMBULANCE_SCHEMA_VERSION:
        raise AmbulanceCheckpointCompatibilityError(
            f"Checkpoint payload schema is {payload.get('schema_version')}"
        )
    payload_contract = (
        (
            int(payload.get("emergency_embed_dim", -1)),
            contract.emergency_embed_dim,
            "emergency_embed_dim",
        ),
        (
            int(payload.get("emergency_graph_layers", -1)),
            contract.emergency_graph_layers,
            "emergency_graph_layers",
        ),
        (
            float(payload.get("residual_bound", float("nan"))),
            contract.residual_bound,
            "residual_bound",
        ),
        (
            float(payload.get("authority", float("nan"))),
            contract.authority,
            "authority",
        ),
    )
    mismatches: list[str] = []
    if payload.get("contract_sha256") != _contract_sha256(contract):
        mismatches.append("contract_sha256")
    for saved, expected, name in payload_contract:
        if isinstance(saved, float):
            matches = (
                math.isfinite(saved)
                and abs(saved - float(expected)) <= 1e-9
            )
        else:
            matches = saved == expected
        if not matches:
            mismatches.append(name)
    if mismatches:
        raise AmbulanceCheckpointCompatibilityError(
            "Checkpoint payload and contract disagree: "
            + ", ".join(mismatches)
        )
    return dict(payload), contract


__all__ = [
    "AMBULANCE_SCHEMA_VERSION",
    "AmbulanceCheckpointCompatibilityError",
    "AmbulanceCheckpointContract",
    "emergency_contract_path",
    "emergency_model_path",
    "load_emergency_checkpoint",
    "save_emergency_checkpoint",
    "sha256_file",
]
