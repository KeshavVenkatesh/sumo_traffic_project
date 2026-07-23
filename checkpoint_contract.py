#!/usr/bin/env python3
"""Strict runtime contract for portable traffic-signal checkpoints.

An observation shape match is not enough: cadence, normalization, feature
ordering, safety timing, and CMPP/residual limits must also match.  This module
writes those semantics next to a checkpoint and refuses incompatible runtime
loads before SUMO is started.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from map_agnostic_tls import (
    GLOBAL_FEATURE_NAMES,
    MAX_MOVEMENTS,
    MAX_PHASES,
    MOVEMENT_FEATURE_NAMES,
    PHASE_FEATURE_NAMES,
)
from safe_residual_controller import CMPPConfig


CONTRACT_FORMAT_VERSION = 1
SAFE_RESIDUAL_SCHEMA_VERSION = 3


class CheckpointCompatibilityError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def feature_schema_payload() -> dict[str, Any]:
    return {
        "max_movements": MAX_MOVEMENTS,
        "max_phases": MAX_PHASES,
        "movement_features": list(MOVEMENT_FEATURE_NAMES),
        "phase_features": list(PHASE_FEATURE_NAMES),
        "global_features": list(GLOBAL_FEATURE_NAMES),
        "action_semantics": "0=hold,1..N=physical_candidate_phase",
        "analytical_observation_normalization": True,
        "vecnormalize_required": False,
    }


def checkpoint_contract_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    return path.parent / f"{path.name}_contract.json"


@dataclass(frozen=True)
class CheckpointContract:
    contract_format_version: int
    controller_schema_version: int
    controller_family: str
    policy_class: str
    decision_seconds: float
    step_length_seconds: float
    minimum_green_seconds: float
    maximum_green_seconds: float
    max_movements: int
    max_phases: int
    movement_features: tuple[str, ...]
    phase_features: tuple[str, ...]
    global_features: tuple[str, ...]
    feature_schema_sha256: str
    analytical_observation_normalization: bool
    vecnormalize_required: bool
    cmpp_config: Mapping[str, float]
    cmpp_config_sha256: str
    residual_authority: float
    residual_bound: float
    max_baseline_regret: float
    adapter_names: tuple[str, ...] = ()
    active_adapter: str | None = None

    @classmethod
    def create(
        cls,
        *,
        decision_seconds: float,
        step_length_seconds: float,
        minimum_green_seconds: float,
        maximum_green_seconds: float,
        residual_authority: float,
        residual_bound: float,
        max_baseline_regret: float,
        cmpp_config: CMPPConfig = CMPPConfig(),
        policy_class: str = (
            "safe_residual_policy.SafeResidualMapAgnosticPolicy"
        ),
        adapter_names: Sequence[str] = (),
        active_adapter: str | None = None,
    ) -> "CheckpointContract":
        features = feature_schema_payload()
        cmpp = cmpp_config.to_dict()
        return cls(
            contract_format_version=CONTRACT_FORMAT_VERSION,
            controller_schema_version=SAFE_RESIDUAL_SCHEMA_VERSION,
            controller_family="normalized_cmpp_bounded_residual",
            policy_class=policy_class,
            decision_seconds=float(decision_seconds),
            step_length_seconds=float(step_length_seconds),
            minimum_green_seconds=float(minimum_green_seconds),
            maximum_green_seconds=float(maximum_green_seconds),
            max_movements=int(features["max_movements"]),
            max_phases=int(features["max_phases"]),
            movement_features=tuple(features["movement_features"]),
            phase_features=tuple(features["phase_features"]),
            global_features=tuple(features["global_features"]),
            feature_schema_sha256=sha256_json(features),
            analytical_observation_normalization=True,
            vecnormalize_required=False,
            cmpp_config=cmpp,
            cmpp_config_sha256=sha256_json(cmpp),
            residual_authority=float(residual_authority),
            residual_bound=float(residual_bound),
            max_baseline_regret=float(max_baseline_regret),
            adapter_names=tuple(str(name) for name in adapter_names),
            active_adapter=(str(active_adapter) if active_adapter else None),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["movement_features"] = list(self.movement_features)
        payload["phase_features"] = list(self.phase_features)
        payload["global_features"] = list(self.global_features)
        payload["cmpp_config"] = dict(self.cmpp_config)
        payload["adapter_names"] = list(self.adapter_names)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointContract":
        required = {field_name for field_name in cls.__dataclass_fields__}
        missing = sorted(required - set(payload))
        if missing:
            raise CheckpointCompatibilityError(
                f"Checkpoint contract is missing fields: {missing}"
            )
        values = {key: payload[key] for key in required}
        values["movement_features"] = tuple(str(x) for x in values["movement_features"])
        values["phase_features"] = tuple(str(x) for x in values["phase_features"])
        values["global_features"] = tuple(str(x) for x in values["global_features"])
        values["cmpp_config"] = {
            str(key): float(value) for key, value in dict(values["cmpp_config"]).items()
        }
        values["adapter_names"] = tuple(str(x) for x in values["adapter_names"])
        return cls(**values)

    def validate_self(self) -> None:
        errors: list[str] = []
        if self.contract_format_version != CONTRACT_FORMAT_VERSION:
            errors.append(
                f"contract format {self.contract_format_version} != {CONTRACT_FORMAT_VERSION}"
            )
        if self.controller_schema_version != SAFE_RESIDUAL_SCHEMA_VERSION:
            errors.append(
                f"controller schema {self.controller_schema_version} != "
                f"{SAFE_RESIDUAL_SCHEMA_VERSION}"
            )
        features = feature_schema_payload()
        if self.feature_schema_sha256 != sha256_json(features):
            errors.append("feature schema hash does not match this code")
        if self.max_movements != MAX_MOVEMENTS or self.max_phases != MAX_PHASES:
            errors.append("padded movement/phase limits do not match this code")
        if tuple(self.movement_features) != tuple(MOVEMENT_FEATURE_NAMES):
            errors.append("movement feature order differs")
        if tuple(self.phase_features) != tuple(PHASE_FEATURE_NAMES):
            errors.append("phase feature order differs")
        if tuple(self.global_features) != tuple(GLOBAL_FEATURE_NAMES):
            errors.append("global feature order differs")
        try:
            config = CMPPConfig.from_mapping(self.cmpp_config)
            if self.cmpp_config_sha256 != sha256_json(config.to_dict()):
                errors.append("CMPP configuration hash is invalid")
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid CMPP configuration: {exc}")
        if self.vecnormalize_required:
            errors.append("safe residual schema must use analytical normalization")
        if len(set(self.adapter_names)) != len(self.adapter_names):
            errors.append("adapter_names contains duplicates")
        if self.active_adapter is not None and self.active_adapter not in self.adapter_names:
            errors.append("active_adapter is not present in adapter_names")
        if errors:
            raise CheckpointCompatibilityError("; ".join(errors))

    def validate_runtime(
        self,
        *,
        decision_seconds: float,
        step_length_seconds: float,
        minimum_green_seconds: float,
        maximum_green_seconds: float,
        atol: float = 1e-9,
    ) -> None:
        self.validate_self()
        expected = {
            "decision_seconds": (self.decision_seconds, float(decision_seconds)),
            "step_length_seconds": (
                self.step_length_seconds,
                float(step_length_seconds),
            ),
            "minimum_green_seconds": (
                self.minimum_green_seconds,
                float(minimum_green_seconds),
            ),
            "maximum_green_seconds": (
                self.maximum_green_seconds,
                float(maximum_green_seconds),
            ),
        }
        mismatches = [
            f"{name}: checkpoint={saved:g}, runtime={runtime:g}"
            for name, (saved, runtime) in expected.items()
            if abs(float(saved) - float(runtime)) > atol
        ]
        if mismatches:
            raise CheckpointCompatibilityError(
                "Incompatible checkpoint runtime settings: " + "; ".join(mismatches)
            )


def write_checkpoint_contract(
    model_path: str | Path, contract: CheckpointContract
) -> Path:
    contract.validate_self()
    path = checkpoint_contract_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_checkpoint_contract(model_path: str | Path) -> CheckpointContract:
    path = checkpoint_contract_path(model_path)
    if not path.exists():
        raise CheckpointCompatibilityError(
            f"Safe-residual checkpoint contract is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCompatibilityError(
            f"Could not read checkpoint contract {path}: {exc}"
        ) from exc
    contract = CheckpointContract.from_dict(payload)
    contract.validate_self()
    return contract


def validate_checkpoint_runtime(
    model_path: str | Path,
    *,
    decision_seconds: float,
    step_length_seconds: float,
    minimum_green_seconds: float,
    maximum_green_seconds: float,
) -> CheckpointContract:
    contract = load_checkpoint_contract(model_path)
    contract.validate_runtime(
        decision_seconds=decision_seconds,
        step_length_seconds=step_length_seconds,
        minimum_green_seconds=minimum_green_seconds,
        maximum_green_seconds=maximum_green_seconds,
    )
    return contract


__all__: Sequence[str] = (
    "CheckpointCompatibilityError",
    "CheckpointContract",
    "checkpoint_contract_path",
    "load_checkpoint_contract",
    "validate_checkpoint_runtime",
    "write_checkpoint_contract",
)
