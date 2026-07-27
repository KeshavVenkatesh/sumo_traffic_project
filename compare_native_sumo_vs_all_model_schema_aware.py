#!/usr/bin/env python3
"""Run native-vs-all-TLS evaluation for both 30- and 33-feature policies.

The legacy Santa Clara policies use the original 30 traffic features.  The
ambulance-aware policy appends three local ambulance features.  This adapter
detects the observation dimension stored in each PPO archive, constructs the
matching VecNormalize dummy environment, and supplies the corresponding
observation vector without changing either trained policy.

It also enables the same ambulance spawning process in the native-SUMO run
that the learned-controller run uses.  The original native evaluator left
``next_ambulance_spawn`` at infinity, making that comparison asymmetric.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

import compare_fixed_vs_single_vs_all_model_realistic as cmp


LEGACY_OBS_DIM = 30
AMBULANCE_OBS_DIM = 33
AMBULANCE_NEARBY_DISTANCE_M = 150.0
AMBULANCE_MAX_DISTANCE_M = 300.0
AMBULANCE_SPEED_REFERENCE_MPS = 22.2
# Raw normalization constants used by the completed 33-feature training run.
# These must be applied before the model's saved VecNormalize transform.
AMBULANCE_POLICY_MAX_GREEN_HOLD_S = 55.0
AMBULANCE_POLICY_SIM_END_S = 7200.0

# One model is evaluated per process.  The parallel launcher creates separate
# processes for different models/seeds, so this process-local schema is safe.
_active_policy_obs_dim = LEGACY_OBS_DIM
_legacy_get_observation = cmp.get_observation


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class AmbulanceOutcomeTracker:
    """Collect actual ambulance outcomes, not merely ambulance occupancy.

    The base evaluator reports mean active ambulance count.  That value is not
    a response-time metric: a controller can have more active ambulances simply
    because trips take longer.  This tracker adds completion, trip-time,
    distance-normalized travel-time, stopped-time, and time-loss measurements.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.last_sim_time = 0.0
        self.next_detail_sample_time = 0.0

    def _record_for(
        self,
        ambulance_id: str,
        sim_time: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        record = self.records.setdefault(
            ambulance_id,
            {
                "spawn_seen_time": sim_time,
                "depart_time": None,
                "end_time": None,
                "completed": False,
                "removed": False,
                "last_seen_time": None,
                "last_speed": 0.0,
                "tracked_seconds": 0.0,
                "stopped_seconds": 0.0,
                "route_distance": 0.0,
                "distance_driven": 0.0,
                "time_loss": 0.0,
            },
        )
        if metadata:
            record["route_distance"] = float(
                metadata.get("route_distance", record["route_distance"]) or 0.0
            )
        return record

    def observe(self, sim_state: dict[str, Any]) -> None:
        try:
            sim_time = float(cmp.sim.traci.simulation.getTime())
            current_ids = set(cmp.sim.traci.vehicle.getIDList())
            arrived_ids = set(cmp.sim.traci.simulation.getArrivedIDList())
        except Exception:
            return

        active_metadata = sim_state.get("active_ambulances", {}) or {}
        for ambulance_id, metadata in active_metadata.items():
            self._record_for(str(ambulance_id), sim_time, metadata)

        current_ambulances = {
            str(vehicle_id)
            for vehicle_id in current_ids
            if str(vehicle_id).startswith("ambulance_")
        }
        detail_sample = sim_time + 1e-9 >= self.next_detail_sample_time

        for ambulance_id in current_ambulances:
            record = self._record_for(
                ambulance_id,
                sim_time,
                active_metadata.get(ambulance_id),
            )

            if record["depart_time"] is None:
                try:
                    departure = float(
                        cmp.sim.traci.vehicle.getDeparture(ambulance_id)
                    )
                except Exception:
                    departure = sim_time
                record["depart_time"] = departure if departure >= 0.0 else sim_time

            try:
                speed = float(cmp.sim.traci.vehicle.getSpeed(ambulance_id))
            except Exception:
                speed = float(record["last_speed"])

            last_seen = record["last_seen_time"]
            if last_seen is not None:
                elapsed = max(0.0, sim_time - float(last_seen))
                record["tracked_seconds"] += elapsed
                if speed < 1.0:
                    record["stopped_seconds"] += elapsed
            record["last_seen_time"] = sim_time
            record["last_speed"] = speed

            if detail_sample:
                try:
                    record["distance_driven"] = max(
                        float(record["distance_driven"]),
                        float(cmp.sim.traci.vehicle.getDistance(ambulance_id)),
                    )
                except Exception:
                    pass
                try:
                    record["time_loss"] = max(
                        float(record["time_loss"]),
                        float(cmp.sim.traci.vehicle.getTimeLoss(ambulance_id)),
                    )
                except Exception:
                    pass

        # A vehicle that vanishes on the current SUMO step is an arrival only
        # when SUMO lists it in getArrivedIDList().  Other disappearances are
        # retained separately instead of being misreported as successful trips.
        for ambulance_id, record in self.records.items():
            if record["depart_time"] is None or record["end_time"] is not None:
                continue
            if ambulance_id in current_ambulances:
                continue
            record["end_time"] = sim_time
            if ambulance_id in arrived_ids:
                record["completed"] = True
            else:
                record["removed"] = True

        if detail_sample:
            self.next_detail_sample_time = sim_time + 5.0
        self.last_sim_time = sim_time

    def summary(self) -> dict[str, float | int]:
        records = list(self.records.values())
        departed = [record for record in records if record["depart_time"] is not None]
        completed = [record for record in departed if record["completed"]]
        removed = [record for record in departed if record["removed"]]
        censored = [
            record
            for record in departed
            if not record["completed"] and not record["removed"]
        ]

        trip_times = [
            max(0.0, float(record["end_time"]) - float(record["depart_time"]))
            for record in completed
        ]
        stopped_times = [float(record["stopped_seconds"]) for record in departed]
        stopped_fractions = [
            float(record["stopped_seconds"])
            / max(1e-9, float(record["tracked_seconds"]))
            for record in departed
        ]
        time_losses = [float(record["time_loss"]) for record in departed]
        seconds_per_km = [
            1000.0
            * max(0.0, float(record["end_time"]) - float(record["depart_time"]))
            / max(1.0, float(record["route_distance"]))
            for record in completed
        ]

        result: dict[str, float | int] = {
            "ambulance_spawned_total": len(records),
            "ambulance_departed_total": len(departed),
            "ambulance_completed_total": len(completed),
            "ambulance_removed_total": len(removed),
            "ambulance_censored_total": len(censored),
            "ambulance_completion_rate": len(completed) / max(1, len(departed)),
            "mean_ambulance_trip_time_s": float(np.mean(trip_times)) if trip_times else 0.0,
            "median_ambulance_trip_time_s": _percentile(trip_times, 50.0),
            "p90_ambulance_trip_time_s": _percentile(trip_times, 90.0),
            "mean_ambulance_seconds_per_km": (
                float(np.mean(seconds_per_km)) if seconds_per_km else 0.0
            ),
            "mean_ambulance_stopped_seconds": (
                float(np.mean(stopped_times)) if stopped_times else 0.0
            ),
            "p90_ambulance_stopped_seconds": _percentile(stopped_times, 90.0),
            "mean_ambulance_stopped_fraction": (
                float(np.mean(stopped_fractions)) if stopped_fractions else 0.0
            ),
            "mean_ambulance_time_loss_s": (
                float(np.mean(time_losses)) if time_losses else 0.0
            ),
        }
        return result


_ambulance_tracker = AmbulanceOutcomeTracker()
_original_run_simulation_steps = cmp.sim.run_simulation_steps
_original_summarize_samples = cmp.summarize_samples
_original_anchorless_reset = cmp.AnchorlessSimulationEpisode.reset


def tracked_run_simulation_steps(*args, **kwargs):
    result = _original_run_simulation_steps(*args, **kwargs)
    sim_state = kwargs.get("sim_state")
    if sim_state is None and len(args) >= 10:
        sim_state = args[9]
    if isinstance(sim_state, dict):
        _ambulance_tracker.observe(sim_state)
    return result


def tracked_summarize_samples(label, seed, samples):
    result = _original_summarize_samples(label, seed, samples)
    ambulance_summary = _ambulance_tracker.summary()
    result.update(ambulance_summary)
    print(
        f"[{label} seed {seed}] ambulance outcomes: "
        f"spawned={ambulance_summary['ambulance_spawned_total']}, "
        f"departed={ambulance_summary['ambulance_departed_total']}, "
        f"completed={ambulance_summary['ambulance_completed_total']}, "
        f"censored={ambulance_summary['ambulance_censored_total']}, "
        f"removed={ambulance_summary['ambulance_removed_total']}",
        flush=True,
    )
    return result


def tracked_anchorless_reset(self):
    _ambulance_tracker.reset()
    info = _original_anchorless_reset(self)
    # The base anchorless evaluator initializes this to infinity even though
    # make_sim_args says ambulances are enabled. Start the same ambulance
    # process used by the Native-SUMO path below.
    self.sim_state["next_ambulance_spawn"] = 0.0
    return info


cmp.sim.run_simulation_steps = tracked_run_simulation_steps
cmp.summarize_samples = tracked_summarize_samples
cmp.AnchorlessSimulationEpisode.reset = tracked_anchorless_reset


def _ambulance_features(controller: dict[str, Any]) -> list[float]:
    """Match traffic_rl_model.get_ambulance_obs exactly."""

    tls_id = str(controller.get("tls_id", ""))
    try:
        junction_ids = cmp.sim.traci.trafficlight.getControlledJunctions(tls_id)
        if not junction_ids:
            raise ValueError("TLS has no controlled junction")
        junction_x, junction_y = cmp.sim.traci.junction.getPosition(junction_ids[0])
    except Exception:
        return [1.0, 0.0, 0.0]

    nearest_distance = float("inf")
    nearest_speed = 0.0
    any_waiting = 0.0

    try:
        vehicle_ids = cmp.sim.traci.vehicle.getIDList()
    except Exception:
        return [1.0, 0.0, 0.0]

    for vehicle_id in vehicle_ids:
        if not str(vehicle_id).startswith("ambulance_"):
            continue

        try:
            x, y = cmp.sim.traci.vehicle.getPosition(vehicle_id)
            distance = math.hypot(x - junction_x, y - junction_y)
            speed = float(cmp.sim.traci.vehicle.getSpeed(vehicle_id))
        except Exception:
            continue

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speed = speed

        if distance <= AMBULANCE_NEARBY_DISTANCE_M and speed < 1.0:
            any_waiting = 1.0

    if not math.isfinite(nearest_distance):
        return [1.0, 0.0, 0.0]

    return [
        min(1.0, nearest_distance / AMBULANCE_MAX_DISTANCE_M),
        min(1.0, nearest_speed / AMBULANCE_SPEED_REFERENCE_MPS),
        any_waiting,
    ]


def schema_aware_get_observation(
    controller: dict[str, Any], episode_seconds: int
) -> np.ndarray:
    traffic_observation = np.asarray(
        _legacy_get_observation(controller, episode_seconds), dtype=np.float32
    )

    if traffic_observation.shape != (LEGACY_OBS_DIM,):
        raise RuntimeError(
            "The base realistic evaluator no longer produced the expected "
            f"30 traffic features; got {traffic_observation.shape}."
        )

    if _active_policy_obs_dim == LEGACY_OBS_DIM:
        return traffic_observation

    if _active_policy_obs_dim == AMBULANCE_OBS_DIM:
        # The legacy evaluator's final two raw traffic features use /60 and
        # /episode_seconds.  The ambulance-aware policy was trained with /55
        # and /7200 respectively.  Preserve the legacy path above, but restore
        # the exact training transform before applying the 33-feature model's
        # own saved VecNormalize statistics.
        traffic_observation = traffic_observation.copy()
        traffic_observation[28] = (
            float(controller.get("phase_elapsed", 0.0))
            / AMBULANCE_POLICY_MAX_GREEN_HOLD_S
        )
        try:
            sim_time = float(cmp.sim.traci.simulation.getTime())
        except Exception:
            sim_time = 0.0
        traffic_observation[29] = sim_time / AMBULANCE_POLICY_SIM_END_S
        return np.concatenate(
            [traffic_observation, np.asarray(_ambulance_features(controller), dtype=np.float32)]
        )

    raise RuntimeError(
        f"Unsupported model observation dimension: {_active_policy_obs_dim}. "
        "Only the legacy 30-feature and ambulance-aware 33-feature schemas are supported."
    )


def schema_aware_get_observation_batch(
    controllers: list[dict[str, Any]], episode_seconds: int
) -> list[np.ndarray]:
    return [
        schema_aware_get_observation(controller, episode_seconds)
        for controller in controllers
    ]


class DynamicPolicyShapeEnv(gym.Env):
    """No-SUMO environment used only to load model/VecNormalize shapes."""

    metadata = {"render_modes": []}

    def __init__(self, observation_dim: int):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

    def _zeros(self) -> np.ndarray:
        return np.zeros((self.observation_dim,), dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        return self._zeros(), {}

    def step(self, action: int):
        return self._zeros(), 0.0, False, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(5, dtype=bool)


def schema_aware_load_model_with_vecnormalize(
    args, seed: int, label: str
):
    """Detect the PPO schema before constructing the normalization wrapper."""

    global _active_policy_obs_dim

    model_path = Path(args.model_path)
    probe = MaskablePPO.load(str(model_path), device=args.device)
    observation_shape = tuple(probe.observation_space.shape or ())
    action_count = int(getattr(probe.action_space, "n", -1))
    del probe

    if len(observation_shape) != 1:
        raise RuntimeError(
            f"Unsupported non-vector observation space for {model_path}: "
            f"{observation_shape}"
        )

    observation_dim = int(observation_shape[0])
    if observation_dim not in {LEGACY_OBS_DIM, AMBULANCE_OBS_DIM}:
        raise RuntimeError(
            f"Unsupported observation size for {model_path}: {observation_dim}. "
            "Expected 30 or 33."
        )
    if action_count != 5:
        raise RuntimeError(
            f"Unsupported action count for {model_path}: {action_count}. Expected 5."
        )

    _active_policy_obs_dim = observation_dim
    print(
        f"[{label} seed {seed}] detected policy schema: "
        f"observation={observation_dim}, actions={action_count}",
        flush=True,
    )

    dummy_raw_env = DummyVecEnv(
        [lambda: Monitor(DynamicPolicyShapeEnv(observation_dim))]
    )
    vecnormalize_path = cmp.find_vecnormalize_path(
        model_path, getattr(args, "vecnormalize_path", None)
    )

    if vecnormalize_path is not None:
        print(
            f"[{label} seed {seed}] Loading VecNormalize stats from "
            f"{vecnormalize_path}"
        )
        normalized_env = VecNormalize.load(str(vecnormalize_path), dummy_raw_env)
        normalized_env.training = False
        normalized_env.norm_reward = False
    else:
        print(
            f"[{label} seed {seed}] Warning: no VecNormalize stats found; "
            "using raw observations."
        )
        normalized_env = VecNormalize(
            dummy_raw_env, norm_obs=False, norm_reward=False
        )
        normalized_env.training = False

    model = MaskablePPO.load(
        str(model_path), env=normalized_env, device=args.device
    )
    return model, normalized_env


# Patch the policy seam before importing the native runner.  That module imports
# the same cmp module object, so it sees these schema-aware functions.
cmp.get_observation = schema_aware_get_observation
cmp.get_observation_batch = schema_aware_get_observation_batch
cmp.load_model_with_vecnormalize = schema_aware_load_model_with_vecnormalize

import compare_native_sumo_vs_all_model as native_runner  # noqa: E402


_original_native_reset = native_runner.NativeSignalEpisode.reset


def fair_native_reset(self):
    _ambulance_tracker.reset()
    info = _original_native_reset(self)
    # Learned-controller episodes start ambulance spawning at time zero.  Use
    # the same setting for Native SUMO so controller comparisons are symmetric.
    self.sim_state["next_ambulance_spawn"] = 0.0
    return info


native_runner.NativeSignalEpisode.reset = fair_native_reset


if __name__ == "__main__":
    native_runner.main()
