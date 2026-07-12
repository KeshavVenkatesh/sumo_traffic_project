#!/usr/bin/env python3
"""
Mixed-TLS training wrapper around the known-good Santa Clara proxy env.

The policy interface remains exactly compatible with the old model:
    observation shape: (30,)
    action space:       Discrete(5)
    reward:             unchanged
    phase construction: unchanged
    action masks:       unchanged

The difference is the training distribution:
    - a balanced shuffled deck of TLS ids
    - a new TLS selected on each episode reset
    - random initial phase and modest random phase age
    - support for multiple parallel SUMO workers
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import traffic_rl_model_santaclara_proxy as base
from traffic_rl_model_santaclara_proxy import *  # noqa: F401,F403


TLS_POOL_FILE = Path(
    os.environ.get("MIXED_TLS_FILE", ".usable_santaclara_tls.txt")
).expanduser().resolve()

TLS_POOL_LIMIT = max(
    0,
    int(os.environ.get("MIXED_TLS_LIMIT", "40")),
)

RANDOM_INITIAL_PHASE = (
    os.environ.get("MIXED_RANDOM_INITIAL_PHASE", "1") != "0"
)

INITIAL_PHASE_AGE_MAX = max(
    0.0,
    float(os.environ.get("MIXED_INITIAL_PHASE_AGE_MAX", "12.0")),
)


# train_santaclara_proxy.py patches globals on the selected environment module.
# The inherited methods are defined in the base module, so synchronize those
# patched values before every reset.
_SYNC_NAMES = (
    "TRAIN_EPISODE_SECONDS",
    "EPISODE_SECONDS",
    "SIM_END",
    "DEFAULT_SIM_END",
    "MAX_NUM_VEHICLES",
    "MAX_ACTIVE_VEHICLE_CAP",
    "MAX_VEHICLES",
    "MAX_VEHICLE_VARIANTS",
    "VEHICLE_VARIANTS",
    "TARGET_VEHICLES",
    "TARGET_ACTIVE_VEHICLES",
    "INITIAL_VEHICLES",
    "INITIAL_ACTIVE_VEHICLES",
    "SPAWN_BATCH",
    "SPAWN_BATCH_SIZE",
    "ROUTE_LOOKAHEAD_EDGES",
    "LOOKAHEAD_EDGES",
    "GREEN_DURATION",
    "DEFAULT_GREEN_DURATION",
    "INTERSECTION_NO_LANE_CHANGE_DISTANCE",
    "TRAFFIC_LIGHT_NO_LANE_CHANGE_DISTANCE",
    "TLS_NO_LANE_CHANGE_DISTANCE",
    "INTERSECTION_LANE_PREP_DISTANCE",
    "TRAFFIC_LIGHT_LANE_PREP_DISTANCE",
    "TLS_LANE_PREP_DISTANCE",
    "USE_AMBULANCES",
    "ENABLE_AMBULANCES",
    "AMBULANCES_ENABLED",
    "SPAWN_AMBULANCES",
    "TRAIN_WITH_AMBULANCES",
    "AMBULANCE_INTERVAL",
    "AMBULANCE_SPAWN_INTERVAL",
    "TRAIN_WITH_SUMO_LOGS",
    "PRINT_SUMO_LOGS",
    "PRINT_TRAINING_SCENARIOS",
    "PRINT_SCENARIOS",
)


def _sync_base_globals() -> None:
    current = globals()

    for name in _SYNC_NAMES:
        if name in current:
            setattr(base, name, current[name])

    # These environment-variable overrides also work inside spawn-based
    # SubprocVecEnv workers, where parent-process module mutations are not
    # guaranteed to be inherited.
    episode_raw = os.environ.get("MIXED_EPISODE_SECONDS")
    if episode_raw is not None:
        episode_seconds = int(episode_raw)
        globals()["TRAIN_EPISODE_SECONDS"] = episode_seconds
        globals()["SIM_END"] = episode_seconds
        base.TRAIN_EPISODE_SECONDS = episode_seconds
        base.SIM_END = episode_seconds

    max_vehicle_raw = os.environ.get("MIXED_MAX_NUM_VEHICLES")
    if max_vehicle_raw is not None:
        max_vehicles = int(max_vehicle_raw)
        globals()["MAX_NUM_VEHICLES"] = max_vehicles
        base.MAX_NUM_VEHICLES = max_vehicles

    base.TRAIN_WITH_SUMO_LOGS = False
    if hasattr(base, "PRINT_SUMO_LOGS"):
        base.PRINT_SUMO_LOGS = False


def _read_tls_pool(fallback_tls_id: str | None) -> list[str]:
    tls_ids: list[str] = []

    if TLS_POOL_FILE.exists():
        tls_ids = [
            line.strip()
            for line in TLS_POOL_FILE.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if TLS_POOL_LIMIT > 0:
        tls_ids = tls_ids[:TLS_POOL_LIMIT]

    if not tls_ids and fallback_tls_id:
        tls_ids = [fallback_tls_id]

    if not tls_ids:
        raise RuntimeError(
            f"No TLS ids found in {TLS_POOL_FILE}. "
            "Generate .usable_santaclara_tls.txt first."
        )

    return tls_ids


class TrafficSignalEnv(base.TrafficSignalEnv):
    """
    Old compatible environment with episode-level balanced TLS sampling.
    """

    def __init__(
        self,
        tls_id=None,
        gui=False,
        randomize_traffic=False,
        route_variants=None,
        max_vehicle_variants=None,
        seed=None,
    ):
        _sync_base_globals()

        super().__init__(
            tls_id=tls_id,
            gui=gui,
            randomize_traffic=randomize_traffic,
            route_variants=route_variants,
            max_vehicle_variants=max_vehicle_variants,
        )

        self._fallback_tls_id = tls_id
        self._mixed_tls_enabled = bool(randomize_traffic)

        self._worker_seed = 42 if seed is None else int(seed)
        self._rng = random.Random(self._worker_seed)

        self._tls_pool = _read_tls_pool(self._fallback_tls_id)
        self._tls_deck: list[str] = []
        self._tls_cursor = 0
        self._episode_number = 0

        print(
            "[mixed-tls] "
            f"worker_seed={self._worker_seed} "
            f"pool_size={len(self._tls_pool)} "
            f"pool_file={TLS_POOL_FILE}"
        )

    def _reshuffle_tls_deck(self) -> None:
        self._tls_deck = list(self._tls_pool)
        self._rng.shuffle(self._tls_deck)
        self._tls_cursor = 0

    def _next_tls(self) -> str:
        if (
            not self._tls_deck
            or self._tls_cursor >= len(self._tls_deck)
        ):
            self._reshuffle_tls_deck()

        tls_id = self._tls_deck[self._tls_cursor]
        self._tls_cursor += 1
        return tls_id

    def _randomize_initial_signal_state(self) -> None:
        if not RANDOM_INITIAL_PHASE or self.controller is None:
            return

        phases = self.controller.get("phases", [])
        if not phases:
            return

        phase_pos = self._rng.randrange(len(phases))

        # Episode initialization does not need a yellow transition.
        base.start_green(
            self.controller["tls_id"],
            self.controller,
            phase_pos=phase_pos,
        )

        max_green = float(getattr(base, "MAX_GREEN_HOLD", 55.0))
        maximum_age = min(
            INITIAL_PHASE_AGE_MAX,
            max(0.0, max_green - 1.0),
        )

        phase_age = (
            self._rng.uniform(0.0, maximum_age)
            if maximum_age > 0.0
            else 0.0
        )

        self.controller["phase_elapsed"] = phase_age

        phase = phases[phase_pos]
        duration = float(phase.get("duration", max_green))
        self.controller["remaining"] = max(
            0.0,
            duration - phase_age,
        )

    def reset(self, seed=None, options=None):
        _sync_base_globals()

        if seed is not None:
            self._rng.seed(self._worker_seed + int(seed))
            self._tls_deck = []
            self._tls_cursor = 0

        if self._mixed_tls_enabled:
            self.tls_id = self._next_tls()
        else:
            self.tls_id = self._fallback_tls_id

        obs, info = super().reset(seed=seed, options=options)

        if self._mixed_tls_enabled:
            self._randomize_initial_signal_state()

            snapshot = base.lane_vehicle_snapshot(
                self.controller.get("all_in_lanes", set())
            )

            self.prev_wait, self.prev_queue = (
                base.total_controlled_wait_and_queue(
                    self.controller,
                    snapshot=snapshot,
                )
            )

            base.reset_run_step_arrival_counter()
            obs = base.get_observation(
                self.controller,
                snapshot=snapshot,
            )

        self._episode_number += 1

        info = dict(info)
        info.update(
            mixed_tls_id=self.tls_id,
            mixed_tls_episode=self._episode_number,
            mixed_tls_pool_size=len(self._tls_pool),
            initial_phase_pos=(
                None
                if self.controller is None
                else int(self.controller["phase_pos"])
            ),
            initial_phase_age=(
                0.0
                if self.controller is None
                else float(self.controller["phase_elapsed"])
            ),
        )

        return obs, info
