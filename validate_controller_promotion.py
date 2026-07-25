#!/usr/bin/env python3
"""Baseline-relative, paired promotion gates for traffic controllers.

This validator never looks at PPO reward.  It consumes fixed-demand evaluation
JSON, verifies paired seeds/schedules, and rejects checkpoints with meaningful
throughput or waiting regressions in any map/load condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    max_throughput_regression_pct: float = 2.0
    max_wait_regression_pct: float = 5.0
    minimum_controlled_tls_fraction: float = 0.95
    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    require_fixed_demand: bool = True


@dataclass(frozen=True)
class PairedMetricResult:
    metric: str
    direction: str
    pairs: int
    baseline_mean: float
    candidate_mean: float
    improvement_pct: float
    improvement_ci_low_pct: float
    improvement_ci_high_pct: float
    win_fraction: float


def _load_runs(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("runs", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list or {{'runs': [...]}} in {path}")
    return [dict(row) for row in rows]


def _rows_by_seed(rows: Iterable[Mapping[str, Any]], label: str) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("controller", "")) != label:
            continue
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"Duplicate row for controller={label!r}, seed={seed}")
        result[seed] = row
    return result


def _paired_rows(
    rows: Sequence[Mapping[str, Any]], baseline: str, candidate: str
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    baseline_rows = _rows_by_seed(rows, baseline)
    candidate_rows = _rows_by_seed(rows, candidate)
    if not baseline_rows:
        raise ValueError(f"No rows found for baseline {baseline!r}")
    if not candidate_rows:
        raise ValueError(f"No rows found for candidate {candidate!r}")
    if set(baseline_rows) != set(candidate_rows):
        missing_candidate = sorted(set(baseline_rows) - set(candidate_rows))
        missing_baseline = sorted(set(candidate_rows) - set(baseline_rows))
        raise ValueError(
            "Seed sets differ: "
            f"missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    return [(baseline_rows[seed], candidate_rows[seed]) for seed in sorted(baseline_rows)]


def _improvement_fraction(candidate: np.ndarray, baseline: np.ndarray, direction: str) -> np.ndarray:
    denominator = np.maximum(np.abs(baseline), 1e-9)
    if direction == "higher":
        return (candidate - baseline) / denominator
    if direction == "lower":
        return (baseline - candidate) / denominator
    raise ValueError(f"Unknown metric direction {direction!r}")


def paired_metric(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    metric: str,
    direction: str,
    *,
    bootstrap_samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> PairedMetricResult:
    baseline = np.asarray([float(left[metric]) for left, _right in pairs], dtype=np.float64)
    candidate = np.asarray([float(right[metric]) for _left, right in pairs], dtype=np.float64)
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError(f"Metric {metric!r} contains NaN or infinity")

    # Report the relative change of pooled means.  Bootstrap paired seeds, so
    # each resample retains baseline/candidate demand identity.
    if direction == "higher":
        improvement = (candidate.mean() - baseline.mean()) / max(abs(baseline.mean()), 1e-9)
        wins = candidate > baseline
    elif direction == "lower":
        improvement = (baseline.mean() - candidate.mean()) / max(abs(baseline.mean()), 1e-9)
        wins = candidate < baseline
    else:
        raise ValueError(f"Unknown metric direction {direction!r}")

    if bootstrap_samples > 0 and len(pairs) > 1:
        indices = rng.integers(0, len(pairs), size=(bootstrap_samples, len(pairs)))
        sampled_baseline = baseline[indices].mean(axis=1)
        sampled_candidate = candidate[indices].mean(axis=1)
        denominator = np.maximum(np.abs(sampled_baseline), 1e-9)
        if direction == "higher":
            samples = (sampled_candidate - sampled_baseline) / denominator
        else:
            samples = (sampled_baseline - sampled_candidate) / denominator
        alpha = (1.0 - confidence) / 2.0
        ci_low, ci_high = np.quantile(samples, [alpha, 1.0 - alpha])
    else:
        ci_low = ci_high = improvement

    return PairedMetricResult(
        metric=metric,
        direction=direction,
        pairs=len(pairs),
        baseline_mean=float(baseline.mean()),
        candidate_mean=float(candidate.mean()),
        improvement_pct=100.0 * float(improvement),
        improvement_ci_low_pct=100.0 * float(ci_low),
        improvement_ci_high_pct=100.0 * float(ci_high),
        win_fraction=float(np.mean(wins)),
    )


def _validate_demand_identity(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    require_fixed: bool,
) -> list[str]:
    failures: list[str] = []
    for baseline, candidate in pairs:
        seed = int(baseline["seed"])
        if require_fixed and (
            int(baseline.get("fixed_demand", 0)) != 1
            or int(candidate.get("fixed_demand", 0)) != 1
        ):
            failures.append(f"seed {seed}: result was not produced with fixed demand")
        baseline_scheduled = int(baseline.get("scheduled_total", 0) or 0)
        candidate_scheduled = int(candidate.get("scheduled_total", 0) or 0)
        if baseline_scheduled != candidate_scheduled:
            failures.append(
                f"seed {seed}: scheduled demand differs "
                f"({baseline_scheduled} vs {candidate_scheduled})"
            )
        if require_fixed and baseline_scheduled <= 0:
            failures.append(f"seed {seed}: scheduled_total is not positive")
        for field in (
            "demand_route_sha256",
            "demand_network_sha256",
            "demand_scenario",
            "demand_map_id",
        ):
            baseline_value = str(baseline.get(field, "") or "")
            candidate_value = str(candidate.get(field, "") or "")
            if require_fixed and not baseline_value:
                failures.append(f"seed {seed}: fixed-demand identity field {field} is missing")
            elif baseline_value != candidate_value:
                failures.append(
                    f"seed {seed}: {field} differs ({baseline_value!r} vs {candidate_value!r})"
                )
    return failures


def evaluate_condition(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
    config: GateConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    pairs = _paired_rows(rows, baseline, candidate)
    failures = _validate_demand_identity(pairs, config.require_fixed_demand)
    metrics: dict[str, PairedMetricResult] = {}
    required_metrics = {
        "total_arrived": "higher",
        "mean_global_wait": "lower",
        "mean_global_queue": "lower",
        "mean_avg_speed_mps": "higher",
    }
    # Some existing fixed-route evaluators do not yet expose SUMO's insertion
    # backlog separately.  Compare scheduled identity and completed throughput
    # now; evaluate not-departed demand as an additional metric when present.
    if config.require_fixed_demand and all(
        "not_departed_total" in left and "not_departed_total" in right
        for left, right in pairs
    ):
        required_metrics["not_departed_total"] = "lower"
    for metric, direction in required_metrics.items():
        if any(metric not in left or metric not in right for left, right in pairs):
            failures.append(f"required metric {metric!r} is missing")
            continue
        metrics[metric] = paired_metric(
            pairs,
            metric,
            direction,
            bootstrap_samples=config.bootstrap_samples,
            confidence=config.confidence,
            rng=rng,
        )

    throughput = metrics.get("total_arrived")
    waiting = metrics.get("mean_global_wait")
    if throughput and throughput.improvement_pct < -config.max_throughput_regression_pct:
        failures.append(
            f"throughput regression {-throughput.improvement_pct:.2f}% exceeds "
            f"{config.max_throughput_regression_pct:.2f}%"
        )
    if waiting and waiting.improvement_pct < -config.max_wait_regression_pct:
        failures.append(
            f"mean-wait regression {-waiting.improvement_pct:.2f}% exceeds "
            f"{config.max_wait_regression_pct:.2f}%"
        )

    # Safety counters are optional in older evaluators, but when present they
    # are hard gates. Invalid applied actions must always be zero.
    counter_metrics = (
        "invalid_actions",
        "collisions_total",
        "teleports_total",
        "route_failures_total",
    )
    for metric in counter_metrics:
        if all(metric in left and metric in right for left, right in pairs):
            baseline_total = sum(float(left[metric]) for left, _right in pairs)
            candidate_total = sum(float(right[metric]) for _left, right in pairs)
            if metric == "invalid_actions" and candidate_total > 0:
                failures.append(f"candidate applied {candidate_total:g} invalid actions")
            elif candidate_total > baseline_total:
                failures.append(
                    f"{metric} increased from {baseline_total:g} to {candidate_total:g}"
                )

    coverage_values = [
        float(right["controlled_tls_fraction"])
        for _left, right in pairs
        if "controlled_tls_fraction" in right
    ]
    if coverage_values and min(coverage_values) < config.minimum_controlled_tls_fraction:
        failures.append(
            f"controlled TLS coverage {min(coverage_values):.2%} is below "
            f"{config.minimum_controlled_tls_fraction:.2%}"
        )

    return {
        "condition": name,
        "baseline": baseline,
        "candidate": candidate,
        "paired_seeds": [int(left["seed"]) for left, _right in pairs],
        "eligible": not failures,
        "failures": failures,
        "metrics": {key: asdict(value) for key, value in metrics.items()},
    }


def evaluate_campaign(
    conditions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline: str,
    candidate: str,
    config: GateConfig = GateConfig(),
    random_seed: int = 20260722,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    reports = [
        evaluate_condition(
            name,
            rows,
            baseline=baseline,
            candidate=candidate,
            config=config,
            rng=rng,
        )
        for name, rows in conditions.items()
    ]
    improvements = [
        report["metrics"]["total_arrived"]["improvement_pct"]
        for report in reports
        if "total_arrived" in report["metrics"]
    ]
    worst_throughput = min(improvements) if improvements else -math.inf
    return {
        "promoted": bool(reports) and all(report["eligible"] for report in reports),
        "baseline": baseline,
        "candidate": candidate,
        "gate_config": asdict(config),
        "condition_count": len(reports),
        "worst_condition_throughput_improvement_pct": worst_throughput,
        "conditions": reports,
    }


def _parse_condition(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--condition must be NAME=RESULTS.json_or_csv")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--condition must be NAME=RESULTS.json")
    return name.strip(), Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True,
                        help="Repeat NAME=RESULTS.json for every map/load condition.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="controller_promotion_report.json")
    parser.add_argument("--max-throughput-regression-pct", type=float, default=2.0)
    parser.add_argument("--max-wait-regression-pct", type=float, default=5.0)
    parser.add_argument("--minimum-controlled-tls-fraction", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--require-fixed-demand", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    condition_paths = [_parse_condition(value) for value in args.condition]
    names = [name for name, _path in condition_paths]
    if len(set(names)) != len(names):
        parser.error("Condition names must be unique")
    config = GateConfig(
        max_throughput_regression_pct=args.max_throughput_regression_pct,
        max_wait_regression_pct=args.max_wait_regression_pct,
        minimum_controlled_tls_fraction=args.minimum_controlled_tls_fraction,
        bootstrap_samples=max(0, args.bootstrap_samples),
        confidence=args.confidence,
        require_fixed_demand=args.require_fixed_demand,
    )
    conditions = {name: _load_runs(path) for name, path in condition_paths}
    report = evaluate_campaign(
        conditions,
        baseline=args.baseline,
        candidate=args.candidate,
        config=config,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Promotion decision: {'PASS' if report['promoted'] else 'REJECT'}")
    for condition in report["conditions"]:
        status = "PASS" if condition["eligible"] else "REJECT"
        print(f"  {condition['condition']}: {status}")
        for failure in condition["failures"]:
            print(f"    - {failure}")
    print(f"Wrote {output}")
    if not report["promoted"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
