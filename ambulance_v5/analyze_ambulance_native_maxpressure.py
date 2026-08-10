#!/usr/bin/env python3
"""Validate and statistically analyze the final three-way ambulance benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from scipy.stats import ttest_1samp
except ImportError as exc:  # pragma: no cover - server environment guard
    raise ImportError("SciPy is required for paired p-values") from exc


NATIVE = "traffic_aware_route_native_sumo"
MAX_PRESSURE = "traffic_aware_route_max_pressure"
LEARNED = "traffic_aware_route_learned_signals"
CONTROLLERS = (NATIVE, MAX_PRESSURE, LEARNED)
EXPECTED_MAPS = ("fremont", "santa_clara", "fresno", "san_diego")
EXPECTED_RATES = (6.0, 12.0, 18.0)
EXPECTED_SEEDS = tuple(range(1001, 1031))


def ambulance_metric(name: str) -> Callable[[Mapping[str, Any]], float]:
    return lambda record: float(record["ambulance"][name])


def ordinary_metric(name: str) -> Callable[[Mapping[str, Any]], float]:
    return lambda record: float(record["ordinary_traffic"][name])


METRICS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], float]], ...] = (
    ("ambulance_mean_response_time_s", "lower", ambulance_metric("mean_response_time_s")),
    ("ambulance_p95_response_time_s", "lower", ambulance_metric("p95_response_time_s")),
    ("ambulance_completion_rate", "higher", ambulance_metric("completion_rate")),
    ("ambulance_mean_time_loss_s", "lower", ambulance_metric("mean_time_loss_s")),
    ("ambulance_mean_stopped_seconds", "lower", ambulance_metric("mean_stopped_seconds")),
    ("ordinary_arrived_total", "higher", ordinary_metric("arrived_total")),
    (
        "ordinary_scheduled_throughput_rate",
        "higher",
        ordinary_metric("scheduled_throughput_rate"),
    ),
    ("ordinary_mean_speed_mps", "higher", ordinary_metric("mean_speed_mps")),
    ("ordinary_mean_queue_vehicles", "lower", ordinary_metric("mean_queue_vehicles")),
    (
        "ordinary_mean_wait_all_departed_s",
        "lower",
        ordinary_metric("mean_wait_all_departed_s"),
    ),
    (
        "ordinary_mean_time_loss_all_departed_s",
        "lower",
        ordinary_metric("mean_time_loss_all_departed_s"),
    ),
)

COMPARISONS = (
    (LEARNED, NATIVE, "learned_vs_native_sumo"),
    (MAX_PRESSURE, NATIVE, "max_pressure_vs_native_sumo"),
    (LEARNED, MAX_PRESSURE, "learned_vs_max_pressure"),
)

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        raise ValueError("At least two paired observations are required")
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    if df > 120:
        return 1.960
    lower = max(key for key in T_CRITICAL_975 if key < df)
    upper = min(key for key in T_CRITICAL_975 if key > df)
    fraction = (df - lower) / (upper - lower)
    return T_CRITICAL_975[lower] + fraction * (
        T_CRITICAL_975[upper] - T_CRITICAL_975[lower]
    )


def condition_rate(record: Mapping[str, Any]) -> float:
    try:
        return float(record["scenario"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric benchmark scenario: {record.get('scenario')!r}") from exc


def validate_payload(payload: Mapping[str, Any]) -> dict[tuple[str, float, int, str], Mapping[str, Any]]:
    expected_mode = "paired_immutable_demand_native_maxpressure_learned_exact_sumo"
    if payload.get("evaluation_mode") != expected_mode:
        raise ValueError(
            f"Wrong evaluation mode: {payload.get('evaluation_mode')!r}; expected {expected_mode!r}"
        )
    if int(payload.get("episode_seconds", -1)) != 1200:
        raise ValueError("Final benchmark must use 1200 simulated seconds")
    if abs(float(payload.get("decision_seconds", -1.0)) - 5.0) > 1e-9:
        raise ValueError("Ambulance-v5 checkpoint contract requires a 5 s decision interval")
    if tuple(int(value) for value in payload.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Final benchmark must use seeds 1001..1030")

    records = list(payload.get("records", ()))
    if len(records) != 1080:
        raise ValueError(f"Expected exactly 1080 records; found {len(records)}")

    by_key: dict[tuple[str, float, int, str], Mapping[str, Any]] = {}
    for record in records:
        key = (
            str(record["map_id"]),
            condition_rate(record),
            int(record["seed"]),
            str(record["ablation"]),
        )
        if key in by_key:
            raise ValueError(f"Duplicate benchmark record: {key}")
        by_key[key] = record

    expected_keys = {
        (map_id, rate, seed, controller)
        for map_id in EXPECTED_MAPS
        for rate in EXPECTED_RATES
        for seed in EXPECTED_SEEDS
        for controller in CONTROLLERS
    }
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:10]
        unexpected = sorted(actual_keys - expected_keys)[:10]
        raise ValueError(
            "Benchmark matrix mismatch. "
            f"missing(first 10)={missing}, unexpected(first 10)={unexpected}"
        )

    for map_id in EXPECTED_MAPS:
        for rate in EXPECTED_RATES:
            for seed in EXPECTED_SEEDS:
                paired = [by_key[(map_id, rate, seed, controller)] for controller in CONTROLLERS]
                for field in (
                    "route_sha256",
                    "network_sha256",
                    "schedule_sha256",
                    "scheduled_ordinary_vehicles",
                ):
                    values = {str(record[field]) for record in paired}
                    if len(values) != 1:
                        raise ValueError(
                            f"Pairing failure for {map_id}/rate={rate:g}/seed={seed}: {field}={values}"
                        )
    return by_key


def summarize_pair(
    baseline_records: list[Mapping[str, Any]],
    candidate_records: list[Mapping[str, Any]],
    metric: str,
    direction: str,
    extractor: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    baseline_values = [extractor(record) for record in baseline_records]
    candidate_values = [extractor(record) for record in candidate_records]
    if len(baseline_values) != len(candidate_values) or len(baseline_values) < 2:
        raise ValueError(f"Insufficient paired values for {metric}")
    if not all(math.isfinite(value) for value in baseline_values + candidate_values):
        raise ValueError(f"Non-finite value in metric {metric}")

    if direction == "higher":
        deltas = [candidate - baseline for baseline, candidate in zip(baseline_values, candidate_values)]
    else:
        deltas = [baseline - candidate for baseline, candidate in zip(baseline_values, candidate_values)]
    n = len(deltas)
    mean_delta = statistics.fmean(deltas)
    sd_delta = statistics.stdev(deltas)
    se_delta = sd_delta / math.sqrt(n)
    margin = t_critical_95(n - 1) * se_delta
    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    if abs(baseline_mean) < 1e-12:
        improvement = math.nan
    elif direction == "higher":
        improvement = 100.0 * (candidate_mean - baseline_mean) / abs(baseline_mean)
    else:
        improvement = 100.0 * (baseline_mean - candidate_mean) / abs(baseline_mean)

    if sd_delta < 1e-12:
        pvalue = 0.0 if abs(mean_delta) >= 1e-12 else 1.0
        effect_dz = (
            math.copysign(math.inf, mean_delta) if abs(mean_delta) >= 1e-12 else 0.0
        )
    else:
        pvalue = float(ttest_1samp(deltas, popmean=0.0).pvalue)
        effect_dz = mean_delta / sd_delta
    return {
        "metric": metric,
        "direction": direction,
        "n": n,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "aggregate_improvement_pct": improvement,
        "paired_delta_mean": mean_delta,
        "paired_delta_sd": sd_delta,
        "paired_delta_ci95_low": mean_delta - margin,
        "paired_delta_ci95_high": mean_delta + margin,
        "paired_t_pvalue": pvalue,
        "paired_effect_size_dz": effect_dz,
        "wins": sum(delta > 0.0 for delta in deltas),
        "losses": sum(delta < 0.0 for delta in deltas),
        "ties": sum(delta == 0.0 for delta in deltas),
    }


def apply_holm(rows: list[dict[str, Any]]) -> None:
    for comparison in {str(row["comparison"]) for row in rows}:
        family = [
            row
            for row in rows
            if row["scope"] == "condition" and row["comparison"] == comparison
        ]
        ordered = sorted(family, key=lambda row: float(row["paired_t_pvalue"]))
        running = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, float(row["paired_t_pvalue"]) * (total - rank))
            running = max(running, adjusted)
            row["holm_adjusted_pvalue"] = running
            row["holm_significant_in_favor"] = (
                "yes"
                if running < 0.05 and float(row["paired_delta_ci95_low"]) > 0.0
                else "no"
            )
    for row in rows:
        row.setdefault("holm_adjusted_pvalue", math.nan)
        row.setdefault("holm_significant_in_favor", "not_applicable")


def safety_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for controller in CONTROLLERS:
        selected = [record for record in records if record["ablation"] == controller]
        ambulance_collisions = sum(
            len(record["ambulance"].get("collision_vehicle_ids", ())) for record in selected
        )
        ambulance_teleports = sum(
            len(record["ambulance"].get("teleported_vehicle_ids", ())) for record in selected
        )
        ordinary_collisions = sum(
            len(record["ordinary_traffic"].get("collision_vehicle_ids", ())) for record in selected
        )
        ordinary_teleports = sum(
            len(record["ordinary_traffic"].get("teleported_vehicle_ids", ())) for record in selected
        )
        failed = sum(int(record["ambulance"].get("failed_total", 0)) for record in selected)
        censored = sum(int(record["ambulance"].get("censored_total", 0)) for record in selected)
        invalid_actions = sum(
            int(record["ambulance"].get("signal_safety", {}).get("invalid_policy_actions", 0))
            for record in selected
        )
        invalid_transitions = sum(
            int(record["ambulance"].get("signal_safety", {}).get("invalid_signal_transitions", 0))
            for record in selected
        )
        unrecovered = sum(int(record["recovery"].get("unrecovered_events", 0)) for record in selected)
        result[controller] = {
            "runs": len(selected),
            "ambulance_collisions": ambulance_collisions,
            "ambulance_teleports": ambulance_teleports,
            "ordinary_collisions": ordinary_collisions,
            "ordinary_teleports": ordinary_teleports,
            "ambulance_failed": failed,
            "ambulance_censored": censored,
            "invalid_policy_actions": invalid_actions,
            "invalid_signal_transitions": invalid_transitions,
            "unrecovered_events": unrecovered,
            "strict_safety_pass": (
                ambulance_collisions == 0
                and ambulance_teleports == 0
                and ordinary_collisions == 0
                and ordinary_teleports == 0
                and failed == 0
                and censored == 0
                and invalid_actions == 0
                and invalid_transitions == 0
                and unrecovered == 0
            ),
        }
    return result


def json_safe(value: Any) -> Any:
    """Replace non-finite floats so the result is strict JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.evaluation_json.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    by_key = validate_payload(payload)

    rows: list[dict[str, Any]] = []
    for map_id in EXPECTED_MAPS:
        for rate in EXPECTED_RATES:
            for candidate, baseline, label in COMPARISONS:
                candidate_records = [
                    by_key[(map_id, rate, seed, candidate)] for seed in EXPECTED_SEEDS
                ]
                baseline_records = [
                    by_key[(map_id, rate, seed, baseline)] for seed in EXPECTED_SEEDS
                ]
                for metric, direction, extractor in METRICS:
                    rows.append(
                        {
                            "scope": "condition",
                            "map": map_id,
                            "rate": rate,
                            "comparison": label,
                            "candidate": candidate,
                            "baseline": baseline,
                            **summarize_pair(
                                baseline_records,
                                candidate_records,
                                metric,
                                direction,
                                extractor,
                            ),
                        }
                    )

    for candidate, baseline, label in COMPARISONS:
        candidate_records = [
            by_key[(map_id, rate, seed, candidate)]
            for map_id in EXPECTED_MAPS
            for rate in EXPECTED_RATES
            for seed in EXPECTED_SEEDS
        ]
        baseline_records = [
            by_key[(map_id, rate, seed, baseline)]
            for map_id in EXPECTED_MAPS
            for rate in EXPECTED_RATES
            for seed in EXPECTED_SEEDS
        ]
        for metric, direction, extractor in METRICS:
            rows.append(
                {
                    "scope": "pooled",
                    "map": "ALL",
                    "rate": "ALL",
                    "comparison": label,
                    "candidate": candidate,
                    "baseline": baseline,
                    **summarize_pair(
                        baseline_records,
                        candidate_records,
                        metric,
                        direction,
                        extractor,
                    ),
                }
            )
    apply_holm(rows)

    csv_path = output_dir / "statistical_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    robustness: dict[str, Any] = {}
    for comparison in ("learned_vs_native_sumo", "learned_vs_max_pressure"):
        robustness[comparison] = {}
        for metric, _direction, _extractor in METRICS:
            metric_rows = [
                row
                for row in rows
                if row["scope"] == "condition"
                and row["comparison"] == comparison
                and row["metric"] == metric
            ]
            finite = [
                row for row in metric_rows if math.isfinite(float(row["aggregate_improvement_pct"]))
            ]
            worst = min(
                finite or metric_rows,
                key=lambda row: float(row["aggregate_improvement_pct"]),
            )
            robustness[comparison][metric] = {
                "conditions": len(metric_rows),
                "clear_positive_conditions": sum(
                    float(row["paired_delta_ci95_low"]) > 0.0 for row in metric_rows
                ),
                "clear_negative_conditions": sum(
                    float(row["paired_delta_ci95_high"]) < 0.0 for row in metric_rows
                ),
                "holm_clear_positive_conditions": sum(
                    row["holm_significant_in_favor"] == "yes" for row in metric_rows
                ),
                "worst_condition": {
                    "map": worst["map"],
                    "rate": worst["rate"],
                    "improvement_pct": worst["aggregate_improvement_pct"],
                },
            }

    records = list(payload["records"])
    safety = safety_summary(records)
    robustness_path = output_dir / "robustness_summary.json"
    robustness_path.write_text(
        json.dumps(
            json_safe({
                "matrix_validation_passed": True,
                "fixed_demand_and_ambulance_schedule_pairing_passed": True,
                "conditions": 12,
                "paired_seeds_per_condition": 30,
                "runs": 1080,
                "statistics": (
                    "paired two-sided 95% Student-t CIs, paired t-tests, paired Cohen's dz, "
                    "Holm adjustment across condition/metric tests within each controller comparison"
                ),
                "safety": safety,
                "robustness": robustness,
            }),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    pooled = [
        row
        for row in rows
        if row["scope"] == "pooled"
        and row["comparison"] in {"learned_vs_native_sumo", "learned_vs_max_pressure"}
    ]
    report_lines = [
        "AMBULANCE-V5 FINAL NATIVE SUMO / MAXPRESSURE BENCHMARK",
        "Matrix: 4 maps x 3 rates x 30 paired seeds x 3 controllers = 1080 runs",
        "Pairing: PASS (background route, network, and deterministic ambulance schedule hashes match)",
        "",
    ]
    for comparison in ("learned_vs_native_sumo", "learned_vs_max_pressure"):
        report_lines.append(comparison)
        for row in pooled:
            if row["comparison"] != comparison:
                continue
            improvement = float(row["aggregate_improvement_pct"])
            imp_text = "n/a" if not math.isfinite(improvement) else f"{improvement:+.2f}%"
            report_lines.append(
                f"  {row['metric']}: {imp_text}; "
                f"95% paired delta CI [{float(row['paired_delta_ci95_low']):.4g}, "
                f"{float(row['paired_delta_ci95_high']):.4g}]; "
                f"p={float(row['paired_t_pvalue']):.4g}; dz={float(row['paired_effect_size_dz']):.4g}"
            )
        report_lines.append("")
    report_lines.append("strict safety by controller")
    for controller in CONTROLLERS:
        item = safety[controller]
        report_lines.append(
            f"  {controller}: {'PASS' if item['strict_safety_pass'] else 'FAIL'}; "
            f"failed={item['ambulance_failed']}, censored={item['ambulance_censored']}, "
            f"collisions={item['ambulance_collisions'] + item['ordinary_collisions']}, "
            f"teleports={item['ambulance_teleports'] + item['ordinary_teleports']}, "
            f"invalid_actions={item['invalid_policy_actions']}, "
            f"invalid_transitions={item['invalid_signal_transitions']}, "
            f"unrecovered={item['unrecovered_events']}"
        )
    report_path = output_dir / "benchmark_summary.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("FULL_MATRIX_VALIDATION=PASS")
    print("PAIRING_VALIDATION=PASS")
    print(f"STATISTICAL_ROWS={len(rows)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {robustness_path}")
    print(f"Wrote {report_path}")
    print()
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
