#!/usr/bin/env python3
"""Paired statistics, robustness gates, and fairness checks for evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_two_map_eval_statistics import METRICS, summarize_metric

try:
    from scipy.stats import ttest_1samp
except ImportError as exc:  # pragma: no cover - installation guard
    raise ImportError(
        "Install requirements-map-agnostic.txt (SciPy is required for "
        "paired p-values)."
    ) from exc


COMPARISONS = (
    ("all_model", "native_sumo", "learned_vs_native"),
    ("max_pressure", "native_sumo", "max_pressure_vs_native"),
    ("all_model", "max_pressure", "learned_vs_max_pressure"),
)


def load_condition(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    controllers: dict[str, dict[int, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            controllers.setdefault(row["controller"], {})[
                int(row["seed"])
            ] = row
    return controllers


def condition_paths(root: Path):
    for path in sorted(root.glob("*/rate_*/paired_runs.csv")):
        yield path.parent.parent.name, path.parent.name.removeprefix("rate_"), path


def paired_or_raise(
    controllers: dict[str, dict[int, dict[str, str]]],
    candidate: str,
    baseline: str,
    path: Path,
):
    missing = [
        label for label in (candidate, baseline) if label not in controllers
    ]
    if missing:
        raise ValueError(f"Missing {missing} in {path}")
    candidate_rows = controllers[candidate]
    baseline_rows = controllers[baseline]
    if set(candidate_rows) != set(baseline_rows):
        raise ValueError(
            f"Unpaired {candidate}/{baseline} seeds in {path}: "
            f"candidate={sorted(candidate_rows)}, "
            f"baseline={sorted(baseline_rows)}"
        )
    return candidate_rows, baseline_rows


def fairness_checks(
    controllers: dict[str, dict[int, dict[str, str]]], path: Path
) -> list[dict[str, Any]]:
    issues = []
    all_seeds = set().union(*(rows.keys() for rows in controllers.values()))
    for seed in sorted(all_seeds):
        rows = [values[seed] for values in controllers.values() if seed in values]
        if len(rows) != len(controllers):
            issues.append(
                {"path": str(path), "seed": seed, "issue": "missing_controller"}
            )
            continue
        route_files = {row.get("demand_route_file", "") for row in rows}
        if len(route_files) != 1 or not next(iter(route_files), ""):
            issues.append(
                {"path": str(path), "seed": seed, "issue": "demand_route_mismatch"}
            )
        modes = {row.get("demand_mode", "") for row in rows}
        if modes != {"fixed_route_replay"}:
            issues.append(
                {"path": str(path), "seed": seed, "issue": "not_fixed_route_replay"}
            )
        sim_seconds = [float(row.get("sim_seconds", 0.0) or 0.0) for row in rows]
        if max(sim_seconds, default=0.0) - min(sim_seconds, default=0.0) > 1e-6:
            issues.append(
                {"path": str(path), "seed": seed, "issue": "sim_time_mismatch"}
            )
        samples = [int(float(row.get("samples", 0) or 0)) for row in rows]
        if len(set(samples)) != 1:
            issues.append(
                {"path": str(path), "seed": seed, "issue": "sample_grid_mismatch"}
            )
    return issues


def pooled_rows(
    conditions: list[dict[str, Any]], controller: str
) -> dict[int, dict[str, str]]:
    result = {}
    index = 0
    for condition in conditions:
        for _seed, row in sorted(condition["controllers"][controller].items()):
            result[index] = row
            index += 1
    return result


def fmt(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.3f}"


def paired_test_stats(
    baseline: dict[int, dict[str, str]],
    candidate: dict[int, dict[str, str]],
    metric: str,
    direction: str,
) -> dict[str, float]:
    deltas = []
    for seed in sorted(baseline):
        base = float(baseline[seed][metric])
        value = float(candidate[seed][metric])
        deltas.append(value - base if direction == "higher" else base - value)
    if len(deltas) < 2:
        return {"paired_t_pvalue": math.nan, "paired_effect_size_dz": math.nan}
    mean = statistics.fmean(deltas)
    standard_deviation = statistics.stdev(deltas)
    if standard_deviation < 1e-12:
        pvalue = 0.0 if abs(mean) >= 1e-12 else 1.0
        effect = (
            math.copysign(math.inf, mean) if abs(mean) >= 1e-12 else 0.0
        )
    else:
        pvalue = float(ttest_1samp(deltas, popmean=0.0).pvalue)
        effect = mean / standard_deviation
    return {
        "paired_t_pvalue": pvalue,
        "paired_effect_size_dz": effect,
    }


def apply_holm_adjustment(rows: list[dict[str, Any]]) -> None:
    """Control family-wise error across all condition/metric tests."""
    for comparison in {row["comparison"] for row in rows}:
        family = [
            row
            for row in rows
            if row["scope"] == "condition"
            and row["comparison"] == comparison
        ]
        ordered = sorted(
            family,
            key=lambda row: (
                float(row["paired_t_pvalue"])
                if math.isfinite(float(row["paired_t_pvalue"]))
                else 1.0
            ),
        )
        running = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            raw = float(row["paired_t_pvalue"])
            if not math.isfinite(raw):
                raw = 1.0
            adjusted = min(1.0, raw * (total - rank))
            running = max(running, adjusted)
            row["holm_adjusted_pvalue"] = running
            row["holm_significant_in_favor"] = (
                "yes"
                if running < 0.05
                and float(row["paired_delta_ci95_low"]) > 0
                else "no"
            )
    for row in rows:
        row.setdefault("holm_adjusted_pvalue", math.nan)
        row.setdefault("holm_significant_in_favor", "not_applicable")


def runtime_and_safety_summary(
    conditions: list[dict[str, Any]]
) -> dict[str, dict[str, float | int]]:
    by_controller: dict[str, list[dict[str, str]]] = {}
    for condition in conditions:
        for controller, seed_rows in condition["controllers"].items():
            by_controller.setdefault(controller, []).extend(seed_rows.values())
    result: dict[str, dict[str, float | int]] = {}
    for controller, controller_rows in by_controller.items():
        runtimes = [
            float(row["evaluation_wall_seconds"])
            for row in controller_rows
            if row.get("evaluation_wall_seconds", "")
        ]
        ordered = sorted(runtimes)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        decisions = sum(
            int(float(row.get("policy_decisions", 0) or 0))
            for row in controller_rows
        )
        invalid = sum(
            int(float(row.get("policy_invalid_actions", 0) or 0))
            for row in controller_rows
        )
        switches = sum(
            int(float(row.get("policy_switches", 0) or 0))
            for row in controller_rows
        )
        forced = sum(
            int(float(row.get("policy_forced_switches", 0) or 0))
            for row in controller_rows
        )
        spillback = [
            float(row["mean_tls_spillback"])
            for row in controller_rows
            if row.get("mean_tls_spillback", "")
        ]
        starvation = [
            float(row["mean_tls_starvation"])
            for row in controller_rows
            if row.get("mean_tls_starvation", "")
        ]
        result[controller] = {
            "runs": len(controller_rows),
            "mean_wall_seconds": (
                statistics.fmean(runtimes) if runtimes else math.nan
            ),
            "median_wall_seconds": (
                statistics.median(runtimes) if runtimes else math.nan
            ),
            "p95_wall_seconds": (
                ordered[p95_index] if ordered else math.nan
            ),
            "policy_decisions": decisions,
            "policy_switch_rate": switches / max(1, decisions),
            "policy_forced_switch_rate": forced / max(1, decisions),
            "policy_invalid_action_rate": invalid / max(1, decisions),
            "mean_tls_spillback": (
                statistics.fmean(spillback) if spillback else math.nan
            ),
            "mean_tls_starvation": (
                statistics.fmean(starvation) if starvation else math.nan
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    conditions = []
    fairness_issues: list[dict[str, Any]] = []
    for map_name, rate, path in condition_paths(root):
        controllers = load_condition(path)
        fairness_issues.extend(fairness_checks(controllers, path))
        conditions.append(
            {
                "map": map_name,
                "rate": rate,
                "path": path,
                "controllers": controllers,
            }
        )
    if not conditions:
        raise RuntimeError(f"No */rate_*/paired_runs.csv files under {root}")

    controller_names = set.intersection(
        *(set(condition["controllers"]) for condition in conditions)
    )
    comparisons = list(COMPARISONS)
    for controller in sorted(
        controller_names
        - {"native_sumo", "max_pressure", "all_model"}
    ):
        comparisons.append(
            (controller, "native_sumo", f"{controller}_vs_native")
        )

    rows: list[dict[str, Any]] = []
    print("=" * 148)
    print("PAIRED COMPREHENSIVE EVALUATION")
    print("=" * 148)
    for condition in conditions:
        for candidate, baseline, comparison in comparisons:
            candidate_rows, baseline_rows = paired_or_raise(
                condition["controllers"],
                candidate,
                baseline,
                condition["path"],
            )
            print(
                f"\n{condition['map'].upper()} rate={condition['rate']} | "
                f"{comparison} | {len(candidate_rows)} paired seeds"
            )
            print(
                f"{'metric':<24} {'baseline':>12} {'candidate':>12} "
                f"{'imp.%':>9} {'wins':>12} {'paired delta 95% CI':>30}"
            )
            for metric, direction in METRICS:
                summary = summarize_metric(
                    baseline_rows, candidate_rows, metric, direction
                )
                test_stats = paired_test_stats(
                    baseline_rows, candidate_rows, metric, direction
                )
                row = {
                    "scope": "condition",
                    "map": condition["map"],
                    "rate": condition["rate"],
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
                    **summary,
                    **test_stats,
                }
                rows.append(row)
                win_text = (
                    f"{summary['wins']}/{summary['n']} "
                    f"({summary['losses']}L)"
                )
                print(
                    f"{metric:<24} "
                    f"{float(summary['native_mean']):>12.3f} "
                    f"{float(summary['model_mean']):>12.3f} "
                    f"{float(summary['aggregate_improvement_pct']):>8.2f}% "
                    f"{win_text:>12} "
                    f"[{fmt(float(summary['paired_delta_ci95_low']))}, "
                    f"{fmt(float(summary['paired_delta_ci95_high']))}]"
                )

    for candidate, baseline, comparison in comparisons:
        candidate_pool = pooled_rows(conditions, candidate)
        baseline_pool = pooled_rows(conditions, baseline)
        for metric, direction in METRICS:
            summary = summarize_metric(
                baseline_pool, candidate_pool, metric, direction
            )
            test_stats = paired_test_stats(
                baseline_pool, candidate_pool, metric, direction
            )
            rows.append(
                {
                    "scope": "pooled",
                    "map": "ALL",
                    "rate": "ALL",
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
                    **summary,
                    **test_stats,
                }
            )

    apply_holm_adjustment(rows)

    learned_rows = [
        row
        for row in rows
        if row["scope"] == "condition"
        and row["comparison"] == "learned_vs_native"
    ]
    robustness = {}
    for metric, _direction in METRICS:
        metric_rows = [row for row in learned_rows if row["metric"] == metric]
        finite_rows = [
            row
            for row in metric_rows
            if math.isfinite(float(row["aggregate_improvement_pct"]))
        ]
        worst = min(
            finite_rows or metric_rows,
            key=lambda row: float(row["aggregate_improvement_pct"]),
        )
        robustness[metric] = {
            "conditions": len(metric_rows),
            "clear_positive_conditions": sum(
                float(row["paired_delta_ci95_low"]) > 0 for row in metric_rows
            ),
            "clear_negative_conditions": sum(
                float(row["paired_delta_ci95_high"]) < 0 for row in metric_rows
            ),
            "holm_clear_positive_conditions": sum(
                row["holm_significant_in_favor"] == "yes"
                for row in metric_rows
            ),
            "inconclusive_conditions": sum(
                float(row["paired_delta_ci95_low"]) <= 0
                <= float(row["paired_delta_ci95_high"])
                for row in metric_rows
            ),
            "worst_condition": {
                "map": worst["map"],
                "rate": worst["rate"],
                "improvement_pct": worst["aggregate_improvement_pct"],
            },
            "median_condition_improvement_pct": statistics.median(
                float(row["aggregate_improvement_pct"])
                for row in (finite_rows or metric_rows)
            ),
        }

    fieldnames = list(rows[0].keys())
    output_csv = root / "statistical_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_json = root / "robustness_summary.json"
    operational_summary = runtime_and_safety_summary(conditions)
    output_json.write_text(
        json.dumps(
            {
                "conditions": len(conditions),
                "fairness_passed": not fairness_issues,
                "fairness_issues": fairness_issues,
                "learned_vs_native_robustness": robustness,
                "runtime_and_policy_safety": operational_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nFairness checks: {'PASS' if not fairness_issues else 'FAIL'}")
    print("Runtime/safety summary:")
    for controller, summary in sorted(operational_summary.items()):
        print(
            f"  {controller:<14} "
            f"mean wall={float(summary['mean_wall_seconds']):.2f}s "
            f"invalid={100.0 * float(summary['policy_invalid_action_rate']):.3f}% "
            f"forced={100.0 * float(summary['policy_forced_switch_rate']):.3f}%"
        )
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    if fairness_issues:
        raise RuntimeError(
            f"{len(fairness_issues)} paired-demand/sample fairness checks failed"
        )


if __name__ == "__main__":
    main()
