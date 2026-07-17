#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys


METRICS = [
    ("total_arrived", "higher"),
    ("mean_avg_speed_mps", "higher"),
    ("mean_global_queue", "lower"),
    ("max_global_queue", "lower"),
    ("mean_global_wait", "lower"),
    ("max_global_wait", "lower"),
    ("recovered_total", "lower"),
]


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
        raise ValueError("At least two paired seeds are required for a confidence interval")
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


def load_paired_rows(path: Path) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    native: dict[int, dict[str, str]] = {}
    model: dict[int, dict[str, str]] = {}

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            controller = row["controller"]
            if controller == "native_sumo":
                native[seed] = row
            elif controller == "all_model":
                model[seed] = row

    if not native or not model:
        raise ValueError(f"Missing native_sumo or all_model rows in {path}")

    if set(native) != set(model):
        native_only = sorted(set(native) - set(model))
        model_only = sorted(set(model) - set(native))
        raise ValueError(
            f"Unpaired seeds in {path}: native-only={native_only}, model-only={model_only}"
        )

    return native, model


def summarize_metric(
    native: dict[int, dict[str, str]],
    model: dict[int, dict[str, str]],
    metric: str,
    direction: str,
) -> dict[str, float | int | str]:
    seeds = sorted(native)
    native_values = [float(native[seed][metric]) for seed in seeds]
    model_values = [float(model[seed][metric]) for seed in seeds]

    if direction == "higher":
        paired_deltas = [m - n for n, m in zip(native_values, model_values)]
    else:
        paired_deltas = [n - m for n, m in zip(native_values, model_values)]

    paired_percentages = [
        100.0 * delta / native_value if native_value != 0 else math.nan
        for delta, native_value in zip(paired_deltas, native_values)
    ]
    finite_percentages = [value for value in paired_percentages if math.isfinite(value)]

    n = len(seeds)
    native_mean = statistics.fmean(native_values)
    model_mean = statistics.fmean(model_values)
    delta_mean = statistics.fmean(paired_deltas)
    delta_sd = statistics.stdev(paired_deltas) if n > 1 else math.nan
    delta_se = delta_sd / math.sqrt(n) if n > 1 else math.nan

    if n > 1:
        margin = t_critical_95(n - 1) * delta_se
        ci_low = delta_mean - margin
        ci_high = delta_mean + margin
    else:
        ci_low = math.nan
        ci_high = math.nan

    if abs(native_mean) < 1e-12:
        aggregate_improvement = math.nan
    elif direction == "higher":
        aggregate_improvement = 100.0 * (model_mean - native_mean) / native_mean
    else:
        aggregate_improvement = 100.0 * (native_mean - model_mean) / native_mean

    wins = sum(delta > 0 for delta in paired_deltas)
    losses = sum(delta < 0 for delta in paired_deltas)
    ties = n - wins - losses

    return {
        "metric": metric,
        "direction": direction,
        "n": n,
        "native_mean": native_mean,
        "model_mean": model_mean,
        "aggregate_improvement_pct": aggregate_improvement,
        "mean_paired_improvement_pct": (
            statistics.fmean(finite_percentages) if finite_percentages else math.nan
        ),
        "paired_delta_mean": delta_mean,
        "paired_delta_sd": delta_sd,
        "paired_delta_ci95_low": ci_low,
        "paired_delta_ci95_high": ci_high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "ci_excludes_zero_in_favor_of_model": (
            "yes" if math.isfinite(ci_low) and ci_low > 0 else "no"
        ),
    }


def find_comparison_files(root: Path) -> list[tuple[str, str, Path]]:
    comparisons = []
    for map_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for csv_path in sorted(map_dir.glob("*_vs_native.csv")):
            model_tag = csv_path.name.removesuffix("_vs_native.csv")
            comparisons.append((map_dir.name, model_tag, csv_path))
    return comparisons


def format_number(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute paired 95% confidence intervals for multi-seed SUMO evaluations."
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="Evaluation directory. Defaults to .last_two_map_eval_dir.",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        pointer = Path(".last_two_map_eval_dir")
        if not pointer.exists():
            print("Missing .last_two_map_eval_dir", file=sys.stderr)
            return 1
        root = Path(pointer.read_text().strip())

    comparisons = find_comparison_files(root)
    if not comparisons:
        print(f"No *_vs_native.csv files found under {root}", file=sys.stderr)
        return 1

    output_rows = []

    for map_tag, model_tag, path in comparisons:
        native, model = load_paired_rows(path)
        print()
        print("=" * 132)
        print(f"{map_tag.upper()} / {model_tag} / {len(native)} paired seeds")
        print("=" * 132)
        print(
            f"{'metric':<24} {'native':>12} {'model':>12} {'imp.%':>9} "
            f"{'wins':>11} {'paired delta 95% CI':>32} {'clear?':>8}"
        )
        print("-" * 132)

        for metric, direction in METRICS:
            summary = summarize_metric(native, model, metric, direction)
            output_rows.append({"map": map_tag, "model": model_tag, **summary})

            ci_text = (
                f"[{format_number(float(summary['paired_delta_ci95_low']))}, "
                f"{format_number(float(summary['paired_delta_ci95_high']))}]"
            )
            win_text = (
                f"{summary['wins']}/{summary['n']}"
                f" ({summary['losses']}L)"
            )
            print(
                f"{metric:<24} "
                f"{float(summary['native_mean']):>12.3f} "
                f"{float(summary['model_mean']):>12.3f} "
                f"{float(summary['aggregate_improvement_pct']):>8.2f}% "
                f"{win_text:>11} "
                f"{ci_text:>32} "
                f"{summary['ci_excludes_zero_in_favor_of_model']:>8}"
            )

    output_path = root / "statistical_summary.csv"
    fieldnames = [
        "map",
        "model",
        "metric",
        "direction",
        "n",
        "native_mean",
        "model_mean",
        "aggregate_improvement_pct",
        "mean_paired_improvement_pct",
        "paired_delta_mean",
        "paired_delta_sd",
        "paired_delta_ci95_low",
        "paired_delta_ci95_high",
        "wins",
        "losses",
        "ties",
        "ci_excludes_zero_in_favor_of_model",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print(f"Wrote: {output_path}")
    print(
        "Positive improvement and positive paired deltas always favor the learned model. "
        "The CI column is a paired two-sided 95% Student-t confidence interval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
