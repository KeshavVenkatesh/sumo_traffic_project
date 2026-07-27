#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    ("total_arrived", True),
    ("mean_avg_speed_mps", True),
    ("mean_global_queue", False),
    ("max_global_queue", False),
    ("mean_global_wait", False),
    ("max_global_wait", False),
    ("recovered_total", False),
    ("ambulance_completion_rate", True),
    ("mean_ambulance_seconds_per_km", False),
    ("mean_ambulance_stopped_seconds", False),
    ("mean_ambulance_stopped_fraction", False),
    ("mean_ambulance_time_loss_s", False),
)

MODEL_LABELS = (
    "ambulance_aware_final_3m",
    "original_40tls_good",
    "robust_v1",
    "mixed_tls_v1",
)

COMPLETED_AMBULANCE_ONLY_METRICS = {
    "mean_ambulance_trip_time_s",
    "median_ambulance_trip_time_s",
    "p90_ambulance_trip_time_s",
    "mean_ambulance_seconds_per_km",
}


T_CRITICAL_95 = {
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
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def load_runs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"No run rows in {path}")
    return runs


def by_seed(rows: list[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in result:
            raise RuntimeError(f"Duplicate seed {seed}")
        result[seed] = row
    return result


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def paired_summary(
    map_name: str,
    model_label: str,
    metric: str,
    higher_is_better: bool,
    native_by_seed: dict[int, dict],
    model_by_seed: dict[int, dict],
) -> dict:
    native_seeds = set(native_by_seed)
    model_seeds = set(model_by_seed)
    common = sorted(native_seeds & model_seeds)
    if native_seeds != model_seeds:
        missing_model = sorted(native_seeds - model_seeds)
        missing_native = sorted(model_seeds - native_seeds)
        raise RuntimeError(
            f"Seed mismatch for {map_name}/{model_label}: "
            f"missing model={missing_model}, missing native={missing_native}"
        )
    excluded_pairs = 0
    if metric in COMPLETED_AMBULANCE_ONLY_METRICS:
        eligible = [
            seed
            for seed in common
            if int(native_by_seed[seed].get("ambulance_completed_total", 0)) > 0
            and int(model_by_seed[seed].get("ambulance_completed_total", 0)) > 0
        ]
        excluded_pairs = len(common) - len(eligible)
        common = eligible

    if not common:
        return {
            "map": map_name,
            "model": model_label,
            "metric": metric,
            "direction": "higher" if higher_is_better else "lower",
            "paired_seeds": 0,
            "excluded_pairs": excluded_pairs,
            "native_mean": float("nan"),
            "model_mean": float("nan"),
            "improvement_percent": float("nan"),
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "paired_delta_mean": float("nan"),
            "paired_delta_sd": float("nan"),
            "paired_delta_ci95_low": float("nan"),
            "paired_delta_ci95_high": float("nan"),
            "paired_effect_dz": float("nan"),
            "verdict": "insufficient",
        }

    native_values = [float(native_by_seed[s][metric]) for s in common]
    model_values = [float(model_by_seed[s][metric]) for s in common]
    if higher_is_better:
        deltas = [m - n for n, m in zip(native_values, model_values)]
    else:
        deltas = [n - m for n, m in zip(native_values, model_values)]

    n = len(deltas)
    delta_mean = mean(deltas)
    if n >= 2:
        delta_sd = statistics.stdev(deltas)
        standard_error = delta_sd / math.sqrt(n)
        margin = t_critical_95(n - 1) * standard_error
        ci_low = delta_mean - margin
        ci_high = delta_mean + margin
    else:
        delta_sd = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    native_mean = mean(native_values)
    model_mean = mean(model_values)
    if native_mean == 0.0:
        improvement_percent = float("nan")
    elif higher_is_better:
        improvement_percent = 100.0 * (model_mean - native_mean) / abs(native_mean)
    else:
        improvement_percent = 100.0 * (native_mean - model_mean) / abs(native_mean)

    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = n - wins - losses

    if n < 2:
        verdict = "insufficient"
    elif ci_low > 0.0:
        verdict = "better"
    elif ci_high < 0.0:
        verdict = "worse"
    else:
        verdict = "unclear"

    if n < 2:
        paired_dz = float("nan")
    elif delta_sd > 0.0:
        paired_dz = delta_mean / delta_sd
    elif delta_mean == 0.0:
        paired_dz = 0.0
    else:
        paired_dz = math.copysign(float("inf"), delta_mean)

    return {
        "map": map_name,
        "model": model_label,
        "metric": metric,
        "direction": "higher" if higher_is_better else "lower",
        "paired_seeds": n,
        "excluded_pairs": excluded_pairs,
        "native_mean": native_mean,
        "model_mean": model_mean,
        "improvement_percent": improvement_percent,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_delta_mean": delta_mean,
        "paired_delta_sd": delta_sd,
        "paired_delta_ci95_low": ci_low,
        "paired_delta_ci95_high": ci_high,
        "paired_effect_dz": paired_dz,
        "verdict": verdict,
    }


def render_table(rows: list[dict], map_name: str, model_label: str) -> str:
    selected = [
        row for row in rows
        if row["map"] == map_name and row["model"] == model_label
    ]
    total_n = max(
        row["paired_seeds"] + row.get("excluded_pairs", 0)
        for row in selected
    )
    lines = [
        "=" * 148,
        f"{map_name.upper()} / {model_label} / {total_n} requested paired seeds",
        "=" * 148,
        (
            f"{'metric':26s} {'native':>11s} {'model':>11s} {'imp.%':>9s} "
            f"{'wins':>14s} {'n':>4s} {'paired delta 95% CI':>28s} {'effect dz':>11s} {'verdict':>12s}"
        ),
        "-" * 148,
    ]
    for row in selected:
        wins = f"{row['wins']}W/{row['losses']}L/{row['ties']}T"
        ci = (
            f"[{row['paired_delta_ci95_low']:.3f}, "
            f"{row['paired_delta_ci95_high']:.3f}]"
        )
        lines.append(
            f"{row['metric']:26s} "
            f"{row['native_mean']:11.3f} "
            f"{row['model_mean']:11.3f} "
            f"{row['improvement_percent']:8.2f}% "
            f"{wins:>14s} {row['paired_seeds']:4d} {ci:>28s} "
            f"{row['paired_effect_dz']:11.3f} {row['verdict']:>12s}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired-seed statistics for the two-map/four-model evaluation."
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="Evaluation root; defaults to latest_two_map_four_model_eval_root.txt",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        pointer = Path(__file__).resolve().parent / "latest_two_map_four_model_eval_root.txt"
        root = Path(pointer.read_text(encoding="utf-8").strip()).resolve()

    all_rows: list[dict] = []
    map_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    for map_name in map_names:
        native_path = root / map_name / "native_sumo" / "merged.json"
        if not native_path.is_file():
            raise FileNotFoundError(native_path)
        native_by_seed = by_seed(load_runs(native_path))

        for model_label in MODEL_LABELS:
            model_path = root / map_name / model_label / "merged.json"
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            model_by_seed = by_seed(load_runs(model_path))

            for metric, higher_is_better in METRICS:
                all_rows.append(
                    paired_summary(
                        map_name,
                        model_label,
                        metric,
                        higher_is_better,
                        native_by_seed,
                        model_by_seed,
                    )
                )

    csv_path = root / "statistical_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    sections = []
    for map_name in map_names:
        for model_label in MODEL_LABELS:
            sections.append(render_table(all_rows, map_name, model_label))
    sections.extend(
        [
            "=" * 148,
            "Positive improvement and positive paired deltas always favor the learned model.",
            "The confidence interval is a paired, two-sided 95% Student-t interval.",
            "Effect dz is paired Cohen's dz: positive favors the learned model.",
            "Ambulance occupancy is deliberately excluded: it is not a response-time metric.",
            "Completed-trip time distributions and ambulance event counts remain available in each merged CSV/JSON.",
            f"Wrote: {csv_path}",
        ]
    )
    report = "\n\n".join(sections) + "\n"
    report_path = root / "statistical_summary.log"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
