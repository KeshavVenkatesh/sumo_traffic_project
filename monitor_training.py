#!/usr/bin/env python3

import re
import subprocess
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("/users/sriramv/sumo_traffic_project/train_traffic_signal_rl_numenv4.log")

# Settings from your current training command.
TARGET_TIMESTEPS = 2_700_000
N_STEPS = 4096
NUM_ENVS = 4
ROLLOUT_SIZE = N_STEPS * NUM_ENVS

STAGE_TARGETS = {
    "dense_warmup": 240_000,
    "simulation_center": 1_560_000,
    "stress_generalization": 599_999,
    "sim_polish": 300_001,
}

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "unknown"

    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def progress_bar(value, total, width=40):
    if total <= 0:
        return "[" + "-" * width + "]"

    fraction = max(0.0, min(1.0, value / total))
    filled = round(width * fraction)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def get_last_metric(text, metric):
    pattern = rf"\|\s*{re.escape(metric)}\s*\|\s*([^|\n]+?)\s*\|"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def parse_integer(value):
    if value is None:
        return None

    try:
        return int(float(value.replace(",", "").strip()))
    except ValueError:
        return None


def parse_float(value):
    if value is None:
        return None

    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None


def find_training_process():
    result = subprocess.run(
        ["pgrep", "-fo", r"python.*train_traffic_signal_rl\.py"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None, None

    pid = result.stdout.strip()

    status = subprocess.run(
        [
            "ps",
            "-p",
            pid,
            "-o",
            "pid=,etime=,%cpu=,%mem=,rss=,stat=",
        ],
        capture_output=True,
        text=True,
    )

    return pid, status.stdout.strip() if status.returncode == 0 else None


def main():
    print(f"Traffic RL training dashboard — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    pid, process_status = find_training_process()

    if pid is None:
        print("PROCESS: not running")
    else:
        print(f"PROCESS: running, PID {pid}")
        if process_status:
            print("PID       ELAPSED  %CPU  %MEM      RSS  STATE")
            print(process_status)

    print()

    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        return

    raw_text = LOG_FILE.read_text(errors="replace")
    text = ANSI_ESCAPE.sub("", raw_text).replace("\r", "\n")

    timestep_matches = list(
        re.finditer(
            r"\|\s*total_timesteps\s*\|\s*([0-9,]+)\s*\|",
            text,
        )
    )

    if not timestep_matches:
        print("No total_timesteps metric has been written yet.")
        print()
        print("Latest log output:")
        print("-" * 78)
        print("\n".join(text.splitlines()[-30:]))
        return

    timestep_values = [
        int(match.group(1).replace(",", ""))
        for match in timestep_matches
    ]

    current_lifetime_steps = timestep_values[-1]

    # When --resume is used, SB3's total_timesteps includes the steps from the
    # previously loaded model. The first logged value is normally one rollout
    # after that starting value, so subtract one rollout to infer the baseline.
    inferred_start_steps = max(0, timestep_values[0] - ROLLOUT_SIZE)
    run_steps = max(0, current_lifetime_steps - inferred_start_steps)

    overall_percent = min(100.0, run_steps / TARGET_TIMESTEPS * 100.0)
    remaining_steps = max(0, TARGET_TIMESTEPS - run_steps)

    stage_matches = list(
        re.finditer(
            r"Starting stage\s+(\d+)/(\d+):\s*([^\n]+)",
            text,
        )
    )

    stage_name = "unknown"
    stage_number = None
    stage_count = None
    stage_steps = None
    stage_target = None
    stage_percent = None

    if stage_matches:
        latest_stage = stage_matches[-1]
        stage_number = int(latest_stage.group(1))
        stage_count = int(latest_stage.group(2))
        stage_name = latest_stage.group(3).strip()

        # Find the last logged timestep before this stage started.
        steps_before_stage = [
            int(match.group(1).replace(",", ""))
            for match in timestep_matches
            if match.start() < latest_stage.start()
        ]

        stage_start_lifetime = (
            steps_before_stage[-1]
            if steps_before_stage
            else inferred_start_steps
        )

        stage_steps = max(0, current_lifetime_steps - stage_start_lifetime)
        stage_target = STAGE_TARGETS.get(stage_name)

        if stage_target:
            stage_percent = min(100.0, stage_steps / stage_target * 100.0)

    fps = parse_float(get_last_metric(text, "fps"))
    eta_seconds = remaining_steps / fps if fps and fps > 0 else None

    print("OVERALL TRAINING")
    print(
        f"{progress_bar(run_steps, TARGET_TIMESTEPS)} "
        f"{overall_percent:6.2f}%"
    )
    print(f"Run progress:       {run_steps:,} / {TARGET_TIMESTEPS:,} timesteps")
    print(f"Remaining:          {remaining_steps:,} timesteps")
    print(f"Resume baseline:    {inferred_start_steps:,} previous timesteps")
    print(f"Lifetime timesteps: {current_lifetime_steps:,}")
    print(f"Estimated ETA:      {format_duration(eta_seconds)}")
    print()

    print("CURRENT CURRICULUM STAGE")
    if stage_number is not None:
        print(f"Stage:              {stage_number}/{stage_count} — {stage_name}")

        if stage_target:
            print(
                f"{progress_bar(stage_steps, stage_target)} "
                f"{stage_percent:6.2f}%"
            )
            print(f"Stage progress:     {stage_steps:,} / {stage_target:,} timesteps")
    else:
        print("Stage marker has not appeared in the log yet.")

    print()
    print("LATEST PPO METRICS")

    metrics = [
        ("ep_rew_mean", "Mean episode reward"),
        ("ep_len_mean", "Mean episode length"),
        ("fps", "Training FPS"),
        ("iterations", "PPO iterations"),
        ("time_elapsed", "Stage time elapsed"),
        ("learning_rate", "Learning rate"),
        ("approx_kl", "Approximate KL"),
        ("clip_fraction", "Clip fraction"),
        ("entropy_loss", "Entropy loss"),
        ("explained_variance", "Explained variance"),
        ("policy_gradient_loss", "Policy gradient loss"),
        ("value_loss", "Value loss"),
        ("loss", "Total loss"),
        ("n_updates", "Gradient updates"),
    ]

    found_any = False
    for key, label in metrics:
        value = get_last_metric(text, key)
        if value is not None:
            found_any = True
            print(f"{label:<24} {value}")

    if not found_any:
        print("No PPO metric table has been written yet.")

    print()
    print("The screen refreshes every 10 seconds. Press Ctrl+C to exit.")
    print("This monitor does not stop or modify training.")


if __name__ == "__main__":
    main()
