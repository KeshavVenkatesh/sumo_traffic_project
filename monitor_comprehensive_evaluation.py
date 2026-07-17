#!/usr/bin/env python3
"""Print one concise progress snapshot for the comprehensive evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "progress_file",
        nargs="?",
        type=Path,
        default=Path("comprehensive_eval_progress.json"),
    )
    args = parser.parse_args()
    if not args.progress_file.exists():
        print("Waiting for evaluation progress...")
        return
    payload = json.loads(args.progress_file.read_text(encoding="utf-8"))
    age = max(0.0, time.time() - float(payload.get("updated_at", 0.0)))
    print(
        f"Status: {payload.get('status', 'unknown')}\n"
        f"Completed: {payload.get('completed_jobs', 0)} / "
        f"{payload.get('total_jobs', 0)}\n"
        f"Progress: {float(payload.get('percent', 0.0)):.2f}%\n"
        f"Failed: {payload.get('failed_jobs', 0)}\n"
        f"Latest: {payload.get('current', '')}\n"
        f"Updated: {age:.1f}s ago"
    )


if __name__ == "__main__":
    main()
