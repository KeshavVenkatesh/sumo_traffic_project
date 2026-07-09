from pathlib import Path
import re
import sys
import time

OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("split_eval_40tls")
EVAL_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

TASKS = [
    ("native", 42),
    ("allmodel", 42),
    ("native", 43),
    ("allmodel", 43),
    ("native", 44),
    ("allmodel", 44),
]

PATTERNS = {
    "native": re.compile(r"\[native seed\s+(\d+)\]\s+step=\s*(\d+)"),
    "allmodel": re.compile(r"\[all-model seed\s+(\d+)\]\s+step=\s*(\d+)"),
}

def latest_step(kind, seed):
    log = OUTDIR / f"{kind}_seed{seed}.log"
    js = OUTDIR / f"{kind}_seed{seed}.json"

    if js.exists():
        return EVAL_STEPS

    if not log.exists():
        return 0

    pat = PATTERNS[kind]
    best = 0

    try:
        lines = log.read_text(errors="ignore").splitlines()
    except Exception:
        return 0

    for line in reversed(lines[-1000:]):
        m = pat.search(line)
        if m and int(m.group(1)) == seed:
            best = int(m.group(2))
            break

    return min(best, EVAL_STEPS)

while True:
    total = 0
    parts = []

    for kind, seed in TASKS:
        step = latest_step(kind, seed)
        total += step
        parts.append(f"{kind}{seed}:{100.0 * step / EVAL_STEPS:5.1f}%")

    pct = 100.0 * total / (len(TASKS) * EVAL_STEPS)

    print(f"[progress] {pct:6.2f}% complete | " + " | ".join(parts), flush=True)

    if pct >= 100.0:
        break

    time.sleep(30)
