#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 MODEL_PATH_WITHOUT_ZIP OUTPUT_TAG"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_TAG="$2"

cd ~/sumo_traffic_project
source .venv/bin/activate

export SUMO_HOME="$PWD/.venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

MODEL_FILE="${MODEL_PATH}.zip"
VEC_FILE="${MODEL_PATH}_vecnormalize.pkl"

test -f "$MODEL_FILE" || {
    echo "Missing model: $MODEL_FILE"
    exit 1
}

test -f "$VEC_FILE" || {
    echo "Missing VecNormalize file: $VEC_FILE"
    exit 1
}

SIM_FILE="realistic_all_intersections_fixed_cycle.py"
BACKUP_FILE="$(mktemp /tmp/realistic_santaclara_eval.XXXXXX.py)"

cp "$SIM_FILE" "$BACKUP_FILE"

restore_sim_file() {
    cp "$BACKUP_FILE" "$SIM_FILE"
    rm -f "$BACKUP_FILE"
}

trap restore_sim_file EXIT INT TERM

python - <<'PY'
from pathlib import Path
import re

path = Path("realistic_all_intersections_fixed_cycle.py")
text = path.read_text()

patterns = [
    (
        r'(?m)^(\s*NET_FILE\s*=\s*os\.path\.join\(\s*BASE_DIR\s*,\s*)'
        r'["\'][^"\']+\.net\.xml["\'](\s*\).*)$',
        r'\1"santa_clara.net.xml"\2',
    ),
    (
        r'(?m)^(\s*NET_FILE\s*=\s*)["\'][^"\']+\.net\.xml["\'](.*)$',
        r'\1"santa_clara.net.xml"\2',
    ),
]

for pattern, replacement in patterns:
    text, count = re.subn(pattern, replacement, text, count=1)
    if count:
        break
else:
    raise RuntimeError("Could not safely locate NET_FILE assignment.")

path.write_text(text)
PY

python -m py_compile realistic_all_intersections_fixed_cycle.py

echo
echo "================================================================================"
echo "Evaluating: $MODEL_PATH"
echo "Output tag: $OUTPUT_TAG"
echo "================================================================================"

grep -n "^NET_FILE" realistic_all_intersections_fixed_cycle.py

python -u compare_native_sumo_vs_all_model.py \
  --episode-seconds 1200 \
  --eval-steps 2500 \
  --compare-seeds 42,43 \
  --max-vehicle-center 1500 \
  --target-vehicle-center 1500 \
  --initial-vehicle-center 300 \
  --spawn-batch-center 20 \
  --model-path "$MODEL_PATH" \
  --vecnormalize-path "$VEC_FILE" \
  --model-update-period 10 \
  --metrics-interval 20 \
  --eval-print-every 100 \
  --stats-csv "native_vs_model_1500_santaclara_${OUTPUT_TAG}_screen.csv" \
  --stats-json "native_vs_model_1500_santaclara_${OUTPUT_TAG}_screen.json"
