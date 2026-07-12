#!/usr/bin/env bash
set -euo pipefail

cd ~/sumo_traffic_project
source .venv/bin/activate

export SUMO_HOME="$PWD/.venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

SIM_FILE="realistic_all_intersections_fixed_cycle.py"
BACKUP_FILE="$(mktemp /tmp/realistic_all_intersections_fixed_cycle.XXXXXX.py)"

cp "$SIM_FILE" "$BACKUP_FILE"

restore_sim_file() {
    cp "$BACKUP_FILE" "$SIM_FILE"
    rm -f "$BACKUP_FILE"
}

trap restore_sim_file EXIT INT TERM

# The current snapshot points to new_map.net.xml.
# Temporarily use Santa Clara for this evaluation.
python - <<'PY'
from pathlib import Path

p = Path("realistic_all_intersections_fixed_cycle.py")
s = p.read_text()

new_map_line = 'NET_FILE = os.path.join(BASE_DIR, "new_map.net.xml")'
santa_clara_line = 'NET_FILE = os.path.join(BASE_DIR, "santa_clara.net.xml")'

if new_map_line in s:
    s = s.replace(new_map_line, santa_clara_line, 1)
elif santa_clara_line not in s:
    raise RuntimeError(
        "Could not safely find the NET_FILE assignment. "
        "Current NET_FILE lines:\n"
        + "\n".join(
            line for line in s.splitlines()
            if "NET_FILE" in line and "=" in line
        )
    )

p.write_text(s)
PY

python -m py_compile realistic_all_intersections_fixed_cycle.py

echo "Evaluation map:"
grep -n "^NET_FILE" realistic_all_intersections_fixed_cycle.py

test -f models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1.zip
test -f models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1_vecnormalize.pkl

python -u compare_native_sumo_vs_all_model.py \
  --episode-seconds 1200 \
  --eval-steps 2500 \
  --compare-seeds 42,43 \
  --max-vehicle-center 1500 \
  --target-vehicle-center 1500 \
  --initial-vehicle-center 300 \
  --spawn-batch-center 20 \
  --model-path models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1 \
  --vecnormalize-path models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1_vecnormalize.pkl \
  --model-update-period 10 \
  --metrics-interval 20 \
  --eval-print-every 100 \
  --stats-csv native_vs_model_1500_santaclara_robust_v1_screen.csv \
  --stats-json native_vs_model_1500_santaclara_robust_v1_screen.json
