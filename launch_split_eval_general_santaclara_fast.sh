#!/usr/bin/env bash
set -euo pipefail

cd ~/sumo_traffic_project
source .venv/bin/activate

export SUMO_HOME="/users/sriramv/sumo_traffic_project/.venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

OUTDIR="split_eval_general_santaclara_fast"
mkdir -p "$OUTDIR"

SEEDS=(42 43)
MAX_PARALLEL=4

MODEL_PATH="models/traffic_signal_maskable_ppo_general_rr"
VEC_PATH="models/traffic_signal_maskable_ppo_general_rr_vecnormalize.pkl"

COMMON_ARGS=(
  --episode-seconds 1200
  --eval-steps 2500
  --max-vehicle-center 1500
  --target-vehicle-center 1500
  --initial-vehicle-center 300
  --spawn-batch-center 20
  --model-update-period 10
  --metrics-interval 20
  --eval-print-every 100
)

launch_job() {
  local kind="$1"
  local seed="$2"

  local log="$OUTDIR/${kind}_seed${seed}.log"
  local csv="$OUTDIR/${kind}_seed${seed}.csv"
  local json="$OUTDIR/${kind}_seed${seed}.json"

  echo "Launching $kind seed $seed -> $log"

  if [[ "$kind" == "native" ]]; then
    nohup python -u compare_native_sumo_vs_all_model_general.py \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-all-model \
      --stats-csv "$csv" \
      --stats-json "$json" \
      > "$log" 2>&1 &
  else
    nohup python -u compare_native_sumo_vs_all_model_general.py \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-native \
      --model-path "$MODEL_PATH" \
      --vecnormalize-path "$VEC_PATH" \
      --stats-csv "$csv" \
      --stats-json "$json" \
      > "$log" 2>&1 &
  fi
}

wait_for_slot() {
  while true; do
    running=$(jobs -rp | wc -l)
    if (( running < MAX_PARALLEL )); then
      break
    fi
    sleep 10
  done
}

for seed in "${SEEDS[@]}"; do
  wait_for_slot
  launch_job native "$seed"

  wait_for_slot
  launch_job allmodel "$seed"
done

echo "All fast general-model eval jobs launched. Waiting..."
wait
echo "All fast general-model eval jobs finished."
