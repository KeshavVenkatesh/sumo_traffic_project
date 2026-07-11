#!/usr/bin/env bash
set -euo pipefail

cd ~/sumo_traffic_project
source .venv/bin/activate

OUTDIR="split_eval_new_map_40tls"
mkdir -p "$OUTDIR"

SEEDS=(42 43 44)
MAX_PARALLEL=3

MODEL_PATH="models/traffic_signal_maskable_ppo_santaclara_rr_40tls_good"
VEC_PATH="models/traffic_signal_maskable_ppo_santaclara_rr_40tls_good_vecnormalize.pkl"

COMMON_ARGS=(
  --episode-seconds 3600
  --eval-steps 10000
  --max-vehicle-center 1500
  --target-vehicle-center 1500
  --initial-vehicle-center 300
  --spawn-batch-center 20
  --model-update-period 10
  --metrics-interval 10
  --eval-print-every 50
)

launch_allmodel() {
  local seed="$1"

  local log="$OUTDIR/allmodel_seed${seed}.log"
  local csv="$OUTDIR/allmodel_seed${seed}.csv"
  local json="$OUTDIR/allmodel_seed${seed}.json"

  echo "Launching allmodel seed $seed -> $log"

  nohup python -u compare_native_sumo_vs_all_model.py \
    "${COMMON_ARGS[@]}" \
    --compare-seeds "$seed" \
    --skip-native \
    --model-path "$MODEL_PATH" \
    --vecnormalize-path "$VEC_PATH" \
    --stats-csv "$csv" \
    --stats-json "$json" \
    > "$log" 2>&1 &
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
  launch_allmodel "$seed"
done

echo "All all-model jobs launched. Waiting..."
wait
echo "All all-model new-map eval jobs finished."
