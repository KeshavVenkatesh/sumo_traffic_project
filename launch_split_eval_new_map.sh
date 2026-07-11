#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

OUTDIR="split_eval_new_map_40tls"
mkdir -p "$OUTDIR"

SEEDS=(42 43 44)

# Number of simultaneous SUMO jobs.
# Use 4 first. If machine is fine, change to 6 later.
MAX_PARALLEL=4

MODEL_PATH="models/traffic_signal_maskable_ppo_sf compare_native_sumo_vs_all_model.py
cat > launch_split_eval_new_map.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

OUTDIR="split_eval_new_map_40tls"
mkdir -p "$OUTDIR"

SEEDS=(42 43 44)

# Number of simultaneous SUMO jobsantaclara_rr_40tls_good"
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

launch_job() {
  local kind="$1"
  local seed="$2"

  local log="$OUTDIR/${kind}_seed${seed}.log"
  local csv="$OUTDIR/${kind}_seed${seed}.csv"
  local json="$OUTDIR/${kind}_seed${seed}.json"

  echo "Launching $kind seed $seed -> $log"

  if [[ "$kind" == "native" ]]; then
    nohup python -u compare_native_sumo_vs_all_model.py \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-all-model \
      --stats-csv "$csv" \
      --stats-json "$json" \
      > "$log" 2>&1 &
  else
    nohup python -u compare_native_sumo_vs_all_model.py \
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

echo "All jobs launched. Waiting..."
wait
echo "All split new-map eval jobs finished."
