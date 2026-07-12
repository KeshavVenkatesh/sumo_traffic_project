#!/usr/bin/env bash
set -uo pipefail

cd ~/sumo_traffic_project
source .venv/bin/activate

export SUMO_HOME="$PWD/.venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export TRAFFIC_NET_FILE="santa_clara.net.xml"

# Change when launching with:
# MAX_PARALLEL=12 nohup bash ...
MAX_PARALLEL="${MAX_PARALLEL:-8}"

SEEDS=(42 43 44)

TAGS=(
  original_40tls_good
  robust_v1
  mixed_tls_v1
)

declare -A MODEL_PATHS

MODEL_PATHS[original_40tls_good]="models/traffic_signal_maskable_ppo_santaclara_rr_40tls_good"
MODEL_PATHS[robust_v1]="models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1"
MODEL_PATHS[mixed_tls_v1]="models/traffic_signal_maskable_ppo_santaclara_mixed_tls_v1"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-parallel_eval_all_three_santaclara_${RUN_ID}}"

mkdir -p "$OUTROOT/native"

for tag in "${TAGS[@]}"; do
  mkdir -p "$OUTROOT/$tag"

  model="${MODEL_PATHS[$tag]}"

  if [[ ! -f "${model}.zip" ]]; then
    echo "Missing model: ${model}.zip"
    exit 1
  fi

  if [[ ! -f "${model}_vecnormalize.pkl" ]]; then
    echo "Missing normalization file: ${model}_vecnormalize.pkl"
    exit 1
  fi
done

printf '%s\n' "$OUTROOT" > .last_three_model_eval_dir

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

PIDS=()
NAMES=()

wait_for_slot() {
  while true; do
    running="$(jobs -rp | wc -l)"

    if (( running < MAX_PARALLEL )); then
      return
    fi

    sleep 5
  done
}

launch_native() {
  local seed="$1"
  local name="native_seed${seed}"
  local dir="$OUTROOT/native"

  echo "Launching $name"

  nohup env TRAFFIC_NET_FILE="$TRAFFIC_NET_FILE" \
    python -u compare_native_sumo_vs_all_model.py \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-all-model \
      --stats-csv "$dir/${name}.csv" \
      --stats-json "$dir/${name}.json" \
      > "$dir/${name}.log" 2>&1 &

  PIDS+=("$!")
  NAMES+=("$name")
}

launch_model() {
  local tag="$1"
  local seed="$2"
  local model="${MODEL_PATHS[$tag]}"
  local name="${tag}_seed${seed}"
  local dir="$OUTROOT/$tag"

  echo "Launching $name"

  nohup env TRAFFIC_NET_FILE="$TRAFFIC_NET_FILE" \
    python -u compare_native_sumo_vs_all_model.py \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-native \
      --model-path "$model" \
      --vecnormalize-path "${model}_vecnormalize.pkl" \
      --stats-csv "$dir/allmodel_seed${seed}.csv" \
      --stats-json "$dir/allmodel_seed${seed}.json" \
      > "$dir/allmodel_seed${seed}.log" 2>&1 &

  PIDS+=("$!")
  NAMES+=("$name")
}

echo "Output directory: $OUTROOT"
echo "Maximum simultaneous jobs: $MAX_PARALLEL"
echo

# Four jobs per seed:
# native + original + robust + mixed.
for seed in "${SEEDS[@]}"; do
  wait_for_slot
  launch_native "$seed"

  for tag in "${TAGS[@]}"; do
    wait_for_slot
    launch_model "$tag" "$seed"
  done
done

echo
echo "All 12 jobs have been submitted."
echo "Waiting for completion..."

failed=0

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"

  if wait "$pid"; then
    echo "FINISHED: $name"
  else
    echo "FAILED:   $name"
    failed=1
  fi
done

echo
echo "Parallel evaluation complete."
echo "Results: $OUTROOT"

if (( failed != 0 )); then
  echo "At least one job failed. Inspect the corresponding log."
fi

exit "$failed"
