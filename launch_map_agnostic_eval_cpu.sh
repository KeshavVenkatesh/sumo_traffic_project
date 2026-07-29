#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

NET_FILE="${TRAFFIC_NET_FILE:-new_map.net.xml}"
MODEL_PATH="${MODEL_PATH:-models/map_agnostic_multiagent_v3_best}"
SEED_LIST="${SEED_LIST:-42,43,44}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SKIP_NATIVE="${SKIP_NATIVE:-0}"
OUTDIR="${OUTDIR:-map_agnostic_eval_$(date +%Y%m%d_%H%M%S)}"
# Optional directory produced by generate_fixed_demand.py. Each seed then
# replays DEMAND_DIR/seed_<seed>.rou.xml for both controllers.
DEMAND_DIR="${DEMAND_DIR:-}"

EPISODE_SECONDS="${EPISODE_SECONDS:-1200}"
EVAL_STEPS="${EVAL_STEPS:-2500}"
MAX_VEHICLES="${MAX_VEHICLES:-1500}"
TARGET_VEHICLES="${TARGET_VEHICLES:-1500}"
INITIAL_VEHICLES="${INITIAL_VEHICLES:-300}"
SPAWN_BATCH="${SPAWN_BATCH:-20}"
MODEL_UPDATE_PERIOD="${MODEL_UPDATE_PERIOD:-10}"
METRICS_INTERVAL="${METRICS_INTERVAL:-20}"
PRINT_INTERVAL="${PRINT_INTERVAL:-100}"
MERGE_RESULTS="${MERGE_RESULTS:-1}"

mkdir -p "$OUTDIR"
printf '%s\n' "$OUTDIR" > .last_map_agnostic_eval_dir

IFS=',' read -r -a SEEDS <<< "$SEED_LIST"

COMMON_ARGS=(
  --episode-seconds "$EPISODE_SECONDS"
  --eval-steps "$EVAL_STEPS"
  --max-vehicle-center "$MAX_VEHICLES"
  --target-vehicle-center "$TARGET_VEHICLES"
  --initial-vehicle-center "$INITIAL_VEHICLES"
  --spawn-batch-center "$SPAWN_BATCH"
  --model-update-period "$MODEL_UPDATE_PERIOD"
  --metrics-interval "$METRICS_INTERVAL"
  --eval-print-every "$PRINT_INTERVAL"
  --device cpu
)

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || true
  done
}

launch_one() {
  local controller="$1"
  local seed="$2"
  local log="$OUTDIR/${controller}_seed${seed}.log"
  local csv="$OUTDIR/${controller}_seed${seed}.csv"
  local json="$OUTDIR/${controller}_seed${seed}.json"
  local args=("${COMMON_ARGS[@]}" --compare-seeds "$seed" --stats-csv "$csv" --stats-json "$json")

  if [[ -n "$DEMAND_DIR" ]]; then
    local demand_file="$DEMAND_DIR/seed_${seed}.rou.xml"
    if [[ ! -f "$demand_file" ]]; then
      echo "Missing fixed demand route: $demand_file" >&2
      return 2
    fi
    args+=(--demand-route-file "$demand_file")
  fi

  if [[ "$controller" == "native" ]]; then
    args+=(--skip-all-model)
  else
    args+=(--skip-native --model-path "$MODEL_PATH")
  fi

  echo "Launching $controller seed=$seed map=$NET_FILE -> $log"
  TRAFFIC_NET_FILE="$NET_FILE" nohup python -u compare_native_sumo_vs_map_agnostic.py \
    "${args[@]}" > "$log" 2>&1 &
}

for raw_seed in "${SEEDS[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ -n "$seed" ]] || continue
  if [[ "$SKIP_NATIVE" != "1" ]]; then
    wait_for_slot
    launch_one native "$seed"
  fi
  wait_for_slot
  launch_one allmodel "$seed"
done

echo "Jobs launched in $OUTDIR; waiting for completion..."
wait
if [[ "$MERGE_RESULTS" == "1" ]]; then
  TRAFFIC_NET_FILE="$NET_FILE" python -u merge_split_eval_results.py \
    "$OUTDIR" "$OUTDIR/map_agnostic_v3_vs_native.csv" "$OUTDIR/map_agnostic_v3_vs_native.json" \
    > "$OUTDIR/merge.log" 2>&1
fi
echo "Map-agnostic evaluation complete: $OUTDIR"
