#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root, regardless of the caller's current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "rl-env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "rl-env/bin/activate"
else
  echo "Could not find .venv/bin/activate or rl-env/bin/activate."
  exit 1
fi

if [[ -z "${SUMO_HOME:-}" ]]; then
  sumo_candidate="$PWD/.venv/lib/python3.10/site-packages/sumo"
  if [[ -d "$sumo_candidate" ]]; then
    export SUMO_HOME="$sumo_candidate"
  fi
fi

if [[ -n "${SUMO_HOME:-}" ]]; then
  export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
fi

EVALUATOR="compare_native_sumo_vs_all_model.py"
MERGER="merge_split_eval_results.py"

for required_file in "$EVALUATOR" "$MERGER" "realistic_all_intersections_fixed_cycle.py"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file"
    exit 1
  fi
done

if ! grep -q "TRAFFIC_NET_FILE" realistic_all_intersections_fixed_cycle.py; then
  echo "realistic_all_intersections_fixed_cycle.py is not map-selection safe."
  echo "Its NET_FILE assignment must read the TRAFFIC_NET_FILE environment variable."
  exit 1
fi

# Map paths can be overridden when launching the script.
SANTACLARA_MAP="${SANTACLARA_MAP:-santa_clara.net.xml}"
FREMONT_MAP="${FREMONT_MAP:-new_map.net.xml}"

if [[ ! -f "$SANTACLARA_MAP" && -f "maps/santa_clara/santa_clara.net.xml" ]]; then
  SANTACLARA_MAP="maps/santa_clara/santa_clara.net.xml"
fi

if [[ ! -f "$FREMONT_MAP" && -f "maps/original_map/new_map.net.xml" ]]; then
  FREMONT_MAP="maps/original_map/new_map.net.xml"
fi

declare -A MAP_FILES
MAP_FILES[santaclara]="$SANTACLARA_MAP"
MAP_FILES[fremont]="$FREMONT_MAP"

if [[ -n "${MAP_LIST:-}" ]]; then
  normalized_maps="${MAP_LIST//,/ }"
  read -r -a MAP_TAGS <<< "$normalized_maps"
else
  MAP_TAGS=(santaclara fremont)
fi

for map_tag in "${MAP_TAGS[@]}"; do
  if [[ -z "${MAP_FILES[$map_tag]+set}" ]]; then
    echo "Unknown map tag '$map_tag'. Use santaclara, fremont, or both."
    exit 1
  fi

  if [[ ! -f "${MAP_FILES[$map_tag]}" ]]; then
    echo "Missing map for $map_tag: ${MAP_FILES[$map_tag]}"
    exit 1
  fi
done

MODEL_TAGS=(
  original_40tls_good
  robust_v1
  mixed_tls_v1
)

declare -A MODEL_PATHS
MODEL_PATHS[original_40tls_good]="models/traffic_signal_maskable_ppo_santaclara_rr_40tls_good"
MODEL_PATHS[robust_v1]="models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1"
MODEL_PATHS[mixed_tls_v1]="models/traffic_signal_maskable_ppo_santaclara_mixed_tls_v1"

for model_tag in "${MODEL_TAGS[@]}"; do
  model_path="${MODEL_PATHS[$model_tag]}"

  if [[ ! -f "${model_path}.zip" ]]; then
    echo "Missing model: ${model_path}.zip"
    exit 1
  fi

  if [[ ! -f "${model_path}_vecnormalize.pkl" ]]; then
    echo "Missing normalization file: ${model_path}_vecnormalize.pkl"
    exit 1
  fi
done

# Explicit SEED_LIST takes precedence. Examples:
#   SEED_LIST="42,43,44"
#   SEED_LIST="42 43 44"
if [[ -n "${SEED_LIST:-}" ]]; then
  normalized_seeds="${SEED_LIST//,/ }"
  read -r -a SEEDS <<< "$normalized_seeds"
else
  SEED_START="${SEED_START:-42}"
  SEED_END="${SEED_END:-61}"

  if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]]; then
    echo "SEED_START and SEED_END must be nonnegative integers."
    exit 1
  fi

  if (( SEED_END < SEED_START )); then
    echo "SEED_END must be at least SEED_START."
    exit 1
  fi

  mapfile -t SEEDS < <(seq "$SEED_START" "$SEED_END")
fi

if (( ${#SEEDS[@]} == 0 )); then
  echo "No seeds were selected."
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed"
    exit 1
  fi
done

# SUMO/TraCI is CPU-heavy. Increase this only when the server has enough cores
# and memory. Every job is still submitted; this limits simultaneous jobs.
MAX_PARALLEL="${MAX_PARALLEL:-16}"

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer."
  exit 1
fi

EPISODE_SECONDS="${EPISODE_SECONDS:-1200}"
EVAL_STEPS="${EVAL_STEPS:-2500}"
MAX_VEHICLE_CENTER="${MAX_VEHICLE_CENTER:-1500}"
TARGET_VEHICLE_CENTER="${TARGET_VEHICLE_CENTER:-1500}"
INITIAL_VEHICLE_CENTER="${INITIAL_VEHICLE_CENTER:-300}"
SPAWN_BATCH_CENTER="${SPAWN_BATCH_CENTER:-20}"
MODEL_UPDATE_PERIOD="${MODEL_UPDATE_PERIOD:-10}"
METRICS_INTERVAL="${METRICS_INTERVAL:-20}"
EVAL_PRINT_EVERY="${EVAL_PRINT_EVERY:-100}"
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"

COMMON_ARGS=(
  --episode-seconds "$EPISODE_SECONDS"
  --eval-steps "$EVAL_STEPS"
  --max-vehicle-center "$MAX_VEHICLE_CENTER"
  --target-vehicle-center "$TARGET_VEHICLE_CENTER"
  --initial-vehicle-center "$INITIAL_VEHICLE_CENTER"
  --spawn-batch-center "$SPAWN_BATCH_CENTER"
  --model-update-period "$MODEL_UPDATE_PERIOD"
  --metrics-interval "$METRICS_INTERVAL"
  --eval-print-every "$EVAL_PRINT_EVERY"
)

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-parallel_eval_two_maps_${RUN_ID}}"

mkdir -p "$OUTROOT"

for map_tag in "${MAP_TAGS[@]}"; do
  mkdir -p "$OUTROOT/$map_tag/native"
  for model_tag in "${MODEL_TAGS[@]}"; do
    mkdir -p "$OUTROOT/$map_tag/$model_tag"
  done
done

printf '%s\n' "$OUTROOT" > .last_two_map_eval_dir

CONFIG_FILE="$OUTROOT/run_config.env"
{
  printf 'EPISODE_SECONDS=%s\n' "$EPISODE_SECONDS"
  printf 'EVAL_STEPS=%s\n' "$EVAL_STEPS"
  printf 'MAX_PARALLEL=%s\n' "$MAX_PARALLEL"
  printf 'MAP_TAGS=%s\n' "${MAP_TAGS[*]}"
  printf 'MODEL_TAGS=%s\n' "${MODEL_TAGS[*]}"
  printf 'SEEDS=%s\n' "${SEEDS[*]}"
  printf 'SANTACLARA_MAP=%s\n' "$SANTACLARA_MAP"
  printf 'FREMONT_MAP=%s\n' "$FREMONT_MAP"
  printf 'MAX_VEHICLE_CENTER=%s\n' "$MAX_VEHICLE_CENTER"
  printf 'TARGET_VEHICLE_CENTER=%s\n' "$TARGET_VEHICLE_CENTER"
  printf 'INITIAL_VEHICLE_CENTER=%s\n' "$INITIAL_VEHICLE_CENTER"
  printf 'SPAWN_BATCH_CENTER=%s\n' "$SPAWN_BATCH_CENTER"
  printf 'THREADS_PER_JOB=%s\n' "$THREADS_PER_JOB"
} > "$CONFIG_FILE"

JOBS_FILE="$OUTROOT/jobs.tsv"
printf 'map\tcontroller\tseed\tlog\tjson\n' > "$JOBS_FILE"

for seed in "${SEEDS[@]}"; do
  for map_tag in "${MAP_TAGS[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$map_tag" "native" "$seed" \
      "$map_tag/native/native_seed${seed}.log" \
      "$map_tag/native/native_seed${seed}.json" \
      >> "$JOBS_FILE"

    for model_tag in "${MODEL_TAGS[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$map_tag" "$model_tag" "$seed" \
        "$map_tag/$model_tag/allmodel_seed${seed}.log" \
        "$map_tag/$model_tag/allmodel_seed${seed}.json" \
        >> "$JOBS_FILE"
    done
  done
done

PIDS=()
NAMES=()

wait_for_slot() {
  while true; do
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if (( running < MAX_PARALLEL )); then
      return
    fi
    sleep 3
  done
}

launch_native() {
  local map_tag="$1"
  local map_file="$2"
  local seed="$3"
  local dir="$OUTROOT/$map_tag/native"
  local name="${map_tag}_native_seed${seed}"
  local json="$dir/native_seed${seed}.json"

  if [[ -f "$json" ]]; then
    echo "SKIP existing: $name"
    return
  fi

  echo "LAUNCH: $name"

  nohup env \
    TRAFFIC_NET_FILE="$map_file" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" \
    MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    python -u "$EVALUATOR" \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-all-model \
      --stats-csv "$dir/native_seed${seed}.csv" \
      --stats-json "$json" \
      > "$dir/native_seed${seed}.log" 2>&1 < /dev/null &

  PIDS+=("$!")
  NAMES+=("$name")
}

launch_model() {
  local map_tag="$1"
  local map_file="$2"
  local model_tag="$3"
  local seed="$4"
  local model_path="${MODEL_PATHS[$model_tag]}"
  local dir="$OUTROOT/$map_tag/$model_tag"
  local name="${map_tag}_${model_tag}_seed${seed}"
  local json="$dir/allmodel_seed${seed}.json"

  if [[ -f "$json" ]]; then
    echo "SKIP existing: $name"
    return
  fi

  echo "LAUNCH: $name"

  nohup env \
    TRAFFIC_NET_FILE="$map_file" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" \
    MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    python -u "$EVALUATOR" \
      "${COMMON_ARGS[@]}" \
      --compare-seeds "$seed" \
      --skip-native \
      --model-path "$model_path" \
      --vecnormalize-path "${model_path}_vecnormalize.pkl" \
      --stats-csv "$dir/allmodel_seed${seed}.csv" \
      --stats-json "$json" \
      > "$dir/allmodel_seed${seed}.log" 2>&1 < /dev/null &

  PIDS+=("$!")
  NAMES+=("$name")
}

total_jobs=$(( ${#SEEDS[@]} * ${#MAP_TAGS[@]} * (1 + ${#MODEL_TAGS[@]}) ))

echo "Output directory: $OUTROOT"
echo "Maps: ${MAP_TAGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total expected jobs: $total_jobs"
echo "Maximum simultaneous jobs: $MAX_PARALLEL"
echo

# Interleave maps for each seed so both maps make progress simultaneously.
for seed in "${SEEDS[@]}"; do
  for map_tag in "${MAP_TAGS[@]}"; do
    map_file="${MAP_FILES[$map_tag]}"

    wait_for_slot
    launch_native "$map_tag" "$map_file" "$seed"

    for model_tag in "${MODEL_TAGS[@]}"; do
      wait_for_slot
      launch_model "$map_tag" "$map_file" "$model_tag" "$seed"
    done
  done
done

echo
echo "All jobs have been submitted. Waiting for completion..."

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

missing=0
while IFS=$'\t' read -r map_tag controller seed log_rel json_rel; do
  if [[ "$map_tag" == "map" ]]; then
    continue
  fi
  if [[ ! -f "$OUTROOT/$json_rel" ]]; then
    echo "MISSING RESULT: $OUTROOT/$json_rel"
    missing=$((missing + 1))
  fi
done < "$JOBS_FILE"

if (( missing > 0 )); then
  echo
  echo "$missing of $total_jobs expected JSON files are missing."
  echo "The completed results are preserved in $OUTROOT."
  echo "Re-run with OUTROOT=$OUTROOT to skip completed jobs and retry missing jobs."
  exit 1
fi

echo
echo "All $total_jobs JSON result files exist. Merging comparisons..."

merge_failed=0

for map_tag in "${MAP_TAGS[@]}"; do
  for model_tag in "${MODEL_TAGS[@]}"; do
    merge_dir="$OUTROOT/$map_tag/merge_$model_tag"
    mkdir -p "$merge_dir"

    find "$merge_dir" -maxdepth 1 -type f \
      \( -name 'native_seed*.json' -o -name 'allmodel_seed*.json' \) \
      -delete

    cp "$OUTROOT/$map_tag/native"/native_seed*.json "$merge_dir/"
    cp "$OUTROOT/$map_tag/$model_tag"/allmodel_seed*.json "$merge_dir/"

    echo
    echo "################ $map_tag / $model_tag ################"

    if python "$MERGER" \
      "$merge_dir" \
      "$OUTROOT/$map_tag/${model_tag}_vs_native.csv" \
      "$OUTROOT/$map_tag/${model_tag}_vs_native.json" \
      | tee "$OUTROOT/$map_tag/${model_tag}_vs_native_table.log"
    then
      echo "MERGED: $map_tag / $model_tag"
    else
      echo "MERGE FAILED: $map_tag / $model_tag"
      merge_failed=1
    fi
  done
done

echo
if [[ -f "analyze_two_map_eval_statistics.py" ]]; then
  echo "Computing paired seed-by-seed confidence intervals..."
  if python analyze_two_map_eval_statistics.py "$OUTROOT" \
    | tee "$OUTROOT/statistical_summary.log"
  then
    echo "Wrote statistical summary: $OUTROOT/statistical_summary.csv"
  else
    echo "Statistical analysis failed. The raw and merged results are still preserved."
    merge_failed=1
  fi
fi

echo
echo "Evaluation campaign complete."
echo "Results: $OUTROOT"

if (( failed != 0 || merge_failed != 0 )); then
  exit 1
fi

exit 0
