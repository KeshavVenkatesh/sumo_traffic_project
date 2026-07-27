#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="${PYTHON:-$PWD/.venv-traffic/bin/python}"
EVALUATOR="${EVALUATOR:-compare_native_sumo_vs_all_model_schema_aware.py}"
PARALLEL_RUNNER="${PARALLEL_RUNNER:-compare_native_sumo_vs_all_model_parallel.py}"
SEEDS="${SEEDS:-$(seq -s, 1001 1030)}"
SEED_COUNT="$(awk -F, '{print NF}' <<<"$SEEDS")"
JOBS_PER_CAMPAIGN="${JOBS_PER_CAMPAIGN:-2}"
EPISODE_SECONDS="${EPISODE_SECONDS:-1200}"
EVAL_STEPS="${EVAL_STEPS:-1300}"
MODEL_UPDATE_PERIOD="${MODEL_UPDATE_PERIOD:-10}"
METRICS_INTERVAL="${METRICS_INTERVAL:-20}"
EVAL_PRINT_EVERY="${EVAL_PRINT_EVERY:-50}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python interpreter: $PYTHON" >&2
    exit 1
fi

for required in "$EVALUATOR" "$PARALLEL_RUNNER" \
    summarize_two_map_four_model_eval.py \
    compare_native_sumo_vs_all_model.py \
    compare_fixed_vs_single_vs_all_model_realistic.py \
    realistic_all_intersections_fixed_cycle.py; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required source file: $required" >&2
        exit 1
    fi
done

MAP_SPECS=(
    "fremont|$PWD/new_map.net.xml"
    "santaclara|$PWD/santa_clara.net.xml"
)

MODEL_SPECS=(
    "ambulance_aware_final_3m|$PWD/models/traffic_signal_ambulance_aware_final_3m.zip|$PWD/models/traffic_signal_ambulance_aware_final_3m_vecnormalize.pkl"
    "original_40tls_good|$PWD/models/traffic_signal_maskable_ppo_santaclara_rr_40tls.zip|$PWD/models/traffic_signal_maskable_ppo_santaclara_rr_40tls_vecnormalize.pkl"
    "robust_v1|$PWD/models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1.zip|$PWD/models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1_vecnormalize.pkl"
    "mixed_tls_v1|$PWD/models/traffic_signal_maskable_ppo_santaclara_mixed_tls_v1.zip|$PWD/models/traffic_signal_maskable_ppo_santaclara_mixed_tls_v1_vecnormalize.pkl"
)

for map_spec in "${MAP_SPECS[@]}"; do
    IFS='|' read -r _map_label map_path <<<"$map_spec"
    if [[ ! -f "$map_path" ]]; then
        echo "Missing map: $map_path" >&2
        exit 1
    fi
done

for model_spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r label model_path vec_path <<<"$model_spec"
    if [[ ! -f "$model_path" ]]; then
        echo "Missing model for $label: $model_path" >&2
        exit 1
    fi
    if [[ ! -f "$vec_path" ]]; then
        echo "Missing VecNormalize file for $label: $vec_path" >&2
        exit 1
    fi
done

"$PYTHON" -m py_compile "$EVALUATOR" "$PARALLEL_RUNNER"

ROOT="${ROOT:-$PWD/two_map_four_model_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"
printf '%s\n' "$ROOT" > "$PWD/latest_two_map_four_model_eval_root.txt"

cat > "$ROOT/evaluation_manifest.txt" <<EOF
root=$ROOT
started_at=$(date --iso-8601=seconds)
seeds=$SEEDS
seed_count=$SEED_COUNT
jobs_per_campaign=$JOBS_PER_CAMPAIGN
episode_seconds=$EPISODE_SECONDS
eval_steps=$EVAL_STEPS
model_update_period=$MODEL_UPDATE_PERIOD
metrics_interval=$METRICS_INTERVAL
evaluator=$EVALUATOR
EOF

COMMON_ARGS=(
    --script "$EVALUATOR"
    --compare-seeds "$SEEDS"
    --jobs "$JOBS_PER_CAMPAIGN"
    --episode-seconds "$EPISODE_SECONDS"
    --eval-steps "$EVAL_STEPS"
    --eval-print-every "$EVAL_PRINT_EVERY"
    --model-update-period "$MODEL_UPDATE_PERIOD"
    --metrics-interval "$METRICS_INTERVAL"
    --max-vehicle-center 750
    --target-vehicle-center 650
    --initial-vehicle-center 200
    --spawn-batch-center 12
    --device cpu
)

launch_campaign() {
    local map_label="$1"
    local map_path="$2"
    local controller_label="$3"
    shift 3

    local campaign_dir="$ROOT/$map_label/$controller_label"
    local seed_log_dir="$campaign_dir/seed_logs"
    local wrapper_log="$campaign_dir/wrapper.log"
    local pid_file="$campaign_dir/wrapper.pid"

    mkdir -p "$seed_log_dir"

    nohup env \
        TRAFFIC_NET_FILE="$map_path" \
        SUMO_USE_LIBSUMO=0 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        "$PYTHON" -u "$PARALLEL_RUNNER" \
        "${COMMON_ARGS[@]}" \
        --log-dir "$seed_log_dir" \
        --stats-csv "$campaign_dir/merged.csv" \
        --stats-json "$campaign_dir/merged.json" \
        "$@" \
        > "$wrapper_log" 2>&1 &

    local pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    printf 'launched %-11s / %-26s PID=%s\n' \
        "$map_label" "$controller_label" "$pid"
}

for map_spec in "${MAP_SPECS[@]}"; do
    IFS='|' read -r map_label map_path <<<"$map_spec"

    # Native SUMO is evaluated once per map, not redundantly once per model.
    launch_campaign "$map_label" "$map_path" native_sumo \
        --skip-all-model

    for model_spec in "${MODEL_SPECS[@]}"; do
        IFS='|' read -r model_label model_path vec_path <<<"$model_spec"
        launch_campaign "$map_label" "$map_path" "$model_label" \
            --skip-native \
            --model-path "$model_path" \
            --vecnormalize-path "$vec_path"
    done
done

echo
echo "Evaluation root: $ROOT"
echo "Campaigns:      10"
echo "Seeds/campaign: $SEED_COUNT"
echo "Total runs:     $((10 * SEED_COUNT))"
echo "Max concurrent seed processes: $((10 * JOBS_PER_CAMPAIGN))"
echo
echo "Monitor with:"
echo "  watch -n 10 -t '$PYTHON monitor_two_map_four_model_eval.py'"
