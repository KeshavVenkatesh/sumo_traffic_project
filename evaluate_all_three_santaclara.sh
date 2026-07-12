#!/usr/bin/env bash
set -euo pipefail

cd ~/sumo_traffic_project
source .venv/bin/activate

export SUMO_HOME="$PWD/.venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

run_model() {
    local model_path="$1"
    local tag="$2"

    echo
    echo "####################################################################################################"
    echo "EVALUATING: $tag"
    echo "MODEL:      $model_path"
    echo "####################################################################################################"

    bash evaluate_santaclara_checkpoint.sh \
        "$model_path" \
        "$tag"

    echo
    echo "FINISHED: $tag"
    echo
}

run_model \
    models/traffic_signal_maskable_ppo_santaclara_rr_40tls_good \
    original_40tls_good

run_model \
    models/traffic_signal_maskable_ppo_santaclara_rr_robust_v1 \
    robust_v1

run_model \
    models/traffic_signal_maskable_ppo_santaclara_mixed_tls_v1 \
    mixed_tls_v1

echo
echo "All three Santa Clara evaluations finished."
