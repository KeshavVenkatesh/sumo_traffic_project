#!/usr/bin/env bash
set -euo pipefail

cd ~/sumo_traffic_project
source .venv/bin/activate

export SUMO_HOME="$PWD/.venv/lib/python3.10/site-packages/sumo"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

export MIXED_TLS_FILE="$PWD/.usable_santaclara_tls.txt"
export MIXED_TLS_LIMIT=40
export MIXED_RANDOM_INITIAL_PHASE=1
export MIXED_INITIAL_PHASE_AGE_MAX=12
export MIXED_EPISODE_SECONDS=900
export MIXED_MAX_NUM_VEHICLES=2000

python -u train_santaclara_proxy.py \
  --env-module traffic_rl_model_santaclara_mixed_tls \
  --env-class TrafficSignalEnv \
  --tls-id cluster_282813104_282813137_5041442783_5041442784 \
  --model-path models/traffic_signal_maskable_ppo_santaclara_mixed_tls_v1 \
  --timesteps 400000 \
  --episode-seconds 900 \
  --max-vehicles 2000 \
  --target-vehicles 1500 \
  --initial-vehicles 300 \
  --spawn-batch 20 \
  --vehicle-variants 900,1100,1300,1500,1800,2000 \
  --num-envs 2 \
  --seed 739 \
  --device auto \
  --torch-threads 1 \
  --resume \
  --no-curriculum \
  --no-progress-bar \
  --lr-start 0.00001 \
  --lr-end 0.000003 \
  --clip-start 0.08 \
  --clip-end 0.04 \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 4 \
  --gamma 0.997 \
  --gae-lambda 0.95 \
  --ent-coef 0.002 \
  --vf-coef 0.70 \
  --max-grad-norm 0.50 \
  --target-kl 0.01 \
  --norm-obs \
  --clip-obs 10.0 \
  --clip-reward 10.0 \
  --save-freq 12500 \
  --eval-freq 0
