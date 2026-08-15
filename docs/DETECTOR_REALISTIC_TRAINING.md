# Detector-realistic traffic-signal RL (schema v4)

Schema v4 is the deployable-observation successor to the full-state schema-v3
controller. It preserves the map-agnostic shared GNN, legal phase action mask,
exact SUMO traffic, persistent multi-map workers, PPO update, fixed demand
banks, and held-out validation. The primary change is the detector-limited
information boundary used by both the actor and its training reward.

The schema-v3 checkpoint remains the oracle/full-state upper bound. Schema-v3
and schema-v4 checkpoints are not interchangeable even though their padded
tensor dimensions match. The validation and comparison entry points require
the v4 metadata sidecar and reject a v3 or otherwise incompatible checkpoint
instead of silently running it against detector observations.

## What the actor can and cannot observe

Each graph node represents one physical incoming detector lane. A lane shared
by several turns remains one node; schema v4 never uses a vehicle route to
split that lane into perfectly known turn demand.

The actor receives:

- stop-bar and advance-detector presence/occupancy;
- detected arrival and departure rates over short and 60-second windows;
- a conservation-based queue estimate, queue trend, and delay proxy;
- speed only when the simulated detector profile supports it;
- downstream occupancy only when that detector is installed;
- current phase, elapsed green, time since service, and legal phase topology;
- static turn permissions inferred from the network.

The actor never receives vehicle routes, destinations, individual waiting
times, exact per-vehicle ETAs, raw vehicle identifiers, or oracle downstream
state when no downstream detector is available. SUMO vehicle positions are
used internally only to emulate aggregate detector zones and crossing pulses.
The PPO reward is likewise built from detected departure pulses and the same
aggregate queue, delay, spillback, and service estimates; raw vehicle identity
is not retained in the policy/reward snapshot.

Profiles:

- `loops`: stop-bar and advance detection; no direct speed or downstream view.
- `camera`: detection-zone speed and downstream occupancy are available.
- `mixed`: heterogeneous capabilities sampled per lane. This is the recommended
  training profile because one checkpoint learns to handle different field
  installations.

Training can also randomize calibration, measurement noise, temporary dropout,
stuck detectors, and one or more decisions of latency. Validation defaults to
deterministic sensors so checkpoint selection is repeatable. Run a separate
corrupted-sensor campaign to report robustness.

### Exact observation tensors

All values are bounded to `[-1, 1]` and padded/masked so one checkpoint can
serve different intersections.

| Scope | Fields |
| --- | --- |
| Incoming detector lane (up to 160) | stop-bar presence; short/60-second stop-bar occupancy; advance presence; short/60-second advance occupancy; short/60-second arrival rate; short/60-second departure rate; estimated queue; queue trend; estimated delay; optional speed ratio; optional downstream occupancy; estimated pressure; detector-call duration; currently green; time since service; left/straight/right lane permissions; speed-available flag; sensor health |
| Candidate phase (up to 16) | current-phase flag; elapsed green; mean estimated queue; mean estimated pressure; mean downstream space; maximum detector-call duration; detector coverage; mean arrival rate |
| Local intersection | elapsed green; minimum-green progress; mean stop-bar occupancy; mean estimated queue; mean arrival rate; mean downstream occupancy; maximum time since service; detector coverage |

There is no network-wide traffic state. Each TLS makes a local decision using
the same shared policy weights.

### One policy decision

1. The virtual roadside detectors aggregate the latest local readings and
   update rolling queue/delay estimates.
2. The graph network embeds incoming detector lanes and passes messages only
   through local phase/topology relationships.
3. A phase scorer assigns one score to each candidate phase; a separate head
   scores `hold`, and a value head supports PPO training.
4. The safety mask removes padded actions, prevents switching before minimum
   green, blocks a phase only when an installed downstream detector reports
   spillback, and forces an alternative after maximum green.
5. The selected legal phase is handed to the existing safe yellow/all-red
   transition controller.

## Files

- `detector_realistic_tls.py`: detector simulator, lane-group topology,
  observation schema, queue estimator, and action mask.
- `detector_realistic_policy.py`: shared permutation-equivariant graph policy.
- `detector_realistic_multiagent_worker.py`: persistent all-TLS SUMO worker.
- `train_detector_realistic_multiagent.py`: central multi-map PPO trainer.
- `validate_detector_realistic_multiagent.py`: held-out checkpoint selection.
- `traffic_rl_detector_realistic_env.py`: checkpoint shape environment and TLS
  compatibility scan.
- `compare_native_sumo_vs_detector_realistic.py`: paired native-versus-v4
  evaluator.
- `tests/test_detector_realistic_*.py`: information-boundary and policy tests.

## 1. External-machine environment

```bash
cd /users/sriramv/sumo_traffic_project
git fetch origin
git pull --ff-only origin main

source /users/sriramv/.venvs/sumo-eval-v3/bin/activate

export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

which python
which sumo
sumo --version
python -c 'import numpy, torch, gymnasium, traci, sumolib; print("Python/SUMO imports OK")'
```

If the existing environment is unavailable, create a project-local one:

```bash
cd /users/sriramv/sumo_traffic_project
python3 -m venv .venv-traffic
source .venv-traffic/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 2. Code tests and SUMO compatibility scan

Run these before starting a long job:

```bash
cd /users/sriramv/sumo_traffic_project
source /users/sriramv/.venvs/sumo-eval-v3/bin/activate
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

python -m pytest -q \
  tests/test_detector_realistic_tls.py \
  tests/test_detector_realistic_policy.py \
  tests/test_detector_realistic_worker_contract.py

DETECTOR_SENSOR_PROFILE=loops \
python -u traffic_rl_detector_realistic_env.py \
  --net-file new_map.net.xml \
  --list-tls-json
```

## 3. Training data

Reuse `generated_map_corpus/manifest.json` and
`training_demand_bank_v3/manifest.json` if their files still exist. Demand is
controller-independent, so it does not need to be regenerated for schema v4.

If the map corpus is missing, create it first:

```bash
nohup python -u generate_map_corpus.py \
  --config map_corpus_regions.json \
  --output-dir generated_map_corpus \
  > generate_map_corpus.log 2>&1 &
echo $! > generate_map_corpus.pid
```

Wait for that job and verify its manifest before generating demand:

```bash
while kill -0 "$(cat generate_map_corpus.pid)" 2>/dev/null; do
  tail -n 3 generate_map_corpus.log
  sleep 30
done
test -s generated_map_corpus/manifest.json

nohup python -u generate_training_demand_bank.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --output-dir training_demand_bank_v3 \
  --rates 4,8,12 \
  --seeds 101,102 \
  --episode-seconds 7200 \
  --workers 8 \
  > generate_training_demand_bank_v3.log 2>&1 &
echo $! > generate_training_demand_bank_v3.pid
```

Do not begin training until demand generation exits successfully and the
demand-bank manifest reports all expected route files.

## 4. Foreground smoke training

This starts one exact SUMO worker, performs one PPO update, exports a schema-v4
checkpoint, and skips held-out validation. It intentionally uses the dynamic
smoke demand so a 7,200-second fixed-demand bank is not paired with a shorter
600-second episode:

```bash
python -u train_detector_realistic_multiagent.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --model-path models/detector_realistic_v4_smoke \
  --best-model-path models/detector_realistic_v4_smoke_best \
  --sensor-profile mixed \
  --rounds 1 \
  --num-map-workers 1 \
  --rollouts-per-map-visit 1 \
  --rollout-steps 8 \
  --episode-seconds 600 \
  --decision-seconds 10 \
  --embed-dim 32 \
  --graph-layers 1 \
  --no-validate-every-round \
  --progress-file detector_v4_smoke_progress.json \
  --no-use-libsumo \
  --restart
```

## 5. Full mixed-infrastructure training

Run one trainer only. Its map workers already collect concurrently and one
central learner owns the checkpoint.

```bash
nohup /users/sriramv/.venvs/sumo-eval-v3/bin/python -u \
  train_detector_realistic_multiagent.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --demand-bank-manifest training_demand_bank_v3/manifest.json \
  --model-path models/detector_realistic_multiagent_v4 \
  --best-model-path models/detector_realistic_multiagent_v4_best \
  --sensor-profile mixed \
  --detector-noise-std 0.02 \
  --detector-calibration-jitter 0.05 \
  --detector-dropout-prob 0.03 \
  --detector-stuck-prob 0.01 \
  --max-detector-latency-decisions 1 \
  --rounds 4 \
  --num-map-workers 4 \
  --rollouts-per-map-visit 16 \
  --rollout-steps 64 \
  --episode-seconds 7200 \
  --decision-seconds 10 \
  --max-vehicle-center 1500 \
  --target-density-range 2,10 \
  --embed-dim 128 \
  --graph-layers 2 \
  --ppo-epochs 4 \
  --minibatch-size 512 \
  --teacher-coef 0.10 \
  --teacher-decay-fraction 0.25 \
  --validate-every-round \
  --validation-splits validation \
  --validation-seeds 9001,9002 \
  --validation-episode-seconds 600 \
  --validation-workers 2 \
  --progress-file detector_realistic_multiagent_progress.json \
  --no-use-libsumo \
  --restart \
  > detector_realistic_v4_training.log 2>&1 &
echo $! > detector_realistic_v4_training.pid
```

Monitor it:

```bash
watch -n 10 'python monitor_multiagent_training.py detector_realistic_multiagent_progress.json --refresh 0'
tail -f detector_realistic_v4_training.log
ps -o pid,ppid,etime,%cpu,%mem,cmd -p "$(cat detector_realistic_v4_training.pid)"
```

To resume, repeat the exact command without `--restart` and append to the log.
The sensor configuration is part of the resume-plan signature, so the trainer
rejects an accidental profile/noise change.

## 6. Direct deterministic validation

```bash
python -u validate_detector_realistic_multiagent.py \
  --manifest generated_map_corpus/manifest.json \
  --splits validation \
  --model-path models/detector_realistic_multiagent_v4_best \
  --output-json runs/detector_v4_validation.json \
  --seeds 9001,9002 \
  --episode-seconds 600 \
  --decision-seconds 10 \
  --sensor-profile mixed \
  --workers 2 \
  --no-use-libsumo
```

## 7. Paired Fremont/Santa Clara/test-map evaluation

The comprehensive launcher now accepts the detector-realistic runner while
retaining the exact same fixed demand for native SUMO, MaxPressure, and the
learned policy.

First run the strict ordinary-loop condition:

```bash
SEEDS=$(seq -s, 1001 1030)

nohup python -u launch_comprehensive_evaluation.py \
  --benchmarks fremont=new_map.net.xml,santaclara=santa_clara.net.xml \
  --manifest generated_map_corpus/manifest.json \
  --manifest-splits test \
  --model-path models/detector_realistic_multiagent_v4_best \
  --all-model-runner detector_realistic \
  --sensor-profile loops \
  --output-dir runs/detector_v4_loops \
  --rates 6,12,18 \
  --seeds "$SEEDS" \
  --episode-seconds 1200 \
  --eval-steps 2500 \
  --metrics-interval 20 \
  --model-update-period 10 \
  --max-vehicles 3000 \
  --max-parallel 8 \
  --demand-generation-workers 8 \
  --progress-file detector_v4_loops_progress.json \
  > detector_v4_loops_eval.log 2>&1 &
echo $! > detector_v4_loops_eval.pid
```

Then run a modern camera/radar condition in a new output directory by changing
`--sensor-profile loops` to `--sensor-profile camera`.

For a sensor-failure robustness campaign, use another output directory and add:

```bash
  --detector-noise-std 0.03 \
  --detector-calibration-jitter 0.05 \
  --detector-dropout-prob 0.05 \
  --detector-stuck-prob 0.02 \
  --max-detector-latency-decisions 2
```

Monitor and analyze with the existing tools:

```bash
watch -n 10 'python monitor_comprehensive_evaluation.py detector_v4_loops_progress.json'
tail -f detector_v4_loops_eval.log

python -u analyze_comprehensive_evaluation.py runs/detector_v4_loops \
  | tee runs/detector_v4_loops/statistical_summary.log
```

Interpret the loop-only campaign as the primary deployment result, camera as
the value of upgraded sensing, and schema v3 as an oracle upper bound. Never
compare controllers on different demand route files or different simulation
durations.
