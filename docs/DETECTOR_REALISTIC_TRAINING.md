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
- `map_corpus_regions_v4.json`: frozen 32/8/8 brand-new geographic split.
- `validate_map_split_protocol.py`: pre/post-generation leakage and manifest
  lock checks used by the trainer and final evaluator.
- `tests/test_detector_realistic_*.py`: information-boundary and policy tests.

## 1. External-machine environment

```bash
cd /users/sriramv/sumo_traffic_project
export SUMO_HOME=/users/sriramv/.local/lib/python3.10/site-packages/sumo
export PATH="$SUMO_HOME/bin:/users/sriramv/.local/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
hash -r

which python
which sumo
which netconvert
sumo --version
python -c 'import numpy, torch, gymnasium, traci, sumolib; print("Python/SUMO imports OK")'
```

This is the user-level SUMO 1.26 installation used on the current CloudLab
node; no virtual-environment activation is required. If a working project
virtual environment is created later, activate it before the exports above.

If `pytest` is missing, install only the development requirements into the
user environment:

```bash
python -m pip install --user -r requirements-dev.txt
```

## 2. Code tests and SUMO compatibility scan

Run these before starting a long job:

```bash
cd /users/sriramv/sumo_traffic_project
export SUMO_HOME=/users/sriramv/.local/lib/python3.10/site-packages/sumo
export PATH="$SUMO_HOME/bin:/users/sriramv/.local/bin:$PATH"
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

## 3. Frozen, brand-new map protocol

Do **not** reuse `generated_map_corpus` or `training_demand_bank_v3` for this
experiment. Those names belong to the historical schema-v3 campaign. Schema
v4 uses `map_corpus_regions_v4.json`, seed `20260817`, and separate output
directories:

| Split | New maps | Permitted use |
| --- | ---: | --- |
| train | 32 | PPO updates and detector randomization |
| validation | 8 | checkpoint selection only |
| test | 8 | one locked final evaluation after development ends |

Every v4 region name is new. The protocol also expands the historical
`map_corpus_regions.json` with its original seed `20260712` and rejects any
final-test crop that overlaps or lies within 75 km of any historical map or
any v4 training/validation crop. The closest planned test/exclusion pair is
more than 129 km apart.

Run the preflight before downloading anything:

```bash
python -u validate_map_split_protocol.py \
  --new-config map_corpus_regions_v4.json \
  --old-config map_corpus_regions.json \
  --output detector_v4_split_preflight.json
```

Then generate the new corpus in its own directory:

```bash
nohup python -u generate_map_corpus.py \
  --config map_corpus_regions_v4.json \
  --seed 20260817 \
  --output-dir generated_map_corpus_v4 \
  --request-delay-seconds 2 \
  > generate_map_corpus_v4.log 2>&1 &
echo $! > generate_map_corpus_v4.pid
```

Wait for generation, then create the post-generation lock. This second check
requires all 48 planned maps to have passed the OSM/SUMO signal filters; it
also hashes the exact manifest used by training and final evaluation.

```bash
while kill -0 "$(cat generate_map_corpus_v4.pid)" 2>/dev/null; do
  tail -n 3 generate_map_corpus_v4.log
  sleep 30
done
wait "$(cat generate_map_corpus_v4.pid)" 2>/dev/null || true
test -s generated_map_corpus_v4/manifest.json

python -u validate_map_split_protocol.py \
  --new-config map_corpus_regions_v4.json \
  --old-config map_corpus_regions.json \
  --manifest generated_map_corpus_v4/manifest.json \
  --output generated_map_corpus_v4/split_protocol_lock.json
```

If the post-generation check fails because a region was rejected, do not
reduce a split count or move a test map into development. Replace the rejected
region in the configuration, assign a new corpus ID/seed, and freeze the
protocol again before training.

Generate a new demand bank for the 32 new training maps:

```bash

nohup python -u generate_training_demand_bank.py \
  --manifest generated_map_corpus_v4/manifest.json \
  --splits train \
  --output-dir training_demand_bank_detector_v4 \
  --rates 4,8,12 \
  --seeds 101,102 \
  --episode-seconds 7200 \
  --workers 8 \
  > generate_training_demand_bank_detector_v4.log 2>&1 &
echo $! > generate_training_demand_bank_detector_v4.pid
```

Do not begin training until demand generation exits successfully and the
demand-bank manifest reports `32 maps x 3 rates x 2 seeds = 192` route files.

## 4. Foreground smoke training

This starts one exact SUMO worker, performs one PPO update, exports a schema-v4
checkpoint, and skips held-out validation. It intentionally uses the dynamic
smoke demand so a 7,200-second fixed-demand bank is not paired with a shorter
600-second episode:

```bash
python -u train_detector_realistic_multiagent.py \
  --manifest generated_map_corpus_v4/manifest.json \
  --split-protocol-lock generated_map_corpus_v4/split_protocol_lock.json \
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
nohup python -u train_detector_realistic_multiagent.py \
  --manifest generated_map_corpus_v4/manifest.json \
  --split-protocol-lock generated_map_corpus_v4/split_protocol_lock.json \
  --splits train \
  --demand-bank-manifest training_demand_bank_detector_v4/manifest.json \
  --model-path models/detector_realistic_multiagent_v4 \
  --best-model-path models/detector_realistic_multiagent_v4_best \
  --sensor-profile mixed \
  --detector-noise-std 0.02 \
  --detector-calibration-jitter 0.05 \
  --detector-dropout-prob 0.03 \
  --detector-stuck-prob 0.01 \
  --max-detector-latency-decisions 1 \
  --rounds 6 \
  --num-map-workers 8 \
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
  --validation-workers 4 \
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
  --manifest generated_map_corpus_v4/manifest.json \
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

Before direct validation, rerun the lock check so a changed manifest cannot be
used accidentally:

```bash
python -u validate_map_split_protocol.py \
  --new-config map_corpus_regions_v4.json \
  --old-config map_corpus_regions.json \
  --manifest generated_map_corpus_v4/manifest.json \
  --output generated_map_corpus_v4/split_protocol_lock.json
```

## 7. Paired four-controller locked final evaluation

The comprehensive launcher accepts the detector-realistic runner and the
previous schema-v3 checkpoint in one campaign. Detector v4, schema v3,
MaxPressure, and native SUMO replay the exact same fixed route file for every
map/rate/seed condition. This makes the v4-versus-v3 result paired rather than
an informal comparison between separate campaigns.

Do not run this section until model design, hyperparameters, and checkpoint
selection are finished. Looking at these results turns the locked maps into
development data. The launcher verifies both the manifest hash and exact test
membership from the post-generation protocol lock.

First run the strict ordinary-loop condition on only the eight frozen maps:

```bash
SEEDS=$(seq -s, 1001 1030)

nohup python -u launch_comprehensive_evaluation.py \
  --benchmarks "" \
  --manifest generated_map_corpus_v4/manifest.json \
  --manifest-splits test \
  --split-protocol-lock generated_map_corpus_v4/split_protocol_lock.json \
  --model-path models/detector_realistic_multiagent_v4_best \
  --schema-v3-model models/map_agnostic_multiagent_v3_best \
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

## 8. Historical maps and the controlled information ablation

Fremont, Santa Clara, Fresno, and San Diego remain useful legacy/external
benchmarks, but they have already been examined in earlier project work. Run
them in a separate output directory without `--split-protocol-lock`; never mix
their statistics into the untouched eight-map primary result.

The frozen schema-v3 checkpoint answers the deployment question: “Does the new
detector-limited system beat the previously shipped model on maps neither
checkpoint trained on?” It does not isolate sensing alone because schema v3
and schema v4 were trained on different corpora.

For a strict observation-boundary ablation, also retrain the schema-v3
architecture on the exact v4 training manifest and demand bank, select it only
on the same validation split, and substitute that matched checkpoint for
`--schema-v3-model` in a second locked campaign. Report the frozen historical
checkpoint and the matched retrain as separate baselines.

Run the matched schema-v3 training only after the schema-v4 trainer has
finished, so the two jobs do not compete for the same CPUs:

```bash
nohup python -u train_map_agnostic_multiagent.py \
  --manifest generated_map_corpus_v4/manifest.json \
  --split-protocol-lock generated_map_corpus_v4/split_protocol_lock.json \
  --splits train \
  --demand-bank-manifest training_demand_bank_detector_v4/manifest.json \
  --model-path models/map_agnostic_multiagent_v3_matched \
  --best-model-path models/map_agnostic_multiagent_v3_matched_best \
  --rounds 6 \
  --num-map-workers 8 \
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
  --validation-workers 4 \
  --progress-file map_agnostic_v3_matched_progress.json \
  --no-use-libsumo \
  --restart \
  > map_agnostic_v3_matched_training.log 2>&1 &
echo $! > map_agnostic_v3_matched_training.pid
```

For its locked campaign, repeat section 7 with a new output/progress/log name
and set
`--schema-v3-model models/map_agnostic_multiagent_v3_matched_best`. The v4,
matched v3, MaxPressure, and native controllers will again replay identical
fixed routes within that campaign.
