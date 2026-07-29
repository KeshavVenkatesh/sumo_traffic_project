# Fast map-agnostic traffic-signal training (schema v3)

> **Safe-residual integration status.** PR #4 contained an older replacement
> trainer and a second all-TLS environment. Those replacements are not the
> schema-v3 training path and have been removed. The persistent worker/central
> learner architecture documented below remains authoritative. The reusable
> safe-residual controller, policy, checkpoint contract, immutable demand-bank
> generator, and promotion gate are available as an opt-in experimental
> deployment layer; schema-v3 checkpoints are not silently reinterpreted as
> safe-residual checkpoints.

Schema v3 replaces the sequential “one full SUMO map per TLS task” campaign.
One persistent SUMO process now controls every compatible traffic light on a
map and emits one transition per TLS per decision. A central parameter-sharing
PPO learner combines several maps while preserving a separate temporal GAE
trajectory for each intersection.

## Why this is faster without changing the simulator

- SUMO remains exact; no surrogate traffic model is used.
- Each map is parsed and started once for many PPO updates.
- One map-wide TraCI snapshot caches lane and vehicle values for every TLS.
- Static topology is transmitted/stored once per TLS, not once per timestep.
- Dense GNN attention crops the 64-slot padding to the largest real
  intersection in each minibatch.
- Pre-generated route banks remove online OD-route construction and guarantee
  repeatable demand.
- Several independent map processes collect concurrently. There is still only
  one learner and checkpoint writer, so updates cannot overwrite one another.

The previous configuration needed 4,096 environment transitions before its
first PPO update and obtained them by repeatedly restarting a 1,500-vehicle map
for one TLS. Schema v3 obtains transitions from every TLS at the same simulated
second. On the included Fremont smoke test, 2 decisions over 10 TLS produced 20
transitions in about 0.2 seconds of collection after startup. Full-density
speed is hardware- and map-dependent, so use the live throughput and ETA rather
than extrapolating that smoke number.

## Quality/generalization protections

- Inputs are physical movements rather than compass/phase slots.
- Queue, density, speed, wait, pressure, downstream storage, arrival rate, and
  starvation are bounded by local capacity or physical reference values.
- Actions are `hold` or a valid native green candidate; minimum green, maximum
  green, yellow/all-red clearance, invalid phases, and spillback safety remain
  hard constrained.
- A movement GNN and shared phase scorer are permutation equivariant.
- Every map receives equal loss weight. Rare topology buckets receive equal
  weight within each map.
- The shared policy controls all TLS simultaneously during training, matching
  deployment and exposing downstream interactions.
- Sensor noise, calibration jitter, and rare channel dropout affect only
  measured traffic features—not phase legality or topology.
- A small normalized-MaxPressure imitation loss stabilizes early exploration,
  then decays to exactly zero after the first quarter of training.
- Held-out validation controls all TLS on each validation map. Best-checkpoint
  selection weights the worst validation map 75% and the mean 25%.
- Training refuses byte-identical/path-overlapping maps across splits.

Schema-v1 checkpoints are incompatible with the movement representation. A v2
deployment checkpoint may still load because its observation/action tensors
match, but it cannot resume the v3 multi-agent optimizer/schedule and does not
contain the v3 training improvements. Use a new v3 model path.

## Safe-residual controller components

`safe_residual_controller.py` implements a deterministic normalized coordinated
MaxPressure-plus-penalty (CMPP) baseline. A learned residual may influence only
actions that remain legal under the map-agnostic action mask and are within a
configured CMPP-regret bound. Setting residual authority to zero, reporting
high uncertainty, or reporting an out-of-distribution observation falls back
to the deterministic baseline.

`safe_residual_policy.py` provides an SB3-compatible
`SafeResidualMapAgnosticPolicy`. It reuses the current movement-GNN observation
and action schema from `map_agnostic_policy.py`; optional adapters use shared
movement/phase scorers rather than positional TLS weights. It has its own
checkpoint class and must not be used to resume a schema-v3 trainer checkpoint.
Call `set_residual_authority()` between rollout batches when applying an
authority curriculum. Do not change authority during a PPO rollout.

Every promoted safe-residual `.zip` must have a sibling
`*_contract.json` created with `checkpoint_contract.py`. The contract binds the
feature order, padding limits, decision cadence, safety timing, CMPP settings,
residual limits, and adapter selection. Loading must fail closed when the
contract is absent or does not match runtime settings. Existing schema-v3
`*_map_agnostic.json` metadata remains the contract for existing non-residual
checkpoints.

The safe-residual implementation is intentionally not wired into
`train_map_agnostic_multiagent.py` yet. That trainer uses a custom centralized
PPO update and persistent rollout workers; replacing it with the PR #4
short-lived per-map SB3 loop would lose map-balanced sampling, temporal GAE per
TLS, topology balancing, validation selection, resumable optimizer state, and
persistent SUMO throughput. Integrating residual training requires adding CMPP
logits consistently to both the worker behavior policy and central PPO
evaluation path, followed by an end-to-end SUMO equivalence campaign.

## Immutable paired demand and promotion gating

The established schema-v3 training bank remains
`generate_training_demand_bank.py`. For controller-promotion experiments,
`generate_fixed_demand_bank.py` adds checksummed immutable route records while
also writing the existing top-level `episode_seconds` and `routes` fields, so
the current schema-v3 demand-bank loader can consume it. Concrete `<vehicle>`
and `<trip>` schedules are accepted; `<flow>` records are rejected because
their realized count is not an immutable paired schedule.

`launch_comprehensive_evaluation.py` stamps each CSV row with the route/network
hashes, map/scenario identity, and scheduled departure count required by the
promotion gate. The gate accepts either its JSON row format or the existing
campaign's `paired_runs.csv`. Source-insertion backlog is evaluated when an
evaluator reports `not_departed_total`; older evaluators still receive strict
scheduled-demand identity and completed-throughput checks.

Use `validate_controller_promotion.py` only with paired result rows produced
from the same checksummed route file. It checks seed pairing, scheduled demand
identity, recovery rate, throughput, delay, gridlock, and bootstrap confidence
gates. Passing this statistical gate does not replace GUI inspection or a
multi-map SUMO safety campaign.

## 1. Environment

```bash
cd ~/sumo_traffic_project

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Adjust this only if SUMO is installed elsewhere.
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
sumo --version
```

Optional: if this succeeds, add `--use-libsumo` to training/validation commands
to remove the TraCI socket boundary:

```bash
python -c 'import libsumo; print("libsumo available")'
```

If an old sequential trainer is still running, inspect and stop only that
trainer before starting v3:

```bash
pgrep -af 'train_map_agnostic_(multimap|policy).py'
pkill -TERM -f 'train_map_agnostic_multimap.py'
```

## 2. Create the train/validation/test map corpus

The generator uses Overpass QL (the same language as Overpass Turbo) and
`netconvert`. Fremont and Santa Clara are not in the supplied training
manifest; keep them as zero-shot benchmarks.

```bash
nohup python -u generate_map_corpus.py \
  --config map_corpus_regions.json \
  --output-dir generated_map_corpus \
  > generate_map_corpus.log 2>&1 &
echo $! > generate_map_corpus.pid

tail -f generate_map_corpus.log
```

The supplied region plan deliberately mixes Bay Area corridors with grid,
radial, irregular, and European layouts. Independently jittered crop sizes
prevent every training domain from having the same rectangular scale. Fremont
and Santa Clara remain outside the training split so they continue to be honest
zero-shot benchmarks. The output is
`generated_map_corpus/manifest.json`.

## 3. Pre-generate the training demand bank

Three map-normalized traffic intensities and two route seeds give each map six
long scenarios. Worker-specific shuffling prevents every round from seeing the
same route first.

```bash
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

tail -f generate_training_demand_bank_v3.log
```

## 4. Smoke test

This confirms SUMO, multiprocessing, PPO, masks, and checkpoint export before a
long run:

```bash
python -u train_map_agnostic_multiagent.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --demand-bank-manifest training_demand_bank_v3/manifest.json \
  --model-path models/map_agnostic_v3_smoke \
  --best-model-path models/map_agnostic_v3_smoke_best \
  --rounds 1 \
  --num-map-workers 1 \
  --rollouts-per-map-visit 1 \
  --rollout-steps 8 \
  --episode-seconds 600 \
  --decision-seconds 10 \
  --embed-dim 32 \
  --graph-layers 1 \
  --no-validate-every-round \
  --progress-file smoke_progress.json \
  --restart
```

## 5. Full training campaign

Run one `nohup` trainer. `--num-map-workers 4` already creates four parallel
SUMO collectors; do not start independent trainers against the same model path.

```bash
nohup python -u train_map_agnostic_multiagent.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --demand-bank-manifest training_demand_bank_v3/manifest.json \
  --model-path models/map_agnostic_multiagent_v3 \
  --best-model-path models/map_agnostic_multiagent_v3_best \
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
  --progress-file map_agnostic_multiagent_progress.json \
  --restart \
  > map_agnostic_v3_training.log 2>&1 &
echo $! > map_agnostic_v3_training.pid
```

The central learner uses CUDA automatically when available; each SUMO worker
uses one CPU thread. Increase map workers only if CPU/RAM headroom remains.

Live partial percentages, per-map rollout percentages, throughput, and rough
ETA:

```bash
watch -n 10 'python monitor_multiagent_training.py map_agnostic_multiagent_progress.json --refresh 0'
```

Useful diagnostics:

```bash
tail -f map_agnostic_v3_training.log
ps -o pid,ppid,etime,%cpu,%mem,cmd -p "$(cat map_agnostic_v3_training.pid)"
nvidia-smi
```

To resume after interruption, repeat the exact full command without
`--restart` and append to the log. The checkpoint stores an update-schedule
fingerprint and refuses a mismatched resume command.

The deployable checkpoints are:

- `models/map_agnostic_multiagent_v3.zip` — last update.
- `models/map_agnostic_multiagent_v3_best.zip` — best held-out validation
  score; use this for final evaluation.
- `*_trainer.pt` — optimizer/resume state, not the deployment file.

## 6. Thorough held-out evaluation

The evaluator generates one route file per map/rate/seed and replays that exact
demand under native SUMO, normalized MaxPressure, and the learned policy. This
eliminates the former “faster controller gets more spawned demand” bias.

Thirty paired seeds, three rates (one above the training range), Fremont, Santa
Clara, and the manifest test split form the exhaustive zero-shot campaign:

```bash
SEEDS=$(seq -s, 1001 1030)

nohup python -u launch_comprehensive_evaluation.py \
  --benchmarks fremont=new_map.net.xml,santaclara=santa_clara.net.xml \
  --manifest generated_map_corpus/manifest.json \
  --manifest-splits test \
  --model-path models/map_agnostic_multiagent_v3_best \
  --output-dir runs/map_agnostic_v3_comprehensive \
  --rates 6,12,18 \
  --seeds "$SEEDS" \
  --episode-seconds 1200 \
  --eval-steps 2500 \
  --metrics-interval 20 \
  --model-update-period 10 \
  --max-vehicles 3000 \
  --max-parallel 8 \
  --demand-generation-workers 8 \
  --progress-file comprehensive_eval_progress.json \
  > comprehensive_eval.log 2>&1 &
echo $! > comprehensive_eval.pid
```

To include the three historical five-action policies in the same fixed-demand
campaign, add these flags (adjust paths to the actual checkpoints):

```bash
  --legacy-model mixed_tls_v1=models/mixed_tls_v1 \
  --legacy-model robust_v1=models/robust_v1 \
  --legacy-model original_40tls_good=models/original_40tls_good
```

Their observation/action semantics remain historical; the analyzer labels each
separately and reports each one against native SUMO.

This is 4 maps × 3 rates × 30 seeds × 3 controllers = 1,080 runs. For a
smaller credible core report, omit `--manifest` and `--manifest-splits`; the
Fremont/Santa Clara campaign is 540 runs.

Monitor it:

```bash
watch -n 10 'python monitor_comprehensive_evaluation.py comprehensive_eval_progress.json'
tail -f comprehensive_eval.log
```

Analysis runs automatically after all simulations. It can be repeated without
rerunning SUMO:

```bash
python -u analyze_comprehensive_evaluation.py \
  runs/map_agnostic_v3_comprehensive \
  | tee runs/map_agnostic_v3_comprehensive/statistical_summary.log
```

The report includes:

- learned vs native, MaxPressure vs native, and learned vs MaxPressure;
- throughput, speed, mean/max queue, mean/max waiting, and recovery metrics;
- paired wins/losses, two-sided 95% paired Student-t CIs, paired effect size;
- Holm-adjusted p-values across all map/rate/metric tests;
- pooled summaries plus the worst map/rate condition for every metric;
- fixed-demand, equal-simulation-time, and equal-sample-grid fairness checks;
- policy switch/forced-switch/invalid-action rates and local
  spillback/starvation diagnostics in raw CSV rows.

Important: interpret condition-level and worst-case results first. A good
pooled mean does not establish map generalization if any held-out map has a
large, clear regression. Do not tune the model after examining test/Fremont/
Santa Clara outcomes and continue calling those maps unseen; make changes using
training/validation maps, then run a fresh test campaign.
