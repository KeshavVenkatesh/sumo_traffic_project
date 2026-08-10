# Ambulance routing and signal priority (schema v5)

Schema v5 adds emergency routing and learned signal priority without replacing
the validated schema-v3 normal-traffic controller. The schema-v3 checkpoint is
frozen. A smaller emergency network may override one legal native green phase
only while an ambulance is in a two-to-three-signal rolling corridor or while
that corridor is recovering.

When neither condition exists, the residual override is disabled and the
frozen schema-v3 policy supplies the action. A shared fail-closed safety shield
may replace an unsafe phase request with a hold if its receiving lanes lack the
required 18 m entrance gap. That shield is identical in every ablation.

Schema v5 supersedes every earlier ambulance checkpoint, including the
unaudited v4 package and all checkpoints trained with the old single-light
file. Start fresh; the checkpoint loader deliberately rejects older schemas.

## What is implemented

| Component | Implementation |
|---|---|
| Ambulance O/D schedule | Deterministic per map, demand seed, and episode; requires a route of sufficient length crossing at least two controlled TLS |
| Concurrency | Planned from free-flow durations before the run; due spawns are never delayed by controller-dependent arrival times |
| Boundary-safe insertion | Each request is queued one SUMO microstep early with its original fixed `depart` time, so an ambulance departing on a policy boundary is visible for that decision |
| Free-flow reference | SUMO `findRoute()` edges and `Stage.travelTime` |
| Traffic-aware routing | Aggregated travel-time route at insertion, then conservative rerouting every 10–15 seconds |
| Reroute hysteresis | Requires at least 8 seconds or 5% savings; disabled within 100 m of the next TLS; jitter uses an independent deterministic stream per ambulance |
| Lifecycle accounting | Exact one-second departed, arrived, starting-teleport, collision, removal, insertion-failure, and episode-censor events |
| Signal observation | Ambulance movement, next-TLS link, distance, ETA, route order, recent time loss/stopping, protected/permissive phase service |
| Signal execution | Validated native green states with explicit nonblocking 4 s yellow and 2 s all-red clearance; every action advances the same simulated time |
| Exit-space safety | Requires 18 m on every receiving lane both when a phase is requested and again after yellow/all-red, immediately before green; otherwise the TLS remains all-red and retries |
| Emergency policy | Permutation-equivariant residual network around the frozen schema-v3 logits |
| Recovery policy | Normalized MaxPressure teacher clears the largest safe ordinary-traffic pressure after preemption |
| Hard authority limits | Preparation starts at ETA ≤25 seconds; no override after 45 seconds unless ETA is ≤8 seconds; no override into a phase with <8% downstream space |
| Curriculum | One ambulance and easier demand first; heavier demand and up to two simultaneous ambulances later |
| Reward | Incremental route progress, time loss, stopping, TLS clearance, arrival/failure/censor, normal-traffic loss, safety, and override cost |
| Ordinary delay | Uses all departed vehicles, including unfinished vehicles at the horizon; wait is integrated per microstep rather than read from SUMO's rolling waiting-memory window |
| Promotion gate | ≤5% ordinary delay increase, ≤2% throughput loss, no safety/lifecycle failures, no per-scenario ambulance regression, and no regression versus deterministic preemption |

The ambulance uses ordinary car-following, lane-changing, red-light, and
collision-avoidance rules. It has no SUMO blue-light device, no forced lane
changes, no red running, and no teleport-based assistance. Its vehicle class
remains `passenger` so passenger-only roads remain routable; its shape and
performance parameters identify it as an ambulance.

SUMO's automatic time-to-teleport is disabled (`-1`) for the whole ambulance
experiment. A stuck trip remains visible as unfinished instead of being moved
or removed by the simulator. Fixed background-demand files bind every vehicle
to an audited passenger vType containing `jmIgnoreKeepClearTime="-1"`.

`max_active_ambulances` limits planned overlap, using 1.5× each route's
free-flow duration. Congestion can cause more actual overlap, but the fixed
spawn schedule is never changed in response; otherwise a faster controller
would receive different emergency demand.

## 1. Environment and base checkpoint

Use the same environment as schema v3:

```bash
cd ~/sumo_traffic_project
source .venv/bin/activate
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

python -c 'import torch, gymnasium, sb3_contrib, traci; print("dependencies OK")'
sumo --version
```

The base checkpoint must be the selected schema-v3 all-intersection model:

```bash
test -f models/map_agnostic_multiagent_v3_best.zip
```

Schema v5 records the SHA-256 of that `.zip`. Loading fails closed if a
different base checkpoint is supplied later.

## 2. Generate checksummed training demand

Use pre-generated controller-independent demand. Rates are normalized by
passenger-lane kilometers, so the same rate means comparable traffic intensity
on differently sized maps.

```bash
nohup python -u generate_training_demand_bank.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --output-dir ambulance_training_demand_v5 \
  --rates 4,8,12 \
  --seeds 301,302,303,304,305,306 \
  --episode-seconds 3600 \
  --workers 8 \
  > ambulance_training_demand_v5.log 2>&1 &
echo $! > ambulance_training_demand_v5.pid
```

The schema-v2 demand manifest stores both network and route-file hashes.
The generator also installs the audited fixed-demand passenger vType. Schema v5
rejects an older route bank that lacks it; generate into the new v5 directory
instead of reusing an old v4 bank.

Generate a separate fixed validation bank using only validation maps and
validation seeds:

```bash
nohup python -u generate_training_demand_bank.py \
  --manifest generated_map_corpus/manifest.json \
  --splits validation \
  --output-dir ambulance_validation_demand_v5 \
  --rates 6,12,18 \
  --seeds 9001,9002 \
  --episode-seconds 1200 \
  --workers 8 \
  > ambulance_validation_demand_v5.log 2>&1 &
echo $! > ambulance_validation_demand_v5.pid
```

Do not reuse test maps or test seeds during checkpoint selection.

## 3. Smoke test

This checks checkpoint loading, SUMO startup, deterministic ambulance route
construction, rollout collection, and one PPO update on the first training
map:

```bash
SMOKE_MAP="$(python - <<'PY'
import json
from pathlib import Path

manifest = Path("generated_map_corpus/manifest.json")
payload = json.loads(manifest.read_text(encoding="utf-8"))
records = payload if isinstance(payload, list) else payload["maps"]
record = next(
    item
    for item in records
    if str(item.get("split", "train")) == "train"
)
path = Path(record.get("net_file") or record["path"])
if not path.is_absolute():
    path = (Path.cwd() / path).resolve()
print(path)
PY
)"

python -u train_ambulance_override.py \
  --manifest generated_map_corpus/manifest.json \
  --splits __explicit_smoke_map__ \
  --maps "$SMOKE_MAP" \
  --demand-bank-manifest ambulance_training_demand_v5/manifest.json \
  --base-model-path models/map_agnostic_multiagent_v3_best \
  --model-path models/map_agnostic_emergency_v5_smoke \
  --best-model-path models/map_agnostic_emergency_v5_smoke_best \
  --rounds 1 \
  --num-map-workers 1 \
  --rollouts-per-map-visit 1 \
  --rollout-steps 360 \
  --episode-seconds 3600 \
  --decision-seconds 10 \
  --emergency-embed-dim 32 \
  --emergency-graph-layers 1 \
  --ambulance-first-spawn 60 \
  --ambulance-interval-min 600 \
  --ambulance-interval-max 600 \
  --max-ambulances-per-episode 2 \
  --no-validate-every-round \
  --progress-file ambulance_v5_smoke_progress.json \
  --restart
```

A map that cannot produce a sufficiently long route through two usable traffic
lights fails explicitly. Inspect the map before reducing the route constraints;
otherwise the task no longer tests coordinated signal priority.

Before full training, inspect the final smoke update line. It must report at
least one arrival, a positive finite `response_s`—never an unconditional
zero-second arrival—and `failed_or_censored=0`. Per-worker files in
`runs/ambulance_v5_sumo_logs/` contain SUMO diagnostics if startup or TraCI
fails.

## 4. Full training

Only the emergency network is optimized. The base schema-v3 weights never enter
the optimizer.

```bash
nohup python -u train_ambulance_override.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --demand-bank-manifest ambulance_training_demand_v5/manifest.json \
  --base-model-path models/map_agnostic_multiagent_v3_best \
  --model-path models/map_agnostic_emergency_v5 \
  --best-model-path models/map_agnostic_emergency_v5_best \
  --rounds 8 \
  --num-map-workers 4 \
  --rollouts-per-map-visit 12 \
  --rollout-steps 64 \
  --episode-seconds 3600 \
  --decision-seconds 10 \
  --emergency-embed-dim 96 \
  --emergency-graph-layers 1 \
  --authority-start 0.5 \
  --authority-end 1.0 \
  --ambulance-first-spawn 30 \
  --ambulance-interval-min 90 \
  --ambulance-interval-max 240 \
  --max-ambulances-per-episode 16 \
  --ambulance-min-route-distance 1500 \
  --ambulance-min-route-edges 12 \
  --ambulance-min-route-tls 2 \
  --reroute-interval 12 \
  --reroute-jitter 2 \
  --reroute-min-savings-seconds 8 \
  --reroute-min-savings-fraction 0.05 \
  --no-reroute-within-tls 100 \
  --recovery-seconds 30 \
  --max-preemption-seconds 45 \
  --prepare-eta-seconds 25 \
  --serve-eta-seconds 12 \
  --validate-every-round \
  --validation-splits validation \
  --validation-seeds 9001,9002 \
  --validation-episode-seconds 1200 \
  --validation-demand-bank-manifest \
    ambulance_validation_demand_v5/manifest.json \
  --validation-workers 2 \
  --ordinary-delay-budget-percent 5 \
  --throughput-budget-percent 2 \
  --progress-file ambulance_v5_progress.json \
  --restart \
  > ambulance_v5_training.log 2>&1 &
echo $! > ambulance_v5_training.pid
```

Four map workers mean four SUMO processes plus one learner. Do not launch
multiple trainers writing the same model path.

Monitor without following the full log:

```bash
python monitor_ambulance_training.py ambulance_v5_progress.json
```

Or render once:

```bash
python monitor_ambulance_training.py \
  ambulance_v5_progress.json --refresh 0
```

Resume by repeating the original command without `--restart`. Resume fails if
the base checkpoint, map plan, demand manifest, model shape, ambulance
semantics, curriculum, or PPO schedule changed.

Each SUMO worker writes its own error log below
`runs/ambulance_v5_sumo_logs/`. If TraCI closes unexpectedly, the Python error
prints the exact corresponding log path.

## 5. Checkpoints

Training writes:

- `map_agnostic_emergency_v5.pt`: inference weights.
- `map_agnostic_emergency_v5_contract.json`: strict schema, feature order,
  frozen-base hash, timing, routing, observation, and corridor contract.
- `map_agnostic_emergency_v5_trainer.pt`: optimizer, completed-update state,
  and strict training-plan signature for safe resume.
- `map_agnostic_emergency_v5_best.*`: best checkpoint that passed every
  validation constraint.
- `map_agnostic_emergency_v5_best_validation.json`: evidence for its selection.

Use the `_best` checkpoint for held-out testing. A checkpoint with higher
training reward is not promoted if it violates the ordinary-traffic budget.

## 6. Five-way held-out evaluation

Generate demand for the exact four-map July 30 held-out snapshot. Fresno and
San Diego are the freshly generated final-test maps; Fremont and Santa Clara
are the additional external zero-shot benchmarks. Passing all four absolute
paths explicitly prevents a two-map `test` split from being mistaken for the
full four-map campaign. This example reproduces the earlier low/medium/high
rates and 30 seeds:

```bash
HELDOUT_MAPS="/users/sriramv/sumo_eval_v3_inputs_20260730/generated_test_maps/fresno/fresno.net.xml,/users/sriramv/sumo_eval_v3_inputs_20260730/generated_test_maps/san_diego/san_diego.net.xml,/users/sriramv/sumo_eval_v3_inputs_20260730/external_zero_shot/fremont.net.xml,/users/sriramv/sumo_eval_v3_inputs_20260730/external_zero_shot/santa_clara.net.xml"

nohup python -u generate_training_demand_bank.py \
  --manifest \
    /users/sriramv/sumo_eval_v3_inputs_20260730/generated_test_maps/manifest.json \
  --splits __explicit_maps_only__ \
  --maps "$HELDOUT_MAPS" \
  --output-dir ambulance_heldout_demand_v5 \
  --rates 6,12,18 \
  --seeds \
    1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030 \
  --episode-seconds 1200 \
  --workers 8 \
  > ambulance_heldout_demand_v5.log 2>&1 &
echo $! > ambulance_heldout_demand_v5.pid
```

Run all five ablations:

```bash
HELDOUT_MAPS="/users/sriramv/sumo_eval_v3_inputs_20260730/generated_test_maps/fresno/fresno.net.xml,/users/sriramv/sumo_eval_v3_inputs_20260730/generated_test_maps/san_diego/san_diego.net.xml,/users/sriramv/sumo_eval_v3_inputs_20260730/external_zero_shot/fremont.net.xml,/users/sriramv/sumo_eval_v3_inputs_20260730/external_zero_shot/santa_clara.net.xml"

nohup python -u evaluate_ambulance_system.py \
  --maps "$HELDOUT_MAPS" \
  --base-model-path models/map_agnostic_multiagent_v3_best \
  --emergency-model-path models/map_agnostic_emergency_v5_best \
  --demand-bank-manifest ambulance_heldout_demand_v5/manifest.json \
  --output-json ambulance_heldout_v5.json \
  --seeds \
    1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030 \
  --episode-seconds 1200 \
  --decision-seconds 10 \
  --workers 8 \
  --ordinary-delay-budget-percent 5 \
  --throughput-budget-percent 2 \
  > ambulance_heldout_v5.log 2>&1 &
echo $! > ambulance_heldout_v5.pid
```

For four maps, three rates, 30 seeds, and five controllers, this is 1,800 SUMO
runs. Each `(map, rate, seed)` group contains:

1. Free-flow route + frozen base signals.
2. Traffic-aware route + frozen base signals.
3. Free-flow route + learned emergency signals.
4. Traffic-aware route + learned emergency signals.
5. Traffic-aware route + deterministic emergency preemption.

The evaluator rejects a group unless all five runs have the same checksummed
background demand and the same ambulance schedule fingerprint. It reports
ambulance mean/p95 dispatch-to-arrival response time, in-network trip time,
insertion delay, time loss, stops, completion/failures; ordinary delay,
throughput, queue, speed; recovery time; collisions; and teleports. Checkpoint
selection uses dispatch-to-arrival response time so source-insertion delay
cannot be hidden.

The primary signal-control comparison is (2) versus (4). The combined system
comparison is (1) versus (4). Routing-only and deterministic-preemption
ablations show where each improvement comes from.

## 7. Promotion criteria

`eligible=true` requires all of the following on the learned traffic-aware
system:

- ambulance response time and completion are no worse than traffic-aware
  routing with frozen base signals, both pooled and in every paired scenario;
- response time is no worse than deterministic emergency preemption;
- ordinary all-departed mean time loss increases by at most 5%;
- ordinary completed throughput decreases by at most 2%;
- no collision, teleport, invalid policy action, invalid signal transition,
  insertion failure, unexplained removal, censored ambulance, or unrecovered
  preemption event.

The selection score is 75% worst-scenario ambulance gain and 25% mean-scenario
gain. This keeps a checkpoint from hiding a bad map or traffic level behind a
good global average.

## 8. Tests

```bash
python -m pytest -q
```

The focused tests cover route-to-movement indexing, deterministic schedules,
boundary-safe insertion, vehicle-type creation order, cosmetic TraCI failures,
exact arrival semantics, terminal censoring, audited demand vTypes, ambulance
exclusion from ordinary observations, the 18 m request/activation safety
checks, safe teacher actions, zero-initialized residual behavior, and
checkpoint/base-hash compatibility.
