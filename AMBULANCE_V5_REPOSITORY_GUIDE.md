# Ambulance v5 Repository Guide

This directory documents the exact reviewed Ambulance v5 round-40 research snapshot, its source code, model artifacts, experiment inputs, and 1,080-run final benchmark.

## Model and scientific status

- Published checkpoint: `unpromoted_final_round_40`
- Inference checkpoint: `map_agnostic_emergency_v5_19train_production.pt`
- Resumable trainer checkpoint: `map_agnostic_emergency_v5_19train_production_trainer.pt`
- There is no Ambulance v5 `_best.pt` checkpoint because round 40 did not pass the validation-promotion gates.
- The checkpoint did not pass the strict zero-event safety gate.
- The completed test matrix is frozen and must not be used for additional model tuning.

The final benchmark contains:

- 4 unseen test maps: Fremont, Santa Clara, Fresno, and San Diego
- 3 traffic rates per map: 6, 12, and 18
- 30 paired seeds per map/rate condition
- 3 controllers: Native SUMO, MaxPressure, and learned signals
- 1,080 total runs

The learned controller reduced pooled mean ambulance response time by approximately 21.52% versus Native SUMO and 14.31% versus MaxPressure. Mean response time improved in all 12 map/rate conditions.

Against MaxPressure, pooled ordinary-traffic differences were not statistically significant. San Diego at rate 6 was the condition-level exception where ordinary-traffic performance became worse. Collision and teleport counts were lowest for the learned controller, but all controllers failed the strict safety criterion.

## Repository layout

| Location | Contents |
|---|---|
| `ambulance_v5/*.py` | Runnable training, simulation, controller, and analysis code |
| `ambulance_v5/*.json` | Training split/configuration records and progress snapshots |
| `ambulance_v5/tests/` | Unit and integration tests |
| `ambulance_v5/docs/` | Training notes, provenance, and release documentation |
| `ambulance_v5/models/` | Checkpoint contract and weight-download instructions |
| `ambulance_v5/manifests/` | Map, demand, archive, checksum, and file inventories |
| `ambulance_v5/results/final_test/` | Human-readable summaries from the final benchmark |
| GitHub prerelease | Weights, raw results, maps, demands, logs, and frozen inputs |

## Source-code files

| File | Purpose |
|---|---|
| `ambulance_checkpoint.py` | Saves, loads, and validates ambulance checkpoints |
| `ambulance_curriculum.py` | Defines the ambulance training curriculum |
| `ambulance_emergency.py` | Emergency-vehicle state, routing, observation, and intervention logic |
| `ambulance_multiagent_worker.py` | SUMO rollout worker for ambulance training |
| `ambulance_system.py` | Integrates emergency routing and traffic-signal control |
| `analyze_ambulance_native_maxpressure.py` | Produces paired statistics and robustness summaries |
| `checkpoint_contract.py` | Enforces checkpoint architecture and metadata compatibility |
| `compare_fixed_vs_single_vs_all_model_realistic.py` | Comparative simulation harness |
| `evaluate_ambulance_system.py` | Runs evaluations and records response, traffic, and safety metrics |
| `fixed_demand.py` | Shared deterministic-demand utilities |
| `generate_fixed_demand.py` | Generates one fixed demand and ambulance schedule |
| `generate_fixed_demand_bank.py` | Generates paired evaluation demand banks |
| `generate_map_corpus.py` | Generates and records the multimap SUMO corpus |
| `generate_training_demand_bank.py` | Generates training and validation demand banks |
| `map_agnostic_multiagent_worker.py` | General map-agnostic rollout worker |
| `map_agnostic_policy.py` | Map-agnostic policy network and action logic |
| `map_agnostic_tls.py` | Traffic-light observations and movement representation |
| `monitor_ambulance_training.py` | Reports training and validation progress |
| `realistic_all_intersections_fixed_cycle.py` | Main realistic SUMO scenario engine |
| `safe_residual_controller.py` | Safety-constrained residual signal controller |
| `traffic_rl_map_agnostic_env.py` | Map-agnostic reinforcement-learning environment |
| `train_ambulance_override.py` | Primary Ambulance v5 training entry point |
| `train_ambulance_override_exit_shield_ablation.py` | Exit-shield ablation experiment |
| `train_ambulance_override_shield_liveness.py` | Shield-liveness experiment |
| `train_ambulance_override_smoke_diagnostic.py` | Short diagnostic training experiment |
| `train_map_agnostic_multiagent.py` | General map-agnostic multiagent trainer |
| `train_map_agnostic_multimap.py` | Multimap training entry point |
| `validate_controller_promotion.py` | Applies promotion and safety gates |
| `run_ambulance_native_maxpressure_benchmark.sh` | Launches the final benchmark |
| `ambulance_native_maxpressure_benchmark.patch` | Preserves the benchmark patch |

## Important configuration files

| File or pattern | Purpose |
|---|---|
| `ambulance_v5_19_5_4_split_lock.json` | Locked 19-training/5-validation/4-test split |
| `ambulance_v5_19_5_4_split_lock_phase32.json` | Phase-capacity-32 version of the split |
| `ambulance_v5_phase_capacity_32_patch.json` | Phase-capacity adjustment metadata |
| `map_corpus_regions.json` | Geographic regions used to create the map corpus |
| `map_corpus_train_validation_v5.json` | Ambulance v5 corpus partition |
| `ambulance_v5_*_progress.json` | Production and diagnostic progress records |
| `models/*_contract.json` | Model schema required to load the checkpoint |

## Tests

| File | What it checks |
|---|---|
| `test_ambulance_checkpoint.py` | Checkpoint save/load and compatibility |
| `test_ambulance_curriculum.py` | Curriculum transitions |
| `test_ambulance_emergency.py` | Emergency-state and control behavior |
| `test_ambulance_evaluation.py` | Evaluation metrics and aggregation |
| `test_ambulance_recovery_accounting.py` | Recovery-event accounting |
| `test_ambulance_system.py` | Integrated ambulance-system behavior |
| `test_map_agnostic_policy.py` | Policy dimensions and map-independent behavior |
| `test_map_agnostic_tls.py` | Traffic-light and movement representations |
| `test_multiagent_ppo.py` | Multiagent PPO behavior |
| `test_multimap_sampling.py` | Map and scenario sampling |
| `test_native_phase_catalog.py` | Native SUMO phase catalog handling |
| `test_safe_residual_controller.py` | Residual-controller safety logic |

## Final-test summaries committed to Git

| File | Purpose |
|---|---|
| `RUN_CONFIGURATION.txt` | Exact benchmark launch configuration |
| `benchmark_summary.txt` | Concise pooled results |
| `statistical_summary.csv` | Estimates, confidence intervals, tests, and effect sizes |
| `robustness_summary.json` | Results across all 12 conditions |
| `smoke.json` | Benchmark smoke-test output |
| `smoke_demand_manifest.json` | Smoke-test demand metadata |
| `analysis.log` | Final analysis log |
| `demand_preparation.log` | Demand-preparation log |
| `status.txt` | Final benchmark status |
| `RESULT_SHA256SUMS` | Checksums recorded by the benchmark |

The complete aggregate JSON, per-run outputs, and evaluation logs are stored in the raw-results release archive.

## GitHub release assets

Release tag: `ambulance-v5-round40-20260808`

| Asset | Contents |
|---|---|
| `ambulance-v5-round40-models.tar.gz` | Policy, trainer state, contract, progress, and v3 base checkpoint |
| `ambulance-v5-training-evidence.tar.gz` | Validation outputs, SUMO logs, and provenance backups |
| `ambulance-v5-training-demand-bank.tar.gz` | Full training demand bank |
| `ambulance-v5-validation-demand-bank.tar.gz` | Full validation demand bank |
| `ambulance-v5-map-corpus.tar.gz` | Dereferenced 28-network SUMO corpus |
| `ambulance-v5-final-test-1080-runs.tar.gz` | Complete final benchmark directory |
| `ambulance-v5-final-test-inputs.tar.gz` | Frozen paired demands and ambulance schedules |
| `RELEASE_ASSET_SHA256SUMS.txt` | SHA-256 checksums |
| `RELEASE_ASSET_MANIFEST.tsv` | Archive names, sizes, and hashes |
| `RELEASE_ASSET_CONTENTS.txt` | Exact contents of every archive |

## Restoring the complete snapshot

From the repository root:

~~bash
mkdir -p ambulance_v5/artifacts

gh release download ambulance-v5-round40-20260808 \
  --repo KeshavVenkatesh/sumo_traffic_project \
  --dir ambulance_v5/artifacts

(
  cd ambulance_v5/artifacts
  sha256sum -c RELEASE_ASSET_SHA256SUMS.txt
)

for archive in ambulance_v5/artifacts/*.tar.gz
do
    tar -xzf "$archive" -C ambulance_v5
done
~~

This reconstructs the model, map corpus, demand banks, training evidence, benchmark inputs, and raw results beneath `ambulance_v5/`.

## Reproduction notes

Use a SUMO-compatible Python environment and run commands from `ambulance_v5/`.

~~bash
cd ambulance_v5
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 -m pytest -q tests
~~

Before reporting new results, verify the checkpoint contract, fixed demand/schedule pairing, map split, and safety-accounting definitions. If the algorithm is changed using the published test results, evaluate the modified system on a new untouched test set.
