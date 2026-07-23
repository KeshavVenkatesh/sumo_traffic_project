Map-agnostic traffic-signal policy (schema v2)

Schema v2 replaces the old compass-slot/five-action controller. It is designed for one shared checkpoint to operate on intersections whose lane counts,movement sets, and safe phase counts differ.

What changed

A movement is an incoming-edge to outgoing-edge connection, not a fixedNB-L, lane-index, or phase-slot input position.

Queue, occupancy, speed, waiting, arrival rate, ETA bins, and downstream storage are normalized by local physical capacity, speed limit, or a fixed time reference. VecNormalize is deliberately disabled.

An action is hold or one of the current intersection's safe candidate phases. The policy scores every candidate with the same neural function.

Candidate greens come from the active native tlLogic program; yellow,red-yellow, and duplicate states are excluded. Schema v2 inserts a conservative yellow plus all-red clearance between selected native greens.Phase membership retains protected (G) versus permissive/stop (g/s)service strength instead of collapsing both to an identical phase.

Minimum green, maximum green, yellow, all-red, invalid candidates, and downstream spillback constraints remain outside the learned policy.

The reward contains only bounded local ratios: discharge, served pressure,queue improvement/level, waiting, spillback, starvation, and switching.

A graph-attention encoder processes an unordered movement graph. Phase embeddings are pooled from the movements each phase serves.

Training rotates one checkpoint through maps and intersections with balanced map sampling and randomized OD traffic.

The old schema-v1 checkpoints have 30/46 inputs and five positional actions.They cannot be safely converted to schema v2. Keep them for historical comparisons, but use a new model path for schema-v2 training.

Install

Use the same SUMO installation as the rest of the repository, then install theRL dependencies:

python -m pip install -r requirements-map-agnostic.txt

Generate a varied map corpus

generate_map_corpus.py submits standard Overpass QL queries—the same query language used at https://overpass-turbo.eu/—and converts accepted regions with netconvert. Large generated .osm and .net.xml files stay outside git.

nohup python -u generate_map_corpus.py \
  --config map_corpus_regions.json \
  --output-dir generated_map_corpus \
  > generate_map_corpus.log 2>&1 &

Every query is also saved as .overpassql, so it can be pasted into OverpassTurbo for visual inspection. The resulting manifest has explicit train,validation, and test splits.

For a valid zero-shot experiment, do not put Fremont or Santa Clara in the training split. They should remain held-out benchmarks. Add many training regions whose intersection primitives cover three/four-way junctions,different lane counts, asymmetric approaches, and different safe phase counts.

Validate topology support

TRAFFIC_NET_FILE=new_map.net.xml \
python traffic_rl_map_agnostic_env.py --list-tls-json

This prints the movement and phase counts for every usable TLS. If any exceedsMAX_MOVEMENTS=64 or MAX_PHASES=16, increase the constant before training;the code fails rather than silently dropping movements.

Controllers with only one stable native green have no meaningful phase choice(these are commonly pedestrian or subordinate signal programs). Schema v2 leaves those programs under native SUMO timing rather than asking PPO to choose between one action and itself.

Train one shared policy across maps

Start with a short smoke test:

python train_map_agnostic_multimap.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --rounds 1 \
  --max-tls-per-map 2 \
  --steps-per-tls 2048 \
  --num-envs 1 \
  --restart

Then run a real campaign:

nohup python -u train_map_agnostic_multimap.py \
  --manifest generated_map_corpus/manifest.json \
  --splits train \
  --model-path models/traffic_signal_map_agnostic_v2 \
  --rounds 4 \
  --max-tls-per-map 24 \
  --steps-per-tls 10000 \
  --num-envs 4 \
  --restart \
  > map_agnostic_training.log 2>&1 &

There is one PPO learner and one checkpoint writer. --num-envs 4 parallelizesSUMO data collection inside each task. Do not launch independent nohup trainers that all write the same model file; they would overwrite each other's updates. Resume an interrupted campaign by repeating the command without--restart.

The traffic scenario sampler randomizes demand, initial population, OD routes,turns, and density around the supplied centers. The map orchestrator samples each map evenly, rather than letting a map with more traffic lights dominate.It samples a target in vehicles per passenger lane-kilometer (default2.0,10.0) and derives each map's active-vehicle target from its drivable lanelength. --max-vehicle-center is a hard compute cap, not the shared demand level for every map. Schema v2 also removes the legacy evaluator's silent750-vehicle clamp; set MAP_AGNOSTIC_MAX_ACTIVE_CAP if the default cap of 2000is too expensive for a machine.

Training also adds small noise to dynamic sensor features by default(--observation-noise-std 0.01); topology, phase membership, masks, and turnsemantics remain exact. Evaluation automatically disables this noise.

Evaluate held-out maps

Santa Clara:

TRAFFIC_NET_FILE=santa_clara.net.xml \
MODEL_PATH=models/traffic_signal_map_agnostic_v2 \
SEED_LIST=42,43,44,45,46 \
MAX_PARALLEL=5 \
nohup bash launch_map_agnostic_eval.sh \
  > map_agnostic_santaclara_launcher.log 2>&1 &

Fremont:

TRAFFIC_NET_FILE=new_map.net.xml \
MODEL_PATH=models/traffic_signal_map_agnostic_v2 \
SEED_LIST=42,43,44,45,46 \
MAX_PARALLEL=5 \
OUTDIR=map_agnostic_fremont_eval \
nohup bash launch_map_agnostic_eval.sh \
  > map_agnostic_fremont_launcher.log 2>&1 &

Use 30 paired seeds for the final report. Model selection should use only validation maps; do not repeatedly tune on Fremont and still call Fremont an unseen test map.

Show partial percentages while the jobs run:

python monitor_map_agnostic_eval.py --episode-seconds 1200 --refresh 30

Important evaluation note

The existing realistic simulator maintains a target number of active vehicles,so a controller that completes trips faster may cause additional vehicles to bespawned. Keep reporting spawned_total alongside arrivals and congestion. For the strongest publication-grade comparison, the next evaluation upgrade shouldpre-generate and replay an identical OD departure schedule for Native SUMO and the learned controller.
