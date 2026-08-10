#!/usr/bin/env bash
set -euo pipefail

# Final ambulance-v5 comparison against Native SUMO and the exact historical
# normalized-MaxPressure score rule. No validation-selected _best checkpoint
# exists for this completed run, so this launcher intentionally benchmarks the
# round-40 production checkpoint as UNPROMOTED_FINAL. It remains fail-closed:
# training must be 240/240, round 40 must be the last validation, that round
# must remain explicitly ineligible, and fixed demand / smoke checks must pass
# before the 1080-run campaign can start.

USER_ROOT="/users/sriramv"
PROJECT="$USER_ROOT/sumo_traffic_project_ambulance_v5_reviewed"
VENV="$USER_ROOT/.venvs/sumo-eval-v3"
PYTHON="$VENV/bin/python"
INPUT_ROOT="$USER_ROOT/sumo_eval_v3_inputs_20260730"
OLD_FULL_ROOT="$USER_ROOT/sumo_eval_v3_full_20260730"
RUN_ROOT="${RUN_ROOT:-$USER_ROOT/ambulance_v5_native_mp_full_20260807}"
PATCH_FILE="${PATCH_FILE:-$PROJECT/ambulance_native_maxpressure_benchmark.patch}"
ANALYZER_FILE="${ANALYZER_FILE:-$PROJECT/analyze_ambulance_native_maxpressure.py}"

EXPECTED_INPUT_SUMS_SHA256="c9f8be53ad7501d9e6473e57dcf9dd2e8ad08ee5edf53f2f80cf6b66aee8da40"
BASE_STEM="$PROJECT/models/map_agnostic_multiagent_v3_best"
EVAL_STEM="$PROJECT/models/map_agnostic_emergency_v5_19train_production"
PROGRESS_FILE="$PROJECT/ambulance_v5_19train_production_progress.json"
EVAL_VALIDATION="$PROJECT/runs/ambulance_v5_19train_production_validation/round_040.json"
EVAL_CONTRACT="${EVAL_STEM}_contract.json"
EVAL_MODEL="${EVAL_STEM}.pt"
EVAL_TRAINER="${EVAL_STEM}_trainer.pt"
CHECKPOINT_KIND="unpromoted_final_round_40"
SEEDS="$(seq -s, 1001 1030)"

FREMONT="$INPUT_ROOT/external_zero_shot/fremont.net.xml"
SANTA_CLARA="$INPUT_ROOT/external_zero_shot/santa_clara.net.xml"
FRESNO="$INPUT_ROOT/generated_test_maps/fresno/fresno.net.xml"
SAN_DIEGO="$INPUT_ROOT/generated_test_maps/san_diego/san_diego.net.xml"
MAPS="$FREMONT,$SANTA_CLARA,$FRESNO,$SAN_DIEGO"

DEMAND_DIR="$RUN_ROOT/demand"
DEMAND_MANIFEST="$DEMAND_DIR/manifest.json"
SMOKE_MANIFEST="$RUN_ROOT/smoke_demand_manifest.json"
SMOKE_JSON="$RUN_ROOT/smoke.json"
FULL_JSON="$RUN_ROOT/ambulance_native_maxpressure_learned.json"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

test -d "$PROJECT" || fail "Missing project: $PROJECT"
test -x "$PYTHON" || fail "Missing evaluation Python: $PYTHON"
test -f "$PATCH_FILE" || fail "Place ambulance_native_maxpressure_benchmark.patch in $PROJECT"
test -f "$ANALYZER_FILE" || fail "Place analyze_ambulance_native_maxpressure.py in $PROJECT"
test ! -e "$RUN_ROOT" || fail "Run root already exists: $RUN_ROOT"

cd "$PROJECT"
source "$VENV/bin/activate"

SUMO_HOME="$($PYTHON -c 'import sumo; print(sumo.SUMO_HOME)')"
LOCAL_LIB="$USER_ROOT/.local/sumo-eval-v3-libs/usr/lib/x86_64-linux-gnu"
export SUMO_HOME
export PATH="$SUMO_HOME/bin:$VENV/bin:$PATH"
export PYTHONPATH="$PROJECT:$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}"
if test -d "$LOCAL_LIB"; then
    export LD_LIBRARY_PATH="$LOCAL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -x "$SUMO_HOME/bin/sumo" || fail "SUMO binary not found under $SUMO_HOME"
PROJ_DB="$(find "$SUMO_HOME" -type f -name proj.db -print -quit 2>/dev/null || true)"
if test -n "$PROJ_DB"; then
    export PROJ_DATA="$(dirname "$PROJ_DB")"
    export PROJ_LIB="$PROJ_DATA"
fi

for required in \
    "$PROGRESS_FILE" \
    "$EVAL_MODEL" \
    "$EVAL_CONTRACT" \
    "$EVAL_VALIDATION" \
    "$EVAL_TRAINER" \
    "${BASE_STEM}.zip" \
    "$INPUT_ROOT/SHA256SUMS" \
    "$OLD_FULL_ROOT/progress.json"; do
    test -f "$required" || fail "Missing required file: $required"
done

for net_file in "$FREMONT" "$SANTA_CLARA" "$FRESNO" "$SAN_DIEGO"; do
    test -f "$net_file" || fail "Missing held-out network: $net_file"
done

actual_input_sums_sha="$(sha256sum "$INPUT_ROOT/SHA256SUMS" | awk '{print $1}')"
test "$actual_input_sums_sha" = "$EXPECTED_INPUT_SUMS_SHA256" || \
    fail "Previous test-input checksum manifest changed"
(
    cd "$INPUT_ROOT"
    sha256sum -c SHA256SUMS >/dev/null
)

"$PYTHON" - "$PROGRESS_FILE" "$EVAL_CONTRACT" "$EVAL_VALIDATION" \
    "${BASE_STEM}.zip" "$OLD_FULL_ROOT/progress.json" "$CHECKPOINT_KIND" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

progress_path, contract_path, validation_path, base_path, old_progress_path = map(Path, sys.argv[1:6])
checkpoint_kind = sys.argv[6]

progress = json.loads(progress_path.read_text(encoding="utf-8"))
assert progress.get("status") == "complete", progress
assert int(progress.get("completed_updates", -1)) == 240, progress
assert int(progress.get("total_updates", -1)) == 240, progress
assert float(progress.get("percentage", -1.0)) >= 100.0 - 1e-9, progress
assert int(progress.get("last_validated_round", -1)) == 40, progress
assert progress.get("best_validation_score") is None, (
    "A constrained-best score exists; do not benchmark the unpromoted final checkpoint",
    progress.get("best_validation_score"),
)

contract = json.loads(contract_path.read_text(encoding="utf-8"))
assert int(contract.get("schema_version", -1)) == 5, contract.get("schema_version")
assert abs(float(contract.get("decision_seconds", -1.0)) - 5.0) < 1e-9

validation = json.loads(validation_path.read_text(encoding="utf-8"))
assert validation.get("eligible") is False, (
    "Round 40 is eligible; selection artifacts should be repaired instead of using unpromoted final"
)
failed_gates = sorted(
    name for name, passed in dict(validation.get("gates", {})).items() if passed is not True
)
assert failed_gates, "Round 40 is ineligible but contains no recorded failed gate"

digest = hashlib.sha256(base_path.read_bytes()).hexdigest()
assert digest == contract.get("base_checkpoint_sha256"), (
    "Ambulance checkpoint was not trained against this frozen base checkpoint",
    digest,
    contract.get("base_checkpoint_sha256"),
)

old = json.loads(old_progress_path.read_text(encoding="utf-8"))
assert old.get("status") == "complete", old
assert int(old.get("completed_jobs", -1)) == 1080, old
assert int(old.get("total_jobs", -1)) == 1080, old
assert int(old.get("failed_jobs", -1)) == 0, old

print("CHECKPOINT_PREFLIGHT=PASS")
print(f"TRAINING_UPDATES={progress['completed_updates']}/{progress['total_updates']}")
print(f"CHECKPOINT_KIND={checkpoint_kind}")
print("VALIDATION_SELECTED_BEST=NONE")
print(f"ROUND_040_ELIGIBLE={validation.get('eligible')}")
print(f"ROUND_040_SELECTION_SCORE={validation.get('selection_score')}")
print("ROUND_040_FAILED_GATES=" + ",".join(failed_gates))
print("PREVIOUS_FULL_CAMPAIGN=PASS")
PY

# Install the opt-in three-way evaluator only if it is not already present.
if ! grep -Fq -- '"--native-maxpressure-benchmark-only"' evaluate_ambulance_system.py; then
    patch --dry-run --batch --forward -p1 < "$PATCH_FILE"
    BACKUP_DIR="$PROJECT/backups/native_maxpressure_benchmark_20260807"
    test ! -e "$BACKUP_DIR" || fail "Source backup already exists: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp -a \
        evaluate_ambulance_system.py \
        map_agnostic_multiagent_worker.py \
        ambulance_multiagent_worker.py \
        "$BACKUP_DIR/"
    (
        cd "$BACKUP_DIR"
        sha256sum -- *.py > SHA256SUMS
    )
    patch --batch --forward -p1 < "$PATCH_FILE"
    echo "BENCHMARK_PATCH=APPLIED"
    echo "SOURCE_BACKUP=$BACKUP_DIR"
else
    echo "BENCHMARK_PATCH=ALREADY_PRESENT"
fi

grep -Fq 'traffic_aware_route_native_sumo' evaluate_ambulance_system.py
grep -Fq 'traffic_aware_route_max_pressure' evaluate_ambulance_system.py
grep -Fq 'native_sumo_signals' map_agnostic_multiagent_worker.py
grep -Fq 'controller_mode == "max_pressure"' ambulance_multiagent_worker.py
"$PYTHON" -m py_compile \
    evaluate_ambulance_system.py \
    map_agnostic_multiagent_worker.py \
    ambulance_multiagent_worker.py \
    "$ANALYZER_FILE"
"$PYTHON" evaluate_ambulance_system.py --help \
    | grep -F -- '--native-maxpressure-benchmark-only' >/dev/null

mkdir "$RUN_ROOT"
trap 'rc=$?; if test "$rc" -ne 0; then printf "FAILED exit=%s at %s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/status.txt"; fi' EXIT

stage() {
    printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_ROOT/status.txt"
}

stage "PREPARING_EXACT_PAIRED_DEMAND"

# Copy the exact 360 background route schedules used by the old non-ambulance
# campaign. Ambulance-v5 requires an audited passenger vType; after adding it,
# verify every departure/route signature is otherwise byte-semantically equal.
"$PYTHON" - "$OLD_FULL_ROOT/demand" "$DEMAND_DIR" "$INPUT_ROOT" <<'PY' \
    | tee "$RUN_ROOT/demand_preparation.log"
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from fixed_demand import (
    count_scheduled_vehicles,
    enforce_fixed_demand_vehicle_type,
    fixed_demand_vehicle_type_is_safe,
    sha256_file,
)
from train_map_agnostic_multimap import passenger_lane_km

old_root = Path(sys.argv[1]).resolve()
new_root = Path(sys.argv[2]).resolve()
input_root = Path(sys.argv[3]).resolve()
assert old_root.is_dir(), old_root

maps = {
    "fremont": input_root / "external_zero_shot" / "fremont.net.xml",
    "santa_clara": input_root / "external_zero_shot" / "santa_clara.net.xml",
    "fresno": input_root / "generated_test_maps" / "fresno" / "fresno.net.xml",
    "san_diego": input_root / "generated_test_maps" / "san_diego" / "san_diego.net.xml",
}
rates = (6.0, 12.0, 18.0)
seeds = tuple(range(1001, 1031))


def route_signature(path: Path):
    root = ET.parse(path).getroot()
    result = []
    for element in root:
        if element.tag == "flow":
            raise AssertionError(f"Flow found in fixed demand: {path}")
        if element.tag not in {"vehicle", "trip"}:
            continue
        attributes = tuple(sorted((key, value) for key, value in element.attrib.items() if key != "type"))
        route = element.find("route")
        edges = "" if route is None else route.get("edges", "")
        result.append((element.tag, attributes, edges))
    return tuple(result)


records = []
for map_id, net_file in maps.items():
    assert net_file.is_file(), net_file
    network_hash = sha256_file(net_file)
    lane_km = passenger_lane_km(net_file)
    for rate in rates:
        rate_tag = str(rate).replace(".", "p")
        period = max(0.05, 3600.0 / (rate * lane_km))
        for seed in seeds:
            source = old_root / map_id / f"rate_{rate_tag}" / f"seed_{seed}.rou.xml"
            assert source.is_file(), source
            destination = new_root / map_id / f"rate_{rate_tag}" / f"seed_{seed}.rou.xml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = route_signature(source)
            assert before, source
            shutil.copy2(source, destination)
            enforce_fixed_demand_vehicle_type(destination)
            after = route_signature(destination)
            assert after == before, f"v5 vType conversion changed demand semantics: {source}"
            assert fixed_demand_vehicle_type_is_safe(destination), destination
            scheduled = count_scheduled_vehicles(destination)
            assert scheduled == len(before)
            records.append(
                {
                    "map_id": map_id,
                    "seed": seed,
                    "net_file": str(net_file),
                    "route_file": str(destination),
                    "period_seconds": period,
                    "trips_per_lane_km_hour": rate,
                    "scenario": f"{rate:g}",
                    "scheduled_records": scheduled,
                    "network_sha256": network_hash,
                    "route_sha256": sha256_file(destination),
                }
            )

assert len(records) == 360
payload = {
    "schema_version": 2,
    "generator": "exact_copy_of_sumo_eval_v3_full_20260730_plus_audited_v5_vtype",
    "episode_seconds": 1200.0,
    "rates": list(rates),
    "seeds": list(seeds),
    "routes": records,
}
manifest = new_root / "manifest.json"
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("OLD_DEMAND_SEMANTIC_IDENTITY=PASS")
print("DEMAND_FILES=360")
print(f"Wrote {manifest}")
PY

find "$DEMAND_DIR" -type f -exec chmod a-w {} +

"$PYTHON" - "$DEMAND_MANIFEST" "$SMOKE_MANIFEST" "$FRESNO" <<'PY'
import json
import sys
from pathlib import Path

source, destination, fresno = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
routes = [
    record
    for record in payload["routes"]
    if Path(record["net_file"]).resolve() == fresno.resolve()
    and int(record["seed"]) == 1001
    and abs(float(record["trips_per_lane_km_hour"]) - 12.0) < 1e-9
]
assert len(routes) == 1, routes
payload["routes"] = routes
payload["rates"] = [12.0]
payload["seeds"] = [1001]
destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

COMMON_ARGS=(
    --base-model-path "$BASE_STEM"
    --emergency-model-path "$EVAL_STEM"
    --episode-seconds 1200
    --decision-seconds 5
    --worker-start-timeout 900
    --max-vehicle-center 3000
    --ambulance-first-spawn 30
    --ambulance-interval-seconds 165
    --ambulance-spawn-jitter 20
    --max-ambulances 16
    --max-active-ambulances 2
    --planned-active-duration-factor 1.5
    --ambulance-last-spawn-buffer 300
    --ambulance-min-euclidean-distance 1200
    --ambulance-min-route-distance 1500
    --ambulance-min-route-edges 12
    --ambulance-min-route-tls 2
    --ambulance-route-attempts 120
    --reroute-interval 12
    --reroute-jitter 2
    --reroute-min-savings-seconds 8
    --reroute-min-savings-fraction 0.05
    --no-reroute-within-tls 100
    --recovery-seconds 30
    --max-preemption-seconds 45
    --clearance-buffer-seconds 3
    --prepare-eta-seconds 25
    --serve-eta-seconds 12
    --ordinary-delay-budget-percent 5
    --throughput-budget-percent 2
    --no-use-libsumo
    --native-maxpressure-benchmark-only
)

stage "SMOKE_TEST_3_RUNS"
"$PYTHON" -u evaluate_ambulance_system.py \
    --maps "$FRESNO" \
    --demand-bank-manifest "$SMOKE_MANIFEST" \
    --output-json "$SMOKE_JSON" \
    --seeds 1001 \
    --workers 3 \
    --sumo-log-dir "$RUN_ROOT/smoke_sumo_logs" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/smoke.log"

"$PYTHON" - "$SMOKE_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_mode = "paired_immutable_demand_native_maxpressure_learned_exact_sumo"
assert payload["evaluation_mode"] == expected_mode
records = payload["records"]
assert len(records) == 3
assert {record["ablation"] for record in records} == {
    "traffic_aware_route_native_sumo",
    "traffic_aware_route_max_pressure",
    "traffic_aware_route_learned_signals",
}
assert len({record["route_sha256"] for record in records}) == 1
assert len({record["network_sha256"] for record in records}) == 1
assert len({record["schedule_sha256"] for record in records}) == 1
print("THREE_WAY_SMOKE=PASS")
PY

{
    echo "RUN_DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "MAPS=fremont,santa_clara,fresno,san_diego"
    echo "RATES=6,12,18"
    echo "SEEDS=$SEEDS"
    echo "CONTROLLERS=native_sumo,max_pressure,ambulance_v5_final_unpromoted"
    echo "AMBULANCE_CHECKPOINT_KIND=$CHECKPOINT_KIND"
    echo "AMBULANCE_VALIDATION_SELECTED_BEST=none"
    echo "AMBULANCE_ROUND_040_ELIGIBLE=false"
    echo "EPISODE_SECONDS=1200"
    echo "DECISION_SECONDS=5"
    echo "MAX_PARALLEL=8"
    echo "DEMAND_FILES=360"
    echo "EVALUATION_RUNS=1080"
    echo "OLD_DEMAND_ROOT=$OLD_FULL_ROOT/demand"
    echo "INPUT_SHA256SUMS_SHA256=$EXPECTED_INPUT_SUMS_SHA256"
    echo "BASE_MODEL_SHA256=$(sha256sum "${BASE_STEM}.zip" | awk '{print $1}')"
    echo "AMBULANCE_FINAL_MODEL_SHA256=$(sha256sum "$EVAL_MODEL" | awk '{print $1}')"
    echo "AMBULANCE_FINAL_CONTRACT_SHA256=$(sha256sum "$EVAL_CONTRACT" | awk '{print $1}')"
    echo "AMBULANCE_ROUND_040_VALIDATION_SHA256=$(sha256sum "$EVAL_VALIDATION" | awk '{print $1}')"
    echo "BENCHMARK_PATCH_SHA256=$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
    echo "MAXPRESSURE_RULE=historical_normalized_score_rule_plus_shared_v5_exit_space_safety_executor"
    echo "NATIVE_SUMO_RULE=original_net_tlLogic_untouched"
} > "$RUN_ROOT/RUN_CONFIGURATION.txt"

stage "FULL_EVALUATION_0_OF_1080"
"$PYTHON" -u evaluate_ambulance_system.py \
    --maps "$MAPS" \
    --demand-bank-manifest "$DEMAND_MANIFEST" \
    --output-json "$FULL_JSON" \
    --seeds "$SEEDS" \
    --workers 8 \
    --sumo-log-dir "$RUN_ROOT/full_sumo_logs" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/evaluation.log"

stage "STATISTICAL_ANALYSIS"
"$PYTHON" "$ANALYZER_FILE" "$FULL_JSON" --output-dir "$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/analysis.log"

sha256sum \
    "$FULL_JSON" \
    "$RUN_ROOT/statistical_summary.csv" \
    "$RUN_ROOT/robustness_summary.json" \
    "$RUN_ROOT/benchmark_summary.txt" \
    > "$RUN_ROOT/RESULT_SHA256SUMS"

stage "COMPLETE_1080_OF_1080"
echo "FINAL_BENCHMARK_COMPLETE"
echo "Results: $RUN_ROOT"
echo "Summary: $RUN_ROOT/benchmark_summary.txt"
