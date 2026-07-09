from pathlib import Path
from types import SimpleNamespace
import json
import sys

import compare_native_sumo_vs_all_model as base

outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("split_eval_40tls")
out_csv = sys.argv[2] if len(sys.argv) > 2 else "native_vs_model_1500_santaclara_rr_40tls_split_parallel.csv"
out_json = sys.argv[3] if len(sys.argv) > 3 else "native_vs_model_1500_santaclara_rr_40tls_split_parallel.json"

rows = []

for path in sorted(outdir.glob("*_seed*.json")):
    payload = json.loads(path.read_text())
    got = payload.get("runs", [])
    print(f"{path}: {len(got)} rows")
    rows.extend(got)

controllers = sorted(set(row.get("controller") for row in rows))
print("controllers:", controllers)
print("total rows:", len(rows))

base.print_native_vs_model_table(rows)

args = SimpleNamespace(stats_csv=out_csv, stats_json=out_json)
base.write_outputs(rows, args)

print(f"Wrote merged CSV: {out_csv}")
print(f"Wrote merged JSON: {out_json}")
