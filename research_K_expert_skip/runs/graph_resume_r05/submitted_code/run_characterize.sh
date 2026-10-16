#!/usr/bin/env bash
# Bounded exploratory characterization; run only inside run_locked.py.
set -euo pipefail
PY=/root/miniconda3/bin/python
OUT=${1:?output directory}
MODEL=/root/autodl-tmp/moe-research-20261002/model
DATA=/root/autodl-tmp/research_K_expert_skip/datasets/wikitext2_serving_r01
"$PY" code/bench_decode.py --model "$MODEL" --prompts "$DATA/validation_bench64.jsonl" --output-dir "$OUT/native_routes" --traceonly --batches 1 2 4 8 16 32 64 --prompt-tokens 128 --trace-steps 32 --gpu-memory-utilization .35
"$PY" code/tune_moe.py --model "$MODEL" --traces "$OUT"/native_routes/trace_native_b*/snapshot.json --output-dir "$OUT/native_tuning" --batches 1 2 4 8 16 32 64
export VLLM_TUNED_CONFIG_FOLDER="$PWD/$OUT/native_tuning/tuned_configs"
"$PY" code/bench_graph.py --model "$MODEL" --prompts "$DATA/validation_bench64.jsonl" --output-dir "$OUT/graph_pilot" --arms-file configs/default_arms.json --arms native global_j4 marginal_high --batches 1 8 32 --repeats 1 --newtokens 128 --prompt-tokens 128 --warmup-seconds 1 --no-energy
# Share native/dense references and model loading between Pareto and T1 at B8.
"$PY" - "$OUT/quality_combined_b8_arms.json" <<'PYCONFIG'
import json,sys
from pathlib import Path
pareto=json.loads(Path("configs/quality_pareto.json").read_text())
if isinstance(pareto,dict):pareto=pareto["arms"]
t1=json.loads(Path("configs/drop_rank_gate.json").read_text())
combined=pareto+[a for a in t1["arms"] if a["spec"] and a["spec"]["mode"]=="rank_gate"]
assert len({a["name"] for a in combined})==len(combined)
Path(sys.argv[1]).write_text(json.dumps({"arms":combined,"t1_semantics":t1["semantics"]},indent=2))
PYCONFIG
for BATCH in 8 1 32; do
  ARMS=configs/quality_pareto.json
  if [ "$BATCH" = 8 ]; then ARMS="$OUT/quality_combined_b8_arms.json"; fi
  "$PY" code/quality_decode.py --model "$MODEL" --articles "$DATA/validation_quality32.jsonl" --output "$OUT/quality_b$BATCH.json" --batch "$BATCH" --tokens 32 --prefix-tokens 128 --limit-docs 32 --arms-file "$ARMS" --gpu-memory-utilization .35
done
