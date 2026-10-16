#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/bin/python
OUT=${1:?output directory}
MODEL=/root/autodl-tmp/moe-research-20261002/model
ARTICLES=/root/autodl-tmp/research_K_expert_skip/datasets/wikitext2_serving_r01/validation_bench64.jsonl
EXPERT_SKIP_TEST_CUDA=1 "$PY" -m unittest discover -s code/tests -v
"$PY" code/bench_decode.py --model "$MODEL" --prompts "$ARTICLES" --output-dir "$OUT/bench_smoke" --arms native global_j4 marginal_mid --batches 1 2 4 8 16 32 64 --newtokens 8 --prompt-tokens 32 --repeats 1 --warmup-seconds 0 --capture-traces --trace-steps 8 --gpu-memory-utilization .35
"$PY" code/smoke_graph.py --model "$MODEL" --output "$OUT/graph_dense.json" --spec-json '{"mode":"dense","backend":"triton"}'
"$PY" code/smoke_graph.py --model "$MODEL" --output "$OUT/graph_marginal.json"
