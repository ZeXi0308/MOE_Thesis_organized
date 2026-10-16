#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/bin/python
OUT=${1:?output directory}
MODEL=/root/autodl-tmp/moe-research-20261002/model
DATA=/root/autodl-tmp/research_K_expert_skip/datasets/wikitext2_serving_r01
"$PY" code/bench_decode.py --model "$MODEL" --quantization fp8 --dtype bfloat16 --prompts "$DATA/validation_bench64.jsonl" --output-dir "$OUT/bench" --arms native global_j4 marginal_mid --dense-counter-check --batches 1 8 --newtokens 8 --prompt-tokens 32 --repeats 1 --warmup-seconds 0 --gpu-memory-utilization .35 --no-energy
"$PY" code/quality_decode.py --model "$MODEL" --quantization fp8 --dtype bfloat16 --articles "$DATA/validation_quality32.jsonl" --output "$OUT/quality_b8.json" --batch 8 --tokens 4 --prefix-tokens 32 --limit-docs 8 --arms dense global1 marginal05 --policy-backend triton --gpu-memory-utilization .35
