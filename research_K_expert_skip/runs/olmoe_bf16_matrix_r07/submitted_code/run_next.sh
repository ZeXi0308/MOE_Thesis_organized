#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/bin/python
OUT=${1:?output directory}
EXPERT_SKIP_TEST_CUDA=1 "$PY" -m unittest discover -s code/tests -v
"$PY" code/microbench_policy.py --output "$OUT/policy_microbench.json"
"$PY" code/teacher_force.py
"$PY" code/smoke_model_native_global.py --model /root/autodl-tmp/moe-research-20261002/model --output "$OUT/native_global_smoke.json"
"$PY" code/quality_decode.py --model /root/autodl-tmp/moe-research-20261002/model --articles /root/autodl-tmp/research_B_tiered_combine/runs/quality_articles_r02/validation.jsonl --output "$OUT/quality_b1.json" --batch 1 --tokens 4 --prefix-tokens 32 --limit-docs 1 --arms dense global1 gate05 marginal05 --policy-backend triton
"$PY" code/quality_decode.py --model /root/autodl-tmp/moe-research-20261002/model --articles /root/autodl-tmp/research_B_tiered_combine/runs/quality_articles_r02/validation.jsonl --output "$OUT/quality_b8.json" --batch 8 --tokens 4 --prefix-tokens 32 --limit-docs 8 --arms dense marginal05 --policy-backend triton
"$PY" code/bench_decode.py --model /root/autodl-tmp/moe-research-20261002/model --prompts /root/autodl-tmp/research_B_tiered_combine/runs/quality_articles_r02/validation.jsonl --output-dir "$OUT/bench_smoke" --arms native global_j4 marginal_mid --batches 1 8 --newtokens 8 --prompt-tokens 32 --repeats 1 --warmup-seconds 0 --capture-traces --trace-steps 4
"$PY" code/smoke_graph.py --model /root/autodl-tmp/moe-research-20261002/model --output "$OUT/graph_dense.json" --spec-json '{"mode":"dense","backend":"triton"}'
"$PY" code/smoke_graph.py --model /root/autodl-tmp/moe-research-20261002/model --output "$OUT/graph_marginal.json"
