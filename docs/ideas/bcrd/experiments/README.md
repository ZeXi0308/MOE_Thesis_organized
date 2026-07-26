# BCRD three-gate falsification harness

This directory implements the minimum code listed in
[`../研究设计与三门验证协议.md`](../研究设计与三门验证协议.md).
It implements the three serial Go/No-Go gates; it does **not** implement or
claim a production BCRD controller.

## Evidence boundary

- `--smoke` artifacts are deterministic fixtures and always end in
  `SMOKE_ONLY`, regardless of their numbers.
- Native route capture proves route identity only.
- Native expert timing is a single-GPU expert-forward service curve, not a
  grouped EP, NCCL, A2A, TPOT, P99, or 8-GPU result.
- Gate 1 cannot issue `PASS_GATE1` without a measured full-path denominator.
- Gate 2 is accepted only when every assignment/hold state was enumerated.
  State-limit overflow returns `INVALID_ORACLE_NOT_EXACT`; a heuristic is never
  relabeled as an Oracle.
- Gate 3 online policies receive the current contribution and prefix state,
  while future arrivals are available only to the offline Oracle.
- Even `A100_CANDIDATE` authorizes only the separately specified 8×A100 Gate.

## Files

| File | Purpose |
|---|---|
| `capture_native_routes.py` | native route capture or normalization into `bcrd-route-v1` |
| `benchmark_expert_service_curve.py` | CUDA native-expert row curve |
| `merge_service_curves.py` | checked merge of per-model/per-layer curves |
| `census_fragmentation.py` | Gate 1 hash/least-load fragmentation census |
| `build_fixed_replica_instances.py` | freeze Gate 2/3 instances and calibration/evaluation split |
| `solve_assignment_oracle.py` | exact bounded assignment+hold Oracle |
| `replay_completion_dag.py` | explicit top-k request fork-join replay |
| `policies.py` | current/random/threshold/greedy/BCRD causal policies |
| `compare_policies.py` | frozen Gate 3 policy replay and calibration |
| `compute_captured_headroom.py` | CI, captured headroom and final decision |
| `run_smoke.py` | complete CPU-compatible code-path check |

## Tests and CPU smoke

From the repository root:

```bash
python3 -m unittest discover -v -s docs/ideas/bcrd/experiments -p 'test_*.py'
python3 docs/ideas/bcrd/experiments/run_smoke.py \
  --output-dir /tmp/bcrd-smoke-20260725
```

The last line must report `SMOKE_ONLY`; a smoke run must never report a
scientific PASS.

## Gate 1: native routes and exposed fragmentation

Capture each frozen model/phase separately. Formal runs should retain the
exact model revision and use at least 128 fresh requests per model/phase.

```bash
python3 docs/ideas/bcrd/experiments/capture_native_routes.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-key olmoe \
  --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --samples 128 --offset 0 --seq-len 128 --phase decode --offline \
  --output docs/ideas/bcrd/experiments/results/olmoe_decode_routes.csv
```

For every layer used by the route trace, measure a curve with the native
expert module. The defaults are the frozen 20 warmups and 200 measurements.

```bash
python3 docs/ideas/bcrd/experiments/benchmark_expert_service_curve.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-key olmoe \
  --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --layer 0 --expert 0 --offline \
  --output docs/ideas/bcrd/experiments/results/olmoe_l0_curve.csv

python3 docs/ideas/bcrd/experiments/merge_service_curves.py \
  --input docs/ideas/bcrd/experiments/results/olmoe_l0_curve.csv \
  --input docs/ideas/bcrd/experiments/results/llmjp_l0_curve.csv \
  --output docs/ideas/bcrd/experiments/results/service_curves.csv
```

Gate 1 requires an independently measured exposure CSV. This prevents expert
work saving from being reported as end-to-end saving. Its schema is:

```text
model,phase,layer,concurrency,total_path_us
olmoe,decode,0,16,1234.5
```

`total_path_us` must use the same wave/accounting boundary as the route census
and include the complete exposed layer/model path. It must not be reconstructed
from the proposed saving.

```bash
python3 docs/ideas/bcrd/experiments/census_fragmentation.py \
  --trace docs/ideas/bcrd/experiments/results/olmoe_decode_routes.csv \
  --trace docs/ideas/bcrd/experiments/results/llmjp_decode_routes.csv \
  --service-curve docs/ideas/bcrd/experiments/results/service_curves.csv \
  --exposure-csv docs/ideas/bcrd/experiments/results/exposed_path.csv \
  --replicas 2 4 8 --concurrency 1 4 16 64 \
  --output-dir docs/ideas/bcrd/experiments/results/gate1
```

Only `PASS_GATE1` authorizes instance construction. Missing denominators,
fewer than 128 requests, non-conserved identities, or an out-of-range service
curve fail closed.

## Gate 2: exact fixed-replica Oracle

Start with the smallest natural windows that contain repeated expert rows.
`tokens-per-instance` controls the exact-state size; the solver reports the
required state count instead of silently approximating.

```bash
python3 docs/ideas/bcrd/experiments/build_fixed_replica_instances.py \
  --trace docs/ideas/bcrd/experiments/results/olmoe_decode_routes.csv \
  --trace docs/ideas/bcrd/experiments/results/llmjp_decode_routes.csv \
  --gate1-summary docs/ideas/bcrd/experiments/results/gate1/gate1_summary.json \
  --replicas 2 --tokens-per-instance 2 \
  --output docs/ideas/bcrd/experiments/results/instances.jsonl

python3 docs/ideas/bcrd/experiments/solve_assignment_oracle.py \
  --instances docs/ideas/bcrd/experiments/results/instances.jsonl \
  --service-curve docs/ideas/bcrd/experiments/results/service_curves.csv \
  --holds-us 0 5 10 20 50 100 \
  --remote-latency-us 0 5 15 --required-remote-latency-us 15 \
  --max-exact-states 2000000 \
  --output-dir docs/ideas/bcrd/experiments/results/gate2
```

If the natural instance needs more exact states than the configured budget,
the correct result is `INVALID_ORACLE_NOT_EXACT`. Increase compute budget or
implement a separately verified exact solver; do not shrink or skew the
workload merely to obtain a positive result.

## Gate 3: simple-policy gap

Use exactly one remote-cost cell matching an exact Gate 2 result. Threshold,
hold, and batching-credit parameters are selected only on the calibration
split; all reported rows come from the frozen evaluation split.

```bash
python3 docs/ideas/bcrd/experiments/compare_policies.py \
  --instances docs/ideas/bcrd/experiments/results/instances.jsonl \
  --service-curve docs/ideas/bcrd/experiments/results/service_curves.csv \
  --gate2-summary docs/ideas/bcrd/experiments/results/gate2/gate2_summary.json \
  --oracle-results docs/ideas/bcrd/experiments/results/gate2/oracle_results.jsonl \
  --remote-latency-us 15 \
  --output-dir docs/ideas/bcrd/experiments/results/gate3

python3 docs/ideas/bcrd/experiments/compute_captured_headroom.py \
  --policy-results docs/ideas/bcrd/experiments/results/gate3/policy_results.jsonl \
  --oracle-results docs/ideas/bcrd/experiments/results/gate2/oracle_results.jsonl \
  --resolved-plan docs/ideas/bcrd/experiments/results/gate3/resolved_plan.json \
  --output-dir docs/ideas/bcrd/experiments/results/gate3
```

The final status is one of `SIMPLE_WINS`,
`KEEP_SIMPLE_CANCEL_CONTROLLER`, `NO_GO_GATE3`, or `A100_CANDIDATE`.
