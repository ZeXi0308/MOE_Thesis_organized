# BCRD three-gate falsification harness

This directory implements the fail-closed code listed in
[`../研究设计与三门验证协议.md`](../研究设计与三门验证协议.md).
It implements the three serial Go/No-Go gates; it does **not** implement or
claim a production BCRD controller.

## Evidence boundary

- `--smoke` artifacts are deterministic fixtures and always end in
  `SMOKE_ONLY`, regardless of their numbers.
- Native route capture proves route identity only.
- Formal Gate 1 requires explicit `bcrd-route-v3` rows with document identity,
  legal replicas, and a measured causal stage ledger. The bundled single-device
  producer emits v3 columns but remains `temporal_ledger_eligible=false`.
- Native expert timing is a single-GPU expert-forward service curve, not a
  grouped EP, NCCL, A2A, TPOT, P99, or 8-GPU result.
- Gate 1 cannot issue `PASS_GATE1` without a measured full-path denominator.
- Formal Gate 1 is currently hard-blocked until the layer-level latency proxy
  is replaced by a consumer for the frozen expert/dtype-complete surface.
- Gate 1 replays one continuous causal prefix per model/phase/layer. It does not
  reset least-load state or re-salt current-hash at census-wave boundaries.
  Exposure CSV coordinates must uniquely and exactly cover the route matrix.
- Seal timeout is a real event: a singleton pays the full hold, same-time
  arrivals precede a zero-hold seal, and launched rows leave the open queue.
- The solver can exactly enumerate a **local single-layer window** only when
  every legal assignment and independent active `(replica, expert)` hold state
  fits the state budget. State-limit overflow returns
  `INVALID_ORACLE_NOT_EXACT`; a heuristic is never relabeled as an Oracle.
- Formal Gate 2 is currently hard-disabled as
  `INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED`: local window delays are not yet
  propagated through later layers and autoregressive decode steps.
- Gate 3 online policies receive a snapshot from the same causal engine used by
  the evaluator. The snapshot omits observed dispatch/expert/combine suffixes
  and does not expose the engine. Remote/controller/seal/launch costs advance
  the timeline; future arrivals are available only to the offline Oracle.
- Calibration/evaluation is frozen once by `document_id` (request fallback),
  and any request/document overlap fails closed.
- Even `A100_CANDIDATE` authorizes only the separately specified 8×A100 Gate.

## Files

| File | Purpose |
|---|---|
| `capture_continuous_decode.py` | unique Gate-0 A manifest-bound natural continuous-decode producer; output remains a candidate until independent audit |
| `capture_native_routes.py` | development native capture or normalization into explicit `bcrd-route-v3` columns |
| `benchmark_expert_service_curve.py` | CUDA native-expert row curve |
| `merge_service_curves.py` | checked merge of per-model/per-layer curves |
| `census_fragmentation.py` | Gate 1 hash/least-load fragmentation census |
| `build_fixed_replica_instances.py` | freeze Gate 2/3 instances and calibration/evaluation split |
| `solve_assignment_oracle.py` | symmetry-reduced exact legal-assignment + per-queue-hold Oracle |
| `core.py` | route-v3 contracts and shared causal seal/launch/finish replay engine |
| `replay_completion_dag.py` | CLI wrapper for one explicit single-layer instance replay |
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

The Gate-0 A producer has a separate frozen experiment card and
preregistration:

- [`../../../current/gate0_continuous_decode_experiment_card_2026-08-02.md`](../../../current/gate0_continuous_decode_experiment_card_2026-08-02.md)
- [`configs/gate0_continuous_decode_v1.json`](configs/gate0_continuous_decode_v1.json)
- [`../../../../EXPERIMENT_AUDIT.md`](../../../../EXPERIMENT_AUDIT.md)
- [`../../../current/gate0_audit_2026-08-02.md`](../../../current/gate0_audit_2026-08-02.md)

Reproduce and byte-check the canonical manifests before commit:

```bash
.venv/bin/python -B docs/ideas/bcrd/experiments/build_continuous_workloads.py \
  --dataset-arrow <wikitext-test.arrow-at-b08601e...> \
  --arrival-csv <BurstGPT_1.csv-at-d895a53...> \
  --check
```

The builder rejects either source unless its full SHA-256 matches the frozen
preregistration source. It never reads route outputs.

Its current preregistration and two immutable workload manifests freeze the
inputs and authorize only the declared cells. Execution still fails closed
until these exact reviewed bytes are committed in a clean checkout with one
visible RTX 5090 and the frozen dependency versions. The command shape is:

```bash
python3 docs/ideas/bcrd/experiments/capture_continuous_decode.py \
  --workload-manifest docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json \
  --preregistration docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json \
  --output-dir artifacts/bcrd_gate0/formal/<run-id>/olmoe \
  --offline
```

Run the LLM-jp cell into a separate new directory. A complete producer bundle
requires `RUN_STATUS.json=COMPLETE` and `CAPTURE_COMPLETE.json`; it remains
`scientific_result_eligible=false`, `gate0_complete=false`, and
`gate1_authorized=false` pending independent audit and the other Gate-0 items.

Formal mode also requires the workload path and full file SHA-256 to match the
canonical committed per-model entry in the preregistration. The one-way chain
avoids self-referential hashes: clean `HEAD` binds the preregistration, which
binds each manifest; the observed executing commit is written to the output.
A caller-chosen manifest cannot self-authorize a formal cell.

## Gate 1: native routes and exposed fragmentation

The command below is only a single-device compatibility/development capture.
It retains the exact model revision and identity, but it cannot become formal
because it does not observe natural continuous arrivals or the full stage
ledger. A formal Gate-1 producer must supply those v3 fields for at least 128
  fresh documents per model/phase and mark independently verified metadata.

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

Only `PASS_GATE1` authorizes instance construction. Missing causal stage
ledger, denominators, fewer than 128 documents, non-conserved identities, a
request/document split leak, or an out-of-range service curve fail closed.

## Gate 2: exact fixed-replica Oracle

Start with the smallest natural windows that contain repeated expert rows.
`tokens-per-instance` controls the exact-state size. Singleton experts with
identical request/timing/source/legal-target state are collapsed only by an
exact exchangeability argument; repeated experts remain explicit. The solver
reports `UNSOLVED_EXACT_STATE_LIMIT` rather than silently approximating.

```bash
python3 docs/ideas/bcrd/experiments/build_fixed_replica_instances.py \
  --trace docs/ideas/bcrd/experiments/results/olmoe_decode_routes.csv \
  --trace docs/ideas/bcrd/experiments/results/llmjp_decode_routes.csv \
  --gate1-summary docs/ideas/bcrd/experiments/results/gate1/gate1_summary.json \
  --replicas 2 --tokens-per-instance 2 --phase decode \
  --gate1-concurrency 16 --gate1-policy current_least_load \
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

Even when all local instances are exact, the current formal output remains
`INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED`. Implement and validate
counterfactual propagation across every downstream layer and decode step before
interpreting completion/SLO gain or enabling Gate 3.

## Gate 3: simple-policy gap

This section is a frozen downstream interface, not a currently executable
formal stage. `compare_policies.py` rejects formal input until Gate 2 can emit a
valid `PASS_GATE2`; smoke remains available for artifact-chain regression.

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
