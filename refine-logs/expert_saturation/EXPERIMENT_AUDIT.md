# Expert Saturation Experiment-Integrity Audit

Date: 2026-08-23  
Reviewer: fresh same-family sub-agent `/root/experiment_integrity_audit`  
Verdict: `FAIL`  
Reason code: `FAIL_OPEN_TELEMETRY_GATES_AND_UNVERIFIABLE_PRODUCER_SOURCE`

This verdict audits the evidence and evaluators as they existed before the
remediation pass. It does not invalidate every observational measurement, but
it blocks every telemetry-qualification, action-headroom, SLO, and controller
claim until the listed fail-open paths are fixed and the GPU Gates are rerun.

## A-F verdicts

| Area | Status | Evidence-bounded conclusion |
|---|---|---|
| A. Ground-truth/reference provenance | `WARN` | The workload is WikiText-derived, while route, token, and timing values are model/runtime outputs. Regrouping is a self-supervised fixed-trace proxy, not ground truth. The five old bundles record producer SHA `fa20398f...`, but that exact producer source is absent. |
| B. Score normalization | `PASS` | Route metrics use physical denominators such as assignment count and expert count; timing percentages use Route-OFF timing. No self-normalization by a model-output maximum/minimum was found. |
| C. Result existence and claim matching | `FAIL` | Existing canonical JSONs reproduce from their sealed artifacts and keep bounded claims, but the seals cannot prove the missing producer source or old vLLM source state. Optimized telemetry and decode-cap GPU action results remain `UNRUN`. |
| D. Dead code / fail-open validation | `FAIL` | Multiple evaluator paths can emit a positive qualification despite token drift, zero route coverage, label-only implementation identity, clean-repeat selection, malformed route artifacts, or large negative timing drift. |
| E. Scope | `WARN` | One model/revision, one RTX 5090 runtime, one correlated WikiText prompt pool, no action-conditioned GPU run, no multi-GPU EP run. The 90 step cells and six trajectories are descriptive, not independent workloads. |
| F. Evaluation classification | `PASS` | Native pivot: non-GT observational system measurement. Regrouping: fixed-trace self-supervised proxy. Valid-window patch: static source review. Decode-cap: prospective native action experiment, still `UNRUN`. |

## Material integrity failures

1. `compare_vllm_telemetry_implementations.py` could return
   `VALID_WINDOW_TELEMETRY_QUALIFIED` with zero comparable route cells because
   `token_drift_keys`, `comparable_cells`, and `exact_route_cells` did not
   participate in the verdict.
2. Stock versus optimized identity was based on editable `runtime_patch_id`
   labels while the exact vLLM source hashes were removed from the cross-runtime
   comparison. The computed `stock_pair` was not part of the verdict.
3. `analyze_vllm_route_probe_bundles.py` could discard token-drift repeats,
   retain a clean subset, and still select `TEST_MARGINAL_PRESSURE_ACTION`.
4. OFF/ON transparency used a one-sided overhead check. Large negative timing
   drift could therefore qualify even though it also demonstrates that the two
   arms are not exchangeable for a pressure/timing join.
5. Route-semantic comparison did not validate row hash, tensor shape, expert
   range, or top-k uniqueness, and downstream analysis trusted precomputed route
   summaries rather than recomputing them from sealed NPZ data.
6. The four-arm implementation comparator accepted one directory per arm even
   though the frozen protocol requires two controlled process repeats.
7. Scientific Gate failures could still exit with status code zero.
8. The five historical native bundles are internally sealed, but the exact
   producer source SHA `fa20398f...` is no longer available. They are therefore
   retained only as `producer-source-unverified` observational evidence.

## Claims that survive

- `WORKING_SET_MEASUREMENT_ONLY` survives as a self-consistent,
  producer-source-unverified observational diagnostic.
- Route-ON/OFF timing does not survive: token drift occurred in `1/36` and
  `6/36` cells, so timing and pressure cannot be joined.
- Fixed-trace regrouping remains structural only. Against the strongest
  route-blind baseline, the prior-route heuristic has no material residual;
  current-route hindsight remains non-actionable.
- The valid-window patch keeps only a static source-review ceiling. GPU and
  asynchronous runtime safety remain unmeasured.
- Decode-cap, SLO-goodput, scheduling headroom, and controller benefit remain
  `UNRUN / NO_METHOD_GO`.

## Required remediation before GPU execution

1. Fail the whole Gate on any required-repeat token drift; remove clean-subset
   action selection.
2. Require two complete controlled repeats, full nonzero route coverage, exact
   route parity, and strict row-hash/shape/range/top-k validation.
3. Bind stock and valid-window arms to validator-approved source hashes and make
   the stock control participate in the verdict.
4. Use a two-sided timing-deviation guard for wall time and P95 TPOT.
5. Recompute decision metrics from sealed NPZ/JSONL inputs and validate every
   referenced artifact.
6. Make every non-qualified scientific Gate return nonzero and add adversarial
   regression tests for every discovered counterexample.
7. Embed the exact producer source in all new bundles. Do not upgrade the old
   bundles by documentation alone; rerun them when reproducible evidence is
   required.

The full reviewer request, response, and metadata are preserved under
`.aris/traces/experiment-audit/2026-08-23_run01/`.
