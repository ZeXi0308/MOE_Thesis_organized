# Qwen r04 targeted result check

**Date:** 2026-09-15  
**Verdict:** `PASS / P0=0 / P1=0 / P2=0`  
**Review status:** existing same-family reviewer, non-fresh, provisional.

This check covers only the new r04 static `32/16/16/32` results. It does not reopen earlier implementation or numerical audits. The four raw captures were recomputed directly; the conclusion below does not rely only on the author analyzer.

## Integrity checks

- The parent launcher exited 0, the child exited 0, no owned process remained, and the group status is `COMPLETE`. The readback contains 144 members / 141 payload files; every payload verified, all 38 frozen inputs matched, and the launch sources were stable before and after. See [readback verification](readback_verification_r01.json), [inspection](readback_r01/inspection.json), and [launch](readback_r01/launch/launch.json).
- All four cells completed four requests and 104 generated tokens. Direct reconstruction gives 16 request executions, 416 generated tokens, and 1,552 scheduled positions, exactly 388 per cell. Every ready decode request was scheduled once; there were no preemptions, recomputed tokens, event actions, or prefill-release actions.
- The realized prefill maxima and scheduler thresholds are `32/16/16/32`. All cells use one engine, cap 48, token budget 64, requested KV 512 MiB, 341 block-rounded KV blocks (536,346,624 allocated bytes), BF16, and the same model/runtime allocations. Measurement initial expert caches are identical. The KV free-block queue order is intentionally not rewritten between cells, so this is a same-resource ABBA comparison rather than bit-identical physical KV state.
- The performance pager contains 7,584 measurement layer calls and 3,072 warmup layer calls, totaling 10,656. All performance records have `validation_run=false`; the separate qualification pager has 864 qualification-only records. Qualification ended before handoff and performance start, and both common warmups ended before each measurement origin.
- The runner lifetime is 13,313.962223 s, exactly partitioned into setup/model load 13,155.146901 s, qualification 95.683093 s, handoff 0.234843 s, performance including warmups and I/O 61.199933 s, and shutdown/final bookkeeping 1.697452 s. Parent launch wall is 13,341.039099 s. Per-cell captures of 7.94--9.01 s are not full-lifecycle denominators, and the shared setup/qualification cost cannot be assigned selectively to either arm. See [session lifetime](readback_r01/results/comparison/session_lifetime.json).

## Independent raw comparison

Percentages are `static16 / static32 - 1` within each predeclared ABBA block. Negative time is lower.

| Metric | Block 0: r1 static16 / r0 static32 | Block 1: r2 static16 / r3 static32 |
|---|---:|---:|
| finite-cohort capture | +4.410% | -8.698% |
| mean request completion | +9.062% | -6.432% |
| mean TTFT | +39.545% | +34.436% |
| mean TPOT | +2.143% | -15.087% |
| maximum request ITL | -6.741% | -14.407% |
| engine calls | +1 | +3 |
| measured layer calls | +48 | +144 |
| expert groups | +1.114% | +5.318% |
| expert tensor-copy payload | +4.801% | +3.000% |

The raw values and all request timings agree with [analysis_r01.json](analysis_r01.json). Each arm generated its own future trajectory: only 1/4 request outputs match in each main block, route/group signatures differ, and no request was filtered from the comparison.

## Numerical and claim boundary

[The attribution gate](readback_r01/results/comparison/numerical_qualification/attribution_gate.json) establishes all-48-layer finiteness and bit-exact agreement with the same-partition reference for the captured layer-47 call. It preserves the original result that 47/48 full-reference comparisons passed tolerance. For layer 47, actual versus full remains non-allclose with max-abs 0.25, while actual versus same-partition full is bit-exact; see [layer47 replay](readback_r01/results/comparison/numerical_qualification/layer47_replay/layer47.json). This is an execution-attribution qualification, not task quality or cross-policy semantic equivalence.

The narrow supported conclusion is: on this one true-over-HBM Qwen BF16 model, one fixed-resource single-engine ABBA episode, reducing static prefill 32 to 16 consistently increases arriving-request TTFT and expert transfer/group work while reducing the observed maximum ITL. Full capture, mean completion, TPOT, and throughput change direction across the two blocks, so neither threshold has a stable full-request advantage here.

It does not support model-quality equivalence, production P99/SLO, sustained capacity, statistical significance or non-inferiority, a globally optimal static threshold, or method GO. The two observations per arm are dependent and order-confounded; differing policy outputs and the non-rewritten KV free-block order also keep the result descriptive.

## Severity

| Severity | Count | Finding |
|---|---:|---|
| P0 | 0 | No phantom run, input substitution, missing request, invalid denominator, or qualification leakage found. |
| P1 | 0 | No error that invalidates the bounded descriptive comparison found. |
| P2 | 0 | No additional artifact defect found; the ABBA dependence, output divergence, KV-state boundary, and quality/lifecycle limits are already explicit and must remain in the result claim. |
