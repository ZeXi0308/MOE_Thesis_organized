# Experiment Audit Report

- Date: `2026-08-10`
- Auditor: fresh `gpt-5.6-sol` ultra reviewer
- Review independence: `same-family`
- Acceptance status: `provisional`
- Evaluation type: `self_supervised_proxy`

## Overall verdict: PASS

No P0 or P1 issue was found. The reviewer independently reproduced the decisive exact metrics from the raw route ledgers and prediction selections, verified the V2 lock and output manifest, and confirmed that the failed attempt is isolated from the authoritative result.

## Checks

- **A. Proxy provenance — PASS.** Labels are route-level `recovered`, `harmed`, and `utility=recovered-harmed`; final-logit labels are not consumed. The evidence is explicitly bounded as a route-stability proxy.
- **B. Leakage and normalization — PASS.** Each broad LODO fold excludes its held document and fits its own normalizer/models; fresh scores reuse the full-broad transform and models. No fresh outcome enters features or profile fitting, and no decision metric is normalized by model output.
- **C. Existence, arithmetic, and hashes — PASS.** All seven V2-locked files and all eight pre-audit manifest entries match SHA256 and byte size. B=33, distinct cells, one rank per cell, residual closure, both exact random baselines, gains, and the decision all recompute.
- **D. Execution and failed-attempt isolation — PASS.** The reported call path executed and wrote the authoritative run. The failed V1 attempt is marked `FAILED_NO_RESULT`; its numerical model/prediction files are byte-identical to run01, confirming that the serialization-only V2 repair did not change scores or selection.
- **E. Scope wording — PASS.** The config, plan, tracker, summary, and result card consistently say retrospective, post-hoc, CPU-only, and non-confirmatory.
- **F. Evaluation type — `self_supervised_proxy`.** It is not external ground truth, model-quality evaluation, human evaluation, production validation, or a fresh confirmatory run.

## Independent decisive recomputation

| Metric | Broad 16-fold LODO | Fresh transfer |
|---|---:|---:|
| Raw action totals `(recovered, harmed, net)` | `(178, 164, 14)` | `(281, 430, -149)` |
| Global matched random net | `77/320` | `-1639/640` |
| Selected-cell uniform-rank net | `-29/8` | `-3` |
| Cell-selection gain | `-1237/320` | `-281/640` |
| Rank-residual ridge outcome / gain | `(1,4,-3)` / `5/8` | `(8,8,0)` / `3` |
| Profile outcome / gain | `(1,6,-5)` / `-11/8` | `(7,15,-8)` / `-5` |
| Min-harm outcome / harm avoidance | `(1,4,-3)` / `5/8` | `(9,4,5)` / `57/8` |

Fresh harm MSE skill independently recomputes to `0.023257565024297322`. This is a diagnostic positive, but the pre-result rule permits no harm-only third policy. Since primary fresh profile gain is `-5`, the unique frozen decision remains:

`STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING`

## Findings and claim impact

- P0: none.
- P1: none.
- P2: `RUN_STATUS.json` is not included in the pre-audit output manifest. Its status and decision duplicate manifest-covered artifacts, so this cannot change the result.
- Supported: exact retrospective route-proxy decomposition and the frozen stop-branch decision.
- Diagnostic only: the positive fresh harm-head result.
- Unsupported: confirmation of a new policy, online dynamic observability, model quality, serving SLO, scientific generalization, or production proof.
