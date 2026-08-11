# Experiment Audit Report

**Date**: 2026-08-11  
**Auditor**: GPT-5.6-Sol ultra, fresh same-family agent, read-only and provisional  
**Project**: SemanticFence SFV2-O1  
**Overall verdict**: `WARN`  
**Integrity status**: `warn`  
**Severity**: `P0=0 / P1=3`

The formal numerical verdict `PIVOT_TO_SHADOW_VERIFY` is supported. No decision-reversing leakage, fabricated result, self-normalization, dead decisive path or artifact mismatch was found. WARN limits the evidence boundary; it does not change the frozen verdict.

## A. Ground-truth provenance: WARN

`semantic_safe` is a model-output-derived counterfactual intervention proxy: it means no downstream ordered top-k route change between an independently injected M1 contribution and the paired-M2 contribution. It is not dataset, human, task-semantic or model-quality ground truth. The runner derives endpoint safety from `route_topk_changed` and pair safety by endpoint AND (`run_semantic_online_observability_5090.py:1491`); current authority now states the proxy boundary explicitly.

Classification: `self_supervised_proxy`, with `synthetic_proxy` as a reasonable secondary tag because both compared surfaces are model-generated.

## B. Score normalization: PASS

Features use train-only unique-endpoint normalization (`run_semantic_online_observability_5090.py:842`). The fixed score is the raw distance ratio `log((d_unsafe+eps)/(d_safe+eps))` (`:993`). Threshold selection is validation-only under a zero-unsafe constraint, and raw candidate evaluations are preserved (`:1142`). Cost records all-M1, gross, measured overhead and net values without self-normalizing by prediction maxima (`:1103`).

The reviewer provisionally rebuilt 760 tensor-derived feature vectors, the 296-safe/84-unsafe train banks, validation/test scores and all seven validation threshold candidates; it found no decision mismatch. This feature-level reconstruction is reviewer evidence, not a cross-family or standalone deterministic receipt.

## C. Result existence and fidelity: WARN

All 22 decisive formal artifacts exist and match `COMPLETE.json`. The no-project-module raw verifier independently rederives 77/128 safe test edges, 77 matching pairs, 154/255 covered rows, 19 admitted endpoints, 4 unsafe executed pairs, -183.3504% net projection and `PIVOT_TO_SHADOW_VERIFY`; `ORACLE_MATCHING.json`, `CERTIFICATE_RESULTS.json`, `COST_PROJECTION.json` and `SUMMARY.json` all compare `MATCH` after recomputation.

WARN remains because the original `audit_semantic_online_observability.py` imports primary runner helpers, strict `mtime(artifact) < mtime(COMPLETE)` is not observable after same-second rsync timestamp preservation, and the pre-gate 2026-08-10 composition verdict remains a historical file. Current README/tracker/verdict explicitly supersede that history without rewriting it.

## D. Dead-code detection: PASS

The formal path reaches intervention and pair-AND labeling (`run_semantic_online_observability_5090.py:1491`), witness construction/validation threshold selection (`:2127`), pre-test admission freezing (`:2197`), Oracle/certificate/cost/verdict (`:2258`) and completion-last write logic (`:2334`). Derived products are present in sealed artifacts. No decisive dead or bypassed function was found.

## E. Scope assessment: WARN

The scope is one pinned OLMoE snapshot, BF16, one RTX 5090 expert-stage stack, one formal run, 12 WikiText documents split 6/2/4, two short capture windows per document and 128 test edges. There is no model/hardware/seed replication. Current authority bounds claims to this proxy pilot and excludes task semantics, cross-model robustness, serving, full-request/queueing benefit, EP/NCCL, multi-GPU, production safety and paper proof.

## F. Evaluation type

Primary: `self_supervised_proxy`. Secondary: `synthetic_proxy`. Not `real_gt`, `human_eval` or `simulation_only`.

## P1 findings and decision impact

1. **Audit independence boundary**: the original full auditor imports the primary runner and calls its scoring/threshold/Oracle/certificate/verdict helpers. The new isolated raw verifier closes outcome-label, matching, cost and mechanical-verdict arithmetic, but consumes frozen score rows instead of independently rebuilding the 64D feature/witness pipeline. Reviewer-side feature reconstruction found no mismatch. **Decision impact: none observed; do not call the full feature pipeline cross-implementation verified.**
2. **Strict chronology is not externally observable from transferred mtimes**: hashes, exclusive-write code order and non-later mtimes support the lock, while `strict_mtime_order_observable=false` is recorded because transferred artifacts share one-second timestamp resolution. **Decision impact: no leakage evidence and no numerical change; claim chronology as supported by chained evidence, not independently timestamp-proved.**
3. **Claim freshness and terminology**: the historical composition verdict predates this Gate, and “semantic-safe/action space” can exceed the exact observed downstream ordered-top-k-stability proxy. **Decision impact: no numerical change; current claim surfaces use the exact proxy term and mark the old verdict historical.**

## Claim impact

- **Supported**: on this sealed single-model/single-stack expert-stage replay, four held-out documents produce 77/128 route/top-k-stable candidate edges; hindsight matching covers 154/255 endpoints and projects 29.27% additive expert-stage headroom; frozen witness-v1 fails with 4/5 unsafe executed pairs, 3.92% coverage and -183.35% net projection.
- **Needs qualifier**: “semantic action space” only when immediately defined as the downstream ordered-top-k-stability proxy.
- **Unsupported**: task/model-quality correctness, online M2 authorization, serving throughput/TPOT/P99, full-request/queueing benefit, EP/NCCL/multi-GPU behavior, cross-model robustness, production/formal safety or paper proof.

## Action items

- Keep `PIVOT_TO_SHADOW_VERIFY`; do not tune witness-v1 on this test split.
- Treat `RAW_LEDGER_RECOMPUTE.json` as the independent arithmetic receipt and the shared-helper auditor as a broader closure check.
- Preserve the strict chronology limitation and proxy terminology in all current claims.
- Run only the next fresh no-actuation shadow-verifier/selective-repair Gate.
