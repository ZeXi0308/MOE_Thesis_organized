## Integrity audit verdict: WARN

`review_independence=same-family`  
`acceptance_status=provisional`  
`P0=0`  
`P1=3`

The formal numerical verdict `PIVOT_TO_SHADOW_VERIFY` is supported. I found no decision-reversing leakage, fabricated result, self-normalization, dead decisive metric, or artifact mismatch. The WARN is driven by proxy semantics, limited scope, and incomplete independence/chronology assurance—not by a numerical contradiction.

### A. Ground-truth provenance: WARN

The “semantic-safe” label is model-output-derived, not dataset or human ground truth:

- The frozen contract defines safety as no downstream ordered-top-k route change under M1-versus-M2 injection: [config](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/configs/semantic_online_observability_v1.json:85>).
- The intervention compares the two model-generated route traces and sets `route_topk_semantic_safe`: [run_semantic_oracle_shadow_replay_5090.py](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_oracle_shadow_replay_5090.py:812>).
- The formal runner copies that predicate into `semantic_safe`, then pair-labels with endpoint AND: [run_semantic_online_observability_5090.py](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:1491>).
- The refreshed verdict correctly says this is not model-quality ground truth: [SEMANTICFENCE_ONLINE_OBSERVABILITY_VERDICT](</Users/leandrozhao/Desktop/毕设论文资料/idea-stage/SEMANTICFENCE_ONLINE_OBSERVABILITY_VERDICT_20260811_002341.md:46>).

Thus it is an explicitly bounded proxy, not fake GT. WARN rather than FAIL because “semantic-safe/action space” remains broader language than the actually observed route/top-k-stability predicate.

### B. Score normalization: PASS

No prediction-statistic normalization fraud was found.

- Features are normalized using only unique train endpoints: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:842>).
- The score is the raw distance ratio `log((d_unsafe+eps)/(d_safe+eps))`: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:993>).
- Threshold selection is validation-only, zero-unsafe, with raw candidate evaluations preserved: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:1142>).
- Cost reports baseline, gross, overhead, net, and raw fractions without own-output max normalization: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:1103>).

My independent read-only recomputation rebuilt:

- 760 tensor-derived feature vectors; cross-platform max normalized error `2.59e-6`;
- train banks: 296 safe, 84 unsafe, 51 cells containing both labels;
- validation and test scores exactly from the sealed feature ledger;
- all seven validation threshold candidates and the selected threshold `0.060565232014776364`.

### C. Result existence and claim agreement: WARN

All decisive formal artifacts exist, their 22-file SHA-256 closure is exact, current bound source hashes match, and the refreshed claims reproduce the artifact values.

The no-project-module raw verifier independently reports:

- `PIVOT_TO_SHADOW_VERIFY`: [RAW_LEDGER_RECOMPUTE.json](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01_audit/RAW_LEDGER_RECOMPUTE.json:281>)
- 77 matching edges: [receipt](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01_audit/RAW_LEDGER_RECOMPUTE.json:840>)
- 77 safe test edges: [receipt](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01_audit/RAW_LEDGER_RECOMPUTE.json:924>)
- 19 admitted endpoints, four unsafe executed pairs, and `-1.833504...` net fraction: [receipt](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01_audit/RAW_LEDGER_RECOMPUTE.json:32>).

The refreshed verdict and authority docs match those numbers: [verdict](</Users/leandrozhao/Desktop/毕设论文资料/idea-stage/SEMANTICFENCE_ONLINE_OBSERVABILITY_VERDICT_20260811_002341.md:21>), [SemanticFence README](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/README.md:39>), [current authority](</Users/leandrozhao/Desktop/毕设论文资料/docs/current/README.md:25>).

WARN reasons:

- The tracker still marks the fresh-agent audit and authority update `IN_PROGRESS`: [tracker](</Users/leandrozhao/Desktop/毕设论文资料/refine-logs/EXPERIMENT_TRACKER_SFV2_O1_20260811_000054.md:23>).
- The dated pre-gate composition verdict still says fresh action/certificate evidence is unverified: [composition verdict](</Users/leandrozhao/Desktop/毕设论文资料/idea-stage/SEMANTICFENCE_COMPOSITION_VERDICT_20260810_204445.md:58>). Current authority supersedes it, but the file is not internally marked historical.

### D. Dead-code detection: PASS

All verdict-bearing functions are reached by the formal pipeline:

- intervention and pair-AND labeling: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:1491>);
- witness construction, validation scoring, and threshold selection: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:2127>);
- frozen admission plan before test execution: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:2197>);
- Oracle, certificate, cost, and verdict: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:2258>);
- `COMPLETE.json` written after summary: [runner](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/run_semantic_online_observability_5090.py:2334>).

### E. Scope assessment: WARN

Actual scope is:

- one pinned OLMoE snapshot, BF16, single RTX 5090 expert-stage stack: [pilot config](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json:4>);
- 12 WikiText documents split 6/2/4, two 16-token windows each, capped at 32 edges/document: [online config](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/configs/semantic_online_observability_v1.json:36>);
- one formal run, four test documents, 128 test edges, no model/hardware/seed replication.

Current claims are mostly calibrated to that scope, especially the explicit exclusions in the refreshed verdict at line 46. The evidence is insufficient for cross-model robustness, production safety, serving performance, or paper-level generality.

### F. Evaluation type

`self_supervised_proxy`, more specifically a model-output-derived counterfactual intervention proxy.

It is not `real_gt`, `human_eval`, or `simulation_only`. “Synthetic proxy” is a reasonable secondary tag because both reference surfaces are generated by the model, but the primary task is self-consistency under controlled intervention.

## P1 findings

1. **Independent-audit boundary is overstated.** The original “independent” auditor imports the producer runner and calls its scoring, threshold, Oracle, certificate, and verdict functions: [audit](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/audit_semantic_online_observability.py:20>), [audit recomputation](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/audit_semantic_online_observability.py:502>). The new raw verifier fixes this for outcome labels, matching, cost, and verdict, but trusts frozen score rows rather than rebuilding features/witness scores: [raw verifier](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/verify_semantic_online_observability_raw.py:414>). My read-only upstream recomputation found no mismatch, so this does not reverse the verdict.

2. **Strict chronology is not independently observable.** Hash chains, exclusive writes, code order, and non-later mtimes support the freeze, but decisive artifacts and `COMPLETE.json` share timestamp resolution; the raw receipt records `strict_mtime_order_observable=false`: [receipt](</Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01_audit/RAW_LEDGER_RECOMPUTE.json:202>). This weakens external proof of pre-outcome chronology without providing evidence of leakage.

3. **Claim-surface freshness/terminology.** Current authority is bounded correctly, but the historical composition verdict is stale and several surfaces use “semantic-safe/action space” where “downstream ordered-top-k-stability proxy” is the exact supported term.

## Highest supported claim

On this sealed single-model/single-stack expert-stage replay, four held-out WikiText documents produced 77/128 route/top-k-stable candidate edges. A hindsight maximum matching covered 154/255 endpoints and projects 29.27% additive expert-stage headroom. Frozen witness-v1 failed on unseen documents: 4/5 executed pairs were unsafe under that proxy, coverage was 3.92%, and projected net saving including prototype overhead was −183.35%.

Unsupported extrapolations include semantic/task correctness, model-quality preservation, online M2 authorization, serving throughput/TPOT/P99, full-request or queueing benefit, EP/NCCL/multi-GPU behavior, cross-model robustness, production/formal safety, and paper proof.

The added verifier materially strengthens downstream arithmetic and verdict confidence; it does not change A, E, F, the claim ceiling, or the provisional same-family acceptance.

No files were edited and no scientific experiment was run. No decisive listed artifact remains unread. I did not exhaustively line-read unrelated historical branches inside the large source-bound helper modules (`run_pilot_5090.py`, partner/cross-companion, StableBatch, and SpectatorRoute); I inspected their SFV2-O1-reachable paths, label provenance, and evidence-boundary declarations.
