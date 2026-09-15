# Fresh-agent response

## Overall verdict: PASS_WITH_LIMITATIONS

- `integrity_status=PASS`
- `reason_code=NO_CORE_INTEGRITY_DEFECT_PROXY_SCOPE_ONLY`
- P0: 0
- P1: 0
- The experiment can continue.

The frozen descriptive verdict reproduces exactly; remaining issues are P2 evidence-boundary limitations.

## Core recomputation

- Parsed all 240 raw result rows and independently checked reward identities, distances, assignments, intervention surfaces, repeats, and summary fields: 0 mismatches.
- Recomputed `A_O=-3`, `A_S=3`, opportunity cells `35`, opportunity victims `8`, O positive/tie/negative `13/209/18`, S `10/220/10`, and verdict `WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE`.
- All 8 acceptance and 12 formal manifest entries match current sizes and SHA-256 values.
- Regenerated every balanced-shuffle rank, arm order, and side-call order from frozen seeds: 0 mismatches; each shuffle rank occurs exactly 30 times.
- Current locked observable-selector tests pass 10/10.

## A. Reference provenance: PASS

The all-M1 reference is explicitly a self-supervised model-internal counterfactual rather than external ground truth. Inputs are fixed hash-bound token windows (`run_single_contribution_pilot.py:312-344`), and rewards are direct downstream top-k membership-set distance changes (`run_observable_selector_pilot.py:822-845`).

## B. Score normalization: PASS

Raw totals are accumulated directly and rates divide by the fixed 240 cells (`run_observable_selector_pilot.py:897-966`). Raw totals and signed counts accompany rates (`summary.json:25-35`, `summary.json:116-124`). No prediction-statistic self-normalization was found.

## C. Result existence: PASS

Acceptance is smoke-only and scientifically ineligible. The formal request binds exact acceptance, runner, base runner, config, and lock hashes. The formal manifest binds all ledgers and status. Independent raw-ledger recomputation agrees on every decisive field.

## D. Selection and path integrity: PASS

Observable scanning captures gate weights and expert IDs without computing side-call outcomes (`run_observable_selector_pilot.py:225-287`). Assignment receives only the allowlisted observable view (`run_observable_selector_pilot.py:290-358`). Assignments and the policy lock are written before result extraction (`run_observable_selector_pilot.py:1296-1344`). The frozen classifier is executed when constructing the summary (`run_observable_selector_pilot.py:1352-1373`).

## E. Scope: WARN, P2 only

One OLMoE revision, one RTX 5090 stack, 16 document windows, 240 layer cells, one formal run, and one balanced-shuffle seed. The three repeats are bitwise-stability repeats, not independent scientific seeds. This supports a descriptive pilot verdict, not general statistical inferiority.

## F. Evaluation type: PASS

Primary classification: `causal_model-internal_measurement`. Reference classification: `self_supervised_proxy`. It is real GPU/model execution, not serving, EP, multi-GPU, prevalence, external-correctness, or generalization evidence.

## Claim impacts

- Frozen pilot verdict: supported.
- MaxGate-v1 improves over matched shuffle: contradicted in this run.
- MaxGate is statistically inferior in general: unsupported.
- Serving, dynamic batching, EP, multi-GPU, prevalence, or external-correctness claims: unsupported.

## Action

Continue; no integrity repair is required. Preserve proxy and scope labels. Additional document-level replication is needed only before elevating this descriptive pilot into a general inferential claim.

## Assurance

- `reviewer_model=gpt-5.6-sol`
- `reviewer_reasoning=ultra`
- `review_independence=same-family`
- `acceptance_status=provisional`
