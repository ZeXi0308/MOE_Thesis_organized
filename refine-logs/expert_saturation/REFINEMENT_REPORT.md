# Refinement Report

## Original formulation

The initial direction combined many candidate signals, a generic runtime cost model and several scheduling actions: request admission, active concurrency, token budget and prefill/decode ratio. That scope risked producing a feature-plus-controller collage and invalid fixed-route counterfactuals.

## Final formulation

The proposal now asks one question:

> Does completed expert/rank pressure modify the paired full-request SLO cost of raising one frozen decode-concurrency budget beyond strong route-free controls?

## Material refinements

1. Reduced the action surface to one active decode cap.
2. Defined full common prestate `Z_t` and action-specific `X_t(b)`.
3. Replaced observational latency correlation with paired action-risk difference.
4. Required policy-specific request/KV/route/queue/completion evolution.
5. Froze a treatment-independent request cohort and complete failure accounting.
6. Reduced online MoE input to completed max expert/rank load.
7. Moved active union, load-distribution statistics, A2A and kernel spans to diagnostics.
8. Made the strongest comparator route-free Token/KV/queue/previous-action/recent-latency control.
9. Made the one-threshold correction the final conditional mechanism.
10. Delayed 8×A100 EP until single-GPU action and Controller Gates pass.

## Scope removed

- future-route prediction;
- router/top-k/precision modification;
- placement/replication/migration;
- token reshuffle;
- joint prefill/decode control;
- GBDT/RL/multi-action search;
- broad native operator localization during N0.

## Final recommendation

Proceed with N0 only. Further proposal refinement has diminishing value; the score ceiling is now scientific evidence. A negative N0/I1 outcome remains useful if classified precisely and not generalized beyond its tested runtime/regime.
