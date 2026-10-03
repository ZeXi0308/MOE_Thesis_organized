# Final same-family novelty review

**Verdict: `4.5/10 — PROCEED WITH CAUTION`.** RouteShape-SLO currently looks
more like a combination of known route signals and known SLO control actions
than an independent mechanism. Only the low-cost P0 -> P1 -> P2 falsification
chain is justified; no Method GO is supported.

The action boundary must be corrected first: the current frozen action is the
next-window `max_running_sequences`, not verified active-token admission. The
two actions have different nearest neighbors and must not be mixed.

## Independent contribution versus feature engineering

The current prior is feature engineering / control-loop recombination. No exact
match was found for the complete loop:

```text
historical route shape <= t
  -> next-window quantile safe capacity
  -> one global concurrency cap
```

But every core ingredient has strong precedent: MoE route statistics and
historical expert pressure; route-shape effects on latency under matched load;
safe batch/token/concurrency prediction; SLO admission and token budgets; and
permutation-invariant rank/distribution abstractions. Only a strong,
counter-intuitive system result could lift their combination into an
independent contribution.

## Closest action-level neighbors

| Side | Closest neighbor | Exact delta |
|---|---|---|
| MoE state | [Gimbal](https://arxiv.org/html/2606.15177v1) | Already combines workload/KV/queue with recent expert-token pressure and activation matrices; acts through DP-engine dispatch, SJF, and placement/migration. RouteShape-SLO instead uses identity-free history to emit one global cap. |
| Physical mechanism | [ELDR](https://arxiv.org/html/2607.00466v2) | Shows active-expert union changes latency at fixed batch and uses route signature plus live load for decode-worker selection. RouteShape-SLO changes local concurrency, not worker routing. |
| Frozen `max_running_sequences` action | [SCORPIO](https://arxiv.org/html/2505.23022v1) | Predicts TPOT from batch/sequence state and performs virtual-batch admission. The main proposed increment is historical route shape. |
| If action later changes to active-token budget | [SLOs-Serve](https://arxiv.org/html/2504.08784v1), [ConServe](https://openreview.net/forum?id=eKfWG67mZB) | These already perform iterative token allocation or safe token budgets; the remaining delta would be MoE-conditioned capacity estimation. |
| Route representation | [MoE-GPS](https://arxiv.org/html/2506.07366v1), [Multi-Node MoE Activation Patterns](https://arxiv.org/html/2604.23150v1), [RaMP](https://arxiv.org/abs/2604.26039), [ZipMoE](https://arxiv.org/abs/2601.21198) | Historical distributions, moving averages, max load, route histograms, and identity-agnostic/rank abstractions already exist; CV/HHI/EWMA combinations are weak method novelty. |

The most accurate positioning is Gimbal/ELDR route-pressure state plus
SCORPIO-style safe admission, frozen to a coarser global concurrency knob.

## Claim-level assessment

- Route-shape features such as CV, HHI, max/mean, active experts, and EWMA:
  `LOW` novelty.
- Route shape predicting next-window tail service time after matched workload:
  `MEDIUM`; ELDR weakens phenomenon novelty.
- Route-conditioned P95/P99 safe-capacity predictor: `MEDIUM`.
- Only changing global `max_running_sequences`: `LOW-MEDIUM`.
- Future route used solely as an Oracle: `LOW` novelty but good rigor.
- Permutation-invariant cross-expert generalization: `LOW-MEDIUM`.
- Full route shape beating workload-only and simple pressure while increasing
  SLO capacity under fixed assignment: potentially `HIGH`, but wholly
  unverified.

## Decisive falsification

Freeze `action(t+1) = max_running_sequences`. Hold placement, replication,
assignment, batch delay, precision, kernel, and migration fixed. Use identical
predictor class, quantile loss, safety margin, hysteresis, and dwell time while
comparing workload-only, workload plus simple pressure, full route history,
latency feedback, and a future-route Oracle.

Split complete request/document/arrival episodes; use at least two frozen
models and two arrival regimes. Randomize the cap in a safe range or rerun each
candidate action deterministically. Changing the cap changes the running set
and future route, so reusing one observed future route for all candidates is
fake counterfactual ground truth. Include route-history shuffle and report
pinball loss, dangerous underprediction, calibration, P99 violation,
SLO-goodput, oscillation, and overhead.

Keep the frozen gates: P1 route-over-pressure improvement >=5% in pinball or
>=15% dangerous-underprediction reduction with consistent model directions;
P2 Oracle SLO-goodput >=5% or violation reduction >=20%; P3 recovers >=40% of
Oracle gain, net goodput >=3%, overhead <1%, and simple pressure/feedback does
not recover >=90% of Oracle.

If prediction improves without closed-loop goodput, demote it to a diagnostic.
If full shape does not beat simple max-pressure plus active-expert-union, use
the simple controller. If benefit requires replica fragmentation, fold into
BCRD. If the action is admitted-set composition, fold into DEPA. Call the
target `incremental decision value`, not historical-route causality: route is
endogenous, while the identifiable causal object is the cap intervention.

Default placement today is a DEPA capacity-estimation/admission submodule, a
BCRD measurement submodule only when fragmentation mediates the effect, or a
negative result if the frozen gates fail.
