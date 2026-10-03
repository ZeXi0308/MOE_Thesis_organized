## Same-family review — provisional

Based only on the supplied comparison set, there is no exact direct match to the complete tuple:

> lagged intra-engine route shape → next-window safe active-token capacity → SLO-constrained budget control, with placement/routing/precision frozen.

The nearest conceptual match is **Gimbal**, because it already combines recent expert-token pressure with ordinary serving state for online admission/dispatch. The nearest pieces are **METRO** for route shape affecting latency and **SLOs-Serve/SCORPIO** for capacity/admission control. RouteShape-SLO is therefore a new composition and isolation of known ingredients, not yet a clearly new systems mechanism.

### Novelty ratings

- **(a) Route-shape measurement finding: 2.5/10.**
  Weak alone. METRO and Gimbal already establish that activated-expert/route pressure correlates with serving behavior. Novelty requires showing that *historical* shape has statistically significant fresh-holdout predictive value after strong serving-state and recent-latency controls—not merely rediscovering current-load imbalance.

- **(b) Route-enhanced capacity predictor: 5.5/10.**
  The strongest component. Predicting the next safe active-token ceiling is more specific than predicting routes, latency, pressure, or expert demand. It becomes defensible only if route features materially reduce calibrated capacity error or unsafe over-admission versus a strong non-route predictor.

- **(c) Active-token controller: 4/10.**
  Active-token/concurrency control is conventional. The potentially novel delta is using lagged route shape as an incremental control signal while all route-side actions remain frozen. A predictive model attached to a standard controller is insufficient unless it produces measurable safe headroom.

### Paper or submodule?

Absent strong evidence, this belongs as a **DEPA capacity-estimation/admission submodule**, not a standalone paper. It is poorly aligned with BCRD because BCRD acts after routing on replica assignment, whereas RouteShape-SLO acts before the next service window on admission capacity.

It earns standalone status only if all three links survive:

1. route history adds fresh-holdout signal;
2. a future-route oracle exposes meaningful capacity headroom;
3. a causal historical-route controller captures a substantial fraction of that headroom without increasing SLO violations.

Failure at link 1 kills the idea. Failure at link 2 means there is no useful action opportunity. Failure at link 3 leaves an observational measurement result, not a controller paper.

### Decisive experiment

Use one fixed model, hardware/runtime configuration, placement, precision, request trace, scheduling policy, and window definition. The **only intervention** is next-window active-token budget.

For every state at window \(t\), safely sweep candidate budgets for \(t+1\) through deterministic replay or repeated controlled execution. Record the largest budget satisfying the frozen tail-SLO criterion. This produces an action-conditioned safe-capacity label.

Train on chronological traces and evaluate on untouched future traces, including at least one workload/sequence-length shift:

- Base predictor: active tokens, running sequences, queue depth, KV length, phase, recent latency/throughput, model/runtime identifiers.
- Route-enhanced predictor: exactly the base predictor plus lagged route-shape features.
- Future-route oracle: receives only genuinely future route information available for each candidate action.
- Historical-route controller: chooses the next budget from information available at \(t\).

Necessary baselines:

1. best fixed budget under the same violation allowance;
2. recent-latency/SLO reactive controller;
3. base non-route capacity predictor;
4. Gimbal-inspired current expert-pressure heuristic;
5. simple single-feature route baseline, such as prior max expert tokens;
6. route-feature time-shuffle/placebo;
7. future-route oracle upper bound.

Report:

- SLO-violation rate and confidence interval;
- admitted tokens or throughput at matched violation rate;
- safe-capacity MAE/quantile loss and calibration;
- oracle-regret and fraction of oracle headroom recovered;
- feature ablations and chronological/OOD stability.

The main validity hazard is policy feedback: changing the budget changes which sequences run and therefore changes future routes. Reusing one observed future route trace as the counterfactual label for every candidate budget would be fake ground truth. The experiment must regenerate or faithfully replay routes separately under each candidate action.

### Verdict

**Overall novelty: 5/10 — PROCEED WITH CAUTION.**

Run the oracle/actionability test before building a controller. The cleanest success claim is not “route shape predicts latency,” but:

> At the same SLO-violation rate, lagged route shape enables materially higher active-token capacity than an otherwise identical, strongly tuned non-route controller on untouched traces.

Without that result, fold it into DEPA as a feature ablation or report it as a negative finding.
