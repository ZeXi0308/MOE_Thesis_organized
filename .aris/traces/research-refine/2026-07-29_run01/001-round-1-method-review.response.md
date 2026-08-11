# Fresh C10 Mechanism Review

**Review independence:** same-family fresh reviewer  
**Acceptance status:** provisional  
**Evidence state:** exploratory proposal; zero pilot results  
**CALIBRATION: none**  
**OVERALL: 5.20 / 10**  
**VERDICT: RETHINK**

All claims about natural-workload prevalence, false-trigger rate, certificate utility, action reversal, SLO impact, overhead, cross-model replication, or multi-GPU behavior are **UNVERIFIED**. Planned gates and experiments are not counted as evidence.

## Authority and scope check

The proposal respects the current authority boundary at document level:

- It remains `EXPLORATORY / NOT_CURRENT_MAINLINE`.
- It does not revive stopped controllers.
- It acknowledges that the current common phenomenon Gate is the only formal execution line.
- It does not promote RTX 5090 replay into EP/TPOT/P99 evidence.
- It acknowledges that full request-DAG replay is not implemented and that optimized multi-GPU EP evidence is unavailable.

This is a document-level compliance finding, not empirical validation. C10 must not compete with or modify the current common Gate execution line.

## Scorecard

| Dimension | Score | Review |
|---|---:|---|
| 1. Problem Fidelity | **8/10** | The proposal stays focused on scheduler-induced observation bias, fixed-ledger intervention, trigger reversal, and migration harm. Its non-goals are disciplined. The main unresolved issue is whether arrival-cohort popularity is actually the correct estimand for declaring an operational trigger “false.” |
| 2. Method Specificity | **5/10** | **CRITICAL weakness:** the conservation identity is specified, but the method does not precisely define the actuator horizon, detector class, cohort construction, cancellation semantics, feasible-set relaxation, or how a generic change-point detector is verified by a “small LP.” **Fix:** freeze one detector family, one cohort definition, and one slow actuator; define the exact feasible polytope and prove solver soundness/completeness. Make the certificate action-horizon-specific. |
| 3. Contribution Quality | **4/10** | **CRITICAL weakness:** backlog conservation is queue accounting; non-identifiability under unseen future routes is generic censoring; unanimous-decision partial identification is generic robust thresholding; and the full-DAG audit is validation rather than method novelty. **Fix:** either derive a genuinely MoE-specific, materially tighter certificate from top-k/layer/progress structure, or reframe this as a systems audit contribution whose novelty rests on demonstrated cross-model harmful decisions and non-vacuous intervention-time certificates. Both empirical premises are currently **UNVERIFIED**. |
| 4. Frontier Leverage | **6/10** | **IMPORTANT weakness:** the proposal correctly concedes PROBE, Director, Gimbal, and CRAFT, but does not yet prove why its residual method is more than an application of generic performative-observation and censored-demand theory. **Fix:** provide a formal observation/intervention/estimand/action matrix against each neighbor and generic theory. State one theorem or capability that those formulations cannot obtain under the same no-prediction assumptions. |
| 5. Feasibility | **4/10** | **CRITICAL weakness:** worst-case future-route sets may remain too wide until nearly complete cohort sealing; arbitrary frozen detectors may not admit efficient exact LP verification; full-DAG replay is absent; formal EP resources are blocked. **Fix:** before implementation, derive a non-vacuity envelope giving the earliest possible certificate time as a function of observed mass, remaining-route mass, and detector margin. Kill the method if this time misses the frozen actuator deadline. |
| 6. Validation Focus | **6/10** | **IMPORTANT weakness:** the validation sketch is disciplined but too broad for the unresolved conceptual stage. Multiple detectors, schedulers, models, predictive oracles, and full EP should not precede the basic non-vacuity and estimand tests. **Fix:** add a Phase −1 analytic impossibility gate and a Phase 0 trace-only paired replay with one hotspot detector and one slow migration action. Planned validation remains **UNVERIFIED**. |
| 7. Venue Readiness | **3/10** | **CRITICAL weakness:** there are zero pilots, a current abstraction-level novelty objection, no demonstrated certificate utility, no full-DAG implementation, and no formal EP environment. **Fix:** venue readiness requires surviving the novelty audit, the analytic non-vacuity gate, natural cross-model phenomenon measurement, action-aware harm validation, and optimized EP reproduction. Every such result is currently **UNVERIFIED**. |

Weighted composite:

\[
0.15(8)+0.25(5)+0.25(4)+0.15(6)+0.10(4)+0.05(6)+0.05(3)=\mathbf{5.20}.
\]

## Mechanism adjudication

### 1. Backlog conservation

The identity is useful and appears internally coherent if:

\[
Q_e^\pi(t)
\]

means all fixed future expert-\(e\) route events belonging to requests that arrived by \(t\) but have not executed by \(t\), including routes of tokens not yet generated. Under no drops and a fixed complete route ledger, it is a queue-flow conservation identity.

That is an accounting foundation, not yet a method contribution. Its online problem is precisely that \(Q_e^\pi(t)\) contains unknown future routes. The proposal recognizes this correctly.

A formal version still needs to specify:

- Whether `A_e(s,t]` includes requests by arrival time or route events by logical production time.
- How EOS, cancellation, timeout, admission rejection, and truncated generation alter conservation.
- Whether remaining route count is exact or only bounded.
- Whether fixed routes remain valid under batching-dependent numerical behavior.
- Whether the replay intervention preserves per-request RNG and token/output identity.

### 2. Online non-identifiability

The impossibility construction is valid in spirit: if unfinished tokens may legally route to different experts, two futures can match the complete observed history but reverse final popularity.

However, this is generic missing-future-label/censoring non-identifiability. MoE supplies the labels and top-k conservation structure, but the current theorem does not exploit that structure beyond instantiation.

To become method-level novelty, the proposal needs a MoE-specific result such as:

- a tighter feasible region induced by layer progress, per-layer expert eligibility, top-k coupling, or known routing constraints;
- a nontrivial early-identification theorem;
- or an action-specific certificate obtainable substantially earlier than generic cohort sealing.

Without one, reviewers can reasonably call this standard partial identification applied to expert counts.

### 3. Partial-identification certificate and the always-abstain risk

Let \(m\) be observed route mass and \(r\) the maximum remaining mass. Even in a simplified fixed-total relaxation, expert \(e\)'s popularity can range approximately over:

\[
\left[
\frac{O_e}{m+r},
\frac{O_e+r}{m+r}
\right].
\]

For hotspot threshold \(\theta\), a positive certificate requires approximately:

\[
O_e > \theta(m+r),
\]

while a negative certificate requires:

\[
O_e+r < \theta(m+r).
\]

If \(r/m\) is larger than the detector margin, both labels remain feasible. Change-point certification is worse because uncertainty from multiple cohorts is combined.

Whether natural requests reach a useful margin before the migration deadline is **UNVERIFIED**. Using `max_new_tokens` rather than a probabilistic remaining-length model may make \(r\) extremely conservative. Rolling arrival cohorts may also remain unsealed because a single long request holds the watermark.

This should be settled analytically before building the ledger or full-DAG machinery. The proposal currently treats always-abstain as an experimental failure mode; it is first a paper-and-pencil feasibility question.

The feasible-set implementation is also underspecified. Variable EOS length makes the denominator uncertain, yielding a linear-fractional or integer feasibility problem. A generic change-point detector is not automatically reducible to a small LP. A relaxed polytope may give a sound certificate, but it will be even more conservative.

### 4. Is the target estimand operationally meaningful?

Arrival-cohort final popularity is meaningful as an **offline causal audit target**: it asks whether the intrinsic route mix of fixed arrivals changed.

It is not sufficient as an **online placement-action target**.

The system actually experiences executed load:

\[
L^\pi_e([t,t+H]),
\]

and an action should be judged by its horizon-specific utility:

\[
\Delta J(a;S_t,H,T_{\text{apply}},C_{\text{migration}}).
\]

Scheduler-induced backlog can create a real near-term hotspot even when final arrival-cohort popularity is invariant. For a fast actuator, reacting to that load can be correct. For a slow migration that becomes effective only after the backlog drains, the same reaction can be harmful.

Therefore, “policy-induced” does not imply “false,” and “arrival-invariant” does not imply “action-optimal.” Source attribution alone cannot establish a wrong reconfiguration.

The proposal currently treats full-DAG action-sign evaluation as an optional supporting contribution. For the harmful-migration claim, it is logically necessary. The smallest adequate repair is not a general new optimizer: freeze one slow placement/migration action, its apply latency, and its useful horizon, then certify the action sign or regret over feasible futures. If the paper remains only an observation audit, it must drop the claim that the certificate identifies harmful triggers.

### 5. Missing fast/slow actuator distinction

This distinction is central:

- **Fast scheduling/routing/replica-selection actuator:** service-window counts may be exactly the relevant operational signal. Backlog-induced load is not automatically contamination.
- **Slow placement/migration actuator:** a transient service-window hotspot may expire before the action pays back. Here the audit may have value, but only relative to migration latency and amortization horizon.

CohortFence cannot soundly provide one generic verdict for both. The method must freeze the slow-actuator case or define separate certificates. Freezing one slow actuator is the smaller and stronger choice.

## Novelty pressure adjudication

On the proposal's own prior-art account:

- PROBE already owns continuous-batching membership churn, popularity volatility, and lagging historical EPLB.
- Director already owns incoming/pending-request prediction as a response to stale completed-history statistics.
- Gimbal already owns scheduler/placement feedback and policy-dependent closed-loop observations.
- CRAFT and production EPLB already own periodic post-routing statistics and reconfiguration.
- Generic performative prediction already covers policy-dependent data generation.
- Generic censored-demand and partial-identification theory already covers unobserved future outcomes and set-valued decisions.

The remaining bundle is:

1. fixed-arrival/fixed-route scheduler intervention;
2. backlog-boundary accounting;
3. an arrival-cohort feasible set;
4. unanimous-detector certification;
5. full-DAG harmful-action audit.

That bundle is coherent and MoE-relevant, but **as written it is not yet a distinct MoE method contribution**. Items 2–4 are generic accounting plus partial identification; item 5 is required evaluation. The bundle could become a publishable systems methodology only if MoE structure yields an early, useful certificate and the audit exposes harmful decisions that existing longer-window, hysteresis, predictor, or static baselines do not avoid. Every such empirical claim is **UNVERIFIED**.

## Fatal novelty objection

**YES, for the current method thesis.**

A reviewer can state:

> “CohortFence renames flow conservation, right-censoring bounds, and robust threshold unanimity for MoE expert counts. The full-DAG replay is an evaluation harness. The proposal has not identified a new MoE algorithmic primitive, a new nontrivial theorem, or an empirically established systems failure that makes this combination itself a contribution.”

No exact title-level collision is required for this objection. Absence of an identical prior system is weaker than showing contribution beyond generic theory.

This objection is potentially curable, but only through a MoE-specific non-vacuity/tightness result or a strong systems-audit result. The latter is currently **UNVERIFIED**.

## Strongest counterexample / impossibility concern

Consider identical arrivals and fixed final routes under two schedulers. One scheduler drains an expert-\(e\)-heavy unfinished subset immediately; the other delays it. The first service window correctly observes \(e\) as hot.

- A fast load-routing or replication action can benefit from reacting immediately.
- A slow migration may complete only after the heavy backlog drains and therefore lose money.
- Final arrival-cohort popularity is identical under both schedulers.

CohortFence can attribute the observation difference to the backlog boundary, but that attribution cannot determine whether the trigger was harmful. Any certificate that labels policy-induced service load “false” without action latency and horizon will fail on one of these actuator regimes.

Further, while enough unfinished route mass remains to concentrate on either \(e_1\) or \(e_2\), every nontrivial hotspot decision may remain feasible. Under the no-prediction/no-assumption rule, no online algorithm can guarantee both early certification and low abstention in this construction.

## Simplification Opportunities

1. **Drop generic change-point support initially.** Certify one frozen expert-hotspot/top-\(M\) threshold whose robust decision can be solved exactly. Change-point support currently adds solver ambiguity without strengthening the core claim.

2. **Freeze one slow actuator.** Use one existing placement/migration action with explicit apply latency, amortization horizon, and migration cost. Do not claim applicability to fast scheduling and slow migration simultaneously.

3. **Remove Phase B veto from the contribution until non-vacuity is established.** First establish an offline action-aware audit. An online guard is unnecessary module expansion unless certificates arrive before the frozen action deadline.

## Modernization Opportunities

**NONE.**

A predictor, RL controller, shadow model, or generic OPE module would add complexity and move into already crowded territory. The needed repair is classical systems timescale and action-utility modeling, not a trendy learned component.

## Drift Warning

The following would drift from the Problem Anchor:

- adding a future-route predictor to narrow intervals;
- turning CohortFence into a placement optimizer;
- introducing randomized scheduler exploration or OPE;
- changing natural workloads, max lengths, cohort widths, or detector thresholds after observing abstention;
- claiming generic performative prediction or censoring theory as the paper's problem;
- treating scheduler-induced execution load as erroneous without a fixed action horizon.

Adding a frozen slow-actuator horizon and action-sign certificate is not drift; it is required to complete the anchor's harmful-reconfiguration claim.

## Required revision of Route A

Route A deserves **one conceptual revision before implementation**, not immediate abandonment.

The revised minimum should be:

1. Define arrival-cohort popularity only as an attribution variable.
2. Freeze one hotspot detector and one slow migration/placement action.
3. Define `T_apply`, evaluation horizon \(H\), migration cost, and full-path utility.
4. Derive an analytic non-vacuity bound from observed mass, remaining mass, detector margin, and action deadline.
5. Certify either the action sign or a sufficient condition for harmful migration, rather than certifying only \(D(p)\).
6. Prove the exact solver for the chosen detector; do not claim generic detector support.
7. Run a tiny fixed-ledger replay only if the analytic bound permits a timely certificate.

If the analytic envelope shows that identification generally occurs after the frozen migration deadline, **kill C10 and pivot**. Do not rescue it by changing cohorts, workloads, thresholds, or adding a predictor. If the envelope is non-vacuous but the method remains generic, retain it only as an audit protocol or supporting analysis rather than the dominant CCF-B contribution.

## GAP

The decisive gap is not implementation polish or missing experimental scale. The proposal currently lacks an action-grounded estimand, an early-certification non-vacuity proof, and a contribution beyond generic queue accounting plus censored-outcome partial identification. Full-DAG and multi-GPU work would be premature until those conceptual gates pass. The natural cross-model phenomenon, harmful decision rate, certificate retention/lag, action reversal, and systems impact are all **UNVERIFIED**, while the current README explicitly records that the required full request-DAG and formal optimized-EP evidence do not exist.

## Final verdict

**RETHINK.**

The problem is focused and worth a bounded audit, but CohortFence is not yet a CCF-B-grade method. Route A can be revised once around a single slow actuator and an analytic non-vacuity gate. Failure of that gate, or failure to establish a MoE-specific contribution beyond a renamed accounting/watermark rule, should kill C10 rather than trigger mechanism expansion.
