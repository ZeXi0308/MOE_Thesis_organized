# Round-2 HorizonFence Method Review

**Review independence:** same-family, same reviewer  
**Acceptance status:** provisional  
**Evidence state:** exploratory proposal; zero pilot results  
**CALIBRATION: none**  
**OVERALL: 4.60 / 10**  
**VERDICT: KILL**

All claims about natural trigger disagreement, harmful-migration prevalence, certificate precision/recall/coverage, deadline non-vacuity, hysteresis advantage, full-path benefit, or optimized EP behavior remain **UNVERIFIED**. The proposed gates are not evidence.

## Problem Anchor verification

**VERBATIM CHECK: PASS.**

The complete Problem Anchor in `round-1-refinement.md` is byte-for-byte identical to Round 0.

- Round-0 anchor SHA-256: `5df382fbe57bd7d28d6d8270c841174c544770edc8977743bda435c524c05587`
- Round-1 anchor SHA-256: `5df382fbe57bd7d28d6d8270c841174c544770edc8977743bda435c524c05587`

The text is preserved, but the revised mechanism only partially preserves the must-solve causal bottleneck. HorizonFence's veto rule does not use scheduler-induced backlog attribution; it can reject any expensive migration, including one triggered by a genuine external shift. Fixed-ledger attribution has moved into validation rather than the certificate. This is a mechanism-level fidelity risk despite the verbatim anchor.

## Scorecard

| Dimension | Score | Review |
|---|---:|---|
| 1. Problem Fidelity | **7/10** | The anchor is unchanged, and restricting the method to slow migration corrects the previous fast/slow mismatch. However, the certificate now answers “can this migration repay a cost bound?” rather than “did scheduler-induced exposure create the harmful trigger?” The causal attribution central to the anchor is not needed by the method. |
| 2. Method Specificity | **5/10** | **CRITICAL weakness:** the action, horizon, detector, and route envelope are much more concrete, but the soundness argument has unresolved definitions and false composition steps: admission timing is incomplete, `J` and `C_a` may double-count migration delay, 1-Lipschitzness does not justify event-to-DAG additivity, and the action commit deadline is missing. **Fix:** count every queued/external arrival from \(t\), define a hard arrival contract, separate gross service benefit from migration disruption exactly once, and require a certificate before the existing policy's irreversible commit point. These repairs do not cure novelty. |
| 3. Contribution Quality | **3/10** | **CRITICAL weakness:** `sup feasible benefit < inf unavoidable cost` is standard robust dominance/futility accounting. The MoE part is a cardinality bound on possible expert-route events. Backlog decomposition and fixed-ledger intervention are not used in the certificate. **Fix:** no non-expansive method-level fix exists. A genuinely new MoE structural bound or causal certificate would be a new route, contrary to the frozen “rethink once” rule. |
| 4. Frontier Leverage | **6/10** | **IMPORTANT weakness:** avoiding predictors and RL remains justified, but the proposal has not separated HorizonFence from generic robust optimization, safe action pruning, or worst-case admission-control accounting. **Fix:** write the rule as a generic robust dominance lemma and identify any strictly MoE-specific tightening. Under the current equations, the only tightening is the elementary one-route-per-expert-per-layer count, which is insufficient for a CCF-B method claim. |
| 5. Feasibility | **2/10** | **CRITICAL weakness:** a sound asynchronous-migration cost lower bound may be zero, while a sound full-DAG benefit upper bound may require large downstream amplification and all future arrivals. Correcting both sides creates a structural vacuity fork: \(C_a^-=0\) or \(G_a^+\) becomes too large. Deadline utility is additionally compromised by the audit holdoff. **Fix:** use \(C_a^-=0\) unless a state-specific causal proof establishes otherwise, include worst-case DAG amplification, and evaluate at the true commit deadline. If the inequality then never holds, kill without implementation. |
| 6. Validation Focus | **7/10** | The Phase −1 kill gate, one detector, one action, and explicit stop conditions are appropriately focused. This score reflects protocol focus only; none of the planned results is evidence, and all empirical success conditions are **UNVERIFIED**. |
| 7. Venue Readiness | **2/10** | **CRITICAL weakness:** the dominant contribution still has a fatal generic-accounting objection, the soundness proof is incomplete, non-vacuity is structurally doubtful, full request-DAG replay is unavailable, and formal EP remains blocked. **Fix:** no non-expansive path reaches venue readiness. Retaining this as an internal engineering guard is possible, but it should not remain a CCF-B paper route. |

Weighted composite:

\[
0.15(7)+0.25(5)+0.25(3)+0.15(6)+0.10(2)+0.05(7)+0.05(2)
=\mathbf{4.60}.
\]

## What the revision fixed

The revision correctly resolved several Round-1 objections:

- Arrival-cohort popularity is now attribution rather than the action target.
- Fast and slow actuators are separated.
- The method freezes one detector and one slow action.
- Generic detector and generic LP claims are removed.
- `PROVABLY_FUTILE` is one-sided; `UNRESOLVED` is not mislabeled beneficial.
- Analytic non-vacuity is placed before implementation.
- The proposal explicitly accepts KILL rather than adding a predictor or controller.

These are real conceptual improvements. They do not resolve the core novelty or soundness objections.

## Core novelty adjudication

HorizonFence's central rule is:

\[
\sup_{\omega\in\Omega(t)} G_a(\omega) < \inf_{\omega\in\Omega(t)} C_a(\omega)
\Rightarrow \text{reject action }a.
\]

This is a standard robust dominance certificate. It says that an action is futile if its most favorable legal future cannot cover its minimum unavoidable cost.

The claimed MoE specialization supplies:

- remaining tokens from logical progress;
- at most one route to expert \(e\) per token and layer;
- a physical route-event capacity ceiling.

These define an opportunity-count envelope. They do not change the underlying certificate, and the one-route-per-expert bound is an elementary cardinality fact. The proposal does not derive a nontrivial MoE coupling theorem, exploit router geometry, or produce a tightness result unavailable to generic work-conserving DAG systems.

More importantly, scheduler-induced backlog decomposition is not present in the veto inequality. The same rule applies to a database shard migration, cache relocation, function warmup, or model-replica move after replacing “route event” with “future access.” The fixed-ledger scheduler experiment may establish an interesting failure case, but it is evaluation, not the method.

Therefore HorizonFence remains **worst-case benefit-versus-cost accounting with MoE labels**.

## Future-admission and capacity-envelope audit

The proposed envelope is not sound as written.

### 1. Admissions during holdoff are omitted

The method defines existing work at trigger time \(t\) and new admissions inside:

\[
I_a(d)=[t+d+\ell_a,t+H].
\]

Requests admitted during:

\[
(t,t+d+\ell_a)
\]

can remain queued and execute expert-\(e\) routes after the migration becomes effective. They are neither in the cohort admitted at \(t\) nor in “new admission within \(I_a\).” Excluding them underestimates possible benefit.

A sound envelope must include:

- requests already admitted at \(t\);
- arrivals already waiting but not admitted at \(t\);
- every external arrival from \(t\) through \(t+H\), including arrivals before activation;
- any action-dependent admission difference.

Adding these terms can only enlarge \(G_a^+\), reducing non-vacuity.

### 2. Admission is endogenous

The proposal needs a bound on **external arrivals**, not merely admissions. Migration can change capacity, queue occupancy, and therefore which requests are admitted. A bound learned from historical admissions under the stay-put policy does not cover the action trajectory.

For an “all feasible futures” theorem, `Lambda_max` must be a hard service contract, traffic-shaper limit, or other deterministic bound. A statistical quantile or calibration maximum cannot support a universal certificate. Whether such a hard natural-workload envelope exists is **UNVERIFIED**.

### 3. Capacity differs between counterfactual trajectories

`K_l^max(I_a)` must cover the maximum legal route-event count under both stay and migrate trajectories, including any temporary duplication, copy interference, changed batching, and action-enabled throughput. Using baseline physical capacity can underestimate the action's opportunity set.

### 4. Horizon membership is ambiguous

`J([t,t+H])` must state whether it covers:

- requests arriving within the interval;
- requests completing within the interval;
- work executed within the interval;
- or lateness clipped at \(t+H\).

These are not equivalent. Selecting requests by completion within the horizon introduces a discontinuity when an action moves a request across the boundary. That breaks the simple Lipschitz argument.

## 1-Lipschitz utility audit

The stated inference is invalid:

> “1-Lipschitz \(J\) makes per-event exposed-latency improvement additively upper-bound completion-cost improvement.”

For:

\[
J(c)=\sum_i \max(0,c_i-d_i),
\]

\(J\) is 1-Lipschitz with respect to the completion-time vector under the \(L_1\) norm:

\[
|J(c)-J(c')|\le \|c-c'\|_1.
\]

That does not imply that reducing one route-event service time by \(\delta\) changes the completion vector by \(L_1\) distance at most \(\delta\).

In a queue, one event shortened by \(\delta\) can release a shared resource earlier and advance \(M\) downstream requests by \(\delta\). Then:

\[
\|c-c'\|_1=M\delta,
\]

and total lateness can improve by \(M\delta\), not \(\delta\).

Batching, all-to-all barriers, synchronization, admission release, and downstream decode steps create the same amplification. Conversely, several event savings on one request can overlap on noncritical paths and fail to add. The full request-DAG mapping is the essential object.

A sound bound needs either:

- an exact counterfactual full-DAG replay;
- or a worst-case causal-cone multiplier for every event.

The first is explicitly unimplemented. The second can be as large as the number of downstream requests and makes \(G_a^+\) much looser. A single-GPU service surface cannot establish this full-DAG sensitivity.

## Full-DAG additivity audit

The proposal assumes an additive bound:

\[
G_a^+=\sum_{l,e}R^+_{l,e}\bar{\delta}_{l,e,a}.
\]

This is sound only if `delta_bar` already upper-bounds the **entire downstream completion-cost effect** of one additional affected event across every legal queue state and DAG. But the proposal says it comes from a service surface over batch sizes, queue states, and paths. A service surface can bound local exposed event latency; it does not automatically bound global request-DAG fanout.

If `delta_bar` is redefined to include worst-case global fanout, then it is no longer a local measured service-surface primitive and will probably be extremely large. Its natural finite bound is approximately local saving multiplied by the maximum number of affected descendants. Deadline non-vacuity under that correction is **UNVERIFIED** and structurally doubtful.

The current README separately records that cross-layer/step full request-DAG replay is not implemented and formal Gate 2 is invalid until it exists. HorizonFence cannot use additive notation to bypass that blocker.

## Asynchronous migration cost-lower-bound audit

The universal positive lower bound \(C_a^->0\) is not justified.

A real asynchronous migration can have:

- copy overlapped with otherwise idle DMA/network capacity;
- synchronization during a period with no request requiring expert \(e\);
- a switch barrier hidden under unrelated critical-path work;
- no future route to the migrated expert after the trigger;
- a destination already holding usable state or benefiting from incremental copy.

In these legal futures, incremental completion-lateness cost can be zero even though bytes were copied and wall-clock migration duration was positive.

A one-sided lower confidence bound from calibration trials is distributional evidence about sampled states. It is not a deterministic lower bound for every feasible future. Using it in a universal theorem is unsound.

A positive lower bound requires a state-specific structural witness, such as:

- a mandatory non-overlappable global stop;
- known already-routed work guaranteed to cross the barrier;
- or a separately priced resource cost charged regardless of overlap.

The first may not describe optimized asynchronous migration. The second may rarely exist at trigger time. The third changes the utility from completion lateness to an operator-defined mixed cost.

There is also a double-counting ambiguity. If \(J_a\) is a full-path completion metric that already includes stop/copy/barrier delay, subtracting \(C_a\) again counts migration harm twice. If \(J_a\) excludes migration disruption, it is not the stated full-path action trajectory. The proposal must choose one decomposition.

Under the declared completion-lateness utility, the safe universal default is:

\[
C_a^-=0.
\]

Because \(G_a^+\ge 0\), the strict certificate:

\[
G_a^+<C_a^-
\]

then cannot fire.

## Deadline and holdoff non-vacuity audit

The certificate becomes easier as \(d\) grows because the remaining benefit interval shrinks. This creates a mechanical near-horizon certificate: eventually every action is futile because insufficient time remains.

The current kill rule rejects very late certificates, but two deeper issues remain.

### 1. The certificate must precede the irreversible commit point

If the existing EPLB begins migration immediately at \(t\), waiting until \(t+d\) to decide is not a veto unless the action is deliberately delayed. If copy or synchronization has already started, some cost is sunk.

Therefore the true deadline is not merely \(d_{\max}\). It is the existing policy's frozen commit time:

\[
d_{\text{commit}}.
\]

A useful certificate must be available before that point.

### 2. Holdoff is itself an intervention

If HorizonFence delays every candidate migration while waiting for evidence, it changes the existing policy. For beneficial migrations, the delay loses \(d\) units of useful horizon. That opportunity cost is not covered by a one-sided “veto only” claim.

A non-controller interpretation is valid only if:

- the existing policy already has the same frozen confirmation/holdoff interval;
- or the certificate is computable at \(d=0\).

Whether useful certificates exist at \(d=0\) or within a pre-existing holdoff is **UNVERIFIED**. If the method requires introducing a new wait, it is no longer merely auditing an existing trigger.

## Fatal novelty objection

**YES.**

A reviewer can state:

> “HorizonFence applies the generic rule ‘reject an action when a worst-case benefit bound is below a best-case cost floor.’ Its MoE specialization only counts the maximum number of future expert accesses. The scheduler-induced backlog decomposition is absent from the certificate, and the full-DAG evaluation is an unimplemented validation harness. Moreover, the cost floor may be zero under asynchronous migration, while the benefit bound ignores queueing fanout.”

This is both a novelty objection and a method-soundness objection. Under the proposal's frozen “RETHINK-ONCE” rule, it is terminal.

## Strongest counterexample

At time \(t\), expert \(e\) triggers migration. The system has an asynchronous copy path.

1. During the holdoff and apply interval \((t,t+d+\ell_a)\), a burst of requests is admitted. These requests are not included in `R_old(t)` and are outside the proposal's `R_new(I_a)` admission interval.
2. After activation, their expert-\(e\) routes form a FIFO chain.
3. Migration shortens one critical expert event by \(\delta\), releasing a shared resource earlier and advancing \(M\) downstream request completions by approximately \(\delta\). Completion-lateness benefit is approximately \(M\delta\), not \(\delta\).
4. The copy overlaps idle network time, and the switch occurs before the queued expert events reach the barrier. Incremental completion-lateness migration cost is zero.

The proposal can underestimate gross benefit by omitting the pre-activation admissions and DAG fanout, while assigning a positive empirical \(C_a^-\). It may therefore emit `PROVABLY_FUTILE` even though the action has positive full-path utility.

If corrected soundly, all arrivals are included, fanout amplification enlarges \(G_a^+\), and \(C_a^-\) becomes zero. The certificate then returns `UNRESOLVED`. This is the central soundness-versus-vacuity fork.

## Simplification Opportunities

1. If retained as engineering infrastructure, call it a **migration futility guard**, not a scheduler-induced-demand method. Drop the CCF-B novelty claim and use it only where the platform provides a hard arrival cap and structural non-overlappable cost.

2. Require certification at \(d=0\) or inside an already-existing immutable confirmation interval. Do not introduce a new holdoff.

3. Before any implementation, recompute the analytic gate with:
   - all arrivals from \(t\) to \(t+H\);
   - the maximum capacity across stay and migrate paths;
   - a worst-case full-DAG amplification factor;
   - and \(C_a^-=0\) unless structurally proven positive.

These simplifications are suitable for falsification. They do not rescue the paper contribution.

## Modernization Opportunities

**NONE.**

A predictor, learned utility model, RL controller, or OPE layer would expand the problem and collide with the rejected neighboring routes. The failure is not lack of a modern component.

## Drift Warning

**TEXTUAL ANCHOR: NONE; MECHANISM-LEVEL PARTIAL DRIFT.**

The anchor requires separating external workload shift from scheduler-induced exposure shift. HorizonFence's decision rule does not use that separation; it is a generic migration-cost guard. Restoring causal attribution as another certificate component would expand the method and risk returning to the rejected CohortFence formulation.

The correct response is not to add a second module. It is to stop treating HorizonFence as the dominant C10 contribution.

## Concrete fix assessment

There are local, non-expansive corrections for soundness bookkeeping:

- include arrivals from trigger time rather than activation time;
- use a hard external-arrival cap;
- define the action commit deadline;
- eliminate `J_a`/`C_a` double counting;
- default the universal asynchronous cost floor to zero.

There is **no non-expansive fix** for the combination of:

- generic robust-accounting novelty;
- zero unavoidable cost under legal asynchronous futures;
- full-DAG queueing amplification;
- and deadline-time vacuity.

Using exact full-DAG robust reachability, a new causal sensitivity theorem, or a learned/stochastic future model would be a materially new and larger method. Those are outside the frozen revision allowance.

## GAP

The remaining gap is now sharper than in Round 1: HorizonFence has an operational action target, but its proof depends on two bounds that move in the wrong directions when made sound. Full-DAG queueing and future admissions enlarge the maximum benefit bound, while asynchronous overlap drives the universal cost lower bound toward zero. The resulting certificate is either unsound or vacuous. Independently, the rule remains generic worst-case benefit-versus-cost accounting and no longer uses scheduler-induced attribution in its mechanism. Natural harmful triggers, early coverage, precision, recall, baseline advantage, full-path impact, and optimized EP behavior are all **UNVERIFIED**. The current authority document also forbids treating the missing full request-DAG or single-GPU qualification as formal evidence.

## Route decision

**KILL C10 as a dominant paper route.**

Do not implement HorizonFence, add a predictor, retune its bounds, or create another controller formulation. At most, retain the soundness counterexample and the corrected \(C_a^-=0\) analytic check as a narrow audit note.

Resources should return to the current authoritative common phenomenon Gate. A future paper route may only reopen if independent evidence establishes a real natural scheduler-induced harmful-migration phenomenon and a nonzero structural cost or nontrivial MoE-specific bound without changing the frozen workload or evidence rules.

## Final verdict

**KILL.**

The revision is cleaner and more honest, but it confirms rather than removes the fatal objection. HorizonFence is a generic futility guard whose universal cost floor can vanish and whose local benefit bound does not compose through the request DAG. Under the proposal's own `RETHINK-ONCE` rule, further revision would be mechanism expansion rather than refinement.
