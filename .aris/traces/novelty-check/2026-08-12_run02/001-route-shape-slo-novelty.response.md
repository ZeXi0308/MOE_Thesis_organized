# Independent novelty verdict

`PARTIALLY_NOVEL`

No checked paper contains the complete chain `historical completed-window route
shape -> next-window SLO-feasible max-running capacity -> causal
admission/max-running action` under fixed placement, replica assignment, and
ordinary workload-state controls. The broad mechanism is not independently
novel: route distribution affects execution cost; historical/current expert
pressure predicts future behavior; SLO systems change running/token budgets;
and Gimbal feeds expert pressure into frontend scheduling.

RouteShape-SLO is therefore presently a narrow measurement/predictor hypothesis
inside DEPA, not an independent paper line. Standalone status requires P1--P3
to show stable cross-model residual value and an action distinct from
workload-only admission.

## Strongest collisions

1. [Gimbal](https://arxiv.org/abs/2606.15177) combines KV, token/queue state,
   and MoE pressure to dispatch incoming requests and manage placement. It is
   the closest systems collision, though it does not forecast a local
   next-window max-running cap.
2. [SCORPIO](https://arxiv.org/abs/2505.23022) estimates ITL/TPOT from running
   batch size, average sequence length, and predicted output length, then
   admits/rejects and changes batch composition. This is the exact action-space
   collision.
3. [SLOs-Serve](https://arxiv.org/abs/2504.08784) performs SLO-constrained token
   allocation and admission. Route-conditioned capacity alone is not a new
   controller.
4. [ELDR](https://arxiv.org/abs/2607.00466) uses prefill expert signatures to
   predict future decode locality and route requests to workers, occupying the
   claim that historical route is deployable predictive telemetry.
5. [RaMP](https://arxiv.org/abs/2604.26039) and
   [DA-MoE](https://arxiv.org/abs/2607.23099) use live route histograms to
   predict kernel cost/configuration, disqualifying a claim that route-dependent
   performance information is itself new.

The defensible narrow claim is to test, in continuous-batching MoE decode with
fixed placement/assignment/kernels/model semantics, whether permutation-
invariant route statistics from completed window `t` add stable cross-model
information about the service-time tail at `t+1` after controlling active
tokens, running sequences, queue, and KV length; if so, quantify improvement of
a frozen max-running policy over workload-only SLO admission.

Disqualifiers include: Gimbal already gating local concurrency with predicted
MoE pressure; a route-aware SCORPIO/SLOs-Serve variant; gains disappearing after
ordinary workload controls; only future/current route working; the effect being
entirely due to kernel dispatch, placement, replication, or migration; or the
action reducing to admission/batch composition with a better estimator.
