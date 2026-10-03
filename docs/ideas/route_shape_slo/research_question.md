# RouteShape-SLO frozen research question

## Question

After matching the ordinary serving state (active work, scheduled batch,
running sequences, queue depth, KV length, prompt/decode stage, model,
hardware, and runtime), does route shape observed by the end of window `t`
improve the safe-capacity estimate for window `t+1`? If it does, is the
remaining action-space headroom large enough to justify a service-level
controller instead of a workload-only controller or a BCRD/DEPA submodule?

The first and only capacity action is the prompt default:

```text
active_token_budget[t+1]
```

The current continuous-decode harness does not expose a verified active-token
admission interface. That is an explicit P2 implementation blocker, not a
reason to silently substitute a max-running fallback. The existing smoke
therefore exercises no action, and no second action is explored in parallel.
The first experiment does not also change batch delay, replica assignment,
precision, placement, migration, or expert dispatch.

## Frozen hypothesis chain

- **H1 -- route-induced capacity uncertainty:** after matching active tokens,
  batch/running/queue state, prompt/decode stage, KV and sequence lengths,
  model, hardware, and runtime configuration, route shape still explains
  next-window service time, throughput, violation risk, or safe capacity.
- **H2 -- incremental predictive value:** adding causal route features improves
  fresh-holdout tail prediction or dangerous-underprediction rate over the
  identical workload-only predictor.
- **H3 -- action-space headroom:** an action-conditioned future-route Oracle
  materially beats the strongest token-only controller under the same SLO or
  throughput constraint.
- **H4 -- causal deployability:** telemetry completed by window `t` improves the
  frozen `active_token_budget[t+1]` decision without future information.
- **H5 -- runtime relevance (cross-cutting validity check):** the signal survives
  a representative runtime and a paired instrumentation-overhead check, rather
  than being produced by the Python expert loop, hooks, synchronization, or
  synthetic arrivals.

The executable order is strictly `H1 -> H2 -> H3 -> H4`; failure at any stage
stops later stages. H5 is checked wherever runtime evidence is admitted and
cannot be used to skip that order.

## Sequential gates

1. **P0 measurement surface:** align route identity, workload state, and
   synchronized service time to the same decode window. Reject representative
   claims if queue semantics, instrumentation overhead, or serving-runtime
   relevance are missing.
2. **P1 incremental signal:** compare a fixed linear/ridge model using workload
   features with the same model plus causal route features. Split complete
   requests/documents/arrival episodes, never adjacent windows at random.
3. **P2 Oracle headroom:** only after P1 is at least weak-positive on eligible
   evidence, replay the frozen active-token-budget grid and compare the
   future-route Oracle with the strongest workload-only policy.
4. **P3 causal controller:** only after P2 passes, use historical route features
   with a safety margin, two-threshold hysteresis, dwell time, and a cold-start
   workload-only fallback.

No later gate may rescue a failed earlier gate.

## Frozen information boundary

At the end of window `t`, M0--M3 may consume only:

```text
workload_state[<=t]
route_shape[<=t]
latency_history[<=t]
```

The target is next-window `step_service_ms`; future route is allowed only in
M4, the diagnostic Oracle. Full future request length, future tokens, future
latency, and future expert load are forbidden.

M4 is not an action Oracle by itself. Changing
`active_token_budget[t+1]` changes which requests execute and therefore
changes `route_shape[t+1]`. P2 must regenerate or faithfully replay routes for
every candidate budget. Reusing one observed future-route trace for all
budgets is fake counterfactual ground truth and invalidates the experiment.

Workload-only features:

```text
active_tokens, running_sequences, queue_depth, mean_kv_length,
max_kv_length, prompt_tokens, decode_tokens, batch_size,
recent_step_ms, recent_tokens_per_second
```

Route features are permutation-invariant:

```text
per-layer max expert tokens, max/mean, coefficient of variation, HHI,
active expert count, top-1 share, cross-layer max/mean pressure,
hotspot persistence, route-shape EWMA, route-shape delta
```

## Frozen exploratory thresholds

- P1: route-augmented P95 pinball loss improves at least 5%, or dangerous
  underprediction falls at least 15%, with directionally consistent results on
  both frozen models. Improvement below 3% or sign reversal stops the route
  controller. One passing model with the second directionally consistent is
  `WEAK_SIGNAL_NEEDS_MORE_EVENTS`.
- P2: future-route Oracle improves SLO-goodput at least 5% at the same SLO, or
  cuts violation rate at least 20% at the same throughput.
- P3: recover at least 40% of Oracle gain; improve net goodput at least 3% over
  workload-only; overhead below 1%; two arrival regimes agree; simple feedback
  must not already recover 90% or more of Oracle gain.

Thresholds are frozen before a fresh holdout is opened.

## Frozen model revisions and workloads

Scientific P1 requires directionally consistent results on both frozen models:

```text
allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5
llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055
```

The holdout must cover steady and bursty arrival episodes, short/long decode,
mixed KV lengths, and matched active-token cells. Natural workloads carry the
main result; synthetic skew is mechanism sanity only. A control-window choice
from 20 ms, 50 ms, 100 ms, or a fixed decode-step window must be calibrated and
then frozen before the test split is opened.

## Frozen confound controls

The next runtime trace must also record physical KV length, left-padding ratio,
effective batch shape, attention/backend/kernel identity, dtype, placement and
replica set. Existing single-target evidence shows that padded physical KV
shape can change router outputs even when logical target state is held fixed;
without these fields, route shape may merely proxy an omitted execution-shape
variable.

## Formulation-collision rule

- If route value disappears after controlling queue/token/KV state, stop and
  use a token-only controller.
- If value exists only through physical-replica fragmentation, fold the
  measurement into BCRD.
- If value changes service-level capacity independently of assignment but its
  action is admission/running-set composition, compare directly with DEPA and
  fold into DEPA unless P2/P3 establish a distinct deployable mechanism.
- If only future route helps, retain it as an Oracle/diagnostic, not an online
  controller.

## Hardware boundary

A single RTX 5090 may establish route-conditioned local expert execution shape,
single-device capacity prediction, and controller overhead. It cannot establish
EP all-to-all, receiver congestion, remote-byte pressure, NCCL/RDMA critical
paths, multi-GPU fork-join tails, or 8xA100 serving gains. No 8xA100 experiment
is authorized before P1, P2, and P3 pass in order.
