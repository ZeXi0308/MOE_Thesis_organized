# RouteShape-SLO action-level related-work check

**Search date:** 2026-08-12
**Boundary:** primary-source pages inspected in a bounded 2024--2026 search.
Absence of a direct match is `PROVISIONAL`, not proof that no such work exists.

## Result

No direct match was found for the complete chain:

```text
historical route-shape telemetry
  -> incremental next-window safe-capacity prediction after ordinary counters
  -> service-level next-window active-token-budget admission
```

This does not make the proposal novel yet. Dynamic token/admission actions are
known, and route/load telemetry is already widely used for MoE placement,
replication, prefetch, redistribution and batching. The only potentially
defensible delta is an unverified empirical bridge between those two bodies.

## Action matrix

| Work | Predicted/observed object | Action | Collision status |
|---|---|---|---|
| [Gimbal](https://arxiv.org/abs/2606.15177) | KV/token/queue state plus MoE expert pressure | frontend dispatch/admission and placement management | `VERIFIED`: closest end-to-end systems collision; not the exact local next-window capacity target. |
| [AugServe](https://arxiv.org/html/2512.04013v3) | reclaimable KV memory and output/state estimates | dynamic token budget and request order | `VERIFIED`: strongest action collision; no route input. |
| [SCORPIO](https://arxiv.org/abs/2505.23022) | output length and batch/sequence state to latency | reject/admit, queue and batch selection | `VERIFIED`: admission baseline. |
| [TAPER](https://arxiv.org/abs/2605.06914) | branch externality, context and slack | per-step reasoning-branch admission | `VERIFIED`: similar control form, different object. |
| [SLOs-Serve](https://arxiv.org/abs/2504.08784) | GPU profile, token allocation and SLO | soft admission, batch size, chunked prefill | `VERIFIED`: route-free SLO capacity control. |
| [Chiron](https://arxiv.org/abs/2501.08090) | queue, utilization and SLO state | backpressure, scaling and batch size | `VERIFIED`: queue/runtime baseline. |
| [SLIM](https://arxiv.org/abs/2607.29575) | model/hardware execution model | batching configuration under latency target | `VERIFIED`: generic capacity-prediction neighbor. |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | prefill/decode token work | chunked prefill and stall-free batching | `VERIFIED`: continuous-batching neighbor. |
| [MoE-Infinity](https://arxiv.org/html/2401.14361v3) | historical/current activation map to future expert activation | cache and prefetch | `VERIFIED`: closest historical-route signal, different target/action. |
| [Scaling Multi-Node MoE Inference Using Expert Activation Patterns](https://arxiv.org/abs/2604.23150) | persistent expert-load structure and prefill/decode activation relation | micro-batch grouping and placement | `VERIFIED`: historical activation is already actionable telemetry; different action. |
| [ELDR](https://arxiv.org/abs/2607.00466) | prefill expert signature to future decode activation | request routing among decode workers | `VERIFIED`: deployable historical-route prediction; not within-worker capacity. |
| [Semantic Parallelism / Sem-MoE](https://arxiv.org/html/2503.04398v5) | token identity and inter-layer affinity to route path | placement, rebatching and token reshuffle | `VERIFIED`: route prediction, not admission. |
| [Mixture-of-Experts Serving](https://arxiv.org/html/2607.17880v1) | current routed expert workload | GPU allocation per expert | `VERIFIED`: route load to resource assignment; closer to BCRD/DEPA. |
| [PROBE](https://arxiv.org/abs/2602.00509) | next-layer activation | replication, token assignment and prefetch | `VERIFIED`: replication collision. |
| [Predictive Prefetching and Expert Replication](https://arxiv.org/abs/2605.11537) | upcoming overloaded experts | replication and prefetch | `VERIFIED`: hotspot prediction, not safe capacity. |
| [Toward Efficient MoE Inference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html) | runtime activation, load correlation and locality | token redistribution, placement and buffering | `VERIFIED`: route-statistic overlap. |
| [HarMoEny](https://www.microsoft.com/en-us/research/publication/harmoeny-efficient-multi-gpu-inference-of-moe-models/) | exact post-route expert load | redistribution and async prefetch | `VERIFIED`: same signal family, different action. |
| [MoE-Gen](https://arxiv.org/abs/2503.09716) | module/expert token accumulation | module batching | `VERIFIED`: BCRD/batching neighbor. |
| [BrownoutServe](https://arxiv.org/html/2507.17133v1) | recent P90 TTFT/TPOT | approximate expert-execution brownout | `VERIFIED`: MoE SLO controller with a different action. |
| [MoE-GPS](https://arxiv.org/abs/2506.07366) | coarse token distribution | expert duplication | `VERIFIED`: distribution-only route prediction and overhead tradeoff; different action. |

## Six required answers

1. Direct route telemetry to serving capacity: no complete direct match found;
   `PROVISIONAL`.
2. Existing work predicts route/activation, expert load, latency/throughput, or
   memory headroom. Route shape to safe running-set capacity remains
   unverified.
3. Existing actions cover placement, replication, prefetch/cache,
   redistribution, batching and admission; neither side of the proposed bridge
   is novel alone.
4. Without matched residual value, Oracle action headroom and a causal policy,
   the Idea is only adding route features to a known predictor.
5. If route only supplies BCRD service estimates, fold it into BCRD.
6. Strong route-free baselines exist. A future P1 must additionally compare
   running-sequences-only, global queue plus tokens, per-expert queue summaries,
   and those counters plus route shape.

## Provisional novelty position

Do not claim “first route-aware capacity predictor.” A defensible contribution
would require a new finding: route shape is a low-overhead incremental
execution-shape statistic that changes a causal next-window admission decision
after strong queue/token/KV/per-expert-queue baselines. That finding has not
been measured.

The authoritative review of the selected active-token action scored the overall
direction **5/10** and recommended `PROCEED WITH CAUTION`. An earlier review of
the unselected max-running fallback is retained only as audit history; it is not
a second action in this protocol. The key objection is compositional: a reviewer
can describe the proposal as SCORPIO/SLIM/TAPER-style predictive SLO admission
plus MoE-GPS/MoE-Infinity/ELDR-style route telemetry. This is a provisional jury
judgment, not an external-family novelty ruling. With the frozen active-token-
budget action, the prior is a **DEPA prediction/admission submodule**;
standalone status would require a cross-model empirical residual, Oracle
headroom, causal recovery of that headroom, and a mechanism not representable
as DEPA admission or running-set composition.

## Fresh independent reviewer, claim by claim

The fresh GPT-5.6-Sol reviewer classified the overall direction as **CAUTION**:
run H1/H2 as falsification gates, but do not claim an independent method. This
is same-family/provisional review evidence.

| Frozen claim | Closest collision | Residual | Verdict |
|---|---|---|---|
| C1: route shape adds information after workload controls | ELDR, METRO, activation-pattern scaling, Sem-MoE and Gimbal already show route/activation affects latency and predicts future use | The exact conditional M3-vs-M1 result under matched cells and document/time/arrival holdouts was not found | `CAUTION` |
| C2: use the signal for `active_token_budget[t+1]` | SLOs-Serve, ConServe, SCORPIO and related SLO-serving systems already choose token/admission/capacity bounds; Gimbal uses expert pressure online | Placement/routing/rebatching stay fixed and only one service-level active-token ceiling changes | `CAUTION` |
| C3: matched test -> counterfactual Oracle -> causal controller | Controlled route studies, scheduler ablations and Oracle-to-policy methodology exist | The full endogeneity-safe chain was not found; useful validation protocol, not new theory | `PROCEED_AS_VALIDATION_PROTOCOL` |
| C4: independent RouteShape-SLO system | Semantic, implementation and empirical neighborhoods are crowded | Only a new, material route-conditioned service-capacity axis plus causal gain could support independence | `CAUTION_UNSUPPORTED_NOW` |

Reviewer collision levels: semantic `HIGH`, theory `MEDIUM-HIGH`,
implementation `HIGH`, empirical `MEDIUM-HIGH`. The strongest validity warning
is endogeneity: changing an active-token budget changes the admitted set and
therefore future routes. Reusing one observed future route trace for every
candidate budget would create fake counterfactual ground truth.

At least four query formulations were used per claim, covering expert
activation plus request-count admission, route-aware serving latency, active
expert union, route-aware SLO capacity bounds, route-conditioned capacity
Oracles, and expert-activation admission controllers. The full trace is in
`.aris/traces/novelty-check/2026-08-12_run02/`.
