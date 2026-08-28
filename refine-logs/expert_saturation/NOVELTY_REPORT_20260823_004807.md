# Action-level Novelty Report

> Topic: Expert-pressure-conditioned feasible decode capacity for SLO-constrained MoE serving  
> Search cutoff: 2026-08-23  
> Review status: primary-source bounded search + same-family provisional peer review  
> Claim policy: no `first`, no exhaustive-search claim, no method novelty before native action evidence

## Verdict

- **Broad formulation novelty: LOW.** “Expert pressure / activated experts influence MoE serving scheduling” has direct collisions.
- **Exact formulation novelty: MEDIUM, conditional.** The remaining residual is not a new feature or controller, but a specific causal question:

```text
ordinary route-free counters + completed expert pressure
→ pressure-conditioned marginal risk of a candidate decode budget
→ same-prestate policy-specific execution
→ complete request-level SLO denominator
→ empirically feasible capacity boundary
```

- **Current scientific status: UNVERIFIED.** The repository does not yet contain native serving, action-conditioned capacity, request SLO, EP, or Controller evidence.

## Action-level Matrix

| Work | Signal / state | Prediction target | Action | Objective / regime | Collision | Remaining residual |
|---|---|---|---|---|---|---|
| [Gimbal](https://arxiv.org/abs/2606.15177) | KV, prefill, queue, recent rank/expert-token pressure | engine pressure and dispatch/placement cost | cross-DP request dispatch, SJF+aging, expert placement/migration | TTFT, TPOT, throughput; vLLM DP+EP | **Highest broad collision** | fixed placement/router, within-engine budget treatment effect and full-request action rerun |
| [SCORPIO](https://arxiv.org/abs/2505.23022) | TTFT/TPOT SLO, deadline, output length, batch state | feasibility and ITL/TPOT risk | reject/admit, virtual batch size, credit batching | SLO attainment / goodput; continuous batching | **High action collision** | MoE pressure's incremental effect on the same action beyond strong route-free state |
| [SLOs-Serve](https://arxiv.org/abs/2504.08784) | profiled capacity, token/stage/SLO state | admitted set and token allocation | dynamic batch, chunked prefill, delay/reroute, soft admission | multi-stage SLO capacity | **High action collision** | fixed decode-only knob with endogenous route regeneration |
| [Chiron](https://arxiv.org/abs/2501.08090) | queue, utilization, RWT, SLO, backpressure | replica and local batch demand | scale-out/in and local batch control | cluster SLO and utilization | Medium | per-epoch MoE/EP pressure without scale action |
| [BrownoutServe](https://arxiv.org/abs/2507.17133) | burst/SLO pressure and MoE workload | degradation level | reduce token/expert computation | SLO-quality trade-off | Medium | model semantics remain unchanged; only concurrency/admission changes |
| [Sem-MoE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html) | co-activation and online activation likelihood | token/expert locality | clustering, placement, rebatching, token reshuffle | A2A and throughput; EP | High for “expert-aware batching” | no rebatching/reshuffle/placement; single budget action and SLO denominator |
| [Scaling Multi-Node MoE Using Expert Activation Patterns](https://arxiv.org/abs/2604.23150) | activation traces, domain pattern, imbalance | workload locality | prefill-informed microbatch grouping and placement | multi-node A2A/decode latency | Medium | no grouping/placement; engine-local capacity boundary |
| [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin) | adjacent-layer popularity / selection | next expert resource need | expert resource assignment and A2A coordination | distributed MoE tail latency | Low–Medium | continuous-serving request SLO and fixed placement |
| [ReXpert](https://arxiv.org/abs/2608.13962) | SLO-limited batch, activated-expert union, route skew | FFN latency/energy capacity | ReRAM residency, multicast, coactivation placement | TPOT and energy | High phenomenon collision | standard GPU/EP, request-level active-set action; no hardware change |
| [METRO](https://arxiv.org/abs/2512.09277) | activated experts, MoE decode load | expert-balanced latency | token-to-replica routing | larger batch under TPOT | High phenomenon/action adjacency | fixed logical routing and request admission with policy-specific trajectories |
| [ELDR](https://arxiv.org/abs/2607.00466) | expert signature beyond ordinary worker load | worker-specific decode cost | request-to-worker dispatch | tail-aware load balancing | High signal adjacency | same-engine per-epoch decode budget; no future-route predictor |
| [TAPER](https://arxiv.org/abs/2605.06914) | context/slack/externality at candidate width | candidate action latency risk | per-step admission with fallback | SLO-aware branch scheduling | High control-form collision | endogenous MoE pressure and request active-set state |
| Generic Token/KV controller | candidate token, KV, queue, padding, ages, recent latency | candidate budget feasibility | active decode cap | SLO-goodput | **Exact baseline** | pressure must add held-out paired action-effect information and realized goodput |

## Claim Independence

| Candidate claim | Independence | Decision |
|---|---|---|
| Expert pressure can drive serving scheduling | LOW | Do not claim |
| Activated-expert saturation is a new phenomenon | LOW | Do not claim |
| A quantile predictor, EMA, hysteresis or fallback is novel | LOW | Engineering only |
| Pressure retains stable residual beyond Token/KV/queue/recent latency | MEDIUM | Test on native fresh holdout; no claim yet |
| Pressure changes the marginal request-SLO cost of raising decode budget | MEDIUM | Central I1 hypothesis |
| Same-prestate, policy-specific rerun under endogenous route | High as rigor, not automatically contribution | Contribution only if it reveals a stable system boundary |
| Regime map of when pressure adds or fails to add a control dimension | MEDIUM; potentially stronger with cross-regime/EP confirmation | Preferred thesis narrative |

## Positioning

The closest collision is a collage rather than one identical system:

```text
Gimbal                         → MoE pressure sensing and serving scheduling
SCORPIO / SLOs-Serve / TAPER  → SLO-aware admission or batch-cap action
METRO / ReXpert               → activated-expert saturation phenomenon
ELDR                           → expert state beyond ordinary load counters
```

Therefore the thesis should not lead with “we propose an expert-aware concurrency controller.” It should lead with:

> **When does completed expert pressure add a usable control dimension beyond Token/KV/queue/recent-latency counters?**

The Controller is conditional evidence that an identified boundary is actionable, not the primary novelty.

## Novelty Gate

Before any independent method claim, require all of the following:

1. native continuous-serving execution;
2. same-`Z_t` action branches with independent KV/route/request/completion evolution;
3. a complete request-level denominator;
4. strong route-free baselines including previous budget, total routed tokens and recent latency;
5. stable pressure-conditioned paired action effect on fresh holdout;
6. material action Oracle headroom;
7. positive-regime confirmation or a reproducible negative/boundary map.

Failure of items 5 or 6 yields `MEASUREMENT_ONLY / NO CONTROLLER`, not a renamed method.

## Bounded-search Caveat

This report checked direct primary-source neighbors available by 2026-08-23. It is not an exhaustive proof of novelty. Venue metadata was not inferred when only an arXiv source was confirmed. A final paper submission requires a fresh search after the I1 signal Gate, because several direct neighbors are 2026 preprints.
