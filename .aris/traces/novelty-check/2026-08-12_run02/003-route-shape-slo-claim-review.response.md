# Independent claim-level novelty verdict

**Overall: `CAUTION` — run H1/H2, but do not issue an independent-method GO.**

The broad ingredients are known: route/activation shape affects MoE execution
time beyond token count; historical activation can predict later expert use;
SLO schedulers already control token/capacity budgets; and MoE schedulers feed
expert pressure into online decisions. No checked primary work established the
complete frozen residual:

```text
historical route shape adds held-out information beyond strong workload state
  -> changes safe next-window active-token capacity
  -> a causal controller captures the gain
```

This residual is a defensible falsifiable systems question. It is not yet an
independent mechanism claim.

## Claim verdicts

- **C1 incremental capacity signal: `CAUTION`.** ELDR, METRO, activation-pattern
  scaling, Sem-MoE, and Gimbal occupy the broad premise. The narrow residual is
  M3 versus M1 after token/running/queue/KV/recent-latency controls, matched
  cells, and document/time/arrival holdouts.
- **C2 route-conditioned next-window active-token budget: `CAUTION`.**
  SlidingServe, SLOs-Serve, SCORPIO, and HyGen occupy SLO-aware token/capacity
  control; Gimbal already uses MoE pressure online. The action alone is not
  novel. The only defensible boundary freezes routing, placement, rebatching,
  sealing, and EP behavior.
- **C3 matched test -> Oracle -> causal controller: `PROCEED_AS_VALIDATION_PROTOCOL`.**
  The complete chain was not found, but Oracle-to-policy validation is not new
  theory. The future-route Oracle must be an upper bound, use the same actuator
  and arrival/service process, and regenerate the admitted set and routes for
  every candidate budget.
- **C4 independent RouteShape-SLO system: `CAUTION_UNSUPPORTED_NOW`.** The
  semantic and implementation neighborhoods are crowded. Independence would
  require two frozen models, natural continuous-batching arrival episodes,
  stable M3-over-M1 gains, material action-conditioned Oracle headroom, causal
  recovery, and an ablation against scalar expert pressure/recent latency.

## Collision matrix

| Dimension | Collision |
|---|---:|
| Semantic | High |
| Theory | Medium-high |
| Implementation | High |
| Empirical | Medium-high |

The strongest validity threat is policy endogeneity: changing the active-token
budget changes the admitted request set and therefore future routes. Reusing
one observed future-route trace for all candidate actions creates fake
counterfactual ground truth.

## Query formulations used

For C1: `mixture of experts expert activation pattern latency prediction token
count admission control`; `MoE route-aware serving latency predictor expert
activation locality`; `active expert union batch latency mixture of experts`;
`expert activation latency predictor continuous batching MoE serving`.

For C2: `mixture of experts SLO admission control active token budget
route-aware`; `MoE serving token budget controller expert load SLO`;
`route-aware admission control MoE continuous batching`; `expert activation
admission control mixture-of-experts serving`.

For C3: `MoE safe capacity route-conditioned oracle counterfactual SLO
goodput`; `expert activation workload-aware batching placement admission MoE
inference`; `MoE latency prediction batch size sequence length expert load`;
`mixture of experts counterfactual capacity controller latency SLO`.

For C4: `MoE route-aware SLO controller active token budget`; `expert
activation admission controller mixture of experts serving`; `route
conditioned capacity control MoE inference`; `route-aware scheduler SLO
mixture-of-experts inference`.

## Decision boundary

KILL if route features collapse to recent latency/scalar expert pressure, gains
disappear on matched held-out cells, the action-conditioned Oracle has no
material SLO-goodput advantage, or the effect requires routing/placement/
rebatching/EP changes. Proceed toward an independent system only if H1 is
stable across models and episodes, H2 exposes a reproducible capacity gap, and
H3 captures a meaningful fraction of that gap without tail-SLO regression.

Bottom line: run only H1 and H2 as falsification gates; the simple controller
is unlikely to carry novelty by itself.
