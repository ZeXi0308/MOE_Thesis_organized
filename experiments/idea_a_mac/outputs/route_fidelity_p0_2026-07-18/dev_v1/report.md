# RouteFidelity P0-A

> Boundary: route hypergraph and logical rank-deduplicated records only.  This
> is not an actual backend, wire, or latency experiment.

| model | max receiver-P99 underestimate | max real overflow using synthetic P99 capacity | max placement regret |
|---|---:|---:|---:|
| LLM-jp | 13.3% | 99.9% | 1.5% |
| OLMoE | 43.0% | 60.8% | 2.1% |

Verdict: **PASS_P0_A**

Compared abstractions:

- `uniform_unique`: architecture-only B/E/k style synthesis;
- `marginal_unique`: preserves layer-level expert popularity in expectation;
- `hyperedge_shuffle`: preserves every routed top-k set but destroys temporal bursts;
- `real`: preserves the complete token-expert hypergraph and ordering.

`PASS_P0_A` establishes only that route correlation can change a logical
receiver-tail or capacity decision.  CCF-C promotion additionally requires two
real backend adapters and at least one reproducible >=5% latency/configuration
ranking inversion.  Without that, this remains a benchmark/workshop seed.
