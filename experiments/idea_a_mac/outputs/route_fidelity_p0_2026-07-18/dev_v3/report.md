# RouteFidelity P0-A

> Boundary: route hypergraph and logical rank-deduplicated records only.  This
> is not an actual backend, wire, or latency experiment.

| model | surrogate | max receiver-P99 underestimate | hypothetical exceedance when underestimate >=5% | max placement regret |
|---|---|---:|---:|---:|
| LLM-jp | `degree_shuffle` | 0.8% | 0.0% | 0.7% |
| LLM-jp | `hyperedge_shuffle` | 0.8% | 0.0% | 0.0% |
| LLM-jp | `uniform_unique` | 13.3% | 99.9% | 1.5% |
| OLMoE | `degree_shuffle` | 10.0% | 11.3% | 2.5% |
| OLMoE | `hyperedge_shuffle` | 10.2% | 8.6% | 2.1% |
| OLMoE | `uniform_unique` | 43.0% | 60.8% | 2.1% |

Verdict: **EXPLORATORY_SIGNAL**

Compared abstractions:

- `uniform_unique`: architecture-only B/E/k style synthesis;
- `degree_shuffle`: preserves exact layer-level expert occurrence counts while
  destroying token-level co-activation and ordering;
- `hyperedge_shuffle`: preserves every routed top-k set but destroys temporal bursts;
- `real`: preserves the complete token-expert hypergraph and ordering.

This is a development-grid maximum scan, so `EXPLORATORY_SIGNAL` is not a
confirmatory pass. The exceedance column applies a hypothetical zero-headroom
logical capacity and is neither a real buffer overflow nor an OOM/drop claim.
Placement regret remained below the 5% gate. CCF-C promotion requires fresh,
pre-registered serving batches, two real backend adapters, and at least one
reproducible >=5% latency or backend-configuration ranking inversion whose
confidence interval excludes zero. Without that, this remains a
benchmark/workshop seed.
