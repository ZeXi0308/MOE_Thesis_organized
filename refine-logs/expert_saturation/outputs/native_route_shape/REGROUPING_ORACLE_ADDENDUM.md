# Fixed-trace regrouping diagnostic addendum

## Canonical artifact

`native_route_regrouping_diagnostic_v3_final2.json` is the canonical result.

- Report SHA-256: `da5280aff25a42e2cd73bc51b595a9f41a358a6ec9e7a5685f2434e17618a940`
- Analyzer SHA-256: `a161f7a7224052c2c03ee62e365e9c49c953f59516651db7b8de120c8d3d174f`
- Test SHA-256: `df6b5b0bb3f11415790c54bbe8f51be15145c26015a27ba143ce7d9c8f7f19e0`
- Reproduction: a second run from the same three sealed bundles is byte-identical (`cmp` exit 0 and the same report SHA-256).

Earlier `native_route_regrouping_oracle_v1.json`, `native_route_regrouping_oracle_v2.json`, `native_route_regrouping_diagnostic_v3.json`, and `native_route_regrouping_diagnostic_v3_final.json` remain immutable history and are superseded. The final correction replaces mathematical upper-bound/action-space wording, aggregates correlated steps into trajectories, strengthens the route-blind ladder, and closes provenance checks.

## Corrected result

The claim ceiling is `STRUCTURAL_FIXED_TRACE_DIAGNOSTIC_ONLY`; the overall verdict is `BUDGET_GATE_COMPOSITION_DIAGNOSTIC_ONLY`.

The primary unit is one complete 15-step `(process_repeat, prompt_length)` trajectory, producing six descriptive summaries. These are not six independent workload samples: adjacent steps are correlated, P128 and P512 are prefix-related views, and all data come from one model and one synthesized 96-request pool. The 90 individual step cells remain diagnostics only.

The route-blind ladder contains original membership, static round-robin, stable request-key hash, and eight deterministic shuffle seeds declared before the v3 reanalysis but after data collection. They are not presented as a pre-data experiment preregistration. Every route-blind partition is fixed across decode steps and process repeats.

- Static round-robin is the strongest balance-oriented route-blind baseline in `6/6` trajectories.
- Relative to that strongest route-blind baseline, `t-1` history greedy changes mean per-layer maximum expert load by median `-0.10%`; all `6/6` trajectories are non-positive. Verdict: `NO_MATERIAL_FIXED_TRACE_HISTORY_RESIDUAL`.
- Current-step future-route local search retains median `7.09%` descriptive reduction relative to the strongest route-blind baseline. This is hindsight regrouping potential, not a mathematical bound or deployable signal.
- The separate working-set coalescing search reduces active experts by median `25.49%`, but deliberately increases HHI; it is not combined with the balance objective.

## Claim boundary and next Gate

The 96-request pool is synthesized from six separately executed B=16 calls, not an observed simultaneous continuous-batching ready set. Repartitioning reuses routes captured under the original composition; a native rerun can change hidden states, tokens, KV state, and later routes in either direction. No TPOT, latency, throughput, SLO-goodput, action-space, or controller-headroom claim is supported.

This diagnostic only says composition must be controlled or stratified in the action-conditioned **budget Gate**. It does not authorize a route-aware regrouping controller. The next mainline experiment should test the real budget action while treating composition as a matched diagnostic factor.
