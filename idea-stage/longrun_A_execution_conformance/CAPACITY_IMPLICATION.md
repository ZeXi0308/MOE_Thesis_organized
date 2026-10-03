# Route Capacity Implication

Date: 2026-08-13

## Decision

- Observational track: `NEEDS_CONTROLLED_REPEAT`.
- Causal action track: `ACTION_ORACLE_STILL_PAUSED`.
- Paper direction: `PIVOT_TO_EXECUTION_CONFORMANCE_MEASUREMENT`.

Batch-dependent route does **not** invalidate the observational question `historical batched route -> next-window batched latency/risk`. It does show that route is coupled to the same batch shape, padding, attention execution, and expert microbatch composition that also affect latency. A route feature can therefore be a stable predictor, a shape proxy, or an unstable runtime artifact; the current evidence does not distinguish them.

The retained M3-vs-M2 P95 pinball diagnostic was `+9.6288%`; another uncontrolled Python-environment run was `-24.0388%`. The positive value is backed by the retained metrics bundle; the negative value is retained only as a disclosed quarantined diagnostic without a sealed manifest and is not independently durable evidence. Dangerous underprediction improved in neither. This task did not run three new isolated full-capture repeats, so it deliberately does not evaluate signal stability or resume capacity. The minimum observational reopening evidence is three retained, process-isolated capture/analysis repeats with consistent M3-vs-M2 direction, fold direction, and dangerous-underprediction behavior.

The causal action question remains `running_set_budget -> SLO-goodput`. No budget was executed, and fixed-route replay is invalid because budget changes request set, batch/KV shape, route, and completion. Readiness requires action-conditioned reruns that independently regenerate those states for every budget and a representative request-level denominator.

What remains alive is a bounded observational hypothesis. What stops is the current interpretation of custom-runtime route telemetry as an already identified safe-capacity variable or controller signal.

The unique next experiment for the thesis is the one-steady plus one-bursty native-runtime conformance transfer defined in `SOURCE_LOCALIZATION_REPORT.md`; capacity repeats should follow only if that transfer preserves the phenomenon.
