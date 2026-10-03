# Route-Conditioned Capacity Envelope P1 development report

Verdict: `PIVOT_TO_EXECUTION_CONFORMANCE`.

| Method | P95 pinball loss | Dangerous underprediction | SLO-risk false negative |
|---|---:|---:|---:|
| M0_constant | 1.53372312 | 0.032258 | 1.000000 |
| M1_workload_only | 1.31690276 | 0.032258 | 1.000000 |
| M2_workload_plus_expert_load | 1.26975268 | 0.032258 | 1.000000 |
| M3_workload_expert_load_plus_historical_route | 1.14749054 | 0.032258 | 1.000000 |
| M4_future_route_latency_diagnostic | 1.45792341 | 0.032258 | 1.000000 |

M3 vs M2 P95 pinball relative improvement: `+9.6288%`.

M3 vs M2 dangerous-underprediction relative reduction: `+0.0000%`.

M4 reads the observed t+1 route and is diagnostic only. No action was executed; this report does not establish safe capacity, action headroom, controller gain, native serving behavior, or a two-model claim.

Serial-vs-batched route conformance: `{'statuses': ['BATCH_DEPENDENT'], 'minimum_expert_assignment_match_fraction': 0.947265625, 'batch_dependent_route_observed': True, 'interpretation': 'serial-vs-batched route differs despite exact token parity; capacity interpretation is not authorized'}`. A batch-dependent route finding overrides the route-signal diagnostic verdict and blocks a capacity interpretation.
