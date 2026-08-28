# Reviewer response

Verdict: **WARN / same-family provisional**, P0=0, P1=3.

1. **Expert-execution grouping is inferred, not measured.** The runner retains target residual, attention output, pre-router hidden, router logits, route, and MoE output, but no per-expert `M`, token order, kernel/config ID, individual expert output, dispatch/sort state, or combine intermediate. Supported wording is “fixed-width, fixed-KV-vector companion identity first changes the observed target at MoE output.”
2. **Historical serial-route reproduction is overstated.** Arm C closes captured batched sources, while historical serial identity is diagnostic only. `steady:olmoe-dev-steady-000:1` has `historical_serial_first_known_layer_match=false` in all repeats. Supported wording is six selected batched states plus stable same-pre-state A/C divergence.
3. **Original-event provenance is not self-contained.** Source captures were model-derived and honestly marked non-scientific ground truth. Their hashes were checked when external capture directories existed, but the raw captures and the development wrapper provenance are absent from the canonical bundle. Fresh A/B/C/D outputs are hash-closed; “independently verifiable historical real events” is unsupported.

A provenance: WARN. B normalization: PASS. C files/numbers/hashes: PASS, including exact post-run seal verification. D called paths: WARN because the grouping mechanism has no measurement path; the material-stage helper also has a dormant late-stage omission. E scope: WARN but otherwise properly bounded. F evaluation: `self_supervised_proxy` over real GPU execution.

Same logical pre-step state, arm independence, A/B/C/D semantics, within-process repeat stability, reported first-divergence stages, propagation KV independence, and primary no-future-leakage checks pass. Propagation teacher-forces captured future tokens. GPU isolation is sampled process isolation only. Prevalence is outcome-bounded, not incidence.

Final claim ceiling: stable single-OLMoE/single-RTX5090/custom-runtime companion-conditioned MoE-output externality with teacher-forced logit/KV persistence. No exact grouping mechanism, native transfer, quality, latency, capacity, controller, or SLO claim.
