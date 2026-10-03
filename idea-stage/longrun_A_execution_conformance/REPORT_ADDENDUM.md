# Post-audit claim addendum

Date: 2026-08-13
Applies to: sealed artifact `artifacts/longrun_A_execution_conformance/20260812T204037Z/report.md`

The artifact remains byte-for-byte sealed. This addendum narrows the narrative after fresh same-family audit:

1. Replace “near-boundary/top-k amplification” with **`NEAR_BOUNDARY_ASSOCIATION_ONLY`**. The frozen margin supports 13/18 association. Lost/gained-expert order crossing is part of the definition of a route flip and is not independent causal evidence of amplification.
2. The smallest native-runtime transfer is **one steady plus one bursty A/C/D event**, not two events from each regime. It must retain target token/KV/position and exact-length companions and observe the internal MoE grouping/operator boundary.
3. “Expert-execution grouping” is a hypothesis, not a measured cause. With width and the full KV-length/padding vector fixed, the measured statement is only that the first observed C/D difference is at MoE output.
4. Arm C reproduces the selected batched sources, but historical serial identity was not a validity condition. One selected event did not match its recorded historical serial route in any repeat, so exact all-six historical serial reproduction is not claimed.
5. Fresh A/B/C/D output artifacts and both post-run seals are self-contained. The raw `/tmp` source captures were hash-checked during execution but are absent locally; original-event selection provenance is therefore not independently replayable from the retained workspace.
6. In the prevalence probe, the first exact A/C difference is attention output in 24/24 cases; the first material/non-allclose difference is attention output in 23/24. One steady width-2 case first becomes material at pre-router hidden.

Machine classifications in `arm_metrics.json` are exact-bit sensitive. The scientific report therefore uses the first non-allclose tensor stage plus actual route membership, not the machine label alone. The three repeat calls per arm ran inside one loaded model process; they are not independent process restarts. No raw metric, tensor, repeat, or sealed artifact was changed.
