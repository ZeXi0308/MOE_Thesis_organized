# Actual first-swap and tail paths

| Baseline → action | width2 calls Δ | width2 host ms Δ | A early four completion ms Δ | A tail two completion ms Δ | Same A tail two |
|---|---:|---:|---|---|---|
| cohort0-block0-least_progress → cohort0-block0-first_most_then_least | -29 | -159.713 | 0003733: +655.758, 0003820: -62.100, 0003941: -62.340, 0004015: -42.323 | 0007877: -184.661, 0008125: -37.735 | True |
| cohort0-block0-most_output → cohort0-block0-first_most_then_least | +50 | +273.718 | 0003733: -752.902, 0003820: -789.507, 0003941: -1326.232, 0004015: -731.051 | 0007877: +156.967, 0008125: +307.493 | True |
| cohort0-block0-least_progress → cohort0-block0-most_output | -79 | -433.430 | 0003733: +1408.659, 0003820: +727.407, 0003941: +1263.891, 0004015: +688.728 | 0007877: -341.628, 0008125: -345.228 | True |
| cohort0-block1-least_progress → cohort0-block1-first_most_then_least | -29 | -161.553 | 0003733: +795.760, 0003820: +76.996, 0003941: +76.666, 0004015: +97.026 | 0007877: -46.105, 0008125: +100.164 | True |
| cohort0-block1-most_output → cohort0-block1-first_most_then_least | +50 | +273.744 | 0003733: -715.153, 0003820: -750.075, 0003941: -1289.913, 0004015: -692.716 | 0007877: +194.394, 0008125: +345.232 | True |
| cohort0-block1-least_progress → cohort0-block1-most_output | -79 | -435.297 | 0003733: +1510.913, 0003820: +827.070, 0003941: +1366.579, 0004015: +789.742 | 0007877: -240.500, 0008125: -245.068 | True |

Manifest roles/blocks; source-ID joins; C uses recorded effective order. Main analysis qualifies successful-swap counts, costs and recovery chains. Output counts exclude recompute. Width bucket time changes and displaced completion are actual paths, not additive causal savings or hard bounds. No new C GPU evidence exists until the full primary campaign qualifies.
