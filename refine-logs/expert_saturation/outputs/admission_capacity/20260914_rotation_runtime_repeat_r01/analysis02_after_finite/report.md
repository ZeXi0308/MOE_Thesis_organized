# Same-cohort runtime repeat

MEASUREMENT_ONLY

| Cell | Status | Complete | Wall s | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|---:|
| cohort2-block0-native | COMPLETE | 32/32 | 24.438142 | 1.309429 | 21.903293 | 4.885155 |
| cohort2-block0-most_output | COMPLETE | 32/32 | 23.478504 | 1.362949 | 21.959875 | 1.064623 |
| cohort2-block1-most_output | COMPLETE | 32/32 | 24.326836 | 1.315420 | 22.803103 | 1.067163 |
| cohort2-block1-native | COMPLETE | 32/32 | 23.927953 | 1.337348 | 21.449650 | 4.710559 |

## Primary paired observations

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort2/0 | native → most_output | DESCRIPTIVE_MATCHED_PAIR | +4.087% | +0.258% | -3.820533 |
| cohort2/1 | native → most_output | DESCRIPTIVE_MATCHED_PAIR | -1.640% | +6.310% | -3.643396 |

## Native A/A observed drift

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|

All four must qualify; native/most per block, never replace or correct baseline.

Same previously observed cohort2, native/most then most/native, four fresh engines. This is an exploratory runtime repeat, not a fresh-document holdout. Retain all calls, including long calls. No cache reset, pressure warmup, drift subtraction, significance, noise bound, quality equivalence or method GO.

Full wall retained; scheduler inclusive + engine non schedule + outside engine; no long-call subtraction.

Full paired metric vectors, per-request changes, costs and same-role block drift are retained in analysis.json.
