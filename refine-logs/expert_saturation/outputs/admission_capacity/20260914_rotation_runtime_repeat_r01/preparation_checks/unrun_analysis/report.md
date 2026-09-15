# Same-cohort runtime repeat

UNRUN

| Cell | Status | Complete | Wall s | Requests/s | Mean completion s | Max ITL s |
|---|---|---:|---:|---:|---:|---:|
| cohort2-block0-native | UNRUN | — | — | — | — | — |
| cohort2-block0-most_output | UNRUN | — | — | — | — | — |
| cohort2-block1-most_output | UNRUN | — | — | — | — | — |
| cohort2-block1-native | UNRUN | — | — | — | — | — |

## Primary paired observations

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|
| cohort2/0 | native → most_output | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |
| cohort2/1 | native → most_output | UNRUN_OR_INCOMPLETE_CAMPAIGN | — | — | — |

## Native A/A observed drift

| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |
|---|---|---|---:|---:|---:|

All four must qualify; native/most per block, never replace or correct baseline.

Same previously observed cohort2, native/most then most/native, four fresh engines. This is an exploratory runtime repeat, not a fresh-document holdout. Retain all calls, including long calls. No cache reset, pressure warmup, drift subtraction, significance, noise bound, quality equivalence or method GO.

Full wall retained; scheduler inclusive + engine non schedule + outside engine; no long-call subtraction.

Full paired metric vectors, per-request changes, costs and same-role block drift are retained in analysis.json.
