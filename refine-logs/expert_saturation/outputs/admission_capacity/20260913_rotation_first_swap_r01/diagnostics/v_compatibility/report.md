# Actual first-swap and tail paths

| Baseline → action | width2 calls Δ | width2 host ms Δ | A early four completion ms Δ | A tail two completion ms Δ | Same A tail two |
|---|---:|---:|---|---|---|
| cohort0-block0-least_progress → cohort0-block0-most_output | -79 | -436.650 | 0003733: +1437.390, 0003820: +755.691, 0003941: +1291.447, 0004015: +716.611 | 0007877: -325.812, 0008125: -330.912 | True |
| cohort0-block1-least_progress → cohort0-block1-most_output | -79 | -431.541 | 0003733: +1491.862, 0003820: +813.649, 0003941: +1347.415, 0004015: +775.217 | 0007877: -250.546, 0008125: -254.717 | True |
| cohort1-block0-least_progress → cohort1-block0-most_output | -79 | -440.042 | 0008366: +1430.288, 0008446: +752.814, 0008542: +1280.130, 0008901: +715.722 | 0011810: -334.464, 0012079: -339.656 | True |
| cohort1-block1-least_progress → cohort1-block1-most_output | -79 | -437.886 | 0008366: +1409.649, 0008446: +726.071, 0008542: +1258.272, 0008901: +688.374 | 0011810: -351.123, 0012079: -355.337 | True |

Manifest roles/blocks; source-ID joins; C uses recorded effective order. Main analysis qualifies successful-swap counts, costs and recovery chains. Output counts exclude recompute. Width bucket time changes and displaced completion are actual paths, not additive causal savings or hard bounds. No new C GPU evidence exists until the full primary campaign qualifies.
