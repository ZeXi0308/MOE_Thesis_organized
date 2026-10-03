# Two-term decomposition with heterogeneous output lengths

| cell | arm | policy | wall s | max ITL s | out-len CV | span s | tail waste s | marg recompute s | preempt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hom_b0 | homogeneous | native | 40.271 | 1.390 | 0.000 | 3.508 | 1.486 | 0.041 | 1 |
| hom_b1 | homogeneous | native | 40.001 | 1.332 | 0.000 | 3.501 | 1.424 | 0.041 | 1 |
| het_b0 | heterogeneous | native | 49.695 | 0.789 | 0.500 | 33.955 | 15.091 | 0.000 | 0 |

## Workload-inherent drain (reported, never subtracted)

| cell | measured tail waste s | workload-inherent estimate s |
|---|---:|---:|
| hom_b0 | 1.486 | 0.000 |
| hom_b1 | 1.424 | 0.000 |
| het_b0 | 15.091 | 0.000 |
