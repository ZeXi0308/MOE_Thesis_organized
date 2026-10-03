Verdict: `NO_TARGET_OUTPUT_DIFFERENCE_IN_TESTED_FIXED_WIDTH_OPERATOR`

Measured calls: 36; distinct target rows: 2.

| Contrast | Comparisons | Input differs | Router differs | Weights differ | IDs differ | Output differs | Max output abs | Counts changed | Absolute locations changed | Internal rows changed | Kernel config changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same_arm_repeat | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| original_vs_permuted | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 12 | 0 |
| original_vs_replaced | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 12 | 12 | 0 |
| cross_process | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Counts are comparisons, not independent samples. All per-repeat contrasts and exact dispatch counts/locations are in analysis.json.

| Process | Target | Comparison | Output differing repeats | Max output abs |
|---:|---:|---|---:|---:|
| 0 | 0 | original_vs_permuted | 0/3 | 0 |
| 0 | 0 | original_vs_replaced | 0/3 | 0 |
| 0 | 16 | original_vs_permuted | 0/3 | 0 |
| 0 | 16 | original_vs_replaced | 0/3 | 0 |
| 1 | 0 | original_vs_permuted | 0/3 | 0 |
| 1 | 0 | original_vs_replaced | 0/3 | 0 |
| 1 | 16 | original_vs_permuted | 0/3 | 0 |
| 1 | 16 | original_vs_replaced | 0/3 | 0 |

- Input comparisons reconstruct the frozen fixture row using retained source indices; per-call input tensors were not separately exported.
- Repeat comparisons and tensor elements are not independent text samples; only two target rows were tested.
- GPU-selected IDs and weights are observed; no CPU top-k reconstruction is used.
- Expert counts and sorted locations are dispatch metadata, not measured HBM traffic, congestion or request benefit.
- Prefill-end activations and isolated layer calls do not establish decode trajectory, task semantics, native graph-serving or performance effects.
