# Old d6 `most_output` victim cost identity

Evidence: `CPU_OBSERVED_PRE_ACTION_IDENTITY_ONLY`. This is an old-data diagnostic; no GPU or future trajectory was run.

| Cell | Rotations | Eligible/feasible rows | Selected in max-block tie | Unique max block | Selected=min recompute | Pooled output/block r |
|---|---:|---:|---:|---:|---:|---:|
| block0-d6-most_output | 38 | 962/962 | 38 | 3 | 1 | 0.999689 |
| block1-d6-most_output | 38 | 962/962 | 38 | 3 | 1 | 0.999689 |

Both repeats are source-level identical across targets, selected victims, candidate sets, and visible values: `True`.

Every one of the 962 eligible candidate rows per repeat can fund the target's full visible history. The selected victim belongs to the maximum-block tie in all 38 actions, but is the unique maximum in only 3. Max-block tie-size histogram: `{'1': 3, '2': 6, '3': 6, '4': 9, '5': 2, '6': 4, '7': 3, '8': 3, '9': 1, '10': 1}`.

The feasible minimum-current-recompute victim differs in 37 of 38 actions. The only equality is the one-candidate final action. When different, `most_output` carries 104.16 more visible recompute tokens on average (range 56–149), and 4–9 more releasable blocks.

The fixed-length confound is exact in these snapshots: all prompts are `3072` tokens; all 962 rows satisfy `computed = prompt + output - 1` and `owned_blocks = ceil(computed / 16)`. Output-to-block ordering has 0 inversions; 2181 of 12443 unequal-output pairs (17.53%) tie after 16-token block quantization.

Exact per-step alternatives and deltas are in `differences.csv`; complete feasible candidate sets and source identities are in `analysis.json`.

Interpretation boundary: this establishes ranking identity under the old homogeneous d6 states. It does not estimate how any alternative victim would affect later batches or completion.
