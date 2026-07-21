# TokenRace-EP P1 Corrected Offline Recompute (2026-07-21)

graph_advantage_us=39.570, rebatch_2=47.952, rebatch_4=87.504

Interpretation: correcting graph placement credits the barrier (smaller B),
which typically makes negative improvements *more* negative. Early-only rebatch
is less pessimistic than unconditional. Expectation: does **not** resurrect TokenRace.

| scenario | model | B | P50 | P99 | pass5% |
|---|---|---:|---:|---:|---|
| corrected_barrier_graph_credit_early_only_2group | llmjp | 32 | -62.60% | -46.39% | False |
| corrected_barrier_graph_credit_early_only_2group | llmjp | 64 | -62.80% | -45.06% | False |
| corrected_barrier_graph_credit_early_only_2group | llmjp | 128 | -61.13% | -49.05% | False |
| corrected_barrier_graph_credit_early_only_2group | olmoe | 32 | -47.38% | -34.85% | False |
| corrected_barrier_graph_credit_early_only_2group | olmoe | 64 | -44.14% | -31.51% | False |
| corrected_barrier_graph_credit_early_only_2group | olmoe | 128 | -43.40% | -31.34% | False |
| corrected_barrier_graph_credit_early_only_4group | llmjp | 32 | -86.71% | -72.95% | False |
| corrected_barrier_graph_credit_early_only_4group | llmjp | 64 | -86.97% | -70.93% | False |
| corrected_barrier_graph_credit_early_only_4group | llmjp | 128 | -84.64% | -75.97% | False |
| corrected_barrier_graph_credit_early_only_4group | olmoe | 32 | -82.26% | -66.00% | False |
| corrected_barrier_graph_credit_early_only_4group | olmoe | 64 | -78.43% | -62.02% | False |
| corrected_barrier_graph_credit_early_only_4group | olmoe | 128 | -77.43% | -61.38% | False |
| corrected_no_graph_early_only_p0_rebatch | llmjp | 32 | -0.94% | 2.71% | False |
| corrected_no_graph_early_only_p0_rebatch | llmjp | 64 | -1.08% | 3.02% | False |
| corrected_no_graph_early_only_p0_rebatch | llmjp | 128 | -0.54% | 1.54% | False |
| corrected_no_graph_early_only_p0_rebatch | olmoe | 32 | 8.14% | 10.80% | True |
| corrected_no_graph_early_only_p0_rebatch | olmoe | 64 | 9.57% | 12.34% | True |
| corrected_no_graph_early_only_p0_rebatch | olmoe | 128 | 9.78% | 12.29% | True |
| legacy_p1_ii_graph_on_race_unconditional | llmjp | 32 | -61.60% | -51.75% | False |
| legacy_p1_ii_graph_on_race_unconditional | llmjp | 64 | -61.77% | -50.93% | False |
| legacy_p1_ii_graph_on_race_unconditional | llmjp | 128 | -60.84% | -53.58% | False |
| legacy_p1_ii_graph_on_race_unconditional | olmoe | 32 | -38.53% | -32.94% | False |
| legacy_p1_ii_graph_on_race_unconditional | olmoe | 64 | -36.07% | -30.25% | False |
| legacy_p1_ii_graph_on_race_unconditional | olmoe | 128 | -35.61% | -30.27% | False |
| legacy_unconditional_p0_rebatch | llmjp | 32 | -11.67% | -8.63% | False |
| legacy_unconditional_p0_rebatch | llmjp | 64 | -11.86% | -8.15% | False |
| legacy_unconditional_p0_rebatch | llmjp | 128 | -11.32% | -9.86% | False |
| legacy_unconditional_p0_rebatch | olmoe | 32 | 5.48% | 6.31% | True |
| legacy_unconditional_p0_rebatch | olmoe | 64 | 7.13% | 8.14% | True |
| legacy_unconditional_p0_rebatch | olmoe | 128 | 7.33% | 7.97% | True |

## Gate summary (corrected scenarios only)
- `corrected_barrier_graph_credit_early_only_2group`: all_cells_pass_5pct=False, min_p99=-49.05%
- `corrected_barrier_graph_credit_early_only_4group`: all_cells_pass_5pct=False, min_p99=-75.97%
- `corrected_no_graph_early_only_p0_rebatch`: all_cells_pass_5pct=False, min_p99=1.54%

**Verdict:** MAINTAIN_KILLED_AFTER_CORRECTION: no corrected scenario passes the 5% P99 gate across all model×B cells. P1 accounting was sloppy but fixing it does not resurrect TokenRace; if anything barrier graph credit deepens negatives.