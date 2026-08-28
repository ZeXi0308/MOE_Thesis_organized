# TokenRace-EP P2: Adaptive Trigger Sweep (corrected thresholds)

rebatch_overhead_us(real, GPU P0)=26.644
threshold_mode=absolute

Note: legacy multiplier grids (0.5×rebatch ≈ 13µs) are usually vacuous because
deterministic predicted_gap is typically ≪ a few µs when per_token_us≈0.03.
Only absolute / gap-percentile grids with non-zero trigger rates are informative.

any_informative_nonzero_threshold_passes_5pct=False

## Moderate-regime gate check

| model | threshold_tag | threshold_us | min_p99_improvement | mean_trigger_rate | passes_5pct_gate | informative |
|---|---|---|---|---|---|---|
| llmjp | abs_us | 0.0000 | -0.0318 | 0.1428 | False | True |
| llmjp | abs_us | 0.1000 | -0.0179 | 0.0114 | False | True |
| llmjp | abs_us | 0.5000 | -0.0000 | 0.0000 | False | False |
| llmjp | abs_us | 1.0000 | -0.0000 | 0.0000 | False | False |
| llmjp | abs_us | 2.0000 | -0.0000 | 0.0000 | False | False |
| llmjp | abs_us | 5.0000 | -0.0000 | 0.0000 | False | False |
| llmjp | abs_us | 10.0000 | -0.0000 | 0.0000 | False | False |
| olmoe | abs_us | 0.0000 | 0.0215 | 0.5205 | False | True |
| olmoe | abs_us | 0.1000 | 0.0093 | 0.4251 | False | True |
| olmoe | abs_us | 0.5000 | 0.0000 | 0.1078 | False | True |
| olmoe | abs_us | 1.0000 | -0.0007 | 0.0254 | False | True |
| olmoe | abs_us | 2.0000 | -0.0000 | 0.0015 | False | True |
| olmoe | abs_us | 5.0000 | -0.0000 | 0.0000 | False | False |
| olmoe | abs_us | 10.0000 | -0.0000 | 0.0000 | False | False |

**Verdict:** MAINTAIN_NO_ADAPTIVE_RESCUE: under absolute/percentile thresholds with non-zero trigger where applicable, no cell restores a cross-batch 5% P99 gate. Delete the claim 'mult>=0.5 proves adaptive impossible'; replace with 'deterministic load imbalance is too small vs rebatch'.