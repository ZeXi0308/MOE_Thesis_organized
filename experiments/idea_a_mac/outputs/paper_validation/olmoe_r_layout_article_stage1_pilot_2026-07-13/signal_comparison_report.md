# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:test` offset `0`, n=`16`
- sequence length: `256`
- target low-bit pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.044678`
- calibrated cumulative tail-mass budget: `0.156616`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 7.420446 | 0.000000 | 0.000000 | 6.507832 | 8.323475 |
| uniform_fp8 | 0.500000 | 0.496094 | 7.447989 | 0.027543 | 0.003206 | 6.530123 | 8.366452 |
| rank_tail4_mxfp4 | 0.625000 | 0.615234 | 7.445008 | 0.024562 | 0.005849 | 6.532956 | 8.353798 |
| gate_threshold_mxfp4 | 0.625662 | 0.615865 | 7.425734 | 0.005288 | 0.005194 | 6.503111 | 8.345865 |
| gate_tailmass_mxfp4 | 0.625018 | 0.615251 | 7.445482 | 0.025036 | 0.005502 | 6.546664 | 8.342919 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 0.615234 | 7.461517 | 0.041071 | 0.005421 | 6.556829 | 8.369130 |
| head4_mxfp4_control | 0.625000 | 0.615234 | 7.591371 | 0.170925 | 0.031741 | 6.691173 | 8.488936 |
| interleaved4_mxfp4_control | 0.625000 | 0.615234 | 7.565026 | 0.144580 | 0.029148 | 6.625804 | 8.494694 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
