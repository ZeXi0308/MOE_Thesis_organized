# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:validation` offset `32`, n=`28`
- sequence length: `256`
- target low-bit pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.044678`
- calibrated cumulative tail-mass budget: `0.156616`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 8.541599 | 0.000000 | 0.000000 | 7.489552 | 9.778139 |
| uniform_fp8 | 0.500000 | 0.496094 | 8.544977 | 0.003378 | 0.003323 | 7.498106 | 9.775140 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 8.750209 | 0.208611 | 0.031585 | 7.704602 | 9.978568 |
| rank_tail4_mxfp4 | 0.625000 | 0.615234 | 8.532795 | -0.008804 | 0.005634 | 7.475534 | 9.767328 |
| gate_threshold_mxfp4 | 0.625924 | 0.616115 | 8.528763 | -0.012836 | 0.005116 | 7.475380 | 9.758722 |
| gate_tailmass_mxfp4 | 0.625433 | 0.615647 | 8.525274 | -0.016325 | 0.005197 | 7.472846 | 9.762567 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 0.615234 | 8.547327 | 0.005728 | 0.005029 | 7.494219 | 9.779429 |
| head4_mxfp4_control | 0.625000 | 0.615234 | 8.795396 | 0.253798 | 0.029510 | 7.733101 | 10.050476 |
| interleaved4_mxfp4_control | 0.625000 | 0.615234 | 8.717076 | 0.175477 | 0.023601 | 7.656302 | 9.963773 |
| block_gate1_mxfp4 | 0.625000 | 0.615234 | 8.532795 | -0.008804 | 0.005634 | 7.475534 | 9.767328 |
| block_gate2_mxfp4 | 0.625000 | 0.615234 | 8.534229 | -0.007369 | 0.005457 | 7.487331 | 9.764578 |
| block_gate4_mxfp4 | 0.625000 | 0.615234 | 8.529230 | -0.012369 | 0.005281 | 7.479465 | 9.760818 |
| block_gate8_mxfp4 | 0.625000 | 0.615234 | 8.547496 | 0.005897 | 0.005146 | 7.490976 | 9.784313 |
| block_gate16_mxfp4 | 0.625000 | 0.615234 | 8.535051 | -0.006547 | 0.005191 | 7.485878 | 9.768724 |
| block_gate32_mxfp4 | 0.625000 | 0.615234 | 8.530609 | -0.010989 | 0.005175 | 7.481191 | 9.764639 |
| block_gate1_residual_mxfp4 | 0.625000 | 0.601562 | 8.514571 | -0.027028 | 0.005147 | 7.470096 | 9.736441 |
| block_gate2_residual_mxfp4 | 0.625000 | 0.601562 | 8.543099 | 0.001500 | 0.005049 | 7.490445 | 9.771687 |
| block_gate4_residual_mxfp4 | 0.625000 | 0.601562 | 8.533650 | -0.007949 | 0.004931 | 7.488517 | 9.763319 |
| block_gate8_residual_mxfp4 | 0.625000 | 0.601562 | 8.531984 | -0.009614 | 0.004850 | 7.483345 | 9.761502 |
| block_gate16_residual_mxfp4 | 0.625000 | 0.601562 | 8.543217 | 0.001619 | 0.004727 | 7.487019 | 9.777836 |
| block_gate32_residual_mxfp4 | 0.625000 | 0.601562 | 8.540076 | -0.001523 | 0.004764 | 7.487858 | 9.770953 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- `block_gate*` is a quality-side proxy: it fixes the FP8/low-bit count per contiguous token block, but does not implement peer-specific packing or a communication kernel.
- `block_gate*_residual_*` sends a low-bit base for every pair and a second low-bit residual for the critical half; its metadata accounting includes both sets of scales.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
