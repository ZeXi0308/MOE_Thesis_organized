# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`2`
- test: `wikitext2_docs:validation` offset `32`, n=`2`
- sequence length: `64`
- target low-bit pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.046387`
- calibrated cumulative tail-mass budget: `0.161255`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 14.663579 | 0.000000 | 0.000000 | 8.816747 | 24.387743 |
| uniform_fp8 | 0.500000 | 0.496094 | 15.004640 | 0.341061 | 0.005557 | 8.859847 | 25.411187 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 15.317199 | 0.653620 | 0.052000 | 9.202154 | 25.495832 |
| rank_tail4_mxfp4 | 0.625000 | 0.615234 | 14.507980 | -0.155599 | 0.007431 | 8.728770 | 24.113532 |
| gate_threshold_mxfp4 | 0.630508 | 0.620485 | 14.740090 | 0.076511 | 0.005723 | 8.901644 | 24.407881 |
| gate_tailmass_mxfp4 | 0.626785 | 0.616936 | 14.870150 | 0.206571 | 0.007365 | 8.830311 | 25.041175 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 0.615234 | 14.726951 | 0.063372 | 0.006236 | 8.801044 | 24.642883 |
| head4_mxfp4_control | 0.625000 | 0.615234 | 15.922383 | 1.258804 | 0.051531 | 9.197615 | 27.563915 |
| interleaved4_mxfp4_control | 0.625000 | 0.615234 | 14.494093 | -0.169486 | 0.042531 | 9.250604 | 22.709731 |
| block_gate1_mxfp4 | 0.625000 | 0.615234 | 14.502155 | -0.161424 | 0.008785 | 8.834966 | 23.804564 |
| block_gate2_mxfp4 | 0.625000 | 0.615234 | 14.720994 | 0.057415 | 0.006813 | 8.819256 | 24.572102 |
| block_gate4_mxfp4 | 0.625000 | 0.615234 | 14.739996 | 0.076417 | 0.007412 | 8.872658 | 24.487303 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- `block_gate*` is a quality-side proxy: it fixes the FP8/low-bit count per contiguous token block, but does not implement peer-specific packing or a communication kernel.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
