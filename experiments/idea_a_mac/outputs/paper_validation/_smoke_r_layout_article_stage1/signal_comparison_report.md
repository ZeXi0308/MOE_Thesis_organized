# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`1`
- test: `wikitext2_docs:test` offset `0`, n=`1`
- sequence length: `64`
- target low-bit pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.046631`
- calibrated cumulative tail-mass budget: `0.161926`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 16.692691 | 0.000000 | 0.000000 | 16.692691 | 16.692691 |
| uniform_fp8 | 0.500000 | 0.496094 | 16.458831 | -0.233860 | 0.003891 | 16.458831 | 16.458831 |
| rank_tail4_mxfp4 | 0.625000 | 0.615234 | 16.759396 | 0.066705 | 0.005447 | 16.759396 | 16.759396 |
| gate_threshold_mxfp4 | 0.631836 | 0.621750 | 16.047975 | -0.644716 | 0.004729 | 16.047975 | 16.047975 |
| gate_tailmass_mxfp4 | 0.628571 | 0.618638 | 16.404689 | -0.288002 | 0.005107 | 16.404689 | 16.404689 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 0.615234 | 16.597709 | -0.094982 | 0.005165 | 16.597709 | 16.597709 |
| head4_mxfp4_control | 0.625000 | 0.615234 | 16.848096 | 0.155405 | 0.036827 | 16.848096 | 16.848096 |
| interleaved4_mxfp4_control | 0.625000 | 0.615234 | 17.229900 | 0.537209 | 0.055381 | 17.229900 | 17.229900 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
