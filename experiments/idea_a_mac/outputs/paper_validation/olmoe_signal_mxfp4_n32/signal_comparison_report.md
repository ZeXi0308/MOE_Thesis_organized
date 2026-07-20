# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2:validation` offset `0`, n=`8`
- test: `wikitext2:validation` offset `128`, n=`32`
- sequence length: `128`
- target INT4 pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.045410`
- calibrated cumulative tail-mass budget: `0.158081`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 18.791504 | 0.000000 | 0.000000 | 16.123690 | 23.097583 |
| uniform_fp8 | 0.500000 | 18.783238 | -0.008266 | 0.004722 | 16.133224 | 23.042301 |
| rank_tail4_mxfp4 | 0.625000 | 18.734759 | -0.056745 | 0.006840 | 16.041733 | 22.960436 |
| gate_threshold_mxfp4 | 0.628959 | 18.745336 | -0.046168 | 0.006151 | 16.083485 | 22.981446 |
| gate_tailmass_mxfp4 | 0.626323 | 18.773909 | -0.017595 | 0.006328 | 16.103160 | 23.044294 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 18.775607 | -0.015897 | 0.005509 | 16.147839 | 22.963805 |
| head4_mxfp4_control | 0.625000 | 20.044932 | 1.253428 | 0.059874 | 17.032165 | 24.788970 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` excludes scale metadata, packing, alignment, and communication-kernel overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
