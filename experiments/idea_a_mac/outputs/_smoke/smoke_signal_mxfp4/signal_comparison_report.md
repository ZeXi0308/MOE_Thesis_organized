# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2:validation` offset `0`, n=`2`
- test: `wikitext2:validation` offset `128`, n=`2`
- sequence length: `64`
- target INT4 pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.045898`
- calibrated cumulative tail-mass budget: `0.158447`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 30.405824 | 0.000000 | 0.000000 | 29.479461 | 31.792318 |
| uniform_fp8 | 0.500000 | 30.016300 | -0.389523 | 0.004026 | 29.212619 | 31.213597 |
| rank_tail4_mxfp4 | 0.625000 | 29.717769 | -0.688055 | 0.006018 | 29.154530 | 30.548691 |
| gate_threshold_mxfp4 | 0.634949 | 30.211588 | -0.194236 | 0.005858 | 29.496678 | 31.272487 |
| gate_tailmass_mxfp4 | 0.625965 | 30.223673 | -0.182150 | 0.005268 | 29.436364 | 31.395511 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 30.060854 | -0.344969 | 0.005140 | 29.108391 | 31.488607 |
| head4_mxfp4_control | 0.625000 | 31.802409 | 1.396586 | 0.089595 | 30.384927 | 33.962367 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` excludes scale metadata, packing, alignment, and communication-kernel overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
