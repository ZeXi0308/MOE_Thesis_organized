# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2:validation` offset `0`, n=`8`
- test: `wikitext2:validation` offset `128`, n=`16`
- sequence length: `128`
- target INT4 pair fraction: `0.5000`
- tail precision: `nvfp4`
- calibrated gate threshold: `0.045410`
- calibrated cumulative tail-mass budget: `0.158081`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 19.834418 | 0.000000 | 0.000000 | 16.807771 | 25.413363 |
| uniform_fp8 | 0.500000 | 19.792591 | -0.041827 | 0.004678 | 16.807796 | 25.262597 |
| rank_tail4_nvfp4 | 0.625000 | 19.676268 | -0.158150 | 0.005845 | 16.712038 | 25.066302 |
| gate_threshold_nvfp4 | 0.629930 | 19.676009 | -0.158409 | 0.005765 | 16.727532 | 25.086623 |
| gate_tailmass_nvfp4 | 0.626535 | 19.787368 | -0.047050 | 0.005805 | 16.841261 | 25.051320 |
| contribution_tail4_nvfp4_oracle | 0.625000 | 19.815913 | -0.018505 | 0.005670 | 16.816495 | 25.218225 |
| head4_nvfp4_control | 0.625000 | 20.185727 | 0.351309 | 0.014985 | 17.193179 | 25.648928 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` excludes scale metadata, packing, alignment, and communication-kernel overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
