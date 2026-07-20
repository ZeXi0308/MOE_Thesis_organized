# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2:validation` offset `0`, n=`8`
- test: `wikitext2:validation` offset `128`, n=`8`
- sequence length: `128`
- target INT4 pair fraction: `0.5000`
- calibrated gate threshold: `0.045410`
- calibrated cumulative tail-mass budget: `0.158081`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 37.878800 | 0.000000 | 0.000000 | 26.929224 | 58.498431 |
| uniform_fp8 | 0.500000 | 37.226029 | -0.652771 | 0.004459 | 26.638564 | 57.106688 |
| rank_tail4_int4 | 0.625000 | 38.277283 | 0.398483 | 0.036094 | 27.426811 | 58.035069 |
| gate_threshold_int4 | 0.633702 | 38.018000 | 0.139200 | 0.025288 | 27.155670 | 58.721644 |
| gate_tailmass_int4 | 0.626931 | 37.486282 | -0.392518 | 0.027695 | 27.073388 | 57.273936 |
| contribution_tail4_int4_oracle | 0.625000 | 37.384727 | -0.494073 | 0.023008 | 26.968542 | 56.697542 |
| head4_int4_control | 0.625000 | 57.606395 | 19.727595 | 0.449879 | 38.058498 | 97.747421 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` excludes scale metadata, packing, alignment, and communication-kernel overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
