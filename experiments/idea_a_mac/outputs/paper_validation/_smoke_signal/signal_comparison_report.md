# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `builtin:validation` offset `0`, n=`2`
- test: `builtin:validation` offset `2`, n=`2`
- sequence length: `32`
- target INT4 pair fraction: `0.5000`
- calibrated gate threshold: `0.500000`
- calibrated cumulative tail-mass budget: `0.749156`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 42482.094999 | 0.000000 | 0.000000 | 34593.023104 | 50847.681891 |
| uniform_fp8 | 0.500000 | 42389.413594 | -92.681405 | 0.000091 | 34489.945065 | 50772.284095 |
| rank_tail1_int4 | 0.625000 | 42078.002415 | -404.092584 | 0.005329 | 33839.478882 | 50916.393128 |
| gate_threshold_int4 | 0.625000 | 42078.002415 | -404.092584 | 0.005329 | 33839.478882 | 50916.393128 |
| gate_tailmass_int4 | 0.625000 | 42078.002415 | -404.092584 | 0.005329 | 33839.478882 | 50916.393128 |
| contribution_tail1_int4_oracle | 0.625000 | 42476.693465 | -5.401534 | 0.004169 | 33692.969300 | 52021.837900 |
| head1_int4_control | 0.625000 | 43154.097602 | 672.002603 | 0.005140 | 34966.783377 | 51876.133757 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` excludes scale metadata, packing, alignment, and communication-kernel overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
