# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2:validation` offset `0`, n=`16`
- test: `wikitext2:validation` offset `128`, n=`32`
- sequence length: `128`
- target INT4 pair fraction: `0.5000`
- calibrated gate threshold: `0.045898`
- calibrated cumulative tail-mass budget: `0.159424`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|
| full | 0.000000 | 18.791504 | 0.000000 | 0.000000 | 16.123690 | 23.097583 |
| uniform_fp8 | 0.500000 | 18.783238 | -0.008266 | 0.004722 | 16.133224 | 23.042301 |
| rank_tail4_int4 | 0.625000 | 19.238104 | 0.446600 | 0.030320 | 16.564886 | 23.593013 |
| gate_threshold_int4 | 0.631258 | 18.973757 | 0.182253 | 0.018361 | 16.305668 | 23.251888 |
| gate_tailmass_int4 | 0.627391 | 18.914243 | 0.122739 | 0.021300 | 16.286339 | 23.203481 |
| contribution_tail4_int4_oracle | 0.625000 | 18.947302 | 0.155798 | 0.018125 | 16.282756 | 23.231952 |
| head4_int4_control | 0.625000 | 24.616759 | 5.825255 | 0.280165 | 20.854740 | 31.096663 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` excludes scale metadata, packing, alignment, and communication-kernel overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
