# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`32`
- test: `wikitext2_docs:test` offset `16`, n=`45`
- sequence length: `256`
- target low-bit pair fraction: `0.5000`
- tail precision: `mxfp4`
- calibrated gate threshold: `0.044922`
- calibrated cumulative tail-mass budget: `0.157227`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 8.378428 | 0.000000 | 0.000000 | 7.389457 | 9.441219 |
| uniform_fp8 | 0.500000 | 0.496094 | 8.391835 | 0.013407 | 0.003359 | 7.402237 | 9.454902 |
| rank_tail4_mxfp4 | 0.625000 | 0.615234 | 8.398173 | 0.019744 | 0.006042 | 7.409099 | 9.459529 |
| gate_threshold_mxfp4 | 0.626074 | 0.616258 | 8.400503 | 0.022074 | 0.005635 | 7.407148 | 9.463550 |
| gate_tailmass_mxfp4 | 0.625267 | 0.615489 | 8.407887 | 0.029458 | 0.005616 | 7.414938 | 9.471254 |
| contribution_tail4_mxfp4_oracle | 0.625000 | 0.615234 | 8.395329 | 0.016901 | 0.005559 | 7.407394 | 9.459905 |
| head4_mxfp4_control | 0.625000 | 0.615234 | 8.658963 | 0.280535 | 0.035938 | 7.658773 | 9.730558 |
| interleaved4_mxfp4_control | 0.625000 | 0.615234 | 8.558377 | 0.179949 | 0.027707 | 7.568076 | 9.611139 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
