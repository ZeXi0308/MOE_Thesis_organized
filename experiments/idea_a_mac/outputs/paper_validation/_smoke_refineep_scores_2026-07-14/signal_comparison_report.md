# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`1`
- test: `wikitext2_docs:validation` offset `2`, n=`1`
- sequence length: `32`
- target low-bit pair fraction: `0.5000`
- residual metadata-matched direct low-bit fraction: `0.4357` (encoded `436`)
- tail precision: `mxfp4`
- calibrated gate threshold: `0.047363`
- metadata-matched calibrated gate threshold: `0.042725`
- calibrated cumulative tail-mass budget: `0.163147`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 56.072254 | 0.000000 | 0.000000 | 56.072254 | 56.072254 |
| uniform_fp8 | 0.500000 | 0.499023 | 55.551559 | -0.520694 | 0.005894 | 55.551559 | 55.551559 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 55.737190 | -0.335064 | 0.062796 | 55.737190 | 55.737190 |
| rank_tail4_mxfp4 | 0.625000 | 0.616699 | 54.898892 | -1.173362 | 0.009140 | 54.898892 | 54.898892 |
| gate_threshold_mxfp4 | 0.635315 | 0.626410 | 56.460069 | 0.387815 | 0.005952 | 56.460069 | 56.460069 |
| block_gate8_mxfp4 | 0.625000 | 0.616699 | 56.290842 | 0.218588 | 0.008559 | 56.290842 | 56.290842 |
| block_contrib8_mxfp4 | 0.625000 | 0.616699 | 55.689158 | -0.383096 | 0.006128 | 55.689158 | 55.689158 |
| block_qenergy8_mxfp4 | 0.625000 | 0.616699 | 53.949792 | -2.122461 | 0.008560 | 53.949792 | 53.949792 |
| block_qerr8_mxfp4 | 0.625000 | 0.616699 | 52.931536 | -3.140718 | 0.007287 | 52.931536 | 52.931536 |
| block_qbenefit8_mxfp4 | 0.625000 | 0.616699 | 53.102818 | -2.969436 | 0.007621 | 53.102818 | 53.102818 |
| block_random8_mxfp4 | 0.625000 | 0.616699 | 58.361808 | 2.289555 | 0.051379 | 58.361808 | 58.361808 |
| block_reversegate8_mxfp4 | 0.625000 | 0.616699 | 57.893968 | 1.821714 | 0.060866 | 57.893968 | 57.893968 |
| block_gate8_residual_mxfp4 | 0.625000 | 0.601562 | 55.518935 | -0.553319 | 0.008640 | 55.518935 | 55.518935 |
| block_contrib8_residual_mxfp4 | 0.625000 | 0.601562 | 54.829354 | -1.242900 | 0.008014 | 54.829354 | 54.829354 |
| block_resenergy8_residual_mxfp4 | 0.625000 | 0.601562 | 54.868001 | -1.204253 | 0.008248 | 54.868001 | 54.868001 |
| block_reserr8_residual_mxfp4 | 0.625000 | 0.601562 | 54.572840 | -1.499414 | 0.009386 | 54.572840 | 54.572840 |
| block_resbenefit8_residual_mxfp4 | 0.625000 | 0.601562 | 54.572840 | -1.499414 | 0.009386 | 54.572840 | 54.572840 |
| block_random8_residual_mxfp4 | 0.625000 | 0.601562 | 53.248607 | -2.823647 | 0.053867 | 53.248607 | 53.248607 |
| block_reversegate8_residual_mxfp4 | 0.625000 | 0.601562 | 55.298880 | -0.773374 | 0.076130 | 55.298880 | 55.298880 |
| block_gate8_f436_mxfp4 | 0.609375 | 0.601990 | 55.168993 | -0.903261 | 0.007832 | 55.168993 | 55.168993 |
| block_contrib8_f436_mxfp4 | 0.609375 | 0.601990 | 54.305524 | -1.766730 | 0.008451 | 54.305524 | 54.305524 |
| block_qenergy8_f436_mxfp4 | 0.609375 | 0.601990 | 55.085351 | -0.986903 | 0.007650 | 55.085351 | 55.085351 |
| block_qerr8_f436_mxfp4 | 0.609375 | 0.601990 | 55.065886 | -1.006368 | 0.005534 | 55.065886 | 55.065886 |
| block_qbenefit8_f436_mxfp4 | 0.609375 | 0.601990 | 55.043056 | -1.029198 | 0.005350 | 55.043056 | 55.043056 |
| block_random8_f436_mxfp4 | 0.609375 | 0.601990 | 55.584039 | -0.488215 | 0.048916 | 55.584039 | 55.584039 |
| block_reversegate8_f436_mxfp4 | 0.609375 | 0.601990 | 57.562083 | 1.489829 | 0.063509 | 57.562083 | 57.562083 |
| gate_threshold_matchedwire_mxfp4 | 0.617188 | 0.609344 | 56.632649 | 0.560395 | 0.005048 | 56.632649 | 56.632649 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- `block_*` is a quality-side fixed-rate proxy: conditional on routed-pair count it fixes the FP8/low-bit composition per contiguous token block, but it neither fixes total message volume nor implements peer-specific packing or a communication kernel.
- `qenergy`/`resenergy` use owner-local low-bit error energy without gate metadata. `qerr`/`reserr` multiply it by gate squared. `qbenefit`/`resbenefit` additionally quantize the alternative representation for all pairs and are expensive score upper bounds, not deployment-ready selectors.
- `random` is a deterministic rank-independent anti-control; `reversegate` intentionally gives scarce precision/refinement to lower-gate pairs.
- `peerblock_gate*` first groups routed pairs by synthetic expert-owner group, then fixes the precision/refinement count inside each peer-local pair tile; it is closer to an EP buffer but still lacks real origin ranks and padding.
- `block_gate*_residual_*` sends a low-bit base for every pair and a second low-bit residual for the critical half; its metadata accounting includes both sets of scales.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
