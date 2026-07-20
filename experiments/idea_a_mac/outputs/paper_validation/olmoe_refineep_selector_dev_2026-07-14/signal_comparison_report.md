# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:validation` offset `32`, n=`16`
- sequence length: `256`
- target low-bit pair fraction: `0.5000`
- residual metadata-matched direct low-bit fraction: `0.4357` (encoded `436`)
- tail precision: `mxfp4`
- calibrated gate threshold: `0.044678`
- metadata-matched calibrated gate threshold: `0.040771`
- calibrated cumulative tail-mass budget: `0.156616`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 9.263349 | 0.000000 | 0.000000 | 7.882340 | 10.838404 |
| uniform_fp8 | 0.500000 | 0.499023 | 9.264525 | 0.001176 | 0.003218 | 7.883325 | 10.847301 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 9.491377 | 0.228028 | 0.029334 | 8.130790 | 11.032051 |
| rank_tail4_mxfp4 | 0.625000 | 0.616699 | 9.254911 | -0.008438 | 0.005567 | 7.869047 | 10.834644 |
| gate_threshold_mxfp4 | 0.625517 | 0.617186 | 9.254878 | -0.008472 | 0.004831 | 7.887888 | 10.827216 |
| block_gate8_mxfp4 | 0.625000 | 0.616699 | 9.274608 | 0.011259 | 0.005058 | 7.908459 | 10.844720 |
| block_contrib8_mxfp4 | 0.625000 | 0.616699 | 9.258285 | -0.005065 | 0.004570 | 7.900330 | 10.816686 |
| block_qenergy8_mxfp4 | 0.625000 | 0.616699 | 9.275773 | 0.012423 | 0.006118 | 7.903931 | 10.856229 |
| block_qerr8_mxfp4 | 0.625000 | 0.616699 | 9.284697 | 0.021348 | 0.004626 | 7.909721 | 10.867537 |
| block_qbenefit8_mxfp4 | 0.625000 | 0.616699 | 9.269051 | 0.005701 | 0.004702 | 7.901016 | 10.847319 |
| block_random8_mxfp4 | 0.625000 | 0.616699 | 9.371457 | 0.108107 | 0.018211 | 8.008883 | 10.917925 |
| block_reversegate8_mxfp4 | 0.625000 | 0.616699 | 9.437726 | 0.174376 | 0.027418 | 8.049477 | 11.016608 |
| block_gate16_residual_mxfp4 | 0.625000 | 0.601562 | 9.266057 | 0.002708 | 0.004602 | 7.900444 | 10.836123 |
| block_contrib16_residual_mxfp4 | 0.625000 | 0.601562 | 9.256388 | -0.006961 | 0.004272 | 7.898296 | 10.810830 |
| block_resenergy16_residual_mxfp4 | 0.625000 | 0.601562 | 9.250736 | -0.012613 | 0.005996 | 7.901958 | 10.787708 |
| block_reserr16_residual_mxfp4 | 0.625000 | 0.601562 | 9.264632 | 0.001282 | 0.004305 | 7.907812 | 10.812386 |
| block_resbenefit16_residual_mxfp4 | 0.625000 | 0.601562 | 9.268169 | 0.004820 | 0.004303 | 7.911203 | 10.827844 |
| block_random16_residual_mxfp4 | 0.625000 | 0.601562 | 9.367255 | 0.103905 | 0.017998 | 8.004946 | 10.904516 |
| block_reversegate16_residual_mxfp4 | 0.625000 | 0.601562 | 9.482611 | 0.219262 | 0.027967 | 8.069450 | 11.097137 |
| block_gate16_f436_mxfp4 | 0.609375 | 0.601990 | 9.272985 | 0.009636 | 0.004648 | 7.912951 | 10.834174 |
| block_contrib16_f436_mxfp4 | 0.609375 | 0.601990 | 9.255908 | -0.007441 | 0.004366 | 7.885386 | 10.830229 |
| block_qenergy16_f436_mxfp4 | 0.609375 | 0.601990 | 9.272927 | 0.009578 | 0.005712 | 7.903484 | 10.849903 |
| block_qerr16_f436_mxfp4 | 0.609375 | 0.601990 | 9.249457 | -0.013892 | 0.004091 | 7.882722 | 10.810112 |
| block_qbenefit16_f436_mxfp4 | 0.609375 | 0.601990 | 9.259050 | -0.004299 | 0.004244 | 7.889830 | 10.821632 |
| block_random16_f436_mxfp4 | 0.609375 | 0.601990 | 9.381872 | 0.118523 | 0.016772 | 7.992893 | 10.957474 |
| block_reversegate16_f436_mxfp4 | 0.609375 | 0.601990 | 9.477100 | 0.213751 | 0.027244 | 8.081837 | 11.062497 |
| gate_threshold_matchedwire_mxfp4 | 0.608363 | 0.601037 | 9.240362 | -0.022987 | 0.004512 | 7.873202 | 10.811775 |

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
