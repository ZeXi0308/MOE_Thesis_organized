# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:validation` offset `16`, n=`16`
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
| full | 0.000000 | 0.000000 | 6.988770 | 0.000000 | 0.000000 | 6.060325 | 8.020151 |
| uniform_fp8 | 0.500000 | 0.499023 | 6.985629 | -0.003141 | 0.003528 | 6.060198 | 8.009786 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 7.187213 | 0.198442 | 0.033794 | 6.227265 | 8.243139 |
| rank_tail4_mxfp4 | 0.625000 | 0.616699 | 6.990359 | 0.001589 | 0.006156 | 6.062787 | 8.020207 |
| gate_threshold_mxfp4 | 0.623612 | 0.615393 | 6.997657 | 0.008887 | 0.005856 | 6.054492 | 8.044455 |
| peerblock_gate64_mxfp4 | 0.624997 | 0.616696 | 6.986053 | -0.002717 | 0.006199 | 6.054306 | 8.017568 |
| peerblock_contrib64_mxfp4 | 0.625001 | 0.616701 | 6.971191 | -0.017579 | 0.005541 | 6.044687 | 7.997332 |
| peerblock_qerr64_mxfp4 | 0.624994 | 0.616694 | 6.994034 | 0.005264 | 0.005465 | 6.065425 | 8.020462 |
| peerblock_random64_mxfp4 | 0.624993 | 0.616692 | 7.111961 | 0.123191 | 0.021463 | 6.158267 | 8.175270 |
| peerblock_reversegate64_mxfp4 | 0.625009 | 0.616707 | 7.172548 | 0.183778 | 0.032400 | 6.210487 | 8.247390 |

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- `block_*` is a quality-side fixed-rate proxy: conditional on routed-pair count it fixes the FP8/low-bit composition per contiguous token block, but it neither fixes total message volume nor implements peer-specific packing or a communication kernel.
- `qenergy`/`resenergy` use owner-local low-bit error energy without gate metadata. `qerr`/`reserr` multiply it by gate squared. `qbenefit`/`resbenefit` additionally quantize the alternative representation for all pairs and are expensive score upper bounds, not deployment-ready selectors.
- `random` is a deterministic rank-independent anti-control; `reversegate` intentionally gives scarce precision/refinement to lower-gate pairs.
- `peerblock_*` groups routed pairs by synthetic expert-owner group. Because each Mac forward has one implicit token origin, this is a one-origin `(owner -> origin)` quality proxy; it still lacks multi-origin traces, actual placement, padding, and a communication kernel.
- `block_gate*_residual_*` sends a low-bit base for every pair and a second low-bit residual for the critical half; its metadata accounting includes both sets of scales.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
