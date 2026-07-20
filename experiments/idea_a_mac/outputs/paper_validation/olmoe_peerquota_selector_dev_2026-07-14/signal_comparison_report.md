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
| peerblock_gate64_mxfp4 | 0.624995 | 0.616694 | 9.233289 | -0.030060 | 0.005304 | 7.861910 | 10.805321 |
| peerblock_contrib64_mxfp4 | 0.625007 | 0.616706 | 9.287318 | 0.023969 | 0.004821 | 7.913064 | 10.863534 |
| peerblock_qenergy64_mxfp4 | 0.625012 | 0.616710 | 9.269649 | 0.006300 | 0.006338 | 7.907514 | 10.835658 |
| peerblock_qerr64_mxfp4 | 0.625000 | 0.616699 | 9.268748 | 0.005399 | 0.004658 | 7.904775 | 10.833245 |
| peerblock_qbenefit64_mxfp4 | 0.624995 | 0.616694 | 9.240971 | -0.022378 | 0.004736 | 7.872919 | 10.806902 |
| peerblock_random64_mxfp4 | 0.624979 | 0.616680 | 9.373384 | 0.110035 | 0.018259 | 7.988901 | 10.953267 |
| peerblock_reversegate64_mxfp4 | 0.625005 | 0.616704 | 9.495195 | 0.231846 | 0.029465 | 8.091777 | 11.087150 |

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
