# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:test` offset `1`, n=`60`
- sequence length: `256`
- target low-bit pair fraction: `0.5000`
- residual metadata-matched direct low-bit fraction: `0.4426` (encoded `443`)
- tail precision: `mxfp4`
- calibrated gate threshold: `0.047119`
- metadata-matched calibrated gate threshold: `0.043213`
- calibrated cumulative tail-mass budget: `0.283936`

## Results

| strategy | theoretical_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | corpus_ppl_ci_low | corpus_ppl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 13.768351 | 0.000000 | 0.000000 | 12.744336 | 14.896939 |
| uniform_fp8 | 0.500000 | 0.496094 | 13.782834 | 0.014483 | 0.004765 | 12.760060 | 14.918617 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 14.652363 | 0.884012 | 0.066271 | 13.561438 | 15.855350 |
| rank_tail8_mxfp4 | 0.625000 | 0.615234 | 13.826354 | 0.058003 | 0.006431 | 12.798755 | 14.956890 |
| gate_threshold_mxfp4 | 0.623907 | 0.614193 | 13.791276 | 0.022925 | 0.006055 | 12.764582 | 14.927520 |
| peerblock_gate64_mxfp4 | 0.624998 | 0.615233 | 13.796979 | 0.028628 | 0.006500 | 12.773174 | 14.928528 |
| peerblock_contrib64_mxfp4 | 0.625003 | 0.615237 | 13.810089 | 0.041738 | 0.005694 | 12.783864 | 14.941958 |
| peerblock_qerr64_mxfp4 | 0.625001 | 0.615235 | 13.803649 | 0.035298 | 0.005788 | 12.773108 | 14.944136 |
| peerblock_random64_mxfp4 | 0.625002 | 0.615236 | 14.552049 | 0.783698 | 0.054462 | 13.461557 | 15.751968 |
| peerblock_reversegate64_mxfp4 | 0.625000 | 0.615234 | 14.661743 | 0.893392 | 0.063410 | 13.560471 | 15.873618 |

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
