# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `wikitext2_docs:validation` offset `0`, n=`16`
- test: `wikitext2_docs:validation` offset `32`, n=`8`
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
| full | 0.000000 | 0.000000 | 10.148635 | 0.000000 | 0.000000 | 8.032889 | 12.549482 |
| uniform_fp8 | 0.500000 | 0.499023 | 10.146318 | -0.002318 | 0.003262 | 8.023401 | 12.561234 |
| uniform_mxfp4 | 0.750000 | 0.734375 | 10.416441 | 0.267805 | 0.030828 | 8.369793 | 12.746470 |
| rank_tail4_mxfp4 | 0.625000 | 0.616699 | 10.156310 | 0.007675 | 0.005252 | 8.054278 | 12.555469 |
| gate_threshold_mxfp4 | 0.624730 | 0.616445 | 10.161793 | 0.013158 | 0.004682 | 8.063541 | 12.538315 |
| peerblock_gate8_mxfp4 | 0.625011 | 0.616710 | 10.146180 | -0.002455 | 0.005757 | 8.049075 | 12.529695 |
| peerblock_contrib8_mxfp4 | 0.624976 | 0.616677 | 10.197929 | 0.049294 | 0.004971 | 8.092883 | 12.580751 |
| peerblock_gate16_mxfp4 | 0.624975 | 0.616676 | 10.204020 | 0.055385 | 0.005404 | 8.102991 | 12.600350 |
| peerblock_contrib16_mxfp4 | 0.625006 | 0.616705 | 10.167680 | 0.019044 | 0.004831 | 8.080268 | 12.550971 |
| peerblock_gate32_mxfp4 | 0.624989 | 0.616688 | 10.186864 | 0.038228 | 0.005308 | 8.111415 | 12.558016 |
| peerblock_contrib32_mxfp4 | 0.624997 | 0.616697 | 10.131454 | -0.017181 | 0.004538 | 8.036878 | 12.495636 |
| peerblock_gate64_mxfp4 | 0.624987 | 0.616687 | 10.130084 | -0.018552 | 0.004928 | 8.028277 | 12.502618 |
| peerblock_contrib64_mxfp4 | 0.625013 | 0.616712 | 10.180993 | 0.032358 | 0.004747 | 8.081998 | 12.587921 |
| peerblock_gate128_mxfp4 | 0.625021 | 0.616719 | 10.179157 | 0.030521 | 0.005013 | 8.091644 | 12.554530 |
| peerblock_contrib128_mxfp4 | 0.624970 | 0.616671 | 10.177985 | 0.029349 | 0.004652 | 8.103178 | 12.551214 |

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
