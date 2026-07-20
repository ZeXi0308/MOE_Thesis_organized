# FP4 Format Comparison for PLTB

## Boundary

This is fake quantization, not a native FP4 communication kernel.  NVFP4 uses
E2M1 values, per-16 E4M3 block scales, and one FP32 global scale per vector.
MXFP4 uses E2M1 values with a per-32 power-of-two scale.  Metadata-aware wire
bytes include scale bytes but exclude message alignment and collective headers.

## Setup

- model: `jamesdborin/tiny-mixtral`; hidden size: `1024`; top-k: `2`
- test: `builtin:validation` offset `4`, n=`2`
- allocation: `kl_profile_3_5` = `[0, 2]`

## Results

| strategy | raw_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 38001.037446 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| uniform_fp8 | 0.500000 | 0.496094 | 38674.782430 | 673.744985 | 0.001247 | 0.000101 | 0.002393 |
| fixed_int4 | 0.625000 | 0.622070 | 38386.508611 | 385.471165 | 0.003111 | 0.003025 | 0.003196 |
| pltb_int4 | 0.625000 | 0.622070 | 38521.589518 | 520.552073 | 0.002040 | 0.000918 | 0.003162 |
| fixed_mxfp4 | 0.625000 | 0.615234 | 38139.063966 | 138.026521 | 0.002981 | 0.002030 | 0.003933 |
| pltb_mxfp4 | 0.625000 | 0.615234 | 38436.684187 | 435.646741 | 0.001758 | 0.000647 | 0.002869 |
| fixed_nvfp4 | 0.625000 | 0.606445 | 38360.598734 | 359.561289 | 0.004294 | 0.001789 | 0.006798 |
| pltb_nvfp4 | 0.625000 | 0.606445 | 38433.479532 | 432.442086 | 0.001591 | 0.000463 | 0.002720 |
