# FP4 Format Comparison for PLTB

## Boundary

This is fake quantization, not a native FP4 communication kernel.  NVFP4 uses
E2M1 values, per-16 E4M3 block scales, and one FP32 global scale per vector.
MXFP4 uses E2M1 values with a per-32 power-of-two scale.  Metadata-aware wire
bytes include scale bytes but exclude message alignment and collective headers.

## Setup

- model: `allenai/OLMoE-1B-7B-0924`; hidden size: `2048`; top-k: `8`
- test: `wikitext2:validation` offset `128`, n=`32`
- allocation: `kl_profile_2_4_6` = `[2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 6, 6, 2]`

## Results

| strategy | raw_payload_saving_vs_bf16 | metadata_aware_wire_saving_vs_bf16 | corpus_ppl | ppl_delta_vs_full | mean_token_kl | mean_token_kl_ci_low | mean_token_kl_ci_high |
|---|---|---|---|---|---|---|---|
| full | 0.000000 | 0.000000 | 18.791504 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| uniform_fp8 | 0.500000 | 0.496094 | 18.783238 | -0.008266 | 0.004722 | 0.003774 | 0.005887 |
| fixed_int4 | 0.625000 | 0.622559 | 19.238104 | 0.446600 | 0.030320 | 0.025831 | 0.036156 |
| pltb_int4 | 0.625000 | 0.622559 | 18.998177 | 0.206673 | 0.019569 | 0.017629 | 0.022000 |
| fixed_mxfp4 | 0.625000 | 0.615234 | 18.734759 | -0.056745 | 0.006840 | 0.005646 | 0.008491 |
| pltb_mxfp4 | 0.625000 | 0.615234 | 18.723308 | -0.068196 | 0.006322 | 0.005497 | 0.007392 |
| fixed_nvfp4 | 0.625000 | 0.606934 | 18.669896 | -0.121608 | 0.005713 | 0.004721 | 0.007021 |
| pltb_nvfp4 | 0.625000 | 0.606934 | 18.700475 | -0.091030 | 0.005810 | 0.005028 | 0.006788 |
