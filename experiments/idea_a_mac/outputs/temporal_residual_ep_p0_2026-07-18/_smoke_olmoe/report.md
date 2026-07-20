# Temporal-Residual EP Mac P0

> Evidence boundary: numerical fake quantization + logical payload only; no GPU kernel,
> collective, NIC bytes, TPOT, or P99 evidence.

- model: `allenai/OLMoE-1B-7B-0924`
- samples / seq_len: `1` / `32`
- stock-vs-patched exactness: `True` (max abs `0.0`)
- adjacent-token expert revisit rate: `38.6102%`
- temporal logical-byte saving vs uniform FP8: `17.9779%`
- temporal/direct same-budget end-to-end KL ratio: `0.707116`
- preregistered-style P0 verdict: **PASS**

The causal control is `revisit_abs_mxfp4`: it compresses exactly the same routed
pairs with the same formats and bytes, but quantizes the absolute output instead
of a closed-loop temporal residual.  A win over this control isolates predictive
coding from merely selecting recurrent experts.

Promotion gates:

1. revisit rate >= 30%;
2. logical-byte saving vs uniform FP8 >= 15%;
3. temporal end-to-end KL <= 0.8x same-budget direct-MXFP4 KL.

Passing this P0 only licenses a fresh held-out replication and a GPU codec
microbenchmark.  It does not establish communication speedup.
