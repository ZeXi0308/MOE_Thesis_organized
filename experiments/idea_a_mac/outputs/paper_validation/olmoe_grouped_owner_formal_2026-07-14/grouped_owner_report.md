# Grouped-Owner Combine Offline Experiment

This experiment models a non-expanded/local-reduction EP combine wire unit:
selected expert outputs on the same owner are multiplied by router weights and
reduced in BF16 before fake FP8/MXFP4. DeepEP/NCCL-style LL or expanded paths
may instead transmit per-expert responses. This is a quality and trace-structure
experiment, not a kernel or network benchmark.

## Configuration

- model: `allenai/OLMoE-1B-7B-0924`
- data: `wikitext2_docs:validation`, offset `48`, n=`12`
- sequence length: `256`
- EP sizes: `[8]`; mappings: `['contiguous']`
- peer tile: `64` present grouped vectors
- exact per-tile FP8 target: `0.5` (implemented as `floor(n*f)`)
- patched full exact: `True`
- grouped EP=1 exact: `True`

## Results

| ep_size | mapping | strategy | n_over_m | collision_pair_fraction | observed_high_fraction | wire_saving_vs_grouped_bf16 | local_relative_mse | mean_token_kl_vs_grouped_bf16 | ppl_delta_vs_grouped_bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | contiguous | grouped_bf16 | 1.4547 | 0.312571 | 0 | 0 | 0 | 0 | 0 |
| 8 | contiguous | uniform_fp8 | 1.4545 | 0.312477 | 1 | 0.499023 | 0.000414695 | 0.00331733 | 0.00653019 |
| 8 | contiguous | uniform_mxfp4 | 1.45481 | 0.312625 | 0 | 0.734375 | 0.0122069 | 0.0349517 | 0.1756 |
| 8 | contiguous | mixed_rank | 1.45465 | 0.312551 | 0.498631 | 0.617021 | 0.00227133 | 0.00652258 | 0.0457608 |
| 8 | contiguous | mixed_gate_mass | 1.45467 | 0.312558 | 0.498648 | 0.617017 | 0.00212471 | 0.00626147 | 0.0371099 |
| 8 | contiguous | mixed_inputnorm_gate | 1.4546 | 0.312528 | 0.498559 | 0.617038 | 0.00217084 | 0.00631045 | 0.0251951 |
| 8 | contiguous | mixed_pair_contribution | 1.45483 | 0.312632 | 0.498594 | 0.61703 | 0.00165812 | 0.00556318 | 0.0392076 |
| 8 | contiguous | mixed_contribution | 1.45457 | 0.31251 | 0.498568 | 0.617036 | 0.00162423 | 0.00542953 | 0.0470823 |
| 8 | contiguous | global_contribution | 1.45468 | 0.312564 | 0.499828 | 0.61674 | 0.00136112 | 0.00521146 | 0.0161667 |
| 8 | contiguous | token_contribution | 1.45515 | 0.312785 | 0.454712 | 0.627358 | 0.00198715 | 0.00620013 | 0.01351 |
| 8 | contiguous | mixed_qerr | 1.45457 | 0.31251 | 0.498591 | 0.617031 | 0.00159445 | 0.00540873 | 0.0292963 |
| 8 | contiguous | mixed_oracle | 1.4548 | 0.31262 | 0.498574 | 0.617035 | 0.00163783 | 0.00549443 | 0.0376228 |
| 8 | contiguous | mixed_random | 1.45513 | 0.312777 | 0.498577 | 0.617034 | 0.00542383 | 0.0153985 | 0.0717724 |

## Metric definitions and boundaries

- `N` is the number of routed expert pairs; `M` is the number of nonempty
  token-owner vectors after owner-local reduction. `N/M` and the collision
  fractions characterize existing local combine aggregation; they are not a
  new compression saving.
- `grouped_bf16` is the precision reference for EP>1. Its logit drift from the
  original single-process path is reported because owner grouping changes BF16
  associativity. Quantized KL is therefore reported against both references.
- Mixed selectors get exactly the same `floor(n*f)` FP8 cardinality within each
  present-vector tile of each owner. Total counts can deviate slightly from
  `f*M` because final short tiles are handled independently. A real decode
  implementation must define fallback/carry behavior and must not wait to fill
  a tile on the critical path.
- `mixed_rank` uses a gate/output-free sum of reciprocal routed ranks.
  `mixed_gate_mass` uses the owner-local sum of router weights.
  `mixed_inputnorm_gate` multiplies that mass by the origin-available input norm.
  `mixed_pair_contribution` uses `sum(g*||o||)` before owner aggregation, while
  `mixed_contribution` uses `||sum(g*o)||` and therefore includes cancellation.
  `global_contribution` and `token_contribution` keep the same score/codec but
  remove the peer quota, isolating its quality cost; their per-peer lane counts
  are variable and they are not the proposed regular layout.
  `mixed_qerr` uses its fake-MXFP4 error energy.
- `mixed_oracle` is an exact *one-owner-group* local FP4-to-FP8 intervention
  score that includes same-token cross-owner terms. Multiple simultaneous
  upgrades still interact, so it is not a global combinatorial optimum.
- Wire accounting includes the fake formats' vector/block scale bytes but omits
  alignment, padding, membership masks, headers, packing, and GPU kernel cost.
- Upstream approximation can change later-layer routes. A formal run should add
  disjoint datasets, larger models, native FP4/FP8 kernels, and real EP traces.
