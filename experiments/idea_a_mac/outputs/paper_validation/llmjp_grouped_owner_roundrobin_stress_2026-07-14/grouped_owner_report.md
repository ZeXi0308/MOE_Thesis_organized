# Grouped-Owner Combine Offline Experiment

This experiment models a non-expanded/local-reduction EP combine wire unit:
selected expert outputs on the same owner are multiplied by router weights and
reduced in BF16 before fake FP8/MXFP4. DeepEP/NCCL-style LL or expanded paths
may instead transmit per-expert responses. This is a quality and trace-structure
experiment, not a kernel or network benchmark.

## Configuration

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- data: `wikitext2_docs:validation`, offset `24`, n=`12`
- sequence length: `256`
- EP sizes: `[8]`; mappings: `['round_robin']`
- peer tile: `64` present grouped vectors
- exact per-tile FP8 target: `0.5` (implemented as `floor(n*f)`)
- patched full exact: `True`
- grouped EP=1 exact: `True`

## Results

| ep_size | mapping | strategy | n_over_m | collision_pair_fraction | observed_high_fraction | wire_saving_vs_grouped_bf16 | local_relative_mse | mean_token_kl_vs_grouped_bf16 | ppl_delta_vs_grouped_bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | round_robin | grouped_bf16 | 2.11798 | 0.527851 | 0 | 0 | 0 | 0 | 0 |
| 8 | round_robin | uniform_fp8 | 2.11805 | 0.527868 | 1 | 0.496094 | 0.000343997 | 0.00491426 | 0.00694533 |
| 8 | round_robin | mixed_rank | 2.11828 | 0.527918 | 0.498974 | 0.615479 | 0.000582264 | 0.00815891 | 0.0289751 |
| 8 | round_robin | mixed_gate_mass | 2.11831 | 0.527925 | 0.499026 | 0.615466 | 0.00056161 | 0.00800704 | -0.00381102 |
| 8 | round_robin | mixed_pair_contribution | 2.11842 | 0.527949 | 0.499038 | 0.615464 | 0.000524945 | 0.00731503 | 0.0535775 |
| 8 | round_robin | mixed_contribution | 2.11822 | 0.527905 | 0.498986 | 0.615476 | 0.000520313 | 0.00710826 | 0.0513721 |
| 8 | round_robin | global_contribution | 2.11833 | 0.52793 | 0.499875 | 0.615264 | 0.000434102 | 0.00669744 | 0.000956054 |
| 8 | round_robin | mixed_qerr | 2.11824 | 0.527911 | 0.49899 | 0.615475 | 0.000510913 | 0.00700137 | 0.0422714 |
| 8 | round_robin | mixed_random | 2.11776 | 0.527803 | 0.49894 | 0.615487 | 0.00681847 | 0.0360612 | 0.196605 |

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
