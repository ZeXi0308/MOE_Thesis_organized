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
- EP sizes: `[8]`; mappings: `['contiguous']`
- peer tile: `64` present grouped vectors
- exact per-tile FP8 target: `0.5` (implemented as `floor(n*f)`)
- patched full exact: `True`
- grouped EP=1 exact: `True`

## Results

| ep_size | mapping | strategy | n_over_m | collision_pair_fraction | observed_high_fraction | wire_saving_vs_grouped_bf16 | local_relative_mse | mean_token_kl_vs_grouped_bf16 | ppl_delta_vs_grouped_bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | contiguous | grouped_bf16 | 2.10597 | 0.52516 | 0 | 0 | 0 | 0 | 0 |
| 8 | contiguous | uniform_fp8 | 2.10578 | 0.525116 | 1 | 0.496094 | 0.000355813 | 0.00645407 | 0.0417126 |
| 8 | contiguous | uniform_mxfp4 | 2.10557 | 0.525069 | 0 | 0.734375 | 0.0143604 | 0.0648929 | 0.848249 |
| 8 | contiguous | mixed_rank | 2.1061 | 0.525188 | 0.499096 | 0.61545 | 0.000956328 | 0.0102864 | 0.0901887 |
| 8 | contiguous | mixed_gate_mass | 2.10604 | 0.525176 | 0.499035 | 0.615464 | 0.000977681 | 0.010705 | 0.049112 |
| 8 | contiguous | mixed_inputnorm_gate | 2.10614 | 0.525197 | 0.499009 | 0.61547 | 0.000984947 | 0.0104862 | 0.0494784 |
| 8 | contiguous | mixed_pair_contribution | 2.1059 | 0.525143 | 0.499072 | 0.615455 | 0.000544552 | 0.00861885 | 0.0152495 |
| 8 | contiguous | mixed_contribution | 2.10576 | 0.525112 | 0.49904 | 0.615463 | 0.000538134 | 0.00830924 | 0.00361664 |
| 8 | contiguous | global_contribution | 2.10625 | 0.525223 | 0.499885 | 0.615262 | 0.000440648 | 0.0067929 | 0.0454031 |
| 8 | contiguous | token_contribution | 2.10625 | 0.525223 | 0.477516 | 0.620592 | 0.000547694 | 0.00885686 | 0.0767416 |
| 8 | contiguous | mixed_qerr | 2.10618 | 0.525208 | 0.499068 | 0.615456 | 0.000539297 | 0.00823482 | 0.0253619 |
| 8 | contiguous | mixed_oracle | 2.10589 | 0.525141 | 0.499047 | 0.615462 | 0.000855721 | 0.00935849 | 0.0575671 |
| 8 | contiguous | mixed_random | 2.10581 | 0.525124 | 0.499076 | 0.615454 | 0.0113136 | 0.052695 | 0.707796 |

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
