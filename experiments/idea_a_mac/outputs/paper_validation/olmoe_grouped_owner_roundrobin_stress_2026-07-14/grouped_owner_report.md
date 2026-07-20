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
- EP sizes: `[8]`; mappings: `['round_robin']`
- peer tile: `64` present grouped vectors
- exact per-tile FP8 target: `0.5` (implemented as `floor(n*f)`)
- patched full exact: `True`
- grouped EP=1 exact: `True`

## Results

| ep_size | mapping | strategy | n_over_m | collision_pair_fraction | observed_high_fraction | wire_saving_vs_grouped_bf16 | local_relative_mse | mean_token_kl_vs_grouped_bf16 | ppl_delta_vs_grouped_bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | round_robin | grouped_bf16 | 1.47502 | 0.322044 | 0 | 0 | 0 | 0 | 0 |
| 8 | round_robin | uniform_fp8 | 1.47448 | 0.321795 | 1 | 0.499023 | 0.000413635 | 0.00309593 | 0.0117709 |
| 8 | round_robin | mixed_rank | 1.47497 | 0.322019 | 0.498561 | 0.617038 | 0.00234163 | 0.00662106 | 0.0307288 |
| 8 | round_robin | mixed_gate_mass | 1.4743 | 0.321711 | 0.498572 | 0.617035 | 0.00215233 | 0.0063949 | 0.0332864 |
| 8 | round_robin | mixed_pair_contribution | 1.4746 | 0.321851 | 0.498528 | 0.617046 | 0.00167327 | 0.00535794 | 0.0149033 |
| 8 | round_robin | mixed_contribution | 1.47479 | 0.321938 | 0.498577 | 0.617034 | 0.00164953 | 0.00540806 | 0.0186419 |
| 8 | round_robin | global_contribution | 1.47498 | 0.322024 | 0.499826 | 0.61674 | 0.00139953 | 0.00482964 | 0.0362622 |
| 8 | round_robin | mixed_qerr | 1.47429 | 0.321709 | 0.498543 | 0.617042 | 0.00161325 | 0.00540375 | 0.030268 |
| 8 | round_robin | mixed_random | 1.47433 | 0.321726 | 0.498545 | 0.617042 | 0.00716377 | 0.0257683 | 0.125093 |

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
