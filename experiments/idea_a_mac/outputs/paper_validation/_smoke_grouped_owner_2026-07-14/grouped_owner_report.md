# Grouped-Owner Combine Offline Experiment

This experiment uses the DeepEP combine wire unit: selected expert outputs on
the same owner are multiplied by router weights and reduced in BF16 before fake
FP8/MXFP4. It is a quality and trace-structure experiment, not a kernel or
network benchmark.

## Configuration

- model: `jamesdborin/tiny-mixtral`
- data: `wikitext2_docs:test`, offset `0`, n=`1`
- sequence length: `32`
- EP sizes: `[2]`; mappings: `['contiguous']`
- peer tile: `8` present grouped vectors
- exact per-tile FP8 target: `0.5` (implemented as `round(n*f)`)
- patched full exact: `True`
- grouped EP=1 exact: `True`

## Results

| ep_size | mapping | strategy | n_over_m | collision_pair_fraction | observed_high_fraction | wire_saving_vs_grouped_bf16 | local_relative_mse | mean_token_kl_vs_grouped_bf16 | ppl_delta_vs_grouped_bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | contiguous | grouped_bf16 | 1.10345 | 0.09375 | 0 | 0 | 0 | 0 | 0 |
| 2 | contiguous | uniform_fp8 | 1.09402 | 0.0859375 | 1 | 0.498047 | 0.000705048 | 0.000699412 | 470.922 |
| 2 | contiguous | uniform_mxfp4 | 1.07563 | 0.0703125 | 0 | 0.734375 | 0.0137229 | 0.00513574 | 1432.11 |
| 2 | contiguous | mixed_rank | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.006392 | 0.00218771 | 1148.52 |
| 2 | contiguous | mixed_gate_mass | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.00576741 | 0.00213082 | 1208.1 |
| 2 | contiguous | mixed_inputnorm_gate | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.00576741 | 0.00213082 | 1208.1 |
| 2 | contiguous | mixed_contribution | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.00575541 | 0.00214258 | 1289.02 |
| 2 | contiguous | mixed_qerr | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.00572072 | 0.00213337 | 1430.21 |
| 2 | contiguous | mixed_oracle | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.00572502 | 0.00212062 | 1274.22 |
| 2 | contiguous | mixed_random | 1.08475 | 0.078125 | 0.5 | 0.616211 | 0.00724702 | 0.00377154 | 1496.98 |

## Metric definitions and boundaries

- `N` is the number of routed expert pairs; `M` is the number of nonempty
  token-owner vectors after owner-local reduction. `N/M` and the collision
  fractions characterize existing local combine aggregation; they are not a
  new compression saving.
- `grouped_bf16` is the precision reference for EP>1. Its logit drift from the
  original single-process path is reported because owner grouping changes BF16
  associativity. Quantized KL is therefore reported against both references.
- Mixed selectors get exactly the same `round(n*f)` FP8 cardinality within each
  present-vector tile of each owner. Total counts can deviate slightly from
  `f*M` because final short tiles are rounded independently.
- `mixed_rank` uses a gate/output-free sum of reciprocal routed ranks.
  `mixed_gate_mass` uses the owner-local sum of router weights.
  `mixed_inputnorm_gate` multiplies that mass by the origin-available input norm.
  `mixed_contribution` uses the norm of the already reduced wire vector.
  `mixed_qerr` uses its fake-MXFP4 error energy.
- `mixed_oracle` is an exact *one-owner-group* local FP4-to-FP8 intervention
  score that includes same-token cross-owner terms. Multiple simultaneous
  upgrades still interact, so it is not a global combinatorial optimum.
- Wire accounting includes the fake formats' vector/block scale bytes but omits
  alignment, padding, membership masks, headers, packing, and GPU kernel cost.
- Upstream approximation can change later-layer routes. A formal run should add
  disjoint datasets, larger models, native FP4/FP8 kernels, and real EP traces.
