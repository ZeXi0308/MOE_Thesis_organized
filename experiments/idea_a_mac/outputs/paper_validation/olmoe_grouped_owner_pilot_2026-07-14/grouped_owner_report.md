# Grouped-Owner Combine Offline Experiment

This experiment uses the DeepEP combine wire unit: selected expert outputs on
the same owner are multiplied by router weights and reduced in BF16 before fake
FP8/MXFP4. It is a quality and trace-structure experiment, not a kernel or
network benchmark.

## Configuration

- model: `allenai/OLMoE-1B-7B-0924`
- data: `wikitext2_docs:test`, offset `48`, n=`1`
- sequence length: `64`
- EP sizes: `[8]`; mappings: `['contiguous']`
- peer tile: `32` present grouped vectors
- exact per-tile FP8 target: `0.5` (implemented as `round(n*f)`)
- patched full exact: `True`
- grouped EP=1 exact: `True`

## Results

| ep_size | mapping | strategy | n_over_m | collision_pair_fraction | observed_high_fraction | wire_saving_vs_grouped_bf16 | local_relative_mse | mean_token_kl_vs_grouped_bf16 | ppl_delta_vs_grouped_bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | contiguous | grouped_bf16 | 1.47047 | 0.319946 | 0 | 0 | 0 | 0 | 0 |
| 8 | contiguous | uniform_fp8 | 1.46889 | 0.319214 | 1 | 0.499023 | 0.000289628 | 0.00500721 | 0.202238 |
| 8 | contiguous | uniform_mxfp4 | 1.47338 | 0.321289 | 0 | 0.734375 | 0.0114796 | 0.0553952 | 0.589301 |
| 8 | contiguous | mixed_rank | 1.471 | 0.32019 | 0.500269 | 0.616636 | 0.00122418 | 0.00801944 | -0.0225108 |
| 8 | contiguous | mixed_gate_mass | 1.47021 | 0.319824 | 0.498923 | 0.616953 | 0.00123361 | 0.00924157 | -0.0611825 |
| 8 | contiguous | mixed_inputnorm_gate | 1.46757 | 0.318604 | 0.499463 | 0.616826 | 0.00125617 | 0.0109314 | -0.171237 |
| 8 | contiguous | mixed_contribution | 1.47021 | 0.319824 | 0.499462 | 0.616826 | 0.000938319 | 0.00694367 | -0.120084 |
| 8 | contiguous | mixed_qerr | 1.47126 | 0.320312 | 0.50018 | 0.616657 | 0.000920899 | 0.00690707 | 0.167101 |
| 8 | contiguous | mixed_oracle | 1.46942 | 0.319458 | 0.499193 | 0.616889 | 0.000960559 | 0.0069406 | -0.0713588 |
| 8 | contiguous | mixed_random | 1.471 | 0.32019 | 0.499192 | 0.616889 | 0.00432108 | 0.052223 | -0.104298 |

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
