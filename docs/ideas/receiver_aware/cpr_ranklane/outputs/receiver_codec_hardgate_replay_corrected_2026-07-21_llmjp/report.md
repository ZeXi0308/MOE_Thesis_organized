# Receiver Codec Hard-Gate Offline Replay (corrected accounting)

## Evidence boundary

offline replay of recorded policy action traces with Phase-A measured pack+unpack+h2d tax; analytic wire times from baseline/policy bottleneck bytes (FP8 high → INT4 low); not real NCCL/RDMA latency; H2D is a pessimistic host-staging bound

## Verdict (primary = once_per_step / fused kernel)

- block_rate_among_low (once_per_step): **25.5%**
- median orig_net_p50 (once_per_step): **106.181 µs**
- block_rate_among_low (serialized_tiles, scaled): **100.0%**
- conclusion: **MIXED_FUSED: see per-file table; do not expand online controller yet**

## Per-file summary

| arm | tax_mode | n_low | blocked | block_rate | orig_net_p50 | hardgate_net_p50 | unblocked_acted_net_med |
|---|---|---:|---:|---:|---:|---:|---:|
| controller_balanced | once_per_step | 30 | 6 | 20.0% | 15.329 | 15.329 | 16.435 |
| controller_balanced | serialized_tiles | 30 | 30 | 100.0% | -21801.262 | 0.000 | nan |
| controller_hotspot | once_per_step | 31 | 0 | 0.0% | 188.339 | 188.339 | 188.836 |
| controller_hotspot | serialized_tiles | 31 | 31 | 100.0% | -16233.454 | 0.000 | nan |
| causal_no_hysteresis_balanced | once_per_step | 31 | 26 | 83.9% | -9.814 | 0.000 | 5.917 |
| causal_no_hysteresis_balanced | serialized_tiles | 31 | 31 | 100.0% | -15354.137 | 0.000 | nan |
| causal_no_hysteresis_hotspot | once_per_step | 31 | 0 | 0.0% | 188.339 | 188.339 | 188.836 |
| causal_no_hysteresis_hotspot | serialized_tiles | 31 | 31 | 100.0% | -16247.334 | 0.000 | nan |
| calib_static_balanced | once_per_step | 32 | 32 | 100.0% | -37.825 | 0.000 | nan |
| calib_static_balanced | serialized_tiles | 32 | 32 | 100.0% | -9339.369 | 0.000 | nan |
| calib_static_hotspot | once_per_step | 32 | 0 | 0.0% | 189.471 | 189.471 | 189.471 |
| calib_static_hotspot | serialized_tiles | 32 | 32 | 100.0% | -14653.322 | 0.000 | nan |
| uniform_low_balanced | once_per_step | 32 | 0 | 0.0% | 24.023 | 24.023 | 24.023 |
| uniform_low_balanced | serialized_tiles | 32 | 32 | 100.0% | -22573.020 | 0.000 | nan |
| uniform_low_hotspot | once_per_step | 32 | 0 | 0.0% | 189.481 | 189.481 | 189.481 |
| uniform_low_hotspot | serialized_tiles | 32 | 32 | 100.0% | -22497.303 | 0.000 | nan |

## Accounting note

- Online policy baseline is FP8 high; low action is INT4 → lookup `homo_int4` by default.
- `serialized_tiles` uses `tiles * unit * (tile_rows / measured_rows)`.
- Primary go/no-go uses `once_per_step`; scaled serialized is sensitivity.
