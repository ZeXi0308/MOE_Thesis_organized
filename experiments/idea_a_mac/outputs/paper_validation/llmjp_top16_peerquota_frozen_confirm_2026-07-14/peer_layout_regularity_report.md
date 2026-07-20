# Peer-Local Fixed-Quota Layout Regularity

- routing trace: `experiments/idea_a_mac/outputs/paper_validation/llmjp_top16_peerquota_frozen_confirm_2026-07-14/test_routes.csv`
- group semantics: one local-origin replay, expert ids mapped to 8 synthetic owner groups by `contiguous`
- expert count used for mapping: 32 (observed max + 1: 32)
- calibrated gate threshold: 0.04711914
- target low-bit fraction: 0.5000
- fixed-quota tile: 64 routed pairs

## Lane-count variability

| level | units | pairs | threshold_low_fraction_weighted | threshold_low_fraction_p01 | threshold_low_fraction_p05 | threshold_low_fraction_p50 | threshold_low_fraction_p95 | threshold_low_fraction_p99 | mean_abs_lane_count_deviation | p95_abs_lane_count_deviation | any_lane_overflow_fraction | low_lane_overflow_fraction | high_lane_overflow_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| peer_tile_64 | 65296 | 3932160 | 0.498815 | 0.1875 | 0.28125 | 0.5 | 0.71875 | 0.827613 | 6.01979 | 16 | 0.940486 | 0.47228 | 0.468206 |
| peer_message | 7680 | 3932160 | 0.498815 | 0.228162 | 0.324498 | 0.509526 | 0.673922 | 0.747935 | 40.6391 | 107 | 0.993359 | 0.533724 | 0.459635 |

`any_lane_overflow_fraction` is the fraction of units whose threshold-selected
FP8/low-bit counts do not fit a preallocated exact-50% two-lane split.
It is a layout regularity statistic, not a measured allocation or latency cost:
a threshold implementation can recover exact sizes with scans/count exchange.

## Metadata estimate

| hidden_size | tile_pairs | membership_mask_bytes | mask_overhead_fraction_of_payload | optional_fp16_gate_bytes | gate_overhead_fraction_of_payload |
|---|---|---|---|---|---|
| 512 | 64 | 8 | 0.000325521 | 128 | 0.00520833 |
| 2048 | 64 | 8 | 8.13802e-05 | 128 | 0.00130208 |

The fixed quota still needs a membership mask because membership is dynamic; it
only removes variable lane cardinality.  Contribution selection may also need a
gate scalar at the expert owner if the dispatch protocol does not already carry
one.  Quantization scales, headers, alignment, scans, pack/unpack, and collective
startup are outside this estimate and require a real EP kernel benchmark.

## Evidence boundary

This trace is sufficient to test fixed-cardinality layout invariants and gate
lane-count variability.  It does not contain real token-origin ranks, queues,
RDMA/NVLink traffic, or timestamps, so it cannot establish TTFT, TPOT, TBT, P99,
or topology-aware benefit.
