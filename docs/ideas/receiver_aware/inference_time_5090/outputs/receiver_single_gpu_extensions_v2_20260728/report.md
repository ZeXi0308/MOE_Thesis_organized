# Receiver Single-GPU Extensions on RTX 5090

- status: `SINGLE_GPU_EXTENSIONS_COMPLETE_NOT_RECEIVER_GATE`
- component breakdown valid: `TRUE`
- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`

## Coarse local MoE components (decode)

| batch_size | observer_ratio_median | moe_fraction_median | gate_fraction_median | routing_setup_fraction_median | expert_loop_fraction_median | unattributed_tail_fraction_median |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.06547 | 0.826662 | 0.00797145 | 0.0568894 | 0.748972 | 0.0128846 |
| 4 | 1.03891 | 0.889071 | 0.00515358 | 0.0356356 | 0.840022 | 0.0080935 |
| 8 | 1.03059 | 0.897454 | 0.00487285 | 0.0330258 | 0.85222 | 0.00748246 |
| 16 | 1.03203 | 0.900532 | 0.00474075 | 0.0321904 | 0.856337 | 0.00726157 |
| 32 | 1.03562 | 0.900693 | 0.00465335 | 0.0320882 | 0.856688 | 0.00722536 |

`expert_loop` includes gather, expert compute, weighting, and local index_add. It is not pure
expert GEMM and is not return-path time.

## Context-length sweep

| prompt_len | phase | batch_size | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | moe_fraction_median |
| --- | --- | --- | --- | --- | --- |
| 128 | decode | 1 | 42.9307 | 45.0882 | 0.827454 |
| 512 | decode | 1 | 43.1351 | 44.7524 | 0.829142 |
| 2048 | decode | 1 | 43.0294 | 47.8699 | 0.828523 |
| 128 | decode | 4 | 73.7503 | 77.4791 | 0.89 |
| 512 | decode | 4 | 72.5066 | 76.5125 | 0.888816 |
| 2048 | decode | 4 | 73.0359 | 77.946 | 0.889526 |
| 128 | decode | 8 | 80.2624 | 86.2945 | 0.898553 |
| 512 | decode | 8 | 78.7583 | 87.1932 | 0.897803 |
| 2048 | decode | 8 | 79.145 | 80.8112 | 0.897446 |
| 128 | prefill | 1 | 83.1806 | 83.8075 | 0.898419 |
| 512 | prefill | 1 | 86.9445 | 87.3915 | 0.899611 |
| 2048 | prefill | 1 | 90.2591 | 96.566 | 0.890515 |
| 128 | prefill | 4 | 87.3866 | 88.1931 | 0.899604 |
| 512 | prefill | 4 | 90.4902 | 94.8044 | 0.890844 |
| 2048 | prefill | 4 | 134.887 | 140.49 | 0.89117 |
| 128 | prefill | 8 | 88.3577 | 89.865 | 0.896744 |
| 512 | prefill | 8 | 106.383 | 113.195 | 0.894031 |
| 2048 | prefill | 8 | 191.538 | 193.079 | 0.876018 |

## Interleaved frozen natural text versus synthetic token sequence

| phase | batch_size | independent_sequence_pairs | synthetic_latency_median_ms | natural_latency_median_ms | natural_delta_pct_median |
| --- | --- | --- | --- | --- | --- |
| decode | 1 | 5 | 43.6577 | 43.6166 | -0.0516996 |
| decode | 4 | 5 | 73.6468 | 73.6893 | 0.057785 |
| decode | 8 | 5 | 79.3293 | 80.4873 | 1.13011 |
| prefill | 1 | 5 | 83.3015 | 83.8398 | 0.214072 |
| prefill | 4 | 5 | 86.4577 | 86.8437 | 0.3086 |
| prefill | 8 | 5 | 87.045 | 88.2227 | 0.0872876 |

These experiments use one GPU and eager local MoE execution. They contain no EP ranks, NCCL
return collective, receiver queue, or RankLane implementation, so they do not answer the formal
Receiver existence or benefit Gate.
