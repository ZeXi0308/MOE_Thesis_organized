# Coarse Local MoE Component Breakdown on RTX 5090

- model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`
- revision: `1d5983076dfc67aee4a77ec06a27027f5bab6055`
- MoE blocks: `16`
- quantitative breakdown valid: `TRUE`
- maximum decode observer tax: `6.55%` (limit `10%`)
- receiver congestion: `NOT_TESTED_NO_EP_TRAFFIC`

| phase | batch_size | unprofiled_n | unprofiled_latency_median_ms | unprofiled_latency_p95_ms | breakdown_latency_median_ms | observer_ratio_median | observer_tax_acceptable | moe_total_median_ms | gate_median_ms | routing_setup_median_ms | expert_loop_median_ms | unattributed_tail_median_ms | moe_fraction_median | gate_fraction_median | routing_setup_fraction_median | expert_loop_fraction_median | unattributed_tail_fraction_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 48 | 43.0943 | 43.9865 | 45.9155 | 1.06547 | True | 37.9648 | 0.36648 | 2.61304 | 34.3973 | 0.592176 | 0.826662 | 0.00797145 | 0.0568894 | 0.748972 | 0.0128846 |
| decode | 4 | 48 | 73.9364 | 76.9004 | 76.8132 | 1.03891 | True | 68.1683 | 0.391968 | 2.73102 | 64.4219 | 0.621215 | 0.889071 | 0.00515358 | 0.0356356 | 0.840022 | 0.0080935 |
| decode | 8 | 48 | 80.906 | 83.7541 | 83.381 | 1.03059 | True | 74.8528 | 0.406624 | 2.75138 | 71.0678 | 0.623456 | 0.897454 | 0.00487285 | 0.0330258 | 0.85222 | 0.00748246 |
| decode | 16 | 48 | 83.2966 | 87.5275 | 85.9643 | 1.03203 | True | 77.4599 | 0.407872 | 2.76075 | 73.6256 | 0.624528 | 0.900532 | 0.00474075 | 0.0321904 | 0.856337 | 0.00726157 |
| decode | 32 | 48 | 83.2231 | 84.3902 | 86.1872 | 1.03562 | True | 77.5952 | 0.401968 | 2.76522 | 73.8082 | 0.624912 | 0.900693 | 0.00465335 | 0.0320882 | 0.856688 | 0.00722536 |
| prefill | 1 | 3 | 83.4782 | 83.9402 | 86.4175 | 1.03521 | True | 77.5757 | 0.406848 | 2.77715 | 73.7485 | 0.643264 | 0.897686 | 0.00470794 | 0.0321236 | 0.853398 | 0.00744368 |
| prefill | 4 | 3 | 87.2358 | 88.3285 | 90.0685 | 1.03247 | True | 80.8582 | 0.421056 | 2.91741 | 76.91 | 0.648 | 0.897742 | 0.00467484 | 0.0322315 | 0.853905 | 0.00719291 |
| prefill | 8 | 3 | 87.8487 | 89.6701 | 90.7064 | 1.03253 | True | 81.1707 | 0.42032 | 2.91469 | 77.1992 | 0.637281 | 0.894872 | 0.00458094 | 0.032051 | 0.851102 | 0.00701657 |
| prefill | 16 | 3 | 90.3969 | 95.8247 | 93.2964 | 1.03207 | True | 82.9939 | 0.433152 | 2.94368 | 78.9667 | 0.650304 | 0.889703 | 0.00463253 | 0.0315147 | 0.846678 | 0.0068785 |
| prefill | 32 | 3 | 99.0461 | 104.725 | 101.662 | 1.02641 | True | 90.9412 | 0.41984 | 2.95994 | 87.0344 | 0.53008 | 0.894592 | 0.00412977 | 0.0289351 | 0.856281 | 0.00524127 |

`routing_setup` contains softmax/top-k/normalization/allocation/one-hot/active-expert discovery.
`expert_loop` contains gather, expert compute, weighting, and local index_add. It is not a pure
expert-GEMM or return-all-to-all measurement. Rows fail closed when decode observer tax exceeds 10%.
