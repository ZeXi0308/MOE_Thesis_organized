# EP Congestion Trace-Replay Report

## Boundary

This report reconstructs combine traffic as `expert_owner_rank -> token_origin_rank` and models repeated MoE layers plus concurrent MoE requests/replicas. It is an analytical trace replay, not measured GPU latency or queueing.

## Configuration

- model: `jamesdborin/tiny-mixtral`
- EP size: `2`
- GPUs per node: `1`
- inter-node bandwidth: `400.0 Gbps`
- concurrent jobs: `[1, 2, 4]`
- origin modes: `['balanced', 'hotspot']`
- selective tail budget: `0.50` of already-safe tail pairs

## Largest-concurrency comparison

| origin_mode | policy | payload_saving_vs_bf16 | remote_wire_saving_vs_bf16 | sum_layer_bottleneck_us | bottleneck_saving_vs_fp8 | mean_layer_receiver_imbalance |
|---|---|---|---|---|---|---|
| balanced | uniform_fp8 | 0.500000 | 0.500000 | 1.515520 | 0.000000 | 1.044071 |
| balanced | rank_tail_all | 0.625000 | 0.584507 | 1.249280 | 0.175676 | 1.031250 |
| balanced | tail_budget_random | 0.562500 | 0.549296 | 1.361920 | 0.101351 | 1.039353 |
| balanced | tail_budget_profile_ports | 0.562500 | 0.584507 | 1.249280 | 0.175676 | 1.031250 |
| balanced | tail_budget_scheduler_receiver | 0.562500 | 0.584507 | 1.249280 | 0.175676 | 1.031250 |
| balanced | tail_budget_greedy_ports | 0.562500 | 0.584507 | 1.249280 | 0.175676 | 1.031250 |
| hotspot | uniform_fp8 | 0.500000 | 0.500000 | 1.331200 | 0.000000 | 1.015625 |
| hotspot | rank_tail_all | 0.625000 | 0.625000 | 1.024000 | 0.230769 | 1.041667 |
| hotspot | tail_budget_random | 0.562500 | 0.568359 | 1.157120 | 0.130769 | 1.022690 |
| hotspot | tail_budget_profile_ports | 0.562500 | 0.625000 | 1.024000 | 0.230769 | 1.041667 |
| hotspot | tail_budget_scheduler_receiver | 0.562500 | 0.625000 | 1.024000 | 0.230769 | 1.041667 |
| hotspot | tail_budget_greedy_ports | 0.562500 | 0.625000 | 1.024000 | 0.230769 | 1.041667 |

## Policy interpretation

- `rank_tail_all`: all fixed tail ranks use INT4; this has more payload reduction than selective policies.
- `tail_budget_random/profile/scheduler/greedy`: use the same limited INT4 count and only change where that safe budget lands.
- `profile_ports` is deployable only if calibration transfers.
- `scheduler_receiver` assumes the request scheduler knows active token-owner counts.
- `greedy_ports` uses current-window sender/receiver loads and is a trace-level upper-bound probe, not a deployable online algorithm yet.

The optimization question suggested by this simulation is therefore broader than receiver-only: allocate a safe tail-INT4 budget to the current critical sender/receiver port while preserving a regular rank-segmented kernel.
