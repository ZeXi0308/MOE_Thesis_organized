# EP Congestion Trace-Replay Report

## Boundary

This report reconstructs combine traffic as `expert_owner_rank -> token_origin_rank` and models repeated MoE layers plus concurrent MoE requests/replicas. It is an analytical trace replay, not measured GPU latency or queueing.

## Configuration

- model: `allenai/OLMoE-1B-7B-0924`
- EP size: `8`
- GPUs per node: `4`
- inter-node bandwidth: `200.0 Gbps`
- concurrent jobs: `[1, 2, 4, 8, 16]`
- origin modes: `['balanced', 'hotspot']`
- selective tail budget: `0.50` of already-safe tail pairs

## Largest-concurrency comparison

| origin_mode | policy | payload_saving_vs_bf16 | remote_wire_saving_vs_bf16 | sum_layer_bottleneck_us | bottleneck_saving_vs_fp8 | mean_layer_receiver_imbalance |
|---|---|---|---|---|---|---|
| balanced | uniform_fp8 | 0.500000 | 0.500000 | 650.117120 | 0.000000 | 1.531125 |
| balanced | rank_tail_all | 0.625000 | 0.624274 | 509.870080 | 0.215726 | 1.536688 |
| balanced | tail_budget_random | 0.562500 | 0.561861 | 579.829760 | 0.108115 | 1.532817 |
| balanced | tail_budget_profile_ports | 0.562500 | 0.621038 | 509.870080 | 0.215726 | 1.523315 |
| balanced | tail_budget_scheduler_receiver | 0.562500 | 0.621038 | 509.870080 | 0.215726 | 1.523315 |
| balanced | tail_budget_greedy_ports | 0.562500 | 0.621038 | 509.870080 | 0.215726 | 1.523315 |
| hotspot | uniform_fp8 | 0.500000 | 0.500000 | 1456.209920 | 0.000000 | 4.066001 |
| hotspot | rank_tail_all | 0.625000 | 0.624396 | 1096.417280 | 0.247075 | 4.072026 |
| hotspot | tail_budget_random | 0.562500 | 0.561817 | 1278.238720 | 0.122215 | 4.072200 |
| hotspot | tail_budget_profile_ports | 0.562500 | 0.619870 | 1096.417280 | 0.247075 | 4.024231 |
| hotspot | tail_budget_scheduler_receiver | 0.562500 | 0.619870 | 1096.417280 | 0.247075 | 4.024231 |
| hotspot | tail_budget_greedy_ports | 0.562500 | 0.619870 | 1096.417280 | 0.247075 | 4.024231 |

## Policy interpretation

- `rank_tail_all`: all fixed tail ranks use INT4; this has more payload reduction than selective policies.
- `tail_budget_random/profile/scheduler/greedy`: use the same limited INT4 count and only change where that safe budget lands.
- `profile_ports` is deployable only if calibration transfers.
- `scheduler_receiver` assumes the request scheduler knows active token-owner counts.
- `greedy_ports` uses current-window sender/receiver loads and is a trace-level upper-bound probe, not a deployable online algorithm yet.

The optimization question suggested by this simulation is therefore broader than receiver-only: allocate a safe tail-INT4 budget to the current critical sender/receiver port while preserving a regular rank-segmented kernel.
