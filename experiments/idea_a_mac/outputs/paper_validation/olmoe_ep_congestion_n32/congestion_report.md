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
| balanced | uniform_fp8 | 0.500000 | 0.500000 | 1229.209600 | 0.000000 | 1.179250 |
| balanced | rank_tail_all | 0.625000 | 0.624431 | 946.257920 | 0.230190 | 1.185650 |
| balanced | tail_budget_random | 0.562500 | 0.561968 | 1085.931520 | 0.116561 | 1.182744 |
| balanced | tail_budget_profile_ports | 0.562500 | 0.623020 | 946.257920 | 0.230190 | 1.181298 |
| balanced | tail_budget_scheduler_receiver | 0.562500 | 0.623020 | 947.691520 | 0.229024 | 1.181298 |
| balanced | tail_budget_greedy_ports | 0.562500 | 0.623020 | 946.257920 | 0.230190 | 1.181298 |
| hotspot | uniform_fp8 | 0.500000 | 0.500000 | 1537.310720 | 0.000000 | 1.812571 |
| hotspot | rank_tail_all | 0.625000 | 0.624799 | 1165.271040 | 0.242007 | 1.816912 |
| hotspot | tail_budget_random | 0.562500 | 0.562282 | 1352.949760 | 0.119924 | 1.816397 |
| hotspot | tail_budget_profile_ports | 0.562500 | 0.622085 | 1165.271040 | 0.242007 | 1.803588 |
| hotspot | tail_budget_scheduler_receiver | 0.562500 | 0.622085 | 1165.271040 | 0.242007 | 1.803588 |
| hotspot | tail_budget_greedy_ports | 0.562500 | 0.622085 | 1165.271040 | 0.242007 | 1.803588 |

## Policy interpretation

- `rank_tail_all`: all fixed tail ranks use INT4; this has more payload reduction than selective policies.
- `tail_budget_random/profile/scheduler/greedy`: use the same limited INT4 count and only change where that safe budget lands.
- `profile_ports` is deployable only if calibration transfers.
- `scheduler_receiver` assumes the request scheduler knows active token-owner counts.
- `greedy_ports` uses current-window sender/receiver loads and is a trace-level upper-bound probe, not a deployable online algorithm yet.

The optimization question suggested by this simulation is therefore broader than receiver-only: allocate a safe tail-INT4 budget to the current critical sender/receiver port while preserving a regular rank-segmented kernel.
