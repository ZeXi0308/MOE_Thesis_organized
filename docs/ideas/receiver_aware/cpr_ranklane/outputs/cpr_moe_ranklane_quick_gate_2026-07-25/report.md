# CPR-MoE RankLane 5090 快验结果

- 决策：`NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20`
- 冻结主门槛：跨模型 AND；raw BF16 exposed return fraction <= 0.20；相对 uniform FP8 的端到端改善 >= 5.00%
- 最有利假设：zero codec, launch, queueing, and metadata overhead
- RankLane 假设状态：`FALSIFIED_WITHIN_FROZEN_DOMAIN`
- P1 回传路径存在性：`NOT_TESTED_REQUIRES_8XA100`

## 主结果

| 模型 | uniform FP8 节省 | 最乐观 RankLane | 节省 | p=20% 零 codec E2E 上界 | 达到 5% 所需 p | PASS |
|---|---:|---|---:|---:|---:|---|
| olmoe | 50.00% | `fp8top2_rest_int4` | 68.75% | 4.17% | 23.53% | FAIL |
| llmjp | 50.00% | `fp8top4_rest_int4` | 68.75% | 4.17% | 23.53% | FAIL |

## Codec 旁证（不参与主门槛）

- 既有 RTX 5090 FP8→INT4 增量 codec gate：0/8 可行。
- p95 是否为独立样本统计：`false`。因此这里只采用 p50 方向性旁证。
- 边界：single-GPU Triton pack/unpack plus pinned H2D; analytic wire time at given Gbps; not NCCL/RDMA; no incast, collective headers, or multi-node queueing

## 裁决解释

本门槛已经允许任意 RankLane tail policy，并把 codec/launch/queueing/metadata 成本全部设为零。若该上界仍低于 5%，加入质量约束或真实执行开销不可能把它救回。该 FAIL 只否定冻结域内的固定 RankLane 执行器；不否定 P1 回传路径可能存在，也不替代 8×A100 的真实 EP profiling。

## 证据边界

This is a deterministic upper-bound synthesis of prior single-GPU artifacts. It is not a new GPU run, EP/NCCL/RDMA measurement, decode SLO result, or proof that the return path is exposed in production.
