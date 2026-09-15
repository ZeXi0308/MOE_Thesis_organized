# CriticalSplit 负结果后的近期系统边界

**生成时间**：2026-08-10 18:11:57 +08:00  
**用途**：约束 `WEAKEN_ACTION_SPACE` 后的新机制候选；只做直接碰撞检查，不作 novelty 完成证明。  
**证据等级**：本地 sealed result 为 accepted bounded simulation；论文条目为原始论文/官方材料；新机制均为 hypothesis。

## 本地负结果边界

- `artifacts/criticalsplit_pilot/20260810_173200/`：8/8 cell 中 expanded split exact Oracle 与 whole-ready exact Oracle 的 flow 完全相同；proper/critical/bulk optimal launches 均为 0；`eligible_cells=0`；结论为 `WEAKEN_ACTION_SPACE`。
- 该结果只否定冻结 DAG、服务曲线和 Critical/Bulk proper-subset 动作；不否定新的 completion semantics、running-state transition、communication resources 或 execution shape。
- 当前 whole-ready Oracle 已经枚举所有 idle executor 上的 eligible queues 与 bounded HOLD；一般性的 queue score / EDF / cross-request ordering 不是新动作。

## 近期直接近邻

| 来源 | 已覆盖机制 | 对本轮候选的约束 |
|---|---|---|
| [AMoE / AEP](https://arxiv.org/abs/2505.08944) | layer-level asynchronous expert parallelism、token micro-queues、adaptive re-batching、defragmentation | “去 barrier + 动态 re-batching”本身不新；需要更具体的 row-completion、exact consumer 或 running-state actuator。 |
| [FinDEP](https://arxiv.org/abs/2512.21487) | 将 DEP 通信/计算切成细粒度任务并联合优化粒度与顺序 | generic chunk pipeline / comm-compute scheduling 高度拥挤；必须证明 MoE join-specific 新 action。 |
| [ExpertPlex](https://arxiv.org/abs/2607.18002) | persistent kernel 的 tile-level adaptive scheduling、attention-initiated communication、跨 phase overlap | tile scheduling 与 comm-compute overlap 不能单独作为贡献；row/token completion exposure 必须是额外系统语义。 |
| [UltraEP](https://arxiv.org/abs/2606.04101) | 每 microbatch / layer exact-load balancing、persistent tile streaming、replica communication | 动态 replica/load balance 已有强近邻；ReplicaSteal 需要不同于 load balancing 的 bounded join-tail claim。 |
| [Gimbal](https://arxiv.org/abs/2606.15177) | frontend/backend pressure 协同、queue ordering、source-aware expert placement | 一般 cross-request / expert-aware scheduling 与 placement 不新。 |
| [Speculative MoE / Semantic Parallelism](https://arxiv.org/abs/2503.04398) | 预测 token/expert 路径并预调度通信与 expert work | 一般 speculative pre-scheduling 已碰撞；新 speculation 必须限定 first-completion hedge 或其他新物理合同。 |
| [SpecPrefetch](https://arxiv.org/abs/2607.24787) | 在不改变 native route 的条件下预测下一层 expert，仅用于异步传输 | RouteAhead/next-expert prefetch 的 method novelty 很低，可作为 control。 |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | request-level activation tracing、expert prefetch/cache/offload | expert cache/prefetch 不是新意。 |
| [fMoE](https://arxiv.org/abs/2502.05370) | fine-grained expert offloading、selection pattern 与 semantic hint 驱动 prefetch/cache | offload-oriented fine granularity 已覆盖。 |
| `literature/papers/Aurora-Optimizing-MoE-Inference-Time.pdf` | deployment 与 communication scheduling 联合优化 | generic communication schedule 不足；本轮应聚焦 join/consumer 边界。 |
| `literature/papers/Lina-Accelerating-Distributed-MoE-Training-and-Inference.pdf` | micro-op scheduling、expert popularity 与 pipeline overlap | generic micro-op pipeline 有历史近邻。 |
| `literature/papers/HOBBIT-Mixed-Precision-Expert-Offloading.pdf` | mixed-precision expert offload/cache | 与本轮 exact completion 语义不同，但压缩一般 offload idea。 |

## 当前可保留的窄空隙

1. **Atomic batch completion → row/token milestone**：保持一次 whole launch、相同成员与最终 service，只让真实完成里程碑提前可见。
2. **Post-expert exact consumer**：只把可交换/可分块且算术顺序固定的 combine/router prefix 从最终 barrier 前移，不预测最终 route。
3. **Running-state action**：cooperative preemption、open/append 或 first-completion hedge；这些不能由 ready-queue 重排表达。
4. **Join-aware multi-resource contract**：通信、compute、reducer/tile consumer 的动作必须显式关闭 top-k join，并与 generic overlap 近邻区分。

## 禁区

- 不继续切 Critical/Bulk、调工作负载或换 split identity。
- 不把 AMoE 式 asynchronous re-batching、FinDEP 式 chunk scheduling、ExpertPlex 式 tile kernel 或 SpecPrefetch 式 next-expert prefetch单独包装成新贡献。
- 不把固定 replica 上换 queue score 叫作 cross-request scheduling；旧 exact Oracle 已覆盖。
- CPU Oracle 只回答 action-space existence；不得外推 GPU kernel、EP/NCCL/RDMA、online policy 或自然负载收益。
