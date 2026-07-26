# RouteShare Gate 0 快速复现：Review 与结果

日期：2026-07-23  
GPU：NVIDIA GeForce RTX 5090 32 GB  
证据边界：真实 BF16 expert 权重、合成可控 route histogram、单 GPU expert-stage executable oracle；不是 serving、fused MoE kernel、多卡 EP 或网络证据。

## Code Review

### 1. 执行链路

`build_plans` 生成固定 total rows / active expert count / expert identity、仅 row histogram 不同的 matched workloads。`prepare_operation` 对同一 activation pool 做 dispatch gather，执行模型真实 expert MLP，再 concatenate 和 inverse-permutation combine。CUDA Event 逐 plan 计时，block 内随机化顺序。前两个 replica 拟合 cost model，第三个 replica held out。

### 2. 已确认正确

- counts 全为正且严格求和为 total rows；matched cell 的 expert identity 不变；
- shape contrast 排除 active=1 等 histogram 实际相同的伪对照；
- 实际加载固定 revision 的 OLMoE / LLM-jp 本地权重，每层 expert 数严格检查；
- 12 blocks × 240 plans = 每模型 2,880 条，无缺失值；
- 本地复跑 5 个 invariant tests 全部通过；
- 两份 summary 的 `source_sha256` 与保存源码闭合：`9fb05be5fd80eb8abbd2c5d359f598cf52875154cb182590483e9c5b46aee7a4`。

### 3. 混杂与局限

- executor 为 Python 逐 active-expert 调用，active-count 效果包含 kernel-launch / framework overhead；不能外推 fused production kernel；
- router、attention、KV cache、continuous batching 和通信不在计时区间；
- 不同完整运行的 OLMoE shape effect 幅度有漂移，但新复现的 95% LCB 仍远低于 10% gate；
- strong model 的置信区间较宽，因此不应声称已获得可部署 latency predictor。

### 4. 修改项

必须修改项：无。  
建议修改项：下一 gate 换成目标 serving/fused executor，直接测试“最小化 union active experts”的 batch pairing，而不再扩展 histogram predictor。

### 5. Dry run / GPU 准入

CPU invariant tests：5/5 PASS。旧输出因 source hash 不闭合被弃用；当前源码独立复现后的两份输出均闭合。  
结论：**GPU Run Approved**，且已完成。

## GPU 结果

| Model | matched histogram effect | 95% CI | strong held-out R² | 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| OLMoE layer 0 | 6.42% | [3.77%, 8.00%] | 0.925 | [0.765, 0.983] | FAIL |
| LLM-jp layer 0 | 1.65% | [0.88%, 1.80%] | 0.939 | [0.788, 0.979] | FAIL |

预注册的 proceed 条件是 histogram effect 95% LCB `>=10%` 且 strong R² 95% UCB `<0.90`。两模型均失败，因此“复杂 row-histogram 中仍有大量不可解释 RouteShare 成本”的强版本为 **NO-GO**。

同时，固定 total rows 时，active experts 从 1 增加到 16 使该 executor 的中位 expert-stage latency 增加约 `8.1–8.7×`。这只支持更简单的后续问题：在目标 executor 上，按 expert-set overlap 组 batch 能否通过减少 union active experts 获得稳定净收益。它不支持直接进入网络拓扑或公平调度 claim。
