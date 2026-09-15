# RouteShare-VTC Gate 0 Phase 2 冻结协议

状态：**FROZEN FOR IMPLEMENTATION / NO GPU RESULT / NO SCIENTIFIC GO-NO-GO**  
冻结日期：2026-07-23

## 1. Scientific question

在单 GPU MoE expert stage 中，batch 的真实执行成本是否因多租户 route coalition 而显著变化，并且这种变化是否超出 `total rows + active experts` 强简单模型可解释的范围？

若答案是否定的，则停止 RouteShare-VTC，不进入 cost sharing、公平调度、虚拟 EP 或网络拓扑扩展。

## 2. Evidence boundary

本实验是 RTX 5090 上单层、真实 BF16 expert MLP 的 executable oracle。执行链包括 route mask/dispatch、逐 expert 三投影 MLP、gate weighting 和 combine/index-add。它不是 continuous decode、vLLM、NCCL/RDMA、真实多租户服务或生产公平性结果。

## 3. 模型和层

- OLMoE：`allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`；
- LLM-jp：`llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055`；
- 主层：`layer 0`；
- 支持层：`layer 15`；
- dtype：BF16；不使用 fake quant、FP8 或 INT4。

## 4. Coalition workload

每个 scenario 有两个 tenant。固定每 tenant token 数和模型原生 top-k，生成合法的无 token 内重复 expert route。

变量：

- tokens/tenant：`{8, 16, 32, 64}`；
- tenant expert-pool overlap：`{0.0, 0.5, 1.0}`；
- histogram regime：`{balanced, skewed}`；
- 每 cell 4 个独立 route/activation seed；
- calibration seeds：`0..3`；sealed seeds：`100..103`；
- 每 tenant pool size：`min(2*top_k, floor(num_experts/(2-overlap)))`；因此 overlap=0 仍为不相交的 E/2 pools，overlap 增大时可使 k=E/2 模型产生非退化 histogram 干预；
- routing weight：每 token top-k 内均匀，避免 gate value 成为混杂。

所有 scenario 保存 route identity、row histogram、active-expert union、tenant-local histogram 和 SHA-256。

## 5. Executable arms

1. `coalition`：两个 tenant 合并后执行一次完整 expert stage；
2. `tenant_separate`：两个 tenant 分别执行同一 expert stage，时间求和；
3. `identity_relabel_sham`：在 `tenant_separate` 执行器中只交换 tenant identity，不改变 aggregate route histogram；其 latency 必须与 `tenant_separate` 等价。

数值闭合：coalition 输出必须与两个 separate 输出按 token 顺序拼接后相等；`max_abs <= 2e-2` 且 `max_rel <= 2e-2`。失败则 artifact `INVALID_EXECUTOR`。

## 6. Measurement

- GPU exact name：`NVIDIA GeForce RTX 5090`；
- 开始与结束记录 UUID、driver、CUDA、PyTorch、clock、power、memory 和 compute apps；
- 禁止 foreign compute process；
- 每 arm 5 次 warmup；
- 每 scenario 30 个 paired trials；
- 每个 trial 内部重复到累计 CUDA-event 时间至少 100 ms；
- arm 顺序由 scenario hash 交替；
- 每 trial 完整 synchronize；
- 主值为 independent scenario 的 median latency，不把 inner repeat 当独立样本。

## 7. Baselines

- `M0 rows`：总 routed rows；
- `M1 rows+active`：总 rows + active expert 数；
- `M2 row-bin`：M1 + 各 expert row-count bin 的 expert 数，bins=`1,2,3-4,5-8,9-16,17-32,33-64,65-128,>=129`；
- `isolated sum`：两个 tenant 单独执行时间之和；
- per-cell calibration mean；
- identity-relabel sham。

所有回归系数只在 calibration scenarios 拟合；sealed 只评估一次。禁止在 sealed 选择特征、层、模型或阈值。

## 8. Primary metrics

1. `M1` sealed absolute relative error；
2. matched `total_rows + active_experts`、不同 histogram scenario 的 latency contrast；
3. `M2` 相对 `M1` 的 held-out squared-error gap recovery；
4. `C(A union B) / (C(A)+C(B))`；
5. sham relative difference。

bootstrap 以 scenario 为单位，2,000 次；模型、层分别报告。不做事后挑选：两模型×两层的四个 cell 必须各自达到同一预注册阈值，才宣告 GO。

## 9. Gate

主模型 OLMoE 与支持模型 LLM-jp 均需满足：

1. matched-histogram contrast 的 median absolute relative difference 95% LCB `>= 10%`；
2. `M1` 不能解释 `>=90%` 的 token-only executable-oracle gap；
3. `M2` 相对 `M1` 的 sealed gap recovery LCB `>=30%`；
4. sham latency relative difference 95% UCB `<=3%`；
5. executor numerical closure 全部通过。

任一模型失败：`NO_GO_ROUTE_COALITION_COST`，不得换 Shapley、bandit、predictor、虚拟链路或公平指标抢救。

## 10. GPU 后续边界

只有 Gate 0 全部通过，才允许 Phase 2B 设计 delayed virtual service、公平 replay 和 topology-aware cost extension。单卡 Gate 0 通过仍不能声称多 GPU receiver、RDMA 或 production tenant isolation。
