# D10 Shape-Lane C=8 最小正确性 Gate

**状态**：`COMPLETE`  
**判定**：`CONTINUE_D10_FIXED_C8_CORRECTNESS_ONLY_COST_AND_SERVING_GATES_PENDING`  
**论文结果**：否；这是 mechanism correctness Gate。  
**GPU**：单张 NVIDIA GeForce RTX 5090，BF16 eager，94.22 秒。

## 本轮唯一问题

> 对已经确认存在 M1/M64 raw delta 的 victim-layer cells，把目标 token 的 top-8 expert contributions 都放入固定 `C=8` 的 expert-row lane 后，同一 focal row 是否还会随真实 companion 身份、row slot、零填充或 expert 调用顺序而改变？

## 冻结设计

- Target：上一轮 32 个 shape-sensitive contributions 机械去重得到 21 个 victim-layer cells、16 个文档。
- Companion donor：与 target 文档不重叠的 doc016..031，每个 512 tokens；只扩充同 layer/expert 的 companion pool，不参与 target 或结果选择。
- 每个 rank 的四种 context：三组互不重复的 7 个真实 companion，focal slot 分别为 0/3/7；另加 7 个 zero padding、slot 5。
- 每个 context 3 次重复，并轮转 8 个 expert 的调用顺序。
- 既有 M1/M64 raw hash 必须在当前栈复现；否则不是负结果而是 integrity failure。

## 结果

| 检查 | 结果 |
|---|---:|
| eligible cells / victims | 21 / 16 |
| rejected cells | 0 |
| 既有 M1/M64 shape-sensitive hash 复现 | 32 / 32 |
| context × rank × repeat | 4 × 8 × 3 / cell |
| raw BF16 mismatch cells | **0 / 21** |
| post-combine / downstream route / final-logit mismatch cells | **0 / 21** |
| 独立重算 | PASS |

run01 保持 `INVALID_INSUFFICIENT_COMPANIONS_OR_INTEGRITY_FAILURE`：只用 16-token target windows 时仅 4/21 cells 满足每 rank 21 个不同 companion。run02 没有降低门槛、改 `C`、改 slot 或改判定，只增加 document-disjoint companion donor，因而解决的是预检数据不足，不是结果后调参。

## 结论边界

- **Observed**：在这 21 个富集 cell 上，固定 `C=8` 足以让 focal expert outputs 对 companion、slot、zero padding 和调用顺序保持 bitwise invariant；注入这些 outputs 后，combine、后续路由和最终 logits 也完全一致。
- **Inferred**：fixed-shape capsule 可作为 D10 shape-lane scheduler 的正确性 primitive；它没有依赖失败的 MaxGate rank selector。
- **未验证**：自然 continuous-decode 发生率、lane 等待/填充开销、端到端 TPOT/P99/goodput、相对 vLLM Batch Invariance 的收益、跨模型/跨 GPU、EP/NCCL/RDMA。
- side-call 计时只用于执行记录，不是可比较 latency baseline，不据此声称加速。

## 决策

D10 从“文献支持的假设”推进为“**通过单卡最小正确性 Gate 的方法候选**”。下一道且唯一需要的 Gate 是 continuous-decode 下的成本 Pareto：同 zero-divergence 标准下比较 native、serial M1、vLLM global Batch Invariance 与 `C=8` lane；在该 Gate 前不实现 selector、bandit 或更复杂 controller。

