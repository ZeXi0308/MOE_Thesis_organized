# 专家驻留税与 KV 容量悬崖：接纳控制第一次有真实约束

2026-09-08。承接 [`ROUND_20260908_SUMMARY.md`](../ROUND_20260908_SUMMARY.md)。
**本目录的 GPU 探针未执行并已主动让位；结论来自对同期另一实验已产出遥测的结构性分析。**

## 数据归属（必须先说清）

本报告的 4 个 GPU cell **不是本目录跑的**，是同期并行工作流
`moe-native-memory-pressure-20260906-r01` 的产物（其冻结设计与原始数据在该目录）。
本目录的贡献只有两项：

1. 回传其轻量遥测（`inherited_telemetry/`，不含大 raw），并做一项该实验未做的
   **专家驻留税结构量化**（`analyze_expert_residency_tax.py`）；
2. 把该量化与本仓库此前建立的捕获成本阶梯连接，定位出容量悬崖的精确位置。

## 我为什么撤下自己的探针

本目录 [`DECISIONS.md`](DECISIONS.md) 冻结的探针用 `gpu_memory_utilization=0.60`
+ 长输出（128 prompt + 1792 output）制造 KV 压力。执行前的 GPU 隔离检查 fail-closed
（检测到他人 14 GiB 进程），零污染、零 cell 产出。

随后核对并行设计后判定：**我的方案是人为制造瓶颈。** 把显存预算从常规 0.90 压到
0.60 属于 AGENTS.md 明令禁止的"人为制造瓶颈"；而并行方案用**真实 3072-token
WikiText 文档** + **常规 `util=0.90`**，是"有现实依据的新运行域"。方法论更强的方案
已在跑，再跑一个更弱的只会消耗 GPU 并制造选择性解释空间。

因此本目录探针状态为 `WITHDRAWN_IN_FAVOUR_OF_STRONGER_DESIGN`，非 BLOCKED、非失败。
代码改动（KV 容量落盘、抢占测量、长输出派生）保留复用。

## 实测的 MoE 结构性事实

来自 `memory-after-init.json`，四个 cell 完全一致：

| 量 | 实测值 | 说明 |
|---|---:|---|
| 参数存储 | 12.89 GiB | |
| **专家权重 `w13`+`w2`** | **12.00 GiB** | **占参数的 93.1%** |
| 非专家参数 | 0.89 GiB | attention、embedding、router 等全部 |
| KV 区域 | 15.00 GiB | |
| 专家 GiB / KV GiB | **0.800** | 专家权重与 KV 几乎对半分 HBM |

`top-k = 8/64`，即任一 token 只激活 **12.5%** 的专家。因此
**闲置专家常驻 10.50 GiB，相当于 KV 区域的 70.0%**。

会计口径：专家存储是参数存储的**子集**，二者从不相加；KV 区域是分配器区域，
不是在用 token 数。

## 容量悬崖：接纳控制第一次决定"能否完成"

`KV 15.00 GiB ÷ 128 KiB/token = 122,848 tokens`（KV 成本由 config 推导：
`16 层 × 16 KV heads × 128 head_dim × 2(K,V) × 2 bytes`）。

| 负载 | 需求 | 相对容量 | 实测结果 |
|---|---:|---:|---|
| 16 × 4096 | 65,536 tokens | −46.7% | 完成 |
| **32 × 4096** | **131,072 tokens** | **+6.7%** | **`CAPACITY_BOUNDARY_STOP`** |

请求级结果（同输入身份、同引擎、仅 prompt 长度与 cap 不同）：

| cell | prompt | cap | 完成 | SLO 达标率 | goodput rps | TPOT p50 |
|---|---:|---:|---:|---:|---:|---:|
| short-cap16 | 128 | 16 | 32 | 0.500 | 0.7663 | 9.27 ms |
| short-cap32 | 128 | 32 | 32 | **1.000** | **2.4364** | 11.47 ms |
| long-cap16 | 3072 | 16 | 32 | 0.500 | 0.5582 | 13.68 ms |
| long-cap32 | 3072 | 32 | **0** | **0.000** | **0.0000** | 20.12 ms |

**这是本仓库首次出现接纳上限改变"可行性"而非仅"快慢"的证据：**

- 短上下文域：`cap32` 压倒性优于 `cap16`（goodput 3.18×，达标率 0.500→1.000）。
  **在此域，限制并发纯粹是损失**——这与前三轮结论一致。
- 长上下文域：**同一个 `cap32` 变成灾难**（0/32 完成），而 `cap16` 完成全部 32 个。
  最优接纳上限随上下文长度反转。

前三轮实验全部运行在 `max_model_len=256`、峰值 KV 利用率约 12%、
**54,622 个 step 零抢占**的域内。那里不存在稀缺资源，所以接纳控制退化为"调 batch
宽度"，这解释了此前的连续负结果——**不是机制错，是运行域里没有可争夺的资源**。

## 专家驻留税是悬崖的直接成因

超出量只有 **6.7%**，而闲置专家占 10.50 GiB。反事实容量核算：

| 情形 | KV 可用 | 容量 | 32×4096 |
|---|---:|---:|---|
| 实测（全专家常驻） | 15.00 GiB | 122,848 tokens | **超出 6.7%** |
| 若释放闲置专家 | 25.50 GiB | 208,864 tokens | 富余 37.2% |
| 等算力密集模型（~1.3B 参数） | 25.46 GiB | 208,590 tokens | 富余 37.2% |

在同一 27.88 GiB 的 `参数 + KV` 预算下，MoE 把 **46.2%** 交给参数、KV 只得 53.8%；
等算力密集模型只需 8.7% 给参数。**同一个 32×4096 工作负载在密集模型上可行，
在 MoE 上越界——差距仅 6.7%，正落在专家驻留税的量级内。**

这是 MoE serving 特有的：为 7B 权重付 HBM，只拿 1B 算力，代价直接以 KV 容量
（进而以并发上限与可行性）结算。

## 诚实的反面细节

以下削弱"只要释放专家就能提升并发"的简单叙事：

1. **`max_num_seqs=32` 常常先钳住。** 满上下文并发从 29.8 提到 50.8（1.70×）后，
   引擎上限仍是 32，故捕获桶不变（32→32），每请求成本仅降 9.4%。
   **KV 容量与引擎并发上限在此配置下几近重合**，这本身是个巧合而非普适。
2. **释放专家不免费。** 反事实列是**容量上界**，不含 paging 传输成本、延迟或对
   route/质量的影响。本分析明确拒绝把它净算成收益。
3. **短域 SLO 达标率与长域相同（均 0.500）于 cap16。** long 比 short 慢 47.6%
   （TPOT 9.27→13.68 ms）却达标率一致——说明该格由 SLO 阈值位置决定，不由 KV 压力决定。
4. **`long-cap32` 是 0 完成，不是"慢"。** 它是 fail-closed 边界事件，
   **绝不可与完整运行比较吞吐**。其 TTFT/TPOT 数字来自中止前的部分请求，不构成性能结论。
5. 悬崖位置由 `max_model_len=4096` 与 32 请求共同决定，是这一组配置的属性，
   不是模型的普适常数。

## 形成的系统规律

```text
expert residency tax converts HBM into a KV capacity cliff,
and admission control only matters on the cliff's edge
```

三轮负结果 + 本轮对照给出一条连贯解释：接纳控制的价值不由并发数或专家负载决定，
而由**是否存在被争夺的稀缺资源**决定。在 MoE serving 中，该稀缺资源由专家驻留税
系统性地制造出来，且悬崖可以离可行区非常近（本例 6.7%）。

这条规律**建设性**地指出下一步该测什么：不是更聪明的接纳预测器，而是
**接纳决策是否应以 KV 可行性（而非延迟信号）为一等约束**。

## 明确不外推

- 4 个 GPU cell 均为单次、单模型、单卡、32 条真实 WikiText 文档、固定 1024 输出、
  单一到达序列。**无重复**，故任何数值差异都未经受控重复确认。
- `CAPACITY_BOUNDARY_STOP` 是该实验为保持非抢占不变量而设的 fail-closed 停止，
  **不代表 vLLM 生产行为**（默认会抢占重算）。本轮不主张抢占策略优劣。
- 反事实容量列不是可达收益，密集模型对比是**按参数量的算术对照**，非实测。
- 未测 U/C 专家信号增量、任务质量、自然 EOS、第二模型、EP/多卡。
  **本轮仍不含专家感知调度贡献。**
- 本目录未产生任何自己的 GPU 数据；并行实验的正反序重复与其余 cell 归其目录裁决。

## 唯一下一最小实验

在长上下文域做一次**KV 可行性接纳**的最小对照：同引擎、同输入、同到达，
比较三臂——
`cap32`（已知越界）、`cap16`（已知可行）、以及一条**只用 KV 占用与队列长度、
不使用任何延迟信号**的接纳规则，使其在越界前停止接纳。
主指标为联合 SLO goodput 与完成率，完整计入采集与决策成本。

这是本仓库第一次有依据地期待接纳动作产生请求级收益，因为它现在有一个**可行性
边界**要守，而不只是一个宽度要调。执行前需冻结设计，并要求正反序受控重复。
该实验 **UNRUN**。

## 复跑

```bash
cd refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_pressure_probe_r01
python3 ../../../experiments/admission_capacity/analyze_expert_residency_tax.py \
  --cell inherited_telemetry/repeat0-short-cap16 \
  --cell inherited_telemetry/repeat0-long-cap16 \
  --cell inherited_telemetry/repeat0-short-cap32 \
  --cell inherited_telemetry/repeat0-long-cap32 \
  --output-dir <新目录>
```

输出路径须不存在。`inherited_telemetry/` 为只读副本，原始数据留在并行实验目录，
未修改、未移动。未 push，未改 `docs/current/README.md`。

| 结束字段 | 边界 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；本目录探针 `WITHDRAWN_IN_FAVOUR_OF_STRONGER_DESIGN` |
| Evidence type | `STRUCTURAL_ACCOUNTING_OVER_MEASURED_MEMORY_TELEMETRY` + 继承的 4 个 native cell |
| 实测关键量 | 专家权重 12.00 GiB（参数 93.1%）；KV 15.00 GiB；闲置专家 10.50 GiB = KV 的 70.0%；32×4096 超容量 6.7% |
| 首次出现 | 接纳上限改变**可行性**：long-cap32 完成 0/32，long-cap16 完成 32/32 |
| Not measured | KV 可行性接纳规则的收益、受控重复、U/C 增量、paging 成本、第二模型、EP |
| Claim ceiling | 一条有实测支撑的 MoE 结构规律与一个已定位的运行域；**无方法 GO** |
