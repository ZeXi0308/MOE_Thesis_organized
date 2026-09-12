# Batched 专家并集：稀疏专家回收是否存在动作空间（执行前冻结）

2026-09-10。冻结时本实验 GPU 执行数为 0。本文件在见到任何 route 数据前写定；
需修改则另立新日期文件，两者均保留。

## 为什么是这个实验

两条独立的线索指向同一个未运行的前置测量：

1. **仓库早已点名要求，且明确记为未启动。**
   [`20260908_kv_budget_r01/REPORT.md`](../20260908_kv_budget_r01/REPORT.md)：
   > 若之后检查稀疏专家回收，应先测实际 batched expert union、跨 step 复用与 KV
   > 压力交集；**单 token 未选中专家算术不构成可回收 HBM**。该结构诊断仍是条件候选，
   > **没有启动**。

2. **我自己犯了那句话点名的错误。**
   [`20260908_kv_pressure_probe_r01`](../20260908_kv_pressure_probe_r01/REPORT.md)
   用 per-token 比例 `1 - 8/64` 算出"闲置专家 10.50 GiB"。均匀零模型下，
   在本仓库实测的 batch 宽度 16–32 处，该值应为 **1.42–0.17 GiB**，
   即原值的 **13.5%–1.6%**。已在
   [`ADDENDUM_BATCHED_UNIT_CORRECTION.md`](../20260908_kv_pressure_probe_r01/ADDENDUM_BATCHED_UNIT_CORRECTION.md)
   撤回，并阻断一切 paging 主张直至本测量完成。

因此本实验既是仓库要求的前置条件，也是我自己结论的证伪测试。

## 研究问题

> 在真实 batched decode 中，每个 step、每一层，被 batch 内任意 token 选中的专家
> **并集**占全部专家的比例是多少？该并集在连续多步上如何增长？

一层必须持有 batch 内任意 token 路由到的全部专家，因此 `1 - U` 是该层专家权重
在该 step **唯一可能不在 HBM 中**的比例。这决定回收动作空间是否存在，
**先于**任何搬运成本问题。

## 冻结的判据（在见到数据前写定）

`action_space_threshold = 0.10`。机械裁决三档：

| 条件 | 裁决 | 含义 |
|---|---|---|
| 全部层的 median idle fraction `< 0.10` | `NO_RESIDENCY_HEADROOM` | 每步都需近全部专家，回收族关闭 |
| 有 headroom 但全部层 2 步内并集饱和 | `HEADROOM_BUT_IMMEDIATE_REUSE` | 搬运成本每步复发，静态驻留决策不成立 |
| 某层跨视野保持闲置 | `CANDIDATE_RESIDENCY_HEADROOM` | 存在条件动作空间，再谈成本 |

同时报告：
- 每层 `U` 的 min/p50/max 与对应 batch 宽度；
- 均匀独立路由零模型 `(1-k/E)^B` 作为对照（**零模型，不是本 router 的行为**）；
- 视野 `W ∈ {1,2,4,8,16,32}` 的滑窗并集增长与 99%/95% 饱和视野；
- 负载偏斜 `C = max(n_e) / mean(n_e)`，mean **包含**零 token 专家；从未被选中的专家数。

## 三个可证伪预测

| # | 预测 | 证伪条件 |
|---|---|---|
| E1 | 实测 median idle fraction 高于均匀零模型（router 有集中性） | 低于或等于零模型 → 路由接近均匀，回收空间由零模型给定（≈0） |
| E2 | 在 batch 宽度 ≥16 时，median idle fraction `< 0.10` | ≥0.10 → 存在瞬时余量，进入视野检验 |
| E3 | 99% 饱和视野 `≤ 4` 步 | `> 4` 或不饱和 → 存在跨步稳定的闲置子集 |

**E2 与 E3 都通过（即 `NO_RESIDENCY_HEADROOM` 或 `HEADROOM_BUT_IMMEDIATE_REUSE`）
是我的预期结果。** 这会关闭一个我自己曾主张过的机制族，并顺带解释一个已知现象：
`FluxMoE v2` 选择"每层执行前装齐该层全部专家"而非 top-k 缺失分页
（见 [`20260908_kv_budget_r01/PRIOR_ART.md`](../20260908_kv_budget_r01/PRIOR_ART.md)），
若 E2/E3 成立，这就不是工程保守，而是**结构上的必然**。

## 查新边界（已核实的最近工作）

来自仓库既有 `PRIOR_ART.md`，不重复扩张：

| 工作 | 动作 | 与本测量的关系 |
|---|---|---|
| WiSP v2 (2026-08-30) | expert-map + host 装入；drained barrier 回收**闲置 KV** 扩大专家池 | 方向相反（KV→expert）。其 1.19× 结果不能当作 expert→KV 已验证 |
| ELDR v2 (2026-07-02) | expert signature 预测 decode 重叠，选最轻 decoder 降低 expert union | 有 union 概念但**无显存回收动作**；本测量问的是回收可行性 |
| FluxMoE v2 (2026-04-30) | 按层流式加载，每层执行前装齐**该层全部专家** | 恰好回避 top-k 分页。本测量可解释该选择是否必然 |

本轮**不主张**任何新机制、不主张首次测量 expert union（ELDR 已用该信号）。
候选残差仅为：**在固定引擎与真实 batched serving 下，量化 top-k 稀疏性能否转化为
可回收 HBM，并给出该转化失效的结构原因。**

## 方法与不变量

- 复用现有冻结输入：32 篇真实 WikiText103 文章、128 与 3072 两种 prompt、
  固定输出、50ms steady 到达。输入身份不改。
- 引擎配置与既有实验一致：pinned BF16 OLMoE `@6d84c485...`、vLLM 0.26.0、
  单 RTX 5090、engine32、context4096、token budget1024、`util=0.90`、FCFS、
  chunked prefill、无 prefix cache、同步 in-process。
- 路由由模型自身 router logits 按其 softmax/top-k 重算，与仓库既有做法一致。
- **不改 router、top-k、精度、placement；不实现任何 paging；不注入任何动作。**
  本轮是纯测量。

## 证据上限与不外推

- `U` 是**结构信号**：不等于实测 HBM 流量，不等于 fused backend 实际加载了什么，
  也不等于闲置专家的字节在实践中可回收。这三点差异必须在报告中重述。
- 单模型、单卡、有限 cohort、固定输出、两个 prompt 长度。
- 不涉及 EP/多卡（EP 下专家分布在 rank 间，并集语义不同）、第二模型、任务质量。
- 若某层从未被选中的专家数很高但集中在少数层，**不得**跨层平均后声称整体余量。
- `NO_RESIDENCY_HEADROOM` 只关闭"**在本运行域、按 top-k 缺失分页**"这一 formulation，
  不否定 FluxMoE 式整层流式、不否定 offload 域、不否定 WiSP 的 KV→expert 方向。

## 保留

全部 episode 无论结果一律保留。原始 route 统计紧凑导出（不逐 token 落盘），
raw 不修改，更正写 addendum。未 push，不改 `docs/current/README.md`。

## 当前状态

新 5090 机器（`connect.weste.seetacloud.com:11155`，32607 MiB，与旧机同型号）
环境搭建中：venv 已建（Python 3.12.3，与旧机逐位一致），vLLM 0.26.0 安装中，
模型下载中（hf-mirror，13 GB）。`ExpertUnionTracker` 与 19 项测试已完成并通过。
**GPU 采集 UNRUN。**
