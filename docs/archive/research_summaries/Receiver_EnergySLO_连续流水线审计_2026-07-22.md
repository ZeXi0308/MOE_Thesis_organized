# Receiver / Energy-SLO 连续流水线审计（2026-07-22）

> **历史跨方向汇总。** 本文用于追溯 DDRC / Route-row 旧流水线，不再充当当前执行入口；现行裁决见 [`../../current/README.md`](../../current/README.md)。

状态：**HISTORICAL DDRC / ROUTE-ROW PIPELINE / SUPERSEDED AS ACTIVE PIPELINE**

> 本文件记录较早的 DDRC / Route-row FP8 连续流水线。后续 CJC / JouleQueue v1 的最严 Phase 4 结果见 [`CJC_JouleQueue_Phase4_交叉审计_2026-07-22.md`](CJC_JouleQueue_Phase4_交叉审计_2026-07-22.md)：两者同样 `BLOCKED`，但阻断原因已变为控制 locus、负载/统计设计、能耗估计和 route→job/oracle 闭合问题。不得用本文件的局部单测或旧 capability probe 替代最新门禁。

本轮按“探索 → 冻结设计 → 实现 → 独立 Review → 正式执行 → 解读”连续推进；没有因为连续执行而跳过门禁。两个方向均在 Phase 4 留有未关闭 P0，因此没有启动 sealed / formal Phase 5，也没有产生新的科学 GO、科学 No-Go 或 RDMA/P99 结论。

## 1. Phase 1：入选与淘汰

### 入选表

| 轴 | 入选候选 | 一句话主张 | 最强简单基线 | 必要条件 / 最大风险 | Phase 1 判定 |
|---|---|---|---|---|---|
| Receiver-awareness | **DDRC：Dispatch-Derived Cross-Sender Return Credits** | 在 request-origin balancing 后，只利用同层 dispatch 已知的跨 sender 来源分解，对共享 receive resource 发显式、收费的低精度 credit；检验其相对 sender-local exact handle 是否仍有至少 3 percentage points 增量 | `sender_local_exact_handle`，并包含 `causal_prev_step`、static、uniform | 固定 top-k 时 receiver 总回程行数已由 `K × valid_origin_tokens` 决定；唯一可能增量是来源分解映射到真实共享 receive resource。若全局 oracle 都不超 sender-local，方向直接停 | **条件入选**；先做必要性 / oracle ceiling，不直接上 RDMA 或 bandit |
| Energy-SLO | **Route-row FP8 full-path break-even gate** | 在真实 continuous decode 中，用 router 后、expert 前的当前 `m[l,e,t]` 只在完整 FP8 expert MLP 同时节能、不增时延且过质量地板时切换，检验其相对 static FP8、batching-only 与 routing-blind joint baseline 的增量 | always-BF16、always-FP8、routing-blind joint、precision-only、batching-only | 需要真实 per-expert BF16/FP8 动态 hot path、独立 KV、完整 arrival/P99/board-energy 会计；普通 static FP8 或 isolated GEMM 不等价 | **条件入选**；不存在 integrated hot path 时只能 BLOCKED，不能给科学 No-Go |

### 淘汰 / 本轮不复活

| 候选 | 判定 | 原因 |
|---|---|---|
| “预测 receiver 总负载”的旧表述 | 淘汰 | 固定 top-k、无 drop 时总行数是 scheduler 已知量；把它重新发现不构成 receiver-awareness 增量 |
| 旧 EWMA / HHI / regime / bandit 叙事 | 本轮不入选 | 必要性尚未过；这些机制会增加调参和因果风险，不能替代对 `sender_local_exact_handle` 的增量证明 |
| Receiver 多节点 RDMA 直接实现 | 后置 | 单 GPU 的 route-real/topology-proxy 必须先过 oracle、quality 与完整 codec/credit 门；本轮不把 H2D 称为 RDMA |
| static FP8 × batch 微基准相乘 | 淘汰为系统结论 | 两个独立微基准不能相乘成 joint Energy-SLO 收益；缺 arrival、KV decode、P99、动态 quantize/scale/cast |
| DVFS、offload、replica/placement、多节点能耗联控 | 本轮不入选 | 当前单 GPU 证据和计量边界不足；会扩大问题而不关闭最小 route-value 命题 |
| 已归档 killed ideas | **默认不复活** | 本轮没有新的、经纠错后足以越过原 Go 门的仓库证据；勘误只撤回错误数字或错误死因，不自动把机制变成可行。禁止引用 additive 勘误中已经作废的倍率 |

## 2. Phase 2：冻结协议

- Receiver：[`../receiver_aware/ddrc/DDRC_Phase2_冻结实验协议_2026-07-22.md`](../receiver_aware/ddrc/DDRC_Phase2_冻结实验协议_2026-07-22.md)
- Energy-SLO：[`../../ideas/energy_slo/route_row_fp8/Phase2_RouteRow_FP8_BreakEven_冻结协议_2026-07-22.md`](../../ideas/energy_slo/route_row_fp8/Phase2_RouteRow_FP8_BreakEven_冻结协议_2026-07-22.md)

协议均明确：主口径、敏感性口径、强基线、数据隔离、Go/No-Go、停止规则和实现验收；未把旧实验或 dev smoke 预先写成结论。

## 3. Phase 3：实现与返修

### Receiver DDRC

实现：

- `docs/archive/receiver_aware/ddrc/experiments/ddrc_policy.py`
- `docs/archive/receiver_aware/ddrc/experiments/run_ddrc_existence_gpu.py`
- `docs/archive/receiver_aware/ddrc/experiments/test_ddrc_policy.py`
- `docs/archive/receiver_aware/ddrc/configs/ddrc_v1.json`

已实现并通过的局部不变量：sender/receiver identity、local/remote closure、显式 16 B header + 8 B record credit、late/duplicate/hardgate rollback、字节闭合、`NOT_RDMA` 边界、两模型集合硬门、formal top-k/drop 门、按 stream/layer 的严格因果历史、`causal_prev_step` 强基线、状态语义分离，以及一次 action 决策后对 `serialized_tiles` / `amortized_once_per_step_proxy` 两套会计。

返修后验证：**23/23 tests PASS**；dev smoke 为 `NOT_TESTED`，`formal_run_valid=false`，`scientific_verdict=null`，`go=false`；28 条主 action 与 28 条敏感性会计的 action signature 一致。双 codec 当前状态仍为 `ACCOUNTING_ONLY_NO_SCIENTIFIC_VERDICT`。

### Energy-SLO

实现：

- `docs/ideas/energy_slo/route_row_fp8/experiments/route_row_policy.py`
- `docs/ideas/energy_slo/route_row_fp8/experiments/run_route_row_surface.py`
- `docs/ideas/energy_slo/route_row_fp8/experiments/continuous_decode_harness.py`
- `docs/ideas/energy_slo/route_row_fp8/experiments/power_accounting.py`
- 对应 `test_*.py` 与 `configs/route_row_break_even_v1.json`

已实现并通过的局部不变量：固定 row bins、`sum_e m = active_tokens × top_k`、不确定 cell 回退 BF16、三投影 FP8 wrapper、双权重驻留账本、completed-output-token 分母、NVML counter 优先、实际采样 gap 门、KV identity / prefill-once / length-1 decode 审计骨架。

返修关闭两个真实 fail-open：

1. JSON 字符串 `"false"` 不再被转成质量门 `True`；quality pass 必须绑定合法 SHA-256。
2. 测试 backend 即使自报 `real_continuous_engine=True`，abstract harness 也不能授予 scientific eligibility。

返修后验证：**32/32 tests PASS**。配置仍明确为 `PHASE3_BLOCKED_PENDING_INTEGRATED_CONTINUOUS_DYNAMIC_EXPERT_HOT_PATH`；surface 永远是 `INELIGIBLE_CALIBRATION_PROXY`。

## 4. Phase 4：独立 Review 与交叉会计

正式报告：

- Receiver：[`../receiver_aware/ddrc/CodeReview_DDRC_Phase4_2026-07-22.md`](../receiver_aware/ddrc/CodeReview_DDRC_Phase4_2026-07-22.md)
- Energy-SLO：[`../../ideas/energy_slo/route_row_fp8/CodeReview_RouteRow_Phase4_2026-07-22.md`](../../ideas/energy_slo/route_row_fp8/CodeReview_RouteRow_Phase4_2026-07-22.md)
- 交叉会计：[`Phase4_交叉会计审计_2026-07-22.md`](Phase4_交叉会计审计_2026-07-22.md)

返修后的门禁状态：

| 方向 | 已关闭的关键错误 | 仍开放的 P0 | Phase 4 |
|---|---|---|---|
| Receiver | 单模型误 GO、causal baseline 漏项与跨流/逆序风险；top-k/drop 的部分 schema；同 action 双 codec 会计 | native route + action-matched quality producer/source hash；冻结数据 manifest；subject/request hierarchical bootstrap 与 forward/burst block bootstrap；G1 route-only 前置状态机；完整 G2 resource/time accounting；正式双 codec decision gate | **BLOCKED** |
| Energy-SLO | quality bool/hash fail-open；backend capability 自证 scientific eligibility；generic power actual-gap gate；当前 bundle hash 漏项 | integrated router→LUT→expert dynamic hot path；真实 continuous backend；in-loop quality producer 与 LUT 绑定；B0–B4/C、Poisson/MMPP、batch grid、full-drain 正式 runner；capture causal producer；两模型 native recipe 验收 | **BLOCKED** |

单测通过只证明局部不变量，不能抵消 P0。两份 Review 均未生成 `SIGNED-OFF` attestation。

## 5. GPU 环境与 dev capability probe（不是 Phase 5）

远端只执行了不上传仓库代码的通用 capability probe：

- 1× RTX 5090；PyTorch `2.8.0+cu128`；隔离 venv 中 vLLM `0.10.2` 可 import，`pip check` 无破损；未修改系统 Python 环境。
- OLMoE 与 LLM-jp 的 **static vLLM FP8** 均能离线加载固定缓存 revision 并生成短输出。
- RTX 5090 的 NVML UUID 可读，`nvmlDeviceGetTotalEnergyConsumption` 可返回 total-energy counter。

这些只证明“模型 / static FP8 / board-energy 计量基础存在”，不证明 route-row 动态 per-expert gate、双驻留公平性、quality、arrival/P99 或 Energy-SLO 增量。vLLM static FP8 不能替代本轮候选。

由于安全策略阻止把未提交研究代码与协议上传到外部主机，仓库 bundle 尚未传到 GPU；因此没有声称运行过本轮实现的 GPU smoke。需要在明确授权该主机接收工作区内容后才能上传。

## 6. Phase 5 / 6 判定

- **Phase 5：未进入。** 原因是 Phase 4 两方向均 `BLOCKED`；没有 formal raw table、没有 sealed verdict。
- **科学结论：未验证。** 当前既不能说 DDRC Go/No-Go，也不能说 route-row Energy-SLO Go/No-Go。
- **问题分类：** 当前主要是“正式实验实现与会计未闭合”，并包含已修复的代码逻辑错误；不是“纠错后机制仍死”的负结果。
- **不要再怎么救：** 不用旧 no-hardgate / hotspot / linear quality proxy 救 Receiver；不用 static FP8、full-forward batch、isolated GEMM 或 random activation 救 Energy；不放宽 3 pp、10%、P99 或质量门；不提前展开 RDMA、bandit、DVFS、offload、placement。

下一轮最短闭环：Receiver 先实现受 hash 管理的 native route/action-quality producer和 G1 两段状态机；Energy 先在固定 serving engine commit 内做最小 router→row-count→动态 expert kernel hook，若 hot path 无法成立则保持工程 `BLOCKED`，不再增加控制器花样。
