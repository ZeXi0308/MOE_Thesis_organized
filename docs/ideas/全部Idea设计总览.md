# 全部 Idea 设计总览（系统纠错后 · 2026-07-22）

> 备份仓库：https://github.com/ZeXi0308/MOE_Thesis.git  
> 本地镜像：`~/Desktop/MOE_Thesis_backup`（只读对照）  
> 每个 idea 目录下现有：
> - **`设计说明.md`** — 思想 / 背景 / 实验设计 / 代码组织 / 演进（**先读这个**）  
> - **`原文/`**（活）或 **`run_conclusions/`**（判死）— 长文材料  
> - **`experiments/`** 或 **`scripts/`** — 代码

权威状态表仍见：[`../01_current_status/研究现状与Idea演进_2026-07-21.md`](../01_current_status/研究现状与Idea演进_2026-07-21.md)  
系统纠错证据与最小重跑协议：[`../01_current_status/Idea系统纠错审计_2026-07-22.md`](../01_current_status/Idea系统纠错审计_2026-07-22.md)
上一轮 Phase 1 创新筛选（已被 v2 取代为历史候选）：[`../01_current_status/Phase1_创新探索与候选收敛_2026-07-22.md`](../01_current_status/Phase1_创新探索与候选收敛_2026-07-22.md)
上一轮 Phase 1 v2（历史排序：CommitMap-EP / FJEC）：[`../01_current_status/Phase1_v2_创新探索与候选收敛_2026-07-22.md`](../01_current_status/Phase1_v2_创新探索与候选收敛_2026-07-22.md)
最新 Phase 1 v3（5090 可判死、高上限筛选；仅条件保留 RouteShare-VTC / RouteCloak）：[`../01_current_status/Phase1_v3_5090可验证高上限Idea筛选_2026-07-22.md`](../01_current_status/Phase1_v3_5090可验证高上限Idea筛选_2026-07-22.md)
元材料（Registry / 时间线 / 三条线长文）：[`../99_archive/killed_ideas/_meta_原文/`](../99_archive/killed_ideas/_meta_原文/)

---

## 活 / 条件（5）

| Idea | 状态 | 设计说明 |
|---|---|---|
| Rank 长尾 + FP8-first | GO（仅 Claim 1 结构） | [A_rank_tail_fp8/设计说明.md](A_rank_tail_fp8/设计说明.md) |
| Receiver-aware | CONDITIONAL（正式硬门槛重跑未完成） | [receiver_aware/设计说明.md](receiver_aware/设计说明.md) |
| Verify, Don’t Predict（B） | INVALIDATED / NEEDS IN-LOOP RERUN | [B_verify_precision/设计说明.md](B_verify_precision/设计说明.md) |
| Energy-SLO | CHARACTERIZATION / SYSTEM UNVERIFIED | [energy_slo/设计说明.md](energy_slo/设计说明.md) |
| Quality debt | NO-GO | [quality_debt/设计说明.md](quality_debt/设计说明.md) |

## 已归档 / 非主线（12）

> 此处不再把 12 条统称为「严格实验判死」：QuotaEP 是 Gate C 未验证，WaveCredit 是 prior-art 筛查后未实验，Additive 的旧可加性否定已撤回。其余狭义判死范围见纠错审计。

| Idea | 设计说明 |
|---|---|
| CreditReduce | [../99_archive/killed_ideas/creditreduce/设计说明.md](../99_archive/killed_ideas/creditreduce/设计说明.md) |
| TokenRace-EP | [../99_archive/killed_ideas/tokenrace/设计说明.md](../99_archive/killed_ideas/tokenrace/设计说明.md) |
| Prefetch | [../99_archive/killed_ideas/prefetch/设计说明.md](../99_archive/killed_ideas/prefetch/设计说明.md) |
| Progressive | [../99_archive/killed_ideas/progressive/设计说明.md](../99_archive/killed_ideas/progressive/设计说明.md) |
| PLTB / Additive | [../99_archive/killed_ideas/pltb_additive/设计说明.md](../99_archive/killed_ideas/pltb_additive/设计说明.md) |
| Residual / Shadow | [../99_archive/killed_ideas/residual_shadow/设计说明.md](../99_archive/killed_ideas/residual_shadow/设计说明.md) |
| RouteFidelity | [../99_archive/killed_ideas/routefidelity/设计说明.md](../99_archive/killed_ideas/routefidelity/设计说明.md) |
| MassCover | [../99_archive/killed_ideas/masscover/设计说明.md](../99_archive/killed_ideas/masscover/设计说明.md) |
| QuotaEP | [../99_archive/killed_ideas/quotaep/设计说明.md](../99_archive/killed_ideas/quotaep/设计说明.md) |
| Graceful / QTree | [../99_archive/killed_ideas/graceful_qtree/设计说明.md](../99_archive/killed_ideas/graceful_qtree/设计说明.md) |
| Mean-balance placement | [../99_archive/killed_ideas/placement_mean_balance/设计说明.md](../99_archive/killed_ideas/placement_mean_balance/设计说明.md) |
| WaveCredit | [../99_archive/killed_ideas/wavecredit/设计说明.md](../99_archive/killed_ideas/wavecredit/设计说明.md) |

---

## 选题史一句话（三代）

1. **早期（~07-13）**：PLTB / R-layout / Graceful / QTree / additive MILP / 选题地图上的 Shadow·Residual·Energy·Quality…  
2. **转折（~07-14–17）**：QuotaEP-H → 统一严格方法论起步  
3. **严格周（~07-17–21）**：CreditReduce / RouteFidelity / WaveCredit / MassCover / TokenRace 连续评估；Receiver / Quality / Prefetch / Energy / Idea B 深化与 GPU 交叉验证  

完整时间表原文：[`../99_archive/killed_ideas/_meta_原文/MoE_全部候选选题完整时间线_2026-07-19.md`](../99_archive/killed_ideas/_meta_原文/MoE_全部候选选题完整时间线_2026-07-19.md)
