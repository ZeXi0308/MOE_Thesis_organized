# 活 / 待验证 Idea 总索引

> 2026-07-22 系统纠错后整理。下列包含一条结构 GO、三条待正确验证/重新定义的路线，以及一条 NO-GO：
> [`../99_archive/killed_ideas/README.md`](../99_archive/killed_ideas/README.md)  
> **现状与演进总览**：[`../01_current_status/研究现状与Idea演进_2026-07-21.md`](../01_current_status/研究现状与Idea演进_2026-07-21.md)  
> **系统纠错审计**：[`../01_current_status/Idea系统纠错审计_2026-07-22.md`](../01_current_status/Idea系统纠错审计_2026-07-22.md)
> **思想 / 实验设计 / 代码 / 演进（先读）**：[`全部Idea设计总览.md`](全部Idea设计总览.md)

| Idea | 状态 | 一句话 | 设计说明 |
|---|---|---|---|
| Rank 长尾 + FP8-first | 仅 Claim 1 结构 GO | matched-byte 下 combine tail 比 head 安全；frontier 严格门槛跨模型未过 | [设计说明](A_rank_tail_fp8/设计说明.md) |
| Receiver-aware | 条件性 | 结构画像可用；旧 Existence Test 未启用纠正后 codec hard gate | [设计说明](receiver_aware/设计说明.md) |
| Verify, Don’t Predict（Idea B） | 正结果失效 | 离线 KL 掩码没有执行混合 KV 策略；待 in-loop 重跑 | [设计说明](B_verify_precision/设计说明.md) |
| Energy-SLO Precision EP | 硬件 characterization | full-sequence batch 与 GEMM-core 杠杆已观测；serving SLO/controller 未验证 | [设计说明](energy_slo/设计说明.md) |
| Quality debt / Isolation | NO-GO | harm 可观测性和 CI 口径不成立，且点估计未过 20% 门 | [设计说明](quality_debt/设计说明.md) |

每个 idea 目录下现含：

- `设计说明.md` — **思想、背景、实验设计、代码组织、演进**  
- `原文/` — 从 GitHub 备份恢复的长文  
- `experiments/` — 脚本 · `outputs/` — 产物  

共享库：[`../../experiments/shared/`](../../experiments/shared/)  
脚本索引：[`../../experiments/SCRIPTS_BY_IDEA.md`](../../experiments/SCRIPTS_BY_IDEA.md)
