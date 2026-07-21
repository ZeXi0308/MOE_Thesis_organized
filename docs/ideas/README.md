# 活 / 条件 Idea 总索引

> 2026-07-21 整理。现行文档只保留下列路线；**纠错后仍判死的 idea 已全部归档**：  
> [`../99_archive/killed_ideas/README.md`](../99_archive/killed_ideas/README.md)  
> **现状与演进总览**：[`../01_current_status/研究现状与Idea演进_2026-07-21.md`](../01_current_status/研究现状与Idea演进_2026-07-21.md)  
> **思想 / 实验设计 / 代码 / 演进（先读）**：[`全部Idea设计总览.md`](全部Idea设计总览.md)

| Idea | 状态 | 一句话 | 设计说明 |
|---|---|---|---|
| Rank 长尾 + FP8-first | 结构证据 GO | combine 尾部低比特远比头部安全；FP8-first Pareto | [设计说明](A_rank_tail_fp8/设计说明.md) |
| Receiver-aware | 条件性 | 结构拥塞画像站得住；在线自适应待/未过 Existence Test | [设计说明](receiver_aware/设计说明.md) |
| Verify, Don’t Predict（Idea B） | 条件 GO | 弱持续性杀预测器；固定周期影子验证在 OLMoE 过线 | [设计说明](B_verify_precision/设计说明.md) |
| Energy-SLO Precision EP | 最稳备线 | batch×FP8 能效/吞吐杠杆真实；待联合 Pareto | [设计说明](energy_slo/设计说明.md) |
| Quality debt / Isolation | 弱正 | predictor-free 债务公平有改善但未过强门槛 | [设计说明](quality_debt/设计说明.md) |

每个 idea 目录下现含：

- `设计说明.md` — **思想、背景、实验设计、代码组织、演进**  
- `原文/` — 从 GitHub 备份恢复的长文  
- `experiments/` — 脚本 · `outputs/` — 产物  

共享库：[`../../experiments/shared/`](../../experiments/shared/)  
脚本索引：[`../../experiments/SCRIPTS_BY_IDEA.md`](../../experiments/SCRIPTS_BY_IDEA.md)

