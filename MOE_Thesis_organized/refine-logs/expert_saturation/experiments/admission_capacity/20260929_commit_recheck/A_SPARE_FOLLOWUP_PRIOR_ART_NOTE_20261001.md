# 余量后续恢复的直接动作近邻（2026-10-01）

范围仅是 [首个 off/on 结果](A_SPARE_FOLLOWUP_PAIR_RESULT_R01_20261001.md)的动作级文献边界。先读已有的 [`RELATED_WORK_RECOVERY_20260914.md`](../RELATED_WORK_RECOVERY_20260914.md)，本次只回读下面两篇原论文/作者论文页；未运行其系统，不能把本地部分实现称为论文复现。

1. **Backfilling 已覆盖“前位暂不能运行，挑后位可容纳工作填空”的基本动作。** Shmueli 与 Feitelson 的 [2005 年论文作者页](https://research.ibm.com/publications/backfilling-with-lookahead-to-optimize-the-packing-of-parallel-jobs)明确把先给首位工作保留进展机会、再让后位工作填充空余资源作为常见 backfilling 做法，并进一步用 lookahead 选择更好的组合。本地 Q1 先让原恢复目标返回真实新输出，再在原生 FCFS 队首完整历史放不下时挑至多一个已等待至少 30 步、完整历史可容纳的后续请求，在动作层与这一既有思想直接相撞。这里没有 HPC 作业的未来运行时估计或保留开始时刻，也不能仅凭“原目标先输出”保证后续服务不延迟它；**不能宣称发明了 backfilling**。

2. **Andes 已覆盖一个资源释放带入多个请求，以及全体 peer 损失需要计价。** [Andes 原论文 §4.1–4.3](https://arxiv.org/html/2404.16283v2#S4)同时约束 batch 大小和总 KV 上下文；其例子明确一次抢占长上下文请求后接纳两个等待请求，并用 refiner 判断接纳收益能否覆盖恢复、抢占及其他请求的 QoE 损失。本地动作在原目标优先、首输出已发生后才试一次额外原生接纳，缺 Andes 的 QoE、batch 目标和 peer 净收益模型，因此是更窄的强简单对照，不是 Andes 复现。值得精确区分的是，Andes 所列的 priority greedy packing 伪码在优先级顺序中遇到首个不适合的请求时使用 `break`；本地“跳过 FCFS 不适合的队首”与该伪码**不逐语等同**，而已由一般 backfilling 先例覆盖。

**当前结论与缺口。** [本地完整 pair](A_SPARE_FOLLOWUP_PAIR_RESULT_R01_20261001.json)观察到 24 条原生接纳→新输出→完成链，已见 H128 上一个 off→on 块的最大 gap 2.967→1.566 s，率和 mean flow 仍在预算内；全体原生抢占与打印传输量却上升，且输出序列及一个 stop 原因不同。这说明此固定实现有值得复核的系统行为，**不建立新的调度原语或动作级因果机制**。同代码反序 on→off 块仍需按原判据独立通过；即使通过，若论文主张独立算法贡献，还需说明相对容量合格 Q1 加简单 backfill 的可复现增量或新的预声明边界，并在新输入上确认，不能把 30 步门槛、一次后位填空或“没有同一步额外 victim”本身作为新颖性。此前 fit-first、Q10 保护和 Q10 提前让位的失败结论保持原样。

**后续状态更新：** [反序 on→off 块](A_SPARE_FOLLOWUP_PAIR_RESULT_R02_20261001.json)也独立通过九项冻结判据；上述一般 backfilling 的直接动作碰撞不变，新输入确认仍未执行。
