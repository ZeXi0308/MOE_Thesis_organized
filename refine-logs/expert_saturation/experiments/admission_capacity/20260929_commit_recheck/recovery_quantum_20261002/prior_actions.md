# 恢复成本、KV 增长与服务量子的动作边界

2026-10-02；有界查新，服务于当前原型。先读最新 A_progress/PAPER_ARGUMENT、既有 recovery related-work 与固定源码；本次重新读取 CacheOPT v2、UniBoost v1、LTR v1 正文和 BidKV 固定源码网页。未运行任何上游系统。

最直接碰撞是 **UniBoost MemGuard**，并非 BidKV。保护一次恢复使其获得有效 decode 服务、以减少重复 KV 移动，本身已有明确先例。剩余候选应解释并实现：**在恢复前，把可兑现的服务量 q、q 内物理页增长、恢复固定成本和其他请求的延迟代价联合决定**。这只是可检验差异，尚不是独立创新结论。

| 方法、固定来源 | 状态与决策时机 | 真正动作与资源耦合 | 量子/保证边界 |
|---|---|---|---|
| [CacheOPT v2 §§3.3–3.5](https://arxiv.org/html/2503.13773v2#S3.SS3) | 每迭代依据 TTFT/TBT 剩余裕度、历史最大迭代时延、预测长度、已用/已分配 KV | 紧急请求先获得基本 KV；必要时驱逐；其余 KV 按需求分配，并使用预测完成释放、提前补给与共享保留；按上下文 profile 在交换/重算之间选择 | 已把紧迫性、KV 资金、释放和恢复成本放入同系统。正文未给恢复成本决定 useful-output quantum 的规则；其预测需求/历史时延不能直接当本机严格输出 SLO 保证。 |
| [UniBoost v1 §3.3、A.1–A.2](https://arxiv.org/html/2606.18431v1#S3.SS3) | 到达时间、phase、已获得 decode 服务、近期尾部统计；按几何 attained-work 边界更新优先级 | 恢复/派发后至下一几何阈值不可驱逐；逐步检查下一 chunk/decode 的 KVNeed，victim 选择考虑优先级与 swap cost | 明确保证最小 useful-service 区间来摊薄移动，给出随输出长度对数增长的优先级重议点数。该计数性质不是任意物理内存压力下的墙钟 gap 保证；排队论部分限定 M/G/1。 |
| [LTR v1 §4.3 / Algorithm 1](https://arxiv.org/html/2408.15792v1#S4.SS3) | 全队列的学习排名、饥饿计数、优先级与剩余 quantum；每次批选择更新 | 饥饿达到阈值则提权并给予固定 PriorityQuantum；实际入 batch 的迭代扣量子；原目标已包括 TTFT 和最长后续 token 等待 | 固定量子并非根据恢复代价/页增长联合选取。原文计数单位是入 batch 的迭代，不应自动替换为 host 新输出数；本地改变此单位只能称组件变体。 |
| [BidKV 固定 selector](https://github.com/vLLM-HUST/vllm-ascend-hust-bidkv/blob/5ee80256d263d58b1e512d9d436d47e9bac564ba/vllm_ascend_bidkv/selector.py) | host 调用 victim selector 时，读取 computed tokens、output/max_tokens、累计抢占数和可选利用率/cooldown | 默认取最大 `computed/(1 + 0.5*completion + 0.3*preemptions + eps)`；候选集由 host 传入。完成进度/反复抢占代价与释放量已有联合排序 | 此 selector 没有恢复目标、恢复后 q 或跨步 KV 承诺。固定仓库不包含其 host 调度循环，不能推断上游如何处理已调度 prefix 或自抢占。 |

版本和移植边界：LTR 本地计数来源固定于 `hao-ai-lab/vllm-ltr@13bbf6ff3dab661791d41362551b089e5f77c91c`；对应官方 scheduler 本次网络读取失败，沿用本地来源注释，不声称重新取回。UniBoost 既有核验固定 `yl3469/sglang@e2a17c3e342cc47dfcbfd8b13990ba94fcb1a4da`，当时只在 schedule_policy.py/test 找到 Boost/Gamma/cache-frontier，未找到论文 MemGuard；本次该固定源码网络仍不可读，因此几何规则应标“论文组件”，不能称官方代码完整移植。以上旧源码范围见 [既有核验](../../RELATED_WORK_RECOVERY_20260914.md)。BidKV 本地固定原件和公式见 [SOURCE.json](../bidkv_pinned_sources_20261002/SOURCE.json)。

可被实验推翻的剩余差异是 q 的**可执行承诺**。若目标在恢复时缺少 H 页、接下来 q 个真实新输出需要 G_target(q) 页，且实际同时执行的 peer 集合增长为 G_peers(q)，则一个直接充分资金条件为 `F + actually_releasable(victims) >= H + G_target(q) + G_peers(q) + inflight_reservations`。其中 F 必须明确是否已扣 in-flight，防止重复计费；不能用“q 步后可能 EOS”的预测释放作确定资金。跨 q 调用的承诺还要约束其它准入不消费这笔资金、给目标真实执行额度，并处理早停释放。若任意 peer 可自由增长，单纯 target-hold 不构成此性质。这个条件是建模候选，不把加法内存核算本身包装成贡献。

便宜且最强的简单量子对照：**同一个 oldest-funded 恢复器 + 固定新输出 Q + 完全相同的增长资金与执行保护**。Q 用小范围开发集选点后固定（起点建议 Q=1、一个物理 block 的 token 数、四个 block；仍需根据真实恢复时延/步时确认范围）。唯一区别是 Q 不随当前恢复成本/peer 状态调整。若 adaptive q 只赢 Q1，却不赢这个选好的固定 Q，则只能归因更长保护，不能归因联合决策。必须报告所有固定 Q 点和完整请求成本，不把离线选到的最好 Q 当部署策略。

首先仍运行已准备的 ordinary free-fit 对照，回答当前 Q1 收益是否仅由绕过队首获得。随后在同资金规则上对比固定 Q 与候选 q；若仍有增量，再加入使用绝对累计新输出几何边界的 UniBoost MemGuard 论文组件。单独几何 quantum 的对照不能重置为每次恢复从零，否则变成另一算法。当前已有 LTR T/Q 负结果只约束该本地 selected-offload 组件与已测域，不否定完整 LTR，也不能跳过同新底座的强简单 Q 对照。

决定继续的证据应是完整请求 gap/throughput/completion 前沿改善，同时有真实事件说明 q 改变了恢复或让出时机；仅恢复次数减少、量子变长、重算变少都不足。没有传输/执行时长上界时，只能陈述给定资源与公平推进条件下的有限调度调用进度，不能声称严格墙钟 SLO。
