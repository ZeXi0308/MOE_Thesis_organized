# 两个短恢复段的直接来源

**两段都是强制恢复的 target，均成功加载已有 host KV；产生 2 / 1 个新输出后，因为其他运行请求增长所需的 GPU blocks 无法满足，被 native scheduler 再次抢占。** 当步 selective 只返回 `swap cooldown`，没有新轮转计划。这里支持继续研究恢复后的资源增长与执行分配，不能将短段归因为保存/加载失效，也不直接支持增加保护窗口。

| 事件 | `memory-train-article-0000406` | `memory-train-article-0000748` |
|---|---|---|
| 已有保存 | prepare 728：2,464 个位置；store job 116 已完成 | prepare 942：3,792 个位置；store job 129 已完成 |
| 强制恢复身份 | prepare 898 / READY commit 899，victim=`0000271`，target=`0000406` | prepare 1157 / READY commit 1158，victim=`0001582`，target=`0000748` |
| 真实加载与执行 | load job 128 已完成；step 901 从 2,464 开始，重算 17 个位置并输出；step 902 再输出一次 | load job 139 已完成；step 1160 从 3,792 开始，重算 7 个位置并输出一次 |
| 首输出保护解除 | step 902 已无 protected target/reserve | step 1161 已无 protected target/reserve |
| 下一次抢占 | native，step 903，GPU free=0，释放该请求 156 blocks | native，step 1161，GPU free=0，释放该请求 238 blocks |
| 同步资源触发 | 6 个 peer 的下一 decode 各跨入一个新 block；下一快照 free=150 | 5 个 peer 的下一 decode 各跨入一个新 block；下一快照 free=233 |

两次受害 target 自身下一 token 均不需要新 block：`0000406` 已计算 2,483 个位置，持有 156 blocks，可容纳 2,496；`0000748` 已计算 3,800，持有 238 blocks，可容纳 3,808。相反，step 903 的 `0001582/0002487/0002680/0003790/0004150/0004787` 和 step 1161 的 `0004150/0004787/0004818/0005057/0005122` 位于各自 block 边界；原件确认它们在对应步执行了一个 decode token。资源变化分别是 `0 + 156 − 6 = 150`、`0 + 238 − 5 = 233`。这些步没有执行 prefill 或历史重算。

`0000748` 的前两步还直接显示了积压：step 1159/1160 的 free 为 4/1，target 保留 1 block；若干 peer 停在同一 block 边界。target 产生首输出后，reserve 解除，下一步 native 抢占使这些 peer 能继续增长。`0000406` 则是在首输出后 pool 已为 0，随后新的 peer 边界需求触发抢占。二者都说明“恢复至首输出所需资源已满足”不足以保证随后共同执行能够持续。

边界：失败的 `allocate_slots` 调用参数没有记录，因此不声称直接观测到哪一个 peer 首先触发 allocator 失败；peer 需求来自同一步已有状态与实际调度，并由 block 收支闭合。GPU 历史被释放也不等于 host 缓存被清空：这两次加载已经明确复用了 2,464 / 3,792 个历史位置。本笔记不推算延长驻留的全请求收益，不重算主表，不判断整个运行域的普遍发生率。

原件：[主分析](../execution_weste_26862/analysis.json)、[raw](../execution_weste_26862/readback/results/diagnostic-current/raw.json)、[selective](../execution_weste_26862/readback/results/diagnostic-current/selective-store.json)。仅保留这两例对应边界前后 2 步及必要保存/加载身份链接：[source.json](source.json)，复算：[extract.py](extract.py)。
