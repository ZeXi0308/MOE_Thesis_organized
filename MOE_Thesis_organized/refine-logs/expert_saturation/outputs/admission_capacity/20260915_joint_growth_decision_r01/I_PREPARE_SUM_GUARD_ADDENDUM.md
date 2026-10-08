# I 的全体 running 增长预筛：一个 native-loop 反例

**结论：反例成立。该 sum 预筛不是共同 staged-save 后端合法性的必要条件，会过滤当前本可合法完成的 prepare→commit。建议下一版本删除这一额外预筛，保留执行后和 commit 的实际进度、归属、队列及 flush 检查。** 这属于移植范围修正，不是 LTR、eager 或研究问题的负结论，也不是删 guard 会改善完整服务的证据。I 已接受包没有修改；GPU 测量仍为 0。

证据是 **CPU_PINNED_NATIVE_SCHEDULE_FAKE_ALLOCATOR_OUTPUT_TRANSFER**。复用已有 `verify_native_recovery_execution`，运行 captured/pinned native 的 `schedule`、`_preempt_request`、cached payload、post-schedule 方法及 I 实际 adapter closures；只在独立内存副本中移除一条 `sum(r.remaining_blocks for r in rows) > free` 条件。两个分支从同一给定状态独立执行两次调用。allocator、同步输出、connector store 注册及 flush metadata 是明确的 CPU 替身；未执行 native OffloadingConnector 传输、加载完成或 GPU tensor 路径。

入口：[fixture](check_prepare_sum_guard.py)、[完整两分支结果](I_PREPARE_SUM_GUARD.json)。结果保留所有 native allocation failure、preemption、丢弃 computed 状态、adapter 决策、store/flush metadata 和两次调用后的状态，没有选择性丢掉 peer。

## 给定状态及真实 native 循环

4 块包括 null block 0；free=0。运行顺序和 arrival 次序使 V 先于 peer，target 已 boosted。

| 请求 | 状态 | prompt / output / computed | 持块 | 下一次正分配所缺块 |
|---|---|---|---|---:|
| V | RUNNING | 16 / 16 / 31 | 1、2 | 0 |
| peer | RUNNING | 8 / 9 / 16 | 3 | 1 |
| target | PREEMPTED | 16 / 1 / 0 | 无 | 2 |

新预筛计算 `0+1 > free(0)`，因而拒绝。可是共同 `prepare` 契约已可成立：V 的 2 块能为 target 的 2 块需求提供资金，完整可保存前缀是 16 个位置、物理块 1。G 和 I 的 `staged_save_contract.py` 逐字相同；G 后端是在 native 调度后检查 victim 是否恰好执行一次，在下一次调用检查 victim/target 状态和归属，没有“全体 running 均不得缺块”的要求。这里证明的是共同后端合同，**不声称完整 G selector 一定会从这个小 fixture 选中同一动作**。

删除该预筛的内存分支：

1. **call 0 / prepare：** native 先给 V 分配 1 个位置，V 不需新块。随后 peer 申请 1 个新块失败（required=1、free=0），native FCFS 抢占尾部 peer，释放块 3。peer 的 GPU computed=16 被丢弃。V 仍持块 1、2，native 分配结果恰为 `{"V": 1}`；CPU connector fixture 注册前缀 16 / block 1 / job 7，实际 `inspect_store_delta` 返回 `NEW_STORES_VALID`。CPU 返回后 V 为 computed=32、output=17，free=1，target 未变。
2. **call 1 / commit：** 共同 `commit_reason(..., save_enabled=True)` 返回 `READY`，I adapter 自身也记录 `commit_check=READY`。V 恰好多执行一次，仍 pure decode，保存前缀和物理归属没变；`free1 + V持2 >= target需2`。实际 native `_preempt_request(V)` 释放 V 的 2 块并记录强制抢占，metadata 含 `flush=[7]`。随后 native 等待循环合法分配 target 的 17 个位置，占 2 块，余 free=1。

最后这 17 个位置是 CPU native 调度中的重算路径，**不是实测加载、不是异步恢复完成，也不是完整服务收益**。它只补充证明 commit 后资源分配合法；关键反例在共同 contract READY 和真实强制 preemption 时已闭合。

## 对照分支及不能省略的成本

| 两次调用内的结果 | 保留 sum guard | 仅去掉 sum guard |
|---|---|---|
| call 0 | 拒绝 prepare；V 仍执行 1，peer 仍被 native 抢占 | prepare V；V 执行 1，peer 被 native 抢占 |
| call 0 后 free | 1 | 1 |
| call 0 注册 V 前缀 | 无 | 16 位置 / 1 块 |
| call 1 | 才 prepare V；注册 32 位置 / 2 块；尚未 commit | commit V；flush job 7；target 分配 17 |
| peer GPU 丢弃 | 1 块、computed=16、1 次 native 抢占 | 同左 |
| V GPU 丢弃 | 两调用内尚无 | 2 块、computed=32；前缀 16 仅有 CPU store/flush 证据 |

因此 guard 在本例**没有避免 peer 抢占**，却使保存准备和可能的 commit 晚一个调用。它也没有永久消灭后续动作：保留 guard 的分支下一调用已接受 prepare，并保存更长前缀。提前分支可能多付尾部重算等代价；两调用 fixture 不决定这些代价的完整服务净值。peer 丢弃的 16 个 computed 位置也不能直接计成未来必然重算 16 或相应时间，后续真实 host 命中与执行尚未测量。

## 对版本和研究选择的影响

把 sum 预筛称为“必要安全检查”已被这一原生调度反例否定。若保留它，就等于额外引入“同一步所有 running 增长必须一起满足”的调度策略，改变了与 G 共同后端的可执行动作集合。下一包应删这一非必要条件，继续复用 victim 恰好执行一次、保存源归属、下一调用 commit、pending store flush、无不相关 pending load 等原有约束；不移除这些实际约束，也不需要新增控制器。

本轮唯一问题已闭合，不追加扫描或大审计。下一最小动作由 root/唯一执行方完成移植修正版与原单格路径资格；**CPU 反例支持修正移植，不能替代 GPU 保存/加载资格或参数性能校准**。

复算命令（输出使用新文件，不覆盖现有结果）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_joint_growth_decision_r01/check_prepare_sum_guard.py --output /tmp/I_PREPARE_SUM_GUARD_recomputed.json
```
