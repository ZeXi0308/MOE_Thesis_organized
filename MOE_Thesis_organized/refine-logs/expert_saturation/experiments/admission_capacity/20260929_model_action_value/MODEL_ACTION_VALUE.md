# 提交时直接恢复：最小动作模型（CPU）

本文件只属于模型与动作价值角色。接续依据为共享 `RESULT_LEDGER.md`、A 的 `20260929_commit_recheck/README.md`、同期 B 的 `b/EVIDENCE.md`、G/H 完整请求结果；不修改这些文件。当前仓库为 `agent/publish-current-moe-code@76d6d88`。G/H 的 selected/eager 是**已测**停顿优先简单参照；native full/no-extra-rotation 仍在吞吐、平均完成及多数请求自身 gap 上占优。兼容 LTR-style 的真实保存/加载/量子生命周期和完整性能仍 `UNRUN`。已停止 cooldown/window/headroom/growth predictor，本模块不重新搜索这些策略。

## 决策、守恒和未知成本

仅保留一个自由度：A 的 prepare 已接受且旧 commit 仍 `READY` 时，比较 `a_old=(同一target,{planned victim},selected-native,native q)` 与 `a_direct=(同一target,∅,selected-native,native q)`。`decision_model.py` 消费 A 的 `staged_save_contract.Plan/RequestState/commit_reason` 和 `service_window_model.Request`；`StateSnapshot` **需要** native pre-commit 观察者填充，当前尚未接线。它检查即时 `free >= target.remaining_blocks`、16-token 块与可用序列槽、队列和 connector retirement、非 pending-load 状态、目标/受害者版本与所有 running 的物理块独占证明。任何未知或失效字段保留旧 swap 或取消过期 intent；`validate_for_commit` 重检离散状态，允许无新输出时观察时间与 age 正常前进。模块不操作 native 对象、不假装 CPU fixture 已验证 GPU 生命周期。

GPU 块账为 `free_after_direct = free_before - native_need(target)`；原 victim 的已持块继续由它占有。预测只限本次**自定义**驱逐可被省去，后续 native 分配仍可能抢占它或其它 peer。A 的有效 GPU 前缀与 pending-load 已占块严格分开；computed、store 完成和部分重算都不算新输出，`output_age_ms` 只随真实输出 receipt 更新。已经登记/完成的 store、host buffer 中的有效旧前缀均不按永久丢失或可删成本计费。没有 action-specific 暴露关键路径、host live 占用和 EOS 距离时，`exposed_marginal_ms` 与完整请求收益为 `UNKNOWN`；不把混合调用、传输字节、少一次驱逐换算成吞吐毫秒。direct 不新增 host staging 时，旧准备的 host 成本仍保留；若 connector 需要新增 staging，必须给真实 host 占用和预算，否则回退。

## 与最强简单规则的动作差异

旧台账的 50 个 commit 中 D859 `free193/need95/victim233`、E1399 `free277/need106/victim184`。两次的**块余额**分别为 98、171，旧 selected/eager 路径仍驱逐了 victim。`predict_before_execution.py` 将它们固定成**条件预测**：只有槽位、物理归属、队列及 connector 屏障在同一 commit 实际通过，本模块才输出 `DIRECT_RESUME`，也就是保留 victim、恢复相同 target；没有这些前态原件，`actual_direct_legal=UNKNOWN`。旧 victim 后续 gap 0.112/0.261 秒且非各轨迹最大值，不能预测最大 gap 改善。G/H 的同类 commit 频率无法从当前轻量摘要识别，B 的证据把 selected 历史 870 次动作的可识别界保留为 `0..870`，不是发生率估计。

这次动作差异**相对于已测 selected/eager**存在；它已被“提交时有空块和槽就直接恢复”的更强简单规则完全覆盖，兼容 LTR-style 组件也允许该动作。因而这里没有新增效用打分器、阈值或第二个 controller。如果公平的简单 recheck/LTR 对照覆盖收益，就删除模型层的任何额外排序，仅保留物理可执行性检查与成本分账。

同期 C 的 `20260929_commit_recheck/c/commit_recheck.patch` 有独立的 native gate，**并未调用本 CPU 模块**；两者当前资格集合不应默认为一致。执行端应以 C 的真实 native gate 为唯一提交裁决，或在集成时建立统一快照与同状态一致性测试。本目录的合成 `DIRECT_RESUME` 只是纯函数行为，不能作为 C patch 已执行该动作的回执。

## 执行前可证伪预测与判定

在一个上述完整门槛均通过的共同 commit 前态，direct 分支应无该 victim 的自定义强制抢占；若两臂目标均存活且产出下一个新 token，direct 的首新输出不应更晚。提前 EOS/终止须单列为未兑现恢复，不能用未来 EOS 当在线判断。prepared store 的已付成本仍入账。若 native 因原 victim 占块立即抢占它或另一个 peer、目标首输出较旧路径推迟、或完整请求 goodput/吞吐/mean flow 的代价抵消局部收益，则 H1 无实用动作价值。旧 D/E 是开发事件；G/H 看过，均不当盲测。需要 A 的同前态真实分叉或事前冻结的成对完整运行，并记录实际 owner/slot/store/load/flush、每请求交付时间、GPU/host 过程占用、输出长度/EOS。不能用旧 victim gap 离线减法替代执行。

完整请求对照由唯一执行入口串行安排：同资源、同到达 cohort、相同 selected 保存的 recheck off/on，并列 native full、selected/eager 和合理校准的兼容 LTR-style；阈值、计时边界和失败/未完成处理须在运行前冻结。**本轮 GPU/完整请求 H1 结果 UNRUN**；未 SSH、未启动 GPU、未改公共环境或主台账。

## CPU 复算

在仓库根目录执行：

```sh
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity/20260929_model_action_value -p 'test_*.py' -v
python3 refine-logs/expert_saturation/experiments/admission_capacity/20260929_model_action_value/predict_before_execution.py
```

8 项测试覆盖直接恢复、资源/槽不足、块归属或引用变化、部分加载、过期/EOS/取消、未知队列/connector/host staging 回退、内部计算不重置输出 age、旧数值只给条件证书。它们是合成 CPU 状态，不是 native load/flush、GPU 时延或完整收益验证。
