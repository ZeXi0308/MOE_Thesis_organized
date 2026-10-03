# 2026-09-29 H1 提交动作的调度 token 预算边界

证据层级：`CPU_COMPONENT`。本页接续 [本轮交接](README.md) 和 [结果](RESULTS.md)，不修改旧 D/E 或已接受 G/H、LTR r02 的原件。H1 的 KV、槽位与 connector 门禁只证明本次直接恢复的物理资格；它不保证恢复目标与 off 臂在同一调用到达首个新输出。

## 可复算反例

[B 的定向脚本](B_H1_TOKEN_BUDGET_TEST.py)从共同源码和 H1 候选包分别提取实际 `_direct_resume_reason`。31 个 pure-decode running 请求（含计划 victim）各需 1 个调度 token，32 个序列槽尚余 1 个，GPU 空闲块 100，目标完整历史需 994 个 token / 63 块，假设无 host KV 命中。两份 H1 gate 都返回 `DIRECT_READY`。在固定 1024 token 调度预算、running 先于 waiting 的原生顺序下，direct 给目标 `1024−31=993`，off 先驱逐 victim 后给目标 `1024−30=994`。因而 direct 在此条件下本轮不能完成目标的历史重算，off 可以到达生成边界。两份脚本结果均 exit 0；脚本执行真实 gate，但调度和 host-miss 后果是 CPU 条件模型，未执行 native worker 或 GPU。

旧 D859/E1399 派生快照只留目标需 95/106 块与 running 26/22，没有精确 prompt/output/computed、host 命中和实际输出。若 PREEMPTED 无 GPU 块且无 host 命中，目标 backlog 分别界于 1505–1520 / 1681–1696 tokens，direct 本轮余量 998 / 1002，off 为 999 / 1003；这两个条件实例均不能一轮重算完。不能由此推断旧事件实际首输出被延迟、可执行 direct 频率或完整请求收益。

同日已恢复的 G 原生诊断逐次快照提供了另一种更直接的动作空间约束：[B 的只读复算](b/EVIDENCE.md)将 2650 个调度调用与 2650 个即时快照对齐，54 次 READY commit 的目标所需块数都大于当时空闲块，缺口 2–193 块。故这条已执行诊断轨迹中 H1 的直接 KV 资助必要条件为 **0/54**；它不是 H1 执行后的负收益。G/H 轻量格另有 126/690 次已执行 selected commit，却无逐次资源快照，合并可识别上界由 `0..870` 收紧到 `0..816`，仍不是发生率。

## 修订后的判别

`DIRECT_READY` 继续仅表示当前物理 KV、槽位和 transfer/ownership 合法。撤销“目标首新输出必不晚于 off”的无条件预测。暂保留默认关闭的 H1 gate 作为一个待资格动作，因为强行要求整个历史在一次调用内装下会抛弃有意义的分段恢复，而且 host 命中和 connector 延迟仍会改变首输出。原生资格应记录 commit 时的调度 token 上限、running 消耗、目标真实需调度量与 host 匹配/加载、目标实际获分配量、首新输出时刻、victim 与其他请求的间隔；缺少这些信息时不作无延迟归因。

下一轮仍只运行已接受 LTR r02 的单格原生生命周期资格（须先核实可用机器、准确包和授权），随后在同一机器/输入/selected 后端固定 H1 off/on，并列 native-full 效率和经 G 校准的兼容 LTR-style。自然 direct 事件若没有、目标延期或 peer/host/未来资源代价抵消收益，就收窄或放弃 H1；即使有正效应，也须证明它超出“空块与槽位足够便直接恢复”简单规则，不能将该简单规则改名当新算法。G/H 与 D/E 已见过，不能作为新假说盲测。

[C 的包兼容检查](check_package_compatibility.py)已在本地核对 manifests 与静态同输入/后端：已接受 H128 与 H1 候选兼容；LTR r02 G64 与 H1 H128 在输入文件和请求数上不同，只能先做生命周期资格，不能直接比较性能。H1 候选尚未接受 GPU 执行。SSH 新主机名当前本地 DNS 无记录、预算范围待用户确认，本轮 GPU 使用为零。
