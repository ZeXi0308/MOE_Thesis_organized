# 恢复完成以后，还获得了多少有效服务

2026-09-14。主问题 OPEN；本轮为已有 NATIVE_SERVING 原件的新增生命周期分析，MEASUREMENT_ONLY。没有新方法收益结论。

当前回答：**首输出保护能兑现恢复，但没有保证足够服务。已测 fit-scan 和首输出保护均存在恢复数千位置、仅输出一两个 token 后再次丢弃且重执行的生命周期。值得验证的是在可执行窗口内能否避免未来重复成本，并控制其它请求损害；不能按已付账单强制保留。**

## 仓库与接续

本轮从 `agent/publish-current-moe-code`、HEAD `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac` 开始；`git ls-remote` 核实远端分支也为该提交。共享工作区有其它会话的更晚修改。主实现隔离在 `/private/tmp/moe-window-main-20260914`。已读 current/ideas 入口、SERVING_RESOURCE_STUDY、RESULT_LEDGER、GPU_COORDINATION 及四个指定方向的实际报告和 raw。本轮不修改科学权威入口，不覆盖已有证据。

更新后的事实：LTR packing 四格、首输出保护四格、KV 往返微基准、默认 native prompt-offload 四格及后续定位均已经完成。资源进展模型的 26 条轨迹验证是旧固定长度域回溯验证；62 个停止点是有限候选面，均不是全局 Oracle。原生 offload 实测只覆盖 `offload_prompt_only=True`；配置 host 容量 16 GiB，实际 host 峰值仍未闭合。不能将微基准复制速度、已少掉的重算位置升级为完整请求收益。

本轮唯一新增问题：恢复中的内部工作在何处被复用、何处在首输出之前丢失、何处已经产生输出但随后需要再次恢复？最弱因果环节是“首输出兑现”到“有效服务摊销”。实验只读已有 8 格，按实际下一次抢占闭合 residency，再连接下一段实际计算前缀。没有沿一条旧轨迹评价另一策略。

## 分账结果

[analysis.json](lifecycle/analysis.json)保留 8 格、256 个完成请求、全部恢复段。每种规则两个重复的下表计数相同；仍是同一旧 workload，不是独立文档样本。prefix 原组与 restore-off 后组的计数一致也不增加 workload 数。

| 每格规则 | 重算位置总数 | 首输出前丢弃：段数 / 位置 | 仅 1–2 新输出后再抢占：段数 / 新输出 / 重算位置 | 已服务后再丢弃的重算位置 | 服务至完成的重算位置 |
|---|---:|---:|---:|---:|---:|
| fit-scan + 原 200/10 | 75,742 | 2 / 5,973 | 11 / 14 / 37,851 | 48,329 | 21,440 |
| rank-prefix + 原 200/10 | 74,982 | 6 / 6,978 | 0 / 0 / 0 | 42,578 | 25,426 |
| rank-prefix + 首输出保护 | 96,955 | 0 / 0 | 4 / 4 / 13,767 | 67,554 | 29,401 |

互斥守恒是：`重算总数 = 首输出前丢失 + 已输出后丢弃 + 服务至完成`。短服务列是“已输出后丢弃”的子集，不能重复相加。fit 的 37,851 和 guard 的 13,767 个短服务段恢复位置，在各自下一 residency 均实际重新计算过；这是工作位置重执行证据，不是可回收毫秒数。

部分重算并非总是零价值。fit 的 22 个恢复链、prefix 的 20 个、guard 的 27 个均出现已有部分 prefix 跨后续调用继续使用。fit 两个零输出中断链也在链内复用了部分 prefix，最后才被丢弃；因此“内部复用过”和“最终未产出新 token”可以同时成立。关闭 APC、无 host KV offload 的这组原件中，实际 preemption receipt 记录 computed=0、blocks=0，输出历史保持不变；不据此推断其它后端也会丢失状态。

例子：fit 中请求 `0003640` 在 step405 开始恢复，408 返回 1 个新 token，409 再次抢占；此前重算 3,273 个位置，后续又重执行这些位置。guard 中同请求在 405–408 恢复并返回 1 个 token，411 再次抢占，同样重执行 3,273 个位置。两者各有真实输出，不能将其称为零进展；也不能将首次恢复后的 1 token 当作服务摊销已经成立。

已有首输出保护主结果仍保留：max ITL 6.669/6.887 → 4.387/4.141 秒，吞吐 −7.104%/+2.309%，平均完成 +8.300%/−1.426%；每对 27/32 请求自身 max ITL 变差。这里是延迟—效率与损害分布的交换，不要求所有指标同时改善，也没有稳定净收益结论。不能用跨组 fit 数字宣布胜出。

时间统一为同步 `LLMEngine.step` 返回后的 host 记录值。它不是客户端收到时间。调度后 computed 计数不能单独确认执行：本分析连接成功 engine call；早期 memory telemetry 的 `model_execution_confirmed=None` 不被改写为 True。恢复调用 inclusive 时长只作诊断，包含同 batch 其它工作及主机/调度时间，不当作纯重算税、不在请求之间累加。

## 最小模型及反例

模型实现与实跑实例见 [模型报告](../20260914_service_window_model_r01/REPORT.md)、[examples.json](../20260914_service_window_model_r01/examples.json)。它只做少量当前状态动作资格；未接入每步在线调度，也没有将旧全轨迹 Python 模型当在线预测器。

状态是当前有效 GPU/host prefix、实际持有/空闲块、host 使用/上限、目标/peer 历史与 output age；动作给出后端已约束的恢复依赖 DAG、同 batch 的新输出事件和执行成本。只有实际新输出重置年龄。恢复期间与后续 KV 增长分开检查，时间按依赖路径取 max，不直接加可能重叠的计算与传输。模型不能凭输入 profile 给未恢复 peer 创造输出；本版输出 peer 限定 resident pending=1。

设剩余、额外、暴露的恢复税为 C，每新输出可接受该税 α，则 `L=max(1,ceil(C/α))`。`U_KV` 来自所有共批请求的增量占用，`U_age` 来自各请求在窗口内的输出间隔。仅当 source 有效且存在 `L ≤ n ≤ min(U_KV,U_age)` 才有满足此预算的机会窗口。C 缺测则不判定摊销；不能把整段恢复调用耗时填作 C。首输出延迟 r 已包含第一 token，窗口末为 `r+(n−1)τ`。

合成支持例：C=4ms、α=1ms/token，L=4；共批给 peer 及时输出，4-token 窗口满足 KV/等待，1-token 窗口不能满足摊销。合成反例：可用块降低后，仅允许 3-token 窗口，`U_KV=3<L=4`，因此这个动作不合格。毫秒/预算均是测试输入，不是 vLLM 测量或所选业务 SLO。真实 step406 仅证实 `142+3≤148` 块、`994+30=1024` token 的当前共同执行可行性。

未知 EOS 下 n 只是可提供的服务机会，完成即释放。执行中只比较从现在到相同未来服务量的保留/切换成本；已经付出的 C 不作为继续保留理由。未来再恢复成本无法从可见状态界定时，保留简单策略和不确定性；新到达、KV 紧张、无效 host prefix 或预测失准触发重估与原生资源回退，不能盲目坚持窗口。

## 近邻与下一项真实区分

[近邻报告](neighbors/REPORT.md)核对 LTR、Andes、UniBoost/MemGuard、TokenFlow 的原文和已取得的官方源码。等待提权、最小有效服务、恢复成本与其它请求损害比较均已有先例。UniBoost 完整官方实现未取得；各论文完整系统的同预算对照仍未完成。当前能主张的是本仓库组件接入的生命周期缺口，不能归因给原论文。

本轮截至 UTC 20:03 的共享终态记录已确认 `fit / guard_all / guard_residual` 反序六格全部 COMPLETE/exit0、192 请求完成，finished=1789329697.549690，冻结包 `f2e20f73711fd5e54906309b90fa7cea845264f17229d1db0a6138db0cac7450`。原执行会话正在统一回读；本会话不另起 driver、不重复拉包或占用 GPU，尚未取得本组完整 raw 分析。它区分：保持首输出 KV 义务时，把本步预算留给可共批 resident pending=1 工作，能否减少额外停顿并保住完整服务量。不能把终态完成写成机制收益。

- 若 guard_residual 保持恢复兑现、改善其它请求且改善相对 fit 的权衡，先保留这个简单执行规则；此结果仍不证明窗口模型必要。
- 若它仍反复恢复后仅得少量输出，才以相同后端、相同 victim 规则比较固定输出服务量、已有几何保护及有限动作窗口模型；简单输出等待计数修正单独消融。
- 若 fit 或旧最强轮转已覆盖收益，吸收强基线，停止包装额外模型。若资源/执行预算不够，只否定该动作在当前状态可行，不宣称负载全局无解。

下一组结果不得用不同 host 能力拼接排名。未知 EOS、持续到达、异构上下文和第二模型尚未完成；首个闭环后应升级这些运行域，而不是继续旧封闭集合的阈值扫描。

## 本轮交付与裁决

- 新增 [生命周期分析器](../../../experiments/admission_capacity/analyze_effective_recovery_service.py)；从实际抢占、逐步 computed、引擎返回与输出事件重新分账。
- 新增 [4 个生命周期定向测试](../../../experiments/admission_capacity/test_effective_recovery_service.py)和 [6 个模型测试](../../../experiments/admission_capacity/test_service_window_model.py)，合并 10/10 通过；实例输出与独立 worktree 完全一致。
- 运行命令见 [COMMANDS.md](COMMANDS.md)。初次开发检查把旧 telemetry 的 None 当失败，依据冻结 producer 源码修正为成功 engine receipt；没有产生或修改旧 raw/旧分析。
- 本轮实现范围说明：旧分析器只有 residency 计数，缺少前缀持有、实际失效与下一次重执行的连接；旧全轨迹模拟又不能充当轻量资格器。新增核心仅完成这两个缺口，测试/实例使总代码略超约 500 行；没有新增 controller、pager、预测器训练或原生 kernel。更小的单 token 计数不能回答内部复用与末端损失，单步块检查也不能表达服务下界和等待/KV 上界冲突。
- fresh Sol 限定复算未导入待审分析器，8 格 14,984 个调用与所有 residency 数值完全一致、无 P0/P1。该复核为 same-family/provisional，不升级科学证据层级；审计文件保存在本目录。

| 裁决字段 | 值 |
|---|---|
| Verdict | OPEN / MEASUREMENT_ONLY；首输出完成性与有效服务摊销分离 |
| Evidence type | 8 个已完成 NATIVE_SERVING cell 的新增只读分析；CPU 模型资格 |
| What was measured | 内部 prefix 复用、实际新输出、下一次失效、下一 residency 实际重执行 |
| What was not measured | 新窗口策略真实收益、host 实际峰值、全历史 offload、未知 EOS/新负载/第二模型 |
| Strongest baseline | 本数据中的 fit；其它既有 native/most/least 仍需同组比较；完整近邻尚未排除 |
| Oracle/headroom | 有当前动作可行性，不存在已验证全请求 Oracle |
| Claim ceiling | 本组件底座在当前域中的恢复生命周期与服务缺口；非新颖性或客户端 SLO 结论 |
| Failure category | 恢复前等待、短服务后的重复恢复、对其它请求的成本转移 |
| Resurrection condition | 问题仍 OPEN；只在强简单规则后保留真实决策残差时增加窗口机制 |
| One next smallest experiment | 接续已有统一六格结果，检验 KV 保留与执行预算分离；不重跑已完成 packing/offload |

一次恢复确实可以变成新输出；当前证据同时表明，**新输出一次不等于足够服务，更不等于延迟—效率边界已改善**。
