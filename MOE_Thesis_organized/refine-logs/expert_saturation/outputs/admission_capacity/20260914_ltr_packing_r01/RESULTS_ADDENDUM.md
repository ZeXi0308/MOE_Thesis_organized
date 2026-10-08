# Ranked-prefix / fit-scan 首次原生四格结果

Verdict：`MEASUREMENT_ONLY`。在同一custom FCFS/current-history-reservation/native-RECOMPUTE后端上，rank-prefix没有消除部分恢复被打断，最长请求停顿两次都更长；吞吐和平均完成差异符号翻转。主问题`OPEN`，不能据此否定完整LTR或资源调度问题。

Evidence type：`NATIVE_SERVING`，同步in-process vLLM0.26.0 / OLMoE-1B-7B BF16 / 单RTX5090；固定32请求、每请求3072输入/1024输出、steady 50ms、token预算1024、APC off、实际KV13960740864字节/6656可用16-token块。沿用旧d6文档，两个反序相关repeat，不是新holdout。

What was measured：两臂均LTR等待提权200/10，只有遇到当前不可行候选时continue（fit_scan）或break（rank_prefix）不同。四格共128请求/131072输出均完成，保留所有初始化、预热、调度、重算、完整请求等待和不利结果。

| Cell | Wall s | Requests/s | Mean completion s | Max request ITL s | Calls | Preemptions | Recomputed positions |
|---|---:|---:|---:|---:|---:|---:|---:|
| block0-d6-packing-fit_scan | 27.670858657 | 1.156451283 | 21.618075060 | 4.834701002 | 1862 | 22 | 75742 |
| block0-d6-packing-rank_prefix | 28.196696216 | 1.134884731 | 22.277188004 | 6.732402464 | 1872 | 25 | 74982 |
| block1-d6-packing-rank_prefix | 27.708315525 | 1.154887960 | 21.775504319 | 6.576920833 | 1872 | 25 | 74982 |
| block1-d6-packing-fit_scan | 28.234744502 | 1.133355395 | 22.144992097 | 4.933291839 | 1862 | 22 | 75742 |

rank-prefix / fit-scan：吞吐−1.864891% / +1.899895%；平均完成+3.048897% / −1.668494%；max ITL增加1.897701 / 1.643629秒。第一对32/32完成更慢、第二对32/32更快；两对均10/32请求max ITL更差、22/32更好。两臂内部实际调度路径和32/32完整输出相同，但fit repeat wall差+0.563886s，prefix差−0.488381s；这些相关repeat不是噪声界。

恢复段以同一请求下一次真实preemption截止，不能把后来恢复产出的token归入前次恢复。fit每次16段实际恢复后再次抢占，其中0/1/2新输出为2/8/3段；prefix每次18段，分别6/0/0段。完成收尾单列，不混入直方图。prefix六个零输出段实际执行重算994、995、996、997、998、1998位置；因此少了1–2输出短段不等于恢复已完成或最长停顿改善。

两fit分别1个有效提权epoch、两prefix分别8个；每个epoch均10次连续调度，3次纯重算、1次混合重算返回新输出、6次decode，实际返回7个新token。没有观测到量子耗尽仍无新输出，不以结果调大200/10。prefix持有而未执行resident累计916 request-calls，fit110。两臂fresh prefill98304和fresh decode32736相同，prefix重算只少760位置。

时钟守恒为wall = scheduler inclusive + engine excluding scheduler + outside engine。decision计时嵌套在scheduler中，不重复相加。四格scheduler总秒1.096043 / 1.209376 / 1.083290 / 1.120247，decision总秒0.248181 / 0.261837 / 0.252802 / 0.257707，均已包含在上表wall。

What was not measured：业务SLO-goodput、生成质量、第二文档组/arrival regime、完整LTR predictor/CPU SWAP、全动作Oracle、生产P99。两对仅25/32跨臂完整输出相同，7条数值轨迹不同，不能宣称质量等价。三段应用warmup已完成，但现有warning_once日志不证明measurement内JIT或机器码重编译，也不用于解释wall漂移；参见前组[JIT语义勘误](../20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md)。

Strongest baseline：本组相同底座fit-scan；加入rank-prefix是补足最近邻LTR容量选择语义的组件对照，并非重现原LTR的完整容量预测、聚合保留与SWAP后端。既有d6 native/least/most/headroom结果保留在共享台账，不把跨组wall当当前匹配比较。

Oracle/headroom status：全请求Oracle未运行。实际零输出重算段给出可作用的残留位置，不能把这些局部重算直接相加为可实现的请求收益上界。

Claim ceiling：只支持上述旧固定预算/cohort下packing动作、恢复生命周期与请求级权衡。原件实际计划=执行、token/KV守恒和过去信息计数已核验；冻结runner未记录resolved long_prefill_token_threshold，所以主分析明确为`LIMITED_ACTUAL_PLAN_AND_RESOURCE_RECEIPTS`，未声称逐步重建完整选择器计划。

Failure category：当前rank-prefix组件未解决恢复中断且最长停顿更差；这是动作规则的局限，非现象不存在、完整LTR失败或整个问题NO-GO。

Resurrection condition：只有新的动作或运行域改变已测恢复中断因果链才重测；不以更换seed或量子阈值救原四格。

One next smallest experiment：在同rank-prefix/200/10后端中，仅将已经实际选中的有输出历史PREEMPTED请求保留到首个新输出可见；off/on/on/off，首输出后立刻回到原优先级，不延长到后续decode。先验证过去信息状态机、资源保留及实际0输出中断变化，再看完整请求收益和其它请求损失。

研究问题的本轮直接回答：rank-prefix自身不能在该固定KV域同时解除已有请求长暂停并稳定保住服务量；实际部分恢复仍被打断，首输出完成边界这一动作仍需真实验证。

输入包SHA256 `1fea4a612af77a0062163cec33566765c2df25ade4021407756ec43b1734ef73`；完整回读SHA256 `2825c8181d893af9482adbc6b5f07e027dea465f3631fb6036b690d75179cacc`。原始文件见[readback](execution/readback)，派生数据见[analysis.json](analysis/analysis.json)。原准备报告保留其历史UNRUN状态；本addendum才记录首次执行结果。
