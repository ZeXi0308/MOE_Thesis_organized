# 新文档恢复调度强基线：八格实测补记

Verdict：`MEASUREMENT_ONLY`。本组两次反序中，most_output 在预先指定的完整请求吞吐、全局最大请求 ITL 两轴均优于 fit_scan 和 guard_residual；停止将当前首输出保护与预算预留组合作为独立方法胜出包装。most_output 的平均完成更慢，不能扩写为所有指标或所有请求占优；KV 压力下的暂停—服务量问题仍为 `OPEN`。冻结 [REPORT](REPORT.md) 的准备/运行状态作为历史保留，本补记以实际八格终态为准。

Evidence type：`NATIVE_SERVING`，单 RTX 5090 / OLMoE BF16 / vLLM 0.26.0。cohort3 为排除前 128 文档后的同一固定 WikiText train shard 新文档，32×3072 输入、每请求 1024 输出，50ms steady，cap32、token budget1024、APC off；实际 KV 13,960,740,864 bytes / 6656 个可用 16-token 块。各臂独立引擎、独立状态演进、相同预热与 reset；实际资源参照旧 d6 的 [most_output engine_args](../20260914_d6_strong_baselines_r01/readback/results/block0-d6-most_output/engine_args.json)。

What was measured：八格共 256 请求全部完成、262144 输出，失败 0；原生 FCFS、固定参数 most_output 轮转、原 LTR200/10 组件 fit_scan、同 200/10 的 rank_prefix＋首输出义务＋pending1 预算预留 guard_residual，按正序再反序执行。既有主分析八格均 COMPLETE/eligible，准备、身份、时间、资源、实际动作和会计检查通过；本补记只汇编已保存分析，不新增审计结论。

| cell | wall s | requests/s | output tokens/s | mean completion s | max request ITL s | calls | repeated positions | preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block0-native | 27.480188 | 1.164475 | 1192.422703 | 21.246994 | 14.397727 | 1869 | 21426 | 6 |
| block0-most_output | 25.327482 | 1.263450 | 1293.772488 | 23.287793 | 2.814454 | 1315 | 163764 | 44 |
| block0-fit_scan | 28.598838 | 1.118927 | 1145.780819 | 22.682030 | 5.044187 | 1862 | 75742 | 22 |
| block0-guard_residual | 28.117486 | 1.138082 | 1165.395803 | 22.062934 | 4.290791 | 1858 | 89448 | 25 |
| block1-guard_residual | 28.450275 | 1.124769 | 1151.763912 | 22.333150 | 4.368955 | 1858 | 89448 | 25 |
| block1-fit_scan | 27.738519 | 1.153630 | 1181.317576 | 21.809044 | 4.854511 | 1862 | 75742 | 22 |
| block1-most_output | 26.579243 | 1.203947 | 1232.841734 | 24.444272 | 2.941606 | 1315 | 163764 | 44 |
| block1-native | 27.065352 | 1.182323 | 1210.699221 | 20.818732 | 14.289536 | 1869 | 21426 | 6 |

[八个实测点的权衡图](analysis/tradeoffs.svg)：左图为吞吐与全局 max ITL，右图为平均完成与全局 max ITL；保留第三指标的损失，无拟合或择优删点。

wall 是请求测量窗口，包含窗口内调度、观察、engine 及调用间成本；模型初始化、预热、测后序列化与网络客户端不在该分母，不能称全进程成本。吞吐以全部完成请求/该窗口计算。各格新执行 prefill 为 98304 位置、decode 为 32736 位置；每请求首个返回 token 来自 prefill，返回输出与 fresh decode 不能混为同一计数。

全部请求分布如下，三项依次为 p50/p95/p99，均为单格 32 请求的线性插值，不能当生产 P99。逐请求值、时间边界与输出指纹保留在 [主分析](analysis/analysis.json)。

| cell | TTFT s p50/p95/p99 | mean TPOT s p50/p95/p99 | completion s p50/p95/p99 | request max ITL s p50/p95/p99 |
| --- | ---: | ---: | ---: | ---: |
| block0-native | 0.397767 / 0.864488 / 0.911079 | 0.019887 / 0.023684 / 0.024306 | 20.635869 / 25.093245 / 25.776192 | 0.083246 / 10.992781 / 13.732959 |
| block0-most_output | 0.334733 / 0.731514 / 0.767357 | 0.022520 / 0.023708 / 0.023926 | 23.449502 / 24.413007 / 24.581521 | 1.770283 / 2.436576 / 2.751340 |
| block0-fit_scan | 0.451657 / 0.902286 / 0.950329 | 0.021295 / 0.024771 / 0.025374 | 22.121520 / 26.242704 / 26.908025 | 0.107022 / 3.496356 / 4.746310 |
| block0-guard_residual | 0.349854 / 0.769924 / 0.806972 | 0.020695 / 0.024421 / 0.025034 | 21.449553 / 25.752394 / 26.416377 | 0.114909 / 4.230203 / 4.289435 |
| block1-guard_residual | 0.379051 / 0.788620 / 0.825580 | 0.020934 / 0.024710 / 0.025337 | 21.715138 / 26.066449 / 26.745713 | 0.108409 / 4.306670 / 4.367632 |
| block1-fit_scan | 0.353493 / 0.780751 / 0.817809 | 0.020504 / 0.024048 / 0.024663 | 21.241867 / 25.382358 / 26.047770 | 0.112494 / 3.375274 / 4.571603 |
| block1-most_output | 0.407405 / 0.878517 / 0.914650 | 0.023571 / 0.024847 / 0.025088 | 24.598805 / 25.640978 / 25.784508 | 1.875834 / 2.555386 / 2.876348 |
| block1-native | 0.342572 / 0.735625 / 0.771341 | 0.019501 / 0.023398 / 0.024038 | 20.208300 / 24.672187 / 25.362134 | 0.086127 / 10.994936 / 13.653132 |

以下 action 相对 baseline；吞吐正值更好，其余延迟正值更差。better/worse 是各自请求的实测变化，不以整组最坏值替代；所有跨臂输出差异和首个分歧位置保留在原 JSON。

| baseline → action | throughput change | mean completion change | global max ITL delta s | completion better/worse | max ITL better/worse | TTFT better/worse | mean TPOT better/worse | equal output sequences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block0-native → block0-most_output | +8.499% | +9.605% | -11.583273 | 5/27 | 7/25 | 31/1 | 5/27 | 25/32 |
| block0-native → block0-fit_scan | -3.912% | +6.754% | -9.353540 | 0/32 | 5/27 | 0/32 | 0/32 | 26/32 |
| block0-native → block0-guard_residual | -2.267% | +3.840% | -10.106936 | 0/32 | 5/27 | 31/1 | 0/32 | 28/32 |
| block0-most_output → block0-fit_scan | -11.439% | -2.601% | +2.229733 | 25/7 | 25/7 | 0/32 | 25/7 | 26/32 |
| block0-most_output → block0-guard_residual | -9.923% | -5.260% | +1.476337 | 26/6 | 24/8 | 0/32 | 26/6 | 25/32 |
| block0-fit_scan → block0-guard_residual | +1.712% | -2.729% | -0.753396 | 32/0 | 2/30 | 32/0 | 32/0 | 28/32 |
| block1-fit_scan → block1-guard_residual | -2.502% | +2.403% | -0.485556 | 0/32 | 26/6 | 5/27 | 0/32 | 28/32 |
| block1-most_output → block1-guard_residual | -6.576% | -8.636% | +1.427349 | 27/5 | 25/7 | 31/1 | 27/5 | 25/32 |
| block1-native → block1-guard_residual | -4.868% | +7.274% | -9.920581 | 0/32 | 5/27 | 2/30 | 0/32 | 28/32 |
| block1-most_output → block1-fit_scan | -4.179% | -10.781% | +1.912905 | 27/5 | 25/7 | 29/3 | 27/5 | 26/32 |
| block1-native → block1-fit_scan | -2.427% | +4.757% | -9.435025 | 0/32 | 5/27 | 0/32 | 0/32 | 26/32 |
| block1-native → block1-most_output | +1.829% | +17.415% | -11.347930 | 4/28 | 7/25 | 1/31 | 4/28 | 25/32 |

most 相对 native 两次吞吐增加、全局最长暂停显著缩短，却有 27/28 个请求完成更晚，且两次各 25 个请求自身 max ITL 变差。fit 与 residual 相对 native 两次均全部 32 请求完成更晚。residual 相对 fit 的吞吐与平均完成方向翻转：第一对全部 32 请求更早，第二对全部更晚；全局 max ITL 两次改善，但逐请求 max ITL 改善/受损为 2/30 与 26/6。输出相同不代表质量通过，跨臂输出不同也不自动等于质量损害。

两个 residual 各 25 个恢复义务均由首个新输出解除、无义务中断，各有 79 个受保护 selected request-call；但仍各有 5 段只产 1 个新 token 后再抢占。fit 的 0/1/2 新输出后再抢占段为 2/8/3，residual 为 0/5/0，两次一致。恢复段以实际下一次抢占关闭，首输出兑现没有保证后续服务或重算成本摊销。

| cell | decision s | wrapped scheduler s | scheduler inclusive s | selected request-calls | held resident-calls | released blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block0-native | 0.024065 | 未单独记录 | 0.921929 | 32884 | 0 | 1342 |
| block0-most_output | 0.200123 | 0.663901 | 0.977834 | 33000 | 0 | 10255 |
| block0-fit_scan | 0.386312 | 1.136606 | 1.455003 | 32932 | 110 | 4744 |
| block0-guard_residual | 0.347959 | 1.149049 | 1.449463 | 32943 | 186 | 5602 |
| block1-guard_residual | 0.354038 | 1.119194 | 1.424032 | 32943 | 186 | 5602 |
| block1-fit_scan | 0.328340 | 1.130183 | 1.425730 | 32932 | 110 | 4744 |
| block1-most_output | 0.252449 | 0.787259 | 1.158234 | 33000 | 0 | 10255 |
| block1-native | 0.022005 | 未单独记录 | 0.864033 | 32884 | 0 | 1342 |

decision、wrapped scheduler 均嵌套在 scheduler inclusive 内，不能相加后再加到 wall；native 的 wrapped 时间未单独记录。held 是 request-call 数，released 是释放事件累计块数，均不是持续时间或峰值占用。most 每次重算 163764 位置，明显多于 residual 的 89448，却只需 1315 次调用，而 residual 为 1858 次；不能把重算位置直接换成可消除秒数。完整测量窗口内的实现成本已计入主结果。

| same-arm block1 − block0 | wall delta s | throughput change | mean completion change | max ITL delta s | equal output sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | -0.414836 | +1.533% | -2.016% | -0.108190 | 32/32 |
| most_output | +1.251761 | -4.710% | +4.966% | +0.127153 | 32/32 |
| fit_scan | -0.860319 | +3.102% | -3.849% | -0.189676 | 32/32 |
| guard_residual | +0.332789 | -1.170% | +1.225% | +0.078164 | 32/32 |

同臂两次实际调度路径相同、输出序列 32/32 相同；跨臂调度路径均不同。上述反序时间波动全部保留，路径一致不构成噪声上界、统计显著性或非劣证明。

固定成本迁移使用旧 native 三项系数，不重拟合，条件起点为各臂实际 step299。全部比较在 [cost_transfer/analysis.json](analysis/cost_transfer/analysis.json)；直接 fit→residual 的预测与实测如下：

| fit → residual | actual conditional mean delta s | estimated mean delta s | actual conditional last delta s | estimated last delta s |
| --- | ---: | ---: | ---: | ---: |
| block0 | -0.348207 | +0.170174 | -0.210614 | +0.142884 |
| block1 | +0.475041 | +0.170174 | +0.662592 | +0.142884 |

第一对条件平均与最后完成的方向均估错；第二对方向相同但幅度不等。这里是从各自条件起点计算的区间结果，不等于上表从请求测量开始计算的完整 wall/mean completion 差。模型输入含各臂真实未来 schedule，属于事后成本诊断，不能作为动作预测、候选反事实或 Oracle；区间含 host gap，非末次完成调用采用区间末端，三项系数及分类跨度都不是物理 kernel 成本或不可消除下界。

[旧 d6 victim-cost 诊断](preparation_checks/victim_cost_identity/REPORT.md)只提供解释边界：固定相同 prompt 长度时，已输出量、当前重算历史与所占 KV 块共同增长，所选 most victim 均处于最大块数并列集合。那是旧 d6 前态候选枚举，不是 U 的替代执行结果，不能把本组收益归因为进度信号的独立价值。

What was not measured：U 未覆盖独立数据来源总体、可变长度与自由 EOS、另一到达过程、质量/业务 SLO、多模型、多 GPU、完整 LTR 或全部已知调度方法。这里未测全动作性能上界；合同中的参考 SLO 不作为事后挑选的赢家分数。后续 [context-victim](../20260914_context_victim_calibration_r01/REPORT.md) 与 [streaming](../20260914_streaming_recovery_r01/RESULTS.md) 已有独立结果，其运行域与证据不并入本组。

Strongest baseline：同组 most_output 是两项预定主轴上的最强已测简单策略；native 保留更早的平均完成，形成另外的权衡。most 不冒称最强已知方法；fit_scan 是相同组件资源 backend 的直接基线，不能冒称完整 LTR。

Oracle/headroom status：没有经过策略独立执行验证的全局 Oracle。现有主指标中，guard_residual 相对普通 most 未留下净增量；这只覆盖本 cohort、资源与四种冻结动作，不穷尽全部合法动作。旧前态枚举与实际未来 schedule 成本诊断均不构成性能 Oracle。

Claim ceiling：同来源新文档、固定长度与 steady 到达下，最强已测简单轮转覆盖当前恢复保护组合的吞吐—全局最长暂停两轴收益；同时保留平均完成与逐请求损失。可以报告这一强基线边界，不能主张 MoE 专属性、新颖性、质量保证或方法 GO。

Failure category：当前保护组合相对强简单基线的收益不足，且 residual 对 fit 的服务量增量未重复；首输出义务本身兑现，失败不能写成恢复正确性失败。成本模型第一对方向错误单独记录，不将其拟合结果用于解释全部时间差。暂停问题及其他资源动作不因此判死。

Resurrection condition：在公平强基线下出现可重复、含全部测量窗口成本的完整请求 residual，并能定位到新的可执行资源动作；换阈值、延长保护量子或事后选择有利 repeat 不构成复活证据。

One next smallest experiment：接续已登记、由原执行方统一运行和回读的 [funding-filter 六格](../20260914_funding_filter_comparison_r01/REPORT.md)，检验先排除不能资助目标完整恢复的 victim 再排序，能否把已定位的不可执行选择转化为请求净收益。该组使用既定上下文工作负载，不另造异构方案；以其独立终态和环境记录写结果，不将 U 的旧 GPU 或本补记当作新组结果。

研究问题的本轮答案：在本组预定的吞吐与全局最长暂停两轴，普通 most_output 已覆盖 fit 与当前首输出＋剩余预算保护；平均完成与个体请求损失仍是真实代价。

原件与来源：[execution.json](execution/execution.json)、[group-status.json](execution/readback/group-status.json)、[主分析](analysis/analysis.json)、[成本诊断](analysis/cost_transfer/analysis.json)。U GPU UUID 为 `GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9`，组终态时间 `1789331335.906304`。冻结包 SHA256 `b7557c2e6e65f3d7b2a316f2c16ac04990dc1f347425e26a63b0641fb018516d`；metadata SHA256 `28f7a20c1eaecd30540aafcc9634e377b68b74423c97b2e2ca4b71e10f883940`；回读 SHA256 `f3c002dd6911b0dc88b9f2cb30d37d53b70eb0df6783df2704f6f21f91b19290`。原 raw、冻结合同和既有分析未覆盖。
