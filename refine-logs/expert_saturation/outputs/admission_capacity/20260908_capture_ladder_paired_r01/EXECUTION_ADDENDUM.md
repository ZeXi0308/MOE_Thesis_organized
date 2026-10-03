# 配对实验已执行：steady 变号，进入一次固定重复

2026-09-08。原 `REPORT.md` 是执行前准备与当时审批阻塞的历史记录。本补充记录后续真实执行；不改原报告、冻结规则或原始数据。

**32个测量episode、1024次请求执行全部完成，两个引擎各15个warmup也保留完整raw。** 仍是同一32条WikiText文本的重复执行。正序启动由并行任务完成，本任务确认该PID/命令后接管回传；正序在本地完整核验后，本任务启动反序（supervisor PID5193），两块均以0退出。

两条ladder在每个引擎中共享预热集合、输入、到达、SLO、资源和代码，测量计划完整反序。实际源码匹配execution.tar.gz，框架版本、vLLM后端代码摘要和引擎参数一致。32/32个raw指标重算与身份检查通过，130048次既有decode推进、history cutoff和全部step/receipt对齐通过；每个cell边界GPU检查通过，不声称连续进程隔离。

| 域 | 顺序 | aligned vs legacy goodput | aligned vs 当次最好static |
|---|---|---:|---:|
| steady | forward | −44.43% | −50.81%（cap16） |
| steady | reverse | +118.74% | +55.21%（cap24） |
| bursty | forward | +54.62% | −4.63%（cap32） |
| bursty | reverse | +61.38% | −0.11%（cap24） |

完整数字与所有静态点见 [analysis.json](analysis/analysis.json)，动作暴露及逐请求时序见 [diagnostics.json](analysis/diagnostics.json)。最强static指同引擎五个已测静态点的探索性上包络，不是可在线选择的Oracle。平均请求TPOT、逐token ITL与TTFT分开保留。

Steady的aligned总时长基本相同（2.927/2.932秒），联合达标却从4/32变成14/32；TPOT通过数18→30，TTFT通过数18→16，首次动作约0.827→0.676秒。这是策略轨迹及SLO权衡发生变化，不能只看goodput百分比归因GPU漂移或padding。Bursty旧规则两次均有8条TTFT失败，aligned消除了这些失败，但高cap静态策略同样全部达标。

结论为 **MEASUREMENT_ONLY / 稳定方法收益未成立**。steady明确变号，bursty相对静态近零，触发原冻结规则中的一次相同配置受控重复，见 [repeat决定](../20260908_capture_ladder_paired_repeat_r01/REPEAT_DECISIONS.md)。重复不能替换当前任何一块，不改变阈值、输入、源码、预热或canonical。重复完成后的合并结论见该目录报告；本文件本身不将未运行结果计入。

[TRANSFER.json](TRANSFER.json)记录两块归档SHA256和本地可读性：32个测量raw、30个完整warmup raw、配置、环境、命令、stdout/stderr和退出记录均已回传，远端原件未删。旧凭据阻塞已由本会话成功登录解除；上传曾被自动审批以数据敏感性阻止，核对包内输入与同主机既有WikiText数据完全相同、无凭据后，同一路径重试获准。未绕过拒绝、未隐藏失败尝试。

不包含专家信号增量、HBM流量、质量、第二模型、EP或生产capacity结论；没有动态Oracle。下一步只做已冻结的一次受控重复，不能用更复杂predictor解释或抢救当前规则。
