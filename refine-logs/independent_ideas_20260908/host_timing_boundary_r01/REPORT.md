# 独立分解：步间host时间很小，prefill暴露需要下一次真实干预

2026-09-08；HEAD `2a37765`，指定分支及已有未跟踪研究材料。
**本线程新增GPU执行为0。** 以下是对其他进程已完成paired-r01的32个测量episode、
1024次请求执行、11711个调度step、131072个输出事件的独立原始数据分析。
输入只有32篇重复文本，不把1024次执行或相邻step当作独立样本。

本轮回答：请求指标是否主要被步间host处理或交付时间戳拖慢？
**在这批数据中，不支持“大量步间host时间造成主要decode延迟”的解释；host与native
首末时间戳差也没有改变任何请求的平均TPOT达标判定。** 这关闭了一个具体优化假设，
不是全部CPU开销、异步服务或真实外部网络交付问题的NO-GO。

## 与其他Codex任务的分工

共享目录显示其他进程已完成aligned ladder与paired ladder，并准备同配置重复；
原报告还认领单步宽度可行性。自然KV容量、Verify Precision等已有冻结准备。
本线程只读这些输入并确认paired数据回传完整，没有重复启动32个episode，
也没有修改其他进程的报告、代码或原始数据。本目录独立拥有host计时边界问题，
下一步固定接纳上限，只测试prefill token预算。

用户授权的原32个测量已由其他进程完成：两个回传归档与TRANSFER/执行状态的SHA256
一致，175个归档文件逐字节匹配本地解包、168个JSON可读；32个测量raw和30个预热raw
全部COMPLETE，分别包含1024和960次完整请求执行，日志/环境/配置/退出状态齐全。
该目录REPORT中哈希前缀有旧值，实际归档与TRANSFER一致；本线程采用后两者。
其steady变号尚不能判定动作被充分否证；受控重复由原线程继续，非本线程的新任务。

## 新测到了什么

按真实host时间区间互斥相交，所有请求TTFT/首末token间隔及episode分母均闭合；
本次浮点重算最大守恒残差为0。所有32个cell的达标数与原metrics重算一致。

| 观测 | 结果 | 可支持的解释 |
|---|---|---|
| 步间host占各episode请求decode时间 | 0.7028%–1.2176% | 没有大份额步间空隙 |
| host平均TPOT与native首末差值的最大绝对偏差 | 10.034微秒/token | 端点时间定义差相对毫秒级TPOT很小 |
| 用host/native平均TPOT判定9ms SLO | 1024/1024一致 | 这批达标计数差异不是首末交付时间戳选择造成 |
| 步间host占各episode请求TTFT | 0.73%–6.77% | TTFT上不能简单套用decode的1%结论 |
| steady实际提交滞后均值 | 3.949ms | 请求可能在上一步未返回时已到达 |
| steady提交滞后的区间来源 | 91.08%在decode步骤poststamp桶，7.07%在prefill桶 | 多数滞后发生于同步step尚未返回期间 |
| steady距9ms TPOT门槛不足100微秒的请求 | 115/512 | 小时间分量仍可能影响门槛附近的计数，不等于可忽略 |

代码中`frontend_gap`准确含义是**步间host区间**：输出遍历/记账、反馈观测、提交、
空闲sleep，以及下一engine.step进入scheduler前的路径。初始段还包括capture初始化。
它并非可全部消除的frontend税。`scheduler_prefix`包括采集、反馈和原生schedule；
`poststamp`还含wrapper尾部、引擎执行、采样与交付，绝不称GPU kernel时间。
TTFT按全局阶段归属的是请求等待期间的墙钟，不是该请求独占的CPU/GPU消耗。

## 一个改变下一步选择的定位

steady aligned两次执行的平均TPOT差约0.283ms。按同一会计边界分解如下：

| 平均每请求TPOT的组成，ms/token | forward | reverse |
|---|---:|---:|
| 步间host | 0.091121 | 0.094996 |
| scheduler时间戳之前 | 0.161310 | 0.164197 |
| 含prefill步骤的poststamp区间 | 0.865126 | 0.569187 |
| pure decode步骤的poststamp区间 | 7.755816 | 7.762103 |
| 总平均TPOT | 8.873372 | 8.590482 |

reverse更快时，步间host和pure-decode分量反而略高；变化主要在请求经历的prefill步骤
区间。这里使用**均值**保证互斥组成能相加，不能把各分量中位数相加。
这个结果只定位观察差异，不说明prefill计算本身变快，也不将其扣减生成反事实收益。
请求组成、prefill相对decode的发生时刻、执行形状均可能改变这个分量。

因此停止“先优化frontend循环”的候选。唯一下一实验是在共同原生引擎、固定接纳上限下，
比较两档prefill/token预算，记录长短混合请求的ITL与TTFT代价。它是不同于其他线程
接纳档位和单步宽度控制的动作；输入与具体准备见 `next_prefill_budget/`。
分块prefill及其ITL/TTFT权衡已有明确先例，本实验只检查这一MoE运行域的响应和残差，
不主张新颖机制。[vLLM 0.26官方说明](https://docs.vllm.ai/en/v0.26.0/configuration/optimization/#chunked-prefill)

## 数据与边界

完整派生数据在 `analysis/analysis.json`，逐cell含源raw SHA256、全部1024请求的
互斥分桶、提交区间及native/host对照；脚本为 `analyze_host_timing.py`。
原始GPU数据仍在 `../../expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_r01/`，
未重复复制或改写。复跑需一个全新的输出目录：

```bash
python3 refine-logs/independent_ideas_20260908/host_timing_boundary_r01/analyze_host_timing.py \
  --source-dir refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_r01 \
  --output-dir /tmp/moe-host-timing-new-run
```

| 结束字段 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；关闭本域“大量步间host空隙主导decode”的解释 |
| Evidence type | REQUEST_LEVEL / 既有原生GPU trace的观察性互斥会计 |
| Measured | 完整请求TTFT/平均TPOT的时间区间归属、native/host首末差值、提交滞后 |
| Not measured | 新frontend/prefill干预、GPU kernel成本、专家信号、真实HTTP服务 |
| Strongest baseline | 原32个episode全部static、shadow与feedback均纳入；没有新策略收益 |
| Oracle/headroom | 未执行；host分量不能直接当可回收上界或反事实goodput |
| Claim ceiling | 当前同步driver中可排除的一个大开销解释及后续定位 |
| Failure category | host优化候选缺大份额暴露；科学执行链仍开放 |
| Reopen | 新runtime、输出处理或自然到达域出现material的步间host成本 |
| One next experiment | 固定cap下prefill token预算的小型真实对照 |

SSH授权已生效，但本线程实际密码登录被服务器拒绝，未启动新的远端任务；
需要有效连接信息才能执行下一GPU实验。没有审批拒绝，也未另寻连接路径。

下一实验已封装为 `next_prefill_budget/execution.tar.gz`，固定cap8，比较256/1024调度预算，
包含8个测量episode及8个warmup。归档逐文件核对、解包后正反序prepare-only及3项针对性检查
均通过，详见 `next_prefill_budget/PACKAGE_CHECK.json`。这些仅证明执行准备，
实际GPU执行数为0，尚未上传；预算效果及ITL/TTFT权衡仍为UNRUN。

## 后续执行更正（2026-09-08）

上文SSH失败的原因已定位为本线程临时传输脚本遗漏`subprocess.pass_fds`，
承载密码的匿名管道未传入sshpass。补齐描述符继承后，同一凭据成功登录；
因此撤回“需要更换有效凭据”的判断。此故障未启动任何GPU执行，也未影响已有raw。
独立prefill实验包已上传并核对SHA256，当前等待其他进程的自然容量实验最后一格排空。
其他进程的paired受控重复已完成，仍不属于本线程执行。

## 独立GPU对照已完成（2026-09-08）

SSH传输修复后，本线程已独立执行并全量回传8个prefill预算测量episode及8个warmup，
两引擎退出码均0。mixed中256相对1024的TTFT均值增加32.06%/34.56%，整批完成
时间增加9.05%/9.14%，部分ITL尾部缩短；这是权衡，无整体方法GO。
详见 [独立实验报告](next_prefill_budget/REPORT.md)、`next_prefill_budget/EXECUTION.json`
及该目录readback全部raw。上文“尚未上传/GPU为0/等待凭据”均仅描述当时准备状态，
当前由本节及STATE.json更新；源paired32的历史会计结论不变。

## 预选中间预算已完成（2026-09-08）

本线程又完成512/1024的4测量＋4预热，全部数据回传。512相对同引擎1024的ITL p99
下降约29%–30%，TTFT均值增加6%–8%，完成墙钟增加约2%，仍是权衡，停止继续扫静态
预算。详见[中间预算报告](prefill_budget_midpoint_r01/REPORT.md)。
下一步改为固定1024预算下的等待队列顺序对照；当前尚未运行新队列策略。
上一轮独立审查已返回WARN/provisional、P0=0/P1=0，时间桶疑点经定向数值复算未复现。

## 等待顺序对照与一次原样重复（2026-09-08）

本线程已独立完成FCFS/短prompt优先的16正式+16预热GPU episode，固定cap8/budget1024，
与其他进程的KV安全cap/原生抢占动作不同。原始对照全部回传；原样重复的forward已回传，
reverse已在远端确认4正式+4预热COMPLETE、退出0，但SSH在归档前开始于握手阶段关闭，
这最后8个episode的raw仍待补传。累计本线程28正式+28预热已执行，其中24正式+24预热
完整在本地；其他进程完成的原始paired32仍单独计数，不冒充本线程执行。

已回传的三个mixed block中，短请求TTFT下降14.06%/10.38%/9.45%，长请求TTFT增加
3.09%/6.44%/5.19%；全体request latency均值只下降2.52%/0.22%/0.03%，长请求尾部
代价没有消失。当前仅支持请求代价转移的测量结论，无稳定净收益或方法GO。
完整重复裁决等最后raw回传；不追加第三轮或调整策略。最新状态见
[原样重复报告](waiting_order_repeat_r01/REPORT.md)与STATE.json。
