# 保留 KV 的完成余量保护：实现与动作窗口

**结果：保 KV 策略消除了抢占和重算，把最长停顿降至约 1.38 秒，但吞吐下降约 3.9%，未达到“减少暂停并保住服务量”的目标。** 四项在 [execution02](execution02/execution.json) 全部完成、回传并校验，128/128 请求；状态为 `MEASUREMENT_ONLY`，研究问题仍 `OPEN`。首次 execution 发现其他 GPU 进程后在加载前退出；更早的自动审批拒绝记录均保留。

## 实际同资源结果

固定 OLMoE BF16/vLLM0.26.0、RTX5090、16,089,350,144-byte KV / 7,671 可用块、cap32、3072/1024、50ms、budget1024。两臂都启用完整历史检查；每项新进程、新引擎、相同三项预热，各策略独立演进。

| 执行顺序 | 完成 | 墙时 s | 请求/s | 平均完成延迟 s | 最大 ITL s | 抢占/重算位置 |
|---|---:|---:|---:|---:|---:|---:|
| repeat0-native | 32/32 | 22.913784 | 1.396539 | 20.556992 | 4.468961 | 2 / 7,685 |
| repeat0-headroom | 32/32 | 23.828437 | 1.342933 | 22.259208 | 1.381322 | 0 / 0 |
| repeat1-headroom | 32/32 | 23.803814 | 1.344322 | 22.240772 | 1.381332 | 0 / 0 |
| repeat1-native | 32/32 | 22.875082 | 1.398902 | 20.506041 | 4.448064 | 2 / 7,685 |

相对同轮 native，headroom 吞吐 **−3.84% / −3.90%**，墙时 **+3.99% / +4.06%**，平均完成 **+8.28% / +8.46%**。最长 ITL 减少 3.088 / 3.067 秒，但 **29/32 请求的各自最长 ITL 变大，31/32 请求完成更晚**。每请求最长 ITL 的中位数从 79.5/79.1ms 升至 1.231/1.230s。因此它显著改变等待分布，不能只依据最坏请求改善宣布成功。

两轮均从第 799 步实际开始 held，暂缓过 31 个请求，最长连续 227 步；held 请求的状态/块表保持，零抢占、零重算。工作守恒为 `131,040 = 98,304 新 prefill + 32,736 新 decode`；native 为 `138,725 = 131,040 + 7,685 重算`。全部请求指标、逐请求变化、实际 held 区间与输出比较见 [analysis.json](analysis/analysis.json)。

## 成本更新

互斥会计为 `wall = scheduler inclusive + engine non-schedule + outside engine calls`；决策计时包含在 scheduler 中，不能再次相加。逐调用嵌套关系和求和已校验，见 [cost_breakdown.json](analysis/cost_breakdown.json) 和 [60 行复算脚本](../../../experiments/admission_capacity/analyze_headroom_cost.py)。

| headroom − native | repeat0 | repeat1 |
|---|---:|---:|
| 总墙时 | +0.915s | +0.929s |
| 调度区间（含决策/检查） | +0.467s | +0.577s |
| 引擎除调度外区间 | +0.449s | +0.343s |
| 引擎调用外区间 | −0.001s | +0.009s |
| 引擎调用数 | +92（1348→1440） | +92（1348→1440） |

决策子区间本身为 headroom 0.456/0.487s、native 0.0136/0.0127s；前 799 步尚未 held 时，headroom 的调度区间已多出 0.222/0.232s。引擎其余区间包括模型执行、采样、同步和主机开销，不能称为纯 GPU 时间。此会计没有证明任一部分可完全消除，也没有把测得区间从墙时扣除来伪造反事实收益。

## 已关闭的实现缺口

两轮原生 full 账本均在第 **799** 步首次触及保护条件：空闲 16 块，候选请求还有 227 个声明输出上限，需要最多 14 个新增 KV 块；全部请求下一步会申请 3 块。原生首次抢占发生在第 806 步，这个未来事件仅作事后比较，不进入策略输入。逐项状态、源码位置与哈希见 [source_probe.json](source_probe.json)。

安装源码支持 RUNNING 请求本步跳过调度并保有 KV。Worker 只将其移出输入 batch，保留 cached state；再次调度时沿普通 running 路径接续。实现只在原生 running 分配前插入 held skip，并在已关闭到达的 cohort 中阻止 WAITING 分配；不修改请求状态、进度、输出或既有块表。

## 最小模型与动作

令 `F` 为当前空闲块，`a_i` 为已分配块，`B` 为初始化确认的块大小。已知 prompt `p_i`、声明输出上限 `m_i`、当前 computed `c_i`：

`H_i = max(0, ceil((p_i + m_i - 1)/B) - a_i)`；
`delta_i = max(0, ceil((c_i + 1)/B) - a_i)`。

最终采样 token 不再输入模型，因此终态为 `p_i + m_i - 1`。本轮要求声明总长不超过模型上限，单 full-attention group、无共享/前缀/推测/KV connector、同步纯 decode。

在全部 32 请求完成 prefill 后，默认继续原生全部 decode。当 `F - sum(delta_i) < H_leader - delta_leader`，锁定声明剩余输出最少的请求，FCFS 破同分。激活必须 `F >= H_leader`；否则记录不可用，不声称保证。Leader 每步继续，其余请求只有在分配后不侵占 leader 余量时才能继续，零新增块可继续。Leader 完成并真实释放后重新选择。

**守恒：** leader 分配时 `F` 与 `H_leader` 同减，其他请求不得侵占其余量。暂缓不释放 KV。当前同长度上限域的 CPU 转移最终完成并归还全部块；这不是任意异构 cohort 的可行性定理。

## 验证与边界

- [实现](../../../experiments/admission_capacity/completion_headroom.py)，198 行；[5 项 CPU 检查](cpu-checks.log)通过，覆盖 150 组同长度上限的独立状态推进、块守恒、最后 token 边界、晚触发拒绝和原生 AST 仅两处插入。
- 保留了初始异构随机生成器暴露的反例：9 块池不能支撑某请求的 10 块终态，完成一个短请求不能证明其余请求也可完成。对应回归明确拒绝不可用余量，未把生成器中的无效资源范围写成方法失败。
- 新结果证据层级为 **NATIVE_SERVING / native in-process**，单模型、单 GPU、同一 32 请求 cohort 的两次描述性配对。第 799 步的旧 baseline 分析仍仅作动作窗口定位，性能来自各策略自身实际运行。
- 最强已测基线仍为原生 full。跨策略 Oracle、任务质量、生产尾延迟、独立文档泛化未测；各配对 31/32 条完整输出序列一致，同策略重复 32/32 一致，不能声称质量等价。
- 主要风险：把少数请求的长暂停分散给更多请求，或因 batch 缩小降低吞吐。所有请求都计入 max-ITL、TTFT、平均完成、墙时、失败和重算，不能通过换指标宣布成功。

## 执行边界与唯一下一步

固定原模型、精度、7,671 可用块、cap32、3072/1024、50ms、budget1024、三项预热，两臂都保留完整历史检查；已按 **native → headroom → headroom → native** 新引擎独立完成。原始归档、12 份预热、配置与状态均在 execution02，不完整与加载前退出记录保留。

当前包 [preparation/execution.tar.gz](preparation/execution.tar.gz)，SHA256 `35c41e5461a8b2c7f28ad7e58c873c38d79471c3df0f1c3abe0de9b92eb3903e`；[协议](preparation/source/DECISIONS.md)、[上传阻塞记录](execution-attempt.json)、[精确运行命令](../../../experiments/admission_capacity/RESEARCH_EXPERIMENTS.md)。审批拒绝前的包也保留，差异仅为替换继承的旧 full/chunk 协议文字。

[独立完整性复核](EXPERIMENT_AUDIT.md)为 PASS，P0/P1 均为 0；fresh GPT-5.6-Sol ultra，same-family / provisional，单轮只读复算。[机器可读记录](EXPERIMENT_AUDIT.json)。held 的精确 block-ID 由在线断言检查，raw 独立保留块数/进度连续性，不包含全部 block-ID 序列。

失败分类为 **请求等待分布的权衡 + 调度/执行成本尚未完全分离**，不是动作无效或缺少提前信息。下一最小工作是在保持本次逐步 held 选择一致的前提下，减少尚未改变调度时的监测/检查开销，再用同资源原生对照检验完整成本；该优化未实现/实测，不改阈值抢救本次结果。只有实际成本变化或适用负载变化才重新评估该方案，不重复同配置寻找有利运行。

本轮回答：**保 KV 暂缓能消除本次重算并削弱最严重停顿，但会让多数请求等待更多，且完整吞吐变差；当前机制尚未满足主目标，整个固定预算请求进度问题仍未被否定。**
