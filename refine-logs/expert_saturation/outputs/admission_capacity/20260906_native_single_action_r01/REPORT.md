# 单次32→16：受控重复后仍未超过简单基线

2026-09-06；`agent/publish-current-moe-code@2a37765`及未提交的小范围实现。

**Verdict：`MEASUREMENT_ONLY / CONDITIONAL_NO_GO_CURRENT_TRIGGERED_DOWN_FORMULATION`。**
本轮实际完成24个GPU episode、768次请求执行；只有32条重复使用的真实文本。
四步历史ITL首次越限后只降一次到16，在全部8组配对中goodput均低于同期hold32
和static16。停止当前触发规则与短cohort上的降档实现，不增加第三个selector。
U/C增量、其它触发/动作及动态Oracle仍为UNRUN，不能扩写成MoE并发控制家族NO-GO。

## 问题、设计和执行

上一组[原生反馈实验](../20260906_native_knee_r01/REPORT.md)中，连续降档未超过最好
实测静态点。本轮只问：取消后续12/8降档、保留首次32→16，能否改善完整请求结果？
最弱环节是动作兑现为请求收益，而非再提高慢step预测准确率。

沿用相同32条WikiText输入、128 prompt/128 fixed output、ignore_eos、seed及到达：
steady每50ms一条，bursty每组8条，首尾1.55s，随后排空。固定单RTX5090，BF16
OLMoE pinned revision，vLLM0.26、PyTorch2.11+cu130、Transformers5.15.1、104CPU线程。
共同engine max32、max_model_len256、token预算1024、FA2/Triton、compiled、FCFS、
prefix cache关闭、memory utilization.70；证据是原生同步进程内请求测量，不含HTTP。

- **hold32**：执行同一观测及单次触发代码，记录意图，实际接纳目标始终32。
- **single-down**：最近4个已完成step的请求ITL中位数再取中位数，首次>9ms且满足
  原4步cooldown后，将目标32→16并锁定到episode结束；不再升降。
- **static16**：从空引擎开始固定16，不收集反馈，是独立推进的强简单对照。

forward按hold/down/static16、各steady/bursty；reverse完全反序，共12episode。
四个warmup档8/12/16/32及两种到达保持上一组共同warmup顺序。首轮全部负，但
reverse相对static16仅−0.576%/−0.689%，steady效应随顺序明显变化，因此依
[事前规则](DECISIONS.md)执行唯一一次不变12episode重复。两轮均保留，不再追加重复。

主SLO仍是TTFT≤200ms且request mean TPOT≤9ms；参考5s/200ms全部768次通过。
goodput=联合达标完成请求数/完整episode host时间；含提交、等待、原生执行、输出、
采集、决策和应用成本。初始化、相同warmup及episode外文件落盘在测量外。
mean TPOT=(最后token−首token)/127，不是每个ITL都≤9ms。

## 全部配对结果

每个goodput单元格为达标请求/秒；保留运行次序，八组不是独立文本样本。

| Block / 顺序 / 到达 | hold32 | 单次down16 | static16 | down相对hold | down相对static16 |
|---|---:|---:|---:|---:|---:|
| 首轮 / 正序 / steady | 2.70545 | 0.32484 | 2.07716 | −87.99% | −84.36% |
| 首轮 / 正序 / bursty | 12.52254 | 12.10715 | 12.31935 | −3.32% | −1.72% |
| 首轮 / 反序 / steady | 1.53854 | 1.37145 | 1.37939 | −10.86% | −0.58% |
| 首轮 / 反序 / bursty | 12.43814 | 12.14056 | 12.22476 | −2.39% | −0.69% |
| 重复 / 正序 / steady | 2.29816 | 1.37840 | 1.72160 | −40.02% | −19.94% |
| 重复 / 正序 / bursty | 12.44775 | 12.05220 | 12.18475 | −3.18% | −1.09% |
| 重复 / 反序 / steady | 1.53591 | 1.37562 | 1.72482 | −10.44% | −20.25% |
| 重复 / 反序 / bursty | 12.47193 | 12.21105 | 12.22833 | −2.09% | −0.14% |

达标总数：hold32为149/256，down为141/256，static16为148/256。总计438/768，
未达标请求仍全部正常完成。不同arm的总达标数不直接替代各自的goodput分母。

bursty全部12个episode均32/32通过，down相对hold的goodput下降2.09%–3.32%，
相对static16下降0.14%–1.72%；后者部分接近运行噪声，不宣称每一小差值可统计区分。
steady中，hold始终没有TTFT失败；down每episode有16个请求违反TTFT（含同时违反TPOT），
down总达标1/4/4/4，hold7/4/6/4。完整请求TPOT部分下降没有补偿排队及完成时间代价。
首轮正序down的1/32是保留的低点；重复恢复到4/32，不能以该低点的−87.99%作为稳定效应量。

## 动作和时序揭示的边界

8个down均真实只写入一次目标；写入时active为13/14/16，waiting均0，**均无需自然排空**。
之后active/decode峰值确实受限于16，出现等待；hold原生实际峰值超过16。
首次满足active≤16，与首次出现接纳约束机会，完全不是同一事件。

下表的约束机会要求本臂有等待、有效上限<32且活跃数达到该上限，并仍有至少128个
prefill token预算。时间取scheduler观测区间起点减目标写入时间；不是执行了hold32反事实。

| Block / 顺序 / 到达 | 写入时active | 到首次约束机会ms |
|---|---:|---:|
| 首轮 / 正序 / steady | 13 | 174.26 |
| 首轮 / 正序 / bursty | 16 | 498.96 |
| 首轮 / 反序 / steady | 14 | 115.60 |
| 首轮 / 反序 / bursty | 16 | 473.79 |
| 重复 / 正序 / steady | 13 | 163.49 |
| 重复 / 正序 / bursty | 16 | 481.29 |
| 重复 / 反序 / steady | 14 | 131.63 |
| 重复 / 反序 / bursty | 16 | 1000.80 |

因此不能用“自然排空过慢”解释本轮失败。它证明目标已经满足时，动作仍可能尚未约束
接纳；这只是当前观测边界，不是MoE独有的新规律或已测因果机制。

8组hold/down各自触发前的记录均不完全匹配：step、请求进度或token前缀已有差异。
保留为相同输入/到达、各自历史触发、各自独立未来执行的**端到端策略对照**；不能称为
同一KV状态下单步动作的精确效应。这没有使请求测量无效，也不要求先造snapshot系统。
static16的历史从起点就不同，它是简单基线，不是同前态动作臂。

## 四项必要检查及留存

主/参考指标从全部24份raw重算，与保存值一致；分析无issue。输入身份、动作历史截止、
完整时钟会计及独立未来轨迹得到检查；共65024次反馈臂已有decode推进均保留，8个实际
动作无抢占、KV adjustment或多token输出chunk。static原始decode推进也由复用分析检查。
四个engine的参数和实际执行模块摘要一致；摘要匹配本目录唯一上传包，不把HEAD当完整源码。
48个共同warmup全部完成；四个JIT警告均位于warmup，测量阶段没有记录到该警告。
这不证明所有未埋点成本都已定位，更不支持用已捕获JIT解释首轮低点。

每个反馈episode的决策/应用合计4.82–6.77ms，已在主分母中；不是全部遥测增量成本。
一次针对实际分析脚本变化的核对确认：前缀截断与原始output_events一致，没有未来前缀。
本次没有新增U/C采集，不声称HBM流量、专家拥塞、任务质量或数值一致性。

先前默认python3路径缺失、GPU被另一会话占用时，本轮均为0测量cell；改用原环境的
绝对解释器并等另一整组任务COMPLETE后才启动，未杀进程或调整显存配置。
本轮queue21943、唯一重复queue23277均正常结束，最终GPU空闲。

原始数据：[首轮](gpu_results/)、[受控重复](controlled_repeat/gpu_results/)；
派生结果：[首轮分析](analysis/analysis.json)、[重复分析](controlled_repeat/analysis/analysis.json)；
[运行记录](commands.txt)、[实际上传包](execution.tar.gz)。共享工作区中的既有读取者输出
保留，不覆盖其文件。9项与本次修改相关的CPU检查通过；未跑全仓测试，未push，
未修改`docs/current/README.md`或既有封存原始结果。

复算：从仓库根目录执行`python3 <本目录>/analyze_single.py --results-dir <本目录>/gpu_results
--output-dir <不存在的新目录>`；重复将results-dir改为`<本目录>/controlled_repeat/gpu_results`。

## 研究裁决与唯一下一步

最强本轮同期比较为hold32和static16，两者均8/8优于该触发规则；这是测试集合内的
端到端结果，不是全局最优、动态Oracle或所有合法接纳动作的必要条件证明。
局部负结果否定的是**当前四步9ms触发、一次32→16、128/128有限cohort**的投资价值。
只有重要自然运行域出现可复现动作权衡、且简单策略留有空间，才重开；不换阈值继续救本规则。

本轮停止该降档链后，唯一下一最小实验转入已保留的conformance条件备选：
对两个预先指定自然target，固定单层输入、全局batch宽度及精度，在原生fused-MoE中
比较原companions、相同companions置换、同宽度替换，另做同臂重复。只检验target输出
是否随内部expert形状发生稳定变化；尚不实现verifier或全模型snapshot。
若现象在原生算子消失，停止从custom结果向通用边界外推；若存在，再定位与用户可见后果。
该探针UNRUN。本轮不同时推进offload、Verify、Receiver或QuotaEP，也没有CCF B方法GO。
