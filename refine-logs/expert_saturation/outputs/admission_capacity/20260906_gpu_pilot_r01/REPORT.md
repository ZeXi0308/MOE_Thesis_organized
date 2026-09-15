# 非抢占 MoE 容量与真实动作 pilot：2026-09-06

**Verdict：`MEASUREMENT_ONLY / U_C_ACTION_INCREMENT_UNVERIFIED / NOT_METHOD_GO`。**
本轮确认有限请求 cohort 中存在真实并发—排队—TTFT/TPOT 权衡，补出了强简单静态基线，
并实际测到了非抢占升降档延迟。没有验证专家结构改变并发边际收益的核心假说，尚不能形成完整 CCF B 方法论文。

开始时为 `agent/publish-current-moe-code@2a37765`，三个既存实验目录未跟踪。
已读 current/ideas 权威入口、9 月 5 日准备/CCF B 评估、Route Capacity lightweight status、
N0d 报告与 v3 addendum/verdict；继承其 conformance 测量边界，不升级为容量证据。
执行顺序采用用户本轮授权，封存结果及 `docs/current/README.md` 未改写。
本轮问题是“真实并发响应是否存在，简单档位与一次真实动作对照能解释多少收益”；
最弱环节是实际接纳/排队/SLO 区间与动作执行是否可辨认。

实际连接用户提供的单张 RTX 5090（32 GB），使用 Python 3.12.3、Torch 2.8.0+cu128、
Transformers 4.57.6、固定 OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，
BF16/eager custom runtime。全部 **64 cells 完成，1,024 次请求执行，但只有 16 条唯一输入文本**；
不是 1,024 个独立样本，也不把相邻 step/layer 当独立样本。每次独立推进 KV、route、queue、
输出和 completion。所有 cell 前后 GPU 进程检查通过；不声称连续 GPU/主机隔离。

输入保持 128 prompt tokens、16 输出 tokens、首尾到达 0–1.5 s，TTFT ≤5 s、mean TPOT ≤0.2 s。
未调 SLO，未隐藏任一 repeat。原 24-cell cap=2/4/8 扫描出现 4→8 变号后，补了
24-cell cap=4/6/8 反序重复。加 midpoint 的目的、随后动作时刻均在 [DECISIONS.md](DECISIONS.md)
中于对应 GPU 执行前记录；这是探索性调整，不是未见数据的确认实验。

| 静态 OFF 结果 | 每 cell 联合达标数 | Goodput 请求/秒 | 主要限制 |
|---|---:|---:|---|
| cap=2，原扫描 | 6/16 | 0.379–0.395 | 排队导致 TTFT 超限 |
| cap=4，局部重复 | 12/16 | 1.201–1.243 | 排队导致 TTFT 超限 |
| cap=6，局部重复 | 15–16/16 | 1.790–1.955 | 少量 TTFT 超限；TPOT 全通过 |
| cap=8，局部重复 | 9–11/16 | 1.239–1.562 | TPOT 超限；TTFT 全通过 |

同轮、同 repeat/regime 下，cap=6 相对 cap=4 的 goodput 提高 **46.30%–62.17%**，
相对 cap=8 提高 **14.61%–57.74%**。这些是固定档位收益，不能记到 U/C 策略名下。
cap2/4/6/8 均实际达到对应活跃/decode 宽度且有等待暴露。
原扫描 cap2/4 的平均队列等待约 6.0–6.3 s / 3.17–3.27 s，平均 prefill 约0.15 s；
cap8 的 median TPOT 0.1949–0.2015 s 紧贴门槛。原扫描4→8 goodput 在两个到达模式中均变号，
说明不能只看总吞吐或一次过线计数选 cap。

一个关键边界：**48 个静态 cells 中，同 cap 的请求分组和输出 token 全部一致；
同 cap 的 ON 逐 step/layer U/C 也全部一致。** 两个到达模式没有产生不同的服务分组/专家轨迹。
因此同 cap 时延/达标波动不能归因于已测 U/C 的变化；跨 cap 的 U/C 趋势又混有 batch 改变，
不能识别普通状态之外的增量。mean TPOT 达标也不保证每个 ITL 达标。

统计 helper 占静态 ON episode 的 **0.968%–1.503%**，已计入完整分母。
24 对 OFF/ON 的 episode 墙钟差为 **−1.376%～+3.984%**，属于独立重跑差值，
不是被精确隔离的完整采集税；router-logit 返回等成本不全在 `telemetry_s` 内。

最后以 A/B/B/A 顺序完成 16 个真实动作 cells：A 保持6并在1.5/3 s执行 sham 回调，
B 在1.5 s升8、3 s降6；两种到达模式和 OFF/ON 都保留。相同输入，独立未来状态，
没有用未来信息或 U/C 选动作。8 对中7对观测到的动作前 token/membership 对齐；
1对未对齐。未保存/比较精确 KV，全部只作受控端到端对照。

| Pulse 相对 hold6 的 goodput 变化 | 重复0 | 重复1 |
|---|---:|---:|
| steady OFF | +4.42% | +4.48% |
| bursty OFF | −9.90% | −2.87% |
| steady ON | +2.00% | +5.41%（观测前缀未对齐） |
| bursty ON | −7.84% | +73.15% |

ON bursty 的大正值必须与前一次负值并列：第二次 hold6 降到8/16达标、pulse为13/16，
存在明显重复时序/SLO门槛敏感性，不能用这一个数主张机制或采集可接受。
这个 pulse 没有稳定的跨到达模式、含采集成本优势。它只测试一个固定动作方案，
不否定所有动态并发控制。

升档从应用到实际达到8耗时 **0.425–0.476 s**；降档应用时仍有7或8个活跃请求，
继续全部 decode，自然达到6耗时 **0.129–0.411 s**。加上循环边界等待，从计划时刻到
动作生效分别为 **0.475–0.696 s / 0.168–0.444 s**。16个动作cells从请求时间重建全活跃集合，
未发现正在执行的请求被裁掉或暂停。信号在延迟期间是否保留选择价值仍未验证。

| 固定交付字段 | 本轮结论 |
|---|---|
| Evidence type / Claim ceiling | 单 GPU `CUSTOM_CONTINUOUS_RUNTIME` 的有限请求级测量；不是 native serving 或稳定容量证明 |
| What was measured | 接纳/全 active decode、队列、TTFT/TPOT/ITL/goodput、U/C proxy、重复、一次 pulse 及生效延迟 |
| What was not measured | U/C 超出普通状态的动作增量、在线 selector、独立新 cohort 泛化、native serving、HBM/硬件拥塞、EP |
| Strongest baseline | 同 cohort 探索选出的静态 cap6；动作试验另有带 sham 的 hold6。不是 holdout 调优结果 |
| Oracle/headroom | 无动态 Oracle；静态改善不能代表动态可捕获上界，pulse 不构成 Oracle |
| Failure category | U/C 条件变化不足；固定 pulse 总体收益不稳定。不是 INVALID 数据，也不是系统/family NO-GO |
| Continue / reopen | 在普通状态可比的新文本上产生真实 U/C 差异且动作响应稳定，才允许讨论专家感知 residual；否则不加 predictor |
| One next smallest experiment | 按源顺序取下一组16条新文本，将动作触发固定到完成的 decode step，重复 hold6/pulse 对照；先核对共同前缀的普通状态与 U/C，再比较独立未来请求结果 |

本轮补了运行域汇总与 post-cell 进程检查失败时的分析排除；4个既有 runtime CPU tests、
7个 analyzer fixture tests 通过，均不是性能证据。64 cells 的输入、实际源码和边界检查核对见
[verification.json](verification.json)。GPU 运行期间本地 runner 出现输入 offset/线程记录变更，
未将它冒充已执行版本：本轮上传时的实际 [runner](executed_run_capacity.py) 单独保留并与64份
环境记录匹配，其余5个 helper 与本地一致。运行结束GPU计算进程为空、显存占用0 MiB。

核心数据与重算：静态 [measurements.json](measurements.json)、动作 [action_measurements.json](action_measurements.json)、
[原扫描](scan/)、[局部重复](bracket/)、[动作原始数据](actions/)、[命令](commands.txt)。

**直接回答：并发控制的请求级取舍和可执行动作已被真实 GPU 证据确认；
专家结构是否改变边际动作收益仍没有得到验证。当前保留研究主线，不能把固定cap收益包装为CCF B创新点。**

后续16-cell同配置受控重复已完成：原steady小幅正结果未复现，详见 [REPEAT_ADDENDUM.md](REPEAT_ADDENDUM.md)。原报告及原始数据保留，不用新运行替换。
