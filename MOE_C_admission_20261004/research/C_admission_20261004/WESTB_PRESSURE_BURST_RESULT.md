# westb-pressure-burst-r01：固定突发窗口诊断

**本轮没有建立恢复信号的增量决策价值。** 六臂各384次到达，合计2,304个请求全部完成；失败、拒绝、超时、未完成均为0。两个recovery运行实际改变的gate决定均为0。相对同cap的KV-only，recovery的20点joint goodput在正序块为0胜20负，反序块为16胜4负；mean flow差异为+1.931%/−0.057%。没有执行恢复追加门控，因此不能把这些差异归因于恢复信号。

本轮只改变预先固定的外部到达轨迹：同一有序384请求人口，前256个按0.1*i s到达（末个25.5 s），最后128个在55 s同时到达；到达时刻不依赖运行时状态，各臂实际prompt token IDs与次序相同。这是稳态结果之后设计的posthoc窗口诊断，非独立确认或自然生产负载。fixed cap128从同版本稳态dev-r03的mean-flow选择原样转移，**未证明是burst最优固定上限**，本次没有burst并发调优。KV和recovery cap256；native上限256、GPU/host预算、运行请求调度、victim规则、恢复执行、传输和服务量子固定。不把稳态时延混入本次比较。

证据层级为NATIVE_SERVING。执行顺序为fixed→KV→recovery→recovery→KV→fixed，两顺序块分别分析；仅有两次运行级重复，不把请求、快照或20个SLO阈值点当独立实验，不给统计显著性或置信区间。

来源`67b1211`的v4 runner仅新增burst profile，远端`/root/moe-c-admission-20261007-v4/pro-pressure-burst-test-r01`，GPU `94203fc3-1021-3a9c-a367-cff792479616`。`run.py` SHA256为`f88d1264a5003054d9150a3c22849cf3aaf3da92c317b43253c047a86279da09`，`admission.py`仍为`c9dda825dade091c5673139f472980f927aa2d72c6fb9d4e52f80d2af37d2ae5`。根会话已确认PID25837退出0并释放共同GPU锁；本分析只读CPU数据。81个已提取文件与归档逐一核验字节数和SHA256，没有覆盖原始数据。归档SHA256为`27cfe1fc28081f565882636f8f8b62ca15485c65c56feb04ca3ea4ad582b7466`，见[raw-index](analysis/westb-pressure-burst-r01.raw-index.json)。

| 执行臂 | Admission cap | 完成/到达 | TTFT mean / p95 (s) | 完整flow mean / p95 (s) | 每请求最大生成间隔 p95 / max (s) | Token/s | 排空 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 fixed | 128 | 384/384 | 22.290 / 47.667 | 64.437 / 84.831 | 0.170 / 0.170 | 2734.81 | 85.403 |
| 01 kv | 256 | 384/384 | 9.014 / 29.096 | 64.456 / 87.953 | 0.186 / 16.476 | 2939.32 | 74.023 |
| 02 recovery | 256 | 384/384 | 9.374 / 30.156 | 65.700 / 90.097 | 0.201 / 16.947 | 2935.45 | 74.990 |
| 03 recovery | 256 | 384/384 | 9.491 / 30.648 | 66.122 / 90.546 | 0.197 / 17.536 | 2932.67 | 75.257 |
| 04 kv | 256 | 384/384 | 9.498 / 31.293 | 66.160 / 91.103 | 0.210 / 18.082 | 2937.62 | 75.265 |
| 05 fixed | 128 | 384/384 | 22.459 / 48.740 | 64.692 / 85.831 | 0.193 / 0.193 | 2695.03 | 86.435 |

所有TTFT和完整flow从外部到达起算，未隐藏新请求排队。服务分母依执行序为140.403295、129.023557、129.989807、130.257310、130.265165、141.434700 s，包含55 s到达窗口和其后的完整排空；表中排空为最后完成减55 s。请求结果没有删失，CDF分母均为384。KV256相对转移fixed128的TTFT p95下降38.96%/35.80%，但flow p95上升3.68%/6.14%；最大生成间隔从固定臂的0.170/0.193 s升到KV的16.476/18.082 s。平均flow同样没有改善，不能只展示TTFT和吞吐。

完整原始分布、逐请求CSV/JSON见[主分析](analysis/westb-pressure-burst-r01.json)。两顺序块分别绘制TTFT/flow全人口CDF：[SVG](analysis/westb-pressure-burst-r01-cdf.svg)、[PNG](analysis/westb-pressure-burst-r01-cdf.png)，图中明确标注cap128/256/256，曲线为单次运行分布。

全部20点沿用预声明探索性联合SLO：TTFT≤2/5/10/20/40 s × maxgap≤0.25/0.5/1/2 s，完整flow≤120 s。每格为**fixed128 / KV256 / recovery256**联合达标请求数，分母各384；goodput为达标数除各臂完整服务时间。不事后挑一个阈值称应用SLO。

| Block1 TTFT | gap≤0.25 s | gap≤0.5 s | gap≤1 s | gap≤2 s |
|---|---:|---:|---:|---:|
| ≤2 s | 130 / 215 / 213 | 130 / 215 / 213 | 130 / 215 / 213 | 130 / 215 / 213 |
| ≤5 s | 137 / 220 / 219 | 137 / 220 / 219 | 137 / 220 / 219 | 137 / 220 / 219 |
| ≤10 s | 137 / 220 / 219 | 137 / 220 / 219 | 137 / 220 / 219 | 137 / 220 / 219 |
| ≤20 s | 142 / 291 / 286 | 142 / 292 / 286 | 142 / 293 / 286 | 142 / 293 / 289 |
| ≤40 s | 317 / 371 / 371 | 317 / 375 / 374 | 317 / 377 / 374 | 317 / 377 / 377 |

| Block2 TTFT | gap≤0.25 s | gap≤0.5 s | gap≤1 s | gap≤2 s |
|---|---:|---:|---:|---:|
| ≤2 s | 131 / 212 / 212 | 131 / 212 / 212 | 131 / 212 / 212 | 131 / 212 / 212 |
| ≤5 s | 137 / 218 / 218 | 137 / 218 / 218 | 137 / 219 / 218 | 137 / 219 / 218 |
| ≤10 s | 141 / 218 / 218 | 141 / 218 / 218 | 141 / 219 / 218 | 141 / 219 / 218 |
| ≤20 s | 145 / 284 / 286 | 145 / 284 / 286 | 145 / 285 / 287 | 145 / 286 / 287 |
| ≤40 s | 314 / 369 / 373 | 314 / 372 / 375 | 314 / 375 / 376 | 314 / 376 / 376 |

| 比较 | Block1 goodput胜/负/平 | Block2 goodput胜/负/平 | Block1达标率胜/负/平 | Block2达标率胜/负/平 |
|---|---:|---:|---:|---:|
| KV 对 fixed | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| recovery 对 fixed | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| recovery 对 KV | 0 / 20 / 0 | 16 / 4 / 0 | 0 / 18 / 2 | 7 / 4 / 9 |

KV/recovery相对转移fixed128的20点优势包含不同并发上限和KV门控，不能算恢复状态贡献，也不能代替burst自身最强固定基线。关键同cap的rec/KV对照：TTFT p95为+3.644%/−2.063%，flow p95为+2.438%/−0.611%，吞吐为−0.132%/−0.169%，排空为+1.305%/−0.010%。完整逐点goodput和达标率同时保留，反序多数goodput点胜出不代表达标率全面胜出。

| 执行臂 | 总输出token | 自然stop / length | Preempt | 步首最大backlog | cap / KV重复defer | controller wall / 服务占比 | controller CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 fixed | 383,977 | 10 / 374 | 0 | 0 | 153,409 / 0 | 1.328 s / 0.946% | 1.363 s |
| 01 kv | 379,242 | 15 / 369 | 39 | 9 | 0 / 14,125 | 0.297 s / 0.230% | 0.297 s |
| 02 recovery | 381,578 | 13 / 371 | 36 | 9 | 0 / 12,020 | 0.281 s / 0.216% | 0.280 s |
| 03 recovery | 382,002 | 12 / 372 | 38 | 9 | 0 / 11,872 | 0.276 s / 0.212% | 0.276 s |
| 04 kv | 382,670 | 11 / 373 | 32 | 10 | 0 / 10,464 | 0.268 s / 0.206% | 0.267 s |
| 05 fixed | 381,171 | 13 / 371 | 0 | 0 | 151,472 / 0 | 1.269 s / 0.897% | 1.303 s |

natural EOS开启，min_tokens=0，max output=1024；所有输出token均计入吞吐。rec对KV的输出量差异为+2,336（+0.616%）/−668（−0.175%），序列不同148/384、131/384，长度不同均为6/384。KV对fixed的输出量差异为−4,735（−1.233%）/+1,499（+0.393%），序列不同222/384、217/384。同策略跨块fixed/KV/recovery输出序列分别有202/147/133个请求不同。未验证任务质量，不将生成更少当等工作提速。逐请求长度、序列、stop差异均在[comparison](analysis/westb-pressure-burst-r01-comparison.json)中。控制开销已包含在服务计时；重复defer不是拒绝，外部到达源未暂停。

快照中fixed的最小free pages为11,337/11,399，最大active均128；KV/rec的最小free均0、最大active依序为221/220/221/221，均未达到cap256。可用KV总页数32,768，KV floor为3,277。这里是快照范围，不声称连续峰值全部被采到。

**恢复口径限制：** `Gate.begin`在调度步首刷新`recovery_since`，gate日志的`recovery_count`为该步首集合大小；gate动作瞬间没有重新扫描。latch约每100 ms更新，free/active在gate评估时读取，因此不能把字段当完全同时采样。只计原生等待循环实际评估的未首次启动请求，native queue-head break后没有被遍历的请求不在日志中。

| 臂 | 首次非零步首backlog snapshot (s) | 首次 / 最后latch snapshot (s) | latch gate评估 / 唯一请求 | latch且age<10、KV/cap允许评估 | 恢复增量改变 |
|---|---:|---:|---:|---:|---:|
| 00 fixed | 无 | 无 / 无 | 0 / 0 | 0 | 0 |
| 01 kv | 34.231 | 34.873 / 54.230 | 2 / 2 | 0 | 0 |
| 02 recovery | 34.157 | 35.065 / 52.571 | 3 / 2 | 0 | 0 |
| 03 recovery | 33.826 | 34.478 / 51.090 | 0 / 0 | 0 | 0 |
| 04 kv | 33.808 | 34.712 / 51.621 | 0 / 0 | 0 | 0 |
| 05 fixed | 无 | 无 / 无 | 0 / 0 | 0 | 0 |

时间相对外部到达原点，仅是观测边界，不能推断连续状态持续时间。rec02仅3条latch gate、2个唯一请求，均来自burst之前的旧等待请求，外部时刻52.502557/52.572379/52.638481 s；年龄29.303/29.272/29.338 s，free为150/73/56，active为199/200/200，步首recovery_count为1/1/0。全部`signal_wait_limit_bypass=true`、`kv_only_denied=false`、`denied=false`、`changed_by_recovery=false`，KV与recovery gate都因age≥10 s豁免，cap仍适用；低free并不意味着这次KV gate实际拒绝。最后一条count=0而latch=true表明采样清除滞后。rec03没有latch gate。

即使不要求latch、只查实际gate日志中的`age_s<10 and active<cap and free_blocks>=3277 and recovery_count>0`，各臂评估数仍均为0。55 s突发到达的请求在55–65 s期间年龄小于10 s，其中尚未启动者可参与新请求gate评估；两个recovery运行最后的latch=true快照均在55 s之前。其非零步首backlog快照可延续至61.286/60.090 s，仍没有观察到满足上述交集的新请求gate动作。这描述本次轨迹与当前阈值/迟滞的动作窗口，不能把零机会外推为所有恢复准入机制无效，也不能证明未运行的native full-ISL allocation会成功。

全请求首次prefill许可顺序均保持同一FIFO次序，但批次和许可时间不同；rec/KV同块首次许可时间的最大绝对差为18.597/7.773 s。顺序相同不等于实际准入时间相同。外部到达至首次prefill许可的mean/p95依执行序为22.178/47.532、8.852/28.727、9.212/29.998、9.322/30.313、9.337/31.162、22.345/48.604 s。许可取schedule返回边界，非prefill计算完成；未隐藏队列等待。完整许可序列、逐请求等待及豁免见[decision summary](analysis/westb-pressure-burst-r01-decisions.json)。

原始launch log第92行在首测量臂`test-00-fixed`记录18:29:16的`fused_moe_kernel`推理期JIT警告。没有可隔离时长，故保留警告和首臂原始结果，不猜测贡献、不减去成本，不用末臂覆盖首臂。

本次完成了指定的最后一次burst窗口诊断：在固定的单卡OLMoE原生运行域和当前规则下，没有恢复增量动作或相对KV-only的可归因收益，当前候选不支持论文机制GO。边界仅覆盖此实现、稳态选择转移和本次固定突发轨迹；burst自身并发最优、独立任务质量、第二模型及其他恢复信号形式均未验证。下一步是把这项负结果合入主报告、保留原始数据；不据此启动新GPU搜索。重新打开本机制的必要证据是已授权的新运行域中出现有真实积压、未超龄、KV/并发允许且native可执行的新请求动作窗口。

复算命令（本目录执行；输出独占创建，已有输出需换新文件名）：

```sh
python3 -B analyze.py runs/westb-20261007/pro-pressure-burst-test-r01/test-{00-fixed,01-kv,02-recovery,03-recovery,04-kv,05-fixed} --output analysis/westb-pressure-burst-r01.json
python3 -B decision_summary.py runs/westb-20261007/pro-pressure-burst-test-r01/test-{00-fixed,01-kv,02-recovery,03-recovery,04-kv,05-fixed} --output analysis/westb-pressure-burst-r01-decisions.json
python3 -B compare.py analysis/westb-pressure-burst-r01.json --output analysis/westb-pressure-burst-r01-comparison.json
```

绘图复现沿用[优化稳态报告](WESTB_PRESSURE_OPTIMIZED_TEST_RESULT.md)末尾的完整importlib/POLICIES命令，仅将analysis路径改为`analysis/westb-pressure-burst-r01.json`、output改为`analysis/westb-pressure-burst-r01-cdf.svg`，保留`--png`；不修改绘图脚本。
