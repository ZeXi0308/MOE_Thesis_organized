# d6 四臂结果：整批加速与平均完成继续分离

状态：MEASUREMENT_ONLY；NATIVE_SERVING 限定为原生 vLLM 进程内，非外部服务部署证据。8/8 COMPLETE、256/256请求、262144输出token，全量归档SHA256校验通过。两block同一旧32文档，固定3072/1024 token、50ms到达、6656可用KV块、OLMoE BF16、vLLM0.26、单RTX5090、APC关闭。没有新holdout、质量或自然EOS检验。

问题：headroom或most_output是否降低least_progress在高压下的平均完成代价，同时保住长停顿改善？本次答案：两种既有简单策略都没有在这两组中实现这一点。most_output更快完成整批，但平均完成仍慢于least_progress；headroom的平均完成和最长停顿均更差。不能据此否定恢复调度问题或其它机制。

## 主结果（均为实测原值）

| block | 策略 | 完整wall(s) | 平均完成(s) | 最大ITL(s) | 全部调用 | 抢占数 |
|---|---|---:|---:|---:|---:|---:|
| 0 | native | 26.502 | 20.434 | 13.996 | 1869 | 6 |
| 0 | headroom | 37.057 | 33.012 | 4.564 | 3772 | 0 |
| 0 | least_progress | 26.476 | 21.960 | 2.802 | 1581 | 39 |
| 0 | most_output | 25.767 | 23.723 | 2.818 | 1315 | 44 |
| 1 | most_output | 25.157 | 23.118 | 2.814 | 1315 | 44 |
| 1 | least_progress | 27.567 | 22.980 | 2.938 | 1581 | 39 |
| 1 | headroom | 36.452 | 32.381 | 4.351 | 3772 | 0 |
| 1 | native | 27.115 | 21.020 | 14.367 | 1869 | 6 |

| block | target / baseline | 吞吐变化 | 平均完成变化 | 最大ITL变化(s) | 完成更慢请求 | 输出序列相同 |
|---|---|---:|---:|---:|---:|---:|
| 0 | least_progress / native | +0.10% | +7.47% | -11.193 | 31/32 | 24/32 |
| 0 | headroom / least_progress | -28.55% | +50.33% | +1.762 | 30/32 | 22/32 |
| 0 | most_output / least_progress | +2.75% | +8.03% | +0.016 | 25/32 | 27/32 |
| 1 | least_progress / native | -1.64% | +9.32% | -11.429 | 32/32 | 24/32 |
| 1 | headroom / least_progress | -24.37% | +40.91% | +1.413 | 29/32 | 22/32 |
| 1 | most_output / least_progress | +9.58% | +0.60% | -0.124 | 24/32 | 27/32 |

## 完成时刻解释

`completion_distribution.json`及`E/analyze_completion_distribution.py`对固定到达过程逐事件积分：未完成请求数的时间积分 = 全部请求延迟之和，八格误差小于1e-8请求秒。它是事后会计，不是在线预测。

block0 native/least/most 首个完成分别19.604/21.012/23.104秒，第16个完成20.610/22.027/24.612秒，最后完成26.500/26.474/25.764秒。most把整批尾部提前，却让更早完成的一批请求后移；相对least，25/32请求更慢，平均增加1.763秒。block1相对least为24/32更慢，平均增加0.138秒，幅度受运行时间波动影响。两次most全调用均1315，least均1581；这些逻辑计数一致不能证明墙钟无噪声。

headroom首个请求约10秒就完成，但第16个要35秒左右；早退少数请求不保证整体面积更小。该例说明单独优化首次/末次完成或迭代数均不充分；不能由这些观察单独归因每一次victim选择的因果损益。

## 基线、完整性与边界

状态时间层级：DECISIONS.md是执行前冻结协议，标题的“未运行”保留历史时点；各格safe-cap-qualification.json是主测量前资格快照，其measurement_status=NOT_YET_RUN也保留。当前完成状态依据各格status.json与raw.status，以及campaign结束记录，不由初始化快照覆盖。顶层STATUS.json的audit字段在审阅进行时为IN_PROGRESS，收到审阅结果后更新；这与GPU测量已完成是不同状态。

- 最强简单策略已纳入headroom和most_output；对本轮三指标不存在一个通吃的基线。native平均完成更好；most整批wall更短；least/most停顿明显短于native。
- 前置APC d2结果只读复用其它会话`20260914_prefix_cache_baseline_r01/analysis/analysis.json`：缓存跳过1008位置但恢复服务未提前，最长停顿仍约4.7秒。APC d6未测，当前轮转不支持prefix sharing，不宣称击败APC配置。
- `analyze_d6_strong_baselines.py`核对实际块数、策略接线、源哈希、软件源、输入哈希、32请求完整、1024输出/请求、调用一对一与token守恒；全部通过。不同策略部分输出序列不同，不能称输出等价或质量已验证。
- 所有调用互斥分为重算混合/prefill/pure等，混合调用有效输出保留，见analysis.json progress；不把混合类全部用时计为纯重算税。
- 整组采用反序两block，共享flock及每格GPU初始化检查，全部边界无外部计算进程；不宣称持续监控保证。8格全exit0，无失败重试或有利运行替换。完整日志readback/campaign.log。GPU已释放给下一会话。
- 无总体噪声底、显著性、非劣、SLO或方法GO。尤其most相对least的block1平均+0.60%仅为观测值；不能称稳定可分辨损害。

## 判决与唯一下一步

Verdict：问题OPEN，既有三策略形成实测权衡；在线模型UNRUN。
Oracle/headroom：headroom是策略名称，不是Oracle；未测动作条件最优上界。
失败类别：当前固定规则未同时保住停顿与平均完成，属于目标/调度权衡；不是动作无效或整个问题无空间。
结论上限：此单模型、固定长度旧cohort、APC off、两反序运行的描述性结果。
复活/继续条件：同问题内候选动作能在计入受害请求恢复、其它请求进展和观测开销后减少完整请求代价。
唯一下一最小实验：在高压域选定一个实际可干预状态，从重建并校验的相同动作前状态分别执行least/most/不交换的独立分支，均推进到全部请求结束，核对恢复等待、受害者延迟与完成退出；先验证动作收益排序所需的状态模型，禁止复用单条未来轨迹或继续只凭全局步数选优。GPU执行尚未准备/未运行，不抢下一队位。

复跑分析：`python3 E/analyze_d6_strong_baselines.py --bundle O/20260914_d6_strong_baselines_r01 --results O/20260914_d6_strong_baselines_r01/readback/results --output <new-output.json>`；E/O分别为experiments/admission_capacity与outputs/admission_capacity，位于refine-logs/expert_saturation下。运行命令及输入见preparation/pkg，归档SHA256为798fa3643e7eea838953d29d87a8c894ec46c25e4b786e67e5d8bfad53fa7271。

定向审阅已完成：fresh GPT-5.6-Sol ultra，A-F PASS、无P0/P1，same-family/provisional；详见EXPERIMENT_AUDIT.md。没有升级方法或统计结论。
