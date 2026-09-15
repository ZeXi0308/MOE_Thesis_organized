# 同策略 CPU 定位：函数计时无效

2026-09-13。六格真实请求执行完成，身份/调度/字节会计 PASS；四份 cProfile 均为 **PROFILER_ACCOUNTING_INVALID**。本轮没有定位此前耗时漂移的原因。

同一 admit2_32 策略、OLMoE/vLLM0.26、expert cap16、KV512MiB、token64；顺序 plain、profile×4、plain。18测量请求/240输出，另96预热请求/1212输出。每格均27次engine调用、974组、91.55859GiB权重copy payload；逻辑前态、完整分组/加载轨迹与最终输出全部一致。物理KV IDs不同，未比较KV tensor bytes。

| 顺序 | 原始capture wall s | 包围capture的thread CPU s |
|---|---:|---:|
| plain_before | 2.409815 | 2.410008 |
| profile_1 | 3.238170 | 3.237778 |
| profile_2 | 3.242532 | 3.242754 |
| profile_3 | 3.238971 | 3.239359 |
| profile_4 | 3.164652 | 3.165012 |
| plain_after | 2.344809 | 2.344896 |

独立时钟与函数归因分开：第一份profile出现torch.split self +56.208s和attention self −56.202s抵消；后三份也有累计时间超过整个约3.2s CPU窗口的记录。全部signed函数/调用者原值保留在[summary.json](summary.json)，不删负项、不重分配、不输出排名或函数占比。前期将后三份初判为有效的说法在此更正。两次plain wall差−65.006ms、thread CPU差−65.112ms；带profile时间保留全部仪器影响，不估计或扣除profile税。

当前安装的Torch源码另提供一个可独立检验的因素：memory_allocated/max_memory_allocated为取current/peak两个标量，每次调用memory_stats递归展开并排序整棵统计树；memory_stats_as_nested_dict返回相同底层字段。expert _apply每层有三次此类观测。这个源码事实支持做等价读取对照，但异常profile中的0.46s读数不能作为已证成本，也不能解释旧漂移。

唯一下一实验：[flat/nested/nested/flat](../admission_memory_observer_v026/protocol.json)，同D策略、同字段、同位置和查询次数、不启用profiler。运行前检查当前/峰值等价，测原始完整成本；若收益与工作量一致则作为共同观测实现，随后恢复持续到达验证。否则保留原读法，不修饰旧结果。

Verdict：MEASUREMENT_ONLY，函数计时子实验INVALID；Evidence：人工专家预算下native请求执行。强基线为同一D的原始flat观测路径；exact Oracle/质量/SLO/真实超显存域均未测。失败类别为profiler会计异常，不能外推为调度问题NO-GO；重新使用函数归因须先有有效时钟/调用账。worker79239已退出；完整进程131.531s、初始化后102.737s。10份源码哈希一致，15次GPU边界检查通过，262个采样只见本worker，不能证明连续独占或CPU隔离。[回读回执](readback_completion.json)保留全部原件与archive SHA 6bc6b036…。
