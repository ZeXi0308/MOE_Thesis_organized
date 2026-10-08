# 默认offload反序共同decode诊断

Verdict: MEASUREMENT_ONLY / NATIVE_PROFILER_DIAGNOSTIC。反序复现了offload-on额外主机侧时间；未证明具体函数、可消除税或方法净收益。两格COMPLETE，GPU已释放。

固定旧d6/32请求/OLMoE BF16/native vLLM0.26/单5090，on16GiB→off0。500..531调度签名均匹配冻结expected_window；32/32完整输出一致；每臂32CPU标记与10274GPU活动，全部GPU活动可匹配trace中的CUDA API correlation。不同物理KV布局仍是on/off的一部分。

| 32调用，ms | off | on |
|---|---:|---:|
| 总窗口 | 640.596 | 844.566 |
| CPU scope并集 | 621.238 | 817.263 |
| GPU活动并集 | 511.943 | 511.607 |
| CUDA API并集 | 515.927 | 517.263 |
| 首次CUDA API前 | 34.487 | 96.400 |
| 首次API至同步返回 | 552.413 | 609.328 |
| 同步返回至scope结束 | 34.338 | 111.535 |

最后三行互斥闭合到CPU scope；其余行互有重叠，不能相加。GPU活动之外窗口增加204.306ms，GPU活动自身变化−0.336ms。旧顺序对应219.330/−0.150ms，方向反序仍在。但不能将两组当新workload确认或统计显著性证明。

主机侧分布有变化：旧组scope额外212.895ms中147.852ms位于首次API前；本组scope额外196.025ms，首次API前仅61.913ms，同步返回后77.197ms。on调用527同步返回后66.556ms，是保留的真实长停顿，不删去，也不能未经观测称GC。持续开销与偶发停顿可能并存。

新增driver只读采样：on窗口1789329779.137..9779.982，由1789329778.572..1789329780.636两样本包围；off窗口1789329861.283..9861.923，由1789329860.811..9862.875包围。两包围区间cgroup nr_throttled/throttled_usec增量均0；主进程CPU时间分别2.09/2.07s。cgroup cpu.max=2500000/100000（25核配额），亲和性208核。全机MHz中位数800不代表实际执行核心频率，不能据此排除频率差异。负载不同、采样约1s且跨窗口，不能提供逐调用因果解释。

新增采样本身只存在于本反序组，两臂一致但与旧组不同；profile/顺序/运行环境仍影响幅度。完整运行wall不作性能GO，先前原生baseline平均完成退化的判断不因此改变。最强对照为同资源同任务offload关闭臂，未建立Oracle上界或方法headroom。失败类别：成本源定位未闭合；未测试full-decode保存、轮转connector兼容、多卡或自然EOS质量。

唯一下一最小实验：固定同一窗口，在offload-on上做一次Python调用级定向采样并记录GC起止，定位首次API前持续开销和同步后尖峰属于哪条实际路径；先定位再消融，不凭无CUDA活动时间直接禁用GC或删除connector逻辑。该观测仅诊断，不增加controller，不用新profile时长宣称性能收益。

归档SHA256 4abf56f05855e5987178986559bf51d5677177d950e9a2158488eada5df1f837。analysis.json、host_phases.json、qualification.json为派生结果；执行者定向核验，未宣称fresh独立审计。
