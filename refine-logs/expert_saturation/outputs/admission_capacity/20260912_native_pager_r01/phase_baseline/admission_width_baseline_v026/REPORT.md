# 完整三请求的准入强基线

2026-09-13；**MEASUREMENT_ONLY / WAITING_TRADEOFF_OBSERVED / TIMING_ORDER_UNRESOLVED**。

问题是普通native admission cap2是否已解决该triplet的成本转移。六格已实际完成：I=immediate32，D=admit2_32，R=admit2_release64，顺序I,D,R,R,D,I；18测量请求/240输出，另96预热请求/1212输出。均同vLLM0.26/OLMoE BF16、expert-none、batched map、cap16、KV512MiB、token64、engine maxseq3。第三请求仍在两旧各4输出时真实入队；D/R保留它的全部排队时间与工作量。不是旧少跑一请求的hold控制。

| 臂 | 完整wall s，正/反 | 旧maxITL ms | 旧完成时刻 s | 新TTFT s | 新完成延迟 s | payload GiB / groups / calls |
|---|---:|---:|---:|---:|---:|---:|
| I | 2.13073 / 2.12044 | 173.73 / 172.83 | 2.13047 / 2.12026 | 0.67242 / 0.67088 | 1.16098 / 1.15341 | 88.66406 / 917 / 16 |
| D | 2.93602 / 2.35821 | 71.56 / 65.68 | 1.67362 / 1.46809 | 1.58248 / 1.20906 | 2.02345 / 1.44902 | 91.55859 / 974 / 27 |
| R | 2.41558 / 2.77852 | 70.59 / 100.74 | 1.64112 / 1.86743 | 1.14890 / 1.39660 | 1.48236 / 1.83176 | 80.05078 / 877 / 25 |

完整逐请求maxITL/TPOT/完成、所有配对差和重复差见[summary.json](summary.json)。同臂实际路由/加载/分组结构和最终输出均重复；I与D/R的一条旧请求最终输出不同，因此只比较各臂自然执行，不声称相同token轨迹或质量等价。初始分配、cache及共同入场前态一致，物理KV IDs另存，KV tensor bytes未比较。

D/R在step4–15的12步中让第三请求保持WAITING/computed0/KV空，已有两条decode每步均推进且无抢占。该实例两旧长度相同，因此两者同step完成；新请求首调度由I的step4推迟到step16。D按[32,32,32,32]完成新prefill，R在旧完成后将threshold32→0、实际按budget64执行[64,64]。所有cell结束均排空并恢复admission cap3/threshold32。

TTFT账本为`到达→首调度 + 首调度→首token`，不能把队列等待再加到已经包含旧请求执行的wall中。D两格是762.17+820.32=1582.48ms、559.31+649.75=1209.06ms；R是708.85+440.05=1148.90ms、921.68+474.92=1396.60ms。延后准入减轻了旧暂停，但本例新TTFT与整批时间均更高。R相对D实际prefill加载26.87109→15.51563GiB、225→128组；完整payload少12.57%，完整wall差却−17.73%/+17.82%翻号，不能据其中一格宣布净收益。

运行漂移必须先定位：D自身wall−577.81ms（−19.68%），R+362.94ms（+15.02%），I仅−10.30ms。D/R共同旧等待阶段的实际rows/routes/loads/groups签名完全相同，12步/192组/16.10156GiB；其时间D754.92→555.98ms、R701.98→913.32ms，R的释放动作尚未发生，故这部分差不能归为release收益或损害。

[有限运行诊断](runtime_variation_diagnostic.json)显示D/R完整主线程CPU差−576.27/+360.83ms，外层observer包络差仅−7.13/+2.75ms，GC真重合差−47.55/+17.88ms，CUDA load span差−40.00/+28.62ms。这些时间包含或重叠，不能相加或从wall中扣掉。没有新增cgroup throttling，但sched_schedstats=0，小runqueue读数不能排除干扰；CPU time含驱动与轮询，不能直接归因Python计算。D快慢均在CPU5；仅钉核不足以解释漂移。主要未定位部分是MoE host残余路径及engine内其他路径。

证据是人工缓存预算、单triplet的真实native请求测量；未测独立到达、真实超显存模型、质量、SLO、统计显著性/非劣性和exact Oracle。最强同轮完整wall/新TTFT基线为I，D/R保留旧暂停这一端的权衡。没有观察最大值噪声界，没有问题级NO-GO。

唯一下一GPU实验改为固定D的plain/profile/profile/profile/profile/plain：只增加CPU函数profile以定位计费路径，其他policy、资源、暖机、GC和affinity不变。profile结果仅作诊断，保留两次未profile的完整时钟，不将profile内部时间当原策略性能。8请求持续到达入口已CPU准备，但在当前符号翻转原因未定位前不扩大策略矩阵。

worker77465已退出；进程131.64s，初始化后完整周期104.36s，空闲等待另计。[完成回执](readback_completion.json)保留源码SHA、原始回读SHA2364c5a2…与所有计数；预检等待及全程采样保留，未杀外部进程。复算：`python3 analyze_admission_width.py --input-dir readback_r01/results --out new_summary.json`，原文件不覆盖。
