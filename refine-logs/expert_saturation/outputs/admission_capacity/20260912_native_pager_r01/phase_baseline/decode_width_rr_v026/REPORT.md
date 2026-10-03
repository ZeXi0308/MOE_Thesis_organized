# Decode宽度2轮转：单组条件成立，完整请求收益未成立

2026-09-13；状态 **MEASUREMENT_ONLY / NO_ADVANTAGE_DEMONSTRATED_FOR_RR2**。

问题是：C16/top-k8下，三条纯decode请求每步轮转执行两条，能否用单层单组的节省抵消额外模型调用和轮空等待？本轮答案是没有取得优于匹配static32的证据；不能把单组条件直接当作请求收益。

执行为同一vLLM0.26/OLMoE BF16/expert-none/batched-map底座，cap16、KV512MiB、token64、maxseq3，source4/5/10的32/32/128输入及16/16/8输出。四臂static8/static16/static32/rr32正序后反序，共8格、24次测量请求/320输出；152次预热请求/1936输出另外保留。模型加载前的一次GPU_BUSY完整保留，重试在连续三次空闲检查后执行。过程总时长205.97s，初始化后含预热/flush/关闭的周期167.24s；410个GPU样本仅自身worker72254，不声称连续独占或CPU无干扰。

| 臂 | capture wall s，正/反 | 旧maxITL ms | 新TTFT ms | 新maxITL ms | 新TPOT ms | payload GiB / groups |
|---|---:|---:|---:|---:|---:|---:|
| static8 | 2.92195 / 2.93548 | 137.64 / 146.35 | 1777.74 / 1787.89 | 38.95 / 38.68 | 32.40 / 32.78 | 117.38672 / 1225 |
| static16 | 3.04207 / 3.04208 | 199.05 / 197.19 | 1493.76 / 1489.98 | 116.90 / 113.30 | 85.63 / 85.57 | 101.58984 / 1032 |
| static32 | 2.12476 / 2.11955 | 170.00 / 172.98 | 664.79 / 665.89 | 82.11 / 81.26 | 69.18 / 68.21 | 89.13281 / 917 |
| rr32 | 2.21501 / 2.72075 | 187.96 / 219.15 | 682.92 / 852.77 | 113.75 / 165.23 | 82.03 / 122.03 | 90.67969 / 874 |

所有请求最大ITL在这8格均等于表中的旧maxITL；完整逐请求TTFT/TPOT/ITL/完成时间见[summary.json](summary.json)。这张表是各策略自然生成的实测读数，不是稳定效应量。rr32相对static32 wall高4.25%/28.36%，新maxITL高38.54%/103.32%；这些百分比受以下漂移边界约束。

每个rr32有11个真实激活步、176个layer调用，全部为两行、top-k8、单组，held保持RUNNING且KV块/computed/status不变，下一非空步均获选。原始旧请求`existing_decode_all_scheduled=False`保留，只有实际验证的held豁免；没有抢占、重算或丢请求。step界不等于时间界。

单组没有减少完整工作：总engine calls16→20，groups917→874（−4.69%），payload89.13281→90.67969GiB（+1.74%）。纯三请求阶段由7次全体decode（219组、16.04297GiB、482.66/475.91ms）变为11次RR调用（176组、18.53906GiB、571.68/848.52ms）。轮转减少一个layer-call内部的组数，却增加跨step调用与实际加载；这些是各臂独立演进的计数，不是固定route反事实。

共同token前缀、入场前逻辑态、分配与缓存均一致；同臂重复完整执行结构和输出相同。rr32与static32的一条旧请求最终输出不同，因此不声称相同token轨迹或任务质量等价。所有动作各自执行真实模型/KV/router。

可分辨性边界：rr32自身wall漂移505.74ms（22.83%），且新TTFT在首次RR动作之前已确定；其相对static32的18.13/186.87ms差不能归于轮转动作。混合prefill时间也在RR首次动作前变化18.09/184.95ms；两臂该阶段字节和分组相同。共同前缀变化1.07/29.59ms也不能归因于RR。旧最大ITL若出现在该前动作阶段，同样不能用来量化轮转的损害。static16/32相对上轮曲线的时间排序变化说明跨campaign不能直接合并时间表。没有把同臂最大漂移当作总体噪声界，没有扣掉GC或筛掉较慢RR格。

[互斥阶段账本](phase_cost_diagnostic.json)将完整wall差重建为各实际engine阶段差加call外gap，残差0；host_apply与CUDA span重叠，不相加。所测是人为OLMoE缓存预算、单triplet、同engine反序两次观察；未测独立到达、真实超显存模型、SLO、任务质量和exact Oracle。最强同域实测wall基线为本轮static32，低旧暂停基线为static8。原生admission cap2且保留第三请求仍是缺失强简单基线，不用旧少一请求hold替代。

停止把当前RR2单组条件作为有益机制推进，不通过换宽度或阈值抢救。研究问题仍OPEN；下一唯一实验是同到达/同三请求工作量下的immediate32、admit2_32、admit2_release64六格：第三请求照常入WAITING并计全部TTFT，等已有请求释放名额，比较普通准入和完成后释放prefill预算是否已解决该小实例的成本转移。

复算命令：`python3 analyze_decode_width_rr.py --input-dir readback_r02/results --out new_summary.json`；`python3 diagnose_phase_cost.py --input-dir readback_r02/results --out new_phase_cost.json`。原始source/protocol和两次尝试不改；[完成回执](readback_completion.json)记录SHA及实际请求计数。独立新动作完整性核查仍在进行，其结果单独追加。

独立定向审计已完成：[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)，GPT-5.6-Sol ultra、same-family/provisional，WARN、P0=0/P1=0。复算一致；两格旧maxITL均确认发生在首次RR动作前，不能归因于RR。首个输出分叉是0507第14个生成token，在调度变化之后；自然执行有效，质量等价未测。该次核查到此结束。
