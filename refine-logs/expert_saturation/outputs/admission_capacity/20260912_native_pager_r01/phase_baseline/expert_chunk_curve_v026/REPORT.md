# Expert执行底座的prefill曲面：6/6 COMPLETE

Verdict：**ACTION_TRADEOFF_REMAINS / MEASUREMENT_ONLY**。相同输入/前态/资源下，chunk16的完整wall最短，8的旧请求max-ITL最小，32的新TTFT最低；三者没有同时覆盖另两者。不是调度方法GO，也没有证伪prefill/paging问题。

继承上一轮实际vLLM0.26/BF16/Triton expert分组、batched map、expert cap16/scratch3GiB、实际KV512MiB/token64/maxseq3/CPU0–7。source[4,5,10]为已观察校准文档，prompt32/32/128，output16/16/8。共同token前缀后入场，测量切expert与chunk8/16/32，各策略真实独立演进。每格共用token16→expert8→expert16→expert32→token16预热，episode写出/释放trace。顺序32/16/8/8/16/32在GPU前固定；第一block由seed2026091303打乱，第二block反序以平衡位置，不称两次独立随机复现。

| chunk | 完整wall，block0/1 ms | 旧最大ITL，ms | 新TTFT，ms | 全程payload GiB / groups |
|---|---:|---:|---:|---:|
| 8 | 2920.98 / 2930.27 | 127.32 / 127.26 | 1772.77 / 1769.53 | 117.387 / 1225 |
| 16 | 2444.07 / 2446.65 | 154.52 / 153.63 | 1143.78 / 1148.21 | 101.590 / 1032 |
| 32 | 2559.17 / 2660.08 | 211.01 / 225.73 | 827.92 / 874.39 | 89.133 / 917 |

8相对16：wall+19.513%/+19.766%，旧max-ITL−17.605%/−17.160%，新TTFT+54.991%/+54.112%。32相对16：wall+4.709%/+8.723%，旧max-ITL+36.563%/+46.935%，新TTFT−27.616%/−23.847%。两次每动作的payload/groups完全相同。所有六格全部输出token相同，共同prefix、逻辑前态、initial cache和分配一致；物理KV block IDs另列，未比较KV tensor字节。

首mixed call中，8/16/32分别为122.57/134.79/201.64ms（block0），121.71/136.85/208.13ms（block1）；payload4.7578125/5.4140625/6.52734375GiB，groups48/48/58，loads406/462/557。旧token执行时间曲线不适用于这个底座。以上是动作执行后的诊断，不能当作决策前已知的未来route。

## 少搬却更慢的阶段账

[phase_cost_diagnostic.json](phase_cost_diagnostic.json)按真实scheduler rows把每个engine call放进一个互斥阶段，另外保留call外online gap。chunk32提前结束prefill，使三请求decode从4次变成7次，全部位于旧请求完成路径内；16有3次新请求decode在旧请求完成后执行。

| 32−16阶段wall，ms | block0 | block1 |
|---|---:|---:|
| 新prefill与旧decode混合 | −317.373 | −274.811 |
| 三请求decode | +402.633 | +452.672 |
| 仅旧请求decode | +82.100 | +87.260 |
| 仅新请求decode | −91.152 | −92.339 |
| 共同前缀 | +32.771 | +34.359 |
| call外online gap | +6.120 | +6.287 |
| 完整wall差 | +115.099 | +213.428 |

重构残差0。共同前缀在动作前，它的差异不能归因于chunk。此分解定位了观测成本所在阶段，尚未测固定batch composition的反事实decode成本。32自身wall重复漂移+100.912ms主要分布在混合prefill和三请求decode；GC回调重叠仅增加5.501ms，无gen2事件，未扣除GC，也不宣称GC已被排除为所有抖动来源。

## 模型修正和唯一下一动作

prefill动作同时决定其自身步骤数，以及新请求何时加入decode集合。因此剩余成本应沿真实状态转移累计：`sum T_step(B_t,C_t,executor) + online gaps`，不能只按总bytes或prefill结束时间排序动作。未来route依然内生，完整后续状态必须重跑。

当前最弱因果链是三请求decode的分组开销是否能通过实际执行决策降低。一个无需未来路由的结构条件为：每token最多k个专家、cache cap C，则M行的专家并集不超过Mk；当M≤floor(C/k)时，none模式expert分组至多一组。本模型C16/k8对应M≤2。这不保证完整收益，因为更多model steps和被延后请求的等待可能抵消节省。

唯一下一实验：仅在纯decode三请求阶段轮转选两条，原生scheduler临时token预算64→2并保留未选请求KV；prefill仍32，所有请求完整代价计入。与static32匹配对照、最佳wall的static16及较低停顿static8做四臂正反序八格。它新增的是decode cadence动作，不能称为只改prefill；当前代码准备中，GPU **UNRUN**，不根据结果调这个由C/k决定的2。

证据层级为NATIVE_SERVING中的单模型/人工小专家池、三请求校准。没有独立到达、任务质量、SLO-goodput、真实超显存模型、exact Oracle或新颖性主张。原始数据/失败与全部重复保留；少量重复差不是噪声上界。

证据：[summary](summary.json)、[原始结果](readback_r01/results/)、[协议](protocol.json)、[实际源码](source/)、[完成记录](readback_completion.json)。协议PREPARED_UNRUN为运行前快照。
