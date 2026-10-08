# LongBench150：共同大奇数批次 warmup 控制

V1 控制块为 **INCOMPLETE**：DFS 格完成150条，但 whole 格在首个普通 warmup 结束前，由首次 KV 分配观察器把 vLLM 内部随机后缀 ID 与外部请求 ID 直接比较而失败；whole 正式150条测量未启动。V1 原件保留，不纳入对照。V2 只将观察器改为通过 `OutputProcessor.request_states` 的内部 ID 键取精确外部 ID，随后同机 **DFS→whole** 两格均完成。固定模型、输入、排序、采样、原生 APC 与资源未因此改变。

V2 两臂在原首尾 warmup 之后、正式测量之前，共同增加源序0题 prompt 前1023 token 加1个新token的形状 warmup。额外 warmup 前后均重置 APC，正式测量前 reset 合格；每臂均实际调度一次1023-token批次。额外过程总墙钟为 **DFS 0.712568s／whole 0.690677s**（其中生成0.711645／0.689753s）；这些是另列的预热部署成本，未计入正式请求 host 观察起点。

| V2 正式测量 | DFS | whole（完整子树密度） |
|---|---:|---:|
| 完成请求 / prompt调度token | 150/150 / 386,498 | 150/150 / 386,498 |
| 实际输出token / 自然EOS / 封顶 | 3,940 / 127 / 23 | 3,951 / 126 / 24 |
| 全输出English QA F1 | 37.13733% | 37.08429% |
| 平均到达至完成 | 5.320528s | 4.560885s |
| p95到达至完成 | 9.447309s | 9.717581s |
| 全批host观察区间 | 9.791680s | 10.049753s |
| 每请求最大不同host返回间隔的最大值 | 0.031218s | 0.031578s |
| after-schedule活跃KV块峰值 | 3,163 | 3,269 |
| 最慢正式step | #74：0.031186s | #234：0.031547s |
| 抢占 | 0 | 0 |

本格 whole 相对 DFS 的平均 flow 下降0.759644s，但 p95 上升0.270272s，全批时间也上升；87条请求 flow 改善、63条恶化。两臂有6条输出ID、2条输出长度、1条结束原因变化，F1下降0.0530个百分点，因而不是等输出工作量加速。V2 DFS 的逐请求输出ID／文本／结束原因与先前同机 primary DFS1、DFS2 **150/150相同**，whole 与 primary density1、density2 也 **150/150相同**；各自434或441次正式调度的 token 总数／请求数及提交顺序也逐次一致。

先前未加共同形状warmup的 primary density1／density2 在同样911-token的正式step380分别耗 **0.701143／0.716933s**；V2 whole 的step380为 **0.025791s**，全格最慢step仅0.031547s。结合[单次选步profile](C_LONG_DOCUMENT_QA_GAP_PROFILE_RESULT_20261002.md)直接观察到的Triton编译调用，此结果与可预热的形状编译解释一致，不能再把旧0.7秒当成子树连续排序的固有执行成本。这里比较的是不同轮次的结构观察；**没有**做匹配的warmup关／开性能实验，也未精确定位完整specialization键或建立预热必然消除停顿的因果保证。

[V2分析摘要](long_document_qa_shape_warmup_v2_analysis/summary.json)SHA-256 `ef3919a90fe6e0e15d49de6f8674dc76288b4eae780846c483e7d63123b79b92`，[完整逐请求对照](long_document_qa_shape_warmup_v2_analysis/dfs-vs-whole.json)SHA-256 `d5257f7d3f4a75f1cbfcda5f94ac118f8f80bbbc2f23f77b3beab82d2072955a`。完整V2原始归档 `shape-warmup-control-dev-v2-complete.tar.gz` 为6,671,544字节，SHA-256 `67bc72dc7052ed862b45818fde06263b5664960af9f425f3fa0ea728bd1fb908`。V2块与两格均为COMPLETE，子进程回收、GPU清空及私有stage移除由原始launcher回执记录。
