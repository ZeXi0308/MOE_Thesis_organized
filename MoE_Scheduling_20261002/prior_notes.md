# MoE offload 下请求依赖调度：四篇定向查新

2026-10-02。范围：实际 router 已产生本层 top-k 后，安排专家加载/执行，使依赖齐备的请求先进入后续层；只核四篇一手原文及 HybriMoE 官方仓库入口，未运行任何系统。以下“未覆盖”仅指所核章节，不是全球新颖性结论。

**判断：异步跨层推进、per-expert queue、top-k 依赖齐备后放行都已有直接先例。** 当前可探测的是受限专家缓存下，按请求最后一个未完成依赖安排真实传输，是否比已能异步推进的简单策略多产生完整请求收益。把已有异步执行与 offload 拼接，尚不构成独立贡献。

| 一手来源 | 已有状态→动作→目标与运行域 | 对本候选的覆盖及剩余边界 |
| --- | --- | --- |
| [ExpertFlow 2410.17954v2，§3.3–3.4](https://arxiv.org/html/2410.17954v2#S3.SS3)，2026-04-02 更新、DAC 2026 | 预测所有层 route；重组相邻两个 batch，减少激活专家并提高复用；跨层分配专家缓存并预取，运行时纠正误预测。单 GPU offload，主要评价吞吐/显存。 | route-aware rebatching、跨层预取、cache/compute overlap 已覆盖。所核设计没有给出按当前真实 top-k 的最后依赖、逐请求释放到后续层的完成时间目标。不能因不使用 predictor 就直接宣称新颖。注意此编号已不是仅看 2024 初稿即可。 |
| [QLLM，EuroMLSys 2025，§3.1–3.3](https://arxiv.org/html/2503.09304#S3) | 每专家 FIFO、每序列保存 KV/route/部分专家结果；只有 top-k 结果完整的 token 才形成下一层 hidden；可在层内替换/暂停 BE 请求并优先 LS。评估使用 A100 80 GB、量化 Mixtral。 | **依赖跟踪、部分完成状态及脱离整批层同步已覆盖。** 论文政策是优先级抢占；未展示受限专家缓存、真实冷专家搬运顺序导致的请求完成收益。不能把每请求 join/跨层放行作为本线首创。 |
| [HybriMoE，§IV-B–D](https://arxiv.org/html/2504.05897#S4)，[官方实现](https://github.com/PKU-SEC-Lab/HybriMoE) | 当前激活负载与缓存状态→GPU 优先 cached/high-load、CPU 优先 uncached/low-load、PCIe 优先 high-load uncached；模拟 CPU/GPU/传输时间线，并按预期影响跨层预取。目标为混合执行延迟。 | cached-first、按专家负载排序传输、基于代价预测分配设备、跨层预取已覆盖。所核模型仍以完成全部专家的时长为目标，未显式优化“本次加载补齐哪些请求的全部依赖”。官方仓库存在，但本轮未逐函数核验或复现。 |
| [AMoE 2505.08944v2，§3.2、3.4](https://arxiv.org/html/2505.08944v2#S3.SS2) | token 元数据与 layer μ-queue；top-k 分支全部到达才入 ready queue，GPU 可执行任一就绪层；defragging 减少批次碎片。多 GPU、attention/expert 分离，resident EP；吞吐提高伴随 ITL 代价。 | **这是最直接的执行动作先例：逐请求 top-k join 后异步跨层推进。** 原文的 cold expert 指低负载，不等于不在 HBM。未覆盖本候选的单卡专家容量、PCIe 加载竞争和缓存淘汰成本；但仅迁移运行域不自动成立新方法。 |

**必要强基线。** 首先固定相同 cache 容量、初态、路由精度与真实传输后端：①普通层同步但允许当前层 compute/copy overlap；②相同依赖引擎的 work-conserving cached/ready-first，oldest-ready 打破平局，且同样允许跨层；③HybriMoE 风格 high-load-first 传输；④一维 oldest-request-first 加载其剩余专家；⑤最小“每毫秒能补齐多少当前请求依赖”的贪心规则。②–⑤必须有同样的异步执行能力，不能只以整批 barrier 为对手。完整 AMoE/HybriMoE 未运行时明确称 policy-style 移植，不冒充原系统复现。

**唯一值得先测的窄残差（推断/假说）。** 在真实容量不足而产生 expert misses 的 decode 域，两个候选加载量和专家负载相近，但一个加载能补齐少数等待请求的最后依赖，另一个只给许多请求增加部分完成结果：优先前者是否在“已异步、已 cached-first、已 age-aware”的对照后仍减少实际 next-token/请求完成时间，并抵消 batch 碎片、重复载入和后续层争用？只知道本层真实 route；后续 route 必须随各自执行重算，不能偷用未来轨迹。所有 top-k 均保留，无跳过/改路由。先限定 decode，不能直接把 prefill 的 token 依赖当成相互独立请求。

**最小裁决。** 同一 pre-action 状态分叉，先测真实 expert load→compute→请求 join→后续层→返回输出；以相同异步引擎只换传输/执行顺序。保留所有 peer 代价、实际传输字节、额外 kernel/attention 调用和缓存峰值。若已就绪请求通常仍被后续 miss 挡住，或请求级 Oracle 在完整成本后无剩余空间，停止 controller；若只胜过同步 barrier，结论归为已知异步机制的 offload 域验证。单层 join 提前只属 LOCAL_KERNEL/ISOLATED_GPU_RUNTIME，不能提前写 TTFT/P99/SLO 或 MoE 专属方法成立。
# 2026-10-02 后续有界候选核查（尚未实现）

单块KV预留完整六臂失败后，仅保留一个不同动作的wildcard：target verification已经进入当前MoE层、看到真实路由及驻留集合后，单调撤销draft尾部，而保留前缀的全部原始top-k贡献。它尚无GPU实现和性能结果；不是已有静态Q、KV reservation或专家删减的改名。

三篇定向primary-source核查：

- [SpecMoEOff §5.3](https://arxiv.org/html/2508.21706#S5.SS3)使用profiling与离线模型选择draft数量等配置；本次未发现层内真实路由驱动的token后缀撤销。
- [EVICT §3.2–3.3](https://arxiv.org/html/2605.00342#S3)每轮按draft置信度在线选择前缀长度，决策在target verification之前；不能把它说成纯固定长度。所用成本来自预先测得的平均值。
- [AcceptMoE](https://arxiv.org/html/2608.02989)已逐层结合target router与当前LRU驻留状态减少nonresident专家使用，但通过删除专家资格、重路由保留token实现，是近似目标模型。真实路由与驻留感知本身已有明确覆盖。

因此仅“层内撤销token后缀、保留幸存token原路由”在这三篇中未见直接覆盖，不能宣称已证明新颖性。因果注意力下前缀数学计算不依赖后缀，也不自动证明随机采样分布保持；数据依赖的截断可能有选择偏差，初始边界只能是greedy。前面已执行层的成本不可追回；专家仍被前缀/其他请求使用时，删后缀也不省加载。

唯一前置弱环节检查是固定min-ngram2/max5/k4对已有原生S输出串的prompt-lookup匹配机会，不扫描参数、不读取holdout输出、不当GPU加速或实际spec接受率。若机会存在，下一步也应先跑原生固定长度spec与强自回归baseline，再决定是否实现层内动作。

随机采样边界有直接反例：p=q=(1/2,1/2)，一token draft Y；仅Y=0保留并按标准spec接受，Y=1则放弃、重新从p采样。最后P(output=0)=1/2+1/2×1/2=3/4，非目标1/2。因此基于候选内容的截断加普通重采样一般不保分布。Greedy在相同target计算/固定tie下无此随机选择偏差，但BF16内核形状/计算顺序仍可能改变临界argmax，不保证位级一致。
