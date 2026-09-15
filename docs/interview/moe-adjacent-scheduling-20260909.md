**MoE 相邻调度：文献对照与面试补充**

核对日期：2026-09-09。供当前论文面试使用；15篇论文、5份官方文档。文献结果属于作者，当前实验事实仍以各轮本地报告为准。本次核验书目与关键机制/实验段，未复现论文性能，也不宣称穷尽最新工作。

直接口述答案已加入 [json.md](../../json.md) 的 `high_pressure_answers`（相邻01–25）；每题的来源映射和版本信息在 `adjacent_scheduling_research`。

先按动作层次理解这些方法：

| 层次 | 看什么状态 | 实际动作 | 代表来源 |
|---|---|---|---|
| 集群作业准入与资源份额 | 租户配额、资源需求、作业优先级 | 允许作业启动、借用资源或回收配额 | [DRF](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types)、[Kueue Overview](https://kueue.sigs.k8s.io/docs/overview/) |
| Pod放置与组调度 | 节点容量、拓扑、组成员与预留 | 选节点、Reserve/Permit或回队 | [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)、[Kueue All-or-nothing Scheduling](https://kueue.sigs.k8s.io/docs/concepts/all_or_nothing/) |
| 副本与阶段分配 | 阶段负载、KV、带宽、专家签名 | 选择PD资源、decode worker或迁移请求 | [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)、[TetriInfer](https://arxiv.org/abs/2401.11181)、[Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)、[ELDR](https://arxiv.org/abs/2607.00466v2)、[Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) |
| 请求选择与迭代合批 | 长度、服务账本、运行阶段与token预算 | 选下一请求、切prefill、迭代边界抢占 | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)、[FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)、[VTC](https://www.usenix.org/conference/osdi24/presentation/sheng) |
| KV和权重驻留 | 物理块、前缀引用、专家工作集与搬运成本 | 分页、复用、回收、offload或预算重分配 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)、[SGLang / RadixAttention](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)、[WiSP](https://arxiv.org/abs/2606.21868v2)、[FluxMoE](https://arxiv.org/abs/2604.02715v2) |
| 专家部署与通信 | 专家热度、traffic matrix、设备与链路 | 复制/放置专家、重排通信 | [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)、[Aurora](https://arxiv.org/abs/2410.17043) |

**论文对照**

| 工作与发表状态 | 状态与动作 | 目标及证据范围 | 与当前工作的关系 |
|---|---|---|---|
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)；Gyeong-In Yu et al.；OSDI 2022 | 请求阶段、迭代完成、batch上限和KV容量；逐迭代调度与selective batching；新请求按max_tokens预留KV | 生成服务的吞吐与延迟。Transformer serving及模型并行；不是MoE专家信号实验 | continuous batching及保守预留已有先例；当前cap29的价值是实测现代原生恢复基线下的等待代价。 |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)；Woosuk Kwon et al.；SOSP 2023 | 动态增长的KV与可共享前缀；分页管理KV并支持共享，减少碎片和重复存储 | 相同延迟水平下的服务吞吐。论文在其模型、基线和解码设置中报告收益；不等同当前vLLM版本 | 本研究复用该类KV管理能力；不能把分页或cap29公式归为自研。 |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)；Amey Agrawal et al.；OSDI 2024 | decode集合、prefill余量、token预算及TBT目标；切分prefill并按预算与decode合批，控制迭代工作量 | 尾部token延迟约束下的服务容量。单GPU、TP与PP；模型和硬件需校准 | 256/512/1024实验是特定原生运行域的预算干预，不是完整复现，也不判死chunked prefill。 |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)；Yinmin Zhong et al.；OSDI 2024 | TTFT/TPOT目标、长度/到达率、profiling与互联带宽；PD放到不同GPU，分别配置资源与并行方案，并按带宽约束放置 | 满足TTFT与TPOT的每GPU goodput。多GPU；部分高跨节点带宽方案为模拟评估 | 当前单引擎预算与排序未测PD传输，不能直接外推分离净收益。 |
| [TetriInfer](https://arxiv.org/abs/2401.11181)；Cunchen Hu et al.；2024 arXiv preprint；本轮未核验到正式venue | prompt长度、预测输出区间及decode实例负载；PD分离；有限排序窗口内FCFS/SJF/LJF；固定prefill chunk与预测式decode放置 | TTFT、JCT与资源效率。多实例分离式服务；已有request-level KV传输 | 与短prompt优先直接相邻；当前没有其排序窗口、预测器和跨实例放置。 |
| [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)；Bingyang Wu et al.；NSDI 2026；2023预印本旧题Fast Distributed Inference Serving for Large Language Models | 输入长度、已执行时间、队列层级与KV压力；Skip-join MLFQ、迭代边界抢占、超时提级及KV换入换出 | 降低长任务阻塞并提高满足延迟目标的吞吐。主要A100/FP16与Dense模型；正式论文vLLM基线为0.6.1 | 当前never-started排序没有MLFQ执行权切换；原生缺KV抢占也不是实现了FastServe。 |
| [VTC](https://www.usenix.org/conference/osdi24/presentation/sheng)；Ying Sheng et al.；OSDI 2024 | 各客户端已获得的输入/输出token服务成本；逐token记账；有接纳容量时优先低counter客户端，并校正重新入队的计数基线 | 服务量公平与闲置份额利用。共享LLM的多个客户端；基本方案无运行中抢占 | 比请求数轮询或短prompt排序多了跨客户端服务账本；当前未实现。 |
| [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)；Biao Sun et al.；OSDI 2024 | 实例KV占用、等待需求、优先级与virtual usage/freeness；分发新请求，并分阶段复制KV以迁移运行中请求 | 跨实例负载、容量、优先级与弹性。论文16张A10、LLaMA-7B/30B多实例 | 跨实例空闲容量碎片不同于PagedAttention解决的KV地址管理；单卡实验未覆盖迁移。 |
| [SGLang / RadixAttention](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)；Lianmin Zheng et al.；NeurIPS 2024 | radix树的匹配前缀、KV引用及可回收缓存；复用前缀KV，按匹配前缀组织请求并回收缓存 | 减少重复prefill和提高执行吞吐。共享前缀、多轮与分支程序等负载 | 按已缓存前缀与按总prompt长度排序会选中不同请求；当前无prefix cache，未验证该动作。 |
| [WiSP](https://arxiv.org/abs/2606.21868v2)；Jiamu Zhang et al.；2026 arXiv preprint，v2（2026-08-30） | 专家工作集、复用与KV需求；专家分页、expert/KV预算分配；动态resize在排空边界执行 | 低显存、低并发下的serving延迟与容量。v2主结果为真实RTX3090 24GiB；OLMoE分配实验另为H100受限预算 | 专家与KV联合分配已被直接研究；当前提高memutil不等于实现可回收专家池。 |
| [FluxMoE](https://arxiv.org/abs/2604.02715v2)；Qingxiu Liu et al.；2026 arXiv preprint，v2（2026-04-30） | KV压力与计算/加载成本；PagedTensor、无损压缩GPU存储与host DRAM分层、按需流水加载专家 | 受显存约束时的总token吞吐。四张L40 48GB、vLLM0.10.2、TP2/4、batch32–256 | 权重驻留回收不同于普通KV参数调优；大batch吞吐不能转写为当前单卡SLO收益。 |
| [ELDR](https://arxiv.org/abs/2607.00466v2)；Sangjin Choi et al.；2026 arXiv preprint，v2（2026-07-02） | prefill expert signature、decode worker负载与KV-block signature cache；PD交接时，在局部性候选worker内按负载选择decode目的地 | 平衡负载并提高decode batch的专家复用。MI300X 192GB/400Gbps IB的PD分离；主结果24GPU，EP扩展最多40GPU；模型有BF16与MXFP4 | 按专家签名分流有直接先例；当前单引擎cap/排序是另一动作域，但不同位置本身不构成新颖性。 |
| [Aurora](https://arxiv.org/abs/2410.17043)；Jialong Li et al.；2024 arXiv preprint，v1；未核验正式venue | 历史traffic matrix、阶段时长与设备能力；跨模型专家共置、GPU分配与all-to-all发送顺序 | 模型推理时间。视觉MoE统计驱动的网络/部署仿真 | 属于部署与通信层；不能用其仿真结果支持单卡连续decode改进。 |
| [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)；Jiamin Li et al.；USENIX ATC 2023 | 前层路由估计的专家热度及偏差；推理专家复制/打包与设备映射；训练另调度all-to-all与allreduce | 分布式MoE训练及一批推理的时间。最多16张A100 40GB、100Gbps IB；推理top1、训练top2 | 需要真实多卡专家与通信路径；当前没有这些动作或共同测量分母。 |
| [DRF](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types)；Ali Ghodsi et al.；NSDI 2011（基础工作） | 租户在各资源上的份额与需求向量；按最大资源份额做多资源max-min公平分配 | 多资源共享的公平与效率。固定资源需求模型及Mesos评估 | 适合解释资源份额公平；VTC计的是随时间获得的token服务，二者对象不同。 |

这些工作给当前材料带来的四点补充：

迭代合批、prefill切分、最大长度预留和按prompt排序都已有明确先例。当前实验更适合解释这些普通动作在特定原生运行域中的完整代价；某个小预算或排序负结果，不构成对应方法家族失败。 参见 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)、[TetriInfer](https://arxiv.org/abs/2401.11181)。

公平性的对象需要先明确：VTC计量已获得的token服务，DRF讨论多资源份额；缓存局部性、平均完成时间和最坏等待也不是同一目标。调度器必须说明让谁受益、谁承担等待，以及保证在哪些条件下成立。 参见 [VTC](https://www.usenix.org/conference/osdi24/presentation/sheng)、[DRF](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types)、[SGLang / RadixAttention](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)。

多副本系统增加了阶段资源配置、decode worker选择和运行中迁移等动作，但同时引入KV复制、网络与目的端容量。单卡cap或队列结果不能直接支持这些系统的收益；核对时按共同动作和完整成本分母比较。 参见 [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)、[Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)。

MoE邻近工作已经覆盖专家/KV预算、权重驻留回收、专家签名路由和通信调度。下一步问题应来自强简单策略之后仍存在的请求损害；不同硬件、不同精度或将特征放在另一个层次，本身都不是独立新颖性。 参见 [WiSP](https://arxiv.org/abs/2606.21868v2)、[FluxMoE](https://arxiv.org/abs/2604.02715v2)、[ELDR](https://arxiv.org/abs/2607.00466v2)。

**容易被追问的边界**

| 面试追问 | 必须接住的区别 | 对应题 |
|---|---|---|
| cap29是不是新容量机制？ | 最大长度预留已有先例；当前贡献需落到真实请求代价。 | 相邻01–03 |
| 小预算不好，是不是Sarathi错了？ | chunk大小受SLO和成本约束，局部ITL与TTFT目标不同。 | 相邻04 |
| 短prompt优先是不是SJF/MLFQ？ | 输入排序、已用服务量和真实剩余工作量是不同信息；MLFQ还改变运行权。 | 相邻05–07 |
| 每租户一请求，或每token换租户，哪个公平？ | 服务量账本、接纳时机和运行中抢占要分开；VTC基本方案无抢占。 | 相邻08–10 |
| PD、迁移和缓存能否直接拿来？ | 必须付KV传输、目的端容量、等待与公平性成本。 | 相邻11–15 |
| 专家与KV联合预算有创新吗？ | WiSP/FluxMoE已有直接动作，ELDR也有专家签名路由。 | 相邻16–19 |
| 有8卡配额为何任务不起？ | 总配额、物理放置与全组启动不是同一保证。 | 相邻20–23 |
| 论文多，是否必须全部复现？ | 按相同动作/资源选择最强简单和最近邻基线。 | 相邻24–25 |

**官方行为核验**

- [vLLM v0.26.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.26.0/configuration/optimization/)：V1重算恢复与chunked prefill预算权衡；文档建议必须在具体配置中验证，默认行为不能覆盖实验实际patch。
- [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)：选择可行节点、排序、临时记账及绑定前等待；不是每个token的调度循环。
- [Kueue Overview](https://kueue.sigs.k8s.io/docs/overview/)：决定workload何时准入或被抢占，不替代kube-scheduler的Pod到Node放置。
- [Kueue All-or-nothing Scheduling](https://kueue.sigs.k8s.io/docs/concepts/all_or_nothing/)：总配额可预留不等于节点布局放得下；等待Ready超时回队属于补救，不能称所有Pod原子启动。
- [Kubernetes Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)：HPA改变副本数；稳定窗口可限制抖动。扩副本不是引擎内立即放行请求。

**阅读与版本备注**

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)：官方书目、摘要、selective batching与Algorithm 1/KV reservation相关正文。最大长度预留保证其模型下后续KV可分配，不保证性能最优；有逐迭代控制成本。
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：作者arXiv书目与摘要；当前行为另核v0.26调优文档，未重审2023全文实现。减少存储浪费不等于物理KV容量无限；当前抢占行为另外查v0.26官方文档。
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)：官方书目与正文4.1–4.3、5.4.1。小chunk存在利用率、重复KV读取及固定开销；不能解释成预算越小越好。
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)：官方书目、正文4.1–4.3、通信代价及评估边界。KV传输、阶段排队和两侧容量均需计费；分离不自带高级抢占与容错。
- [TetriInfer](https://arxiv.org/abs/2401.11181)：arXiv v1书目与正文3.2、3.3.1–3.3.4。长度预测有误差和成本；排序窗口限制新到短请求持续插队，非任意过载下SLO保证。
- [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)：NSDI正式PDF 2.3、3、4.1–4.2、6.1及baseline说明；旧预印本去重。抢占有KV搬运和恢复等待；MLFQ不需要真实总输出长度，也不保证任意请求必达SLO。
- [VTC](https://www.usenix.org/conference/osdi24/presentation/sheng)：OSDI正文2.1、3、4.1、Theorem 4.4与4.3加权变体。论文模型下持续backlogged客户端的服务差有界，不是逐请求TTFT或SLO保证。
- [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)：官方书目；作者正文4.2、4.4、5、6.1。迁移需要目的端余量、复制带宽及最终切换停顿；低停顿是实测结果，非零成本保证。
- [SGLang / RadixAttention](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)：NeurIPS正式PDF第3节、Theorem 3.1与附录相关算法。离线缓存命中定理有给定请求集及容量条件；在线贪心可能饥饿，命中最优不等于TTFT最优。
- [WiSP](https://arxiv.org/abs/2606.21868v2)：v2摘要、设计与第5节设置；核对v1/v2硬件变化。PCIe带宽决定预取是否有利；输出一致是作者测试结论。v1模拟小卡结果不能混入v2。
- [FluxMoE](https://arxiv.org/abs/2604.02715v2)：v2摘要、正文4–6及评估指标。主评估不以TTFT/TPOT为目标；未核实统一dtype，不擅填BF16；搬运必须进入完整成本。
- [ELDR](https://arxiv.org/abs/2607.00466v2)：v1机制与设置初读，增量核对v2摘要、机制、实验设置及输出声明；这些段落未见实质变化。维持gate/kernel；作者报告输出不变，不等于任意batch位级一致保证。
- [Aurora](https://arxiv.org/abs/2410.17043)：本地PDF前3页初筛，再定位前提与实验设置；arXiv核验书目。理论最优性受建模假设限制；仿真平均1.07倍最优值不是一般近似保证，也无原生请求级证据。
- [Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin)：本地PDF前3页及相关推理/环境段，官方ATC书目。批次推理时间不是在线请求TTFT；训练通信优先级不能直接迁移为decode策略。
- [DRF](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types)：官方书目与正文摘要、动机、分配定义及性质段。理论性质依赖需求/分配模型；异构GPU效率、动态KV增长和不可切分gang不能直接套用。

当前唯一执行计划仍是已冻结的cap32 KV预算四项对照；上述相邻方法用于建立比较与回答设计追问。没有新增GPU实验、算法收益或正式论文主张。
