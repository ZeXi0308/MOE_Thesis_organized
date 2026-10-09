# G 容量账表与适用边界（封存）

**决定：关闭当前候选投入，作为边界材料归档。** 本页由 `archive_capacity.py` 从G目录现有原始记录和冻结输入生成；不执行实验。dense保持未测，关闭投入不等于一般问题被否定。

| 账目 | 数值 | 证据类型 | 解释与边界 |
|---|---:|---|---|
| KV token / block | 128 KiB / 2 MiB | 结构推导 | BF16 TP1；16 tokens/block；按模型配置复算 |
| compact 总 / 可用 KV | 36828 / 36827 blocks | 启动实测 | 总数扣除1个null block；可用容量不是服务实测占用 |
| compact 实际 KV storage / 可用字节等价 | 71.929688 / 71.927734 GiB | 实测 / 算术换算 | live unique storage / 可用blocks×2 MiB |
| low（12请求）最大占用包络 | 2107 blocks / 4.115234 GiB | 条件上界，非实测峰值 | Σ ceil((prompt tokens + max output tokens)/16) |
| compact 相对 low 包络余量 | 34720 blocks / 67.812500 GiB | 实测容量减上界 | 已测配置在声明语义下容量不生效 |
| high（160请求）最大占用包络 | 34430 blocks / 67.246094 GiB | 条件上界，非实测峰值 | Σ ceil((prompt tokens + max output tokens)/16) |
| compact 相对 high 包络余量 | 2397 blocks / 4.681641 GiB | 实测容量减上界 | 已测配置在声明语义下容量不生效 |
| 使 high 充足证书失效的最小可用块损失 | 2398 blocks / 4.683594 GiB | 离散算术界 | 降至34429才低于34430；不是graph驻留阈值或实测瓶颈 |
| compact 实际图数 | 19 | 启动实测 | 10 PIECEWISE + 9 FULL；非请求桶数 |
| graph 原生启动估计 | 132 MiB | 运行时估计 | 用于KV分配前的预留；不是最终graph独立驻留 |
| 正式 capture 原生差额 | 76 MiB | 阶段实测 | 原生设备空闲内存差；含阶段影响，不能独立归因 |
| 共享 workspace | 128 MiB | 持有storage实测 | 共享池；不乘层数或图数，不保证可返还KV |
| PyTorch graph 池 reserved | 28 MiB | allocator快照 | 非默认segment pool；不含全部driver图元数据 |
| 显式清cache释放 | 106 MiB | 阶段实测 | 释放后已分配KV不扩容 |
| 清cache前 / 后 KV blocks | 36828 / 36828 | 启动实测 | 均含null block；不是在线KV扩容 |
| 整设备占用减KV storage | 14.037292 GiB | 设备计数器与storage之差 | 含模型、context、driver、allocator、workspace和graph，非graph独占 |
| Torch allocated / reserved | 85.021563 / 85.222656 GiB | allocator快照 | 与其他分项重叠，不可相加为集合成本 |
| graph独立驻留 / 完整瞬时峰值 | NA / NA | 未单独测得 | capture差和reserved均不能替代精确归因或完整峰值 |
| 估计 / capture / 引擎启动时间 | 2.291 / 3.316 / 35.932 s | 一次启动观测 | 含观测开销、缓存冷热影响；非执行速度比较，无重复噪声估计 |
| dense 容量 / graph成本 / 服务效果 | NA / NA / NA | 4次尝试均锁忙，未初始化CUDA | 不是负性能结果，不能继承compact证书 |
| native集合 / 其他tile或backend | NA | 未测 | 不假定桶集合包含关系意味着内存单调，不声称存在实测速度—容量取舍 |

单位均为二进制MiB/GiB；精确字节数和输入SHA见 [capacity_ledger.json](evidence/capacity_ledger.json)。compact原件为 [compact.json](evidence/review_20261008/compact.json)，其中阶段事件保留固定测量顺序。原生流程为估计→KV分配→正式capture→清cache；0.9是原生显存利用策略，不是精确90%整设备硬上限。以上估计、capture差、workspace、graph池和整设备非KV数值互有重叠，不相加。没有第二个已测G配置，无法估计计划之间的驻留/容量差或排序。

**为什么当前证据不足以检验graph/KV取舍。** 已测compact可用容量高于整个high有限批次的最大需求，甚至把所有请求同时推到声明输出上限仍有4.681641 GiB余量。执行速度改变批次或完成时间也不能越过这个包络；因此进一步释放少量graph内存无法扩大这批请求的可容纳集合。至少损失2398个可用块才会失去这个充分保证，但失去保证仍不代表实际达到瓶颈。这个“块损失”等价量不能换写为新增多少graph字节：估计预留、取整、共享池和实际分配顺序会影响K(P)。132 MiB估计或76 MiB capture差均不是“可优化的两计划差值”。dense未运行，所以整个冻结负载对所有计划是否容量充足仍未知；当前数据不能完成设想的跨计划取舍检验。G服务吞吐、尾延迟和正确性比较均未运行。

**可进入论文的适用边界说明。** 在单张RTX PRO 6000、BF16 OLMoE、TP1及既定vLLM配置下，compact计划一次启动提供36827个可用KV块，每块2 MiB。对于冻结的12/160请求有限负载，在单输出、full attention、无prefix/speculation/connector、同步调度、零lookahead/watermark且无其他请求共池的条件下，由逐请求最大长度得到的容量包络分别为2107与34430块。因此，该已测计划在这两批请求上不会因KV容量不足触发分配或准入约束；此时计划驻留的少量下降本身不能带来容量收益，执行速度与启动开销仍可能有价值。该结论是启动测量与分配语义共同支持的条件边界，不是完整服务收益、实测峰值或所有执行计划的结论。dense及其他配置未测，尚未建立graph内存差异改变服务能力的证据；也不能据此否定更接近容量边界的部署中的联合选择价值。

**什么部署变化可能重新使问题重要。** 重新立题需先声明有业务依据的改变，例如持续到达并实际保留更多并发上下文、更长的输入/输出或常驻会话契约，或实际迁到更小显存设备/具有明确资源配额的部署；本次不构造或执行这些负载。新设备（包括5090）必须重新核对模型、后端与启动峰值可行性，不能按总显存大小推断收益。决定性条件是合法计划的实测K(P)差异足以跨越真实服务决策所需容量：在匹配前态的单事件诊断中，可检验 K_small < R_required ≤ K_large（R含当前持块和此次准入/执行所需块）；完整策略则保持外部负载与预算可比，允许内部轨迹改变。若所有计划仍覆盖需求，或容量虽紧张但计划差异不足以改变准入、抢占或可承载上下文，联合KV项仍无增量价值。有限trace包络是离线诊断，不向在线方法提供未来信息；未经论证不移用于持续到达或不同KV语义。

历史D运行仅作背景：不同GPU身份、不同调度代码，既有high峰值32031/可用36764块，graph capture差约0.17 GiB为舍入日志值；不能与本次compact拼成受控的计划比较。完整历史口径与未运行服务表保留在 [service_comparison.md](service_comparison.md)。所有失败启动原件保持不变。

最小复现（从毕业设计目录运行，仅Python标准库，全部输入位于G目录）：

```bash
python3 -B G_execution_plans/archive_capacity.py
```

命令只重写本页与 `evidence/capacity_ledger.json`，核验冻结输入、源码快照和实际配置，复算账目并保留dense未知状态；不导入torch/vLLM，不连接服务器，不获取GPU锁。
