# 容量计算的静态 cap29：完整可行，小幅吞吐改善与尾延迟代价

2026-09-08，分支 `agent/publish-current-moe-code`，HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`，继承未提交的准入实验实现。所有新数据保留本地，未push、未改写权威入口或旧raw。

**本轮问题已回答：从真实 KV 池推导的 cap29 可以无抢占地完成当前长上下文负载，相对 cap16 的完整吞吐提高3.75%/4.53%，但 TTFT 尾部和请求平均 TPOT 均变差。** 这是普通容量配置的可重复权衡，不能写成全面时延改善或专家感知方法收益。

## 继承事实与执行范围

执行依据为既有权威入口、准入实验 README、[上一轮自然容量结果](../20260906_native_memory_pressure_r01/EXECUTION_REPORT_20260908.md) 和本轮 [DECISIONS.md](DECISIONS.md)。上一轮已测到：cap16可完成长输入，cap32则在全部prefill完成后的decode增长中耗尽KV；该cap32被保护钩子终止，因此还没有默认抢占/重算策略的完整请求性能。

本轮最弱链路是：比cap16更大的、按声明最大长度预留的静态上限，能否进入完整请求收益。复用同32篇真实文章、3072-token输入、固定1024-token输出、每50ms一条的到达序列；它们不是128篇独立文档。保持 BF16 OLMoE、单RTX5090、vLLM0.26.0、engine32、context4096、token budget1024、显存预算0.90、FCFS、chunked prefill、同步in-process运行和关闭prefix cache。

四个新引擎依次执行 `baseline16 → safe → safe → baseline16`，每个都使用相同三次暖机。所有策略独立推进KV、batch、输出和完成时间；只有新接纳受cap限制，已有decode继续推进。原有首次抢占保护保持启用，任何保护终止都不能参与完整吞吐比较。

## 测量前的实际容量计算

四个引擎均通过了同一live资格检查：一个完整attention组，16层共享该组block table，无prefix共享、speculative/lookahead、context parallel或非零watermark；实际coordinator为 `KVCacheCoordinatorNoPrefixCache`。池包含7678块，其中一块是null block，可用7677块；每块16 tokens。

```text
每请求完整长度预留 = ceil((3072 + 1024) / 16) = 256 blocks
safe_cap = min(32, floor(7677 / 256)) = 29
满预留 = 29 × 256 = 7424 blocks；余253 blocks
```

这些值在暖机及主测量前写入每个cell的 `safe-cap-qualification.json`，没有根据观察到的峰值或性能提高cap。cap16也记录了相同推导。这个公式是当前布局和声明最大长度下的保守容量规则，不是实测最优cap或性能Oracle。

首次r01初始化发现了实现错误：资格检查误要求 `UnitaryKVCacheCoordinator`，而安装版本在关闭prefix cache时选择 `KVCacheCoordinatorNoPrefixCache`。它以UNRUN、退出2停止，0次warmup、0次请求测量；完整环境、资格记录及日志已回传到 [r01](../20260908_kv_safe_static_r01/analysis/report.md)。随后读取实际factory及单组分配路径，仅修类型识别和block_size读取位置，在本r02重新执行原计划，见 [ADDENDUM.md](ADDENDUM.md)。原尝试、源码和执行包未覆盖，公式及输入未变。

## 四项完整请求结果

全部4个测量episode、128次请求执行均完整结束；另有12个完整warmup，共264次16-token请求执行，单独保留。表中TPOT是每请求平均生成间隔的分布；p99由每cell32条请求计算，只作有限样本描述。

| 重复 / 策略 | 完整时长s | 吞吐req/s | TTFT p50 s | TTFT p99 s | 请求平均TPOT p50 ms | 请求平均TPOT p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| 0 / cap16 | 28.34621 | 1.12890 | 6.78208 | 13.60198 | 13.55194 | 13.68674 |
| 0 / cap29 | 27.32225 | 1.17121 | 0.36015 | 18.04667 | 18.64872 | 18.85659 |
| 1 / cap29 | 27.16423 | 1.17802 | 0.35620 | 17.90815 | 18.51731 | 18.73819 |
| 1 / cap16 | 28.39588 | 1.12692 | 6.80451 | 13.62870 | 13.56218 | 13.67471 |

两次成对比较的吞吐变化为 **+3.7477%/+4.5341%**，完整时长减少3.6124%/4.3375%。同时，TTFT p99增加4.445/4.279秒，请求平均TPOT p50增加5.097/4.955ms。不同指标方向并不一致，不能只用吞吐或TTFT中位数宣布策略整体更好。

参考SLO仍为TTFT5s且平均TPOT200ms，联合达标由16/32变为29/32；全部请求都满足这个较宽松的TPOT参考阈值，变化来自TTFT。在cap29下，最后三条请求的TTFT为17.81–18.06秒，而cap16对应为13.57–13.63秒。多数请求更早得到首token，但末尾请求承担更长等待，参考SLO计数不能取代这些连续时延结果。

请求分母始终为同一host时钟的episode终点减原点，包含排队、prefill、decode、采集及调度循环。局部计时不重复加算；平均TPOT不等于逐token ITL约束。完整精度、参考SLO goodput、所有请求及比较见 [analysis.json](analysis/analysis.json) 和 [可读分析](analysis/report.md)。

## 动作与容量确实改变了什么

cap16两次实际active/decode峰值均为16，等待峰值16，KV峰值4079/7677=53.13%；cap29两次实际active/decode峰值均为29，等待峰值9，KV峰值7357/7677=95.83%，最少仍余320块。四项均无分配失败、抢占保护或已有decode漏推进。

因此cap29并非只改了配置数值：实际并发和占用均提高，并能排空完成。实测峰值低于7424块保守上界，不授权把cap继续向上调；更早完成的请求可以释放块，这与最大长度预留的上界含义不同。

KV占用属于预分配KV池内部；expert存储属于总参数子集；Torch reserved包含allocated，均不可重复相加。本轮没有专家路由遥测或释放专家内存的动作。共享分析中的10.50GiB来自 `12GiB × (1−8/64)` 的单token结构估算（见 [原计算脚本](../../../experiments/admission_capacity/analyze_expert_residency_tax.py)）；没有测量batch专家并集、后续复用或allocator释放，不能当成本轮可回收容量。

## 证据与留存

4/4输入身份、prompt hash、到达与完整性核对通过；live公式、实际cap、原始请求指标重算通过；五份执行源码（含safe helper）、引擎参数及软件字段一致。所有暖机raw可读且完整。分析只执行了一次必要的资格、身份和会计核对，没有追加同配置GPU重复或阈值搜索。

四个进程均退出0。每项结束后先完整回传并核对归档SHA256，再开始下一项；四份无损归档合计56966915 bytes。原始请求/step/KV记录、配置、命令、环境、stdout、stderr及退出信息在 [gpu_results/](gpu_results/)，传输证据在 [execution-state-20260908.json](execution-state-20260908.json)。r01初始化失败也另行完整保留，远端原件未删除。

实际执行包为 [execution.tar.gz](execution.tar.gz)，SHA256 `de8aea2f8286babb1178086e6388c7827e34e62a50d4194c18f49b0047df0bc5`；改动见包内source patch。最后一次 [GPU读取](FINAL_GPU_OBSERVATION.json) 未见计算进程；临时凭据读取助手已删除，凭据未入库。进程检查只覆盖边界，不声明连续隔离。

## 当前判断与唯一下一步

本轮证明了 `容量可行 != 时延全面改善` 在这个具体运行域中的表现：保守容量上限避免了此前的KV边界，但增加并发仍需支付生成时延与队列尾部代价。这里只形成普通静态策略的测量结论，未得到MoE专属贡献。

**唯一下一实验是补齐默认原生基线：让cap32允许实际抢占/重算，直到32条请求完整完成，再与本公式cap29做同配置正反序对照。** 不能复用此前保护终止的0完成结果作为默认策略吞吐；要把真实等待、重算与完成时间全部计入host请求分母。

实现只需观测并调用原 `_preempt_request`，将原生抢占事件与保护终止分开。当前局部阶段分类会把已生成token的重算算成decode，且重算步可能没有新token输出，因此该对照应以完整请求计时为主，另记录抢占/重算事件；不能复用“每一步都产出receipt”或“已有decode必须每步推进”的非抢占资格结论。这里是补默认基线，不是新调度控制器。尚未执行该实验。

| 结束字段 | 本轮结论 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`：cap29可行，吞吐小幅增加，尾部TTFT与平均TPOT变差 |
| Evidence type | 单OLMoE、单RTX5090、原生vLLM同步in-process的完整请求测量 |
| What was measured | 两个静态上限、live KV公式、独立正反序、实际并发/等待、完整请求时延与吞吐 |
| What was not measured | 默认抢占/重算的完整代价、专家信号/回收增量、任务质量、第二模型、EP、生产SLO |
| Strongest baseline | 本轮cap16；cap29现为可行的普通容量基线，仍非全局最优静态点或完整prior-art复现 |
| Oracle/headroom | 无动态Oracle；仅测到相对cap16的小幅吞吐空间及明确时延代价 |
| Claim ceiling | 当前同长度、固定输出上限域的容量/时延权衡，没有Pareto或MoE方法GO |
| Failure category | r01是测量前类型资格检查错误；r02执行完整，结论受多指标权衡约束 |
| Resurrection condition | 默认原生/普通KV基线之后仍有可重复完整请求代价，且存在能改变它的具体专家动作 |
| One next smallest experiment | 原生cap32抢占重算至完整完成 vs公式cap29；保持输入/到达/预算，正反序，不再扫cap |

直接回答本轮问题：**安全静态cap确实提高了可行并发和少量完整吞吐，但它把更多代价转移到了生成时延和末尾请求，尚不能证明比默认原生策略更好。**
