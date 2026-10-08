# 保 KV 同动作实现优化：完整请求仍有代价

**结果：四项完成，128/128 请求。优化后仍消除抢占与重算，但相对本轮原生基线，吞吐下降 2.86%–3.25%，未满足主目标。** 当前为 `MEASUREMENT_ONLY / NATIVE_SERVING`：单 OLMoE BF16、vLLM0.26、RTX5090、固定池、同一 32 请求 cohort 的两次描述性配对。研究问题继续 `OPEN`。

## 实测同资源对照

实际 KV 16,089,350,144 bytes / 7,671 可用块，cap32、3072 输入/1024 输出、50ms 到达、budget1024。两臂使用相同 fast 观察设置，完整历史检查均开启；每项新引擎、新进程及原三项预热。无新实例、环境安装或模型下载。

| 执行顺序 | 完成 | 墙时 s | 请求/s | 平均完成延迟 s | 最大 ITL s | 抢占 / 重算位置 |
|---|---:|---:|---:|---:|---:|---:|
| repeat0-native | 32/32 | 22.892435 | 1.397842 | 20.537725 | 4.487516 | 2 / 7,685 |
| repeat0-headroom | 32/32 | 23.660591 | 1.352460 | 22.100533 | 1.386702 | 0 / 0 |
| repeat1-headroom | 32/32 | 23.588630 | 1.356586 | 22.030503 | 1.385728 | 0 / 0 |
| repeat1-native | 32/32 | 22.913837 | 1.396536 | 20.544998 | 4.473709 | 2 / 7,685 |

同轮 headroom − native：吞吐 **−3.25% / −2.86%**，墙时 **+3.36% / +2.94%**，平均完成 **+7.61% / +7.23%**。两轮各 **29/32 请求自身 max-ITL 更大、30/32 完成更晚**；每请求最长间隔中位数从 94/95ms 升至约 1.235s。最坏请求改善不能覆盖多数请求的代价。完整逐请求结果见 [analysis.json](analysis/analysis.json)。

## 本轮修改与一致性

旧 adapter 在每个 active step 对所有 running 请求提取完整块 ID 表。现在先从经过资格校验的单 full-attention owner list 读取长度，只在确定 held 集合后，对 held 请求保存精确 ID tuple，并在调度后检查同一 ID/进度。Leader 后置检查也只读块数。Leader 选择、余量公式、激活时机和 held 规则不变；原 checked 分支保留。

- [CPU 完整 adapter 回放](cpu_adapter_conformance.json)：两条旧轨迹分别 1,440 步，checked/fast 决策与实际动作全字段一致。回放的原生执行层使用记录状态，block IDs 为构造夹具，明确不证明 GPU 物理状态或性能。
- [真实 GPU 历史路径比较](analysis/historical_paths.json)：两次新旧 headroom 各 1,440 步决策和 scheduled/computed 区间一致，完整输出各 32/32 一致。从首次 active step98 起，1,342 步 before/after memory 均一致。
- 全轨迹 memory 比较仍为 `DIVERGED`：激活前分别有 17/9 个 step 的 waiting 数量与新入队请求记录不同，首次在 step12。不能把整个跨轮运行称为 matched-prestate 性能对照。物理 block IDs 与 route tensors 不在 raw 中。
- 当前 headroom 决策子区间为 0.221/0.219s；旧 checked 记录为 0.456/0.487s。旧值仅作历史参考，优化后的完整效果以本轮新 native 配对为准；不扣除观察时间来推算未执行的收益。

## 剩余成本来自哪里

互斥会计：`wall = scheduler inclusive + engine non-schedule + outside engine calls`；decision 已包含于 scheduler。见 [cost_breakdown.json](analysis/cost_breakdown.json)。

| headroom − native | repeat0 | repeat1 |
|---|---:|---:|
| 总墙时 | +0.768s | +0.675s |
| 调度区间 | +0.259s | +0.239s |
| 引擎除调度外区间 | +0.497s | +0.438s |
| 引擎调用外区间 | +0.013s | −0.002s |

约三分之二的额外墙时落在非调度引擎区间。按[实际执行工作](analysis/work_cost.json)再拆分：两臂新 prefill 都为 98 次调用；headroom 少了 8 次含重算调用，却多了 100 次纯新 decode 调用，总调用 1348→1440。纯 decode 的非调度时间增加 0.713/0.663s，含重算调用桶减少 0.228/0.228s。原生这 8 次调用还执行了 233 个新 decode 位置，不能把整个桶称为纯重算成本。

其中，**decode width=1 的调用从 129 增至 290**，该桶增加 0.724/0.720s，超过整个纯 decode 净增量，其余宽度抵消了一部分。该分解定位的是串行 host engine 区间，仍包含执行、采样、同步和主机处理；不能将其称为纯 GPU 时间或证明某项成本不可消除。

## 结论、边界与下一步

- 已关闭的最弱链路：在不改变本次在线动作的前提下，可减少整表读取；真实调度和输出路径保持一致。但降低观察成本后，完整吞吐仍负，继续只优化整表复制不足以解释剩余代价。
- 当前失败分类：**等待转移与小 batch 执行成本暴露**。数据证明单请求 decode 调用增加，尚未隔离 GPU kernel 时间与 worker/同步开销。不能把此次方案失败写成整个 KV 进度问题失败。
- 最强已测基线为本轮原生 full。跨策略 Oracle、独立文档/到达过程、质量及生产尾延迟未测；每个 native/headroom 配对 31/32 输出序列一致，不声称质量等价。所有失败、未完成、暂停和预热保留；本轮各项均完整完成。
- 下一共享实验：按用户提出的互为基线方式，比较 `native32 / safe29 / headroom-fast / rotate32-c20`，检验轮转能否在当前保 KV 实测基线之外改善完整吞吐与请求间隔分布。配置和状态见[四臂对照](../../../experiments/admission_capacity/RESEARCH_EXPERIMENTS.md#四臂共享对照待执行)。轮转推演不替代实测；四臂尚未执行。本轮提出过双请求完成保护的假说，但不另起并行实现或运行。

## 可复现证据

- [单轮独立完整性复核](EXPERIMENT_AUDIT.md)：fresh GPT-5.6-Sol ultra，same-family provisional，PASS / P0=0 / P1=0；不提升方法结论。
- [执行、配置、归档哈希与四项原始结果](execution/execution.json)，12 份预热及完整 raw 在 execution/gpu_results。
- [冻结协议](preparation/source/DECISIONS.md)与 [执行包](preparation/execution.tar.gz)，SHA256 `6ed591759959f8e354c1dae73ceab959f9b1ae45c54b62ae6d4d0b92e3e0b5c6`；[实际源码](execution/frozen/completion_headroom.py)。
- [完整请求分析器](../../../experiments/admission_capacity/analyze_completion_headroom.py)、[成本复算器](../../../experiments/admission_capacity/analyze_headroom_cost.py)、[工作量分解器](../../../experiments/admission_capacity/analyze_headroom_work_cost.py)、[历史轨迹比较](../../../experiments/admission_capacity/compare_headroom_paths.py)、[CPU adapter 验证](../../../experiments/admission_capacity/verify_headroom_fast.py)。

本轮回答：**实现复制成本可以削减，保 KV 的最坏暂停改善仍复现，但多数请求等待和吞吐代价仍在；该版本进入共享四臂对照，研究问题保持 OPEN。**
