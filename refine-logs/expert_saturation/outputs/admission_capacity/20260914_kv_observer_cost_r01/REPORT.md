# KV观察器低分配消融

Verdict: MEASUREMENT_ONLY；当前微优化无稳定完整请求收益，停止继续优化此实现。八格真实native运行全部COMPLETE，256请求完成；GPU已释放。不判死KV保存/恢复研究问题。

固定旧d6/OLMoE BF16/单5090/vLLM0.26/6656可用块/APCoff，offload开/关×原/direct观察器，各做反序重复。未启用任何profiler，GC默认。direct避免get_blocks临时tuple/generator/KVCacheBlocks，仅保留新counts list，观测频率和字段不改。

| direct相对original | mean完成Δ | wallΔ | 最长ITLΔ |
|---|---:|---:|---:|
| offload-off 正序 | −2.81% | −2.04% | −2.49% |
| offload-off 反序 | +2.94% | +3.74% | +3.57% |
| offload-on 正序 | +5.06% | +5.12% | +3.98% |
| offload-on 反序 | −2.77% | −2.62% | −3.24% |

两个offload状态都翻转，不能称稳定加速、显著效应或总体噪声底。on仍慢于off，但当前对照含native connector/layout/主机成本，不宜仅凭profile指认必要物理税。原/direct都没有改变off/on重算21426/2994、各6次原生抢占；offload-on实际load2415919104/store12884901888字节，两模式及重复的sizes多重集一致。没有用lookup轮询匹配量冒充重用字节。

## 等价性及首次差异

四配对逐步执行签名和32完整输出序列全部相同。完整memory逐步完全相同这一严格条件未通过，分析保留DIVERGED_REQUIRES_LOCALIZATION。首次原始比较还包含随机internal request ID，已用每轨迹一对一internal_to_source规范化；旧analysis.json保留，规范化结果见analysis_identity_aligned.json。

规范化后memory差异全部发生于最初69步之内，首差异为墙钟到达导致的waiting_count/已加入请求集合不同。四对所有共享请求的computed/prompt/output/preemptions/block_counts差异数均0；第69步起完整非时钟memory一致。因而没有观察到块数读取值不等价，但不能重写成全运行状态完全相同。memory_difference_localization.json保留全部差异步。

资源核验见resource_check.json及raw的逐步pool。全请求mean、wall、maxITL均包含当前观察器成本，未删除不利重复或尖峰。原始归档SHA256 fd9ccf40c52447b3fd34b29efb0a6b0f1b54f616787c4d9a764ee0ffd5ce9b04；未改旧实验原件。

## 研究影响和唯一下一步

此前cProfile约107ms观测链并不代表这些时间可通过删包装收回；当前简单消融未得到稳定改善，不能用它解释或扣除offload的完整请求退化。停止此观察器微优化，不再加profiler、换阈值或禁用GC以抢救它。新读取方式仅保留为探索代码，不替换共享冻结runner。

回到同一主问题：轮转减少长缺席却支付重复恢复，native默认prompt-only保存又有额外执行成本。唯一下一步是CPU核对现成native connector的保存完成、驱逐通知和恢复可见性契约，确定能否仅对将要驱逐的KV做有明确完成条件的保存；先验证动作合法性和底座接口，不另造pager或直接删除当前rotation/connector不兼容保护。

证据上限NATIVE_SERVING的有限旧固定长度cohort观察器消融；没有机制GO、独立workload收益、多卡、free-generation质量或轮转+connector验证。最强对照为同资源同offload底座的原观察器；Oracle/headroom未由本实验建立。失败类别为此微优化效果不稳定，非问题空间不足。仅在新的明确分配热点/成本变化时重开此实现。执行者定向核验，非fresh独立审计。
