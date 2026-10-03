# 专家工作集与 KV 预算：局部文献核对

2026-09-08；只核对 3 篇原文及 WiSP 官方仓库。本地入口已引用 ELDR；`papers/`、`literature/` 未找到这三篇同名 PDF。以下均按 arXiv 预印本处理，未核实正式录用 venue；WiSP 正文明确标注 Under Review。本页不解读尚未完成的本轮 GPU 结果。

| 工作 | Signal / prediction | Action / objective | Regime / guarantee 与边界 |
|---|---|---|---|
| Zhang 等，**WiSP v2 / MV-WSA**，2026-08-30，[§3、§5.3、附录 B](https://arxiv.org/html/2606.21868v2) | 实际路由集、LRU reuse-distance miss curve、KV 用量与准入下限；估计每字节边际时延价值。 | expert-map 将专家映射到 GPU scratch，缺失专家从 host 装入；配置或调整 expert/KV 分配。实测在线动作在 drained barrier 回收闲置 KV、扩大专家池。 | 模型不能完整常驻、低并发、24 GiB RTX3090；作者报告同预算在线完成时间改善最高 1.19×。要求使用前权重字节到位、先释放再增长、不越预算；一致性受底层引擎可复现性约束。不能把其 idle-KV→expert 结果当作当前 expert→KV 已验证。 |
| Choi 等，**ELDR v2**，2026-07-02，[§4–6](https://arxiv.org/html/2607.00466v2) | prefill 离散 expert counts，经 IDF/选层形成 signature，预测 decode 专家重叠；即时 decoder 在途请求数。 | 离线均衡聚类；handoff 时在相似度带内选最轻 decoder，改变共批请求；目标为降低 expert union 与 TPOT。 | PD 分离、最高 40 张 MI300X；作者报告较最强负载均衡基线 median TPOT 降 5.9%–13.9%。signature 与 KV block 同生命周期；模型路由/权重不改。没有专家显存回收动作，也没有零抢占或 SLO 硬保证。 |
| Liu 等，**FluxMoE v2**，2026-04-30，[§3.1、§4.3、§5–6](https://arxiv.org/html/2604.02715v2) | KV 用量、已 profile 的 compute 时间与 loading 时间比；不依赖 request expert signature。 | 稳定虚地址上的两层流式加载/释放；调整 GPU 压缩存储与 CPU offload 比例，为 KV 腾空间并控制 I/O 暴露；目标为 decode 吞吐。 | 4×L40、batch32–256、长上下文；每层执行前装齐该层全部专家，非 top-k 缺失分页。事件约束保护读取/复用，压缩无损；作者报告最高 3×吞吐，未用 TTFT/TPOT 作主指标。原型无未压缩永久常驻分量。 |

**代码可用性。** [WiSP 官方 README](https://github.com/nokia-applied-research/WiSP#what-this-release-is-and-isnt)明确：当前 release 提供 pager、iso-VRAM 与 byte-identity 复现，尚未提供在线 dual-resize controller；测试版本为 vLLM 0.11.2。ELDR/FluxMoE 本轮未核实可运行官方代码，不能据论文描述声称已可移植到本仓库 vLLM 0.26。

**本轮应如何改变判断。** 如果固定 cap32 的普通显存预算调整已消除秒级暂停，那么该冻结运行域的直接解释就是 KV 配额不足；尚缺“专家结构使普通 token/KV/queue 策略留下可利用 residual”的证据。MoE 模型上发生抢占，本身不构成 MoE 专属调度问题。需要另证相同总预算下实际专家工作集允许可执行的回收或批次动作，并计入搬运、重算、等待及请求尾部代价；单纯联合分配或专家重叠信号已有上述先例。

**最便宜的一个结构诊断。** 若继续检查专家回收空间，只记录同一自然 episode 的每 step/layer **实际 dispatched top-k union**、跨 step 使用间隔及累计工作集，并与真实 KV free/used blocks 和抢占时刻对齐；prefill、正常 decode、重算分别标注。由历史前缀形成容量/重用诊断，未来 next-use 只作离线描述，不能作为在线可知标签。若高 KV 压力时 union 或短时工作集已接近全专家池，则停止当前“稀疏性留下闲置专家空间”配方。若确有空隙，也只得到 STRUCTURAL 机会；仍需真实 allocator 释放和传输成本才能谈收益。该检查不判死 FluxMoE 一类按层流式动作。

`12 GiB × (1−8/64)=10.50 GiB` 只是每 token 未选中专家的算术：本地 [原分析脚本](../../../experiments/admission_capacity/analyze_expert_residency_tax.py)直接用 `experts_per_token/experts_total` 推算，未测 batched union、再使用时间或实际可释放 allocation。它既不是可回收 HBM，也不是把这些字节变成 KV 后的反事实容量证据。本轮唯一在执行的实验仍是普通预算对照，上述结构诊断尚未运行。
