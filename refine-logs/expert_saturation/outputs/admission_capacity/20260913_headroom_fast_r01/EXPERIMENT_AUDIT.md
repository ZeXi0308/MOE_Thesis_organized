# Experiment Audit

2026-09-13。审计者 `/root/fast_headroom_integrity`，fresh GPT-5.6-Sol ultra，read-only；`review_independence=same-family`，`acceptance_status=provisional`。

**PASS，P0=0，P1=0。实验完整性通过，不是方法 GO。**

- 来源与实执行：当前源码、冻结包、配置、请求身份、四项归档及 readback 一致，128/128 请求 COMPLETE；独立复算匹配原始指标、逐步动作与配对差值。
- 在线输入：只用当前 request/KV 状态与声明输出上限；未用未来 route、完成信息或另一策略轨迹。fast 仍逐步校验 held 的精确块 ID、进度与 RUNNING，并检查 leader reservation。
- CPU 夹具：recorded-state fixture 已明确限定；不充当 GPU 状态或性能。GPU 结论来自四项原生 vLLM 独立执行。
- 路径边界：active step98 起 1342 步一致；激活前 17/9 步差异保留。报告不做全轨迹 matched-prestate 性能外推。
- 会计与范围：工作、块和墙时守恒；调度子区间不重复相加。原生含重算的 8 次调用还含 233 个新 decode 位置，没有把整个桶称为纯重算时间。
- 主张：吞吐 −2.86%～−3.25%、平均完成 +7.23%～+7.61%；每轮 29/32 请求自身 max-ITL 更大、30/32 完成更晚。限于单模型/单 GPU/同 cohort 两次描述性配对。

证据：[实际执行](execution/execution.json)、[冻结模块](execution/frozen/completion_headroom.py)、[完整请求复算](analysis/analysis.json)、[工作量分解](analysis/work_cost.json)、[CPU adapter](cpu_adapter_conformance.json)、[真实历史路径](analysis/historical_paths.json)。文件哈希见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)。

无需追加完整性修复。质量等价、Oracle、独立负载、生产尾延迟、方法 GO 均未验证；不支持机制族 NO-GO。下一共享四臂与正反序八项已登记，仍为 UNRUN。
