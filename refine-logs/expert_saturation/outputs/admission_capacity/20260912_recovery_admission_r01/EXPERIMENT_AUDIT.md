# 四项恢复准入对照完整性复核

**PASS；P0 = 0，P1 = 0。** Fresh reviewer：`/root/recovery_integrity`，GPT-5.6-Sol / ultra，未继承对话；另由其子任务独立复算原始账本。这是 same-family / provisional 复核，不是跨模型或外部复现。

覆盖范围为本次四项实际执行、原始数据、指标、会计、因果定位和报告边界。没有增加 GPU 运行，没有修改原始证据。

- **执行与身份：** 四个独立进程均返回 0，各完成 32/32 请求，每请求 1024 输出 token；token ID 与时间对齐且单调。12 份预热均完成并保留。四份 readback raw 与对应执行归档中的 raw 逐字节 SHA256 一致。见 [execution.json](execution/execution.json)、[run_probe.py](execution/frozen/run_probe.py)、[run_campaign.py](execution/frozen/run_campaign.py)。
- **公平对照与独立状态：** GPU UUID、模型 revision/BF16、软件版本、vLLM 源码、16,089,350,144-byte KV 与 7,671 可用块一致；EngineArgs 仅 `scheduler_reserve_full_isl` 不同，实际 scheduler 属性符合各臂。各臂使用新引擎独立推进状态。
- **指标和会计：** 独立从 raw 重算的 TTFT、TPOT、ITL、完成延迟、吞吐、参考 SLO/goodput、抢占与工作量，均与存档指标一致。每臂首次执行工作为 `98,304 + 32,736 = 131,040`；full/chunk 重算为 7,685/123,927。68 次无新输出重复抢占丢弃 116,242 个位置，准确解释额外重算量。
- **原因定位：** 两臂实际调度区间到 step 806 含一致，step 807 首次分叉；chunk 在 step 810 需要新增 50 块，实际仅余 41 块。两轮复现。见 [recovery_cause.json](analysis/recovery_cause.json)。与旧低预算运行的 1,348 步比较由单独源码定位核查，本复核不覆盖该跨运行比较。
- **输出与结论边界：** 各对 31/32 条完整生成序列一致；同一请求在 zero-based index 854 首次不同。未声称质量等价、生产尾延迟、总体显著性、方法 GO 或整个问题 NO-GO。

发现并修正一个非 P1 的分析输入路径问题：重分析器现直接读取本次冻结输入，原路径输入与其逐字节一致。复算仅改变四个 provenance 路径，其他全部字段及表格一致；原分析保留，见 [INPUT_PROVENANCE_ADDENDUM.json](analysis/INPUT_PROVENANCE_ADDENDUM.json)。

**测量限制：** GPU 外部进程隔离证据是初始化前、测量前和测量后三点快照，未连续监测全程。时延包含采集成本；两次同 cohort 反序配对仅支持描述性结果。

**允许结论：** 在本次固定 OLMoE/vLLM/RTX 5090/KV 池/workload 中，直接关闭完整历史可容纳检查造成反复恢复失败，并在两轮降低完整请求结果。该反例只否定这一准入放松方案；保留 KV、完成余量保护及跨策略 Oracle 尚未验证。
