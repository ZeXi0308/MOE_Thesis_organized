# 四臂实验完整性核对

**PASS；P0=0，P1=0。** GPT-5.6-Sol ultra，fresh same-family/provisional；2026-09-13。

这是完整性通过，不是统计显著、吞吐非劣或方法GO。Reviewer只读，未运行GPU、未改文件、未创建`.aris`；按audit stop rule本轮结束。

## 已完成范围

- **identity_and_inputs — PASS**：256 formal requests across 8 complete cells match 32 frozen inputs, arrivals, prompt hashes and 1024 output tokens; source manifest and all archive hashes verified.
- **online_visibility — PASS**：Only current running/waiting, KV/free blocks, declared output limit and own observed history; no future route/completion or other arm trace.
- **independent_state — PASS**：Eight unique Python processes/engines/output directories; execution intervals do not overlap; forward/reverse order; pre-init process checks retained.
- **accounting — PASS**：Scheduled work partitions into new prefill/new decode/recompute; host wall partitions into scheduler/engine remainder/outside; used+free=7671. Held work retained and charged.
- **baselines_and_claim_scope — PASS**：All four arms same model/GPU/runtime/engine and KV; safe29 only changes admission. Report matches actual descriptive measurements and disclaims significance, noninferiority and method GO.

## 关键证据

- [冻结执行顺序](execution/frozen/run_campaign.py:12)：每格新进程/引擎。
- [轮转在线状态](execution/frozen/rotation_native.py:125)与[selector](execution/frozen/absence_rotation.py:114)：当前状态驱动。
- [互斥工作分类](execution/frozen/native_capture.py:64)、[主分析](analysis/analysis.json)、[成本守恒](analysis/cost_breakdown.json)、[工作桶](analysis/work_cost.json)、[恢复账本](analysis/rotation_accounting.json)。
- [执行记录](execution/execution.json:8)：8/8归档哈希、32/32完成、退出0。
- [最终报告](REPORT.md:3)：表格、配对差值及恢复边界均与独立复算一致。

## 允许的结论

- Eight complete NATIVE_SERVING_INPROCESS_HOST_CAPTURE cells in one fixed 32-request regime.
- Actual within-block throughput, completion, max-ITL, preemption and mutually exclusive host/work costs.
- Actual rotation events, recovery to new tokens and held-state accounting.
- Both repeats improve headroom throughput/mean completion/max-ITL; native throughput delta changes sign and mean completion worsens.
- Output identity as an execution diagnostic only.

## 未覆盖及禁止外推

- Statistical significance, throughput noninferiority, no throughput cost, method GO.
- Quality equivalence or output identity as a quality proof.
- Cross-workload/model/hardware/production/EP generalization.
- Host-time buckets as pure GPU kernel causal cost.
- Recovery protection necessity or attribution: held=0.
- Exact Oracle, per-request dominance, global strongest baseline, family GO/NO-GO.
- Independent complete worker block-table proof; runtime adapter checks exact IDs but independent trace covers scheduler snapshots/conservation.

最终报告SHA256：`02fd31b33e88ba2dea9b443a0eaba5cb053e8448236b0b10b740dab0df44c22d`。全部关键输入hash及review元数据见[JSON](EXPERIMENT_AUDIT.json)。

本轮没有新增修复项；唯一下一步仍是固定参数后的独立请求/到达episode与同负载A/A重复。
