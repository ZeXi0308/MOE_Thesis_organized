## 最终审计结论

**Overall verdict：PASS（同模型家族 fresh review，结论为 provisional）**

未发现明确 P0/P1。实验可以继续按当前 `WEAKEN_ACTION_SPACE` 结论封存；这里的 PASS 指“负结论完整可信”，不代表 CriticalSplit 获得正向支持。

评估类型：`simulation_only`，不是 real ground truth，也不是科学/论文结果。

### A. Oracle / reference provenance — PASS

Oracle 是冻结模拟器内的 future-known 穷举最优解，不是伪造 GT。WHOLE 与扩展 action space 使用同一物理状态机、服务曲线和字典序目标；扩展空间严格包含 WHOLE，因此比较不构成“模型拿自己输出当标签”的循环自评。[Oracle 实现](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/criticalsplit_full_dag_pilot.py:439)明确执行 token replay 和多指标闭合，[协议](/Users/leandrozhao/Desktop/毕设论文资料/artifacts/criticalsplit_pilot/20260810_173200/pilot_protocol.json:11)也明确标为 future-known、simulation-only。

### B. Normalization / raw metrics — PASS

Capture 使用 `(immediate - candidate) / (immediate - expanded_oracle)`，是固定 Oracle-headroom 归一化，不是预测结果自归一化；原始 flow、tardiness、miss、launch/service 均保留。[归一化与机械判定](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/bcrd/experiments/criticalsplit_full_dag_pilot.py:558)

本次 `eligible_cells=0`，所以正式中位数为 `null`，没有用非 eligible 单元中的 `capture=1` 冒充支持证据。

### C. Result integrity / completeness — PASS

直接验证结果：

- 28/28 单测通过：FrontierCredit 13，CriticalSplit 15。
- `verify_complete`、payload validation 均通过。
- fresh rerun 经 JSON round-trip 后与 sealed `pilot_results.json` 完全相等。
- 8/8 单元均为 `actual_split_flow == whole_flow`；proper/critical/bulk launch 全为 0；deadline miss delta 全为 0。
- 状态数为 `1786, 1786, 15741, 15741, 75390, 75390, 37737, 37737`，均低于冻结上限 500000。
- manifest、COMPLETE、SOURCE_PRE/POST 和源码哈希闭合；COMPLETE 的 authority 仅为 `artifact_completeness_only`。[封口边界](/Users/leandrozhao/Desktop/毕设论文资料/artifacts/criticalsplit_pilot/20260810_173200/COMPLETE.json:2)

### D. Dead / phantom mechanism — PASS

Critical/Bulk 不是完全不可达的死代码：测试能构造 proper split、验证未揭示 route 不可见、subset 保留剩余 ready nodes，并验证 sham 可改变 action availability。只是 exact optimal trace 在冻结八格中一次也没有选择 split——这是实质性负结果，不是指标未接线。

Action token 固定精确 node IDs，陈旧或篡改 subset 会被拒绝；decision 使用的 eligible、capture、identity gap、sham applicability、deadline delta 都有实际计算和输出路径。

### E. Scope boundary — PASS

证据严格限定为八个冻结、确定性 CPU synthetic cells。当前文件反复排除了 natural prevalence、在线策略、GPU serving、EP/NCCL/RDMA、生产收益及论文结果。[冻结计划范围](/Users/leandrozhao/Desktop/毕设论文资料/refine-logs/EXPERIMENT_PLAN_20260810_164938.md:11)

### F. Evaluation classification — PASS

正确分类为 `simulation_only`；`scientific_result_eligible=false`、`paper_result=false`。Oracle 只可视为模型内 exact reference，不能称真实 GT。

### P0/P1

**无 P0/P1。**

### Claim impact

- **C1：不支持，并被本轮证据削弱。** 冻结八格中 CriticalSplit 没有产生任何相对 WHOLE 的 flow headroom，也没有一次进入最优 trace。
- **C2：未建立。** `sham_applicable_eligible_cells=0`，因此不能据此声称 identity 有效，也不能外推声称 identity 普遍无效。
- **唯一成立的结论：bounded WEAKEN。** 即“在冻结八格、当前服务曲线与 exact simulator action space 内，未发现 Critical/Bulk proper-subset 的增量价值”。不得扩展为对自然负载、在线策略或 GPU 系统的一般否定。
