# 有界实验完整性审阅

2026-09-14。Fresh reviewer `/root/pressure_result_audit`，gpt-5.6-sol / ultra。`review_independence=same-family`，`acceptance_status=provisional`。用户要求不额外建立 `.aris`，本记录与实验同包保存。以下为审阅回复；执行方采纳其范围限定，不追加第二轮审计。

## 审阅原文

同模型族、provisional 审计结论：**Overall WARN；未发现 P0/P1，也未发现影响主表或 M1/M2/M4 冻结证伪结论的数值、状态或会计错误。**

- **A 来源：PASS。** 这是 native runtime 性能测量，没有质量 GT。输入来自冻结 WikiText 文章及固定选择规则（`pkg/.../long/config.json:17-38`），runner 校验 workload/prompt 哈希后执行（`pkg/run_probe.py:40-52`）；指标直接来自实际请求时间线（`run_probe.py:158-179`），未用模型输出伪造 reference。
- **B 分母/会计：PASS。** 吞吐分母为实际 observation duration，完成吞吐只计 completed，请求时延保留全部到达请求（`pkg/metrics.py:62-155`）。调用按 recompute→prefill→pure_decode 互斥分类，并验证 returned token 与最终保留 token 守恒（`analyze_call_progress.py:26-62`）。混合重算调用产生新输出，没有被全算成“纯税”；报告也明确拒绝这种解释（`REPORT.md:40-48`）。
- **C 文件/数值/状态：PASS，附一项范围限制。** 我逐行读取 `analysis.json:1-1409`，并独立对照全部16格的 metrics/status/config/qualification/environment：16格、6个有效配对均无字段或重算差异；总计512 completed、524288 output tokens；14格 COMPLETE，2格 d0-rotate 保持 INCOMPLETE/INVALID_NO_ACTION，且不进入机制主表。d4/d6变慢请求数独立复得32/32、31/32、32/32、30/32。运行日志两格退出1及其余退出0也一致（`campaign-resume01.log:1-33`）。新增 `zero_action_check.json:1-14` 存在并报告两组调度顺序及32/32输出相同；因未提供其生成脚本，本轮未重新加载四份巨大 raw 独立复算该附加断言，但它不影响 d2/d4/d6 主表。
- **D 脚本输出：PASS。** pressure analyzer 实际导入并调用 call-progress 分析（`analyze_pressure_review.py:8,38`），输出16格及配对（`:70-95`）；`analysis.json` 每格均有 progress、raw hash、调用数和token守恒字段。指定两脚本内未见只定义未调用的度量路径。
- **E 范围/未来信息/基线：WARN。** 分析器只使用已发生的轨迹与完整请求结果，未见未来泄漏；但轮转策略内部实现不在本轮给定审计文件中，因此“动作本身完全无未来信息”仍为 **UNVERIFIED**。native 是同资源直接基线，但两个反向区组复用同一32篇输入，并非独立样本；d6还缺 headroom/most_output 强简单基线，报告已如实限定（`REPORT.md:5,19,65-73`）。
- **F 类型：WARN。** 合适分类是 **native in-process vLLM runtime performance measurement / MEASUREMENT_ONLY**。runner 自身冻结的 ceiling 是 `NATIVE_INPROCESS_CAPACITY_QUALIFICATION`（`run_probe.py:71-77`），所以 `REPORT.md:3,66` 的 `NATIVE_SERVING` 标签偏宽；不改变数值，但应避免被理解为外部服务栈/生产部署证据。

最多三项实质风险：同一workload的两区组不足以支持稳健性或显著性；高压点缺最强简单策略；证据层级标签需收窄。当前可支持的主张仅是：在该单模型、单卡、固定长输入、vLLM in-process运行域内，least_progress轮转相对native在d2/d4/d6两区组均降低最大ITL，同时高压下多数请求完成时间变慢。冻结检验的 d4/d6 M1未触发、d6 M2触发、d0 M4触发均与原值和实测算术一致。

## 提示与处理记录

首次提示仅列：两份分析脚本；REPORT、PRE_EXECUTION_ADDENDUM、analysis、comparison_checks、jit_window_check、run.sh、pkg/run_probe、pkg/metrics、冻结long/config；父PREDICTIONS；readback/campaign日志及16格raw/metrics/status/config/environment/qualification路径。要求A–F检查、文件行号、最多3个实质风险；只读、无GPU/SSH、不另开agent、不重复全量巨大raw审计。后续增加zero_action_check路径，并要求收敛、允许未查项保留UNVERIFIED。

执行方处理：主表与冻结检验数值保持；报告证据标签增加“原生vLLM进程内”和冻结runner ceiling，明确非外部服务部署。保留同一输入两区组、高压强基线缺失、策略内部未在本次重审和d0额外断言未被fresh复算的限制。未将WARN改写为PASS，不升级方法GO。
