# G 执行计划与 KV 容量：边界归档

**当前候选投入已关闭。** 已测compact在冻结low/high上容量充足；dense未运行，跨计划内存差与服务收益仍未知。本轮将结果作为适用边界材料归档，不再搜索桶或tile，不补跑GPU、不候锁，不将本域结论外推为执行计划问题没有研究价值。

核心交付：

- [可复算容量账表、论文适用边界与重新立题条件](capacity_boundary.md)
- [研究卡及关闭判决](DECISION.md)
- [完整服务对照状态与历史背景](service_comparison.md)
- [精确字节数、算式结果和来源SHA](evidence/capacity_ledger.json)

最小复现，从毕业设计目录运行：

```bash
python3 -B G_execution_plans/archive_capacity.py
```

仅Python标准库，全部输入已在G目录内。命令核验冻结输入、compact配置与容量语义源码快照，从原始启动事件重算账表；只重写 `capacity_boundary.md` 和 `evidence/capacity_ledger.json`。不需要模型权重、相邻研究目录、服务器或GPU，不获取资源锁。未测状态不会被替换为零或推断值。

`evidence/review_20261008/` 保留compact单次启动、四次dense锁忙原件、实际执行脚本、阶段事件及执行回执。`evidence/review_endpoint_config.json` 保留冻结引擎参数，`inputs/` 保留原请求，`native_sources/` 保留对应源码快照。原始记录未修改；旧静态限制字符串以记录状态和阶段事件解释，详见服务表。

[假设与竞争解释](hypotheses.md)、[候选成本](candidate_costs.md)、[历史实验协议](research_contract.md)、[近邻工作](related_work.md)及原探针保留供追溯。它们不构成后续执行安排。`capacity_envelope.py` 的跨计划诊断仍为 `UNDETERMINED`，因为dense未测；归档完成不代表研究假设得到验证。重新投入须有独立部署依据和计划差异改变服务决策的证据，不能仅凭更小显存或新的桶配置。
