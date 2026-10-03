# 等待越过约束：完成测量，停止当前规则扩展

2026-09-11。12正式和12预热已全部执行并回传。一次越过限制把每组mixed的总越过由SPT的6降为3、最大由3降为1，但没有满足两组都保留相对FCFS短请求TTFT收益的预定条件：短TTFT分别变化+1.32%和−7.13%。相对SPT，长完成p95改善1.77%/2.78%，短TTFT却恶化9.68%/7.80%。当前结论为MEASUREMENT_ONLY，不继续扫描次数阈值。

- [结果与完整解释](REPORT.md)：原值、全部比较、请求代价、证据范围与下一问题。
- [执行前冻结](DECISIONS.md)：动作、资源、输入、正反顺序与停止条件。
- [全表](analysis/TABLE.md)、[全部派生数据](analysis/summary.json)、[逐请求分解](analysis/REQUEST_ACCOUNTING.md)。
- [一次fresh核对](EXPERIMENT_AUDIT.md)：WARN；真实执行和会计通过，正向条件失败。
- raw在 `readback/results/{forward,reverse}/`；回传记录为 `TRANSFER-{forward,reverse}.json`。原始归档及远端原件均保留。

执行前下载、缓存切换和旧驱动中断的记录保留在 `model-cache-source.json`、`model-cache-verification.json`、`runtime-preflight.json` 和 `local-driver-attempt1-*`。最终驱动退出0；旧主机缺失的R1 reverse数据没有被本轮替代。

后续只推进一个不同动作：保持FCFS与总预算1024，测试原生单请求prefill限额。依据是实际观察到空闲请求名额与耗尽token预算同时出现；准备和新结果独立保存在 `../../independent_ideas_20260911/per_request_prefill_share_r01/`。本目录raw不再修改。
