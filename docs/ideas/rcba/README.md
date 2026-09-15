# Route-Conditioned Barrier Amplification Boundary

> Candidate status：`PRIMARY_NEXT_CANDIDATE / SAME_FAMILY_PROVISIONAL / UNVALIDATED`  
> Preparation verdict：`BLOCKED_PROTOCOL_AMBIGUITY`  
> Formal Oracle Gate：`NOT_RUN`  
> Runtime mechanism：`NOT_DESIGNED`

RCBA 只验证一个 Oracle-first 问题：自然 continuous-decode workload 中，MoE route identity 与 expert-tail 是否会经真实 runtime barrier 放大为 full-request critical-path latency；在 arrivals、work、resource capacity 和真实数据依赖完全相同时，只删除预先证明可移除的 runtime barrier edge，是否仍保留 material charged full-request headroom。

本轮在实现 evaluator 前按协议 fail closed。两个模型和 frozen input source 已明确，主指标 `J/H/A` 也可由任务合同锁定；但现有工作区没有唯一的 common natural regime、可移除 barrier whitelist、完整 edge-by-edge dependency 分类或 full-DAG resource-capacity model。自行补写这些字段会直接改变 10% Gate verdict，因此没有创建 evaluator、schema、fixture 或测试结果。

此外，两个模型均没有完成的 identity-complete canonical runtime trace，也没有覆盖 full DAG operator/layer/expert 的可连接实测 service surface。这些是协议歧义解除后的独立输入阻塞，不能用 StableBatch ledger、单 expert curve、aggregate MoE block timing、JoinStream FP16 microbenchmark 或 synthetic duration 替代。

- [Oracle Gate protocol](ORACLE_GATE_PROTOCOL.md)
- [Preparation summary](../../../artifacts/rcba_oracle_gate/preparation/20260810_230514/PREPARATION_SUMMARY.md)
- [Preparation tracker](../../../refine-logs/EXPERIMENT_TRACKER_RCBA_ORACLE_GATE_PREPARATION.md)

JoinStream 保持 `FROZEN / WEAKENED / NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION`；CriticalSplit 保持 `WEAKEN_ACTION_SPACE / eligible cells = 0`。RCBA 当前仍不是已成立的新 Idea 或机制。
