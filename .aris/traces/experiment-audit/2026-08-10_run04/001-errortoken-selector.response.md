审计结论：`PASS_WITH_LIMITATIONS`（确定性重算通过；fresh reviewer 被要求中断，因此无独立同族语义验收，`acceptance_status=review_unavailable`）。

核心重算：

- calibration：29,803 calls、6,793 keys；生成的风险表逐项完全一致，最大误差 `0.0`。
- 0/1 rank join：32 targets、16 victims、每 victim 2 个候选；`topk_rank + 1` 后覆盖 `28/32=87.5%`，缺失 `target-03/14/23/29`。
- ErrorToken causal-first B=1：`6/16` hits。
- gate-weight baseline：`6/16`，与主策略重合 15/16。
- top-k-rank baseline：`6/16`，与主策略重合 14/16。
- 精确 \(2^{16}=65,536\) null：
  - `{4:4096, 5:16384, 6:24576, 7:16384, 8:4096}`
  - mean=`6.0`
  - `p_ge=P(H>=6)=0.6875`
- 因为主策略 `6 <= null mean 6`，`NO_RETROSPECTIVE_ENRICHMENT` 按冻结规则机械正确。

关键限制：

- outcome JSON 的内容确实在 `SELECTION_FROZEN.json` 写入后才解码；freeze 前只计算文件字节哈希。对应 [run_cpu_pilot.py](/Users/leandrozhao/Desktop/毕设论文资料/docs/ideas/errortoken/experiments/run_cpu_pilot.py:551)。
- 但阈值是 retrospective config，没有前瞻预注册证据；32 个候选本身也是 outcome-informed enriched targets。因此无直接 selector-field 泄漏，但存在上游 cohort enrichment，不能解释为总体频率或泛化能力。
- “causal-first”只在每个 victim 的两个已筛候选之间成立，不代表完整在线 decode 序列上的因果策略。
- 只有 6,528/32,234 个 calibration key-row 覆盖完整六档 M；其余“first mismatch”是首次观测到的 mismatch，不一定是真正全网格 onset。
- summary、selection、runner、config、两侧输入和 COMPLETE 封存哈希全部闭合；不过各输出 mtime 同秒，没有独立 syscall trace，实际先后主要由已封存源码控制流证明。
- action 代码只生成 `NOT_EXECUTED_PLAN_ONLY`，未调用 replay、GPU 或 runtime guard。

允许声称：在这个 retrospective、outcome-enriched、固定 16-pair CPU screen 中，ErrorToken selector 没有优于 matched null 或两个简单基线。

禁止声称：在线有效、提升延迟/质量、可部署 guard、自然样本 prevalence、跨 workload 泛化或因果运行时收益。

下一步：冻结同一 selector 后，在未参与阈值设计的自然 heterogeneous co-batch held-out targets 上做前瞻验证；本结果不应通过调阈值抢救。

```json
{
  "verdict": "PASS_WITH_LIMITATIONS",
  "evidence_status": "deterministically_recomputed",
  "review_independence": "fresh_reviewer_interrupted",
  "acceptance_status": "review_unavailable",
  "mechanical_decision": "NO_RETROSPECTIVE_ENRICHMENT",
  "coverage": "28/32",
  "primary_hits": "6/16",
  "null_mean": 6.0,
  "p_ge": 0.6875,
  "baselines": {"gate_weight": "6/16", "topk_rank": "6/16"},
  "action_status": "NOT_EXECUTED_PLAN_ONLY",
  "closure": "PASS"
}
```
