# ConfidenceGuard-MoE v3 冻结实验设计

状态：**DESIGN FROZEN / SEALED UNOPENED / GPU NOT YET APPROVED**  
日期：2026-07-23

## 1. 起点与证据边界

TriageAudit v2 在 32-document calibration 上得到正的风险排序，但两模型均未通过精确 `{2,4,8}` 三分档 assignment stability，因此 v2 verdict 为 `NO_GO_UNSTABLE_TRIAGE_SIGNAL`，sealed 未上传、未读取。

v3 是只使用 calibration 形成的探索性重定义。calibration 上的改善只用于冻结机制，不能作为 confirmatory claim；唯一 confirmatory 结果来自随后一次性打开既有 64-document sealed split。

## 2. Scientific question

当请求级风险信号不足以支持精确三分档时，bootstrap 不确定性是否仍能稳定识别“可安全降低审计频率”的请求和“需要高频保护”的请求，并在完全相同 period multiset 下优于与风险无关的 hash 分配？

## 3. System mechanism

对每个模型在 32 篇 calibration 文档上固定 9 项 prefill 特征和 document-CVaR90 same-state discrepancy 标签：

1. 拟合一个 full-data ridge point model；以 32 个 calibration score 的中位数定义 full binary safe/risk assignment。
2. 做 2,000 次 document bootstrap；每次重拟合 ridge，并以该模型在完整 calibration feature set 上的 score 中位数作为 safe cut。
3. 对新请求只计算 9 项 prefill 特征；2,000 个低开销线性模型投票，得到 `p_safe`。
4. `p_safe >= 0.8 -> period 8`；`p_safe <= 0.2 -> period 2`；其余请求 abstain 到 `period 4`。
5. online audit/lockout 状态机、same-state diagnostic、INT4 expert proxy 和 common phase 均保持 v2 不变。
6. hash baseline 对 ConfidenceGuard 产生的 exact period multiset 做 document-hash 重排，只破坏 risk-document 对应关系，不改变总审计预算。

这不是降低 v2 gate：v2 的精确三分类保持 NO-GO；v3 把模型不确定性变成显式 fallback 系统状态。

## 4. Calibration-only reformulation gate

两模型均必须满足：

- frozen point score 的 document-bootstrap Spearman 95% LCB `> 0`；
- binary assignment probability 中位数 `>= 0.8`；
- assignment probability `>= 0.6` 的文档比例 `>= 0.8`。

该 gate 只决定新机制是否值得消耗 sealed，不构成论文证据。

## 5. Sealed arms 与变量

独立单位为 document；两模型使用相同 64 篇 sealed 文档、prompt 64、teacher-forced decode 32。

- `triage_2_4_8`：实际机制为 ConfidenceGuard `{8,4,2}`；
- `hash_budget_matched_2_4_8`：exact period-multiset hash 重排；
- `fixed_2`, `fixed_4`, `fixed_8`；
- `always_low`, `always_bf16`, `full_shadow`。

控制变量：模型 revision、文本、token budget、INT4 proxy、threshold、phase rule、lockout、reference logits、随机种子与统计代码全部冻结。

## 6. Confirmatory pass/fail

沿用 v2 的 paired-document 5,000 bootstrap + Holm 主检验，但主机制解释改为 selective/abstaining audit allocation：

- 必须先通过 8-arm、64-document、step replay、period multiset、forward/clone ledger 完整性；
- ConfidenceGuard 相对 matched hash 的 dangerous recall 不劣超过 5 pp；
- quality ratio UCB `<= 1.05`；
- total candidate forward reduction LCB `>= 0.10` 或在相同 exact period multiset 下证明更好的 tail protection；
- 两模型分别报告；跨模型结论不允许用一正一负平均掩盖。

若 ConfidenceGuard 与 matched hash 无显著差异，或任一模型出现明显 tail harm，则 Gate M 为 NO-GO；不进入 topology/scheduling 扩展。

## 7. 明确不声明

本实验是单 RTX 5090、teacher-forced、dequantized-BF16 W4A16 quality proxy 的机制验证。它不证明 native INT4 speedup、continuous batching、TPOT/P99、能耗、跨 GPU 通信或真实网络拓扑收益。

