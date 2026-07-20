# 实验结论文档（从 outputs 抽出）

原始 CSV/日志仍在 `experiments/idea_a_mac/outputs/<run>/`。  
本目录只放**可读结论**（移动或复制），按路线分子目录。

| 子目录 | 对应路线 |
|---|---|
| `receiver/` | Receiver-aware v2/v3 |
| `quality_isolation/` | Quality Isolation |
| `prefetch/` | 路由可预测性 / Expert Prefetch |
| `tokenrace/` | TokenRace-EP |
| `creditreduce/` | CreditReduce |
| `masscover/` | MassCover-EP |
| `pltb_additive/` | PLTB + additive-KL |
| `quotaep/` | QuotaEP-H |
| `graceful_qtree/` | Graceful / QTree |
| `energy/` | Energy-SLO |
| `misc/` | Idea A 早期汇总、文献审计、综合结论 |

完整 run 索引见：`experiments/idea_a_mac/outputs/README.md`
