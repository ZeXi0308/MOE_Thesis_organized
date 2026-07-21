# Quality Isolation / Quality Debt

## 主张

在均匀降级基线上，用 **跨请求质量债务**（predictor-free）限制同一租户反复承担近似误差；文档级难度信号在机制间相关，但部署级 proxy 需零成本特征。

## 关键证据与边界

- Oracle 配额：小样本上 P95 KL 改善空间存在。
- Predictor-free debt：约 11–12% 量级改善，**未过预注册 20% 强门槛** → 弱正 / 第二贡献候选，非主杀也非主线。
- 旧「test 上选 best_proxy」claim 无效（leakage）。

## 脚本与产物（本目录）

- [`experiments/`](experiments/) · [`outputs/`](outputs/)
- `run_per_request_quality_isolation_p0.py` / `run_quality_debt_fairness_p0.py` / `run_quality_routing_synergy_check.py`
- 文档：本目录 [`原文/`](原文/)
