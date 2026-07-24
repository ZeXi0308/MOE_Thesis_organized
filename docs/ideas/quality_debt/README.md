# Quality Isolation / Quality Debt

## 主张

用跨请求 quality debt 拉平累积降级 harm。现有实现不预测未来，但它使用相对 BF16 reference 的 KL，没有 shadow 前向时并不可观测。

## 关键证据与边界

- Oracle 配额：小样本上 P95 KL 改善空间存在。
- 16 个固定 harm 值的合成流上点估计约 11–12%，**未过预注册 20%**，且合成 trial CI 不是文档总体不确定性。判决 **NO-GO**，不作第二贡献。
- 旧「test 上选 best_proxy」claim 无效（leakage）。

## 脚本与产物（本目录）

- [`experiments/`](experiments/) · [`outputs/`](outputs/)
- `run_per_request_quality_isolation_p0.py` / `run_quality_debt_fairness_p0.py` / `run_quality_routing_synergy_check.py`
- 文档：本目录 [`原文/`](原文/)
