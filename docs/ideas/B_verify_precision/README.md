# Idea B：Verify, Don’t Predict（专家精度影子验证）

## 主张

短 horizon 上专家量化风险 **弱持续**（杀预测器），但可用 **固定周期影子验证 + 精度上调** 做反应式控制，而不预测未来脆弱性。

## 关键证据与边界

- H2：固定 period≈4 影子验证在 OLMoE 上 harm 降幅过线；LLM-jp 近门槛。
- AIMD 自适应周期、双轴联合 POC：**负结果**（附属，不扩展主线）。
- 边界：真实 INT4 打包内核 / wall-clock 仍可能受环境影响；离线质量分析 ≠ 已证系统吞吐。

## 脚本与产物（本目录）

- [`experiments/`](experiments/) · [`outputs/`](outputs/)
- 主实验：`experiments/run_expert_precision_persistence_shadow_verify_p0.py`
- 负结果附属：`analyze_adaptive_shadow_verify_controller.py`, `analyze_dual_axis_joint_controller_poc.py`
