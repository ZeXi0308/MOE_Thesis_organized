# Idea B：Verify, Don’t Predict（专家精度影子验证）

## 主张

用 **固定周期 low+high 影子验证 + 精度上调** 取代预测未来脆弱性，是待检验的假设；当前 H1 NO-GO，H2 实现失效。

## 关键证据与边界

- `SUPERSEDED`：H2 固定 period≈4 在 OLMoE 过线。脚本只在 all-INT4 KL 轨迹上掩码 high step，没有执行混合 KV 策略。
- AIMD 自适应周期、双轴联合 POC：**负结果**（附属，不扩展主线）。
- 下一步必须从同一 KV 状态双算，每个 policy 独立 in-loop 执行并记双算/cache/能耗成本；现有 CSV 不能修复该问题。

## 脚本与产物（本目录）

- [`experiments/`](experiments/) · [`outputs/`](outputs/)
- 主实验：`experiments/run_expert_precision_persistence_shadow_verify_p0.py`
- 负结果附属：`analyze_adaptive_shadow_verify_controller.py`, `analyze_dual_axis_joint_controller_poc.py`
