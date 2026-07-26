# MassCover-EP 历史 P0：有限 shadow 预算能否约束故障质量爆炸半径

> 自动判决：**KILL_OR_REFORMULATE_MASSCOVER**

## 证据边界

- 这是 route-level historical retrospective screen，不是新 sealed confirmatory experiment。
- C1、gate share 和 missing-mass 都是逻辑/语义代理；没有测 wire、恢复时延、TTFT、TPOT 或 P99。
- shadow expert 在正常路径上假定 inactive；这里只计 expert-copy 数量，不计真实 HBM、加载和路由表更新成本。
- `oracle_cvar` 在 historical test 上选副本，只表示上限，不能部署。

## 核心假设与门槛

在 10% layer-expert shadow budget 下，calibration-only CVaR greedy 必须在每个模型至少 75% 的冻结 primary placements 上，相对 frequency、gate-mass 和 random median 中的最佳者再降低至少 10% 的 historical-test CVaR95；同时 gate 对真实 contribution 的 pair-level Spearman 必须在两模型均不低于 0.50。

## Gate → contribution proxy 审计

| model | pairs | requests | pair Spearman | layer-expert mass Spearman |
|---|---:|---:|---:|---:|
| olmoe_e64k8 | 131072 | 8 | 0.717 | 0.784 |
| llmjp_e32k16 | 262144 | 8 | 0.586 | 0.592 |

## Primary placement 的正常通信与故障风险

| model | placement | C1 cross-domain fraction | no-shadow mean missing mass | no-shadow CVaR95 | no-shadow P99 |
|---|---|---:|---:|---:|---:|
| llmjp_e32k16 | calibration_coactivation_balanced | 0.500 | 0.500 | 0.772 | 0.802 |
| llmjp_e32k16 | calibration_frequency_lpt | 0.500 | 0.500 | 0.764 | 0.798 |
| llmjp_e32k16 | contiguous | 0.500 | 0.500 | 0.762 | 0.788 |
| llmjp_e32k16 | round_robin | 0.500 | 0.500 | 0.785 | 0.820 |
| olmoe_e64k8 | calibration_coactivation_balanced | 0.499 | 0.500 | 0.931 | 1.000 |
| olmoe_e64k8 | calibration_frequency_lpt | 0.500 | 0.500 | 0.876 | 0.916 |
| olmoe_e64k8 | contiguous | 0.501 | 0.500 | 0.868 | 0.914 |
| olmoe_e64k8 | round_robin | 0.500 | 0.500 | 0.858 | 0.903 |

## 10% shadow budget 生死比较

| model | placement | MassCover CVaR95 | best baseline | baseline CVaR95 | relative gain | gate |
|---|---|---:|---|---:|---:|---|
| llmjp_e32k16 | calibration_coactivation_balanced | 0.6566 | gate_mass | 0.6998 | 6.2% | FAIL |
| llmjp_e32k16 | calibration_frequency_lpt | 0.6567 | gate_mass | 0.7020 | 6.5% | FAIL |
| llmjp_e32k16 | contiguous | 0.6521 | gate_mass | 0.6890 | 5.4% | FAIL |
| llmjp_e32k16 | round_robin | 0.6548 | gate_mass | 0.6942 | 5.7% | FAIL |
| olmoe_e64k8 | calibration_coactivation_balanced | 0.7580 | gate_mass | 0.8067 | 6.0% | FAIL |
| olmoe_e64k8 | calibration_frequency_lpt | 0.7381 | gate_mass | 0.7778 | 5.1% | FAIL |
| olmoe_e64k8 | contiguous | 0.7332 | gate_mass | 0.7697 | 4.7% | FAIL |
| olmoe_e64k8 | round_robin | 0.7309 | gate_mass | 0.7828 | 6.6% | FAIL |

## 判决解释

- Proxy gate：PASS。
- Route-risk gate：FAIL；按模型为 `{'llmjp_e32k16': False, 'olmoe_e64k8': False}`。
- **最终：KILL_OR_REFORMULATE_MASSCOVER**。

即使本 P0 通过，下一步仍必须直接 mask 整个失败域的 expert outputs，在相同 shadow HBM 下比较 frequency/CRAFT-like、gate-mass、random、MassCover 与 test oracle 的 KL/PPL/任务质量；随后才轮到 2–4 GPU 的 failure detection、mutable routing 和恢复时延。若端到端质量排序不复现 route proxy，方向立即死亡。

## 输出

- `risk_results.csv`：每模型、placement、policy、budget 的完整指标。
- `placement_tradeoff.csv`：无 shadow 时的通信/故障风险。
- `proxy_fidelity.csv`：gate 与真实 contribution 的小样本相关性。
- `verdict.json`：冻结门槛与机器判决。
- `manifest.json`：输入哈希与实验元数据。
