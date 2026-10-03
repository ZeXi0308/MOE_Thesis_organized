# Fresh novelty jury response

结论：14 个候选中，只有 C10 值得直接 `PROCEED`；真正的 Top 3 是 **C10 → C14 → C09**。C14 是最稳的 2–4 周审计型课题，C09 是较强论文命题但更依赖多 GPU 环境。其余最多作为共同 Gate 的分析模块，不能取代 `docs/current/README.md` 主线，也不能借新包装复活已停止机制。

评分顺序均为：新颖性 / 影响力 / 可行性。

## 关键近邻

- AMoE/AEP 已覆盖异步专家并行、token 队列、动态重批和依赖追踪，但没有给出完整请求 DAG 的 exact counterfactual Oracle。
- UltraEP 已做逐 microbatch、逐层、post-gating exact-load balance，但 serving 主要是 prefill，不等价于自然 continuous decode。
- METRO 指出 decode 不应只平衡 token 数；ExpertPlex 又显式使用 executed tile count，基本封死 C07 的独立方法贡献。
- Gimbal 使用在线 source-aware expert statistics，但统计量由自身调度产生，没有处理 active-set / scheduler endogeneity，这是 C10 的核心余量。
- Mixture-of-Experts Serving 已形式化动态 GPU 分配、重配置成本及静态近似算法，但每步需求作为观测后的外生向量处理。
- Activation Patterns 已覆盖大规模、逐层、跨 workload trace 与 prefill/decode correlation，压低 C12/C13。
- RepetitionCurse 是路由集中型 DoS，不是自然共租户 route-overlap timing channel；C08 尚有窄缺口。
- EEP 与 Tarragon 处理显式 failure/recovery，没有解决 silent expert slowdown 的 identifiability。
- 关键新增近邻：CRAFT、Director、DODOCO、Vidur、The Early Bird。DODOCO 压缩 C03/C06，Director 压缩 C11/C13，CRAFT 压缩 C11/C12/C14。

## 逐项评审

| ID | 新颖/影响/可行 | 决定 | 核心裁决 |
|---|---:|---|---|
| C01 | 6.5 / 8.5 / 5.5 | CAUTION | full-DAG replay 是基础设施；只有跨模型 action reversal 或显著局部误差才形成结果贡献，应与 C04 合并。 |
| C02 | 6.0 / 6.5 / 5.5 | CAUTION | 只有 ontology 可机械检查并有 soundness/completeness 或非平凡等价定理时才是方法，否则只是规范。 |
| C03 | 5.0 / 7.5 / 6.5 | CAUTION | DODOCO 已覆盖多种 imbalance/dispatch 关系；新 index 必须在 held-out 模型显著胜过简单基线。 |
| C04 | 6.5 / 8.0 / 7.5 | CAUTION | 2^3 析因本身标准；只有归因翻转 Gate/controller 结论才有结果新颖性，适合作为 C01/C10 组件。 |
| C05 | 2.5 / 6.5 / 9.0 | ABANDON | 测量卫生，不是独立研究；保留为全实验强制 protocol。 |
| C06 | 3.0 / 6.5 / 6.0 | ABANDON | 接近 DODOCO 风格 characterization，也容易变相复活 PhaseMap。 |
| C07 | 3.5 / 7.0 / 5.0 | ABANDON | 被 METRO、ExpertPlex、DODOCO 和 roofline 直接压缩。 |
| C08 | 7.5 / 8.5 / 3.5 | CAUTION | 若证明真实 route-mediated timing leakage 可新，但需现实共驻威胁模型与多租户 EP。 |
| C09 | 7.5 / 8.0 / 6.5 | CAUTION | 若有 incidence-matrix identifiability theorem/coverage boundary 才是方法新；否则是 route-aware sparse regression 的应用。 |
| C10 | 8.5 / 8.5 / 8.0 | PROCEED | 最近工作没有处理 scheduler-induced active-set endogeneity；兼具方法与结果新潜力。 |
| C11 | 4.5 / 7.5 / 5.0 | CAUTION | MoE Serving、CRAFT、Gimbal、Director 已覆盖更新、漂移、迁移税；half-life 可能只是新指标名。 |
| C12 | 2.5 / 6.0 / 5.0 | ABANDON | 逐层异质性和 per-layer replication 已是显式设计维度。 |
| C13 | 2.0 / 6.5 / 4.5 | ABANDON | Director 与 Activation Patterns 已直接做预测/迁移，不得变相复活 Prefetch。 |
| C14 | 5.5 / 8.5 / 8.0 | CAUTION | min-max robust optimization 不是方法新，但它是最强、最有否证价值的动态 placement baseline。 |

## 总排序

1. C10 — PROCEED
2. C09 — PROCEED WITH CAUTION
3. C14 — PROCEED WITH CAUTION
4. C04 — PROCEED WITH CAUTION
5. C01 — PROCEED WITH CAUTION
6. C08 — PROCEED WITH CAUTION
7. C03 — PROCEED WITH CAUTION
8. C02 — PROCEED WITH CAUTION
9. C11 — PROCEED WITH CAUTION
10. C05 — ABANDON as standalone
11. C07 — ABANDON
12. C06 — ABANDON
13. C12 — ABANDON
14. C13 — ABANDON

实际执行顺序建议是 **C10 → C14 → C09**：C09 学术新颖性更高，但 C14 更适合先做 2–4 周 fail-closed 审计。

## Top 3 fail-closed pilot

### C10

- 固定两模型、自然且 document-disjoint 请求流；冻结 arrival、request ID、route、output length 和 output hash。
- 重放 FCFS continuous batching、pressure-aware scheduler 和 schedule-invariant negative control。
- 用同一 detector 比较 wall-clock、request-index、token-index popularity：hotspot Jaccard、change-point false positive、触发迁移成本、full-DAG P99/SLO-goodput。
- Kill：两模型自然 workload 中 hotspot Jaccard 均 ≥0.90 且 false change point <5%；或错误迁移 full-path 影响 <5%。identity/ledger/denominator 不闭合直接 INVALID。
- 硬件：CPU exact replay + 1×RTX 5090 可做资格；正式 migration/TPOT/P99 需要真实多 GPU optimized EP。

### C14

- 两模型、至少三类自然任务，document-disjoint calibration/evaluation；所有方案用相同 replica、HBM、capacity。
- 比较 default static、per-domain static、pooled static、robust min-max static、含保守迁移成本的 dynamic offline Oracle。
- Kill：robust static 对 simple static worst-cell 改进 <5%，且不改变 dynamic net Oracle 是否 >10% 的结论；只在单模型或 synthetic 成立也停止。
- 若 robust static 捕获 ≥90% Oracle，形成有价值负结果并取消 dynamic controller；若 dynamic 相对 robust static 的 95% LCB 仍 ≥10%，C14 降为强制 baseline。
- 硬件：CPU + 5090 可做 surface pilot；正式动态结论需 full-DAG 与多 GPU。

### C09

- 两模型冻结自然 route incidence matrix；exact replay 注入单 expert 5/10/20% slowdown，以 5090 isolated kernel 校准。
- 比较 route-aware NNLS/sparse/change-point 与 GPU-wide baseline，并加 label permutation、random slowdown、correlated slowdown controls。
- Kill：20% slowdown 下跨模型 top-1 localization <0.8、AUC advantage <0.1、matrix 不可辨识，或 full-path P99 变化 <5%。
- 无 ground-truth expert mapping 的相关 slowdown 不能作为定位证据。
- 硬件：CPU+5090 做 identifiability pilot；正式 reliability claim 需多 GPU EP fault injection。

## 边界

- 不调整当前共同 Gate，不把任何候选写成已验证机制。
- C05 是测量规则；C01/C04 是 C10 的评估基础设施。
- C03/C06/C07/C11/C12/C13 不应形成独立主线。
- C08 无真实共租户威胁模型和正式 EP 时不进前三。
- secondary fresh-agent 在收口时中断，未形成完整书面 verdict，因此不可写成 fully independent acceptance。

`review_independence=same-family`  
`acceptance_status=provisional`

