# Idea A 论文证据总索引

> **题目**：Profile-Guided Receiver-Aware Rank-LUT Mixed-Precision Combine for MoE Serving
> **模型**：OLMoE-1B-7B-0924 (主) / Mixtral-TinyMistral (交叉验证) / LLM-jp top-16 (stress test)
> **平台**：Mac M5 Pro / 48GB，CPU-only
> **整理日期**：2026-06-24

---

## 证据链总览

```
Proposal (C1/C2 claims)
  ↓
前置实验设计 → 生死实验 (C1 长尾 + rank-aware 优势)
  ↓ ✅ 通过
主实验 (3 模型 + receiver_group + serving simulation)
  ↓
补充实验 (layer sensitivity + drift attribution C2)
  ↓ ✅ 全部 claim 验证
LUT optimizer (MILP + δ 标定 + 端到端评估 + additivity check)
  ↓ ✅ 方法有效，发现 cascading 94x
论文写作
```

---

## 目录结构

```
thesis_evidence/
├── INDEX.md                          ← 本文件
├── EVIDENCE_SUMMARY.md              ← 证据汇总（一句话证明 + 关键数字）
├── supplement_experiments_report.md ← 补充实验综合报告
├── lut_final_report.md              ← LUT optimizer 综合报告
│
├── 00_proposal/
│   ├── PDF论文的选择.md              ← 完整 proposal（含 C1/C2 claim、MILP 公式、评估计划）
│   ├── IdeaA_前置实验设计_MacM5Pro.md  ← 前置实验设计（生死实验标准）
│   └── IdeaA_前置实验_TaskList.md     ← 任务清单（含完成状态）
│
├── 01_main_experiment/
│   ├── IdeaA_主实验报告_论文版.md      ← 主实验报告（论文版，最完整）
│   ├── main_experiment_report.md     ← 主实验原始报告
│   ├── olmoe_approx_results.csv      ← OLMoE 近似策略结果
│   ├── olmoe_serving_sim.csv        ← OLMoE serving simulation
│   ├── olmoe_receiver_rank_share.csv ← OLMoE receiver group 频次表（LUT optimizer 输入）
│   ├── mixtral_approx_results.csv    ← Mixtral 交叉验证
│   ├── llmjp_top16_approx_results.csv← top-16 stress test
│   └── figures/                      ← 主实验图表 (rank_sweep_kl, serving_latency 等)
│
├── 02_layer_sensitivity/
│   ├── layer_sensitivity.csv        ← 逐层 KL/MSE（16 层）
│   └── layer_sensitivity_report.md  ← 报告（KL ratio 5.74x → layer 维度成立）
│
├── 03_drift_attribution/
│   ├── drift_per_sample.csv          ← 逐样本 drift 分解（96 rows）
│   ├── drift_summary.csv            ← 按策略聚合
│   └── drift_attribution_report.md  ← 报告（rank8_int4 drift fraction 48.14%）
│
├── 04_delta_profile/
│   └── delta_profile.csv            ← 264 entries: (layer, rank, precision) → KL
│
├── 05_lut_optimizer/
│   ├── optimizer_comparison.csv     ← 6 ε × 3 方法 + baselines
│   ├── optimizer_report.md          ← Pareto 曲线 + 方法对比
│   ├── lut_milp_eps0.1.json         ← MILP LUT (ε=0.1)
│   └── lut_rank_only_eps0.1.json    ← Rank-only LUT (ε=0.1)
│
└── 06_lut_evaluation/
    ├── lut_evaluation.csv            ← actual vs predicted KL + additivity ratio
    └── lut_evaluation_report.md      ← 端到端评估报告
```

---

## Claim → 证据映射

### C1：top-k 内部 rank 长尾

| 证据 | 文件 | 关键数字 |
|---|---|---|
| OLMoE top-8 (主证据) | `01_main_experiment/olmoe_approx_results.csv` | rank-8 median share 4.91%, rank1/rank8 ratio 5.43x |
| Mixtral top-2 (交叉验证) | `01_main_experiment/mixtral_approx_results.csv` | rank-2 median share 0.014%, ratio 14656x |
| LLM-jp top-16 (stress test) | `01_main_experiment/llmjp_top16_approx_results.csv` | rank-16 median share 2.05%, ratio 9.39x |
| 判定标准 | `00_proposal/IdeaA_前置实验设计_MacM5Pro.md` | 强成立: share < 10% 且 ratio > 3 |

### C2：routing drift 是精度损失的重要来源

| 证据 | 文件 | 关键数字 |
|---|---|---|
| 单层 drift attribution | `03_drift_attribution/drift_summary.csv` | rank8_int4 drift fraction 48.14% |
| 多层 cascading (additivity check) | `06_lut_evaluation/lut_evaluation.csv` | actual/predicted KL = 94.1x |
| 解释 | `03_drift_attribution/drift_attribution_report.md` | drift > 30% → cascading correction needed |

### LUT 维度验证

| 维度 | 证据 | 文件 | 关键数字 |
|---|---|---|---|
| rank | rank8_int4 vs rank1_int4 | `01_main_experiment/olmoe_approx_results.csv` | KL 低 58.1x |
| layer | 逐层 sensitivity | `02_layer_sensitivity/layer_sensitivity.csv` | KL ratio 5.74x |
| receiver_group | MILP traffic balancing | `05_lut_optimizer/optimizer_comparison.csv` | group spread 95K→122 |

### LUT optimizer 有效性

| 对比 | 证据 | 文件 | 关键数字 |
|---|---|---|---|
| MILP vs uniform INT4 | 端到端 actual KL | `06_lut_evaluation/lut_evaluation.csv` | 9.41 vs 27.15 → 2.9x 更低 |
| MILP vs greedy | byte saving | `05_lut_optimizer/optimizer_comparison.csv` | ε=0.02 时 +5.2pp |
| MILP Pareto | 6 个 ε 值 | `05_lut_optimizer/optimizer_comparison.csv` | 33.8%→67.8% byte saving |

---

## 关键数字速查

| 指标 | 值 | 来源 |
|---|---|---|
| OLMoE baseline PPL | 15.01 | `01_main_experiment/` |
| rank8_int4 KL vs full (单层) | 0.361 | `01_main_experiment/olmoe_approx_results.csv` |
| rank1_int4 KL vs full (单层) | 20.99 | 同上 |
| rank8/rank1 KL ratio | 58.1x | 同上 |
| layer KL ratio (max/min) | 5.74x | `02_layer_sensitivity/layer_sensitivity.csv` |
| drift fraction (rank8_int4) | 48.14% | `03_drift_attribution/drift_summary.csv` |
| additivity ratio (MILP ε=0.1) | 94.1x | `06_lut_evaluation/lut_evaluation.csv` |
| MILP actual KL (ε=0.1) | 9.41 | 同上 |
| uniform INT4 actual KL | 27.15 | 同上 |
| MILP byte saving (ε=0.1) | 55.8% | `05_lut_optimizer/optimizer_comparison.csv` |
| uniform INT4 byte saving | 75.0% | 同上 |
| MILP traffic spread (ε=0.1) | 122 | 同上 |
| BF16 traffic spread | 95620 | 同上 |

---

## 文档阅读顺序建议

1. `00_proposal/PDF论文的选择.md` — 了解 C1/C2 claim 和 MILP 公式
2. `01_main_experiment/IdeaA_主实验报告_论文版.md` — 主实验全貌
3. `supplement_experiments_report.md` — layer sensitivity + drift attribution
4. `lut_final_report.md` — LUT optimizer + additivity check
5. 各 CSV — 原始数据备查
