# StableBatch Selectability Decomposition Gate — Experiment Audit

**Date**：2026-08-10  
**Auditor**：GPT-5.6-Sol ultra，fresh same-family reviewer，read-only，provisional  
**Overall Verdict**：`WARN`  
**Rapid-experiment blockers**：`P0=0 / P1=0`

## Integrity status

本轮结果可以在冻结范围内解释。`WARN` 来自审计时 tracker/idea report 尚未回写，以及若干 config 字段属于声明性元数据而非动态开关；两者都不会改变 `STOP_PREACTION_STABLEBATCH`。

## A. Ground-truth provenance — PASS

- 输入是真实 WikiText 文档，但没有 dataset label。
- all-M1 是模型自身产生的 self-supervised behavioral proxy，配置明确标记为 `not ground truth`。
- reward 只度量相对 all-M1 的 downstream top-k membership 恢复与新增 harm。

因此结果只支持 execution-shape route-surface 因果结论，不支持模型质量、accuracy 或 human preference。

## B. Score normalization — PASS

- 原始 signed reward 为 `recovered - harmed`。
- 主指标为预冻结的 `(R_selector - R_shuffle) / (R_oracle - R_shuffle)`。
- 原始 reward/recovered/harmed 与 deterministic shuffle 同时报告；uniform random expectation 仅作诊断。

没有以模型自身最大值归一化或隐藏负收益。

## C. Result existence and exactness — WARN

- `RUN_STATUS.json` 为 `COMPLETE`、`scientific_result_eligible=true`。
- run manifest 绑定 selector lock、raw ledger、summary、environment、runtime 与 status。
- 原独立汇总为 `PASS`、`mismatch_fields=[]`。
- 新增 raw-route verifier 从 240 个 U arm 与 1,920 个 action arm 的逐层 top-k route 重新推导 changed-layer sets，`mismatch_count=0`、`summary_mismatch_fields=[]`。

审计时 tracker 的 S1–S4 仍为 TODO，旧 idea report 仍保留此前候选结论；这是文档滞后，不是 phantom result。本轮随后已按正式结果回写。

## D. Dead code/checks — WARN

部分 frozen config 字段是声明性 contract，而不是运行时读取的独立开关；对应行为仍由代码硬约束：feature allowlist、exact unique B=33、reward/LODO、freshness 与 lock ordering 均实际执行。当前值与执行语义一致，因此无 verdict impact。

## E. Scope — PASS

- calibration：旧 16 requests × 15 layers × 8 ranks。
- fresh evaluation：16 个全新 document-disjoint requests × 15 layers = 240 cells，1,920 candidate actions。
- 单 OLMoE revision、单 RTX 5090、BF16 eager、一次正式 run。
- fresh 与 calibration 的全文 hash、window hash 交集均为 0。

不外推到 serving、自然 prevalence、task quality、第二模型、多 GPU、EP/NCCL/RDMA。

## F. Evaluation type — PASS

`self_supervised_proxy on real GPU causal intervention replay`

它不是 `real_gt`、`human_eval` 或 `simulation_only`。

## G. Frozen-gate correctness — PASS

- exact 19-file frozen lock 在执行前验证。
- native-only feature scan 先完成，`SELECTOR_LOCK.json` fsync/hash 后才创建 outcome ledger。
- lock 明确记录 `outcome_rows_existed_at_seal=false`、`result_path_existed_at_seal=false`。
- R/U/A0–A7 action surface、四臂 exact B=33 unique cells、每 cell 一 rank 均闭合。
- full-cohort 与 budget-preserving LODO 按冻结 ranking 执行。
- 远端启动时 GPU 空闲；结束快照只有 formal runner 自身。
- run manifest、raw ledger、selector lock、本地/远端独立汇总和 raw-route verifier 全部 hash/数值一致。

## Claim impact

- **Fresh Oracle opportunity**：支持。Oracle `57`，shuffle `-4`，恢复 `57/84=67.86%`，正收益覆盖 12 个 requests。
- **Static compatibility map**：Gate fail。reward `-7`，低于 shuffle，ROG `-0.04918`，LODO `0/16`。
- **Online observable selector**：Gate fail。reward `-7`，低于 shuffle，ROG `-0.04918`，LODO `0/16`。
- **StableBatch paper mechanism**：当前 pre-action StableBatch 不成立；保留的是 fresh hindsight action opportunity 与 execution-shape propagation 现象，不是可部署或论文 ready 的系统机制。

## Non-blocking notes

- remote project 不是 Git checkout；本轮 provenance 依赖精确 source/data/model/runtime hashes，而不是 commit reachability。
- 明文 SSH 凭据属于独立安全风险，应轮换；它不改变实验结论。

