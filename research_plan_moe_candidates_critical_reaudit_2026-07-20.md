# 研究计划：MoE 系统候选方向批判性重审计（2026-07-20）

## 查询类型判定

本任务属于**深度（depth-first）审计任务**：核心问题单一（"当前 PASS/PROMISING 候选是否真的有价值、能否重构"），但需要从多个独立角度（科学假设有效性、实现/评估bug、系统重构可行性、外部文献novelty、最小决定性实验设计）反复审视同一批候选。同时具有一定 breadth 成分：需要覆盖receiver-aware、Quality Isolation、Expert Prefetch、以及潜在遗漏的"其他存活候选"（Energy-SLO Precision EP）。

## 信息来源规划

1. **内部文档**（本仓库 `/Users/leandrozhao/Desktop/毕设论文资料/`）：已按时间顺序读取 2026-07-20 当天全部关键文档，包括三份"批判性设计/深化"文档、四轮GPU有效性实验报告（receiver-causal+prefetch、Quality Isolation、Receiver Progressive/Codec、Prefill→DecodeFragility）、已死路线复查文档，以及项目历史候选清单（16候选进展报告、全部候选时间线、MoE_Approach_Registry、PartialGuard严格报告）。用 code-explorer 子代理补充核对是否存在遗漏的"仍存活候选"及Approach Registry完整表。
2. **外部文献 grounding**：用 research_subagent 检索 contextual bandit/regret-minimizing在线控制器、incast/tail-latency感知拥塞控制（RCC/RFCC/ICI等）、quality-debt/harm-aware公平调度（VTC之外的LLM serving公平性工作）、以及2025-2026年MoE通信优化新论文（COMET/DeepEP等），评估三条候选重构后的novelty冲突风险。
3. **微信公众号检索**（按规则要求执行，已加载 wechat-article-search skill）：鉴于本题目属于高度专业化的英文系统研究领域（MoE EP通信、RDMA拥塞控制、LLM serving公平性），公众号文章预期信息密度和时效价值有限，仅作补充/风险意识来源，不作为主要证据。

## 分析框架

对每条候选（receiver-aware、Quality Isolation、Expert Prefetch、Energy-SLO Precision EP）逐一回答：
1. 核心科学假设 vs. 实现/评估失败的区分；
2. 新机制/控制策略/反馈信号/在线优化/硬件感知/问题重构能否重新建立价值；
3. receiver-aware专项：queue/latency/NIC telemetry、regret-minimizing/contextual bandit、critical-path/incast/tail-latency感知、无需regime分类的鲁棒策略——逐项评估可行性与文献支持度；
4. Quality Isolation专项：action-conditioned harm estimation、uncertainty-aware allocation、streaming quality debt、tenant-level fairness——逐项结合最新一轮（第四轮decode fragility）证据评估必要性和可行性；
5. 每个可救方向给出1-3个最小成本决定性实验（假设/信号/基线/阈值/confound）；
6. 归类：主贡献 / 第二贡献 / 负结果或经验分析 / 应停止投入。

最终交付：候选排序、当前评分与重构后潜力评分、推荐统一论文主线、两周实验清单。保持批判性，不为保留已有工作强行救idea，优先寻找能在真实多GPU MoE serving、真实通信、端到端TPOT/P99/质量指标上成立的系统问题。

## 执行状态

内部文档梳理、Approach Registry核对、外部文献grounding均已完成（见对话记录）。产出报告：`research_report_moe_candidates_critical_reaudit_2026-07-20.md`。
