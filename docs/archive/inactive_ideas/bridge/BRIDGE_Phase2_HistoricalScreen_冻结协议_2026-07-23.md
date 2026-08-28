# BRIDGE Phase 2：历史预算分配筛选冻结协议

状态：**FROZEN FOR IMPLEMENTATION / HISTORICAL DATA ONLY / NO FORMAL SCIENTIFIC RESULT**  
冻结日期：2026-07-23

## 1. Scientific question

在高精度保护名额固定时，动作发生前可见的弱风险信号，是否比随机分配更能降低请求级 fixed-tail INT4 伤害的上尾风险？

本阶段不检验 KL 点预测，也不把已查看数据重新声明为 sealed evidence。它只决定是否值得生成新的、时序闭合的 first-chunk GPU 标签。

## 2. System mechanism under test

给定一批请求和保护预算 `b`，按冻结风险分数从高到低选择 `floor(bN)` 个请求执行高精度；未选请求执行 fixed-tail INT4。历史 replay 中，高精度请求的相对 KL 记为 0。

## 3. 两个主验证 locus

### H1：Arrival allocation

- 特征：`arrival_lexical`；
- 标签：LLM-jp 新文档 same-prompt fixed-tail INT4 文档级 mean token KL；
- 可观测性：动作前可见；
- 数据：offset 96–159 的 64 篇冻结复现实验文档；
- predictor：仅使用 offset 0–63 的历史 train+validation 拟合，alpha 沿用旧 validation 冻结值。

### H2：Post-prefill to future decode allocation

- 特征：`post_prefill_all`；
- 标签：LLM-jp teacher-forced future decode 的 `fp8top8_rest_int4` mean KL；
- 可观测性：decode 动作前可见；
- 数据：offset 184–231 的 48 篇文档；
- predictor：仍只使用 offset 0–63 的 same-prompt 历史数据拟合，不在 decode 数据上重新选 feature 或 alpha。

`post_prefill_all → same-prompt`、`early_router → full-prompt label` 仅作诊断，不参与 verdict，因为 observation 与 intervention 的时序或标签范围不闭合。

## 4. 固定变量

- 模型：`llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`；
- 主动作：top-16 中前 8 rank FP8、后 8 rank INT4；
- 请求预算：`{0.10, 0.25, 0.50}`；
- 独立样本：document/request；token 不作独立样本；
- 主指标：upper-tail `CVaR90`；
- 次指标：P95、mean、oracle-headroom recovery；
- bootstrap：2,000 次 document bootstrap；
- 每次 bootstrap 的随机策略期望：128 次相同名额的随机选择；
- point random trials：10,000；
- seed：`2026072302`。

## 5. Baselines

1. random protection；
2. frozen `arrival_lexical`；
3. frozen `prefill_nll_only`；
4. frozen `full_router_plus_lexical`；
5. frozen `post_prefill_all`；
6. oracle protection。

只有 H1 的 `arrival_lexical` 和 H2 的 `post_prefill_all` 是主策略。其余是解释性对照，不能替换主策略救结果。

## 6. Pass / fail

单个 locus 在至少两个相邻预算点上同时满足：

1. 相对 random 的 CVaR90 reduction bootstrap 95% LCB `>= 10%`；
2. oracle-headroom recovery `>= 30%`；
3. point estimate 不出现相邻预算反向恶化。

总判定：

- H1、H2 都通过：`HISTORICAL_SCREEN_GO_NEEDS_FRESH_SEALED`；
- 仅一个通过：`PARTIAL_NEEDS_TEMPORAL_FIRST_CHUNK_LABEL`；
- 都失败：`NO_GO_FOR_CURRENT_FROZEN_SIGNALS`。

即使历史筛选 GO，也不能进入系统/网络 claim；下一步必须生成未查看的 first-chunk temporal-tail 标签并重新冻结正式协议。

## 7. 直接判无效条件

- source fit 与 target document ID 重叠；
- target 上选择 feature、alpha、预算或主指标；
- 将 token 当独立样本；
- 保护预算未精确闭合；
- 非有限值、重复 ID、缺列；
- 修改或覆盖历史输入文件；
- 把 fake quant KL replay 表述成实际 latency、energy 或 serving 收益。

## 8. Evidence boundary

本阶段最多支持：“旧数据中是否存在值得进行新 sealed GPU 实验的预算分配 headroom”。不能支持在线 controller、任务质量、真实 INT4 加速、continuous batching、网络拓扑或多 GPU SLO claim。
