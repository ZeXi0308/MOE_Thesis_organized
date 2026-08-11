# Research Proposal: When the Scheduler Creates the Hotspot

> 中文工作名：**调度器何时制造了热点：MoE Serving 中策略诱导 expert demand 的可识别性审计**  
> 方法工作名：**CohortFence — Backlog-Boundary Demand Certificate**  
> 状态：`EXPLORATORY / NOT_CURRENT_MAINLINE / NO_EMPIRICAL_RESULT`  
> 日期：2026-07-29

## Problem Anchor

- **Bottom-line problem**：在固定自然 arrival、request identity、完整 route ledger 和 exact model semantics 的条件下，判定反应式 MoE expert load balancer 使用的有限窗口 post-scheduling expert statistics 是否会被 scheduler-induced backlog composition / right censoring 显著改变，进而触发错误 reconfiguration；若该现象成立，给出不伪造未来 route 的可识别证据边界和最小证书。
- **Must-solve bottleneck**：必须把真正的外生 workload shift 与策略诱导的 exposure shift 分开，并把归因差异连接到 hotspot/change-point 决策反转和 full request-DAG 净损失；只展示不同时间索引的曲线不算解决问题。
- **Non-goals**：不提出新的 expert placement optimizer、prefetch、router predictor、route/weight/precision 修改或 receiver controller；不声称 scheduler 永久改变长期真实需求；不复活 PhaseMap、FJRC、ConfidenceGuard、RankLane、CreditReduce、MassCover、TokenRace、QuotaEP、PLTB、RouteFidelity 或 Prefetch。
- **Constraints**：保持当前 `docs/current/README.md` 的共同现象 Gate 为唯一正式执行线；本方向仅为隔离探索。实验必须 exact semantics、自然 continuous arrival、document-disjoint、至少两个模型、冻结 detector/gate/denominator；失败后不得改 workload、阈值、cohort 或索引救活。现有 OLMoE/LLM-jp 与 RTX 5090 只够资格性 pilot；正式 EP/TPOT/P99 主张需要至少一个 serving-scale open MoE 和真实 optimized multi-GPU EP。
- **Success condition**：全文 prior-art 查杀后仍不存在等价的 fixed-arrival/fixed-route scheduler intervention 与 partial-identification certificate；两个模型自然 workload 均出现预注册的策略诱导假热点或假变点；certificate 不退化为 always-abstain，并保留真实热点；错误 reconfiguration 在完整路径上的影响达到预注册系统显著性门；最终在真实 multi-GPU EP 上复核，且 fresh same-family CCF-B mock review 无 fatal novelty/method objection。任一硬门失败即 NO-GO。

## Technical Gap

### 已被近邻覆盖的部分

截至 2026-07-29 的全文核对显示，原始 C10 表述过宽，以下内容不能再当贡献：

- [PROBE](https://arxiv.org/html/2602.00509) 已明确指出 continuous batching 中请求 join/depart 会使 batch composition churn、expert popularity 快速波动，历史统计滞后，并用 lookahead prediction/prefetch 解决下一层瞬时负载；因此“首次发现 active-set churn”不新。
- [Director](https://arxiv.org/html/2607.08782) 已明确区分 completed/past requests 与 incoming requests，并用预测器做 proactive placement；因此“改用 incoming-request predictor”不新。
- [Gimbal](https://arxiv.org/html/2606.15177) 同时做 request scheduling、按 profiling window 统计 expert load/source-expert matrix、expert placement，并把 expert pressure 反馈给 scheduler；它已经形成真实闭环，但没有把统计量分解为外生 arrival demand 与 policy-induced backlog boundary。
- [CRAFT](https://arxiv.org/html/2603.28768) 和生产 EPLB 会对 post-routing load 做周期 rebalancing；[vLLM EPLB](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py) 的 `window_size` 是 expert load recording window，[SGLang](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md) 也建议按请求/iteration 周期重平衡。问题不是狭义 wall clock，而是所有 **post-scheduling service clock**。
- [Mixture-of-Experts Serving](https://arxiv.org/html/2607.17880) 把每步 workload vector 当作算法已观察到的输入，并研究 latency/reconfiguration trade-off；它没有建模“换 policy 后 workload observation sequence 本身会变化”的反事实比较。
- [Activation Patterns](https://arxiv.org/html/2604.23150) 在 batch size 1 下记录 per-token route，证明 domain-specific activation；它没有 continuous-serving scheduler intervention。

### 仍可能存在的最窄缺口

现有工作把 batch/step/window 中已经执行的 route count 当作 workload state，或者直接预测下一个 batch/layer；尚未看到工作同时完成：

1. 对同一 natural arrival/request/route ledger 只干预 scheduler，定义 policy-induced exposure/censoring 的因果 estimand；
2. 证明有限 service window 的 observed expert counts 包含一个 scheduler-dependent backlog boundary，而长期稳定极限又应当 policy-invariant；
3. 证明仅看已执行 route history 时，未完成 arrival cohort 的最终 demand 在无额外假设下不可点识别；
4. 输出 `IDENTIFIED / AMBIGUOUS / INVALID` 的 partial-identification certificate，而不是再训练一个预测器；
5. 把被证书判为证据不足的 trigger 放进 full request-DAG，验证它是否真的造成 migration 决策反转或 SLO 净损失。

这仍只是 **provisional gap**。若后续查到等价 fixed-ledger counterfactual 或 certificate，方向立即 KILL。

## Method Thesis

- **One-sentence thesis**：有限窗口 post-scheduling expert count 等于 arrival-cohort demand 加上 scheduler-dependent backlog boundary；CohortFence 用 cohort sealing 与 partial identification，只在所有可行剩余 routes 对 hotspot/change-point 给出相同判定时出具证书，从而区分“已识别的 demand evidence”与“尚被 scheduler censoring 污染的 trigger”。
- **Why this is the smallest adequate intervention**：它不预测未来 route、不优化 placement、不修改 serving semantics，只增加一个异步 identity-complete ledger、守恒检查和小型区间求解器；若这都不能避免假 trigger，就没有理由先做更复杂 controller。
- **Why this route is timely**：近期 PROBE、Director、Gimbal、CRAFT 与 production EPLB 都开始更快地观察、预测或反馈 expert load；控制回路越快，窗口内统计的 policy dependence 越可能与迁移/复制时间尺度冲突。这里的及时性来自新系统闭环，而非堆叠 foundation-model buzzword。

## Contribution Focus

- **Dominant contribution**：MoE continuous serving 的 backlog-boundary demand decomposition、在线不可点识别边界，以及一个不依赖未来-route predictor 的 cohort-sealed partial-identification certificate。
- **Optional supporting contribution**：fixed-arrival/fixed-route paired replay，将证书判定与现有 EPLB/Gimbal-like trigger 的 full-DAG action sign 和 SLO 影响连接起来。
- **Explicit non-contributions**：不把 active-set churn 本身写成首次发现；不提出 placement/prefetch/predictor；不把 request-index/token-index 当成天然 ground truth；不把单卡 replay 写成 EP/TPOT/P99 结果；不把数学恒等式本身包装成系统贡献。

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**：冻结 pretrained MoE、tokenizer/model revision、router/top-k、生成 token、arrival ledger、现有 scheduler、现有 hotspot/change-point detector、现有 placement action set、service surface 和 full-DAG evaluator。
- **New trainable components**：0。
- **New runtime components**：一个异步 cohort ledger、一个守恒/完整性校验器、一个按 trigger 调用的可行区间求解器。
- **Tempting additions intentionally not used**：future-route predictor、low-bit shadow model、RL/OPE controller、new placement heuristic、learned change-point detector、synthetic-skew rescue cell。

### System Overview

```text
natural arrivals + frozen model/routes
               |
               v
      existing serving scheduler pi
               |
     +---------+-------------------------+
     |                                   |
     v                                   v
post-service expert counts        identity-complete cohort ledger
(existing detector input)         arrival/request/token/layer/route/EOS
     |                                   |
     v                                   v
candidate hotspot/change-point    observed demand + feasible remaining set
     |                                   |
     +----------------+------------------+
                      v
             CohortFence certificate
          IDENTIFIED | AMBIGUOUS | INVALID
                      |
                      v
        paired fixed-ledger causal replay
                      |
                      v
     full request-DAG action sign / SLO impact
```

Phase A 只产出审计证书，不影响 runtime action。只有共同现象 Gate、方法 Gate 和独立 review 均通过后，Phase B 才允许把 `AMBIGUOUS` 作为 existing trigger 的 veto-only guard；仍不新增 placement policy。

### Core Mechanism 1：Backlog Conservation

请求 `i` 在逻辑 route event `j` 对 expert `e` 的固定贡献记为 `R_ije`，arrival time 为 `a_i`。scheduler `pi` 决定执行时间 `tau_ij^pi`，但在 exact-ledger 条件下不改变 `R_ije`。

有限 service window `(s,t]` 的 observed count：

```math
N_e^pi(s,t] = sum_{i,j} R_ije * 1{s < tau_ij^pi <= t}.
```

令 `A_e(s,t]` 为该 arrival cohort 最终产生的 expert-e route 总量，`Q_e^pi(t)` 为截至 `t` 已到达但尚未执行的潜在 expert-e route 总量，则：

```math
N_e^pi(s,t] = A_e(s,t] + Q_e^pi(s) - Q_e^pi(t).
```

同一 arrival/route ledger 下两种 scheduler 的观测差异完全来自 backlog boundary：

```math
N_e^{pi1}(s,t] - N_e^{pi2}(s,t]
  = Delta Q_e(s) - Delta Q_e(t).
```

必须同时证明反向边界：若系统稳定、无 drop、所有请求最终完成且 `Q_e^pi(T)/T -> 0`，则长期平均 expert rate 与 scheduler 无关。因此论文只主张 **controller timescale 的有限窗口污染**，不主张永久需求改变。

### Core Mechanism 2：Cohort-Sealed Partial Identification

对 arrival cohort `C`，已执行 route count 为 `O_e(C,t)`。对于未完成请求，只使用在线已知的合法上界：当前 logical progress、`max_new_tokens`、MoE layer 数、top-k、EOS/cancel ledger；不预测具体 future expert identity。

令 `X_C(t)` 为与当前历史、剩余长度上界和 top-k conservation 一致的所有 future-route count 向量，则 cohort 最终 popularity 的可行集合为：

```math
P_C(t) = { p : p_e = (O_e + x_e) / sum_f(O_f + x_f), x in X_C(t) }.
```

将冻结 detector `D` 应用于 `P_C(t)`：

- 所有 `p in P_C(t)` 给出同一 hotspot/change-point 结论：`IDENTIFIED`；
- 合法 `p` 之间结论分歧：`AMBIGUOUS`；
- route/top-k/completion/hash/denominator 不闭合：`INVALID`。

cohort 全部完成后 `X_C(t)={0}`，证书收缩为 exact point。输出 schema：

```text
status
expert_intervals
sealed_cohort_estimate
cohort_watermark
certificate_lag
scheduler_contamination_bound
detector_decision_set
reason
model/router/scheduler/ledger hashes
```

单 expert bound 尽量闭式求解；完整 top-k conservation 只在 detector trigger 时运行小 LP。每个 route event 仅做 `O(k)` 异步累加。

### Identification Boundary

对任一未完成请求，可以构造两条与当前所有观测完全一致的 future route：其一后续集中到 `e1`，其二集中到 `e2`。两者当前 history 相同，却使最终 arrival-cohort demand 与 hotspot 判定相反。因此，只依赖已执行 tokens 的在线 point estimator，在不引入 future-route model、lookahead 或统计假设时不可统一识别。

这也是为何简单 reindex 不够：

- request completion order 由 scheduler 控制；
- token-index 只包含已到达该 step 的请求，仍有 policy-dependent right censoring；
- request-weighted 与 token-weighted demand 在变长输出下是不同 estimand；
- logical index 移除了时间分母，不能直接回答 capacity/migration amortization；
- 只有完整封口的 fixed arrival cohort 才是 scheduler-invariant offline reference。

### Rejected Route B：Counterfactual OPE

备选路线曾考虑把 serving 写成 semi-Markov queue，用 logging scheduler 与 reference scheduler 的 occupancy density ratio 估计 counterfactual expert rate。该路线被拒绝，因为 deterministic FCFS/SRPT 常无 positivity，组合 batch action space 巨大，需要注入随机探索和完整 state propensity，且极易变成把标准 OPE/DICE 换成 MoE outcome。它更复杂、更难验证，也不比 partial-identification 路线更贴近 Problem Anchor。

### Modern Primitive Usage

- **Used**：无新 foundation-model primitive。
- **Reason**：Director 与 PROBE 已证明 predictor/lookahead 是可行且拥挤的路线；本问题需要的是证据语义和可识别性，而不是另一个预测网络。强行加入 LLM/RL 只会制造 contribution sprawl 和 exactness 风险。

### Integration into Existing Pipeline

1. 复用当前 `capture_native_routes` 类资产，补齐 continuous arrival、request ID、logical step/layer、EOS/cancel、output hash 和 scheduler policy ID。
2. 复用 shared causal replay / full request-DAG 基础，但当前 `docs/current/README.md` 已明确 full-DAG 仍未实现；在它闭合前不报告 migration harm。
3. 先做纯 measurement Gate：相同 ledger 重放 FCFS、pressure-aware scheduler 和 schedule-invariant negative control，不执行真实 migration。
4. Gate 通过后才将冻结 detector 的 triggers 映射到既有 placement action set，在 exact replay 中计算 action reversal 和 net cost。
5. 单卡只做 route producer / service-surface qualification；真实 optimized EP 复核才形成系统结果。

### Training Plan

无训练。calibration 只用于冻结 cohort duration、detector window 和 confidence procedure；evaluation split 不再调参。若未来加入预测收窄区间，视为新论文路线，必须重新查新和重新冻结，不在本方案内。

### Failure Modes and Diagnostics

- **Future bounds too wide / always abstain**：报告 `AMBIGUOUS` 比例与 certificate lag；若超过冻结门，方法 KILL，不增加 predictor 抢救。
- **Observed difference is only throughput scaling**：同时报告 raw count、per-token composition、arrival-cohort target 和 backlog boundary；若统一 denominator 后消失，记 measurement bug，不算论文现象。
- **Scheduler changes outputs/routes**：逐请求 token/output/route hashes 不同即 `INVALID`。
- **Timeout/drop changes completion set**：Phase A 冻结 completion policy；若 completion set 仍不同，单独记 policy-mediated admission effect，不混入 demand-identification claim。
- **Synthetic-only effect**：只作为 fixture，不进入主 claim。
- **Small-model-only effect**：只能保留 qualification，不形成 CCF-B claim。
- **PROBE/Director already solve practical issue**：二者作为强 baseline/contrast；若 incoming prediction or per-layer lookahead 消除所有 harmful triggers，CohortFence 无独立系统价值，方法 KILL。
- **Mathematical identity but no decision harm**：若没有跨模型 action reversal 或 full-path impact，降级为 protocol note，不写成 CCF-B idea。

### Novelty and Elegance Argument

新意不能落在“continuous batching 让热点变化”，该点已被 PROBE 覆盖；也不能落在“过去统计滞后”，Director/PROBE 已覆盖。唯一可辩护的新意是：

> 把 service-window load 明确定义为 policy-dependent outcome，以 fixed arrival/route ledger 给出 backlog 守恒和不可识别边界，再用不预测 future route 的 partial-identification certificate 判断 existing trigger 是否有充分证据。

它保持一个 dominant contribution：**证据识别**。paired replay、full-DAG 与 veto-only integration 都服务于同一个问题，不是并列 controller。

## CCF-B Candidate Hard Gates

这些 Gate 预先冻结；语义评分不能覆盖硬门失败。

| Gate | 通过条件 | 当前状态 |
|---|---|---|
| G0 Authority | 全部产物保持 exploratory，不修改当前主线 | PASS（文档级） |
| G1 Novelty | 近邻全文查杀无等价 fixed-ledger intervention + partial-ID certificate；无 fatal collision | OPEN；PROBE 已杀掉原始宽 claim |
| G2 Natural phenomenon | 两模型自然 cell：hotspot Jaccard `<0.90` 或 false change-point `>=5%`，且 negative controls 通过 | UNVERIFIED，0 pilots |
| G3 Decision significance | 两模型均有错误 trigger；至少一个 common natural cell 的 full-DAG action sign reversal 或净影响 `>=5%` | UNVERIFIED |
| G4 Certificate utility | `AMBIGUOUS <=50%`，replay-confirmed true-hotspot retention `>=90%`，不晚于可摊销 migration horizon | UNVERIFIED |
| G5 Exactness/protocol | arrival/request/token/layer/route/output/completion/denominator 全闭合，split 与阈值冻结 | OPEN |
| G6 Formal systems proof | 至少一个 serving-scale open MoE + 独立模型，在 optimized multi-GPU EP 复核 TPOT/P99/SLO-goodput | BLOCKED_CURRENT_RESOURCES |
| G7 Review | fresh same-family mock CCF-B review 无 fatal novelty/method objection；标 provisional | OPEN |

只有 G0–G7 全部通过，才能称为“已达到 CCF-B-ready evidence”；当前只是一条 paper-shaped hypothesis。

## Claim-Driven Validation Sketch

### Claim 1：有限窗口 load 含有足以改变决策的 scheduler-dependent backlog boundary

- **Minimal experiment**：冻结两模型、两类以上 natural document-disjoint arrivals、每请求 exact route/output ledger；paired replay FCFS、pressure-aware scheduler、仅时间平移/不改 active-set 的 negative control。对 service-time、iteration/step 和 sealed-arrival-cohort 三种 ledger 使用同一 detector。
- **Baselines / ablations**：raw EPLB count、token-normalized count、request-index、token-index、sealed cohort oracle、PROBE-like next-layer oracle、Director-like incoming oracle；去掉 backlog term、固定输出长度、关闭 continuous batching。
- **Metric**：hotspot Jaccard、change-point FPR/FNR、backlog-boundary share、decision disagreement、route/output equality。
- **Expected evidence**：不是预设正结果；必须在两个模型自然 cell 达到 G2，否则 C10 NO-GO。

### Claim 2：CohortFence 能拒绝 harmful false trigger，而不是永远 abstain

- **Minimal experiment**：在冻结 detector triggers 上运行 certificate；把 raw trigger、CohortFence-veto、sealed-cohort oracle 和 robust static/no-migration 放入相同 full request-DAG。先 exact CPU replay，后 optimized multi-GPU EP。
- **Baselines / ablations**：no certificate、只等 cohort completion、简单 hysteresis、longer window、PROBE/Director predictive signal、robust static placement；删除 partial-ID solver，仅用 point estimate。
- **Metric**：AMBIGUOUS rate、true-hotspot retention、false-trigger precision/recall、certificate lag/overhead、migration count/bytes、action sign reversal、P99/TPOT/SLO-goodput。
- **Expected evidence**：G3/G4/G6 同时通过；否则 certificate 仅是正确但无用的 protocol。

## Experiment Handoff Inputs

- **Primary claim**：post-scheduling expert-load observation 在 controller timescale 上具有 policy-dependent backlog boundary，足以造成跨模型自然 workload 的 reconfiguration decision error。
- **Supporting claim**：partial-identification certificate 能在不预测 future route 的前提下识别可行动 trigger，并以低 abstention/lag 保留真实热点。
- **Anti-claim**：若长期稳定且 drain complete，平均 expert demand 应 policy-invariant；若短窗差异无 action harm，C10 不成立。
- **Must-run ablations**：fixed-length、no continuous batching、longer window、simple hysteresis、request/token reindex、PROBE/Director oracle、robust static/no migration、zero migration cost/full cost。
- **Critical datasets / models**：现有 frozen OLMoE + LLM-jp 只做 qualification；formal 至少加入 Qwen3-30B-A3B 或 Mixtral-8x7B 等 serving-scale open MoE，并用独立模型复现；natural chat/math/code/general arrivals，document-disjoint。
- **Highest-risk assumptions**：route invariance under scheduler intervention；partial-ID 区间不会长期过宽；full-DAG evaluator 忠实；真实 migration horizon 足够长；closest work 未覆盖 fixed-ledger identification。

## Compute & Timeline Estimate

- **Qualification**：CPU paired replay + RTX 5090 native route capture，约 2–4 周、20–40 GPU-hours；只给 G2/G4 的资格信号。
- **Full method implementation**：identity-complete continuous producer、cohort ledger、interval solver、full request-DAG，约 4–8 周；当前 full-DAG 是最大工程 blocker。
- **Formal systems validation**：真实 4–8 GPU optimized EP、serving-scale model、迁移/queue/full-path 测量，约 80–160 GPU-hours，具体取决于可用后端与模型 fit；没有该环境就不能完成 G6。
- **Data / annotation**：使用公开 natural workloads，无人工标注；需要冻结请求/文档/arrival manifest 与 hashes。
- **Total**：若 G2 在第一轮失败，2–4 周停止；若全部通过，约 8–12 周形成完整候选证据，尚不等于录用保证。

## Current Verdict

`REVISE-CANDIDATE / NOT YET CCF-B READY`。

当前新增价值是把 C10 从宽泛现象收缩成可形式化、可被杀死的证据方法；当前没有任何 pilot 或系统结果，且 PROBE 已构成重大 novelty 压力。下一步必须由 fresh reviewer 决定它是 `REVISE` 还是 `RETHINK`，不能由本提案自评为 READY。
