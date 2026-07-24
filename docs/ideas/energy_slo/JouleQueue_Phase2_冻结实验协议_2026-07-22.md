# JouleQueue-MoE Phase 2 冻结实验协议

状态：**SUPERSEDED_FOR_FORMAL_EXECUTION / PHASE4_BLOCKED / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-22  
协议版本：`joulequeue-v1`  
本轮证据上限：真实 RTX 5090 BF16 expert-stage energy/latency surface + route-real arrival replay。**不是完整模型 serving、集群总能耗或多 GPU EP 结论。**

> **2026-07-22 Phase 4 勘误：** 本协议不能直接进入正式执行。独立 Review 发现能量计数窗口的固定 envelope 被不同 repeat 分母除开、同一 activation prefix 的重复测量被误当独立 input event、route-energy-mass coverage 和 thermal/provenance 门缺失，且现有 `max_exact_jobs=12` 小于 LLM-jp 单 token `top_k=16`。详见 [`JouleQueue_CodeReview_Phase4_2026-07-22.md`](JouleQueue_CodeReview_Phase4_2026-07-22.md) 与 [`../../01_current_status/CJC_JouleQueue_Phase4_交叉审计_2026-07-22.md`](../../01_current_status/CJC_JouleQueue_Phase4_交叉审计_2026-07-22.md)。`SUPERSEDED` 只针对 `joulequeue-v1` 的正式协议与 runner；**不构成 Energy-SLO 机制的科学 No-Go**。必须返回 Phase 2 重定义测量估计量、独立样本、job granularity 与可分解 oracle，再重新冻结、实现并 Review。

sealed 输入打开后，任何门槛、baseline、数据窗口、能耗来源或统计单位变更都会使已有结果标记 `SUPERSEDED`。Phase 4 签字前禁止正式执行。

## 1. 冻结主张与边界

> 在 routing、expert、BF16 数值格式和输出集合不变时，利用 per-expert ready-row fragmentation、request slack 与实测 marginal joules，安全地 defer/coalesce expert invocations，并让紧急请求走 fast path；若 energy-optimal launch 决策不同于 fixed timeout、EDF和 throughput μ-queue，则有机会在硬延迟约束下降低 GPU board J/completed token。

本轮第一刀只回答：**存在多大的 expert-stage oracle headroom，以及它是否被最强简单 batching baseline 吃掉。** 通过只表示 `GO_TO_INTEGRATED_ASYNC_PROTOTYPE`，不能写成完整 Energy-SLO 科学 Go。

动作不量化、不 DVFS、不 drop/skip、不 reroute/replicate、不 offload，不改变 expert 权重或 routing。所谓 exact-quality 指无有意近似；浮点 kernel 因 batch shape 造成的数值差异仍必须过预注册门。

## 2. 假设与递进 hard gates

### H0

任一条件成立即接受 H0：

- BF16 expert execution 没有可重复的 non-convex launch/underfill energy tax；
- best fixed timeout、EDF 或 throughput μ-queue 已达到 oracle；
- oracle 节能来自少完成、延后 drain、免费 idle、错分母或不同功耗来源；
- SLO/数值门失败；
- 两模型任一主 workload cell 不过 10% / 1.03 门。

### H1

在相同 route/arrival、相同完成集合、完整 drain、同 GPU board-energy source 和数值门下，clairvoyant launch oracle相对每个 SLO 合格强基线均满足：

- `J/completed-token` 改善的 paired 95% CI 下界 **≥10%**；
- P99 completion/TPOT proxy ratio 的 paired 95% CI 上界 **≤1.03**。

### 递进停止门

1. **E0 Measurement**：能耗/延迟 surface 通过 UUID、同窗、重复性、单位和 thermal gates。
2. **E1 Non-convexity**：至少两个有质量的 row bins 显示合批的 marginal J/row下降，且 route trace 中有足够覆盖。
3. **E2 Oracle necessity**：oracle 相对全部 SLO-qualified 强基线过 10%/1.03 门。
4. **E3 Causal approximability**：若 oracle 过门，至少一个 calibration-frozen causal policy 保留正 headroom；否则只能记“clairvoyant ceiling”，不进入 runtime。

E1/E2 失败即 `NO_GO_JOULEQUEUE_V1`；模型/驱动/NVML capability 缺失为 `BLOCKED`。

## 3. 状态、动作、目标与守恒

job：

```text
j = (request_id, forward_id, layer_id, token_id,
     expert_id, arrival_us, rows, deadline_us)
```

- **状态**：每 expert ready rows、job age、earliest deadline/slack、冻结的 `E_e(m),T_e(m)` 及 CI。
- **动作**：`launch(queue subset)`、`defer`、`urgent_flush`；同 expert 的 jobs 才能合并。
- **目标**：最小化 GPU board J/completed output token；本轮另报 expert-stage J/routed-token，二者不得混名。
- **约束**：routing/precision固定、每 job exactly-once、full drain、deadline/starvation、request/KV identity、数值门。

正式守恒：

```text
admitted_jobs = completed_jobs + explicitly_rejected_jobs
admitted_rows = completed_rows + explicitly_rejected_rows
queue_end = inflight_end = 0
drop = duplicate = early_termination = 0
```

本协议不允许 admission reject，故主 run 中 admitted=completed。少完成请求或只统计已被策略偏爱的 token 会使 run 无效。

## 4. Route/arrival identity 与数据隔离

沿用 CJC v1 的 identity-complete producer契约；旧 `capture_moe.route_rows()` CSV 不合格。每条 route 必须具有 request/forward/layer/token/slot/expert identity、valid mask和 model/data manifest hashes。

模型 revision 固定：

- `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`；
- `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055`。

数据使用 `wikitext-103-raw-v1:train`，与 CJC 相同的 calibration `[20000:22000)`、sealed `[40000:44000)` candidate windows和 SHA 选择规则，但生成独立 manifest。calibration 64、sealed 128 文档，每模型 128 tokens；hash 与历史/两 split 不重叠。

route producer 给出逻辑 expert jobs；arrival workload 独立生成并冻结：

1. `poisson_rho50`；
2. `mmpp_rho80`，二态参数仅由 calibration service rate确定。

主 seeds `2026072201..05`；CI 跨门才允许 `...06..10`。所有 arm 逐 job 共享 arrival、route、deadline和 service/energy sample identity。

## 5. 真实 5090 expert surface

只测 BF16 routed expert MLP，使用真实模型 revision、真实 expert 权重和真实/审计过的 expert-input activations；若只能使用 deterministic synthetic activation，产物必须标 `SHAPE_REAL_INPUT_PROXY`，只能作 dev，不得打开 E2。

固定 row grid：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

每模型按 `sha256(seed,layer,expert)` 在 calibration 前冻结 4 layers × 每层4 experts。每 cell：

- 20 warm-up calls；
- paired AB/BA order，至少 10 independent trials；
- 每 arm inner repeat 自动翻倍，直至 workload window ≥2 s；
- CUDA event 测 device latency，monotonic wall time另报；
- 同一次 trial 中 total-energy source必须一致；不同 arm/source 不可比较；
- 环境记录 GPU UUID、driver、power limit、clock、温度、模型/权重 hash。

对连续 jobs rows `m_1...m_q`：

\[
Saving_E=\sum_i E(m_i)-E(\sum_i m_i),
\qquad
Saving_T=\sum_i T(m_i)-T(\sum_i m_i).
\]

只允许对 grid 内或预注册单调/保守 upper interpolation求值；越界回退逐 job immediate，不得外推。

E1 要求：至少两个 row bins 在两个模型中均有 `J/row`下降的 95% LCB>0，且这些 bins 各覆盖 calibration expert-energy mass ≥10%。若 surface 近线性、噪声过大或 trace 不覆盖，停止。

## 6. GPU board-energy 会计

复用 `power_accounting.py` 的 monotonic boundary、UUID、counter-first 和 completed-token denominator原则，但新实现必须补齐以下缺口：

- sampler 配置周期固定 **5 ms**，正式 observed max gap hard gate仍为 **≤20 ms**；
- total-energy counter start/end 均记录 monotonic timestamp；counter 与 sample workload boundary必须统一，不能把不同窗口当同窗；
- background sampler异常必须在 `stop()` 重抛；
- baseline/candidate 必须使用同一 `nvml_total_energy_counter` 或同一 `monotonic_power_integral` source；中途 fallback 使整对 trial无效；
- start 前、end 后 CUDA synchronize；禁止截断未完成 kernel；
- total board energy 为主口径，idle-subtracted dynamic energy仅敏感性；
- idle calibration记录窗口、温度/clock与 CI，不能用单个拍定常数。

主总能耗：

\[
E_{total}=counter(t_1)-counter(t_0)
\]

或在 counter 不支持时对显式同窗 boundary samples 作梯形积分。`t0` 在首个 measured job release 前同步，`t1` 在最后 job完成、queue/inflight为零并同步后。

本轮 oracle replay的 `expert_stage_J/completed_token` 必须把 defer 等待期间的 board idle energy计入；不得只相加 kernel dynamic energy。由于 attention/router/KV/CPU/NIC 未执行，它不能称 full-serving J/token。

## 7. 数值与质量门

合批前后使用相同 BF16权重、相同 rows和 canonical output split。每个 profiled expert/input cell 比较 separate-vs-coalesced outputs：

- finite且 shape/row identity完全一致；
- `max_abs_error ≤ 2e-2`；
- `mean_abs_error ≤ 2e-3`；
- cosine error `1-cos ≤1e-4`。

上述只是 kernel-level数值门。若进入 integrated runtime，必须另做独立 KV、无 route replay 的 end-to-end teacher-forced quality gate；本轮通过不能写“完整模型质量等价”。

## 8. 强基线与 oracle

| Arm | 冻结定义 |
|---|---|
| `immediate` | job 到达即按当前 rows launch |
| `best_fixed_timeout` | calibration grid `0/5/10/20/50/100/200 us`，sealed 固定 |
| `best_static_rows` | queue rows达到 `8/16/32/64/128`即 launch，另有 max-age hard gate |
| `edf` | earliest-deadline first；只在同 expert安全合批 |
| `throughput_muqueue` | 最大 tile occupancy/rows，deadline与 max-age flush |
| `amoe_style` | per-layer μ-queue + ready-job defrag；不读 energy LUT |
| `festina_like_profiled` | calibration-only energy/latency profile选固定 operating point；不读未来 |
| `clairvoyant_energy_oracle` | 可读本 sealed episode未来 arrival，仅作 necessity ceiling |
| `causal_joulequeue` | 只读当前 queue/slack与 frozen LUT；比较 marginal joule saving和等待风险 |

oracle 按 expert/layer episode做 exact dynamic programming或可校验的 exhaustive solution；若使用 relaxation，必须同时给 upper/lower bound，未闭合 gap不能打开 E2。deadline、max-age和 full-drain均进入优化；禁止漏算 defer idle energy。

最强 baseline 是所有数值与 SLO 合格 baseline 中 J/token最低者，不能只挑 immediate。若 `best_fixed_timeout` 或 `amoe_style` 覆盖 oracle，本 idea失败。

## 9. SLO、指标与统计

本轮主延迟是 route-real expert-stage token completion proxy：token 的当前层 completion为其 top-k expert jobs 全部完成时刻；跨层按冻结 dependency串联。它不是 TTFT/TPOT实测。

calibration 用 `immediate` 定义每 cell：

```text
SLO = 1.10 × P99 token-completion proxy(immediate)
max_age = 0.25 × SLO
```

主指标：

- `expert_stage_board_J / completed_output_token`；
- P99 token completion ratio；
- SLO violation rate；
- full-drain makespan、launch count、mean rows/launch、queue age/starvation。

另报 kernel-sum energy、idle energy、dynamic energy、P50/P95/CVaR、per-expert/load bins；这些不能替代主口径。

统计单位为独立 document/request episode；paired hierarchical bootstrap先按 document、再按 seed，`n=2000`。surface CI以 `(layer,expert,input event)` 为独立单元，不把 inner repeats当独立样本。主结果逐模型×workload cell报告，不允许只报 pooled。

## 10. Go / No-Go 与停止规则

`GO_TO_INTEGRATED_ASYNC_PROTOTYPE` 要求全部满足：

1. E0/E1通过；
2. 两模型×Poisson/MMPP四个主 cell 中，oracle 相对**每个** SLO-qualified强基线的 J/token改善95% LCB≥10%；
3. 同四 cell P99 ratio 95% UCB≤1.03，violation增量UCB≤1pp；
4. 零 drop/duplicate/unfinished，full drain，完成集合完全相同；
5. numerical gate通过；
6. causal_joulequeue在四 cell均有正改善且未被 best fixed timeout/AMoE-style 覆盖；
7. 结果不依赖不同 energy source、idle-subtracted口径或单一 row bin。

任一主 cell失败即 `NO_GO_JOULEQUEUE_V1`。CI 跨门只可追加预留 seeds；仍跨则 `INCONCLUSIVE/STOP`。不得通过加入 FP8、DVFS、offload、replica、drop、学习 controller、删 baseline 或放宽10%/1.03来救。

## 11. Phase 3 实现验收用例

至少覆盖：

1. job identity exactly-once；duplicate/missing/unfinished hard-fail；
2. route row closure与 top-k identity；旧 route CSV formal hard-fail；
3. 仅同 expert jobs可合批；request/token输出可逆拆分；
4. future arrival变化不改变 causal_joulequeue当前动作，oracle可变化；
5. timeout/row-threshold calibration与 sealed隔离；
6. full drain后 queue/inflight=0，所有 arm完成集合相同；
7. `100 W × 10 s / 100 tokens = 10 J/token`；单位 mW/W、mJ/J测试；
8. sampler interval配置5 ms，observed gap>20 ms formal hard-fail；
9. counter/sample同窗；background exception传播；counter回退hard-fail；
10. CUDA/NVML UUID、energy source、power/clock环境一致；
11. surface paired AB/BA、≥2 s、trial独立单位；
12. interpolation越界回退 immediate；不能静默外推；
13. wait期间 idle board energy计入；kernel-only不能生成主指标；
14. numerical separate/coalesced门；不同 rows不得错配；
15. oracle exactness用小规模 exhaustive fixture交叉验证；relaxation gap未闭合禁止GO；
16. baseline缺失、SLO不合格或完成集合不同时禁止比较；
17. formal runner缺 Phase 4 `SIGNED-OFF` 或 protocol/config/source/data hash漂移时拒绝运行；
18. dev只能输出 `NOT_TESTED/PARTIAL`，不能生成 scientific GO。

正式产物至少包括：`protocol.json`、`environment.json`、`source_manifest.json`、`data_manifest.json`、`route_trace.jsonl`、`expert_surface_trials.csv`、`expert_surface.csv`、`arrival_manifest.json`、`job_trace.jsonl`、`action_trace.csv`、`power_trace.csv`、`accounting.csv`、`per_episode.csv`、`paired_bootstrap.json`、`decision.json`、`status.json`、`report.md`。
