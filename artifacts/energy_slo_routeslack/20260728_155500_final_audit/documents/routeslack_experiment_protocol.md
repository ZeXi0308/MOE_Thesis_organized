# RouteSlack-MoE 冻结实验协议

> 版本：RouteSlack v1 / 2026-07-28  
> 状态：`PREREGISTERED_BUT_BLOCKED_AT_GATE0`  
> 原则：先测量、再 Oracle、再强简单 baseline；Gate 0–3 全通过前不实现 controller。

## 1. 语义和 actuator 冻结

`[Observed]` 本协议只允许：

1. fixed-replica assignment；
2. bounded expert microbatch sealing；
3. 粗粒度 GPU power/clock tier；
4. dispatch ordering。

`[Observed]` 禁止改变 top-k、skip expert、FP8/INT4、低秩近似、权重、expert placement、admission、输出 token 数、语义、SLO 或完成集合。所有 arm 必须有相同 router/top-k/expert/weight/dtype/output hash，即 `Delta Q = 0`。

## 2. 实验对象与独立单位

| 项 | 冻结值 |
|---|---|
| 模型 | `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`；`llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055` |
| dtype | BF16 exact path |
| workload | natural document/request trace；每模型/负载 cell 至少 128 个独立 input event；数据文件存 SHA-256 |
| synthetic skew | 仅 sensitivity，不得用于主判定 |
| independent unit | surface：fresh input event/document；serving：request/input event；thermal：AB/BA block |
| 非独立单位 | inner repeat、同 activation 的重复 kernel、token sibling、同 request 内层 |
| split | 按 document/request cluster 一次性分到 calibration/evaluation；禁止同 request 跨 split |
| SLO | 在独立 calibration 中以 unmodified default 的 P99 × 1.10 生成每模型/负载 deadline，写入 config 后冻结 |
| admission/output | 同 arrival trace、同 admission、同每请求 output token IDs |
| random seed | `20260728`；bootstrap seed 从此 seed 和 cell ID 确定派生 |

## 3. 必须保存的 identity ledger

每个 top-k contribution 至少包含：

```text
request_id, input_event_id, token_id, decode_step,
layer_id, expert_id, topk_slot, source_rank, target_replica
```

其中 `target_replica` 是 actuator，不进入不可变 mathematical contribution key；但 dispatch 之后必须在 dispatched/executed/combined 三个 ledger 中一致。强制断言：

```text
routed immutable contribution multiset
= dispatched immutable contribution multiset
= executed immutable contribution multiset
= combined immutable contribution multiset
```

任一 duplicate、drop、extra、wrong target、top-k sibling 不闭合或不同 completed identity set 都使该 pair `INVALID`。

## 4. Gate 0：代码与测量协议

### 4.1 Continuous decode

- prefill 只执行一次，`use_cache=True`；
- 每个 decode step 只输入 1 个新 token，传入独立 `past_key_values`；
- attention mask 和 position IDs 每步增长；KV length 每步恰好 +1；
- router hook 按 request/decode-step/layer 闭合；每层 top-k cardinality 恒定；
- EOS 不执行；max output length 强制生效；
- 同 token prefix 的 cached logits 与 full recomputation 在 `rtol=1e-4, atol=1e-5` 内一致；
- patched/unpatched final output hash 一致。

batch-size-1 的 autoregressive loop 仅是 development prerequisite；formal producer 还必须有 mutable active set、natural arrival、真实 per-request KV ownership 和 serving timeline。

### 4.2 Latency window

每个 trial 同时保留：

- host `monotonic_ns` wall-clock；
- 正确 CUDA stream 上的 start/end events；
- enqueue、seal、dispatch、A2A/send、expert start/end、combine/return、first output、completion timeline；
- request TTFT/TPOT/TBT/P50/P95/P99；
- 任何被排除的 stage 都不得称为 E2E。

仅在 trial 边界允许 synchronize；不允许在每个 kernel 后全局 synchronize 改变执行。

### 4.3 Energy window

- 主来源：GPU cumulative total-energy counter；只有不支持时才用 `monotonic_ns` 高频 power integration；
- sampling target ≤10 ms，observed max gap 不得超过 20 ms；任何采样线程/NVML 异常必须同步传播并使 trial 失效；
- CUDA workload start/end、counter read、boundary power sample 都保存原始时间戳；不得把 sequential bracketing 声称为 atomic alignment；
- 每 arm 的实际 workload window 至少 10 s 且至少 100 个 completed output token；
- A/B 使用相同 inner repeat；固定 meter overhead 作为单独校准量保存，raw board energy 不被隐式改写；
- counter wraparound 只在硬件明确提供 modulus 时处理，否则 fail closed；
- 主指标：`raw board J / matched SLO-completed output token`；
- 辅指标：idle-adjusted dynamic J/token，必须同时报告 idle baseline 敏感性。

### 4.4 Thermal/power state

每个 sample 保存 timestamp、temperature、graphics clock、memory clock、power limit、power draw、utilization、throttling reason、GPU UUID。协议冻结：

- 每个 workload/tier 先 warmup 至少 60 s，且最后 30 s 温差 ≤2°C；
- 使用 A/B/B/A 和 B/A/A/B 交叉 block；
- pair 内 arm 开始温度差 ≤2°C；graphics clock 中位数差 ≤1%；memory clock/power limit 与目标 tier 一致；
- 出现新 throttling reason、后台 GPU workload 或 UUID 不一致时丢弃整个 pair；
- 报告丢弃数量和原因，不得只删除负收益样本。

### 4.5 Gate-0 PASS 规则

PASS 必须由 raw artifact/hash 导出，不接受 caller 自报 boolean。上述 decode、identity、latency、energy、thermal、matched completion、exactness、Oracle isolation 全部 PASS，且 P0=0，才能进入 GPU formal Gate 1。

## 5. Gate 1：物理现象

### Experiment A：rows × power/clock tier surface

- 两模型分别执行；不池化判定；
- 选择多 layer、多 expert、至少 30 个 fresh input event/cell；
- rows 覆盖 natural route P1/P5/P25/P50/P75/P95/P99 及边界；
- tier 至少为 default 和两个硬件合法的 coarse tier，记录实际 clock/power-limit realization；
- 输出 latency/row、raw Joules/row、Joules/token、throughput/W、paired 95% CI、layer/expert/input-event 方差。

### Experiment B：natural continuous-decode census

按 layer/step 统计 expert rows、activated experts、expert union、route entropy、真实 replica fragmentation、virtual-rank slack、common cells，并报告每 cell 的 token/time/raw-energy mass。主结论使用 natural workload；synthetic 仅敏感性。

### Gate-1 阈值

两模型 AND：至少一个两模型共同的 natural cell 在各模型中 effect ≥10%、95% LCB >5%；至少一个共同 cell 在各模型中 effect ≥15%；各模型 actionable natural energy mass ≥20%。任一模型所有 common cell 的 effect 都 <5%、actionability <20%、收益只在 synthetic 出现，或 accounting/thermal 不闭合即 KILL；effect 落在 5%–10% 且未达到 PASS 时只保留 measurement characterization。

## 6. Gate 2：Conservative Oracle

Oracle 仅使用冻结 actuator，但可看到小窗口未来 route/arrival/service realization。在类型上与 `OnlineObservation` 隔离。目标是在相同 matched completion set 上最小化 raw board energy，必须计入：

```text
queueing + expert latency + dispatch/combine/return
+ sealing wait + power switching latency/energy
+ decision/controller tax + board idle energy
```

输出 gross saving、switching tax、holding tax、controller tax、net saving、SLO violation、raw J/SLO-completed-token、unsupported-cell mass。两模型任一 net saving <10% 即 KILL，不得事后修改阈值。

## 7. Gate 3：强简单 baseline

必须同 trace/SLO/admission/output/completion set 比较：

1. immediate execution；
2. fixed row threshold；
3. fixed timeout；
4. earliest deadline first；
5. least-loaded replica；
6. min-predicted-finish；
7. LPLB-like token balancing；
8. two-tier static power；
9. min-finish + two-tier power；
10. route-unaware batch/KV/phase energy controller；
11. future-known Oracle。

```text
CaptureRatio = (E_default - E_strongest_simple)
             / (E_default - E_oracle)
```

若任一模型的最强 simple capture ≥90%，取消复杂 controller。分母 ≤0 时 capture ratio 无定义，直接 KILL。

## 8. Gate 4：Controller feasibility

只有 Gate 0–3 PASS 才实现最小 controller。测量 decision P50/P99、telemetry/hook/logging/surface lookup、state freshness、transition latency、hysteresis、dwell、fallback 和 out-of-range behavior。`controller tax / gross saving >20%` 即 KILL 或降为 measurement-only。

## 9. 统计方法

- 按 input event/document 做 paired/hierarchical bootstrap，2,000 replicates；
- 每模型分开报告 mean、median、P95/P99、95% CI、paired difference、N、missing/filtered 数量及原因；
- inner repeat 仅用于扩大 meter window，不进入样本数；
- 不删除负收益 cell，不只报告最佳运行，不用池化遮蔽单模型失败。

## 10. 图表和 artifact

正式运行必须由 raw artifact 自动生成 12 图：latency surface、J/row surface、natural energy-mass coverage、fragmentation、rank slack、Pareto frontier、Oracle/baseline saving、capture ratio、SLO–energy curve、calibration error、no-op/controller overhead、thermal/clock/power timeline。

本轮正式策略 latency/energy 样本为 0，因此不制造 synthetic “科学图”；`figures/README.md` 明确记录该阻塞。RTX 5090 development bundle 中的双模型 batch-1 route trace 和 NVML capability window 只用于 Gate-0 兼容性诊断，不进入任何 surface、CI、Oracle 或 Gate-1 图。每次运行保存 manifest、environment、config、commands、git commit/diff、raw、processed、figures、logs、verdict 和全文件 SHA-256。

`[Observed]` `artifacts/energy_slo_routeslack_gpu/20260728_144600/` 是按上述边界密封的补充资格 bundle：两冻结模型的 batch-1、4-step exactness 全通过，NVML counter/UUID/采样 gap 能力通过；但 ΔT=23°C 使 thermal pair 明确失败。该负结果不会放宽 60 s warmup、最后 30 s ΔT≤2°C、pair 起始温差≤2°C 或 ≥10 s/≥100 completed-token 的冻结条件，也不授权 Experiment A–E。

## 11. 已执行的 GPU development probe 与正式阻塞

```bash
# 已在隔离 RTX 5090 上离线执行；每个模型仅 1 prompt × 2 decode steps。
python3 docs/ideas/bcrd/experiments/capture_native_routes.py \
  --model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M \
  --model-key llmjp \
  --model-revision 1d5983076dfc67aee4a77ec06a27027f5bab6055 \
  --dataset builtin --samples 1 --seq-len 16 --decode-steps 2 \
  --dtype bfloat16 --phase decode --offline \
  --output <NEW_RUN>/llmjp/routes.csv

python3 docs/ideas/bcrd/experiments/capture_native_routes.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-key olmoe \
  --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --dataset builtin --samples 1 --seq-len 16 --decode-steps 2 \
  --dtype bfloat16 --phase decode --offline \
  --output <NEW_RUN>/olmoe/routes.csv

# 现有 benchmark 缺 energy/tier/thermal，即使有 GPU 也不是 formal Gate-1
python3 docs/ideas/bcrd/experiments/benchmark_expert_service_curve.py \
  --model <PINNED_MODEL> --model-key <MODEL_KEY> --output <NEW_CURVE.csv>
```

`[Observed]` 结果为 LLM-jp 512 rows、OLMoE 256 rows，恰好等于 `steps×layers×top-k`；所有 validation check PASS，metadata 仍为 non-formal。完整 artifact：`artifacts/energy_slo_routeslack/20260728_144614_gpu_dev/`。

`[Observed]` `run_model_patch_parity_probe.py` 已在两冻结 revision 上执行：同一 prompt/forced token 下的 prefill/2-step logits、KL、route ID、route weight error 全为 0，KV length 均为 `[17,18]`。该结果仅关闭 batch-1 patch exactness 子项，artifact 始终 `formal_eligible=false`。

`[Observed]` 本轮还新增 `run_rtx5090_development_probe.py`，以 synthetic BF16 equal-work jobs 验证同一次 execution 的 CUDA event/raw-board-energy 包围、统一 repeat、ABBA、UUID 和 thermal telemetry；mock/unit 9/9 PASS。第一次 GPU 启动检测到外部进程占用约 14.5 GB 显存，按设计在 0 个 window 处 `DEVELOPMENT_PROBE_FAILED_CLOSED`；该失败不能解释为能耗结果。

`[Observed]` 在随后的新目录中，GPU/NVML meter preflight 五项检查全部 PASS，两冻结 revision 的 model-patch parity 也均为 `DEVELOPMENT_PARITY_PASS`。synthetic ABBA 又独立尝试两次，每次都完成 12 windows，但结束检查分别发现竞争 CUDA PID 8042 和 8240，整次 run 因而 fail closed。这 24 个已写入 window 是被拒绝的调试记录，不进入 effect、CI 或任何 Gate。latest bundle 为 `artifacts/energy_slo_routeslack/20260728_151500_rtx5090_dev/`。

`[Observed]` `run_rtx5090_energy_characterization.py` 还按 current validator 执行了两模型 default-tier isolated-expert characterization：每模型 rows 1/8/32/128 × 4 个 request-disjoint outer trial，每窗目标至少 10 s，采样目标 5 ms、最大 gap 20 ms、窗内最大温差 2°C。LLM-jp 有 11/16 窗有效，状态 `CHARACTERIZATION_COMPLETE_WITH_FILTERED_WINDOWS`；OLMoE 仅 4/16 有效且 rows=128 为 0，状态 `CHARACTERIZATION_INCOMPLETE_INVALID_WINDOWS`。artifact 为 `artifacts/energy_slo_routeslack/20260728_154500_current_energy_characterization/`。

`[Blocked]` 该 runner 明确使用 calibration prefill activation、单 expert、provider-default 575 W、expert-row denominator，没有 natural cached decode、power tier actuator、matched completed-token、EP 或 A/B policy。因此它只新增 development physical windows，不改变 formal strategy latency/energy N=0，也不能开启 Gate 1。

`[Observed]` 独立 follow-up `artifacts/energy_slo_routeslack/20260728_150422_gpu_followup/` 也给出同一边界：一次启动前竞争检测在 0 window 拒绝，重试虽完成 12 windows，但结束时检测到约 14.2 GB 的外部 CUDA workload，故整次拒绝。其 19 个声明文件下载后逐一复算无缺失或 hash 漂移；该负结果只验证 fail-closed，不增加 energy N。

`[Blocked]` 仓库仍无符合本协议的双模型 natural continuous-serving + rows×tier raw-energy formal runner；因此不存在可诚实列出的一键 Experiment A–E 命令。扩大 batch-1 capture 或运行 latency-only benchmark不会改变这个结论。
