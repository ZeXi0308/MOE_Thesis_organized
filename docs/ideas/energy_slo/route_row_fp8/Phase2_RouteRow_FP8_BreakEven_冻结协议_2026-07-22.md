# Route-row FP8 Break-even Energy-SLO Phase 2 冻结协议

状态：**PHASE2_FROZEN / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-22  
范围：单 GPU、真实 KV decode、真实 continuous serving engine、GPU board energy。**不外推 CPU/NIC/RDMA/多节点 EP。**

任何实现若退回 full-forward、isolated pre-cast GEMM 或分析模拟，只能标 `PROXY/BLOCKED`，不得生成正式结果。sealed 后修改协议必须使旧产物 `SUPERSEDED`。

## 1. 冻结主张

> 在真实 continuous-batched MoE decode 中，当前层 router 已产生的每专家 token-row 数 `m[l,e,t]` 可决定完整 FP8 expert-MLP 路径是否真正节能；仅在节能置信下界为正、延迟不劣于 BF16 且质量地板满足时选择 FP8。检验其相对最强静态精度与 routing-blind 联合策略，能否在 matched P99 SLO 下减少至少 10% GPU J/completed-output-token。

只动态控制 decode 阶段 routed expert FFN；prefill、attention、router、shared expert 和 LM head 保持 BF16。不得声称“首个动态 MoE 精度控制器”。

## 2. 假设与 existence stop

- **H-Existence**：真实 decode workload 中同时存在 populated FP8-positive 与 BF16-fallback row-count 区域。
- **H-RouteValue**：读取实际 row count 的策略优于只读 active batch、queue 和 slack 的 routing-blind 策略。
- **H-Interaction**：route-row precision gate 与 batching 联合后严格优于 batching-only 和 precision-only。
- **H-Main**：相对所有质量/SLO 合格强基线，J/token 改善 95% CI 下界 ≥10%，P99 TTFT/TPOT ratio 的 95% CI 上界 ≤1.03。

正式 sealed 前必须通过：

1. 至少一个 FP8-safe bin 和一个 BF16-fallback bin；
2. 两类区域分别覆盖 calibration 中至少 10% 的 BF16 expert-energy mass；
3. calibration 上候选相对 `B1/B2/B3/B4` 最强者达到原 10% 门；
4. routing-blind 若覆盖候选，直接科学 No-Go；
5. quality calibration 失败，直接 No-Go。

若 continuous engine、FP8 kernel 或计量实现失败，状态是 `BLOCKED`，不是科学 No-Go，也不得以旧微基准替代。

## 3. 因果动作与 row bins

\[
m_{l,e,t}=\#\{(request,token,topk\ slot):selected\_expert=e\}.
\]

`m` 只能在当前层 router/top-k 完成后、当前层 expert GEMM 前读取；不得读取后续层、未来 token 或未来 arrival。

固定 bins：

```text
1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, >=129
```

每模型分别标定完整 expert MLP：

\[
\Delta E(m)=E_{BF16}(m)-E_{FP8-full}(m),
\]

\[
\Delta T(m)=T_{FP8-full}(m)-T_{BF16}(m).
\]

动作固定为：

```text
FP8 iff:
  bin 样本充分
  AND LCB95(delta_energy) > 0
  AND UCB95(delta_latency) <= 0
  AND calibration in-loop quality gate 通过
else BF16
```

缺失/样本不足/CI 跨零一律 BF16；`m=0` 不启动 expert kernel；同一次 expert 调用的 gate/up/down 或 w1/w3/w2 必须同精度。

LUT key 至少包含 `(model_revision,GPU_UUID,FP8_recipe,expert_MLP_shape,row_bin)`，禁止按 `expert_id` 或跨模型复用。

## 4. 完整 FP8 路径与权重驻留

固定 recipe：E4M3FN；weight per-tensor absmax scale、初始化时一次 cast；activation 每次非空 expert 调用执行 float/amax/scale/cast；`torch._scaled_mm` 输出 BF16；三层 expert projection 全覆盖。

完整账本包含：activation quantize/amax/scale/cast、三次 GEMM、SiLU/mul、输出转换、mixed-precision 分组与额外 launch、policy lookup、route histogram、scheduler、attention、KV、prefill、queueing 和 GPU idle。

本轮冻结为**双驻留**：BF16 原权重与 FP8 expert 权重/scales 在正式计量前全部创建；所有 arm 保持相同驻留状态。记录 BF16/FP8/scales bytes、峰值 HBM 和由此损失的 batch/KV capacity。测量窗口内禁止 weight cast/allocation；超过容量 hard-fail。

权重初始化时间可排除，但双份驻留容量不能排除。

## 5. 模型与数据

模型 revision 固定：

- `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5`；
- `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055`。

数据：`wikitext-103-raw-v1:test`，不 shuffle。

- calibration candidate window：raw articles `[1024:1536)`，筛选两个 tokenizer 编码后长度均 ≥129 的前 128 篇；
- sealed candidate window：`[2048:3072)`，同规则前 256 篇；
- prompt=64 tokens；teacher-forced decode=64 steps；
- Phase 3 先写 canonical text SHA-256 manifest；calibration/test hash 必须不重叠；窗口不足不得静默换 offset。

旧文档“WikiText-103 从未使用”的注释已过时，不能作为隔离证明。

## 6. Continuous serving engine

正式实现必须满足：

- 每请求只 prefill 一次；
- decode forward 输入长度恒为 1，并复用该 policy 自己的 KV；
- arrival 在 iteration boundary 加入，完成请求被移除；
- active batch 在 trace 中实际增减；
- 禁止每步重跑 prefix；
- 禁止每步 O(KV-size) merge/split cache；
- 每个 arm 独立产生 KV、hidden state 和后续 routes；禁止 BF16 route replay/`lock_routing`。

推荐固定可审查的 serving engine/commit。若 vLLM 或等价引擎无法在相同 hot path 支持 OLMoE、Mixtral 和 per-expert BF16/FP8，则 Phase 3 必须标 `BLOCKED`，不可降级为 Transformers full-forward 或 `run_serving_sim.py`。

### Arrival

calibration 用 `max_num_seqs=1, always-BF16` 测参考容量 `mu`。

- 主 Poisson：`lambda=0.7mu`；
- 主 MMPP：`lambda_low=0.2mu`、`lambda_high=1.2mu`，平均状态驻留 2s，长期均值 0.7mu；
- 敏感性：Poisson 0.4mu、0.9mu；
- 每 trace 32 warm-up requests，随后 256 measured requests，最后完整 drain；
- seeds `2026072201..2026072205`；仅 CI 跨门时允许追加预留 `2026072206..2026072210`。

## 7. 强基线与 batch grid

所有 arm 使用同一 engine、arrival、request、FP8 recipe 和计量窗口：

| arm | 定义 |
|---|---|
| `B0` | `max_num_seqs=1 + always-BF16` |
| `B1` | max-feasible continuous batching + always-BF16 |
| `B2` | max-feasible continuous batching + always-FP8 |
| `B3` | routing-blind joint lookup；用 `active_tokens×top_k/num_experts` 的期望行数替代实际 `m`，其余 LUT/校准预算相同 |
| `B4` | route-row gate + `max_num_seqs=1`，precision-only |
| `C` | route-row gate + max-feasible continuous batching |

batch grid：`max_num_seqs ∈ {1,4,8,16,32,64}`。每种 policy 仅在 calibration 选择满足 SLO 的最大可行值；另报 common-cap sensitivity。候选因额外权重/KV 导致的容量下降不得隐藏。

## 8. 事件、SLO、质量与能量账本

每个 serving event 至少包含：

```text
request_id, token_id, phase, arrival_ns, enqueue_ns, batch_id,
batch_seal_ns, gpu_start_ns, gpu_end_ns, first_output_ns,
completion_ns, precision_action, completed, slo_deadline_ns
```

Calibration 用 B0 定义：

\[
SLO_{TPOT}=1.10\,P99_{TPOT}(B0),\quad
SLO_{TTFT}=1.10\,P99_{TTFT}(B0).
\]

TTFT=`arrival→first_output`；request TPOT=`(completion-first_output)/(output_tokens-1)`；P99 在 request 级计算。另报 token-level P99 TBT、violation rate、drop/backlog。

主能耗：

\[
E_{total}=\int_{t_0}^{t_1}P_{GPU}(t)dt,
\quad
J/token=E_{total}/N_{completed-output-token}.
\]

`t0` 为首个 measured arrival 释放前同步点，`t1` 为最后 request 完成并 GPU synchronize 后。必须完整 drain、零 drop、各 arm 输出 token 数相同。未完成 token 不进入分母。

若 NVML total-energy counter 可用，以 counter delta 为主；否则使用 monotonic timestamp、≤20ms sampling 并显式补边界点。主口径包含 idle。辅助口径：

\[
E_{dynamic}=\int\max(P_{GPU}-P_{idle},0)dt.
\]

`P_idle` 在相同时钟、power cap、weight residency、KV 状态下独立测量。NVML 只能称 GPU board energy。

质量：每个 policy 独立 fresh prefill/KV/in-loop decode，保留 deeper-layer route divergence；文档级 paired bootstrap；主门为 `mean token KL(reference||candidate)` 95% CI 上界 `<0.05`。PPL、delta NLL 和 MMLU next-token accuracy 仅为敏感性。

## 9. Go / No-Go

正式 GO 要求两个模型 × Poisson/MMPP 四个主 cell 全部满足：

1. 质量门通过；
2. 相对每个质量/SLO 合格强基线，GPU J/token 节能 95% CI 下界 ≥10%；
3. P99 TPOT 和 P99 TTFT ratio 的 95% CI 上界 ≤1.03；
4. SLO violation rate 增量 95% CI 上界 ≤1 percentage point；
5. 零 drop、零提前终止；
6. routing-blind baseline 未覆盖候选。

实现正确但无 crossover/headroom：科学 No-Go。CI 跨门只允许五个预留 seeds；仍跨则 `INCONCLUSIVE/STOP`。不得通过改 bins、加 feature、DVFS、bandit、RDMA 或放宽 10%/3% 抢救。

## 10. Phase 3 验收用例

至少覆盖：

1. `sum_e m[l,e,t] == active_decode_tokens × top_k`；
2. row signal 位于 router 后、expert 前，policy API 不含未来信息；
3. missing/underpowered bin 回退 BF16；
4. OLMoE target linears=`16×64×3=3072`，LLM-jp=`16×32×3=1536`；
5. FP8 arm 的 scaled-mm/activation-cast counters >0，BF16 arm为0；
6. 空 expert 不执行 amax/GEMM；
7. 每次非空 FP8 expert 调用都 activation cast，weight cast 仅初始化一次；
8. 每 policy KV/cache 独立；禁用 route replay；
9. 每 request prefill count=1、decode input length=1、KV 单调增长、active batch实际增减；
10. 无 O(KV) cache repack；
11. arrival/config/document hash 在 arm 间一致；
12. synthetic energy 100W×10s/100 completed tokens=10J/token；idle30W 时 dynamic=7J/token；
13. 未完成 request 进入 backlog/右删失但不进入完成分母；
14. NVML UUID 与 CUDA device 一致；
15. dual-weight bytes/scales/峰值 HBM/可行 batch cap 闭合；
16. formal runner 缺 Phase 4 `SIGNED-OFF` 或 code/config hash 漂移时拒绝运行；
17. 缺任一强基线产物时禁止生成 GO verdict。

建议 Phase 3 文件：`route_row_policy.py`、`power_accounting.py`、`continuous_decode_harness.py`、`run_route_row_surface.py`、`run_route_row_energy_slo.py`、`test_route_row_*.py`，冻结配置写入 `experiments/configs/route_row_break_even_v1.yaml`。

