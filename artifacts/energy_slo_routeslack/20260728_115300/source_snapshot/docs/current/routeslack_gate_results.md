# RouteSlack-MoE Gate 结果

> 执行时间：2026-07-28  
> 主 artifact：`artifacts/energy_slo_routeslack/20260728_115300/`  
> 证据标签：本报告严格区分 `[Observed] / [Inferred] / [Hypothesis] / [Blocked]`。

## 1. 环境与可执行性

- `[Observed]` 当前节点为 macOS 26.5.1 arm64、Apple M5 Pro；`.venv` 为 Python 3.9.6、PyTorch 2.8.0。
- `[Observed]` `torch.version.cuda=None`、`torch.cuda.is_available()==False`、CUDA device count `0`；`nvidia-smi`、`nvcc`、`pynvml/nvidia-ml-py` 均不可用。
- `[Observed]` MPS 在该 venv 中也不可用；因此没有把 Apple GPU/CPU proxy 当 NVIDIA board-energy 证据。
- `[Observed]` 两个冻结模型 cache 可离线读取：OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，LLM-jp revision `1d5983076dfc67aee4a77ec06a27027f5bab6055`；“权重存在”不等于 Gate 可运行。
- `[Observed]` WikiText-103 `test` 按 document/min-chars 解析只有 61 个单位，默认 `--samples 128 --split test` 实测 fail closed；train 有分片 Arrow，但当前 sandbox 的 datasets lock 不可写且 fallback 尚不支持 shard concat。
- `[Blocked]` RTX 5090 远端实例已关闭；没有当前可用 host/凭据可取得 `nvidia-smi`、NVML 或运行 Experiments A–E。

环境原始记录见 [`environment.json`](../../artifacts/energy_slo_routeslack/20260728_115300/environment.json) 与 [`manifest.json`](../../artifacts/energy_slo_routeslack/20260728_115300/manifest.json)。

## 2. CPU unit tests

`[Observed]` 最终快照 **96/96 PASS，0 failed suite**：

| suite | tests | 结果 | 能证明什么 | 不能证明什么 |
|---|---:|---|---|---|
| BCRD | 20 | PASS | tiny cached/full logits、route-v2 closure、target replica 稳定、简化 replay legality | 两冻结 checkpoint、真实 serving/EP/energy |
| RouteSlack contracts | 31 | PASS | identity、slack、surface fallback、future isolation、repeat/counter/ABBA、dry-run fail-close | physical surface、真实 Oracle 收益 |
| Route-row contracts | 17 | PASS | continuous-decode ledger test double、power accounting 纯算术、sampler error/wrap | NVIDIA workload window 与 thermal gate |
| JouleQueue | 28 | PASS | 现有 CPU helper 与 source hash 主路径 | 旧 surface meter 的 scientific validity |

完整 stdout：[`logs/unit_tests.log`](../../artifacts/energy_slo_routeslack/20260728_115300/logs/unit_tests.log)。单测是代码证据，不是 Gate 1 数据。

## 3. Development-only cached decode

- `[Observed]` 随机初始化 tiny OLMoE 在 CPU 上完成 3 个 forced decode step；cached logits 与 full prefix recomputation 在 `rtol=1e-4, atol=1e-5` 内一致，每步 2 层×top-2，共 12 contribution。
- `[Observed]` 本地 tiny-Mixtral development capture 完成 1 request、2 decode steps、2 layers、top-2，共 **8 contribution**；CSV 为 route-v2，target replica 仍为 `-1`，arrival 来自 CLI 合成值。
- `[Observed]` 该 metadata 明确 `formal_eligible=false`、`scientific_result_eligible=false`，列出无 continuous batching、无正式双模型 exactness、无 dispatch/execute/combine/latency/energy ledger。

原始文件：[`development_tiny_cached_decode_v2.csv`](../../artifacts/energy_slo_routeslack/20260728_115300/raw/development_tiny_cached_decode_v2.csv) 与 [`meta.json`](../../artifacts/energy_slo_routeslack/20260728_115300/raw/development_tiny_cached_decode_v2.meta.json)。

## 4. Dry-run

`[Observed]` synthetic pipeline 完整运行：

- 每 stage 16 contribution；`routed/dispatched/executed/combined` 四个 stage 共保存 64 raw ledger rows；
- 10 个冻结 online baseline 接口与 1 个 future-known Oracle 接口均执行；
- unknown/out-of-range surface cell 返回 `FALLBACK_DEFAULT`，`action_eligible=false`；
- `physical_energy_samples=0`，`physical_latency_samples=0`，`confidence_intervals=null`；
- run status `DRY_RUN_COMPLETE`，但 `formal_result=false` 且 `gate0=FAIL`；没有部分成功被标为 formal。

数据见 [`processed/dry_run_summary.json`](../../artifacts/energy_slo_routeslack/20260728_115300/processed/dry_run_summary.json)、[`raw/contributions.jsonl`](../../artifacts/energy_slo_routeslack/20260728_115300/raw/contributions.jsonl) 和 [`raw/policy_results.jsonl`](../../artifacts/energy_slo_routeslack/20260728_115300/raw/policy_results.jsonl)。policy 中的值是明确标注的 `synthetic_cost_units`，不是 Joules。

## 5. No-op / host framework micro-cost

`[Observed]` 本机 CPU-only fixture 每项 25 个 outer trials、每 trial 2,000 次调用；表中是相对 empty loop 的 host overhead：

| 操作 | P50 增量 (µs/call) | paired mean 增量 (µs) | 95% CI (µs) |
|---|---:|---:|---:|
| instrumentation field read | 0.012729 | 0.011203 | [0.005994, 0.014534] |
| synthetic route-hook append/clear | 0.034584 | 0.033602 | [0.027668, 0.038906] |
| JSON logging | 2.866625 | 2.905731 | [2.855718, 2.965092] |
| online-interface decision call | 0.086521 | 0.095192 | [0.084725, 0.106695] |

Raw outer-trial rows：[`raw/noop_host_overhead.jsonl`](../../artifacts/energy_slo_routeslack/20260728_115300/raw/noop_host_overhead.jsonl)。

`[Inferred]` logging 明显比纯接口调用重；但这些数字不含 CUDA、真实 router hook、NVML、线程争用或 serving load，**不能**作为 controller tax，也不能与不存在的 gross energy saving 相除。

## 6. Physical experiments A–E

| Experiment | physical n | CI | 状态 | 原因 |
|---|---:|---|---|---|
| A rows×tier service-energy surface | 0 | N/A | `[Blocked]` | 无 NVIDIA GPU/NVML；旧 BCRD curve 无 energy/tier |
| B natural continuous-decode census | 0 | N/A | `[Blocked]` | 无 natural serving timeline；旧 census 有跨 step 因果错误 |
| C actionability | 0 | N/A | `[Blocked]` | 无 A/B 输入及真实 replica/power transition |
| D conservative Energy Oracle | 0 | N/A | `[Blocked]` | 旧 Oracle 是 latency replay，缺 energy/tax/DAG |
| E strong simple baselines | 0 | N/A | `[Blocked]` | 无合格 Oracle/physical metric/completion set |

因此没有生成 12 张 scientific figures。拒绝生成的说明保存在 [`figures/README.md`](../../artifacts/energy_slo_routeslack/20260728_115300/figures/README.md)；用 synthetic 数据画 surface/Pareto/capture ratio 会制造不存在的结果。

## 7. Gate table

```text
Gate 0: FAIL
Gate 1: FAIL
Gate 2: FAIL
Gate 3: FAIL
Gate 4: NOT RUN
```

- Gate 0 open items（formal 语义）：native natural continuous decode/serving backend、两模型 KV/exact qualification、latency window、energy window、thermal state。
- Gate 1–3 的 `FAIL` 表示“未取得 PASS 证据、不得晋级”，不是 H1/H2/H3 已被物理反证。
- Gate 4 按顺序没有运行，也没有 controller 实现。

## 8. Simple baseline versus Oracle

```text
E_default = NOT MEASURED
E_best_simple = NOT MEASURED
E_oracle = NOT MEASURED
CaptureRatio = NOT COMPUTABLE (0 physical samples; denominator absent)
```

`[Blocked]` 不能把 dry-run 的 arbitrary cost units 或 BCRD latency completion gain代入 energy CaptureRatio；报告伪数字会违反 matched SLO、matched completion set 与 raw-board-energy 分母。

## 9. 假设结果

| 假设 | Observed | Inferred | Hypothesis | Blocked |
|---|---|---|---|---|
| H1 energy variation | 0 formal energy sample | 现有 curve 无法控制 batch/KV/util/thermal | route residual 可能存在 | 两模型 natural surface |
| H2 actionability | 合法 action 接口可 dry-run | 单卡/回放不能证明 EP action 可执行 | actionable energy mass ≥20% | real replica/tier/sealing execution |
| H3 residual headroom | baseline/Oracle 接口隔离测试通过 | arbitrary smoke cost不能代表上限 | Oracle net ≥10%、simple capture <90% | conservative physical Oracle |

本轮没有把任何 `[Blocked]` 写成负结果，也没有把任何 CPU PASS 写成论文证据。
