# RouteSlack-MoE Gate 执行结果

> 执行日期：2026-07-28  
> canonical artifact：`artifacts/energy_slo_routeslack/20260728_115300/`  
> supporting audit bundle：`artifacts/energy_slo_routeslack/20260728_120340/`  
> manifest SHA-256：canonical `9c661c0bb90fbffd2cfc99b34d798feb04455cc9160e76bd9f610a57a94bde7c`；supporting `f70e0ba95811bec292f93c8d0cb50124ed8be875a248cd8a1aa6e8af818c9be1`  
> base commit：`26cc135f9ea3e4f2b778de38fae8f31666bf31bc`  
> 证据类型：CPU unit/integration + development capture + synthetic dry-run；`formal_result=false`

## 1. 环境与 GPU Gate 尝试

| 项 | 真实结果 | 标签 |
|---|---|---|
| 主机 | macOS 26.5.1, arm64 | `[Observed]` |
| 项目 Python / PyTorch | 3.9.6 / 2.8.0 | `[Observed]` |
| CUDA available / version / device count | `False` / `None` / `0` | `[Observed]` |
| `nvidia-smi` | command not found | `[Observed]` |
| formal service-curve probe | exit 1：`CUDA is mandatory; this benchmark never falls back to CPU` | `[Observed]` |
| formal route-capture probe | exit 1：`CUDA is required ... --allow-cpu is development-only` | `[Observed]` |
| physical latency / energy samples | 0 / 0 | `[Observed]` |

`[Blocked]` 当前机器没有 NVIDIA GPU/CUDA/NVML，正式 Experiment A–E 无法启动；两个入口都按协议 fail closed，没有生成伪 GPU artifact。即使换到 GPU，代码审查中 9 个开放 P0 也会阻止 formal Gate。

## 2. 测试结果

| suite | tests | 结果 | 实际验证内容 |
|---|---:|---|---|
| BCRD | 20 | PASS | core/replay invariants、cached decode、identity 和 provenance fail-closed |
| RouteSlack contracts | 31 | PASS | cache/identity/slack/surface/fallback/Oracle isolation/energy/ABBA/thermal/completed-token contracts |
| route-row contracts | 17 | PASS | development continuous harness 和 shared power accounting |
| JouleQueue | 28 | PASS | development capture/accounting helpers和 source-hash 回归 |
| **合计** | **96** | **PASS** | `[Observed]` CPU/合成协议正确性；不是 GPU 物理结果 |

完整 stdout/stderr：`artifacts/energy_slo_routeslack/20260728_115300/logs/unit_tests.log`。

## 3. Cached-decode development capture

- `[Observed]` tiny cached route-v2 capture 为 `1 request × 2 decode steps × 2 layers × top-2 = 8` 行。
- `[Observed]` CSV SHA-256 为 `8cecb415cdf991cc92fc051aefbf444b2442ca3ed87b25d0b987979c66fee8dc`；metadata SHA-256 为 `79551525d70386de465f560086d7a32372ae98a4a0da5ac92d4fbd63e6217e13`。
- `[Observed]` 独立 tiny-random OLMoE 单元测试执行 3 个 forced decode steps；每步 KV 长度 +1，cached logits 与 full-prefix recomputation 在 `rtol=1e-4, atol=1e-5` 内一致，EOS 不执行且 max-step 生效。
- `[Blocked]` capture metadata 明确为 `formal_eligible=false`、`scientific_result_eligible=false`；它不验证 natural continuous batching、两个冻结模型、GPU hook、EP 或 E2E SLO。

## 4. Synthetic dry-run

- `[Observed]` routed/dispatched/executed/combined 四个 stage 各有 16 个 synthetic contribution，identity conservation 通过。
- `[Observed]` 10 个 online baseline **名称**和 1 个 future-known Oracle 接口被 fixture 调用；Oracle/online 使用不同输入类型。
- `[Observed]` 这些调用只验证 registry、类型隔离和 artifact plumbing；并未实现或运行 10 个真实 baseline 算法。
- `[Observed]` 越界 surface 请求返回 `FALLBACK_DEFAULT` 且 `action_eligible=false`。
- `[Observed]` `physical_latency_samples=0`、`physical_energy_samples=0`、formal CI=`N/A`，最终 `Gate0=FAIL`、`formal_result=false`。

## 5. Host-only no-op 数字

每项为 25 个 timed outer trials，每 trial 2,000 次调用；95% CI 对 outer-trial paired mean increment 做 2,000 次 bootstrap。独立单位是 outer trial，不是 50,000 次 inner call。

| operation | P50 µs/call | P99 µs/call | paired mean Δ vs empty µs | 95% CI |
|---|---:|---:|---:|---:|
| empty loop | 0.029959 | 0.071612 | 0 | [0, 0] |
| instrumentation fixture | 0.042688 | 0.056042 | 0.011203 | [0.005994, 0.014534] |
| route-hook fixture | 0.064542 | 0.095774 | 0.033602 | [0.027668, 0.038906] |
| JSON logging fixture | 2.896584 | 3.293015 | 2.905731 | [2.855718, 2.965092] |
| decision framework | 0.116479 | 0.194960 | 0.095192 | [0.084725, 0.106695] |

`[Observed]` 这些是本机 CPU/Python fixture 的真实 timing。`[Blocked]` 它们不包含真实 router hook、CUDA/NVML、serving scheduler、GPU energy 或 E2E SLO，因此 GPU no-op tax、energy tax 和 `tax/gross saving` 都是 `N/A`。

raw timing：`artifacts/energy_slo_routeslack/20260728_115300/raw/noop_host_overhead.jsonl`。

## 6. 统计结果边界

```text
physical latency N = 0
physical energy N = 0
paired physical difference = N/A
formal 95% CI = N/A
missing physical samples = all requested GPU samples
filtered physical samples = 0
independent physical unit = N/A
```

`[Hypothesis]` H1、H2、H3 保持待检验；当前没有 effect size 可以与预注册 kill threshold 比较。`[Inferred]` 缺样本不是“收益为 0”，也不是 H1–H3 已被反证。

## 7. Gate table

```text
Gate 0: FAIL
Gate 1: FAIL
Gate 2: FAIL
Gate 3: FAIL
Gate 4: NOT RUN
```

| Gate | 依据 |
|---|---|
| Gate 0 | 9 个 P0 仍开放；natural continuous serving、同窗 CUDA/energy、thermal、双模型 exactness 和完整 E2E ledger 未闭合 |
| Gate 1 | Gate 0 顺序阻断；0 个 physical sample，0/2 frozen models 完成 |
| Gate 2 | 没有合格 Gate-1 energy surface/trace；synthetic Oracle 只是接口 fixture |
| Gate 3 | 10 个真实 baseline 算法未实现或物理运行；无 matched raw-energy Oracle |
| Gate 4 | Gate 0–3 未全部 PASS，按协议禁止 controller |

Gate 1–3 的 `FAIL` 表示未达到 PASS 条件且被 Gate 0 顺序阻断，不表示相应物理假设已被反证。

## 8. Simple baseline versus Oracle

```text
E_default = N/A
E_strongest_simple = N/A
E_oracle = N/A
CaptureRatio = N/A
95% CI = N/A
physical N = 0
```

dry-run 的 `synthetic_cost_units` 是固定测试常量，且每行 `scientific_result_eligible=false`。把它们代入 CaptureRatio 只会得到代码路径演示值，不是能耗测量，因此禁止报告。

## 9. 图表状态

`figures/README.md` 拒绝在 physical sample=0 时生成预注册的 12 张科学图。当前没有 rows×tier latency/energy surface、natural energy-mass heatmap、Pareto、baseline/Oracle、SLO–energy 或 thermal timeline；使用 synthetic fixture 制图会伪造物理证据。

## 10. Artifact 完整性

canonical artifact 保存 `manifest.json`、environment、config、commands、git diff、raw CSV/JSONL、processed summary、figures marker、完整测试日志和 verdict。`manifest.json` 记录 base commit、dirty status、seed、执行命令以及每个已纳入文件的 SHA-256。

supporting audit bundle 另保存五份报告的运行时快照、关键源文件和 `logs/gpu_gate_attempts.log`；它只补全 provenance，不替换 canonical artifact 中已报告的 host timing。

`[Inferred]` 当前唯一可复核结论是 measurement contract、development decode helper、fallback 和 fail-closed pipeline 可运行；没有证据授权 RouteSlack controller 或 `8xA100_CANDIDATE`。
