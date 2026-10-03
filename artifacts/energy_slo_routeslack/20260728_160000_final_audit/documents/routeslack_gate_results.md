# RouteSlack-MoE Gate 执行结果

> 执行日期：2026-07-28  
> canonical artifact：`artifacts/energy_slo_routeslack/20260728_115300/`  
> supporting audit bundle：`artifacts/energy_slo_routeslack/20260728_120340/`  
> final supporting dry-run：`artifacts/energy_slo_routeslack/20260728_151412_final_support/`  
> final current audit bundle：`artifacts/energy_slo_routeslack/20260728_155500_final_audit/`  
> RTX 5090 development bundle：`artifacts/energy_slo_routeslack/20260728_144614_gpu_dev/`  
> RTX 5090 parity/failed-energy follow-up：`artifacts/energy_slo_routeslack/20260728_150422_gpu_followup/`  
> latest RTX 5090 development bundle：`artifacts/energy_slo_routeslack/20260728_151500_rtx5090_dev/`  
> current-validator energy characterization：`artifacts/energy_slo_routeslack/20260728_154500_current_energy_characterization/`  
> repeated physical characterization：`artifacts/energy_slo_routeslack/20260728_153000_rtx5090_physical_dev/`  
> sealed GPU qualification：`artifacts/energy_slo_routeslack_gpu/20260728_144600/`  
> manifest SHA-256：canonical `9c661c0bb90fbffd2cfc99b34d798feb04455cc9160e76bd9f610a57a94bde7c`；supporting `f70e0ba95811bec292f93c8d0cb50124ed8be875a248cd8a1aa6e8af818c9be1`；GPU dev `491faee358570fcba18e1564c5dbde57695e09c1309acc5409a46ae536816480`  
> latest GPU dev manifest SHA-256：`d23f18a0455c4ebd04f3611746bc8150eac674ba1bd97e91d24ee5e7517f05c8`  
> current energy manifests：LLM-jp `152317db94a008df4f51a9c87cae3faccd3fcf1ab96c98056f1be42daf758716`；OLMoE `b2550018de3b8d481cf13d6039715affb4e0c4cb1ed0ba60224d102f7bff8a1c`  
> repeated physical manifest SHA-256：`7727e487b3035122e763b519b493af9aa081d4dfbf0d7ad7144f91708ead4ef9`  
> final current audit manifest SHA-256：`5760d1c8e24aee6986647468c50d8b186de310ade616c6a6820d052336a3531d`  
> follow-up artifact manifest SHA-256：`e588eafb78926fe35abdfa15727d414d50498e4fe0ba04568abf34621bbf654e`  
> final supporting manifest SHA-256：`b3e51f274f34148f477a1f6392dc7eb5620f733ba054faea97c690e84d501da1`  
> sealed GPU qualification manifest SHA-256：`69e6303bfaadaec93bfa6f15fee0abe154325c0ce097e8d58914d9fc197a0f37`  
> base commit：`26cc135f9ea3e4f2b778de38fae8f31666bf31bc`  
> 证据类型：CPU unit/integration + development capture + synthetic dry-run；`formal_result=false`

## 1. 环境与 GPU Gate 尝试

| 项 | 真实结果 | 标签 |
|---|---|---|
| 本地主机 | macOS 26.5.1, arm64；Python 3.9.6 / PyTorch 2.8.0；CUDA device 0 | `[Observed]` |
| 隔离 GPU 主机 | Linux x86_64；RTX 5090 32,607 MiB；UUID `GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc` | `[Observed]` |
| GPU 软件栈 | driver 595.71.05；Python 3.12.3；PyTorch 2.8.0+cu128；Transformers 4.57.6；capability 12.0 | `[Observed]` |
| 冻结模型 revision | LLM-jp `1d5983…bab6055`；OLMoE `6d84c4…ade9c5`；均 offline cache 命中 | `[Observed]` |
| development route capture | LLM-jp 512 contributions；OLMoE 256；所有 validation check PASS；metadata non-formal | `[Observed]` |
| 双模型 native/patch parity | 16-token follow-up 的 prefill + forced 2-step：logit/KL/route-ID/route-weight max error 均为 0；KV `[17,18]` | `[Observed]` |
| NVML capability | total-energy counter supported；200 samples；requested 5 ms；observed gap median 5.090 ms / max 12.024 ms | `[Observed]` |
| latest meter preflight | CUDA/NVML UUID match、energy counter、telemetry 和采样 gap 五项 PASS；`formal_result=false` | `[Observed]` |
| fixed-revision model patch parity | OLMoE 与 LLM-jp 均为 `DEVELOPMENT_PARITY_PASS`；prefill + 2 decode steps 的 logits/KL/route-weight 误差均为 0，selected experts 相等 | `[Observed]` |
| synthetic ABBA energy attempts | 2 次各完成 12 windows，均因结束时出现竞争 CUDA 进程而 `DEVELOPMENT_PROBE_FAILED_CLOSED`；可接受样本 0 | `[Observed]` |
| isolated-expert development energy | LLM-jp 首跑 11/16、同协议复跑 10/16 valid；OLMoE 4/16 valid，rows=128 无合格窗；均为 default tier、prefill activation、expert-row denominator | `[Observed]` |
| formal strategy latency / energy samples | 0 / 0 | `[Observed]` |

最新 sealed qualification 的独立复核结果：96/96 protocol-critical tests PASS；LLM-jp/OLMoE 的 4-step batch-1 cached-decode exactness 均为零误差且 argmax/KV length 一致；development capture 分别为 2,048/1,024 contributions。NVML probe 记录 299 点、max gap 10.925 ms、counter Δ=1,724.861 J，但 ΔT=23°C，故 formal thermal check 为 FAIL。以上数字不进入 physical model energy denominator，该样本数仍为 0。

`[Observed]` GPU 可用性阻塞已经解除，但 Gate 0 没有因此自动通过。代码审查中 9 个开放 P0 仍阻止 formal Experiment A–E：当前已有 batch-1 route compatibility、双模型 development patch parity 与 meter capability，但没有 natural continuous serving、可接受的同窗 service-energy/matched completion、thermal pair、真实 actuator/EP、Oracle 或强 baseline。

## 2. 测试结果

| suite | tests | 结果 | 实际验证内容 |
|---|---:|---|---|
| BCRD | 20 | PASS | core/replay invariants、cached decode、identity 和 provenance fail-closed |
| RouteSlack contracts/runners | 59 | PASS | 原 contracts + RTX 5090 measurement-path、model parity、energy-characterization duration/status fail-close 测试 |
| route-row contracts | 17 | PASS | development continuous harness 和 shared power accounting |
| JouleQueue | 28 | PASS | development capture/accounting helpers和 source-hash 回归 |
| **当前合计** | **124** | **PASS** | `[Observed]` CPU/合成协议正确性；不是 124 个 GPU 或科学样本 |

原 96-test stdout/stderr：`artifacts/energy_slo_routeslack/20260728_115300/logs/unit_tests.log`。GPU dev snapshot 另在远端执行 BCRD 20/20 与当时 RouteSlack 31/31，日志已下载；current runner 的本地 RouteSlack 59/59 与总计 124/124 已在 `20260728_153700_final_audit` 中重新固化。

## 3. Cached-decode development capture

- `[Observed]` tiny cached route-v2 capture 为 `1 request × 2 decode steps × 2 layers × top-2 = 8` 行。
- `[Observed]` CSV SHA-256 为 `8cecb415cdf991cc92fc051aefbf444b2442ca3ed87b25d0b987979c66fee8dc`；metadata SHA-256 为 `79551525d70386de465f560086d7a32372ae98a4a0da5ac92d4fbd63e6217e13`。
- `[Observed]` 独立 tiny-random OLMoE 单元测试执行 3 个 forced decode steps；每步 KV 长度 +1，cached logits 与 full-prefix recomputation 在 `rtol=1e-4, atol=1e-5` 内一致，EOS 不执行且 max-step 生效。
- `[Observed]` RTX 5090 上的 LLM-jp frozen revision 完成 2 steps、16 layers、top-16，共 512 contributions；CSV SHA-256 `eff7a374d898fbec0243a26478f5f50af769dde213f6b19f309cfbf2989aaa7e`。
- `[Observed]` OLMoE frozen revision 完成 2 steps、16 layers、top-8，共 256 contributions；CSV SHA-256 `8d13ba97f781399f3f78e94c5d114f724949adca6e904ee6c810e4ef0cf8f86d`。
- `[Observed]` 两模型均精确闭合 decode steps `[0,1]`、每步 16 层、每 input-event/layer 的 top-k、单 token identity、revision 和 artifact hash。首次 LLM-jp 尝试因最小同步包遗漏 `creditreduce_reference` 静态依赖失败；失败日志保留，补齐依赖后同命令成功。
- `[Observed]` 双模型 parity follow-up 进一步比较 native decoder/lm-head 与 shared `full` patch：LLM-jp 和 OLMoE 的 prefill、两个 cached step 的 max absolute logit error、max KL、selected expert mismatch 和 route-weight error均为 0；本次长度 4 prompt 的两臂 KV length 都是 `[5,6]`，layer/top-k 跨 prefill/decode 闭合。
- `[Blocked]` GPU capture/parity metadata 仍明确为 `formal_eligible=false`、`scientific_result_eligible=false`；它不验证 natural continuous batching、ready/dispatch/combine timeline、EP、matched completion 或 E2E SLO。

## 3A. Current-validator isolated-expert characterization

所有数值均为 development-only、provider-default 575 W、单 expert BF16；independent unit 是 non-overlapping captured request activation group，inner repeat 不计样本。95% CI 是对有效 outer trial 的 2,000 次 bootstrap；`N=1` 的退化 CI 不构成统计证据。

| 模型 | rows | valid N/4 | CUDA us/logical batch mean [95% CI] | raw J/expert-row mean [95% CI] |
|---|---:|---:|---:|---:|
| LLM-jp | 1 | 3/4 | 55.148 [53.796, 57.040] | 0.008206 [0.008120, 0.008355] |
| LLM-jp | 8 | 3/4 | 56.779 [54.925, 57.850] | 0.001004 [0.000977, 0.001019] |
| LLM-jp | 32 | 2/4 | 55.965 [54.086, 57.845] | 0.000280 [0.000275, 0.000285] |
| LLM-jp | 128 | 3/4 | 55.164 [54.240, 55.948] | 0.0000856 [0.0000838, 0.0000870] |
| OLMoE | 1 | 2/4 | 51.904 [51.786, 52.022] | 0.007445 [0.007425, 0.007465] |
| OLMoE | 8 | 1/4 | 53.109 [53.109, 53.109] | 0.000869 [0.000869, 0.000869] |
| OLMoE | 32 | 1/4 | 59.825 [59.825, 59.825] | 0.000278 [0.000278, 0.000278] |
| OLMoE | 128 | 0/4 | N/A | N/A |

`[Observed]` LLM-jp 状态为 `CHARACTERIZATION_COMPLETE_WITH_FILTERED_WINDOWS`（11 valid / 5 invalid）；OLMoE 为 `CHARACTERIZATION_INCOMPLETE_INVALID_WINDOWS`（4 valid / 12 invalid）。这些值没有 tier 对照、natural route、matched SLO token 或 policy arm，不能计算 route-conditioned effect、actionability、Oracle saving 或 CaptureRatio。

`[Observed]` 同一冻结 validator 的独立 LLM-jp 复跑为 10 valid / 6 invalid，rows 1/8/32/128 = 3/2/3/2。复跑与同一 OLMoE 失败证据密封在 `20260728_153000_rtx5090_physical_dev`：32 个尝试窗中 14 个有效，12 个因 gap>20 ms、6 个因 ΔT>2°C 失效。所有 host window 均≥12.254 s，因此不存在短窗分母漂移。

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
development isolated-expert physical windows = 15 accepted / 17 rejected
development LLM-jp same-protocol replication = 10 accepted / 6 rejected
development models with complete row grid = 1/2 (filtered); formal eligible = 0/2
development GPU route model revisions = 2
development GPU route contributions = 768
NVML capability samples = 200 (not independent workload samples)
development fixed-revision parity = 2/2 PASS (not formal serving exactness)
rejected synthetic ABBA attempts = 2 (12 windows each; accepted N=0)
paired physical difference = N/A
formal 95% CI = N/A
missing formal strategy samples = all Experiment A-E samples
filtered formal physical samples = 0
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
| Gate 0 | 9 个 P0 仍开放；natural continuous serving、可接受的同窗 CUDA/energy、thermal、双模型 formal-serving exactness 和完整 E2E ledger 未闭合 |
| Gate 1 | Gate 0 顺序阻断；2/2 models 仅完成 development route capture，0/2 完成 formal natural service-energy experiment |
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

`figures/README.md` 拒绝在 formal strategy sample=0 时生成预注册的 12 张科学图。当前没有 rows×tier latency/energy surface、natural energy-mass heatmap、Pareto、baseline/Oracle、SLO–energy 或 thermal timeline；route CSV 或 NVML capability 点不能替代这些图。

## 10. Artifact 完整性

canonical artifact 保存 `manifest.json`、environment、config、commands、git diff、raw CSV/JSONL、processed summary、figures marker、完整测试日志和 verdict。`manifest.json` 记录 base commit、dirty status、seed、执行命令以及每个已纳入文件的 SHA-256。

supporting audit bundle 另保存五份报告的运行时快照、关键源文件和 `logs/gpu_gate_attempts.log`；它只补全 provenance，不替换 canonical artifact 中已报告的 host timing。

final supporting dry-run 在新 timestamped 目录重新执行当前 123/123 tests，并把 follow-up、latest GPU dev、sealed qualification 的清单/摘要纳入同一 17-file manifest；17/17 文件复算一致，`formal_result=false`、`Gate0=FAIL`。

final current audit bundle 重新执行 124/124 tests，并纳入两模型 current-validator summary/trials、parity、meter、两次 ABBA failure 与五份报告快照；manifest 声明 29 个文件，29/29 复算一致，`formal_result=false`、`Gate0=FAIL`。

repeated physical bundle 另密封了 132 MiB 双模型 activation capture、双模型 4-step zero-error exactness、32 个 physical window、所有 raw telemetry/失败重试和 source snapshot。顶层 manifest 声明 368 个文件，368/368 复算一致，状态为 `PHYSICAL_CHARACTERIZATION_INCOMPLETE_GATE0_FAIL`。

GPU development bundle 保存两份原始 route CSV/meta、成功与失败日志、冻结 revision/config/token IDs、源码 SHA、nvidia-smi/NVML raw state、环境、commands、verdict 和 manifest。下载后复核 21/21 声明文件，无 missing/hash/extra；它自己的 `formal_result=false`、`Gate0=FAIL` 是授权边界。

parity/failed-energy follow-up 保存 19 个 hash-bound 声明文件：两模型 16-token prefill + 2-step parity、20/20 runner tests、一次启动前竞争失败和一次 12-window 后竞争失败。本地复算 19/19，无 missing/hash/extra；12 个窗口属于整次被拒绝的调试记录，有效 energy N=0。

latest GPU development bundle 另固化了修正后的 GPU/NVML 环境 provenance、双模型 parity、meter preflight 和两次 ABBA fail-closed 记录。本地复核 manifest 列出 70 个文件，0 missing、0 hash mismatch，唯一 unlisted 文件是 `manifest.json` 自身。

sealed GPU qualification bundle 另保存 29 个 hash-bound file，包括 raw route/exactness/meter、全部成功与失败日志、96-test log、environment/config/commands/verdict、processed summary 和三支新增 GPU 工具的 `source_snapshot/`。远端封装后与本地下载后均复算为 29/29 匹配；archive 传输 SHA-256 为 `e409f1d9f9bd338d30aef8575fb04c6d398329c5827673aacd98c3ca6f2c9264`。

`[Inferred]` 当前唯一可复核结论是 measurement contract、development decode helper、fallback 和 fail-closed pipeline 可运行；没有证据授权 RouteSlack controller 或 `8xA100_CANDIDATE`。
