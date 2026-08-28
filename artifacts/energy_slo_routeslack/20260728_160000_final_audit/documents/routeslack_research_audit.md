# RouteSlack-MoE 研究审计与假设冻结

> 日期：2026-07-28  
> 裁决：`MEASUREMENT_ONLY`  
> 证据边界：Gate 0 失败；RTX 5090 上完成两个冻结模型的 batch-1 cached-route development probe，但正式 latency/energy 策略样本数仍为 0；本轮没有证实或证伪 H1/H2/H3。
> latest update：双模型 fixed-revision development parity 与 GPU/NVML meter preflight 通过；两次 synthetic ABBA 均因竞争 CUDA 进程 fail closed。current-validator 的 default-tier isolated-expert characterization 中，LLM-jp 首跑 11/16、同协议独立复跑 10/16 窗有效，OLMoE 仅 4/16 窗有效且 rows=128 无合格样本；formal strategy sample 仍为 0。

## 1. 直接结论

- `[Observed]` `docs/current/README.md` 仍是当前权威入口：没有已经通过 formal Gate 的 MoE 系统主机制；BCRD 仍是 `DESIGNED_AND_IMPLEMENTED / NOT_FORMALLY_RUN`。
- `[Observed]` 历史 `capture_native_routes.py --phase decode` 只是对一次 `model(**inputs)` 标签 decode，不是 KV-cache decode。本轮 working tree 已增加 prefill + 逐 token `past_key_values` 路径，并在 tiny OLMoE 上完成 cached logits 对 full recomputation 的 CPU 等价测试。
- `[Observed]` 该新路径仍只支持 batch size 1 的 development capture；双模型 4-step patched/native exactness 已通过，但 arrival/deadline 由 CLI 合成，仍没有 natural continuous batching、真实 ready time、dispatch/execute/combine ledger 或 matched SLO energy window，因此 metadata 正确保持 `formal_eligible=false`。
- `[Observed]` BCRD service curve 只测 isolated expert CUDA latency，不含 Joules、power/clock tier、thermal state 或 natural input-event 方差；本轮已禁止它被自动升格为 RouteSlack Gate-1 证据。
- `[Observed]` 本地 Mac 仍无 NVIDIA GPU；随后在隔离远端 RTX 5090（32,607 MiB、driver 595.71.05、PyTorch 2.8.0+cu128、Transformers 4.57.6）上执行了严格标为 non-formal 的最小探针。两冻结 revision 均从本地 HF cache 离线解析，GPU UUID 和源码 hash 已落盘。
- `[Observed]` LLM-jp 完成 2 个 cached one-token step，得到 `2×16 layers×top-16=512` 条 contribution；OLMoE 得到 `2×16×top-8=256` 条。每步 layer/top-k/token identity、revision 和输出文件 hash 全部闭合；两份 metadata 均保持 `formal_eligible=false`。
- `[Observed]` 随后的双模型 native-vs-shared-patch parity 也通过：相同 16-token prompt 与强制 2-step decode 下，两模型的 prefill/decode max absolute logit error、KL、selected-expert mismatch 和 route-weight error 均为 0；native/patched KV length 均为 `[17,18]`。这关闭的是 batch-1 instrumentation exactness 子项，不是 serving/E2E exactness。
- `[Observed]` RTX 5090 的 NVML 累计 energy counter 可读；一次无竞争 compute process 的 200 点能力窗口中，请求采样间隔 5 ms，实际 gap median 5.090 ms、max 12.024 ms，低于 20 ms gate。该窗口不是 idle calibration、workload energy 或 A/B 策略样本。
- `[Observed]` canonical artifact 内嵌 96 个 CPU/合成协议测试，并完成 host-only no-op 开销测量；formal energy/latency sample 都是 0。历史 BCRD `SMOKE_ONLY` 只作为支持性代码路径证据，不进入 RouteSlack 物理判定。
- `[Observed]` 新增 GPU development bundle `artifacts/energy_slo_routeslack/20260728_144614_gpu_dev/`，manifest SHA-256 为 `491faee358570fcba18e1564c5dbde57695e09c1309acc5409a46ae536816480`。bundle 内 21 个声明文件逐一校验无缺失、无 hash 漂移，并保留首次因遗漏静态依赖而失败的原始日志。
- `[Observed]` 扩展 GPU development bundle `20260728_1500_remote_5090_gpu_dev` 又完成远端 96/96 tests、LLM-jp 8,192 与 OLMoE 4,096 条 route contribution，以及 rows 1–64 的 14 个 isolated CUDA latency point；34 个 manifest entry 全部 hash 通过。所有结果仍显式 non-formal，strategy energy window N=0。
- `[Observed]` latest bundle `20260728_151500_rtx5090_dev` 固化了正确的 RTX 5090/NVML provenance、五项 meter preflight PASS、两冻结 revision 的 `DEVELOPMENT_PARITY_PASS`，以及两次竞争失败记录。manifest SHA-256 为 `d23f18a0455c4ebd04f3611746bc8150eac674ba1bd97e91d24ee5e7517f05c8`；70 个声明文件 0 missing / 0 hash mismatch。
- `[Observed]` current-validator energy characterization 位于 `artifacts/energy_slo_routeslack/20260728_154500_current_energy_characterization/`。LLM-jp 在 layer 1/expert 16 的 rows 1/8/32/128 上分别有 3/3/2/3 个 request-disjoint 合格窗（共 11/16）；OLMoE 在 layer 1/expert 5 上仅 rows 1/8/32 有 2/1/1 个合格窗（共 4/16），rows=128 为 0。LLM-jp/OLMoE manifest SHA-256 分别为 `152317db94a008df4f51a9c87cae3faccd3fcf1ab96c98056f1be42daf758716` / `b2550018de3b8d481cf13d6039715affb4e0c4cb1ed0ba60224d102f7bff8a1c`，各 8/8 声明文件验哈希通过。
- `[Observed]` 同一冻结 validator 的独立 LLM-jp 复跑为 10/16 有效（rows 1/8/32/128 = 3/2/3/2）；与同一 OLMoE 4/16 证据一起密封在 `artifacts/energy_slo_routeslack/20260728_153000_rtx5090_physical_dev/`。该 bundle 记录 32 个尝试窗、14 个有效窗、18 个无效窗（12 个 gap、6 个 thermal），368/368 文件验哈希通过，manifest SHA-256 `7727e487b3035122e763b519b493af9aa081d4dfbf0d7ad7144f91708ead4ef9`。两次 LLM-jp 并未改变 OLMoE row-grid 不完整的双模型 AND 失败。
- `[Observed]` 这些窗口只测 provider-default 575 W 下、来自 calibration prefill 的单 expert BF16 execution，分母是 processed expert row。它们不是 natural cached-decode、没有 power/clock tier 干预、没有 matched SLO-completed token、EP dispatch/combine 或 route-conditioned A/B，因而只能作为局部能耗 characterization，不能进入 H1/H2/H3 判定。
- `[Observed]` 新密封的最小 Gate-0 qualification bundle `artifacts/energy_slo_routeslack_gpu/20260728_144600/` 在同一 RTX 5090 上重新执行 96/96 protocol-critical tests。冻结 revision 的 4-step batch-1 capture 得到 LLM-jp 2,048、OLMoE 1,024 条 contribution；native 与 instrumented cached decode 对比得到 4/4 step logits/argmax/KV-length 全通过，逐步 max/mean absolute logit error 均为 0。
- `[Observed]` 同 bundle 的 3.0016 s NVML capability workload 得到 299 个 sample、max gap 10.925 ms、累计 counter 增量 1,724.861 J；但温度从 35°C 升至 58°C（Δ23°C），超过正式 pair 的 2°C 门限，且 power integration 为 1,328.708 J，与累计 counter 明显不一致。它只证明 meter capability，不是 model/service energy 样本；主账本继续使用累计 counter，formal physical-model energy N=0。
- `[Observed]` 该 bundle 的 29 个 manifest entry 已在远端与下载后各复算一次，均无缺失或 hash 漂移；manifest SHA-256 为 `69e6303bfaadaec93bfa6f15fee0abe154325c0ce097e8d58914d9fc197a0f37`，并保留两次失败 exactness 启动日志及三支 GPU 工具的 `source_snapshot/`。
- `[Inferred]` 现有资产可以作为 measurement/protocol characterization，不能支持 controller 论文主张，也不能升格为 `8xA100_CANDIDATE`。

## 2. 交叉核验后的证据层级

| 资产 | 当前定位 | 可以证明 | 不可以证明 |
|---|---|---|---|
| `docs/current/README.md` | `[Observed]` 当前权威 | 正式 Gate 状态和执行边界 | 新机制已成立 |
| BCRD experiments | `[Observed]` latency-oriented logical replay | route schema、小窗口 assignment replay、smoke 全链 | 真实 EP、Energy–SLO、matched completion |
| route-row FP8 | `[Observed]` development/proxy asset | monotonic power-accounting helper、completed-token denominator contract | RouteSlack BF16 exact path 或 formal energy surface |
| JouleQueue | `[Observed]` superseded/development asset | NVML/counter 接口和部分测试思路 | natural decode、等 repeat AB/BA、thermal-closed formal result |
| RouteSlack CPU contracts | `[Observed]` audit-only | identity conservation、cache audit shape、counter wrap、fallback、Oracle/online 接口隔离 | 任何 GPU 物理效应 |
| `20260728_115300` artifact | `[Observed]` canonical dry-run | 96 tests、tiny cached-decode development capture、四阶段 synthetic identity ledger、10 baseline 名称与 Oracle 的接口/管线调用、host-only no-op、raw artifacts 和逐文件 hash | 仍为 `formal_result=false`；没有实现或执行 10 个物理 baseline 算法，也不增加物理样本 |
| `20260728_120340` artifact | `[Observed]` supporting audit bundle | 最终五份报告快照、GPU fail-closed 原始日志、关键源文件和 96-test rerun 的 hash-bound bundle | supporting only；不替代 `115300` 的已报告 host timing，也不增加物理样本 |
| `20260728_144614_gpu_dev` artifact | `[Observed]` RTX 5090 development bundle | 双模型真实 cached-step 路由闭合、冻结 revision/config、NVML counter/采样能力、远端 20+31 tests 与完整 raw log/hash | batch=1、synthetic arrival；没有 continuous serving、同窗策略能耗、EP 或 SLO denominator；`Gate0=FAIL` |
| `20260728_150422_gpu_followup` artifact | `[Observed]` RTX 5090 follow-up bundle | 双模型 native/patch exact parity、KV/layer/top-k closure、20/20 runner tests，以及竞争 workload 下 energy runner 的 fail-closed negative result；19/19 文件验哈希闭合，manifest `e588eafb…bbf654e` | parity 仍为 1 prompt/model、batch=1；一次在 0 window 拒绝、一次完成 12 window 后整次拒绝，accepted energy N=0，不能给出策略能耗差或 Gate-1 结论 |
| `20260728_1500_remote_5090_gpu_dev` artifact | `[Observed]` expanded RTX 5090 development bundle | 远端 96 tests、双模型共 12,288 route contributions、14 个 isolated CUDA latency point、完整 GPU/environment/raw logs | latency trials 是 inner repeats，不是 natural independent unit；无 aligned energy window；`formal_result=false`、`Gate0=FAIL` |
| `20260728_151500_rtx5090_dev` artifact | `[Observed]` latest RTX 5090 development bundle | 双模型 prefill + 2 decode step native/patched parity 零误差；GPU/NVML preflight PASS；修正后环境 provenance；123-test protocol bundle | 两次 ABBA 各 12 windows 均因竞争进程被整体拒绝；accepted energy N=0；`formal_result=false`、`Gate0=FAIL` |
| `20260728_154500_current_energy_characterization` artifact | `[Observed]` current-validator isolated-expert energy characterization | 两模型真实 BF16 activation、累计 NVML counter、CUDA event、request-disjoint outer trial、raw telemetry 和 bootstrap CI | LLM-jp 11/16 有效但有过滤窗；OLMoE 4/16 有效且 rows=128 缺失；default tier only、prefill activation、expert-row denominator，不能作为 natural RouteSlack surface |
| `20260728_153000_rtx5090_physical_dev` artifact | `[Observed]` physical-characterization replication bundle | 双模型 4-step zero-error exactness、132 MiB activation capture、32 个物理窗、全部 raw telemetry/失败重试与 368-file hash closure | 只有 14/32 窗合格；OLMoE rows=128 仍 0/4；formal strategy-energy N=0 |
| `energy_slo_routeslack_gpu/20260728_144600` artifact | `[Observed]` sealed single-GPU Gate-0 qualification | 96/96 tests、双模型 4-step cached-decode zero-error exactness、3,072 条 development route contribution、NVML UUID/counter/10.925 ms max-gap 能力、29-file hash closure | batch=1、synthetic arrival；ΔT=23°C formal thermal FAIL；无 continuous serving、EP、matched completion、service-energy surface 或策略比较；`formal_result=false`、`Gate0=FAIL` |

`ask` 作者历史检索因 git-ai daemon lock 未取得可用会话；因此本轮不引用作者意图。关于旧 `--phase decode` 仍是一次 tokenizer + `model(**inputs)` full forward 的结论，只来自客观代码与 metadata 审查。

## 3. 原研究问题复审

1. `[Inferred]` 原命题只有在 route、完成 identity、SLO、actuator、latency/energy window 同时冻结时才可证伪。当前 replay 将 workload observation、预测 surface 与物理 execution 混合，原实现不能直接回答该命题。
2. `[Blocked]` 没有证据表明 route-conditioned energy variation 独立于 batch size、token 数、expert rows、activated experts、queue length、GPU utilization、KV length 与 phase。
3. `[Observed]` 以下都能伪造“节能”：少完成 token、降低 throughput、拒绝/超时更多请求、放宽 SLO、改变 repeat、改变 meter window、比较不同 completion set。冻结协议必须对这些全部 fail closed。
4. `[Observed]` 本轮已证明单卡可执行两模型 batch-1 KV-cache route capture，且 NVML counter/≤20 ms sampling 接口可用。`[Inferred]` 同卡原则上可进一步测局部 BF16 rows×tier surface、patched/native logits、power transition 和 instrumentation tax；但当前代码/证据仍不能证明真实 EP assignment、A2A/NCCL/RDMA、跨 rank queue、dispatch/combine、EP TPOT/P99 或多卡 board energy。
5. `[Blocked]` source-rank→target-replica 可执行性、通信/计算 overlap、rank slack、multi-board switching/idle tax 和端到端 matched-SLO energy/token 必须留给真实 8×A100 EP serving。
6. `[Hypothesis]` `min-finish + two-tier power` 或 route-unaware batch/KV/phase controller 可能吸收大部分 Oracle 上限；当前没有物理比较。

## 4. 冻结研究边界

仅允许 fixed-replica assignment、bounded expert microbatch sealing、粗粒度 GPU power/clock tier、dispatch ordering。禁止动态 top-k、expert skipping、精度/权重/语义变化、低秩近似、动态 placement migration、RL 和 BO。所有策略必须满足：

```text
router、top-k、expert identity、weights、dtype、admission、输出 token 完全一致
Delta Q = 0
```

## 5. 三个可证伪核心假设

### H1：natural route 中存在独立 energy variation

- 自变量：route identity/shape 与冻结 power tier。
- 因变量：raw J/SLO-completed-token（主），raw J/row、CUDA latency、host latency（辅）。
- 控制：model revision、BF16、input/output、rows、top-k、layer/expert、repeat、clock/power limit、temperature、utilization、window。
- independent unit：fresh input event/document；inner repeat 不是样本。
- 反例：差异被 batch/KV/phase/utilization baseline 吸收，或小于 thermal/meter drift。
- 通过：至少一个两模型共同的 natural cell 在各模型中 paired effect ≥10%、95% LCB >5%，至少一个共同 cell 在各模型中 effect ≥15%，且可测/可执行 cell 覆盖各自 ≥20% natural energy mass。
- 一票否决：任一模型所有 common cell effect <5%、actionability <20%、只在 synthetic/单模型出现、identity/accounting 不闭合、thermal drift 大于策略差或依赖语义变化。

### H2：variation 对合法 actuator 可执行

- 自变量：fixed assignment、bounded seal、coarse tier 或 dispatch order。
- 因变量：action flip、switch/hold 后净 raw energy、completion latency/SLO。
- 控制：相同 arrival、admission、route、output、completion set、surface version、device state。
- independent unit：natural action opportunity/wave，按 input event/request 聚类。
- 反例：只有一个可行 replica/tier；switch latency 超过 slack；seal 需要未来信息；action 不改变 execution。
- 通过：两模型各自 actionable natural energy mass ≥20%，且 matched SLO 下 paired energy effect 非零。
- 一票否决：actionability <20%；future leak；零成本迁移/切换；deadline 前不可完成。

### H3：强简单 baseline 后仍有 residual headroom

- 自变量：10 个 online baseline 与 future-known conservative Oracle，使用不同输入类型。
- 因变量：raw J/SLO-completed-token、P50/P95/P99、SLO violation、goodput、capture ratio。
- 控制：同 trace/SLO/admission/output/completion identity；参数只在 calibration split 冻结。
- independent unit：paired request/input-event trace；AB/BA block 是热状态配对单位。
- 反例：Oracle 净收益很小，或 `min-finish + two-tier` 已近似 Oracle。
- 通过：两模型分别满足 Oracle net saving ≥10%、SLO 不劣化、最强 simple capture ratio <90%。
- 一票否决：任一模型 Oracle <10%；simple capture ≥90%；completion set 不匹配；controller/no-op tax > gross saving 的 20%。

## 6. 当前假设状态

| 假设 | 状态 | 原因 |
|---|---|---|
| H1 | `[Blocked]` | formal strategy-energy sample 仍为 0；虽有双模型 default-tier isolated-expert raw-energy 窗口，但没有 natural continuous-serving、tier 对照、完整 row-grid AND 或 route-conditioned matched pair |
| H2 | `[Blocked]` | 无真实 EP replica choice；switch/seal/dispatch 未进入同窗物理测量 |
| H3 | `[Blocked]` | 没有合格 H1/H2 输入；Oracle/baseline 只有 synthetic dry-run |

`MEASUREMENT_ONLY` 表示当前资产的可用边界：保留协议、meter 和 characterization 脚手，停止 RouteSlack controller 主线。它不是对 H1/H2/H3 的正例。
