# RouteSlack-MoE 研究审计与假设冻结

> 日期：2026-07-28  
> 裁决：`MEASUREMENT_ONLY`  
> 证据边界：Gate 0 失败；物理 GPU 样本数为 0；本轮没有证实或证伪 H1/H2/H3。

## 1. 直接结论

- `[Observed]` `docs/current/README.md` 仍是当前权威入口：没有已经通过 formal Gate 的 MoE 系统主机制；BCRD 仍是 `DESIGNED_AND_IMPLEMENTED / NOT_FORMALLY_RUN`。
- `[Observed]` 历史 `capture_native_routes.py --phase decode` 只是对一次 `model(**inputs)` 标签 decode，不是 KV-cache decode。本轮 working tree 已增加 prefill + 逐 token `past_key_values` 路径，并在 tiny OLMoE 上完成 cached logits 对 full recomputation 的 CPU 等价测试。
- `[Observed]` 该新路径仍只支持 batch size 1 的 development capture；arrival/deadline 由 CLI 合成，没有 natural continuous batching、真实 ready time、dispatch/execute/combine ledger、GPU latency/energy window 或双模型 exactness qualification，因此 metadata 正确保持 `formal_eligible=false`。
- `[Observed]` BCRD service curve 只测 isolated expert CUDA latency，不含 Joules、power/clock tier、thermal state 或 natural input-event 方差；本轮已禁止它被自动升格为 RouteSlack Gate-1 证据。
- `[Observed]` 当前 Mac 无 NVIDIA GPU：没有 `nvidia-smi`；项目 `.venv` 中为 PyTorch 2.8.0，`torch.cuda.is_available()==False`、`torch.version.cuda is None`、CUDA device count 为 0。
- `[Observed]` 本轮只完成 95 个 CPU/合成协议测试、一条 `SMOKE_ONLY` BCRD 全链和 host-only no-op 开销测量；物理 energy/latency sample 都是 0。
- `[Inferred]` 现有资产可以作为 measurement/protocol characterization，不能支持 controller 论文主张，也不能升格为 `8xA100_CANDIDATE`。

## 2. 交叉核验后的证据层级

| 资产 | 当前定位 | 可以证明 | 不可以证明 |
|---|---|---|---|
| `docs/current/README.md` | `[Observed]` 当前权威 | 正式 Gate 状态和执行边界 | 新机制已成立 |
| BCRD experiments | `[Observed]` latency-oriented logical replay | route schema、小窗口 assignment replay、smoke 全链 | 真实 EP、Energy–SLO、matched completion |
| route-row FP8 | `[Observed]` development/proxy asset | monotonic power-accounting helper、completed-token denominator contract | RouteSlack BF16 exact path 或 formal energy surface |
| JouleQueue | `[Observed]` superseded/development asset | NVML/counter 接口和部分测试思路 | natural decode、等 repeat AB/BA、thermal-closed formal result |
| RouteSlack CPU contracts | `[Observed]` audit-only | identity conservation、cache audit shape、counter wrap、fallback、Oracle/online 接口隔离 | 任何 GPU 物理效应 |
| `20260728_114917` artifact | `[Observed]` 主 dry-run | 95 tests pass、tiny cached-decode development capture、GPU fail-closed probes、10 baselines + Oracle 接口可执行 | latency/energy saving、SLO、capture ratio |
| `20260728_114933` artifact | `[Observed]` supporting smoke | BCRD `SMOKE_ONLY` Gate 1–3 全链可打包复现 | natural route、physical Oracle 或正式 Gate |

`ask` 历史检索未取得原始会话（git-ai daemon lock）。因此对原代码意图的结论只来自客观代码与 metadata：它最初是 native router-output capture shim，不是 serving decode engine。

## 3. 原研究问题复审

1. `[Inferred]` 原命题只有在 route、完成 identity、SLO、actuator、latency/energy window 同时冻结时才可证伪。当前 replay 将 workload observation、预测 surface 与物理 execution 混合，原实现不能直接回答该命题。
2. `[Blocked]` 没有证据表明 route-conditioned energy variation 独立于 batch size、token 数、expert rows、activated experts、queue length、GPU utilization、KV length 与 phase。
3. `[Observed]` 以下都能伪造“节能”：少完成 token、降低 throughput、拒绝/超时更多请求、放宽 SLO、改变 repeat、改变 meter window、比较不同 completion set。冻结协议必须对这些全部 fail closed。
4. `[Inferred]` 单卡可测局部 BF16 rows×tier latency/energy surface、KV-cache route、exact logits、power transition 和 instrumentation tax；不能证明真实 EP assignment、A2A/NCCL/RDMA、跨 rank queue、dispatch/combine、EP TPOT/P99 或多卡 board energy。
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
- 通过：两模型各至少一个 common natural cell 的 paired effect ≥10%、95% LCB >5%，且可测 cell 覆盖各自 ≥20% natural energy mass。
- 一票否决：只在 synthetic/单模型出现；identity/accounting 不闭合；thermal drift 大于策略差；依赖语义变化。

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
| H1 | `[Blocked]` | 0 个 formal energy sample；无双模型 natural continuous-decode surface |
| H2 | `[Blocked]` | 无真实 EP replica choice；switch/seal/dispatch 未进入同窗物理测量 |
| H3 | `[Blocked]` | 没有合格 H1/H2 输入；Oracle/baseline 只有 synthetic dry-run |

`MEASUREMENT_ONLY` 表示当前资产的可用边界：保留协议、meter 和 characterization 脚手，停止 RouteSlack controller 主线。它不是对 H1/H2/H3 的正例。
