# RouteSlack-MoE 实验代码审查

> 审查日期：2026-07-28  
> 结论：`Gate 0 FAIL`  
> 当前未闭合：P0 = 14，P1 = 7，P2 = 3。  
> 本轮已修复/降级误用风险 7 类，但没有将 development path 声称为 formal path。

## 1. 未闭合 P0

| 文件 | 行号 | P0/P1/P2 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---:|---|---|---|---|---|---|
| `capture_native_routes.py` | 103–217, 367–520 | P0 | 已有 batch-1 cached decode，但不是 natural continuous batching；arrival/deadline 仍合成，无 real ready-time/serving timeline/双模型 GPU exactness | 不能将离线自回归 route 外推为 online serving、SLO 或 EP 证据 | 接入真实 continuous engine，保存 active set、KV owner、per-step ready time，对两模型 patched/unpatched exactness | `test_cached_decode.py` 已验 development cache；尚缺真实 engine E2E | `OPEN_FORMAL`; metadata 已 fail closed |
| `core.py`; replay 调用链 | 177–308 | P0 | v2 identity/stage validator 存在，但只有 census 强制 v2；replay 未写 dispatch/executed/combined ledger，也未在主路径调用 validator | drop/duplicate/wrong-target 仍可在 aggregate 指标中被隐藏 | 将 immutable contribution key 和 target action 分离，每 stage 写 ledger 并强制闭合 | drop/duplicate/reorder/wrong-target/top-k sibling 集成测试 | `OPEN_INTEGRATION`; CPU contract 已有 |
| `benchmark_expert_service_curve.py` | 95–187 | P0 | 只测 isolated expert latency；无 energy、tier、thermal、input-event | H1 需要 rows×tier latency-energy surface，latency proxy 不是 Joules | 新建同 trial CUDA+raw-energy surface runner，以 input event 为独立单位 | two-model natural surface + AB/BA + energy window | `OPEN`; 已改为 `formal_eligible=false` |
| `core.py` `ServiceCatalog` | 356–423 | P0 | key 仅 `(model, layer)`，丢失 model revision、expert、tier、dtype、GPU UUID 和 tier realization | 不同 expert/设备/档位可被错误合并，导致 Oracle 使用错 surface | exact key 至少包含 model/revision/layer/expert/rows/tier/dtype/GPU UUID/driver/config hash；越界 fail closed | expert/tier/context collision 负测试 | `OPEN` |
| `census_fragmentation.py`; `build_fixed_replica_instances.py` | 35–70; 46–96 | P0 | 一个 request 的多个 decode token 被同 wave/window 处理；后续 route 可能被当作时刻 0 已知 | 制造不可执行的 sealing/batching 与 future leak | 用实测 per-step release event 推进；后续 token 只能在前步 completion 后 ready | 追加未来 route 不改变已发生 action | `OPEN` |
| `census_fragmentation.py` | 100–188 | P0 | `fragmented work - consolidated work` 除 aggregate path 时间，没有同时钟 DAG/overlap；target replica 是虚拟策略 | total-work proxy 不等于 exposed latency 或 energy mass | 建立同边界 stage timeline/deletion replay；proxy 与 physical exposed mass 分开报告 | 量纲、overlap、critical-path 反例测试 | `OPEN` |
| `census_fragmentation.py` | 209–228 | P0 | `passing10` 和 `passing15` 可由不同 cell 拼出 PASS；KILL 谓词也不是冻结的两模型同-cell 规则 | 可产生 false PASS/false KILL | 将 effect、LCB、15%、actionability 在同一 cell 原子判定，两模型 AND | 多 cell 反例测试 | `OPEN` |
| `solve_assignment_oracle.py`; `core.py` | 57–134; 514–601 | P0 | Oracle 优化 on-time/latency/service，不是 raw energy；漏 dispatch/combine/return、switch、idle、controller tax，默认不保守 | 无法回答 H3 的 net Energy–SLO headroom | 改为 matched-set raw-energy objective，计入所有合法成本并使用保守 surface | 每项税的 ablation + impossible action 负测试 | `OPEN` |
| `solve_assignment_oracle.py`; `compare_policies.py`; `compute_captured_headroom.py` | 115–134; 45–82; 37–89 | P0 | 只保存 aggregate completion/on-time，无 exact completed request/token IDs；headroom 为 latency | 策略可通过改变完成集合伪造收益 | 每 arm 输出 completion identity/hash；不匹配即 INVALID；主指标改为 raw J/SLO-completed-token | 相同数量但不同 IDs 必须失败 | `OPEN` |
| `policies.py`; `compare_policies.py` | 19–43; 17, 129–177 | P0 | `expert_rows` 不随 seal 清零；策略预测与 replay 执行模型不一致；只有 6 个 policy，缺冻结强 baseline | 可高估 batching credit，并在 Gate 3 未完成时输出 verdict | 显式 open-batch/seal state；补齐 10 个 online baseline；缺项直接 INVALID | 远隔 singleton 不获得 credit；全 baseline registry 测试 | `OPEN` |
| `capture→census→instances→oracle→gate3` | 多文件 | P0 | 本轮只在 census 增加 trace/curve eligibility 检查；后续 hash/provenance 链仍不完整；split 按 `instance_id` 使同 request 可跨 split | 弱证据、换件或 calibration leakage 可进入 evaluation | 全链验证 manifest/hash/revision/config/mode；按 request/document cluster split | 任一 artifact 替换都 fail；request split cardinality=1 | `OPEN_AFTER_CENSUS` |
| `compare_policies.py`; `compute_captured_headroom.py` | 119–127; 37–68 | P0 | Gate2 多 remote-cost 运行与 Gate3/headroom join 协议漂移；已复现 `need exactly one Oracle remote-cost row` | 文档化全链在正常配置下失败，也可 join 错 Oracle cell | 把 remote cell 写入 resolved plan，按 `(instance_id, remote)` 严格 join | documented-command integration test | `OPEN_REPRODUCED` |
| `run_joulequeue_expert_surface.py`; shared meter | 176–251, 401–470, 674–763; `power_accounting.py` 276–324 | P0 | counter/sample/workload 仅 sequential bracketing；两 arm 可用不同 repeat；重用同 `pool[:rows]` 却将 outer windows 当 independent trials | 固定 meter/window overhead 被不同分母摊销，伪造 energy delta；CI 伪重复 | 同 repeat ABBA，fresh input event，原始 counter/sample/workload timestamp 同 artifact，只按 document/request bootstrap | fixed-overhead、equal-work、different-repeat、window alignment 集成测试 | `OPEN`; pure contracts 已覆盖部分反例 |
| JouleQueue/route-row/formal runner | 多文件 | P0 | 未统一保存 temperature/clocks/power limit/util/throttle/idle calibration、实际 tier 和 matched completed set；也无真实 rows×tier energy runner | energy accounting 不闭合，thermal/clock drift 可大于策略差 | 统一 meter + thermal timeline + raw/idle 双口径 + UUID binding + matched completion | fake NVML round-trip、thermal pair、idle sensitivity、UUID 负测试 | `OPEN_BLOCKS_GPU_FORMAL` |

## 2. 未闭合 P1/P2

| 文件 | 行号 | P0/P1/P2 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---:|---|---|---|---|---|---|
| `benchmark_expert_service_curve.py` | 95–124 | P1 | 同一 random tensor 做 200 inner trials，无 event/layer/expert 方差、host wall 和 CI | 把 inner repeat 误当样本会大幅高估置信度 | fresh event 分层采样与 hierarchical bootstrap | 样本 ID 唯一性/聚类测试 | `OPEN`; formal 已禁止 |
| `power_accounting.py` | 351–400 | P1 | UUID helper 存在，但 sampler/runtime 没有强制 NVML 与 CUDA device 绑定 | 可读错 GPU 的能耗 | runner 启动时强制 UUID 对齐 | 多卡错 UUID 集成测试 | `OPEN` |
| `power_accounting.py` | 163–218 | P1 | idle power 完全由 caller 提供，无独立 idle window/CI/provenance | 动态能耗对 idle 选择敏感 | 运行前单独配对 idle calibration 并保存 raw | idle baseline sensitivity 已有 unit；缺 runner | `OPEN_INTEGRATION` |
| JouleQueue/route-row surface | 多处 | P1 | latency 与 energy 在不同执行中采样，快 arm 可落入过短 window | 热状态和工作量不匹配 | 同 trial 同时产生 CUDA events 和 raw energy；最小窗口按最快 arm 校准 | 相同 window/repeat 集成测试 | `OPEN` |
| `run_routeslack_dry_run.py` | host no-op section | P1 | 当前 no-op 只是 CPU Python microbenchmark，无真实 hook/logging/GPU energy/E2E SLO | 不能作为 controller tax | 在真实 serving 中对 no-hook/hook/no-op decision 做配对 GPU 测量 | serving no-op ABBA | `OPEN_GPU`; host 数字仅 development |
| `routeslack_protocol.py`; `gate0_contracts.py` | 25–444; 1–420 | P1 | 两套 CPU contract 仍存在漂移；boolean `Gate0Evidence`/`formal_gate_status` 不能作为物理签署 | caller 可能自报 PASS，而不是从 artifact 推导 | 合并为一个 contract；formal gate 只接受 hash/provenance validator 输出 | self-attestation 必须被拒绝 | `OPEN`; dry-run 明确 non-formal |
| `gate0_contracts.py` | 369–405 | P1 | thermal validator 尚未完整检查 finite、时间顺序、power draw/utilization 可比性 | 异常 telemetry 可通过 | 完整 schema/range/timestamp/pair validation | NaN、逆时、util drift 负测试 | `OPEN` |
| 三套 NVML/meter 实现 | 多文件 | P2 | shared、JouleQueue、route-row 边界和错误语义不一致 | 协议漂移使后续审计成本升高 | 统一到 shared meter，其他仅做 adapter | shared conformance suite | `OPEN` |
| `run_energy_slo_power_probe.py` | 66–118 | P2 | 历史 `time.time()`/50 ms/full-forward sampler 未强制 historical-only | 容易被误复用为 formal | 文档和代码中标记 `HISTORICAL_ONLY` 并拒绝 formal flag | formal invocation 负测试 | `OPEN` |
| Oracle/replay parsers | 多文件 | P2 | 重复 contribution parser，混用 `ValueError/ProtocolError/AssertionError` | schema 易漂移，错误难审计 | 统一 strict artifact loader 与错误分类 | malformed artifact matrix | `OPEN` |

## 3. 本轮已修复或 fail-closed 的项目

| 项目 | 关键改动 | 验证 | 证据边界 |
|---|---|---|---|
| 一次 full forward 假 decode | 新增 prefill + `past_key_values` + 逐 token 路径，prefill route 清空，KV+1、EOS/max-step 检查 | tiny OLMoE cached-vs-full 2 tests PASS | development batch-1，不是 formal continuous engine |
| identity schema | `Contribution` 新增 input event/token/decode step/layer/slot/source/target；新增 stage conservation helper | identity/unit contracts PASS | replay 尚未接入全 stage |
| 弱 curve 自升格 | latency-only benchmark 固定 `formal_eligible=false` / `routeslack_gate1_eligible=false` | metadata 检查 | 关闭 false PASS，未建成 energy surface |
| Gate1 来源证明 | non-smoke census 强制 explicit v2 trace meta 与 RouteSlack-eligible curve meta | missing meta 负测试 PASS | 仅关闭 census 入口 |
| NVML sampler 异常/绕回 | 后台错误同步传播；显式 modulus counter delta | wrap、thread failure、gap、idle sensitivity tests PASS | workload/counter atomic alignment 仍未闭合 |
| JouleQueue capture NameError | `_source_hash()` 使用正确 `CJC_EXPERIMENTS` | source hash test PASS | 仅工程可运行修复 |
| fail-closed dry-run | 新增 RouteSlack contracts、10 baseline + Oracle synthetic pipeline、manifest/hash、out-of-range fallback、host no-op | 96/96 tests PASS；dry-run `formal_result=false` | 不是 GPU/科学证据 |

## 4. 计数口径

- `[Observed]` 未闭合 P0 14 项；任一项都阻断 GPU formal Gate。
- `[Observed]` 未闭合 P1 7 项；正式结果前必须闭合。
- `[Observed]` 未闭合 P2 3 项。
- `[Observed]` 本轮修复/降级误用风险 7 类，但不从未闭合数量中扣除“部分修复”。
