# RouteSlack-MoE 实验代码审查

> 审查日期：2026-07-28  
> 结论：`Gate 0 FAIL`  
> 发现总数：`P0=16（CLOSED 7 / OPEN 9），P1=8，P2=2`。`CLOSED` 仅表示对应的代码级误用已修复或 fail closed，不表示 formal Gate 已通过。

## 1. P0：会改变科学结论

| ID | 文件 | 行号 | 级别 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---|---:|---|---|---|---|---|---|
| P0-01 | `capture_native_routes.py` | 103–217, 367–463 | P0 | 历史 `--phase decode` 用一次 full forward 冒充 decode | route/KV/step identity 均不真实 | 增加 prefill + `past_key_values` + 单 token step、KV 长度、EOS/max-step 和逐步 route closure | `test_cached_decode.py:94–173` | `CLOSED_DEVELOPMENT`；仍非 continuous serving |
| P0-02 | `core.py`; `capture_native_routes.py` | 50–335; 315–363 | P0 | v1 缺少显式 input-event/decode-step/top-k/source/target identity | aggregate 数量可隐藏 drop、duplicate、错层或错 replica | 新增 route-v2、immutable contribution key、四阶段 conservation 和 strict-v2 loader | `test_identity_conservation.py:27–103` | `CLOSED_DEVELOPMENT` |
| P0-03 | `benchmark_expert_service_curve.py` | 95–124, 147–189 | P0 | latency-only isolated expert curve 可能被误当 RouteSlack service-energy surface | latency proxy 不是 Joules/E2E，也无 natural independent unit | 强制 `formal_eligible=false`、`routeslack_gate1_eligible=false` 并列出 blocker | `test_bcrd_gate.py:60–84` | `CLOSED_FAIL_CLOSED` |
| P0-04 | `census_fragmentation.py` | 73–98, 133–136 | P0 | development trace/curve 可进入非 smoke Gate-1 | 弱 provenance 可被升格为 formal evidence | 非 smoke 强制 route-v2、trace metadata 和 Gate-1-eligible curve metadata | `test_bcrd_gate.py:60–84` | `CLOSED_FAIL_CLOSED` |
| P0-05 | `power_accounting.py` | 221–345 | P0 | sampler 后台异常可静默；counter 倒退可被猜成 wrap | 可漏掉功率区间或制造能耗差 | 在 `stop()` 同步传播线程异常；只有显式 modulus 才接受 wrap | `test_power_accounting.py:117–143` | `CLOSED_UNIT` |
| P0-06 | `capture_joulequeue_expert_inputs_gpu.py` | source-hash dependency block | P0 | source-hash 使用错误符号，正式 capture 启动即 `NameError` | artifact provenance 无法生成 | 使用实际 `CJC_EXPERIMENTS` 路径并加入回归测试 | `test_capture_joulequeue_expert_inputs_gpu.py:39–42` | `CLOSED` |
| P0-07 | `gate0_contracts.py` | 151–217 | P0 | completed-token denominator 曾按 token×layer 重复计数 | J/token 会按层数被系统性缩小 | denominator 按输出 token key 聚合并要求 sibling closure | `test_gate0_contracts.py:87–112` | `CLOSED_UNIT` |
| P0-08 | `capture_native_routes.py`; serving integration | 220–223, 367–463 | P0 | 当前仅 batch-1 离线 decode；arrival/deadline 合成，无 mutable active set、ready time 或因果 wave | 不能推断 continuous batching、queueing、TPOT/P99 或 action slack | 接入真实 continuous-serving producer，落盘 per-request KV owner、ready/completion timeline | 尚缺双模型 serving E2E | `OPEN_BLOCKS_GATE0` |
| P0-09 | `benchmark_expert_service_curve.py`; RouteSlack runner | 95–124 | P0 | 无双模型 natural rows×tier 同 trial latency-energy surface | H1 没有物理自变量/因变量，Oracle 也无成本表 | 新建 CUDA event + raw board-energy + tier/thermal runner，以 fresh input event 为样本 | 尚缺 Experiment A | `OPEN_BLOCKS_GATE0` |
| P0-10 | `run_joulequeue_expert_surface.py`; shared meter | 176–251, 401–470, 674–763; `power_accounting.py:163–324` | P0 | 现有能耗路径是 sequential bracketing，arm 可有不同 repeat/window，缺 matched completion meter | 固定读表开销、窗口长度和完成集合可伪造节能 | 同 repeat ABBA；保存 counter/sample/workload 边界；主分母用 exact matched completed tokens | pure contracts 已有；缺 GPU integration | `OPEN_BLOCKS_GATE0` |
| P0-11 | JouleQueue/route-row/formal runner | 多文件 | P0 | 缺 fresh input-event variance、temperature/clock/power-limit/util/throttle/idle raw timeline | thermal/clock drift 可能大于策略差，inner repeat 会造成伪重复 | 统一 meter/thermal schema，按 document/request bootstrap，pair gate fail closed | thermal/counter unit tests 已有；缺真实 NVML | `OPEN_BLOCKS_GATE0` |
| P0-12 | `build_fixed_replica_instances.py`; `census_fragmentation.py` | 46–96; 35–51, 100–165 | P0 | replica/tier 选择仍是 replay/虚拟 action；未绑定真实 placement、拓扑和 power-tier realization | H2 的“可执行性”可能只是模拟器选择空间 | 绑定 source-rank→合法 target、device UUID/topology/tier，并在真实 executor 验证 action | 尚缺 EP/actuator integration | `OPEN_BLOCKS_H2` |
| P0-13 | `solve_assignment_oracle.py`; `core.py` | 57–134; 514–601 | P0 | Oracle 目标仍以 latency/service 为主，漏 raw energy、dispatch/combine/return、idle、switch/hold/controller tax | 会高估 H3 上限且可选择物理不可执行 action | matched-set raw-energy objective；用保守 surface 并计入全 DAG/tax | 只有接口隔离测试；无 energy Oracle | `OPEN_BLOCKS_H3` |
| P0-14 | `policies.py`; `compare_policies.py`; RouteSlack dry-run | 19–43; 17–177; dry-run policy fixture | P0 | dry-run 只用任意 cost 常量调用 10 个策略名称；不是 10 个冻结算法的实现或物理比较 | 不能计算 strongest-simple、CaptureRatio 或 Gate 3 | 实现各 baseline 的真实状态机/校准/动作，锁定同 trace/SLO/completion set | registry/interface 测试仅覆盖 plumbing | `OPEN_BLOCKS_GATE3` |
| P0-15 | capture→surface→census→Oracle→Gate3；`compute_captured_headroom.py` | 多文件；43–68 | P0 | 全链 revision/config/hash join 与 request/document cluster split 未闭合；README 的 Gate-2 remote=0/5/15 sweep 会为每个 instance 生成三条 Oracle row，但 headroom 仍要求恰好一条 | artifact 换件、calibration leakage 或 remote-cell 错配可产生 false PASS；文档化 Gate-3 链也会直接报 `need exactly one Oracle remote-cost row` | 每阶段严格验证 manifest/hash/model revision/config；按 request/document cluster 一次 split；将 selected remote cell 写入 plan 并按 `(instance_id, remote_latency_us)` 严格 join | artifact-substitution/split E2E + 三 remote-cell documented-command integration test | `OPEN_BLOCKS_FORMAL` |
| P0-16 | 两冻结模型的 instrumented serving path | 多文件 | P0 | 未完成 patched/unpatched final-output exactness、真实 routed→combined ledger 和 E2E matched denominator | 无法证明 `Delta Q=0`，也不能排除少做工作换能耗 | 在两模型真实 serving 上逐请求校验 output hash、stage multiset 与 completion set | tiny CPU tests 只证明 helper | `OPEN_BLOCKS_GATE0` |

## 2. P1：正式结果前必须修复

| ID | 文件 | 行号 | 级别 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---|---:|---|---|---|---|---|---|
| P1-01 | `benchmark_expert_service_curve.py` | 95–123 | P1 | 同一 random tensor 的 inner trials 无 event/layer/expert 方差 | 把 inner repeat 当独立样本会缩窄 CI | fresh input event + hierarchical bootstrap | 样本 cluster 测试 | `OPEN` |
| P1-02 | `power_accounting.py` | 348–400 | P1 | UUID helper 未强制接入每个 CUDA runner | 可能把另一块 GPU 的 NVML 数据配给 workload | 启动时强制 NVML/CUDA UUID 一致 | `test_power_accounting.py:90–94`；缺 runner test | `OPEN_INTEGRATION` |
| P1-03 | shared meter/formal runner | `power_accounting.py:163–218` | P1 | idle power 由 caller 传入，缺独立 idle window/CI/provenance | dynamic energy 对 idle 选择敏感 | 配对 idle calibration，raw 指标保持主结果 | `test_power_accounting.py:126–134` | `OPEN_INTEGRATION` |
| P1-04 | `census_fragmentation.py` | 205–229 | P1 | common-cell/exposure/threshold 逻辑仍未与 energy surface、同一 action opportunity 和两模型 manifest 原子 join | 可能跨 cell 拼接门槛或错配 exposure | 将 effect/LCB/15%/actionability/energy mass 按同 cell、双模型 AND 计算 | 多 cell 反例 E2E | `OPEN` |
| P1-05 | `test_cached_decode.py`; capture path | 94–173 | P1 | tiny random wrapper 只覆盖 CPU development；未覆盖两个冻结模型原生 hook | helper 正确不等于真实模型集成正确 | 两模型 GPU cached/full 和 patched/unpatched exactness | 现有 tiny 测试 | `OPEN_FORMAL` |
| P1-06 | `experiments/shared/prompts.py`; formal runner | 多处 | P1 | formal prompt manifest、revision、每 cell 128 个独立 event 尚未锁定 | workload 漂移或 N 不足会改变 route coverage | 保存 dataset split/revision/hash 和 cluster IDs | manifest/cardinality tests | `OPEN` |
| P1-07 | `core.py` `ServiceCatalog` | 356–423 | P1 | legacy 插值/key 缺 revision、expert、tier、dtype、GPU UUID；不是统一 conservative lookup | Oracle 可能跨设备/档位复用错误成本 | 迁移到 exact/ceiling surface key，越界 default fail closed | `test_routeslack_protocol.py:123–134` | `OPEN_MIGRATION` |
| P1-08 | `run_routeslack_dry_run.py` | 263–378 | P1 | no-op 只是 CPU Python fixture，无真实 hook/logging/CUDA/NVML/E2E | 不能作为 controller tax | 在真实 serving 中做 paired no-hook/hook/no-op GPU 测量 | host-only raw timing 已保存 | `OPEN_GPU` |

## 3. P2：工程与可维护性

| ID | 文件 | 行号 | 级别 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---|---:|---|---|---|---|---|---|
| P2-01 | `gate0_contracts.py`; `routeslack_protocol.py`; 三套 meter | 多处 | P2 | contract、identity 和 meter 实现重叠，错误语义可能漂移 | 后续修复可能只落在一套实现 | 合并 canonical contract，其余只做 adapter | shared conformance suite | `OPEN` |
| P2-02 | legacy replay/parser/helpers | 多文件 | P2 | 重复 parser，混用 `ValueError/ProtocolError/AssertionError`，历史 probe 未统一标为 historical-only | 审计与故障定位成本高 | 统一 strict loader、错误 taxonomy 和 evidence label | malformed-artifact matrix | `OPEN` |

## 4. 计数与 Gate 解释

- `[Observed]` P0 总数 16：7 项已在 development/unit/fail-closed 层关闭，9 项仍开放。
- `[Observed]` P1 总数 8，P2 总数 2。
- `[Inferred]` 任何一个开放 P0 都足以让 Gate 0 保持 `FAIL`；因此当前不能运行或解释 GPU formal Gate。
- `[Blocked]` 当前环境没有 CUDA/NVML；即便换到 GPU，P0-08 至 P0-16 也必须先闭合。
