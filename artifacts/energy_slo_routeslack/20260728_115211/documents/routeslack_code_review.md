# RouteSlack-MoE 实验代码审查

> 快照：2026-07-28 working tree  
> 裁决：`16 P0（6 个代码级关闭，10 个未关闭）/ 8 P1 / 2 P2`  
> Gate 含义：任何未关闭 P0 都使 `Gate 0: FAIL`；“已修复”只表示 CPU 可验证的代码条件关闭，不等于 GPU/formal 资格。

## 1. P0 清单

| 文件 | 行号 | 级别 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---:|---|---|---|---|---|---|
| `docs/ideas/bcrd/experiments/capture_native_routes.py` | 103–202, 411–462 | P0 | 原 `--phase decode` 是一次 full forward；当前改为 prefill 后 `use_cache=True`、传 `past_key_values`、单 token 循环并分 step 清空 hook | full forward 的 token position 不是 decode step，会伪造自然 route census | 新增 `run_cached_decode_steps`、cache 长度/EOS/layer-top-k closure；metadata 永久 `formal_eligible=false` 直到正式资格补齐 | `test_cached_decode.py:94–179` | **代码级已修复；双正式模型/serving 资格未关闭** |
| `docs/ideas/bcrd/experiments/core.py` | 22–42, 50–174, 204–344 | P0 | v1 identity 缺 `input_event/decode_step/token/layer/slot/source/target`，stage 检查原先也允许 dispatch 后改变 target | contribution 丢失、重复或改投 replica 会改变工作量和能耗分母 | 新增 v2 schema、显式 manifest closure、route semantic identity、四阶段守恒和 target 稳定性 | `test_identity_conservation.py:44–114` | **代码级已修复；真实 stage ledger 未接入** |
| `docs/ideas/bcrd/experiments/benchmark_expert_service_curve.py` | 159–181 | P0 | latency-only curve 曾可被误认 formal | 下游可能把无 energy/tier/thermal 的随机微基准当 Gate 1 surface | 强制 `formal_eligible=false`、`routeslack_gate1_eligible=false` 并列 blocker | `test_bcrd_gate.py:59–78`（consumer fail-close） | **已修复为不可晋级** |
| `docs/ideas/bcrd/experiments/census_fragmentation.py` | 73–101, 133–136 | P0 | consumer 原先不强制 route/curve provenance | smoke、v1 或开发态 capture 可被晋升为 formal census | 非 smoke 强制 route-v2 `formal_eligible=true` 与 Gate-1 curve metadata | `test_bcrd_gate.py:59–78` | **已修复为 fail-closed** |
| `docs/ideas/energy_slo/route_row_fp8/experiments/power_accounting.py` | 221–347 | P0 | sampler thread 异常可静默退出；累计 counter 回绕无显式语义 | 欠采样或负 delta 可能被包装成策略节能 | 保存并同步抛出 background exception；仅在声明 modulus 时处理 wrap | `test_power_accounting.py:117–159` | **代码级已修复** |
| `docs/ideas/energy_slo/joulequeue/experiments/capture_joulequeue_expert_inputs_gpu.py` | 104–112 | P0 | `_source_hash()` 引用不存在的 `RECEIVER_EXPERIMENTS`，主路径直接 `NameError` | capture 根本不能产生可追溯输入，却会被浅层单测漏掉 | 改为真实 `CJC_EXPERIMENTS` 并直接测试 hash | `test_capture_joulequeue_expert_inputs_gpu.py:39–43` | **已修复运行时错误** |
| `docs/ideas/bcrd/experiments/census_fragmentation.py` | 35–52, 103–131 | P0 | 同一 request 的多个自回归 decode step 被合入同一 virtual wave | 未来 step 本应依赖前一步 completion；错误合批会夸大 fragmentation/batching headroom | 必须改为逐 `input_event_id + layer_ready timestamp` DAG；当前未修改以免越过正式 serving schema | 无 | **未关闭** |
| `docs/ideas/bcrd/experiments/benchmark_expert_service_curve.py` | 85–124 | P0 | 只有单 expert、固定 `torch.randn`、CUDA latency；无 energy、tier、fresh input event、raw thermal | 无法回答 H1 的 rows×tier 能耗差异，也不能估 input-event CI | 另建 BF16 exact service–energy runner；不能把旧 CSV 补字段伪装 | 无 GPU 测试 | **未关闭** |
| `docs/ideas/energy_slo/joulequeue/experiments/run_joulequeue_expert_surface.py` | 160–252, 687–719 | P0 | counter window 包含不同 arm 的固定 envelope，且两 arm 独立选 repeats 后分别相除 | 固定 meter/线程开销经不同分母会系统性制造 energy delta | formal pair 必须相同 repeat、同 workload；保存 meter-only calibration，counter 边界与 workload 显式对应 | RouteSlack 反例测试覆盖数学条件；旧 runner 未修 | **未关闭** |
| 同上 | 396–470, 639–674 | P0 | 每个 trial 重复使用同一 `pool[:rows]` activation，却把时间重复 bootstrap 成 independent trials | CI 只反映仪器重复性，不能代表 route/input 异质性 | 每 trial 使用独立 `forward_id/input_event` activation，并按 document/event 聚类 | 无 | **未关闭** |
| 同上 | 238–252, 723–769 | P0 | 不保存 counter 原值和全部 power samples；没有 temperature/clock/power limit/util/throttle timeline | 无法复核 window、thermal drift 或后台干扰，能耗会计不能闭合 | raw JSONL 保存每次读数/样本/状态；pair-level thermal gate | 无 | **未关闭** |
| `docs/ideas/bcrd/experiments/solve_assignment_oracle.py` | 57–135, 138–225；`core.py` 492–590 | P0 | “exact Oracle”只枚举 replica 与一个全局 hold，目标仍是 latency/service | 缺 power tier、raw board idle、switch/decision/holding tax、dispatch order、跨层/decode DAG；不是 RouteSlack Energy Oracle | 保留为 `BCRD_LATENCY_REPLAY_ONLY`；待 Gate 1 surface 后另建冻结 Energy Oracle | 现有 oracle legality 仅验证简化 replay | **未关闭** |
| `docs/ideas/bcrd/experiments/compare_policies.py` | 17, 45–85, 104–180 | P0 | 只有 hash/least-load/random/threshold/greedy/BCRD，且指标不是 J/SLO-token | 弱 baseline 会夸大 residual headroom，无法计算冻结 CaptureRatio | 按协议补齐 10 个 online baseline 与同 actuator Oracle；在 Energy Oracle 前禁止 Gate 3 | dry-run 只验证 10 个接口名，非算法结果 | **未关闭** |
| `docs/ideas/bcrd/experiments/build_fixed_replica_instances.py` | 45–68, 89–94 | P0 | 先切 token window，再按 `instance_id` split；同一 request/document 可跨 calibration/evaluation；缺 split 时还强制改首尾标签 | 参数调优可看到 evaluation 同源样本，CI 与 generalization 失真 | 先按 document/request cluster split，再构造 window；assert ID 交集为空 | 无 | **未关闭** |
| `docs/ideas/bcrd/experiments/compare_policies.py` | 104–127；`compute_captured_headroom.py` 55–77 | P0 | Gate 3 未逐行绑定 instance/surface/action-space hash；set join 隐藏重复；多 remote-cost 与 headroom key 不一致 | 可能把不同 Oracle/config 的结果错误拼接并报告 capture ratio | 完整 key + 唯一性 + hash equality assertion | 无 | **未关闭** |
| `capture_native_routes.py` | 501–519；`continuous_decode_harness.py` 367–403 | P0 | 当前 producer 仍无 natural continuous batching、真实 ready/dispatch/execute/combine、两冻结模型 exactness 和 end-to-end completion denominator | 即使单步 cache 正确，也不能把 route observation 归因成合法 actuator 的物理节能 | 接入真实 serving backend，保存 stage ledger/output hash；在此之前保持 formal false | tiny OLMoE/cache test 与 dev capture只证明开发路径 | **未关闭** |

## 2. P1/P2 清单

| 文件 | 行号 | 级别 | 问题 | 为什么影响科学结论 | 修复方式 | 对应测试 | 当前状态 |
|---|---:|---|---|---|---|---|---|
| `power_accounting.py` | 276–315 | P1 | counter 起点早于首 power boundary，终点晚于末 boundary；counter delta 与 sample trace 不是同一时间窗 | raw 与 dynamic 指标的窗口不可逐 ns 对齐 | 分别落盘 workload/counter/sample 四类边界与原值；formal runner使用同窗 pair | 无 GPU | 未关闭 |
| `power_accounting.py` | 351–380 | P1 | backend 按 physical index 打开；UUID helper 不强制绑定当前 CUDA logical device | CUDA workload 与 NVML meter 可能指向不同卡 | runner 启动时强制 CUDA/NVML UUID equality | UUID helper 单测存在 | 未接入 formal runner |
| `census_fragmentation.py` | 209–225 | P1 | `passing10` 与 `passing15` 可由不同 cell 分别满足 | 两个弱 cell 可被拼成一个 PASS | 要求同一 common cell 的 effect/LCB/actionability 全部通过 | 无 | 未关闭 |
| `census_fragmentation.py` | 54–70 | P1 | exposure 只按 model/phase/layer/concurrency 聚合，未逐 wave/identity join | energy/time denominator 可能属于另一批请求 | 逐 wave join并核对 completed identities | 无 | 未关闭 |
| `test_cached_decode.py` | 33–65, 68–152 | P1 | 随机 tiny OLMoE 的 cached/full 都走同一 wrapper；未证明两个冻结 checkpoint 的 patch 前后 exactness | 单测通过不能授权正式模型 | 两模型各自保存 upstream-unpatched 对照 logits/output hash | 当前 2 个测试 PASS | 未关闭 |
| `experiments/shared/prompts.py` | 84–90, 130–152；`capture_native_routes.py` 26–32 | P1 | WikiText-103 test 只有 61 个合格 document，但 capture 默认请求 128；train 是分片 Arrow，当前 fallback 只找单文件 | 默认 formal command 必然在采样前失败或改变 independent unit | 明确用 train+冻结 offset；fallback 支持 shard concat；保存 document manifest | 已复现 `requested [0:128] from only 61` | 未关闭 |
| `core.py` | 334–389 | P1 | latency catalog按 `(model,layer)` 聚合并线性插值；缺 expert/input-event/tier，端点 P95 连线不是自动 upper envelope | Oracle 可能低估未测 cell 成本 | RouteSlack surface 用 exact cell或保守 ceiling；超范围 default fallback | `test_routeslack_protocol.py:104–124` | 新契约已实现；旧 BCRD 未迁移 |
| `docs/ideas/bcrd/experiments/README.md` | 20–34 | P1 | 文档仍称 route-v1，与当前 producer v2 不一致 | 复现实验可能遗漏显式 identity 列 | 更新 README 与 schema migration | 无 | 未关闭 |
| `routeslack_protocol.py` 与 `gate0_contracts.py` | 全文件 | P2 | 两套 CPU contract 存在重叠 dataclass/energy helper | 后续维护可能出现阈值或分母漂移 | Gate 0 正式实现前合并为单一 schema；本轮不为美化而重构 | 两套共 30 tests | 未关闭，不阻断当前 fail-closed |
| 多个 legacy runner | 多处 | P2 | 长行、重复 meter/percentile/helper | 增加审计成本，但本身不产生当前裁决 | 正式路径确定后再清理 | 不适用 | 延后 |

## 3. 审查结论

- `[Observed]` 代码级修复关闭了“full forward 冒充 decode”、隐式 route identity、target 漂移、silent sampler failure、counter wrap 未声明、JouleQueue source hash 崩溃和 provenance 误晋级。
- `[Observed]` 仍有 **10 个 P0**：自然 serving timeline、因果 wave、physical surface、能耗原始证据、independent event、Energy Oracle、强 baseline、cluster split、artifact binding、两模型 exactness 均未闭合。
- `[Inferred]` 因此禁止把 95 个 CPU tests、tiny capture、BCRD smoke 或 dry-run 当作 Gate 0 PASS，更禁止继续实现 controller。
