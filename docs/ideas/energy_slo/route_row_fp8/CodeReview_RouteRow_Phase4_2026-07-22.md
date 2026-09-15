# Route-row FP8 Break-even — Phase 4 严格 Code Review

状态：**BLOCKED**  
日期：2026-07-22  
审查角色：独立 Code Review / 指标会计门禁  
冻结协议：`Phase2_RouteRow_FP8_BreakEven_冻结协议_2026-07-22.md`（SHA-256 `8ff2f11142d4cd15d3d5c820811333e177e56b8c896d6792fc8431bb092cd736`）  
审查基线 commit：`242066439f05e183c6a0e907383075bc18871291`；新增实现尚未提交，因此以下文件 SHA-256 同时构成本次审查快照。

| 文件 | SHA-256 |
|---|---|
| `experiments/route_row_policy.py` | `d95fea01f00ffd874960e437720ff9926597a013e0adabaede048b7c3d7caf9e` |
| `experiments/run_route_row_surface.py` | `deac85a10d18b8636c8852fb3d435421dee6134c85b20e8401bb17440c191b8f` |
| `experiments/continuous_decode_harness.py` | `487a411462a5a4f0e7cfcc8c13376f9791e47d36e6ef7575f650702edb103f7a` |
| `experiments/power_accounting.py` | `6f2d61c48d217a4673a7e8f7a609e04986e9ee7bbadba9678b7173871e97fb00` |
| `experiments/configs/route_row_break_even_v1.json` | `a1bb15f67fdffee954e53ffd9337e22e454daa761e2f1fb83c5f548aa00176eb` |

## 结论

**不得进入 Phase 5 正式实验。** 当前交付是一个明确标注为 calibration proxy 的离线 expert-surface 工具、一个抽象 scheduler/ledger harness 和会计工具；它不是冻结协议要求的真实 continuous-serving、router 后/expert 前动态 per-expert BF16/FP8 hot path。存在多个未关闭 P0，其中两个还是可直接复现的 fail-open：字符串 `"false"` 会被解析为质量门已通过；任意测试 backend 只要自报 capability，就会被标为 `scientific_result_eligible=True`。

本结论是 **实现 BLOCKED**，不是机制科学 No-Go，也没有任何科学结论可翻转。`run_route_row_surface.py` 对自身代理边界的标注是诚实的，不能用其 local crossover 产物替代 Phase 5。

## 主张—实现一致性

| 冻结要求 | 当前证据 | 判定 |
|---|---|---|
| 真实 continuous engine、一次 prefill、长度 1 KV decode | 只有 `ContinuousDecodeBackend` Protocol 和可自报的 capability；无真实 backend | **P0 未实现** |
| 当前层 router 后、expert GEMM 前读取实际 `m[l,e,t]` 并执行 LUT | `RouteRowLUT.decide()` 只被 existence summary 调用；surface 对保存的 activation 手工分别跑 BF16/FP8 | **P0 未实现** |
| 动态 per-expert 完整三投影 FP8 | wrapper 覆盖 gate/up/down、每投影动态 activation cast、`torch._scaled_mm` 输出 BF16 | 机制代码存在；**未接 hot path / GPU 未验证** |
| 双权重驻留且所有 B0–B4/C 公平 | surface 同时持有 BF16/FP8 expert 权重并报字节；无全 arm runner、KV/batch capacity 选择 | **P0 未闭合** |
| Poisson/MMPP、batch grid、B0–B4/C、full drain | 没有正式 runner，也没有这些 workload/arm 实现 | **P0 未实现** |
| TTFT/TPOT、P99 ratio CI、violation、J/completed token | 有基础 ledger 与能量积分；无联合主指标、bootstrap/seed/GO 门 | **P0 未实现** |
| 独立 KV/in-loop KL 质量门 | surface 只读取外部 JSON；无质量 runner，且 JSON 未绑定具体 LUT | **P0 不可信** |
| Phase-4 signoff + 全代码/config hash | surface 有 hash gate，但 manifest 只含 4 个文件，遗漏 harness、power、未来 backend/正式 runner/capture/quality producer | **P0 不完整** |

## P0 — 必须关闭，否则禁止正式运行

### P0-01：核心 causal hot path 不存在

- 冻结协议第 97–110 行要求真实 continuous serving、独立 KV、禁止 full-forward/route replay，并要求在同一 hot path 动态切换。
- `run_route_row_surface.py:2-9` 明确承认不集成 serving hot path；`653-713` 只是在保存的 expert activation 上重复执行两个精度 arm；`1011-1020` 把产物标为 `INELIGIBLE_CALIBRATION_PROXY`。
- 全实现中没有 runtime 对 `RouteRowLUT.decide()` 的调用；唯一调用位于 `route_row_policy.py:323` 的 existence 汇总。实际 router 结果从未驱动实际 expert 执行。
- 仓库中不存在协议建议的 `run_route_row_energy_slo.py` 或等价正式入口。

**关闭条件：** 在固定 engine/commit 内实现可审查 backend；同一层 router/top-k 产出后计算 `m`，在任何 expert GEMM 前调用 LUT，随后在该 expert 的同一次调用内完整执行 BF16 或三投影 FP8。必须输出层级 row closure、action、三类 counter、独立 KV identity、engine commit 和 hook 代码 hash；不得通过保存 activation 后离线 replay 代替。

### P0-02：LUT 反序列化可绕过质量 hard gate

- `route_row_policy.py:361-374` 使用 `bool(value["quality_gate_passed"])`。Python 中非空字符串 `"false"` 会变成 `True`。
- `RouteRowLUT.__post_init__` 允许 `quality_gate_passed=True` 同时 `quality_artifact_sha256=None`。
- 实际复现：把序列化字段改为字符串 `"false"` 后，`from_dict()` 得到 `True / None / FP8`，即质量证据缺失仍选择 FP8。

**关闭条件：** JSON 字段必须严格为 boolean；质量通过时必须要求格式正确、存在且可校验的 artifact SHA-256，并绑定 model/data/arrival/LUT/code hash。新增回归测试：字符串、整数、null 和缺失字段全部 hard-fail，不能静默转型。

### P0-03：formal eligibility 由 backend 自证，可把测试替身伪装成科学运行

- `continuous_decode_harness.py:240-263` 的 capability 全是 backend 自报布尔值；`419-433` 只检查这些布尔值；`623-634` 直接以 `formal` 参数决定 `scientific_result_eligible`。
- 实际复现：把现有 `AuditOnlyBackend.real_continuous_engine` 从 `False` 改为 `True`，无需真实 GPU/engine，formal harness 返回 `COMPLETED True`。
- harness 自身没有 Phase-4 signoff/code hash gate，也没有 engine identity/commit 的强绑定。

**关闭条件：** 删除“`formal=True` 即科学合格”的路径。只有顶层正式 runner 在具体验收过的 backend 类型、固定 engine commit、完整代码 hash、GPU/数据/config 身份全部通过后，才能生成 eligibility；测试 backend 即使伪报 capability 也必须 BLOCKED。新增对应负向测试。

### P0-04：质量 artifact 未绑定候选 LUT，且生产路径缺失

- `run_route_row_surface.py:425-468` 只核对模型、recipe、data/arrival hash 和若干自报字段，不要求候选 LUT/policy hash。
- 质量 artifact 在 LUT 于 `971-980` 构造之前加载；同一份 PASS JSON 可被不同 surface/LUT 重用。
- 没有实现独立 KV、允许 deeper-layer route divergence 的 in-loop KL 质量 runner，因此当前 PASS JSON 没有可审查 producer。

**关闭条件：** 先冻结候选 policy/LUT 内容并生成 hash，再由独立质量 runner 消费该精确 hash；artifact 必须绑定 LUT、policy code、model revision、canonical document manifest、arrival/config 和 engine commit。任何一项漂移必须回退 BF16/BLOCKED。

### P0-05：B0–B4/C、arrival、batch-capacity 与联合 Go/No-Go runner 缺失

- 冻结协议第 111–186 行规定 Poisson/MMPP、5+5 seeds、32 warm-up/256 measured/full drain、batch grid 和 B0–B4/C。
- 当前 config 没有 arm/workload/batch-grid 实例，harness 也没有 `max_num_seqs`、arrival generator、warm-up/measured 分界、baseline completeness gate、P99 ratio CI 或 GO 判定。
- surface 只记录权重字节与 CUDA allocated peak（`run_route_row_surface.py:920-949,1046-1049`），没有在相同双驻留条件下为每个 arm 求可行 batch/KV capacity。

**关闭条件：** 实现单一正式 runner，逐 arm 复用完全相同请求/arrival/hash，保持双驻留，按 calibration 选择 max-feasible cap，完整 drain 后再关能量窗口；缺任一 B0–B4/C cell 必须禁止 verdict。主口径先算，敏感性后算。

### P0-06：capture 的因果与 engine 属性只靠 JSON 自述

- `_load_capture()` 在 `run_route_row_surface.py:243-278` 核对 metadata 值，但没有 capture producer，无法证实 hook 真位于 router 后/expert 前、没有未来信息、没有 BF16 route replay，也无法证实 `O(1)` cache metadata。
- 现有检查能验证 row/expert identity 和 active-set 变化（`326-419`），但不能验证声明的因果来源。

**关闭条件：** capture producer 必须作为受 hash/signoff 管理的代码交付；trace 需包含 hook point、iteration/layer 顺序、request/KV identity 和 producer hash。metadata 自报不能作为 causal proof。

### P0-07：formal hash manifest 没有覆盖正式执行链

- config `formal_gate.hash_manifest` 仅含 policy、surface、一个 test 和 config（`route_row_break_even_v1.json:101-105`）。
- 它遗漏本次要求审查的 `continuous_decode_harness.py`、`power_accounting.py`，也不可能覆盖尚不存在的 backend、capture/quality producer 和正式 runner。

**关闭条件：** manifest 覆盖所有生产代码、冻结 config/协议、backend adapter、capture/quality producer 和正式 runner；依赖版本、engine commit 另行写入不可变 run manifest。任何生产文件漂移必须拒跑。

### P0-08：native GPU recipe 与两模型 target 尚未验收

- CPU 单测明确不执行 CUDA FP8；没有审查产物证明 RTX 5090 上三次 `_scaled_mm`、每投影 activation cast、初始化一次 weight cast、空 expert 零 kernel，以及实际 OLMoE/LLM-jp target 分别为 3072/1536。
- config 常量测试不能替代加载两个 pinned revision 后的实际 module identity 验证。

**关闭条件：** 在目标 GPU 上对两个 pinned revision 运行只读 smoke，保存环境、CUDA/PyTorch/driver/GPU UUID、module count、kernel probe、counter 与数值误差产物。未过时保持 `UNVERIFIED/BLOCKED`，不能改用 CPU 或 random-activation 数字。

## P1 — P0 关闭前一并修复

### P1-01：SLO violation 口径不符合冻结协议

`EventLedger.summary()`（`continuous_decode_harness.py:197-236`）只以 `completion_ns > slo_deadline_ns` 计算 violation；冻结协议分别以 B0 定义 TTFT/TPOT SLO，还要求 P99 ratio CI 和 violation-rate 增量。一个 completion deadline 不能替代两个门。

**修复：** 分别存储/计算 request TTFT 与 TPOT violation；顶层按 paired seed/bootstrap 计算 P99 ratio UCB 和 violation-rate delta UCB。不能拿 completion deadline 充当联合 SLO。

### P1-02：`teacher-forced decode=64 steps` 存在 off-by-one

formal 检查要求 `len(output_token_ids)==64`（`continuous_decode_harness.py:427-433`），但 token 0 在 prefill 产生（`485-510`），length-1 decode 从 index 1 开始，因此只执行 63 次 decode。冻结文档写的是 64 decode steps。

**修复：** 在 Phase 2 口径不变的前提下实现并断言每请求恰好 64 次 length-1 decode；若原意是“64 个输出 token、其中首 token 来自 prefill”，需先修订冻结协议并把旧协议标 `SUPERSEDED`，不可运行后再解释。

### P1-03：fallback power sampling 只约束目标周期，不审计实际间隔

`MonotonicNVMLSampler` 只检查构造参数 `interval_s<=20ms`，`PowerTrace` 不检查实际相邻 timestamp；surface 内另一套 NVML meter 也相同。100 ms 相邻样本的 trace 当前会被接受。线程调度抖动可使真实间隔超过 20 ms。

此外，`MonotonicNVMLSampler.start()` 没有内建 GPU synchronize/t0 release 契约；总能量 counter 起点与显式 power sample 边界也不是同一个读数时刻。

**修复：** formal trace 对实际最大 gap 做 hard gate；顶层 runner 明确在首 measured arrival 释放前 synchronize，并记录同步/释放/计量边界。若 counter 不支持且任一 gap 超门，整个 cell BLOCKED。

### P1-04：generic NVML backend 的设备身份必须在 runner 中强制执行

`PynvmlPowerBackend` 按 NVML physical index 打开设备（`power_accounting.py:268-279`）；在 `CUDA_VISIBLE_DEVICES` 重映射下不等价于 CUDA logical index。虽然提供了 `assert_matching_gpu_uuid()`，目前无正式 runner 强制调用。

**修复：** 与 surface meter 一样按 CUDA UUID 寻找 NVML handle，或在开始计量前无条件核对并把 UUID 写入 run manifest。

## P2 / P3

- **P2：重复会计实现。** surface 自带 `NvmlBoardEnergyMeter`，没有复用 `power_accounting.py`，两套边界/UUID/采样语义容易漂移。正式路径只保留一个经签字实现；surface 若保留，继续标 proxy。
- **P2：artifact 覆写风险。** `write_json_atomic()` 对固定输出文件直接 replace，没有 run-id/manifest 防止覆盖旧证据。正式 runner 应以 immutable run directory 写入，并拒绝覆盖 sealed 产物。
- **P3：无。** 当前问题均不是样式问题。

## 已通过但仅限基础逻辑的项目

- row bins 与 `sum_e m = active_tokens × top_k` 的纯逻辑检查正确；重复 top-k expert id 会拒绝。
- 缺失/underpowered/CI 跨零/错误 SurfaceKey 在直接构造的正常类型 LUT 上回退 BF16；`m=0` 返回 SKIP。
- FP8 wrapper 的代码路径覆盖 gate/up/down 三投影，weight cast 在初始化，activation 每投影量化，scaled-mm 输出 BF16；但这是代码阅读结论，GPU 仍未验证。
- surface 对自身 `CALIBRATION_PROXY_ONLY` / `INELIGIBLE` 的标识清楚，没有在当前 runner 中生成 GO。
- power trapezoid、completed-token 分母和 synthetic `100W×10s/100=10J/token`、idle 30W 得 7J/token 的基础恒等式通过。

## 本次验证记录

```text
PYTHONPYCACHEPREFIX=/tmp/energy_phase4_pycache \
  ./.venv/bin/python -m unittest discover \
  -s docs/ideas/energy_slo/route_row_fp8/experiments -p 'test_*.py' -v

结果：27 tests passed（CPU logic only）
```

`py_compile` 通过；`git diff --check -- docs/ideas/energy_slo` 通过。`git-ai` 作者上下文查询因 daemon lock 未取得历史，因此所有判断均来自当前冻结协议、代码和可复现实验，不推测作者意图。

## 重审入口

只有以下条件全部满足才接受下一次 Phase 4：P0-01 至 P0-08 全部关闭；P1 会计项修复；真实 backend/正式 runner/capture producer/quality producer/全量 tests 均进入 hash manifest；两个目标模型的 GPU smoke artifact 可复核。重审前不得创建 `SIGNED-OFF` JSON，不得把 surface 的 local diagnostic 写入 Phase 5 结果表。

## 返修复审（2026-07-22）

复审状态：**BLOCKED（维持）**。返修关闭了两个真实 fail-open，并补齐当前 proxy bundle 的大部分 hash 覆盖；这不等于实现了被冻结的正式实验链。

返修快照：

| 文件 | 返修后 SHA-256 |
|---|---|
| `experiments/route_row_policy.py` | `9bdac6ba2ed59749c608efb6ce89b88be38c17d0ae4cf5778b2f742882f22d89` |
| `experiments/continuous_decode_harness.py` | `5a129b43071e41f03b9898055300a19b98e24b7287326a614a4c540571208601` |
| `experiments/power_accounting.py` | `bcf8cb2adb9cd02189b5aaa4037da1d2bf466798f6bbb6e87a5244dbcaaad2d3` |
| `experiments/configs/route_row_break_even_v1.json` | `2c950053beb157fda45fff87a03601c09b0d090ceffbf4e8e324b70776509d1b` |

当前 manifest 的 `code_config_sha256` 为 `617931899f0647146a473ac6ff6337ff02c9ad295d34f38a084304a61554fbc4`。

### 已关闭

- **P0-02 已关闭。** `RouteRowLUT` 现在严格要求 JSON boolean；`quality_gate_passed=true` 必须带格式合法的 lowercase SHA-256。独立复现确认原 `"false" → True → FP8` 路径现在抛出 `TypeError`。新增测试覆盖字符串、整数、null、缺失字段及非法/缺失 quality hash。
- **P0-03 已关闭。** abstract harness 在 formal 模式完成调度后仍强制返回 `BLOCKED`，不再以 backend 自报 capability 授予科学资格。独立复现中伪报 `real_continuous_engine=True` 的 test backend 返回 `BLOCKED / scientific_result_eligible=False`。
- **P1-03 在 `power_accounting.py` 路径已关闭一半。** formal sampler 和 `account_power_trace(formal=True)` 都会按实际 timestamp 检查最大相邻 gap；100 ms 负例现在抛出 `RuntimeError`。
- **P0-07 的“遗漏当前已有 harness/power/tests”已关闭。** config manifest 已覆盖当前 4 个生产/工具文件、3 个测试和 config，并有集合相等测试。

### 仍开放的 P0

- **P0-01 仍开放：** 没有真实 continuous-serving backend，也没有 router 后/expert 前调用 LUT 的动态 hot path；`run_route_row_surface.py` 未变，仍是离线 calibration proxy。
- **P0-04 仍开放：** 新校验只保证 `quality_artifact_sha256` 的字符串格式和存在性；并未验证该文件，也未把 artifact 绑定到具体 LUT/policy hash。in-loop、独立 KV、允许 deeper-route divergence 的 quality producer 仍不存在。
- **P0-05 仍开放：** B0–B4/C、Poisson/MMPP、batch grid、warm-up/measured/full-drain、P99 ratio CI、联合 Go/No-Go 正式 runner 均未实现。
- **P0-06 仍开放：** capture producer 与 hook 仍不存在；causal/engine/cache 属性继续只靠 capture metadata 自报。
- **P0-07 仍部分开放：** 当前 proxy bundle 已纳入 manifest，但冻结协议、尚不存在的 backend、capture/quality producer 和正式 runner 不在执行链中，因而现在仍不能创建 Phase-4 `SIGNED-OFF`。这些文件实现后必须先加入 manifest 再重审。
- **P0-08 仍开放：** 没有两个 pinned 模型在目标 GPU 上的 module-count、三投影 native FP8、counter、空 expert 和数值误差验收 artifact。

### 仍开放的 P1

- **P1-01：** completion deadline 仍未变成冻结协议的独立 TTFT/TPOT SLO 与 paired CI。
- **P1-02：** formal 请求仍为 64 output tokens，其中首 token 来自 prefill，实际只有 63 次 length-1 decode；冻结文档的“decode=64 steps”歧义未解决。
- **P1-03：** `run_route_row_surface.py` 自带的 `NvmlBoardEnergyMeter` 未修改，fallback 路径仍只限制目标 sampling interval、不拒绝实际 gap 超过 20 ms；generic 会计修复不能自动保护 surface proxy。
- **P1-04：** generic NVML physical-index 与 CUDA logical-index 的 UUID 强制绑定仍等待正式 runner。

### 复审验证

```text
32 tests passed（CPU logic only）
py_compile: passed
git diff --check -- docs/ideas/energy_slo: passed
```

三个针对原缺陷的独立负向复现结果：

```text
lut_fail_open=NO TypeError
formal_spoof=BLOCKED False
gap_fail_open=NO RuntimeError
```

因此本次返修可以作为 Phase 3 局部纠错合入，但 **Phase 4 仍不得签字，Phase 5 正式 Energy-SLO 实验仍禁止启动**。允许继续运行的只有明确标为 `INELIGIBLE_CALIBRATION_PROXY` 的开发 smoke，且不能进入科学结果表。
