# SemanticFence-MoE 实验跟踪器

> 生成时间：2026-08-09 20:21 +0800  
> 对应计划：`refine-logs/EXPERIMENT_PLAN_20260809_202112.md`  
> 总状态：`PLANNED_NOT_RUN`；当前没有新 GPU 结果。

## 唯一实验

`SF-P0`：OLMoE / RTX 5090 calibration-to-fresh-eval expert rebatching Pilot。

| ID | 阶段 | 目的 | 资源 | 状态 | 完成条件 / artifact |
|---|---|---|---|---|---|
| `SF-P0-DATA` | local preflight | 生成 fresh-32 eval，排除 historical + 旧 42 docs | Mac CPU | TODO | 新 manifest、provenance、exclusion report 和 SHA；未查看结果 |
| `SF-P0-UNIT` | local preflight | 实现/验证 contract、packer 和 fail-closed 规则 | Mac CPU | TODO | CPU tests 全过；无漏/重 row、无 split 泄漏、unknown→M1、tamper fail |
| `SF-P0-ACCEPT` | GPU acceptance | 观测并冻结实际 5090 stack | RTX 5090 | WAITING_FOR_VERIFIED_GPU | acceptance artifact 绑定 UUID/driver/CUDA/torch/transformers/cuBLASLt/model/source |
| `SF-P0-CAL` | calibration | 建立并 seal executor contract | RTX 5090 | TODO_AFTER_ACCEPT | calibration-only evidence、M1 refs、10 repeats、expected signatures、`CONTRACT.json` hash |
| `SF-P0-EVAL` | fresh evaluation | 在同一 fresh row multiset 上执行 A/B/C/D | RTX 5090 | TODO_AFTER_CAL | raw per-row bits、call ledger、timing、trace replay、coverage、summary |
| `SF-P0-DECIDE` | aggregation | 按冻结判据输出 support/weaken/unable | CPU | TODO_AFTER_EVAL | parent recompute 完成；`COMPLETE.json` 最后写入 |

## 前三个实际 launch

### 1. `SF-P0-UNIT`

- 预期命令（实现后）：`python -m unittest docs/ideas/semanticfence/experiments/test_executor_contract.py docs/ideas/semanticfence/experiments/test_run_pilot.py`
- 失败动作：只修复实现或测试装置；不改变 claim、M grid 或支持阈值。

### 2. `SF-P0-ACCEPT`

- 预期命令骨架：`python docs/ideas/semanticfence/experiments/run_pilot_5090.py acceptance --config docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json --output-dir <new-accept-dir>`
- 注意：runner/config 尚未实现，该命令当前不可执行；先记录骨架，禁止伪装成已运行。
- 失败动作：若无 GPU 或 stack 不匹配，记录资源阻塞；不复用旧 Spectator UUID/lock。

### 3. `SF-P0-PILOT`

- 预期命令骨架：`python docs/ideas/semanticfence/experiments/run_pilot_5090.py run --config docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json --acceptance-artifact <acceptance.json> --frozen-lock <new-lock.json> --output-dir <new-run-dir>`
- 单次原子流程：calibration → contract seal → fresh A/B/C/D → trace replay → aggregate → `COMPLETE-last`。
- GPU hard cap：90 分钟；允许的“复测”只是在同一冻结配置下重跑一次，不允许结果后调阈值。

## 冻结判据

### SUPPORT

- A：全部 real rows 10/10 raw-bit stable；
- B：至少 8/64 victims 有稳定 mismatch；
- D：admitted fresh rows 0 mismatch，覆盖至少 8 victims、至少两个自然 `M>1`、0 padding；
- D：paired median CUDA latency 相对 A 至少下降 10%；
- C：没有在 exactness 与 latency 上同时不差于 D。

### WEAKEN

- B 为 0 mismatch；或
- D 退化为全 `M=1` / 单一 fixed M / padding；或
- D 任一 admitted fresh row 稳定 mismatch；或
- C 同时覆盖 D；或
- D exact 但速度收益低于 10%（只削弱局部收益 claim）。

### UNABLE

- A 不稳定；
- row/route/weight/dtype/order 不守恒或 actual M/signature 未变化；
- split 泄漏；
- raw evidence、trace closure、environment、artifact 或 completion 不完整；
- GPU 污染、超时或 signature 漂移。

## Claim coverage

| Claim | 关键 run | 主指标 | 当前状态 |
|---|---|---|---|
| C1：存在可泛化的非平凡 exact executor class | `SF-P0-CAL` + `SF-P0-EVAL` | admitted row mismatch = 0；≥2 个 M>1；≥8 victims | 未验证 |
| C2：保留局部 latency headroom | `SF-P0-EVAL` | D vs A paired median CUDA latency ≤ 0.90×；不被 C 覆盖 | 未验证 |

## 当前风险与处理

1. **GPU 未验证可用**：先完成不依赖 GPU 的 DATA/UNIT；实际 stack 由 acceptance 观测后生成新 lock。
2. **Contract 覆盖塌缩**：这是可解释负结果，不放宽 calibration 门槛救结果。
3. **Signature 漂移**：记为 unable/fail-closed；显式 algorithm selection 延后，不塞入本 Pilot。

## 明确裁剪

- 不跑旧 sealed replay 作为正式结果。
- 不加第二模型、第二 GPU、serving、EP、NCCL、RDMA 或 tenant baseline。
- 不修改任何旧 N05/Spectator artifact、lock、runner 或结论。

当前实验已经足以推进下一轮探索，停止继续扩展审计项。

