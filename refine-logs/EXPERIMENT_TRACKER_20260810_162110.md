# SemanticFence-MoE 实验跟踪器

> 更新时间：2026-08-10 16:21 +0800  
> 对应冻结计划：`refine-logs/EXPERIMENT_PLAN_20260809_202112.md`  
> 总状态：`GPU_PILOT_COMPLETE / WEAKEN_CURRENT_FORMULATION`。`run03` 是唯一写出科学 `COMPLETE.json` 的权威运行；`paper_result=false`，不授权论文结果或 full-layer / serving / EP / 多卡外推。

## 当前研究状态

- **研究方向**：SemanticFence-MoE：Route-Sealed, Regime-Certified Expert Rebatching。
- **Bottom-line problem**：MoE serving 为吞吐跨请求合并 expert rows 时，batch composition / execution shape 会改变底层执行结果；任意合批与全局隔离都不能同时满足可复现语义和性能。
- **Must-solve bottleneck**：找到一个既保持所选 exact semantics、又能放行非零 `M>1` 的可观测 operating region。
- **当前核心机制**：先密封 route ledger，再由版本与 stack 绑定的 calibration contract 决定哪些 expert rows 可稳定合批，未知状态回退 `M=1`。
- **目标结果**：先降低 expert-stage latency，后续才验证 request-level latency / throughput 传播。
- **当前阶段**：阶段 B“机制分解”已获得正式单卡证据；阶段 C“可控性”在当前 coarse executor-class formulation 上未通过。
- **最强证据**：isolated `M=1` reference 全部稳定，而 unrestricted batching 在 64/64 victims 上产生稳定 raw-BF16 mismatch。
- **最重要负结果**：4,237 个 calibration contract entries 全部 `all_repeats_exact=false`、`allowed=false`；SemanticFence 因此 0 coverage、0 个自然 `M>1`，并比 A 慢 0.2364%。
- **当前最大未知**：exactness 是否由当前 descriptor 未表示的 row composition / hidden-state 特征决定，还是本 stack 上 raw-BF16 exact rebatching 基本只存在于 `M=1`。

## 唯一实验执行账本

`SF-P0`：OLMoE / RTX 5090 calibration-to-fresh-eval expert rebatching Pilot。

| ID | 阶段 | 最终状态 | 权威证据 |
|---|---|---|---|
| `SF-P0-DATA` | fresh data | `DONE_FROZEN` | calibration 8 docs，SHA-256 `bfb89125...2630`；fresh 32 docs，SHA-256 `2608ef5d...695d7`；四源排除 overlap=0 |
| `SF-P0-UNIT` | implementation/preflight | `DONE` | 41/41 CPU tests 通过；`py_compile` 通过 |
| `SF-P0-ACCEPT` | real GPU acceptance | `DONE_REAL_GPU` | RTX 5090 UUID `GPU-64c06f...c4d7`；driver 595.71.05；PyTorch 2.8.0+cu128；CUDA 12.8；stack digest `7931d33b...09b2`；acceptance file SHA-256 `c842be78...c0e7` |
| `SF-P0-SEAL` | pre-science lock | `DONE` | lock file SHA-256 `2ea8d424...a56`，status `SEALED_BEFORE_SCIENCE` |
| `SF-P0-CAL` | calibration/contract | `DONE_ZERO_ADMISSION` | 4,237 entries；0 allowed；contract SHA-256 `801b4074...1c1` |
| `SF-P0-EVAL` | fresh A/B/C/D | `DONE` | A stable；B mismatch victims=64；D mismatch rows=0 但 coverage=0、自然 `M>1`=0、padding=0；D/A latency reduction=-0.002364；C 未支配 D |
| `SF-P0-DECIDE` | parent recompute | `WEAKEN` | 40 份 raw BF16、8,192 rows、trace 与 worker/parent closure 完整；`COMPLETE.json` status `SUCCESS_COMPLETE` |

## 本轮新增结果解释

### 实际观察

| 观察量 | run03 结果 |
|---|---:|
| `reference_all_stable` | `true` |
| `unrestricted_mismatch_victims` | `64` |
| `allowed_contract_entry_count` | `0 / 4,237` |
| `semanticfence_mismatch_rows` | `0` |
| `semanticfence_covered_victims` | `0` |
| `semanticfence_distinct_m_gt_1` | `0` |
| `semanticfence_padding_rows` | `0` |
| `semanticfence_latency_reduction_fraction` | `-0.0023640132` |
| `fixed_control_dominates` | `false` |
| trace / numeric / signature / worker-parent closure | 全部 `true`，errors=`[]` |
| wall time / budget | 791.900 s / 5,400 s，预算内 |

原始证据为 40 个 33,554,432-byte BF16 文件，总计 1,342,177,280 bytes；parent 从 raw BF16 独立重算得到相同 decision fields。正式 summary 的 `decision=WEAKEN`、`paper_result=false`、`authorized_next_step=null`。

### 支持的判断

1. **现象成立**：在固定 OLMoE、BF16、RTX 5090 stack 与相同 route rows 下，跨 row batching 确实能稳定改变逐 row raw-BF16 结果；B 覆盖 64/64 victims。
2. **reference 可用**：A 的 all-row isolated `M=1` 10/10 稳定，负结果不是由 canonical reference 漂移造成。
3. **主要死因不是 calibration support 不足**：4,237 entries 中 document support 以 8 docs 为主（2,496 entries），但所有 entry 都因跨 pack/repeat raw exactness 不成立而拒绝。

### 被削弱的判断

被削弱的是：仅用 `(layer, expert, M, kernel signature, stack)` 这一 coarse descriptor，能够从 calibration 得到可泛化且非退化的 raw-BF16 exact executor class。该 formulation 在本次冻结条件下退化为全 `M=1`，因此没有 latency headroom。

### 尚不能得出的结论

- 不能断言所有 SemanticFence / batch-invariance 机制无效。
- 不能断言所有模型、GPU、precision 或 backend 都不存在 exact rebatching operating region。
- 不能写成 full-layer、serving、request-level latency、EP/NCCL、租户隔离或形式化正确性结果。
- `semanticfence_mismatch_rows=0` 不是正向机制证据；它来自 0 admission 后的 `M=1` fail-closed fallback。

## 失败尝试与 authority 边界

- `run01` 在 evaluation trace 阶段检测到 foreign GPU PID 7704，写出 `failure.json`，未写 `COMPLETE.json`。
- `run02` 在 calibration 前检测到 foreign GPU PID 8168，写出 `failure.json`，未写 `COMPLETE.json`。
- 两次仅作为 operational forensics，不进入 scientific aggregation，也不覆盖 run03。
- GPU 连续空闲后启动的新目录 `run03` exit 0，且只有它拥有 completion-last authority。

## 更新后的因果链

batch composition / execution shape  
→ raw-BF16 输出变化 **[已支持：B 64/64 victims]**  
→ coarse executor descriptor 能识别安全的 `M>1` equivalence class **[被削弱：0/4,237 allowed]**  
→ SemanticFence 放行非退化的 exact rebatching **[被削弱：coverage=0，M>1=0]**  
→ expert-stage latency 下降 **[被削弱：-0.2364%]**  
→ request-level latency / throughput 改善 **[尚未验证]**

## 假设更新

- **保留的主假设**：存在输入条件化、版本绑定、未知即回退的安全 rebatching 机制，可以在保持所选语义边界时利用部分跨请求 batching 空间。
- **被支持的子假设**：`M=1` canonical 稳定；unrestricted `M>1` 会产生稳定 composition-sensitive mismatch。
- **被削弱的子假设**：只按 layer/expert/M/kernel regime 聚合即可得到可泛化的 exact allowlist。
- **当前最主要替代解释**：raw exactness 取决于 batch 内 hidden rows 的具体数值组成，而不是仅由执行形状与 kernel signature 决定。
- **新出现的控制变量线索**：calibration 的 16,117 个 `M=2` packs 中有 121 个 pack 在 10 repeats 上 all-row exact，分布于 16 layers、52 experts、108 个 layer-expert cells；这说明 `M>1` exact pack 并非绝对不存在，但当前 descriptor 无法区分。该统计是对已封存 calibration artifact 的只读派生，不是 fresh 正向结果。
- **方向判断**：**保留方向，修改信号**；先测试 composition-aware、input-only descriptor，不改变 raw-BF16 exact 标准，不修改本轮冻结结果。

## 下一轮唯一核心问题

> 在保持同一 raw-BF16 exact semantics、模型、stack 与 fresh split 不变时，calibration-only 的 input-composition descriptor 能否在 fresh `M=2` packs 上做到零错误放行，并恢复非零、跨 layer/expert 的 exact admission？

## 下一轮最小实验

- **核心假设**：少量 all-exact `M=2` packs 由 pre-call 可观测的 hidden-row composition 决定，coarse class 聚合掩盖了该 operating region。
- **自变量**：coarse descriptor 与 frozen input-composition descriptor。
- **因变量**：fresh false-admission count、admitted exact pack/row coverage。
- **baseline**：本轮 `(layer, expert, M, signature)` contract（fresh coverage=0）。
- **必要对照**：在 layer/expert 内置换 exact 标签后使用同一选择流程，排除稀有正例与选择偏差造成的伪信号。
- **固定变量**：同一模型权重、RTX 5090 stack、route/row ledger、BF16、row order、fresh-32、`M=2`、10 repeats；禁止读取 fresh 输出后调 descriptor 或阈值。
- **主指标**：fresh admitted mismatch 必须为 0；其次报告 admitted packs、covered rows 与 layer-expert cells。
- **可复用资产**：run03 的 calibration/evaluation captures、row context、A raw reference、trace parser 与 parent recompute。
- **最小新增实现**：一个只读特征提取与冻结 descriptor 脚本；一个只执行 fresh `M=2` replay 的小 runner。
- **资源**：先离线分析，再单 RTX 5090 局部 replay；预计 GPU 不超过 45 分钟。该时间是计划值，未观测。

## 预定义结果解释

- **支持修改后的信号**：fresh admitted mismatch=0，且至少放行 8 个 packs、覆盖至少 4 个 layer-expert cells；只授权把 composition descriptor 接入 planner 并重新测局部 latency。
- **削弱修改后的信号**：calibration holdout 上无法冻结可用 rule，或 fresh 出现任一错误放行，或 fresh admission 仍为 0；届时 raw-exact rebatching 在本 stack 上被进一步削弱，优先重新审视 semantic/action boundary，而不是调数据或阈值。
- **无法判断**：descriptor 使用了 post-call/output 信息、发生 fresh label leakage、GPU/reference 不稳定或 trace/row identity 不闭合。
- **本轮不能外推**：即使正向，也不能证明完整 SemanticFence、serving 收益或跨 stack 泛化。

## 验证与审计

```bash
PYTHONPYCACHEPREFIX=/tmp/semanticfence-pycache python3 -m unittest discover \
  -s docs/ideas/semanticfence/experiments -p 'test_*.py'
```

结果：`Ran 41 tests ... OK`。

```bash
PYTHONPYCACHEPREFIX=/tmp/semanticfence-pycache python3 -m py_compile \
  docs/ideas/semanticfence/experiments/*.py
```

结果：exit 0。远端与本地 parent recompute 均为 PASS；限界 fresh-agent 结果完整性复核为 `PASS`，P0=0、P1=0，独立确认 run03 authority、summary/parent 一致性、0/4,237 admission、trace closure、失败尝试排除与单卡证据边界。

## 当前非阻塞问题

- P1：只有单一 OLMoE / RTX 5090 / BF16 stack，适用边界尚未扩展。
- P1：本轮是 decode-style expert-stage proxy，没有验证收益传播到 full layer 或 request latency。
- P2：正式论文需要跨 stack、模型和更完整 baseline，但不阻止下一次 composition-signal Pilot。

> 当前实验已经足以推进下一轮探索，停止继续扩展审计项。

## 下一步

冻结并执行唯一的 `M=2` composition-signal Pilot；不重跑或调参抢救 SF-P0。
