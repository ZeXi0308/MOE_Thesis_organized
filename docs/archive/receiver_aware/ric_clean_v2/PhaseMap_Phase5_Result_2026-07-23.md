# PhaseMap-MILP Phase 5 结果（2026-07-23）

状态：**FORMAL STOP / HOLDOUT NOT OPENED**  
冻结协议判定：**`BLOCKED_UNINFORMATIVE_DEADLINE_GRID`**  
机制诊断判定：**当前 closed-pair PhaseMap 形式无可测 headroom，不再继续调参救活。**

## 1. 证据边界

- [Observed] RTX 5090 上真实测量 row-1 BF16 `pack / unpack / combine`；20 warmup +
  100 measured trials，数值 correctness 全通过。
- [Observed] 两模型 clean-v2 calibration native-route artifacts 构造 16 selection pairs/model。
- [Observed] B0/Q/J/R 为 L2 queue replay + exact policy enumeration；不是 RDMA、NCCL、
  多节点或 serving TPOT/P99。
- [Observed] selection gate 失败后没有生成 selection manifest，也没有读取 holdout split。

正式 GPU LUT 内记录的 artifact digest：
`e7277f3d32e64c087eabd3c245d0915f0ea26a7d15908ee70e79293003f9f11e`。

| Model | pack median CUDA us | unpack median CUDA us | combine median CUDA us |
| --- | ---: | ---: | ---: |
| OLMoE, hidden 2048, top-8 | 11.6320 | 9.7280 | 75.8880 |
| LLM-jp, hidden 512, top-16 | 11.6480 | 9.6480 | 144.7360 |

## 2. Formal selection stop

冻结规则要求：先取 pooled B0 miss 最接近 50% 的共享 `kappa`，且两个模型各自都必须落在
`[20%,80%]`。

| kappa | OLMoE B0 miss | LLM-jp B0 miss | pooled |
| ---: | ---: | ---: | ---: |
| 1.25 | 100% | 100% | 100% |
| 1.50 | 100% | 100% | 100% |
| 2.00 | 100% | 50% | 75% |
| 3.00 | 0% | 0% | 0% |

按冻结规则选中 `kappa=2.00`，但 OLMoE 为 100%，违反 informativeness gate，故正式运行
在 selection 阶段停止。不得用 per-model kappa 或事后扩大 grid 改写本次判定。

## 3. Selection-only 机制诊断

以下只使用已经打开的 selection split，不是正式 holdout 结果。

- [Observed] 细扫显示 `kappa=2.00` 时 `(OLMoE, LLM-jp)=(100%,50%)`；到
  `kappa=2.01` 已变为 `(50%,0%)`。`2.01..2.40` 均为 `(50%,0%)`，从 `2.50`
  起均为 `(0%,0%)`。没有两模型共同的可辨识 slack 区间。
- [Observed] 分别在各模型自己的 50% B0 点检查信息价值：

| Model / diagnostic kappa | B0 miss | Q miss | J miss | R miss | B0/Q/J/R CVaR90 | actionable | strict flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMoE / 2.01 | 50% | 50% | 50% | 50% | 0.1154569 | 0% | 0% |
| LLM-jp / 2.00 | 50% | 50% | 50% | 50% | 0.0005262 | 0% | 0% |

- [Observed] `R` 相对 exact best of Q/J 的 absolute/relative miss reduction 与 CVaR reduction
  全为 0。
- [Observed] 代表 pair 的四个 joint first-actions 全部属于最优集合；增加 observation
  partition 没有改变可实现结果。
- [Inferred] 当前构造只有 receiver 信息差，没有 receiver 决策价值。closed two-request、
  work-conserving reorder 不改变关键 join/deadline 结果，MILP/controller 都没有可利用的动作空间。

因此科学结论不是“Receiver-awareness 普遍无效”，而是：

> **在当前 closed-pair、只允许 sender first-order reorder 的 PhaseMap 问题定义中，queue +
> join phase 信息没有增量价值。**

## 4. 不要再怎么救

- 不要扩大 kappa grid、改 per-model kappa 或放宽 `[20%,80%]`，因为在各自 50% 点仍是
  `B0=Q=J=R`。
- 不要在这个 closed-pair action space 上继续加 predictor、bandit、MILP 目标或启发式 controller。
- 不要直接升级 RDMA/多节点；单卡校准 replay 已显示 actionability 为 0。
- 不要读取 holdout 为当前对象寻找正例；本轮 holdout 保持未打开。

## 5. 下一候选：Receiver-Stamped Slack Credits

建议保留 receiver-congestion 主题，但改变问题定义而不是救 PhaseMap：

> receiver 根据已承诺工作与 request deadline 发布可消费的 slack credits；多 sender 连续
> incast 时，admit/defer 动作改变进入 receiver 的工作集合，从而可能改变 SLO miss，而不只是
> 重排守恒的 closed makespan。

本质增量：引入连续到达、有限 admission、request deadline 和 receiver committed-work state。
建模可用 online admission / marginal lateness shadow price；系统动作是 admit/defer，不改模型
router 语义。第一步只做 Phase 1 novelty 与 oracle necessity 审计：若 receiver-aware oracle 相对
最强 sender-only EDF/least-laxity 仍无 gap，立即判死，不写 controller。

## 6. 本地证据文件

- `outputs/phasemap_formal_20260723_run1/lut.json`
- `outputs/phasemap_formal_20260723_run1/source_manifest.json`
- `outputs/phasemap_formal_20260723_run1/selection_stop_diagnostics.json`
- `outputs/phasemap_formal_20260723_run1/phasemap_runner_tests_postpatch_20260723.log`：远端
  RTX 5090 环境最终 runner tests，32/32 PASS。
- `outputs/phasemap_formal_20260723_run1/remote_prepatch_tests_88_of_89.log`：唯一失败为远端
  非 Git checkout 的环境元数据阻断；移除该非科学硬门后，远端 runner tests 32/32 PASS。
