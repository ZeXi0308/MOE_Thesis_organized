# RouteGuard-KV：严格评审补充与收紧后判死协议

> 日期：2026-07-26
>
> 当前状态：`R0A_GPU_SMOKE_INTEGRITY_PASS / CALIBRATION_NOT_RUN / KILL_PROBE_ONLY / NOT_CURRENT_MAINLINE`
>
> 证据等级：冻结设计与数据、实现、25项 CPU/真实 tiny OLMoE 集成测试，以及单张 RTX 5090 engineering smoke 完整性 PASS；**尚无 calibration/formal R0-A 科学裁决、尚无系统机制证据、尚无多卡证据**。
>
> 执行权威：[当前研究状态与唯一执行线](../../current/README.md)。若本文与新的 machine-readable sealed/formal decision 或当前状态冲突，以后者为准。

## 0. 直接裁决

RouteGuard-KV 中真正值得验证的科学问题是：

> **在既有稀疏 MoE 的连续解码中，KV-cache 量化是否通过 router-input 扰动，引发 top-k expert 集合或 gate weight 漂移，并形成相对普通数值误差可分离、相对既有 KV 敏感度信号有增量价值的质量损失？**

当前裁决分为三层：

- `[Observed]` KV-cache 精度是长上下文、高并发解码的真实容量/带宽变量；逐层 K/V 混合精度和离线配置本身已有强 prior art。
- `[Observed, engineering smoke only]` 冻结的2文档 smoke 在 RTX 5090 上完成50/50条 trajectory，完整性、identity、patched-BF16、route-lock 和 quantizer-ledger 门全部 PASS；该样本不进入 formal 统计。
- `[Inferred]` RouteGuard-KV 可能剩余的新意，不是“把 KVTuner 用到 MoE”，而是 **KV 扰动 → 路由变化 → 额外质量损失** 的 post-training 因果链，以及该信号对精度分配的增量价值。
- `[Hypothesis]` 该链可能被残差流、归一化和路由 margin 稀释；即使存在，也可能被 attention-error signal 或“保护少数敏感层”简单策略几乎完全捕获。

因此只授权廉价、fail-closed 的存在性探针。它不是当前主线，不改变 BCRD/DEPA 共同 Gate 0/1 的执行顺序，也不授权先实现完整分配器或系统。

## 1. 对原执行顺序的纠正

原 2026-07-26 候选长稿仍把 U01 RankLane 作为执行顺位 1，这已经被更新证据取代：

- fixed RankLane 在 `p_return≤20%`、所有 codec/launch/queue/metadata 税为零的冻结域内，最乐观 E2E 收益只有 `4.1667%`，正式裁决为 `NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20`；
- 真实 optimized return-path existence 仍需 8×A100，但该硬件 Gate 不占用当前单卡优先级；
- 当前唯一正式动作仍是先关闭共同 Gate 0，再运行共同 Gate 1。

RouteGuard-KV 的位置是：**可登记、可预注册、可在不干扰唯一主线时运行最小 R0-A；未过门不得升格。**

## 2. Prior-art 边界与剩余新颖性

| 最近工作 | 已覆盖内容 | 对 RouteGuard-KV 的约束 |
|---|---|---|
| [KVTuner](https://arxiv.org/abs/2502.04420) | 离线逐层敏感度、K/V 精度对、硬件友好混合精度搜索 | “离线逐层 K/V 位宽表”不能作为新意 |
| [MoE-nD](https://arxiv.org/abs/2604.17695) | 逐层 `(eviction ratio, K bits, V bits)` 联合配置 | “多轴、逐层、K/V 分离配置”不能作为独立贡献 |
| [TriRoute](https://arxiv.org/abs/2607.06601) | attention、expert selection 与 KV bit-width 的联合学习式路由 | “MoE 与 KV 决策耦合”已出现；必须突出既有稀疏 MoE 的 post-training route-drift 诊断与约束 |
| [vLLM quantized KV cache](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/) | 全局 KV dtype，并可跳过指定层的量化 | “保护敏感层”已经是可部署强简单基线 |
| [LMDeploy KV quantization](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html) | 全局 INT4/INT8/TurboQuant 等真实路径 | 可验证统一档快区，但不能替代 RouteGuard 逐层表的系统验证 |

全文碰撞审计前，允许主张的最窄空白仅为：

1. 在固定 pretrained sparse-MoE 上，测量 KV 量化诱发的 route-set/gate-weight 漂移；
2. 用明确定义的 router-lock counterfactual 分离其质量贡献；
3. 检验 route signal 在控制 attention-error signal 后，是否仍能改善 matched-byte 的逐层保护决策。

若 route signal 与 attention-error signal 高度共线，或简单 skip-sensitive-layers 捕获 `≥80%` Oracle headroom，停止 RouteGuard 分配机制主张。

## 3. 因果对象：不用 MoE/dense 比值作主门

不同 MoE 与 dense 模型之间同时存在权重、训练语料、优化过程、结构和归一化差异。因此：

> `ΔKL_MoE / ΔKL_dense` 只能作为辅助描述，不能作为“路由放大”的主因果证据。

主估计量必须来自**同一 MoE、同一文档、同一 token 流、同一量化配置**下的三个 counterfactual 臂。设 BF16-KV 参考轨迹在层 `l`、位置 `t` 的 expert 集合和 gate weights 为 `(S*_{l,t}, g*_{l,t})`：

1. `free(b)`：量化 KV，router 正常产生集合和权重；
2. `set_locked(b)`：量化 KV，但强制使用 `S*`；在量化态完整 router softmax 上仅对 `S*` gather，并严格遵循模型原生 `norm_topk_prob` 语义；OLMoE 的冻结配置为 `false`，因此不额外归一化；
3. `fully_locked(b)`：量化 KV，同时强制使用 `(S*, g*)`。

对相对 BF16 参考输出的文档级损失 `L(·)`，报告：

\[
\Delta_{set}=L(free)-L(set\_locked),
\]

\[
\Delta_{weight}=L(set\_locked)-L(fully\_locked),
\]

\[
\Delta_{numeric}=L(fully\_locked).
\]

三者是预注册 counterfactual contrasts，不预设严格可加或恒为正。若存在显著负交互、符号不稳或 `L(free)` 过小，必须原样报告，不能裁剪负值后计算“归因份额”。可同时报告总体 router-mediated contrast：

\[
\Delta_{router}=L(free)-L(fully\_locked).
\]

teacher forcing 只用于对齐输出 token；三个臂必须维护**彼此独立**的 KV、position、router 和 decode state。不得共享量化 cache，也不得用离线 shadow mask 代替策略特异的逐步状态。

## 4. R0-A：最小存在性判死实验

### 4.1 目标

只回答两个问题：

1. aggressive KV 量化是否在自然 decode state 中稳定改变 MoE 路由？
2. 该变化是否形成非平凡、可复现的 router-mediated 质量损失？

不在 R0-A 中证明 MoE 普遍比 dense 更脆弱，不搜索完整逐层表，不报告吞吐或并发收益。

### 4.2 冻结最小矩阵

- **模型：** `allenai/OLMoE-1B-7B-0924` 一个模型；通过后才加第二个开放 MoE。
- **数据：** 同32篇 sealed 自然文档构造 prompt 512/2048 两个 paired condition；用归一化文本 SHA-256 与历史 calibration/sealed manifest 做排除。仅使用新的 offset 不能证明数据独立。
- **decode：** 每文档 32 个 teacher-forced decode step；逐步写入和读取真实 KV state。
- **配置：** 唯一 primary 为 `INT4-K-only @ prompt 2048`；512、V-only 与 KV 均为不能 rescue 的 secondary；另加 identity/no-op quantizer 负控。位宽格式、group size、scale 粒度、RoPE 前后顺序必须与目标后端谱系一致。
- **counterfactual：** `free`、`set_locked`、`fully_locked`。
- **主指标：** 文档级 KL、route-set Jaccard/flip rate、gate-weight drift、`Δset`、`Δweight`、`Δnumeric`、`Δrouter`，以及逐层/逐位置分布。
- **统计：** 文档级 paired bootstrap 5,000 次；同时公开 effect size、95% CI、正负交互和每个长度桶结果。

### 4.3 上 GPU 前必须通过的不变量

1. BF16 下三个臂逐 token、逐层 output 与 route identity 一致；
2. identity/no-op quantizer 的 route flip 为零，输出在冻结容差内相等；
3. 每臂 cache object、storage pointer/fingerprint 和 decode journal 独立；
4. K 在 RoPE 后量化；若目标后端不同，必须记录差异并禁止等价外推；
5. top-k tie-break、gate weight 重归一化和 position 对齐确定且有单测；
6. 任一 dtype 不支持、发生 silent cast/fallback、cache 跨臂污染或 route 注入失败时立即 `INVALID`。

### 4.4 R0-A 裁决

进入 R0-B 的必要条件：

- `INT4-K-only @ prompt 2048` 的文档级 `L_free ≥1e-4`，且 non-tie route-set flip rate `≥1%`；
- 该唯一 primary cell 的 paired-bootstrap `Δrouter` 95% CI 下界 `>0`；
- `Δrouter / L(free)` point estimate `≥40%`，且 95% CI 下界 `>25%`；
- leave-one-document-out 的 `Δrouter` point estimate 全正，且至少90%的 flip 不是 top-k boundary tie；
- 512、V-only 与 KV cell 原样报告，但不得救回或推翻 primary。

以下任一成立即停止机制线：

- primary 总 KL `<1e-4`、non-tie set flip `<1%` 或 route-mediated share `<25%`；
- 份额符号不稳，或只在 INT2/异常格式出现；
- route flip 存在但 `Δrouter` 接近零；
- 普通数值残余几乎解释全部损失；
- 只有 synthetic prompt、单一离群文档或实现不对应真实后端时成立。

R0-A 失败只允许留下“该模型/格式/长度域内未观察到可操作的 route-mediated KV 量化损失”这一窄结论。

## 5. R0-B：跨模型与辅助 dense 对照

R0-A 通过后才执行：

- 加入第二个 routing regime 不同的开放 MoE；
- calibration 与 sealed evaluation 按文档 hash 冻结分离；
- dense 对照尽量选同数据谱系、相近规模和归一化结构的模型，但只作辅助描述；
- 主门仍是两个 MoE 中 router-mediated contrast 同方向且过门，不使用跨架构 `MoE/dense ≥1.5×` 比值授权机制；
- 报告 route margin、attention-output error 与 route flip 的相关性，检查 RouteGuard signal 是否只是已有敏感度信号的别名。

## 6. R1：只有增量信号成立才叫 RouteGuard

在相同 calibration/sealed split、相同平均 KV bytes 和相同可执行位宽集合下比较：

1. uniform precision；
2. KVTuner-style attention-error signal；
3. vLLM-style skip-sensitive-layers；
4. 最佳单阈值或保护 top-N 层；
5. RouteGuard route-drift signal；
6. future-known per-layer Oracle，只提供上界。

必须同时满足：

- RouteGuard 相对 best simple baseline 的 matched-byte 质量改善 paired CI 下界 `>0`；
- RouteGuard 额外捕获 `≥20%` simple-to-Oracle residual；
- best simple baseline 捕获 `<80%` Oracle headroom；
- 两个 MoE 同方向，不能依赖逐模型临时改阈值；
- 最优表不是“只保护已知某一层”或等价于 vLLM skip-layers 的二值规则。

若失败，允许的结论是“route drift 可测，但不能提供独立分配价值”；停止新 allocator/controller，只保留 measurement 与简单配置建议。

## 7. R2：系统闭环必须执行同一张表

### 7.1 实现资格门

进入 RouteGuard 系统结论前，真实 engine 必须：

- 执行 sealed R1 产生的逐层配置，而不是全局统一档；
- 支持协议所声称的 K/V 分离和位宽集合；
- 逐层记录 effective dtype、scale/granularity、page/block layout 和 kernel path；
- 没有 silent BF16 fallback、隐式 dtype promotion 或只在 Python fake-quant 生效；
- baseline 与 RouteGuard 使用同一 optimized attention/backend、同一 admission/HBM budget 和同一 workload。

如果只能运行 LMDeploy/vLLM 的全局统一档，则该实验只能证明**通用 KV 量化快区**，不能通过 RouteGuard H4。vLLM 的 skip-layers 可以作为二值可部署候选；若它等价捕获 RouteGuard 表，则复杂机制停止。

### 7.2 系统主比较

RouteGuard 的系统收益必须相对“质量门内的 best deployable uniform/skip-layer baseline”计算，而不是只相对 BF16：

- matched task-quality 与上下文长度；
- 同一 HBM 上限和 continuous-arrival trace；
- 报告 capacity、SLO-goodput、TTFT、TPOT、P99、HBM 分解和 kernel/quantization tax；
- 进入完整系统论文叙事至少要求相对 best simple baseline 的 SLO-goodput 净收益 `≥10%`、95% CI 下界 `>0`，且 P99 不恶化超过 `2%`；
- `B_max≥2×` 或相对 BF16 吞吐 `≥1.4×` 只能作为通用 KV 压缩结果，不能单独归因给 RouteGuard。

## 8. 论文与表述边界

| 通过范围 | 允许结论 | 不允许结论 |
|---|---|---|
| 只过 R0-A/R0-B | MoE KV 量化存在可分离的 route-mediated quality effect | RouteGuard 系统成立、CCF-B 机制成立 |
| 再过 R1 | route signal 对 matched-byte 精度分配有跨模型增量价值 | 已有真实吞吐、并发或尾延迟收益 |
| 再过真实实现资格门和 R2 | 同一张 RouteGuard 表在 optimized engine 上产生系统净收益 | 外推到未测模型、MLA、GPU、拓扑或多卡 EP |
| R0 失败 | 冻结域内的窄负结果与配置边界 | “MoE 与 dense 普遍无差异” |

`H1+H2` 只足以形成 characterization；只有跨模型 R1、真实逐层 fast path 和公平 R2 闭合后，才有完整系统论文故事。投稿级别不在实验前承诺。

## 9. 当前最短动作

1. 不修改[当前唯一执行线](../../current/README.md)；
2. R0-A 实现、数据冻结、25项 CPU/真实 tiny OLMoE 集成测试和 RTX 5090 smoke v2 已完成，审计见[Code Review 与执行手册](experiments/R0A_5090_CodeReview与执行手册_2026-07-27.md)；
3. smoke v2 只授权下一步运行8文档 calibration；当前 calibration 未运行，formal 仍为 `NOT_APPROVED`；
4. smoke 的方向性数值不能宣布 R0-A PASS；calibration 不得调阈值、位宽、模型、prompt、steps 或样本数；
5. R0-A 未过门立即停止，不扩模型、不做 R1/R2；
6. 任何新 sealed result 都必须写 machine-readable decision，并同步更新本页与当前状态。
