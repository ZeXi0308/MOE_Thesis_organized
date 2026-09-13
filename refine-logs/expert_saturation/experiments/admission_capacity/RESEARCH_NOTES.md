# 准入／容量研究简记

**当前执行状态：** 目标恢复后连续三个回合实时确认同一Qwen3 PID13275占用GPU，再次标记BLOCKED_RESOURCE；最新已处理4/16分片，第5片3.10GB。新32文档四臂八项保持STAGED、GPU 0/8，本地及远端无已启动结果；准备与检查已完成，无后台等待或执行器。主问题OPEN，资源释放后接续同一包。详见[状态与接续入口](../../outputs/admission_capacity/20260913_rotation_strong_baseline_r01/STATUS.json)。

探索记录，不替代 sealed verdict 或 `docs/current/README.md`。每轮一条：假说 → 命令 → 结果路径 → 解释 → 下一动作。

**新轮先查[共享结论台账](RESULT_LEDGER.md)。** 最新[首次交换六项实测](../../outputs/admission_capacity/20260913_rotation_first_swap_r01/RESULTS_WESTE_ADDENDUM.md)：C相对A吞吐+0.166%/−0.425%，平均完成−0.032%/+0.644%；C相对持续B吞吐−1.332%/−1.513%，平均完成−1.065%/−0.850%。首次交换减轻了B对早完成请求的拖延，但没有同时保住其吞吐结果。B相对A仍是吞吐+1.518%/+1.105%、平均完成+1.044%/+1.507%的权衡；不作方法GO。原八项与20项结果及各自审计状态保持独立，下文历史“下一动作”只描述当时状态。

**最新成本模型修正：** A/C进入同一末两请求阶段时剩余输出87/99→58/103，双请求调用少29、单请求调用多33，抵消约90–91%的双请求桶时间减少，总引擎调用1257→1261。只看双请求阶段长度或剩余token总和会漏掉落后请求的单独收尾。观察阶段满足T_tail≈r·c2+(s−r)·c1（r≤s，期间无新到达/抢占）；它不是使用事后尾请求身份的在线预测或跨策略Oracle。

**唯一下一项：** 新文档cohort，同预算native/native A/A/completion_headroom/持续most_output四臂×两反向block已封存并STAGED，等待GPU空闲后按[接续命令](../../outputs/admission_capacity/20260913_rotation_strong_baseline_r01/RESUME_COMMAND.sh)执行。新32与旧96的文档/输入hash及源区间互斥，仍为同语料同运行域；主native预先指定，A/A只记录漂移。当前无八项GPU结果，不继续扫描首次交换次数。

最新模型修正见 [KV 诊断 addendum](../../outputs/admission_capacity/20260912_kv_deficit_law_r01/ADDENDUM.md)：早期条件耗尽预测仍成立，但两个 CPU 反例推翻了原诊断的跨策略下界表述。代码已修正，原始报告保留。

---

## 主问题（2026-09-12 收窄后）

> 固定 KV 预算下，怎样减少已有请求的长暂停，并保住有效服务量？

运行域：单卡 RTX 5090、OLMoE-1B-7B BF16、vLLM 0.26.0、3072 prompt / 1024 output、
32 请求闭合 cohort、50 ms 到达间隔、`max_num_batched_tokens=1024`。
主评价：同任务同实际资源下的完整请求吞吐与每请求最大 ITL；同时报告 TTFT、平均完成时间、墙钟和全部完成/失败数。
当前没有统一业务 SLO，先展示权衡，再用独立校准确定确认阶段约束。亚请求 KV 回滚是未实现的机制假说；
原生恢复准入、保KV完成保障与轮转均已各自实测；具体适用边界见上方最新结果，不以单个机制代价判死问题。

## 当前动作与已有工作的重合（2026-09-13定向核查）

**轮转、aging、恢复优先、KV预留及计入重算成本本身均不足以构成新颖性。** 本轮查新只追以下四项直接相邻工作及调度路径，未实现新的Controller，也未修改已准备的20项包。

| 工作与版本 | Signal/state → action | Objective / regime / guarantee | 公开实现与本轮含义 |
|---|---|---|---|
| Liu等，Andes，arXiv2404.16283v2（2024-12-13；未核实正式会议） | token时间线、消费曲线、context及批宽 → 按预期QoE增益/内存选择运行集合，计入抢占/恢复对全体请求的成本 | 用户消费QoE；vLLM、4/8×A100 TP，含Phi-3.5-MoE；未给server max-ITL硬界 | [论文§4](https://arxiv.org/html/2404.16283v2#S4)同时讨论swap/recompute；[作者公开实现](https://github.com/AmberLJC/vllm-0.6.1/tree/3cc5c8458bd2a2744b8392ab6aa311c645105655)默认真实RECOMPUTE，发布代码与论文refiner有差异，见下文 |
| Wu等，FastServe，NSDI2026正式版 | 输入长度、已获服务、距上次执行及KV状态 → skip-join MLFQ、aging提权、token边界抢占及预先KV换入换出 | 完成延迟与延迟约束吞吐；OPT/Llama、A100单/多卡。经验防饥饿，不是max-ITL硬界 | [正式论文§4.1–4.2](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf#page=5)、[官方scheduler](https://github.com/MachineLearningSystem/NSDI26-FastServe/blob/main/fastserve/scheduler.py)已有prevent_starvation和reserve_free_blocks；SwiftTransformer+CPU/GPU swap不直接兼容本轮原生重算路径 |
| Shen等，FastSwitch，arXiv2411.18424v1（2024） | 优先级、KV连续块组、近期交换 → 动态块组、同步/异步swap及CPU副本复用 | TTFT/TBT尾部、吞吐；vLLM0.3.3、A10/A100、每GPU60GB CPU swap；优先级由离线Random/Markov过程产生，不是本轮缺席驱动控制 | [论文§3–4](https://arxiv.org/html/2411.18424v1#S3)。本次在论文/作者页未定位官方可执行代码，不能称其未实现。作者页仅记[投ATC2025](https://ziolee.xyz/)，未核实正式接收 |
| Sheng等，VTC，OSDI2024 | 客户端加权输入/输出服务counter → 低counter客户端准入；附录另提抢占式扩展 | 客户端服务差公平，非单请求max-ITL；主体算法无抢占，服务差界需双方持续backlogged等假设 | [正文与保证](https://www.usenix.org/system/files/osdi24-sheng.pdf#page=8)、[C.3](https://www.usenix.org/system/files/osdi24-sheng.pdf#page=25)已提出抢占运行请求换入低counter客户。代码[vtc_req_queue.py](https://github.com/Ying1123/VTC-artifact/blob/192c2e2014c69c8c6c699d7113c3822e4db632e6/slora/server/router/vtc_req_queue.py#L92-L152)为counter准入；所核查路径无resident轮转 |

FastServe以[2026正式版](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)为本次权威，不沿用2023预印本的题名和旧性能数字。论文讨论kill/recompute的代价与livelock风险，不能解释为所有重算轮转必亏的定理；其§6.1 per-token latency为请求完成时延除以输出长度，不等于最长相邻token间隔。公开scheduler的换入顺序按队列优先级，本次未在该路径确认论文ENST预测公式，故不把论文设计和已读代码完全等同。

FastSwitch真正处理的是KV交换成本；其异步策略也指出少量短请求时等待换入并扩大batch可能更好。它支持“恢复成本与batch效率要一起考虑”的邻近认识；我们的重算/收尾抵销仍需独立数据，不能把这句宽泛原则作为首次发现。论文未提供本轮c20策略或秒级硬保证；代码可得性仍未验证。

VTC定理4.4在无抢占Algorithm2、双方全区间队列持续有等待请求时，约束服务差 `2·max(w_p·L_input,w_q·M)`；定理4.11的dispatch等待界另需服务容量正下界等条件。单闭合cohort中“每请求当客户端”并不自动满足条件或给max-ITL保证。C.3是明确提出但未验证的抢占扩展；据此既不能说VTC已给出我们的等价实测解，也不能说前人未考虑过该动作。

Andes v2 §4.3已按context长度离线测量swap/recompute开销，并比较动作QoE收益与全体在途请求的损失；因此“无CPU swap、重算计费”不能作为本轮独立差异。当前server输出max-ITL也不等于客户端消费停顿：缓冲可能吸收输出间隙。没有消费需求与客户端测量，不能把本轮停顿改善升级为QoE收益。

作者仓库固定commit `3cc5c8458bd2a2744b8392ab6aa311c645105655`：入口[llm_engine.py:418](https://github.com/AmberLJC/vllm-0.6.1/blob/3cc5c8458bd2a2744b8392ab6aa311c645105655/vllm/engine/llm_engine.py#L418)选择AndesScheduler；[andes_scheduler.py:33](https://github.com/AmberLJC/vllm-0.6.1/blob/3cc5c8458bd2a2744b8392ab6aa311c645105655/vllm/core/andes_scheduler.py#L33)默认RECOMPUTE，605–623行执行真实重算抢占及swap失败回退。[knapsack_solver.py:24](https://github.com/AmberLJC/vllm-0.6.1/blob/3cc5c8458bd2a2744b8392ab6aa311c645105655/vllm/core/andes_utils/knapsack_solver.py#L24)使用固定0.02s token latency和单位开销系数，按第10百分位slack限制抢占数，`latency_function`在所读调度函数中未使用；不能把该路径等同于论文逐动作累计QoE损失的refiner。官网Coming soon不能再作为代码未公开的证据。这里只核查源码，未安装、移植或复现实验。

**尚待验证的本轮问题：** 固定实际KV下，控制恢复次序和驻留时间能否稳定改善server max-ITL—完整服务量权衡；模型必须同时解释额外重算与batch收尾。该问题与已有动作重合，具体的新贡献尚未定位。旧八项及两组新文本的20项均已测量；四个新轮转格held=0、对native吞吐差仍变号，未证明保护贡献或独立方法新颖性，也不足以判死整个问题。

**对后续比较的约束：** 已完成的[20项独立文本+A/A](../../outputs/admission_capacity/20260913_rotation_holdout_r01/REPORT.md)确认固定参数在这两组文本的权衡和运行漂移，不因此升级论文级方法验证。当前rotate已包含最长缺席恢复和aging；另加同义aging标签可能只是重复。唯一下一步是驱逐服务量排序消融：同一合格集合内，现有最少进度与已交付输出最多者对照，保持其它触发/恢复/保护条件；它不是完整公平基线或FastServe/VTC复现。正式公平策略仍需定义客户端映射、输入/输出计费、backlog及抢占触发，不能事后按结果选择。Andes迁移还须区分发布实现和论文策略，独立校准批宽/恢复成本与消费需求；不能选择消费率制造无动作弱基线。CPU swap系统另计主机空间与传输资源。

---

## 稳定事实

- 短请求静态 cap8 优于 cap6（同引擎，已独立复测）；普通 ITL 反馈与单次非抢占降档未稳定超过强静态点。
  数据：`outputs/admission_capacity/20260906_native_{fixed_engine,fresh_cohort,knee,single_action}_r01/`。
- 短文本 steady 域已有可行包络诊断；它使用了实测执行构成，不能作为所有调度动作的上界。
  四个旧机制的测量结论保留其原 formulation 范围。
- 长上下文 KV 压力域**可判别**（cap 扫描改变联合达标数），是当前主线运行域。
- KV 预算 0.90→0.95 消除秒级暂停，吞吐 +3.99%/+4.70%，但平均完成延迟 +1.02%/+0.15%。
  这是配置测量，不是同预算策略收益。见 `20260912_kv_budget_r01/`。

---

## 轮次记录

### 2026-09-12 · KV 赤字定律（本轮，零 GPU）

**假说.** 同预算收益缺口有三个互斥候选原因：信息不够 / 动作窗口关闭 / 动作粒度不匹配。
三者导向完全不同的下一步，必须先分开。

**命令.**

```bash
(cd refine-logs/expert_saturation/experiments/admission_capacity && \
  python3 -m unittest test_kv_deficit_model -v)          # 36/36

R=refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01
O=refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_deficit_law_r01
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_kv_deficit.py \
  --cell $R/gpu_results/repeat0-budget90 --pause-ledger $R/cpu_analysis/current-repeat0-budget90.json \
  --cell $R/gpu_results/repeat1-budget90 --pause-ledger $R/cpu_analysis/current-repeat1-budget90.json \
  --cell $R/gpu_results/repeat0-budget95 --pause-ledger $R/cpu_analysis/current-repeat0-budget95.json \
  --cell $R/gpu_results/repeat1-budget95 --pause-ledger $R/cpu_analysis/current-repeat1-budget95.json \
  --output $O/analysis/kv_deficit.json
```

**结果路径.** [`20260912_kv_deficit_law_r01/REPORT.md`](../../outputs/admission_capacity/20260912_kv_deficit_law_r01/REPORT.md)、
`analysis/kv_deficit.json`。代码 [`kv_deficit_model.py`](kv_deficit_model.py) + [`analyze_kv_deficit.py`](analyze_kv_deficit.py)。

**结果.**

| 量 | 值 |
|---|---|
| 准入时刻静态可行性判据 | 抢占有无 **4/4** 一致 |
| 空闲块耗尽轨迹（因果窗 [119,519] 外推） | 斜率 −2.00000 块/步，401 点最大残差 1.31 块 |
| 耗尽步预测 | 预测 807，实测 806，**误差 1 步** |
| 受害者停顿律（等第 rank 个自然完成） | 4/4 受害者绝对误差 **< 1 ms**；错 rank 负控误差 > 100 ms |
| 停顿构成 | **等待 96.4%，重算 3.6%** |
| 桥接算术 | 需 436 块，单受害者 237 块 → 最少 2 个，实测正好 2 个 |

**解释.** 当前纯 decode 路径的空闲块趋势容易估计；这不证明所有策略信息已充足。
最后一次新准入在步 94，之后只调 admission cap 无法改变已入场集合。
终态需求差额为 521 块 = 1.018 GiB = 终端需求的 6.36%；两个已测端点呈现显著等待权衡——
cap29 保守准入把最长 ITL 压到 0.108 s，却把最长 TTFT 推到 17.9/18.1 s、墙钟 +17.0%/+17.1%；
cap32 原生抢占保住 TTFT，但集中 4.47/4.51 s 停顿在 2 个受害者上。两者互不支配，
增加实际 KV 能消除观测到的抢占，却略增加平均完成时间，因此也不能称为所有指标上的支配。
原生抢占在时机和受害者数量上是否最优尚未证明。

在“所有 prompt 都在首次完成前常驻、只改变其 decode 进度”的受限错峰模型中，
prefill 占 footprint 75%，动态范围为 `decode_blocks = 64` 块/请求；覆盖 521 块需延后 ≥ 9/32 个请求各 ≥ 928 步。
这不覆盖延后完整 prefill、改变 decode 推进或完成释放时刻的策略。
同一公式对短 prompt 长输出域（128/4096，prefill 份额 < 0.05）给出相反预测，`UNRUN`。

**旧结论边界.** 这些诊断解释了原运行中降 cap 生效太晚和保守容量的代价，
还没有证明它们同属一个不可绕过的结构限制，不能外推为“KV 压力域没有调度收益”。

**不主张.** 零执行，无反事实；停顿律样本仅 4 个受害者、1 个工作负载点；
竞赛判据用声明输出上限，本负载全部生成满 1024 token 是最有利情形；
静态判据在 budget95 上保守 84 块；尾部 KV 回滚是设计，无实现无实测。

**同场完成的源码核查（v0.26.0，纯阅读，未安装未实现）.**

- 所需原语 `SingleTypeKVCacheManager._remove_blocks_in_range` **存在**且在生产路径被使用，
  但唯一入口 `remove_skipped_blocks` 是头部语义，且 `FullAttentionManager.get_num_skipped_tokens -> 0`，
  OLMoE 配置下部分释放**完全不可达**。
- **硬阻塞**：worker 侧块表 append-only，释放只置 `null_block` 不缩表；
  而 `allocate_new_blocks` 用 `cdiv(num_tokens, B) - len(req_blocks)`，
  nulled 尾槽仍计长度 → 回滚后再生长会算出 0 并写进 null block。**不变量冲突，不是参数问题。**
- `Request.num_computed_tokens` 是标量，无"算到 N 但有洞"的表示；`_preempt_request` 全量清零并入 waiting 队首。
- 需新增实现 4 处，**风险集中在第 3 处**（解除 worker append-only 或走既有"resumed 请求重发块表"通道）。
  裁决：`IMPLEMENTABLE_BUT_REQUIRES_INVARIANT_CHANGE`。
- **独立验证**：FCFS victim = `self.running.pop()` = 最新加入者；实测两个受害者正是第 31、32 个到达者，
  与源码行为一致，说明账本解读没有错位。

**核查暴露的两个配置混淆项（新增，必须带进后续实验）.**

- **全部封存 cell 都是 `enable_prefix_caching: False`。** 而上游 `free()` 特意逆序释放，
  注释写明是为了 "tail blocks evicted first **when caching is enabled**"——
  即 vLLM 本来就有让抢占重算变便宜的机制，封存实验把它关了。
  不推翻"停顿 96.4% 是等待"，但"重算 3.6%"是在最不利于原生抢占的配置下测的，且非默认 serving 配置。
- **抢占步不调度任何 waiting 请求**（`if not preempted_reqs and ...` 才进 WAITING 循环）。
  这是受害者等待的二阶来源，模型未计入，量级未测。
- `scheduling_policy=PRIORITY` 是**零实现成本**的 victim 选择旋钮，但按停顿律它只改变"谁承担"与重算量，
  不改变停顿长度。

**下一动作.** 纯阅读，确认 worker 侧 append-only 不变量的确切作用域：
读 `vllm/v1/worker/block_table.py` 与 `gpu_model_runner.py` 的 `CachedRequestData` /
`NewRequestData` 块表写入路径，判定被截断请求能否走既有 "resumed 请求重发块表" 通道绕过，
而不改 worker 内核。产出是二选一判定 + 对应实现量。
可绕过 → 第一笔 GPU 支出是"2 次大回滚 vs 32 次小回滚"成本对照；
不可绕过 → 先做零实现成本的 `enable_prefix_caching=True` 四 cell 重测，
因为它可能直接改变"重算 3.6%"这一前提。
在此之前不实现选择器、不扫参数、不启动第二条实验链。

---

## 保持冻结、本轮不推进

- [expert-union 采集](../../outputs/admission_capacity/20260910_expert_union_r01/DECISIONS.md)：`UNRUN`，
  代码与输入已备。它回答 offload 域的另一个问题，本轮结论不改变其有效性也不提升其优先级。
- 真实 expert pager：新增 vLLM0.26 eager Triton / OLMoE / WiSP 接入已完成 2 个短请求 cell，局部 kernel 校验通过；同预算动作效果仍未测。见 [新增数据接续记录](RESEARCH_EXPERIMENTS.md)。
- Qwen3-30B-A3B 自然超显存对照：`UNRUN`，数据盘可用约 28.19 GiB，checkpoint 量级约 60 GB，**空间不足**。
- 自然长上下文 prefill 尾块（16 篇全文 / 29,103 tokens）：已备，GPU 0 次。

---

## 2026-09-12 · 恢复准入源码核查与解释修正

本节补充同一固定预算问题的直接源码证据；上文原始诊断保留，但不能据事后拟合排除其他请求级动作。

1. **静态终端需求超过池不等于必然抢占。** `8192 > 7671` 说明最大上下文不能同时常驻；合法调度可能在该状态出现前完成并释放请求。`predicted_preemption` 只在相应推进/释放假设下成立。
2. **等待误差 < 1 ms 是账本重建。** 原分析使用了真实未来完成时间与测得重算跨度；它没有验证一个在线可用的恢复时间预测器。原耗尽步分析也从整条轨迹/后续受害者取得 width 和 block size，不能把全部实现统称为因果窗输入。修正版单独保留输出，不覆盖原报告。
3. **桥接受害者数和错峰算术不是跨策略下界。** 它们分别依赖当前完成时间、受害者 yield、恒定宽度，以及所有 prompt 在首次完成前已入场的简化。尚未证明“原生时机已最省”或所有请求级动作“无杠杆”。
4. **固定资源必须读实际池。** 同 `gpu_memory_utilization` 的两次 `.95` 初始化仍相差 48 块。下一对照直接固定 `kv_cache_memory_bytes=16089350144`，并核验 7,671 可用块。

[安装源码与账本核查](../../outputs/admission_capacity/20260912_recovery_admission_r01/preparation/SOURCE_LOCALIZATION.md)发现：vLLM 对 waiting/recovery 请求先检查完整已有历史是否可容纳，再分配本次 chunk。旧两次低预算各有 **187 次首 chunk 可放、完整历史不可放**的失败，另有 32 次两者均不可放。本轮已通过原生配置干预检验其作用。

**`scheduler_reserve_full_isl=True/False` 四项对照已完成。** 固定模型、精度、实际 7,671 可用块、cap32、3072/1024、50ms、budget1024，四个新引擎按 `full → chunk → chunk → full` 独立执行，128/128 请求完成。结果见 [REPORT.md](../../outputs/admission_capacity/20260912_recovery_admission_r01/REPORT.md)。

放松检查使第 807 步更早恢复，却在第 810 步因末 chunk 需 50 块、仅剩 41 块而再次抢占。两轮各发生 68 次无新输出的再抢占，增加 116,242 个重算位置。最大 ITL 从 4.589/4.572 秒升至 5.973/5.866 秒，吞吐降低 6.89%/4.74%，平均完成延迟增加 7.83%/5.68%。原生 full 是本轮更强基线；成本模型需表达恢复期间的后续增长保障。当前没有同预算方法收益，也没有问题层面的判死证据。

最小在线模型使用到达/阶段、computed/generated tokens、声明输出上限、实际分配块、最近 token 时间和恢复状态。无 sharing 时，`delta_i = max(0, ceil((computed_i + scheduled_i)/block_size) - allocated_i)`；块守恒为 `F_next = F_now - allocations + frees`。准入上限下降不直接释放既有 KV。时间评价保留全部请求，分别报告完整吞吐、最大 ITL、TTFT、平均完成时间和失败，重算 span 不重复加到总延迟。当前没有统一业务 SLO，先展示权衡，不以未经校准的阈值证明贡献。

用户授权后的四项运行、预热、日志与 raw 均已回传。两次配对各 31/32 条完整输出相同；当前是原生自由生成测量，没有位级一致或质量结论。下一步针对“保留 KV 并在耗尽前保住一个请求完成余量”的最小动作；该动作未实现、未实测。命令与证据见 [RESEARCH_EXPERIMENTS.md](RESEARCH_EXPERIMENTS.md)。

WiSP 小权重 CUDA pager smoke 已有 6 项通过，见[留存执行记录](../../outputs/admission_capacity/20260912_admission_paging_r01/pager_smoke/readback/execution.json)；它不能替代完整模型 paging 性能或证明 vLLM 0.26 移植已完成。本轮不启动另一套 pager 机制。

## 2026-09-12 · 保 KV 完成余量保护（实现完成，GPU UNRUN）

承接 full/chunk 负结果，最弱链路转为保 KV 暂缓是否有合法运行路径。两轮原生 full 的第 799 步都有 F=16、leader 剩余上限需 H=14、全 batch 下一步申请 3 块的可行动作窗口；未来第 806 步抢占不参与触发输入。原生 running skip 保留 KV、computed 与 status，worker 可从 cached state 接续，未进入 PREEMPTED。

[completion_headroom.py](completion_headroom.py) 只在闭合 cohort 的纯 decode 阶段保护一个请求的完成余量，其余请求按原顺序使用非保留空闲块，零新增块进度仍允许。5 项 CPU 检查通过，含 150 组同长度上限状态推进；没有模拟时延或性能结论。模型与原生适配边界见 [REPORT.md](../../outputs/admission_capacity/20260912_completion_headroom_r01/REPORT.md)。

唯一下一实验为同池 native/headroom/headroom/native。当前机器只读检查为空闲，但上传执行被自动审批拒绝，理由是此前明确授权仅覆盖旧四项恢复准入。新包、代码和命令已齐备，GPU 0 次；方法/质量/Oracle 未验证。无新的科学判死，不继续增加 CPU 审计来替代 GPU 数据。

### 同一保 KV 方案的真实结果：机制生效，但未保住吞吐

用户授权后4/4项完成、128/128请求。两轮均在step799开始held，31个请求被暂缓，零抢占/重算；maxITL从4.469/4.448s降到1.381/1.381s。但吞吐−3.84%/−3.90%，平均完成+8.28%/+8.46%；每轮29/32自身maxITL变差、31/32完成更晚。该机制把等待分散，未满足原主目标，不能用少数最坏请求改善掩盖代价。

互斥成本分解表明额外调度0.467/0.577s、额外非调度引擎区间0.449/0.343s；少7685重算位置却多92次引擎调用。当前不能断言全部退化来自Python或batch计算。成本模型需同时表达调度观察成本、批次执行和每请求暂停分布。下一最小优化保留逐步held选择，削减尚未改变调度时的监测/检查；优化未实现，收益未测。所有结果和失败见[完整报告](../../outputs/admission_capacity/20260912_completion_headroom_r01/REPORT.md)。证据是单模型单GPU原生in-process两次同cohort描述性配对，质量/独立负载/Oracle均未验证。

### 2026-09-13 · 同动作 fast 实现与共享对照

优化完整块表读取后，同池 `native/headroom/headroom/native` 四项全部完成，128/128 请求。两轮动作与旧 headroom 各 1440 步一致；全轨迹早期入队状态差异保留为 `DIVERGED`，不做跨轮时延归因。当前吞吐 −3.25%/−2.86%、平均完成 +7.61%/+7.23%，零抢占/重算；两轮各 29/32 请求自身 max-ITL 增大，30/32 完成更晚。见[本轮报告](../../outputs/admission_capacity/20260913_headroom_fast_r01/REPORT.md)。

剩余成本定位：width=1 decode 调用由 129 增至 290；引擎非调度区间净增 0.497/0.438s。含重算的 8 个原生调用还包括 233 个新 decode 位置，整段时长不是纯重算成本。解释停在实际 host 调用分解，尚不能断言固定税或 GPU 成本不可消除。

下一实验统一为四臂共享对照，以 fast headroom 作为轮转的竞争基线。轮转的约 0.39s 仍是原生轨迹上的条件推演，未验证实际 max-ITL 或吞吐，不能用算术点声明已支配实测策略。双请求保护不另行并行实现。GPU 运行前留存设备与进程查询结果，失败或占用即退出；当前不启动新运行。
