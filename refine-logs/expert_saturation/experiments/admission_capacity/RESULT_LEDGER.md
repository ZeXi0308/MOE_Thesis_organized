# 共享数据结论台账

更新：2026-09-13。新一轮先按输入与指标查本表；复用已有结论，必要的独立复核注明目的。共享 raw 只读，修正保留原结果并链接 addendum。本表是唯一共享结论表；CONCLUSION_LEDGER.md仅为兼容入口。本表登记来源，不划分结论所有权；不替代各实验的原始证据和最新裁决。

`O` = `refine-logs/expert_saturation/outputs/admission_capacity`。一行一条结论；同文档/cohort 的不同 GPU 运行分别登记，离线估算与实际策略执行分开。

| ID / 日期 | 已有结论及口径 | 原始输入 | 已有分析 / 边界 |
|---|---|---|---|
| retention-reuse / 09-13 | full/guard保护项约84.8%/84.5%在下一同层调用复用，共同需求少/多miss1278/1153、973/931，净收益大量抵消 | `O/20260912_wisp_olmoe_r01/group_guard_performance` 12实际episode | [reuse](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/group_guard_performance/retention_reuse.json)。POST_HOC实际寿命；ensure不算使用，未见使用右删失；不是纯cache因果收益，重复非独立 |
| layer-budget-calibration / 09-13 | cap24精确复现1952calls；384槽固定trace少miss5513/636组 vs uniform5594/636，少groups5554/624；预热组数分别3017/2982 vs2939 | 同上首none episode，含五预热 | [calibration](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/group_guard_performance/layer_budget_calibration.json)。STRUCTURAL_FIXED_TRACE_CALIBRATION_ONLY；非GPU反事实/Oracle。三静态映射独立48层数值资格通过；跨文档真实结果见下一行 |
| layer-budget-transfer / 09-13 | 六engine48测量：U/M/G实际loads5224/5195/5209、groups631/627/627；M对U capture−1.477%/−0.811%、cycle−1.047%/−0.017%，G/M capture变号 | `O/20260912_wisp_olmoe_r01/layer_budget_performance`，校准外3文档、144测量请求/1920输出及240预热保留 | [analysis](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/layer_budget_performance/analysis.json)。MEASUREMENT_ONLY；少搬转移0.555%/0.287%，未确认稳定完整收益。4attempt/3零请求失败及慢样本全留，非连续独占/噪声界。下一项仅共享池kernel接口资格 |
| rotation-first-swap / 09-13 | C首次most后least对A吞吐+0.166%/−0.425%，对持续B−1.332%/−1.513%；末两剩余87/99→58/103，width2少29、width1多33，局部减少被抵消90–91% | `O/20260913_rotation_first_swap_r01/execution02_weste_23478`，6格192请求/196608输出，同一复用cohort0 | [实测补充](../../outputs/admission_capacity/20260913_rotation_first_swap_r01/RESULTS_WESTE_ADDENDUM.md)。NATIVE_SERVING/MEASUREMENT_ONLY；六项合格、独立重算零差异，same-family provisional PASS；首次动作不足，不是问题NO-GO/噪声界/新holdout/方法GO |
| rotation-victim-order / 09-13 | most_output对least_progress四块吞吐+1.134%–+1.560%、max-ITL少19.638–24.030ms，平均完成+1.004%–+1.471%；+4910重算、纯decode−104，末两请求width2收尾87→8 | `O/20260913_rotation_victim_order_r01/execution`，8格256次完整请求，复用H两cohort | [本轮报告](../../outputs/admission_capacity/20260913_rotation_victim_order_r01/REPORT.md)。NATIVE_SERVING / MEASUREMENT_ONLY；首次选择step836分离，forced8→9/natural均2/held0；同角色漂移非noisebound，非新holdout/全面净收益/公平或方法GO |
| control-drift-review / 09-13 | waiting全短三臂首次入场相同，但逐步调度不同（forward306/301/304步）；waiting/per_request各自负控TTFT均值最大绝对差2.824%/5.904%，max-ITL最坏值22.969%/60.933% | 两实验既有 `analysis/summary.json` 及waiting六个all_short raw | [定向复核](../../outputs/admission_capacity/20260913_runtime_drift_review_r01/REPORT.md)。CPU重析；六个共享臂差值非独立A/A、非噪声界、不能校准mixed。全通过时goodput等于吞吐；host_timing的115/512是共享raw重分析 |
| pause-budget90 / 09-12 | 两个 victim 的等待占总暂停 96.42% / 96.38%，剩余为恢复调用跨度 | `O/20260912_kv_budget_r01/gpu_results/repeat{0,1}-budget90/raw.json` | [kv_deficit_law §2.3](../../outputs/admission_capacity/20260912_kv_deficit_law_r01/REPORT.md)。CPU 重析已测 native 轨迹；不是纯 GPU 重算时间。其跨策略下界表述已由 [ADDENDUM](../../outputs/admission_capacity/20260912_kv_deficit_law_r01/ADDENDUM.md)撤回 |
| pause-native32 / 09-12 | 最长 ITL 4.473448 = 等待 4.357883 + 恢复调用跨度 0.115565 秒；重复 4.512153 = 4.396435 + 0.115718 秒 | `O/20260908_native_preemption_r01/gpu_results/repeat{0,1}-native32/raw.json[.gz]` | [已有复算](../../outputs/admission_capacity/20260912_admission_paging_r01/REPORT.md)。与上一行同类现象，但 raw 和暂停分母不同；不把约 96.3% 与 96.4% 的差值直接归因为窗口差异 |
| preemption-spacing / 09-12 | 两次抢占 step809、931，相隔 122 steps；两 victim 缺席 221、95 steps | 同上 native32 两条轨迹 | [absence_rotation](../../outputs/admission_capacity/20260912_absence_rotation_r01/REPORT.md)。原路径实测间距，不是所有策略的固定周期或不可减少的缺席下界 |
| rotate-c20-estimate / 09-12 | cooldown20 的最大缺席约 0.39 秒、额外重算成本比例 2.6%（合成时长分母） | 同上 native32 轨迹；重算调用相对纯 decode 的边际成本估计 | [修正上界表](../../outputs/admission_capacity/20260912_absence_rotation_r01/ADDENDUM_MARGINAL_COST_AND_FEASIBILITY.md)。历史离线算术；分母为 steps×纯 decode 中位时间，而非实际 cohort wall。后续分母修正见 rotate-c20-denominator-fix 行；两版均不是已执行轮转的 max-ITL/吞吐，队列、完成分布及实现税未闭合 |
| headroom-checked / 09-12 | 最大 ITL 1.381322 / 1.381332 秒；同轮吞吐 −3.84% / −3.90%；29/32 请求各自 max-ITL 增大 | `O/20260912_completion_headroom_r01/execution02/gpu_results`，四个独立策略 cell | [completion_headroom](../../outputs/admission_capacity/20260912_completion_headroom_r01/REPORT.md)。128/128 请求，NATIVE_SERVING / MEASUREMENT_ONLY；零抢占/重算，平均完成变差 |
| headroom-fast / 09-13 | 最大 ITL 1.386702 / 1.385728 秒；同轮吞吐 −3.25% / −2.86%；29/32 请求各自 max-ITL 增大 | `O/20260913_headroom_fast_r01/execution/gpu_results`，四个独立策略 cell | [headroom_fast](../../outputs/admission_capacity/20260913_headroom_fast_r01/REPORT.md)。128/128 请求；同动作减少观察成本的最新已测基线。新比较使用此实现，旧版继续保留 |
| headroom-work-cost / 09-13 | width=1 decode 调用 129→290；引擎非调度区间净增 0.497/0.438s | 同上 fast 四项调用账本 | [work_cost.json](../../outputs/admission_capacity/20260913_headroom_fast_r01/analysis/work_cost.json)。互斥 host 调用分解；含执行/采样/同步。原生 8 次含重算调用还含 233 个新 decode 位置，非纯重算成本 |
| actual-kv-budget / 09-12 | 实际 KV 增大后吞吐 +3.99% / +4.70%，秒级暂停消失，平均完成仍 +1.02% / +0.15% | `O/20260912_kv_budget_r01` 四项 | [预算对照](../../outputs/admission_capacity/20260912_kv_budget_r01/RESEARCH_REPORT.md)。NATIVE_SERVING / MEASUREMENT_ONLY；资源配置收益，非同预算机制 |
| recovery-chunk / 09-12 | 放松恢复完整历史检查：抢占 2→70，吞吐 −6.89% / −4.74% | `O/20260912_recovery_admission_r01` 四项 | [恢复准入对照](../../outputs/admission_capacity/20260912_recovery_admission_r01/REPORT.md)。128/128 请求；仅该规则失败，不否定整个恢复问题 |
| expert-group-baseline / 09-12 | token→expert 分组 wall 3.204/3.366→1.483/1.232 秒；payload 124.911→37.686 GB | `O/20260912_wisp_olmoe_r01/expert_baseline/performance` | [paging 报告](../../outputs/admission_capacity/20260912_admission_paging_r01/REPORT.md)。0.26、cap24、KV1GiB、4短请求；已有分组思路的强执行基线，晚到请求 max-ITL 有代价，不是调度创新 |
| release8-expert / 09-13 | 旧请求完成后解除 chunk8：wall 2.176/2.206→1.979/2.007 秒，约 −9%；后续 decode 多搬 0.365 GB | `O/20260912_wisp_olmoe_r01/release_baseline/results/same_engine_release_abba` | [paging 报告](../../outputs/admission_capacity/20260912_admission_paging_r01/REPORT.md)。同 engine 3短请求；普通阶段反馈基线，新请求 max-ITL 略增；不与 resident 长上下文 headroom 数字直接排名 |
| rotate-c20-denominator-fix / 09-13 | 最新修正仍预计缺席 0.389864 / 0.390988 秒；预计重算增加 0.683576 / 0.679790 秒，占原轨迹实测跨度 2.95168% / 2.91701% | 同上 native32；分母改为 23.158868 / 23.304316 秒 | [修正数据](../../outputs/admission_capacity/20260913_rotation_vs_headroom_r01/bound_corrected/rotation_bound.json)。替代旧合成分母；仍为离线估算。增加时间/基线时间不是吞吐降幅，不证明轮转优于实测 headroom；对应原生动作现已实跑，实测1.006s及实际成本见rotation-fourarm行，旧估算保留为被修正的历史 |
| pager-map-batch / 09-13 | 同资源 static8 的逐标量 expert-map 写改整表复制：四格输出/调度/163.089844 GiB 专家 payload 相同；配对 wall −29.41% / −12.27% | `O/20260912_native_pager_r01/phase_baseline/batched_map`；原始 result 保留在远端 `map-batch-r01` | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/batched_map/summary.json)。vLLM0.11.2/WiSP cap16，KV512MiB，3短请求；工程税移除，幅度有波动，不是 scheduler 收益 |
| phase-optimized-free / 09-13 | 同优化 pager 的 static8→phase8：wall −1.62% / −1.03%；旧请求 max-ITL +0.879 / +2.859 ms；专家 payload 少 1.300781 GiB | `O/20260912_native_pager_r01/phase_baseline/optimized_holdout`；远端 `phase-optimized-holdout-r01` | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/summary.json)。四格同 action 前状态；新请求自由输出跨策略不同，整体结果包含后续轨迹变化，非纯执行加速 |
| phase-optimized-controlled / 09-13 | 三请求输出跨四格一致时，static8→phase8 wall −0.91% / −1.72%，engine calls 27→24，专家 payload 只少 60 MiB | `O/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens`；远端 `phase-controlled-token-r01` | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_holdout/controlled_tokens/summary.json)。新请求重复 token 的诊断；旧 max-ITL +0.624 / +0.339 ms。primer 后状态与自由生成 campaign 不同，不能把两者字节差直接当因果分解；MEASUREMENT_ONLY |
| decode-retention / 09-13 | 同release8四机制八次：decode保护payload+0.41%、groups+30.08%、wall+2.28%/+8.97%；频率保护少搬1.70%但wall未改善 | `O/20260912_wisp_olmoe_r01/retention_performance` | [原始分析](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/retention_performance/analysis.json)。24请求执行/320输出、同3篇文本；当前完整decode集合保护无净收益，保留时间尖峰，不外推cache家族失败 |
| optimized-static-cost / 09-13 | 8/32端点在16执行前预测首步299.833ms，16实测305.031/305.745ms（残差+1.734%/+1.972%）；32相对8 wall−5.70%/−4.09%，旧maxITL约203→562ms | `O/20260912_native_pager_r01/phase_baseline/optimized_static_calibration`；远端同名-r01六格 | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_calibration/summary.json)。同前状态/任务/资源；未见动作的首步预测，不是独立文档/完整请求预测。不同cap自由输出不同；MEASUREMENT_ONLY |
| optimized-static-transfer / 09-13 | 冻结旧曲线迁移到新3篇文档：8/16/32首步排序保持；实测分别约150/268/481ms，比预测175/300/549ms低10.7%–14.3%（以预测为分母） | `O/20260912_native_pager_r01/phase_baseline/optimized_static_transfer`；远端同名-r01六格 | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_transfer/summary.json)。同资源与每域同状态，所有预测先于GPU；域内首步重复差0.01%–0.15%，但新旧文档跨进程，尚不能把偏差归因文档或专家状态。MEASUREMENT_ONLY |
| rotation-fourarm / 09-13 | 8/8完成：轮转最大ITL 1.006/1.006s；对同块native吞吐−0.548%/+0.055%、平均完成+2.244%/+1.600%；对headroom吞吐+2.435%/+2.242%且最大ITL更低 | `O/20260913_rotation_fourarm_r01/execution/gpu_results`，八个独立引擎cell | [报告](../../outputs/admission_capacity/20260913_rotation_fourarm_r01/REPORT.md)。256/256请求；8 forced+2 natural，0 held；多重算30,876位置但总调用少91。两个新受害者停顿约0.95s；旧0.39s漏计恢复队列与冷却。探索性权衡，未证明原生吞吐非劣或泛化 |
| rotation-holdout / 09-13 | 20/20完成：轮转最大ITL 1.004–1.014s；四块对native吞吐−0.619%–+0.475%、平均完成+1.109%–+2.284%；对headroom吞吐+1.943%–+2.781%、平均完成−4.257%–−5.067% | `O/20260913_rotation_holdout_r01/execution03/gpu_results`，两新文本cohort各两block、四臂与native A/A | [报告](../../outputs/admission_capacity/20260913_rotation_holdout_r01/REPORT.md)。640次完整请求/64新文档；四对A/A吞吐最大已观察绝对差0.782%，非噪声界。新受害者约0.953–0.960s；8 forced+2 natural、0 held，计数路径与完整成本保持。MEASUREMENT_ONLY，无native吞吐非劣/质量/新域/方法GO |
| retention-order / 09-13 | 固定P晚加载达到当前层miss-once最少组数；实跑frequency 810→714组、bytes−0.87%；相对none少搬2.40%，四块wall−0.19%/−0.44%/+0.19%/+51.56% | `O/20260912_wisp_olmoe_r01/retention_order_performance` | [完整比较](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/retention_order_performance/analysis.json)。两engine16episode、48请求执行；组数/搬运改进已测，稳定请求收益未确认。末块动作前已变慢，全部保留，不将none重复范围当噪声界 |
| runtime-variance / 09-13 | 同程序none八次的row-topk/输出/636组/70.389GB均相同；两次call11为198.7/256.3ms，与77.8/135.3ms gen2 GC包络对齐；此前持续动作前漂移未复现 | `O/20260912_wisp_olmoe_r01/runtime_variance_performance` | [GC与CPU交集](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/runtime_variance_performance/analysis.json)。24测量请求/320输出，40预热保留；GC不是可直接扣除成本，观测自身5–25ms计费，不构成方法收益或噪声界。下一步仅对照trace保留生命周期并完整计费 |
| document-state-cost / 09-13 | 同engine fixed16四格：old/new首step为134/119分组、1164/1012加载（13.640625/11.859375GiB），各组重复完全一致 | `O/20260912_native_pager_r01/phase_baseline/document_state_abba/attempt02`；远端document-state-abba-r02 | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/document_state_abba/attempt02/summary.json)。new首步短10.56%/4.83%，同文档时间仍漂移；全episode差−4.98%/+0.11%。文档组前状态不同，计数是post-action；确认执行工作量差异，非稳定性能或scheduler GO |
| trace-lifecycle / 09-13 | 四新engine memory/episode/episode/memory，32测量的row-topk/分组/搬运/输出一致；episode释放使已观测GC交集减少0.672/0.645s，完整cycle仅−1.283%/−0.200% | `O/20260912_wisp_olmoe_r01/trace_lifecycle_performance` | [完整周期及请求账](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/trace_lifecycle_performance/analysis.json)。96测量请求/1280输出、160预热保留；首对进程−14.84%主要来自cycle外，不归因释放；GC观测不含关闭后的尾部。仅测量实现改进，无噪声界/方法GO。下一步用共同episode写出复测none与late-frequency |
| incoming-content-cost / 09-13 | 同旧pair/前态 fixed16四格，仅第三incoming文档改变：首步1164→994加载，旧请求跨步间隔约308→266ms | `O/20260912_native_pager_r01/phase_baseline/incoming_document_abba`；远端incoming-document-abba-r01 | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/incoming_document_abba/summary.json)。4/4 COMPLETE；13.640625→11.6484375GiB tensor payload；两次约−41/−43ms。post-action输入内容因果诊断，非scheduler加速/总体噪声上界；首chunk信息仅能指导下一动作 |
| fourarm-noise-floor / 09-13（口径修正） | 继承报告的同策略跨正反序最大已观测绝对差：\|Δwall\| 0.2684秒、\|Δmax-ITL\| 0.2126秒；仅描述重复差异，不是噪声上下界 | 同 rotation-fourarm 八格 metrics | [原离散度报告](../../outputs/admission_capacity/20260913_completion_dispersion_r01/REPORT.md)保留。每格n=1且与顺序混淆；“rotate吞吐差小于已观测重复差”不证明无代价、非劣或统计不可分辨。safe29近零分母影响相对百分比，不会抬高绝对秒差；改善量/该最大差的倍数也不是显著性证据 |
| width-unit-cost / 09-13 | 同 episode 内每 token 成本 g(w)=c(w)/w：w32 0.608、w29 0.613、w3 2.205、w1 4.451 ms/token | 同上八格 raw 的纯 decode step（排除含 prefill/recompute 步） | [dispersion.json](../../outputs/admission_capacity/20260913_completion_dispersion_r01/analysis/dispersion.json)。step 时长用相邻 `start_s` 差，因 `end_s` 仅为 schedule() 返回时刻。该量含与高宽度步重叠的 prefill/记账，**不是 kernel 时间** |
| tail-dispersion-model / 09-13 | Δwall = Δ边际重算 + Δ尾部浪费 闭合：safe29 解释 87.7%、headroom 71.1%；rotate 的 +0.326 秒重算被 −0.336 秒尾部节省抵消。完成跨度 native 2.459、rotate 2.146、headroom 6.074、safe29 7.953 秒；safe29 有 941 步跑在宽度 3 | 同上 | [recompute_cost.json](../../outputs/admission_capacity/20260913_completion_dispersion_r01/analysis/recompute_cost.json)。**描述性会计，非反事实**；残差全为正（+0.10~+0.50 秒）未拆分。前提是同构输出长度使完成天然聚集，**异构域未测**，该前提是结论范围而非附注 |
| deficit-victim-identity / 09-13 | 结构赤字 32×256−7,671 = 521 块 = 2.035 请求。native32 抢占的 `-0003640`/`-0003571` 与 safe29 三个 SLO 失败请求中的两个完全重合 | `O/20260913_rotation_fourarm_r01/execution/gpu_results/repeat0-{native,safe29}` | 同上 REPORT §4。同一负载同一到达序下的重合，**不是所有策略/负载的固定受害者集合** |
| resume-gate-256 / 09-13 | 恢复受 `full_sequence_must_fit` 的 256 块门槛限制，实际只需 64 块；缺席期 `free≥64` 成立 85.3%/100% 的步，`free≥256` 只 0.9%/1.0%；两 victim 均在首批完成后 0.4 ms 内恢复 | 同上 native32；KV 时间线由 `computed_after` 重建（完成即释放，自检无负值、峰值 7670/7671 块） | 同上 REPORT §4。源码 `vllm/v1/core/kv_cache_manager.py:411` 与 `scheduler.py:944`，实例级开关对 WAITING/PREEMPTED 无差别。解释了 `recovery-chunk` 单纯关闭该检查为何失败，也解释轮转为何须同 step 驱逐+恢复；**不构成放松该检查的授权** |
| one-next-chunk / 09-13 | 共同first16后next8/16/32，两个随机block共12格；B-next8相对16 wall−43.45/−40.50ms、oldmax−9.36/−8.05ms，新TTFT+6.14/+5.66ms；A oldmax反而增加 | `O/20260912_native_pager_r01/phase_baseline/next_chunk_branch`；远端next-chunk-branch-r02 | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/next_chunk_branch/summary.json)。同doc共同分支前态且全部输出相同；B payload−2.1328125GiB但calls19→20；普通首chunk比例模型失配。B-next32 wall符号翻转、未胜全程强static、无SLO/method GO |
| retention-lifecycle / 09-13 | 同episode写出下两engine16次：frequency少搬2.395%、多12.264%组；capture八对四升四降，完整净收益未成立 | `O/20260912_wisp_olmoe_r01/retention_lifecycle_performance` | [实际两臂账](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/retention_lifecycle_performance/analysis.json)。48测量请求/640输出、80预热保留；同模式轨迹相同，不据此建噪声界。当前完整frequency formulation，不是paging家族NO-GO |
| group-budget-opportunity / 09-13 | 当前frequency身份保持不变，要求gP=g0：每次none实际状态126/384、frequency状态112/384层调用仍可改变保护plan，各8次一致 | `O/20260912_wisp_olmoe_r01/retention_lifecycle_performance/group_budget_opportunity.json` | [CPU结构结果](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/retention_lifecycle_performance/group_budget_opportunity.json)。6144调用下界/覆盖/容量检查；POST_HOC_STRUCTURAL_ONLY。该结构分析时guard未接入/未实跑，不估计未来bytes或时间；后续实际对照见group-guard行 |
| one8-static-holdout / 09-13 | 两个新incoming×五策略×两随机block，20/20完成；c11的one8被static16在wall/oldmax/TTFT两次全部覆盖，c10仅留权衡 | `O/20260912_native_pager_r01/phase_baseline/one8_static_holdout`；远端one8-static-holdout-r01 | [summary](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/one8_static_holdout/summary.json)。one8相对16在c11 wall+74.92/+45.74ms、oldmax+24.03/+23.21ms、TTFT+104.59/+92.19ms；两臂输出相同。停止固定first16-next8-then16规则；不判paging/prefill家族NO-GO，无方法GO |
| group-guard / 09-13 | 当前gP=g0保护逐调用成立；三臂none/full/guard为636/714/637组，guard少搬0.840%、119 applied/71拒绝；capture四块一升三降，repeat cycle两升两降 | `O/20260912_wisp_olmoe_r01/group_guard_performance` | [三臂实际账](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/group_guard_performance/analysis.json)。36测量请求/480输出、60预热；guard保留full节省35.075%，后续并集变化带来额外一组，无稳定净收益。保护寿命/实际复用已完成，见retention-reuse及2026-09-15成本增补；无质量/方法GO |
| execution-baseline-cap16 / 09-13 | 同预算token/expert ABBA4/4：expert wall−26.15%/−25.95%、oldmax−50.11%/−49.61%、新TTFT−45.28%/−44.71%；payload−34.81% | `O/20260912_native_pager_r01/phase_baseline/execution_baseline_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/execution_baseline_v026/REPORT.md)。0.26/cap16/KV512MiB/token64，同batched map；全部输出、共同前缀与逻辑态一致，物理KV块不同。已有执行基线收益，非新scheduler；一个triplet/两次每臂，不造噪声界。下一expert静态动作曲面UNRUN |
| expert-chunk-curve / 09-13 | 6/6同前态expert8/16/32：wall最佳16、旧maxITL最佳8、新TTFT最佳32；32少搬约12.46GiB却wall+4.71%/+8.72% | `O/20260912_native_pager_r01/phase_baseline/expert_chunk_curve_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/expert_chunk_curve_v026/REPORT.md)。18请求/240输出跨6格相同；三请求decode4→7次，阶段增量抵消prefill节省。prefix漂移不归因动作、无噪声界；后续RR2见下行 |
| decode-width-rr / 09-13 | 8/8完成：RR2每层单组成立，但完整calls16→20、payload+1.74%、新请求等待增加，未显示优于static32 | `O/20260912_native_pager_r01/phase_baseline/decode_width_rr_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/decode_width_rr_v026/REPORT.md)。24请求/320输出；11轮空均下一步服务、KV不变。RR自身wall漂移22.83%，动作前TTFT/混合阶段差不能归因RR；无稳定效应量/方法GO。后续同任务admission cap2已完成，见下行 |

| admission-width / 09-13 | 6/6完整三请求：D/R保护旧decode，旧maxITL约173→66–101ms，新TTFT与wall均增加；R较D少搬12.57%，wall差−17.73%/+17.82%翻号 | `O/20260912_native_pager_r01/phase_baseline/admission_width_baseline_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/admission_width_baseline_v026/REPORT.md)。18测量请求/240输出；D/R新请求真实WAITING且KV为空12步，所有工作量/等待计费。同臂路由与输出重复，CPU计费漂移原因未定。下一同一D六格CPU profile定位；无噪声界/方法GO |

| length-dispersion-domain / 09-13 | 新运行域（42请求/1024输入/均值2048输出）成立：同构臂峰值并发42、峰值KV 7826/7671=102.0%、结构赤字393块=2.047请求足迹、抢占1次、42/42完成 | `/root/het-r01/results/block{0,1}-hom-native`、`block0-het-native`（远端 `/root` 系统盘，非 autodl-tmp） | [冻结设计](../../outputs/admission_capacity/20260913_heterogeneous_length_r01/DECISIONS.md)。两臂预留块数均8064、总输出均86,016 tok、均值均2048，仅 CV 0 vs 0.500 不同，因此长度离散是唯一变量。赤字以请求足迹计与原域 2.035 对齐。**这是新域：prompt 3072→1024、输出 1024→2048、N 32→42、足迹 256→192 块，不得与原域绝对数字混排名** |
| length-dispersion-impl / 09-13 | 上机定位并修复四处冻结 runner 的 32/1024 硬编码：`safe_static` 维度断言、`run_probe --cap` 选项、`rotation_native install(expected_requests=32)` 两处调用 | patch 见 [patch_safe_static_lengths.py](patch_safe_static_lengths.py)、[prepare_heterogeneous_package.py](prepare_heterogeneous_package.py) | 均为实现阻塞，非科学负结果。`expected_requests=32` 使 42 请求永不形成 cohort，四个 rotate 格因此 INCOMPLETE（`closed synchronous text cohort invariant changed`），失败原件保留在远端 `FAILED-rotate-expected32/`。另有一格 het-native 因与其它会话共占显存 OOM，同样保留 |
| length-dispersion-analyzer / 09-13 | 新分析器在同构四臂上逐项复现旧值 5/5（native32 尾部1.133/重算0.082、safe29 尾部4.747、rotate 尾部0.797/重算0.408），偏差均 ≤20 ms | [analyze_length_dispersion.py](analyze_length_dispersion.py)，`--expect` 回归开关 | 尾部浪费改为纯成本定义（w ≤ w_max/2 的步），不再要求低宽度步连续或位于末段，否则异构臂会被计入其负载固有的中途排空。另报告 `unavoidable_drain`（由冻结长度向量估计的负载固有排空），**只陈列、从不相减** |

| length-dispersion-executed / 09-13 | 8格执行，4个native有效（两block一致）：同构 wall 40.271/40.001秒、峰值KV 102.0/101.9%、抢占1、完成跨度3.51/3.50秒；异构 wall 49.695/49.561秒、峰值KV 68.7%、抢占0、完成跨度33.96/33.80秒。两臂预留块与总输出token相同 | `/root/het-r01/results/block{0,1}-{hom,het}-native`（远端 `/root` 系统盘）；本地 `O/20260913_heterogeneous_length_r01/gpu_results`，归档 SHA256 `e7eb84e0…1b5352bf` 双向一致 | [报告](../../outputs/admission_capacity/20260913_heterogeneous_length_r01/REPORT.md)。H1（长度异构→完成离散）成立，跨度9.7倍。**异构在同等token下 wall +23.6%、吞吐 −19.1%，与策略无关**。每格n=1，两block差0.27/0.13秒只是已观测重复差，非噪声界 |
| het-relieves-kv-pressure / 09-13 | 长度异构使峰值KV 102%→68.7%、抢占1→0。原因：`full_sequence_must_fit` 准入时按 prompt 预留（1024→64块），短请求在L=1024退场时长请求仅到~1024，全体峰值 42×128=5376块=70.1%，与实测68.7%吻合 | 同上四个 native 格 | 同上 REPORT §2。**我的赤字对齐只对齐了预留总和，未对齐时间峰值**——设计缺陷，非机制失败。因此异构臂无抢占、轮转无动作，两个 het-rotate 由实现自报 `rotation never applied a forced preemption`，按冻结设计记 `INVALID_NO_ACTION`。这也是关于 runtime 的结构事实：异构会自行削平KV峰值 |
| step-count-law / 09-13 | 无拟合口径：同构 2189步/84,130 tok/均宽38.4/步均17.73ms；异构 3156步/84,251 tok/均宽26.7/步均15.32ms。均宽降30.5%但步均只降13.6%，故步数涨44.2% | 同上 | 同上 REPORT §2。异构臂同episode内 g(42)=0.374、g(21)=0.697 ms/tok（1.87×）。这是步成本对宽度强次线性的第三个实例（前两：safe29 的941步宽度3、轮转靠维持高宽度） |
| step-cost-model-unidentifiable / 09-13 | 三项模型 `t=γ+α·w+β·C` 域内留出好（同构b1 +0.67%、异构 +3.94%，γ×Δ步数解释两臂decode差95.9%），但跨域高估11.8%~24.8%、p90逐步误差63–116%；系数跨域摆动 γ 4.39–9.45ms、α 3.70–281μs、β 54.6–102.6ns | 新域四格 + 原域 native/safe29/rotate/headroom | [cost_model.json](../../outputs/admission_capacity/20260913_heterogeneous_length_r01/analysis/step_cost/cost_model.json)。**serving trace 中 w 与 C 强共变，三项不可分别辨识**；同trace内β分段斜率比0.48–1.23，safe29为负。**保留无拟合的步数×步均口径，不主张物理系数**。分离宽度与上下文需固定上下文的受控微基准 |
| het-impl-blockers / 09-13 | 冻结 runner 四处原域硬编码已定位修复：`safe_static` 断言(3072,1024,32)、`run_probe --cap` 选项、`rotation_native install(expected_requests=32)`×2、缺 `HF_HOME`/`VLLM_USE_FLASHINFER_SAMPLER=0` | patch: [patch_safe_static_lengths.py](patch_safe_static_lengths.py)、[prepare_heterogeneous_package.py](prepare_heterogeneous_package.py)；CPU 检查 [test_heterogeneous_length.py](test_heterogeneous_length.py) 10/10 | 均为实现阻塞，非科学结果。失败原件保留远端 `FAILED-*`、`DIAG-cap32-no-pressure`（后者是"同负载并发受限"对照：峰值78.2%、抢占0）。CPU 检查抓到一个会在GPU上立刻崩溃的真实 bug（`arrival_traces_s` 写成列表而非按regime索引的字典）。另一格 hom-rotate 因与其他会话共占显存 OOM，保留不重跑 |

| step-floor-metric / 09-13 | 绝对效率度量 `步数/max(L)`：同 max(L)=1024 的原域四臂给出 rotate 1.092 < native 1.212 < headroom 1.310 < safe29 1.913 | 原域四臂 + 新域三格 raw 的纯 decode 步计数 | [下界推导见异构报告](../../outputs/admission_capacity/20260913_heterogeneous_length_r01/REPORT.md)。依据：decode 逐 token 串行，需 L token 的请求至少占 L 步，全部同时入场时纯 decode 步数 ≥ max(L)。**取代此前相对的"尾部浪费"口径**，因后者只相对本臂高宽度相、跨负载不可比。边界：最后一个 prefill chunk 步也产出首 token，故下界有约 2 步误差 |
| floor-metric-validated / 09-13 | `kv_budget_r01` 回溯验证：budget95（赤字 −233/−281，无压力）给出 步/下界 = **0.998 / 0.998**、max-ITL 0.083/0.099 秒、抢占 0；budget90（赤字 521）给出 1.212/1.212、max-ITL 4.591/4.560、抢占 2 | `O/20260912_kv_budget_r01/gpu_results` 四格 | 无赤字时系统恰好跑在物理下界上，**独立确认下界概念**，并提供天然的"零赤字"参照点。两次重复逐位一致（1241/1241、1022/1022 步） |
| deficit-cost-absolute / 09-13 | 以无赤字点为参照，赤字 521 块造成 **21.4% 额外步** 与 **4.353 秒停顿**。同显存下轮转回收 **56% 步浪费、79% 停顿**，吞吐 −0.55%（小于已观测重复差）；加显存（+754 blocks）回收 100% 但改变资源 | 同上两组 | 这是目前最强的结果陈述：把机制收益放在"物理下界 ↔ 加显存"两个参照之间。仍是单一压力点，适用区间由 `20260913_pressure_sweep_r01` 扫描确定 |
| het-wall-penalty-corrected / 09-13（更正） | 异构 23.6% wall 代价的归因更正：**106% 由 max(L) 从 2048 升到 3072 解释**，两臂各自都跑在自身下界的 2.7%/6.9% 以内 | 新域 hom/het native 格 | **撤回**同日 `step-count-law` 行把该代价归为"宽度塌缩"的表述。宽度塌缩是高下界的结果，不是独立原因。推论：固定均值下任何长度离散都必然抬高 max(L)（均值 2048 且 max=2048 ⇒ 全部 2048），故该代价是 makespan 被最长请求锁定，不是调度可回收的低效 |

| step-efficiency-deterministic / 09-13 | **步效率零运行间方差**：holdout 20 格中同策略纯 decode 步数逐 token 相同——rotate 1118、native 1241（含 4 格 native_aa 零动作对照）、headroom 1341、safe29 1959，CV 恰为 **0.0000%**；同 20 格墙钟 CV 为 0.05–0.43% | `O/20260913_rotation_holdout_r01/execution03/gpu_results` 全 20 格（另一执行链产生，本会话只读重算） | [发现记录](../../outputs/admission_capacity/20260913_pressure_sweep_r01/STEP_FLOOR_FINDING.md)。两 cohort 为**完全不相交文档**：请求 ID、prompt sha256、输出序列交集均 0/32。**这解决全线的 n=1 困难——步效率单格即为该策略在该工作点的值，不需重复估噪声**。机制：调度器只看队列与块计数、不看 token 内容，故决策序列确定；内容只影响每步耗时。**不替代墙钟**：策略可步少而每步更贵，两者须同时报告 |
| step-floor-precondition / 09-13 | 下界 `max_i L_i` 的前提「所有请求在首个完成前入场」在 **29/29 个 cell 上成立**（末次入场 1.55–2.06 秒 vs 首个完成 15.7–36.8 秒） | 原域四臂、kv_budget 两档、新域同构/异构、holdout 20 格 | [verify_step_floor.py](verify_step_floor.py)，产物 `O/20260913_pressure_sweep_r01/step_floor_validation.json`。不成立的 cell 会标 `FLOOR_NOT_APPLICABLE` 而不计分。下界比真值高约 1–2 步（末个 prefill chunk 步也产首 token），故无赤字时为 0.998 而非 1.000 |

| psweep-d2-reproduced / 09-13 | 压力扫描 d2 点四格（两 block × 两臂）全 COMPLETE，**步数逐位复现**：native 1241（步/下界 1.212、抢占 2）、rotate 1118（1.092、抢占 10）；墙钟 native 22.589/22.655、rotate 22.600/22.640 | `/root/psweep-r01/results/block{0,1}-d2-{native,rotate}`（旧机 `/root` 系统盘） | [发现记录](../../outputs/admission_capacity/20260913_pressure_sweep_r01/STEP_FLOOR_FINDING.md)。**第三次独立确认且用的是改过的执行包**（KV 字节参数化、池断言按请求推导、cap 选项），运行期间宿主有另一会话进程占 3650 MiB。墙钟跨执行包漂移 0.215–0.330 秒（CV 0.40–0.65%），**步数变异为 0**。这也是扫描包的内建复现检查：d2 字节精确等于 16,089,350,144，池断言推出同样的 7671 |
| psweep-impl-blocker-3 / 09-13 | 冻结 runner 第三处原域硬编码：`run_probe.py:116` 断言 `usable_blocks != 7671`，使 d0/d4/d6 十二格全部 INCOMPLETE（`actual reservation policy or KV pool differs`） | 失败原件保留远端 `/root/psweep-r01/FAILED-pool-assert-7671/`（12 格） | 修复 [patch_pool_assertion.py](patch_pool_assertion.py)：期望池由请求字节按封存比例（16,089,350,144 / 7672 总块，usable=total−1）推导，容差 ±2 块，**保留检查强度**而非删除。d2 点推导值仍为 7671，原有保护不变。实现阻塞，非科学结果 |

## 长上下文竞争机制：四臂协议与实测（8/8 COMPLETE）

本节记录接替轮转旧 addendum 三臂计划的执行协议；原封存文件不改。2026-09-13已按以下顺序完成8项；当时下一步的独立文本+A/A现已完成20项，见rotation-holdout行，当前接续见其报告。四臂为：

1. `native32`：原生抢占与恢复。
2. `safe29`：保守静态准入，engine 最大能力与其它臂一致。
3. `headroom32-fast`：采用 20260913 已测的单请求完成保护实现，保留 KV；与其它臂使用相同观察设置。
4. `rotate32-c20`：显式配置交换间隔20、初次等待30、恢复后驻留30；独立演进请求、KV、队列、输出和后续执行。原生接入与CPU路径夹具已完成，见[准备包](../../outputs/admission_capacity/20260913_rotation_fourarm_r01/REPORT.md)；已完成实际worker与四臂GPU执行，见[实测报告](../../outputs/admission_capacity/20260913_rotation_fourarm_r01/REPORT.md)。本轮held为0，不能据此声称保护必要。动作包括强制抢占、恢复队列优先与完整历史余量保护，held请求计入完整代价，不称为仅改顺序。

共同条件：同一 OLMoE BF16 revision、vLLM0.26/Triton 与执行模式、同一组32条3072输入/1024输出、50ms到达、budget1024、固定 **16,089,350,144-byte KV / 7,671 usable blocks**；prefix/speculation 与已有 resident 基线一致。不得只传 `gpu_memory_utilization` 而容许实际池变化。保守臂只改准入上限，不降低 engine 能力。统一预热、采样和观测，保留所有请求与失败。

顺序为上述四臂正序一次、反序一次；每个 cell 独立引擎，沿各自策略完整执行。复用原始数据来准备输入和分析口径，但旧 headroom 的两个数不替代新 campaign 同资源实跑。冷却20沿用已预选值，不因引入强基线改阈值。无联合 arm，不同时开启轮转与保 KV。

比较完整 cohort 墙时/吞吐、TTFT、每请求 max-ITL 和完成时间分布、改善/受损请求数、重算与暂停分解。只有同口径实测超过 `headroom32-fast` 及保守基线的暂停—服务量权衡，才支持相应贡献；只改善最坏请求或超过 `safe29` 不够。联合 SLO-goodput 需要运行前给定完整的 TTFT、mean-TPOT、max-ITL 门槛和失败分母；本修正不从既有结果反选 SLO。

## GPU 跑前检查与留痕

初始化前记录 UTC、启动进程 PID、GPU UUID/状态、计算进程 PID/name/显存，以及查询失败的错误。发现其它进程或无法确认占用时退出并保留记录；不终止其它任务。多 cell 每次加载前重查；同 engine 的重复边界只允许本 worker PID，结束再记状态。

这是边界占用检查，检查通过不保证下一刻无人启动。所有共享入口必须遵守同一规则；未统一接入的历史冻结脚本保留其原边界，不能追认为连续独占。当前推广入口为 [run_session.sh](run_session.sh)、[run_native_pager.py](run_native_pager.py) 与 [gpu_preflight.py](gpu_preflight.py)（optimized static calibration 父/子 runner）。后者已于 UTC 2026-09-12 16:43:18 因外部 PID36942 占用，在模型初始化前真实 ABORT/91；[记录](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/optimized_static_calibration/gpu_preflight_readback.json)。

当前 paging retention 实验属于固定小专家池、短 mixed batch 域，按 `none/frequency/decode/matched_hash` 四种机制（正反序共八次）比较。其全部臂继承 release8；长上下文四臂的 `headroom32-fast` 不是该域的同资源已测基线。两者共用本台账与 GPU 检查，各工作流一次只推进自身一条执行链，GPU串行占用。

| admission-cpu-profile / 09-13 | 6/6同D请求/轨迹/输出会计PASS，四份函数profile全部INVALID，未完成漂移归因；plain wall2.409815→2.344809s | `O/20260912_native_pager_r01/phase_baseline/admission_cpu_profile_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/admission_cpu_profile_v026/REPORT.md)。18请求/240输出，保留负函数时间及全部原始clock；禁用函数排名和成本占比。下一无profile显存统计等价读取对照，依据源码而非异常0.46s读数 |

| admission-memory-observer / 09-13 | 4/4同D，nested与flat同字段/同工作量，wall两对−78.171/−74.112ms；采用共同nested观测 | `O/20260912_native_pager_r01/phase_baseline/admission_memory_observer_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/admission_memory_observer_v026/REPORT.md)。12请求/160输出、每格1104个allocator值全等；新maxITL方向混合。不解释旧漂移，不计scheduler贡献；下一8请求clock arrival六格 |

| shared-pool-interface / 09-13 | r01 FAILED：负控allclose仍真；r02新4篇资格48影子调用按位通过，固定错误map可区分，8请求/64输出共两轮 | `O/20260912_wisp_olmoe_r01/shared_pool_qualification` | 仅384物理槽索引接口；额外shadow/reference，非同预算pager或质量/性能。首轮保留，下一真实跨层生命周期与同私有LRU对照 |

| continuous-admission / 09-13 | 6/6完成48请求/1584输出；phase仅step0实际64，之后无动作，较静态cap2 TTFT更差，较cap3 wall慢5.63%/6.80%；停止本域phase32 | `O/20260912_native_pager_r01/phase_baseline/continuous_admission_v026` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/continuous_admission_v026/REPORT.md)。每格928计算位置、ready decode全推进；等待/提交/执行三段闭合，跨策略输出不同保留，无质量/SLO/方法GO。下一真实超显存模型加载资格化，不调当前episode救规则 |

| shared-pool-lifecycle / 09-13 | r01固定stream假设失败、0测量；r02两模式各160真实层资格通过，oneshot160按位/640weight checks，8请求64输出 | `O/20260912_wisp_olmoe_r01/shared_pool_lifecycle_qualification` | 总384槽/KV1GiB；切流以event串行。跨臂cache/输出不同，仅各自生命周期资格，性能另由已启动U/M/S/X四臂两组新文本验证 |

| shared-pool-performance / 09-13 | 八格全部完成、失败0：X/U两block capture−5.542%/−5.931%、cycle+2.715%/+2.871%；X/M capture−6.729%/−6.893%、cycle+2.246%/+1.396%；X/S两者均改善 | `O/20260912_wisp_olmoe_r01/shared_pool_performance`；192测量请求/2560输出、320预热/10368输出、125184calls/24576测量层、152GPU边界，全repeat保留 | [analysis](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/shared_pool_performance/analysis.json)。384唯一槽/KV1GiB、H2D/D2D及全部成本计费；五warmup实验cycle不是持续服务摊销测量，请求正收益非net GO。两文档block与顺序混淆，相关重复、独立策略route/output，不作显著性/质量主张。下一普通20×16+64 full-stage资格/性能，CPU通过、GPU UNRUN |

| qwen3-native-qualification / 09-13 | r03四请求32输出/284位置完成；16片18,867源张量/435目标闭合；48层finite，47/48原allclose诊断通过，layer47差异待解释 | `O/20260912_native_pager_r01/phase_baseline/qwen3_native_qualification_v026/attempt03` | [报告](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_native_qualification_v026/attempt03/REPORT.md)。真实56.87GiB BF16权重/31.84GiB GPU，cap48/实际KV511.5MiB；GPU峰26.19GiB，cgroup总峰88.39GiB含文件缓存，不是独占RAM要求。原分析器预算等于分配的错误已用block公式修正；参考/JIT计时不作性能，静态对照UNRUN，无质量或method GO |

| global-lru-fixed-X / 09-13 | STRUCTURAL：旧X两格repeat0固定轨迹下global384 measurement loads9185/9541，对X实际5860/6258，多56.74%/52.46% | `O/20260912_wisp_olmoe_r01/new_host_20260913/global_lru_diagnostic.json` | 按原reset与五warmup连续回放自己的全局LRU，但后续输入仍为X观察轨迹；不是G实际状态/请求/性能或FreeToken复现，不相抵H2D/D2D估收益，不判死global基线 |

| shared-pool-warmup-cost / 09-13 | 旧X/U五预热正增量全部定位到3个protected普通expert路径，合计+0.496/+0.540s；两种共享执行预热合计−0.127/−0.144s | `O/20260912_wisp_olmoe_r01/new_host_20260913/warmup_cost_localization.json` | 80组ordinal×phase配对全保留。protected private21/24伴随更多H2D/groups，未隔离cap与继承状态等因果贡献；旧raw分析，非新GPU/持续服务净收益，不改X/F下一步 |

| qwen3-localized-static-return / 09-13 | 旧weste r01中断已回读：8完整分片/0数值0性能；返回host r02实际启动、仍LOADING | `O/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913` | [接续记录](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/README.md)。原始RUNNING未改、终止原因未知；新轮17源码与输入不变、系统盘分片，layer47资格通过才做静态ABBA。westc r02连接断开后状态未知；当前没有新性能/质量或method GO |
| qwen3-localized-static-return / 09-13 观察状态补记 | observer34278已rc255；最后live为UTC11:21:00.364910，parent13263/worker13275存活、第7片1,806,696,448/3,999,975,472B；当前远端worker/结果UNKNOWN | 同上 `attempt02/{observation_interruption01,reconnect_after_observer01,reconnect_after_observer02,handshake_diagnostic02}.json` 与 `watch_attempt01.jsonl` | [中断记录](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/observation_interruption01.json)、[握手](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/handshake_diagnostic02.json)：两次重连255；TCP建立后、SSH认证前收到HTTP/1.1 502 Bad Gateway。目标116.172.94.204本地路由经utun4，仅路径观察，未定位根因；不推断机器关机或实验终止。最后qualification为空，无新科学结果；原数据保留、未启停 UTC11:42:20.570[控制台](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_localized_static_v026/return_host_20260913/attempt02/browser_console_observation01.json)显示5条实例全off；F04配置相符但SSH端口映射未核验，目标终态回执仍UNKNOWN，无现成live网页终端；未开机/付费。 |

| gpu-startup-collision / 09-13 | weste GPU UUID0a66cc34，09:57:06.779/11.969 UTC两次查询空闲；FQ PID12185随后启动显存不足17.34/31.36GiB，8.761s退出 | `O/20260912_wisp_olmoe_r01/weste_resume_20260913/revision03/qualification_startup_failure` | 0pager层调用/0请求，非机制失败。边界PASS不保证连续独占；该时段其他实验需结合自身启动/实际KV记录检查资源重叠，本条不代判其结果。下一次需协调独占窗口 |

| A-call-progress-review / 09-13 | 8份既有holdout raw：native/rotate全调用1348/1257、pure含末步1242/1119、混合重算8/40；其实际新输出233/1185，每格总32768闭合 | `O/20260913_pressure_sweep_r01/call_progress_review_r01.json`；`E/analyze_call_progress.py`；根目录`A_LINE_RESEARCH_REVIEW_20260913.md` | 撤回“总缺席=额外步”及pure/maxL物理浪费；保留真实长停顿收益。新westc压力包`execution_review_westc_r02`已准备，启动检查发现其他会话cohort2强基线PID1770，占卡后退出93，0GPU初始化/0测量。待该整批结束再跑，勿抢逐格间隙。 |

| fullstage-lifecycle-qualification / 09-14 | F20+64与X21+48各4请求32输出完成；各224层调用状态核对、160按位参考及640权重检查通过 | `O/20260912_wisp_olmoe_r01/westc_return_20260913/qualification_check.json` | 同384槽/实际KV1GiB；资格含参考/JIT/额外显存，不作性能或质量结论。新host GPU70fa1c0a；四格性能UNRUN，原controller4540暂停，等待rotation整组交接后续跑 |

| fullstage-performance / 09-14 | 同机X/F/F/X全部完成；X/F两block capture−6.244%/−5.514%、五预热cycle−4.330%/−4.079%、新TTFT−6.845%/−6.357%；旧maxITL有5/16对变差 | `O/20260912_wisp_olmoe_r01/westc_return_20260913/attempt01`；32测量/96请求/1280输出、160预热、62592calls、76GPU边界；547文件回读 | [简表](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/westc_return_20260913/performance_summary.json)。384槽/KV1GiB；两block描述性联合实现效果，无独立噪声或持续服务GO；旧U/M权衡保留，F不是优化FreeToken。GPU已交A-review；下一CPU准备有限时钟到达U/M/F/X同资源比较。 |

## 2026-09-14 新增实测

| 登记 | 结论 | 输入 | 说明 |
|---|---|---|---|
| rotation-strong-baseline-cohort2 / 09-14 | 新32文档八格256请求全部完成：持续most_output对主native吞吐−1.362%/+3.241%、平均完成+5.962%/+1.031%、maxITL4.572/4.760→1.520/1.020秒；对headroom吞吐+3.604%/+5.303%、平均完成−3.730%/−5.003% | `O/20260913_rotation_strong_baseline_r01/execution02_westc_53036` | [结果](../../outputs/admission_capacity/20260913_rotation_strong_baseline_r01/RESULTS_WESTC_ADDENDUM.md)。单新cohort/两block、固定KV、原生in-process；完整性WARN/P0/P1=0，same-family/provisional。中断间隔保留，不作方法GO、非劣或噪声界 |
| rotation-identical-path-time / 09-14 | 两most的1162步执行选择、决策和输出相同；wall差0.971142秒，主要同一恢复末次调用0.766318/0.031701秒；单请求尾部两次均约0.052秒 | 同上两most；新增问题为同策略时间差来源，不重算旧封存数据 | [定位](../../outputs/admission_capacity/20260913_rotation_strong_baseline_r01/diagnostics/most_output_block_difference.md)。只定位引擎内调度外，JIT/温度/CPU或GPU原因未验证。完整慢调用保留；唯一后续同配置四项重复已准备、未上传/未运行 |

| rotation-candidate-feasibility / 09-14 | 两cohort2 most与一旧least，共26实际交换前态、506/506合法victim均可资助完整恢复；0次rank-first漏动作，选中者26/26实际free差额闭合，最小余量8块 | `O/20260914_rotation_candidate_feasibility_r01/analysis.json`，只读既有三格raw | [报告](../../outputs/admission_capacity/20260914_rotation_candidate_feasibility_r01/REPORT.md)。CPU_OBSERVED_PRE_ACTION_FEASIBILITY_ONLY；候选按已观察动作前态，不是独立实验/未来收益/Oracle；关闭本域资助排序漏动作解释，不外推APC/异构或其它压力点。 |

| A-pressure-review-westc / 09-14 | 16终态/512请求/524288输出：d0可行零抢占；d4 onset458、native pure1492；d6 onset202、pure1748，原2543预测误差−31.26%触发M2。d6最大ITL native14.20–14.40s→rotate2.805–2.816s，平均完成慢4.49–8.90%，吞吐随block变号 | `O/20260913_pressure_sweep_r01/execution_review_westc_r02/REPORT.md`及analysis.json/readback | NATIVE_SERVING/MEASUREMENT_ONLY；两格d0轮转原INVALID_NO_ACTION/INCOMPLETE保留，实际32请求完成。16文件源/输入/块数与全部调用输出检查通过；fresh同族审阅WARN，无主表P0/P1；证据仅原生vLLM进程内，高压强基线仍缺。GPU已交接，无自动追加。撤回d0不可行及缺席线性预测的高压适用；下一d6加入headroom/most_output强简单基线，尚未运行。 |

| rotation-runtime-repeat / 09-14 | 同cohort2四项128请求：most/native吞吐+4.087%/−1.640%、平均完成+0.258%/+6.310%；maxITL4.885/4.711→1.065/1.067秒；首forced恢复末call32.029/32.457ms，旧766ms本轮未见 | `O/20260914_rotation_runtime_repeat_r01/execution02_after_finite` | [结果](../../outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/RESULTS_AFTER_FINITE_ADDENDUM.md)。NATIVE_SERVING/MEASUREMENT_ONLY；两most1162步/输出相同但wall仍差0.848s，未定位JIT/GC/硬件原因；保留首初始化失败与全四项，不作净收益/非劣/新holdout/方法GO。 |

| rotation-runtime-repeat-cross-run / 09-14 | 四次most的1162步实际调度/决策/输出一致；首次恢复末调用旧0.766318/0.031701、新0.032029/0.032457秒；新两most完整wall仍差0.848332秒，最大单调用增量仅解释3.24% | `O/20260914_rotation_runtime_repeat_r01/diagnostics_r02/comparison.json`与`new_block_difference.json`；原八项两most及新四项两most | [解释](../../outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/diagnostics_r02/INTERPRETATION.md)。同cohort重复，非新holdout；完整慢调用保留，旧单次异常不能解释全部计时差，内部根因未验证；四项主结果由接手方主分析及runtime_integrity_review负责，不重复主审计。 |

| finite-arrival-execution / 09-14 | U/M/F/X/X/F/M/U 8格完成128请求/4096输出；X/F capture+3.394%/−0.948%、process+1.771%/+0.636%；D2D−52.061%/−53.862%，没有稳定X/F净收益 | `O/20260912_wisp_olmoe_r01/finite_arrival_r01/attempt01`，162文件/7856calls/6960测量calls/24边界 | [报告](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/finite_arrival_r01/REPORT.md)。U同臂capture漂移−20.900%，非总体噪声底；额外延迟定位layer0包络，JIT未逐次归因；首M启动与外部初始化重叠11.799s，测量无直接重叠，后效未知。保留全格，下一仅受控编译事件定位。 |

| native-APC-baseline / 09-14 | off/on/on/off四格128请求全完成；APC实际重算7685→6677，两次各复用1008位置；恢复step1030/1026不变；吞吐+0.498%/+0.040%、平均完成−0.576%/+0.067%、maxITL4.663/4.687→4.686/4.709秒 | `O/20260914_prefix_cache_baseline_r01/execution`及analysis | [结果](../../outputs/admission_capacity/20260914_prefix_cache_baseline_r01/RESULTS_ADDENDUM.md)。NATIVE_SERVING/MEASUREMENT_ONLY，cohort2/固定7671块/两相关repeat；缓存降低计算未解除本域恢复等待，非APC普遍NO-GO，无SLO/质量/非劣/方法GO；下一APC兼容同轮转四格仅CPU准备，GPU排A d6及B编译定位后。 |

| A-d6-strong-baselines / 09-14 | 8格256请求：least/native平均完成+7.47%/+9.32%，maxITL约14→2.80/2.94s；most/least吞吐+2.75%/+9.58%，平均完成+8.03%/+0.60%；headroom/least完成+50.33%/+40.91%、maxITL也更差 | `O/20260914_d6_strong_baselines_r01/REPORT.md`及analysis.json/readback | 原生in-process/MEASUREMENT_ONLY，同旧32文档、两反序block、APC off；无显著性/非劣/质量/方法GO。完成时刻分布表明整批更快不等于平均更快；全量回读、调用token守恒通过，定向fresh审计A-F PASS/P0/P1=0，同族provisional。GPU已释放B编译定位，无追加组。 |

| 普通U冷/保留Triton-cache两格：7个长layer0调用97.40–98.25%包络与实测编译/加载重合；保留cache仍有2次新编译，非稳定性能差 | jit_localization_r01 | 2026-09-14 | 32请求/1024输出；同源9/16输出一致、49/51step；MEASUREMENT_ONLY，旧尖峰逐次归因未测。outputs/admission_capacity/20260912_wisp_olmoe_r01/jit_localization_r01/REPORT.md |

| APC-compatible-rotation / 09-14 | 全APC-on native/most/most/native四格128请求完成；most最大ITL4.760/4.673→1.047/1.042秒，吞吐+1.692%/−0.334%、平均完成+2.586%/+4.791%；31/32及32/32完成更慢 | `O/20260914_apc_rotation_r01/execution`及analysis | [结果](../../outputs/admission_capacity/20260914_apc_rotation_r01/RESULTS_ADDENDUM.md)。NATIVE_SERVING/MEASUREMENT_ONLY；同cohort2/7671块/两相关repeat；真实9次forced、2次natural、held=0；重复6677→36272，1029提案/18资格汇总每格闭合，输出本组相同不等于质量。无净收益/非劣/Oracle/方法GO。下一协同A的同前态victim三分支，不扫cooldown或重复controller。 |

| A-neighbor-service-components / 09-14 | 新问题为近邻服务计数/保护与实际恢复的相容性：4旧轨迹6055步/57次恢复均4–5次目标调度即有新输出，均少于完整10次量子（非实际LTR剩余量子）；d6首次抢占共同前态全32请求未达512输出门槛，F=0、最少仍需13块才能有一个达门槛 | 独立worktree `agent/a-recovery-components-20260914`；[报告](/private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_components_r01/REPORT.md)，只读压力block0 d2/d6 native/rotate | OBSERVED_DIAGNOSTIC + CPU组件/原生adapter接口；9/9 CPU测试，完整native/GPU UNRUN。条件性guard-only不可达，非UniBoost死锁或反事实收益；LTR200/10+FCFS+共享容量适配off/on/on/off四格包已准备，未上传/无GPU队位。TokenFlow官方源码已核实，不重复funded-resume补丁。 |

| A-d6-action-prefix / 09-14 | 四份轮转raw在step329前均8909输出token、实际执行序列及当时request/KV计数一致；早期waiting遥测不同，保留差异 | `O/20260914_d6_strong_baselines_r01/BRANCH_PREPARATION.md`及branch_prefix_check_r02.json | 新增问题为共同动作前态重建；CPU只读检查，不证明KV张量一致。已按实际FLASH_ATTN布局实现KV指纹并通过7类CPU检查，CUDA未初始化；native集成/候选分支UNRUN，未上传或新增GPU组。 |

| A-d6-native-KV-state / 09-14 | 两次真实重建step329前记录状态一致；16层/512分段（496非空）、每次13,619,429,376有效KV字节，SHA差异0；64请求完成 | `O/20260914_d6_action_state_r01/REPORT.md`及analysis/readback | 原生in-process状态资格/MEASUREMENT_ONLY；诊断耗时12.326/11.754s，不作性能对照，不证明完整checkpoint或候选排序。源/字节数/请求核对通过，仅执行者定向检查。GPU已释放，无新增分支。 |

| compile-domain-r01 / 09-14 | 首F初始化后snapshot vars(slotted state)报错；42.952s、0/320覆盖调用、无请求测量，后三格UNRUN | `O/20260912_wisp_olmoe_r01/compile_domain_r01/attempt01`，62成员全回读 | [报告](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/compile_domain_r01/REPORT.md)。INVALID_EXPERIMENT/接口实现失败，非F/X NO-GO；修正六stats显式读取与真实slots CPU替身，新r02同协议排LTR后，旧raw不变。 |

| A-d6-single-action-branches / 09-14 | 六格192请求，所有前态/KV匹配；4/4/5恢复调用预测两block命中。首次most/least平均剩余+0.04%/−1.84%，defer/least−0.51%/−0.64%但最后完成更晚；least/most总1581、defer1583调用 | `O/20260914_d6_action_branches_r01/REPORT.md`及analysis/readback | 条件未来MEASUREMENT_ONLY，不是未扰动性能/显著性/方法GO；同一旧cohort的两反序block。首次most真实改变1240步后续调度，非零动作，但完整优势未确认。下一仅CPU资源进展模型校准，不追加GPU矩阵。 |

| A-resource-progress-model / 09-14 | 26条既有轨迹调度/块计数/完成步匹配；native两轨迹拟合后，12条非拟合轨迹条件mean/last最大误差2.10%/2.60%；31个首次victim无预测mean收益 | `O/20260914_recovery_progress_model_r01/REPORT.md`及validation/cost/candidate JSON | STRUCTURAL+回溯预测/MEASUREMENT_ONLY；同旧固定长度cohort、APC off，无新GPU/在线GO/Oracle/尾部误差保证；仅执行者定向检查。下一CPU后续决策状态单动作空间检查，不追加首次victim GPU扫描。 |

| compile-domain-r02 / 09-14 | F/X/X/F 64请求2048输出；1280准备调用通过、测量零Triton compiler；X/F capture−3.873%/+0.086%、平均完成−5.705%/−0.873%、D2D−54.040%/−54.623%，稳定净收益未成立 | `O/20260912_wisp_olmoe_r01/compile_domain_r02/attempt01`，134成员回读/4032全调用 | [报告](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/compile_domain_r02/REPORT.md)。MEASUREMENT_ONLY，同新16文档两模式各2引擎；X同臂capture+2.999%非噪声界，全启动/准备成本保留，r01接口失败独立保留；GPU释放，唯一后续CPU定位互斥成本与arrival分叉。 |

| A-later-single-event-surface / 09-14 | 33事件802个victim替换最佳预测mean−0.074%；33个取消动作最佳mean−0.373%但gap+11.22%；均无新GPU | `O/20260914_recovery_later_victim_r01/REPORT.md`及surface/skip_surface | MODEL_ONLY，同旧状态、固定长度；前缀/动作/守恒检查通过，非在线收益或全局Oracle。单事件预测收益不足，下一CPU持续轮转停止与完成释放诊断。 |

| LTR-component-native / 09-14 | 同custom后端boost off/on/on/off四格128请求完成；吞吐−2.188%/+3.215%、平均完成+2.838%/−3.273%、maxITL4.692/4.905→4.928/4.868s；每个on一个10-call量子实际返回7新token | `O/20260914_ltr_component_probe_r01/execution`、analysis及audit | [结果](../../outputs/admission_capacity/20260914_ltr_component_probe_r01/RESULTS_ADDENDUM.md)。NATIVE_SERVING/MEASUREMENT_ONLY，d6/6656块/APCoff/旧cohort，29/32跨策略输出相同、质量未测；全部日志含episode内JIT，不作稳态性能。fresh审计PASS/P0/P1=0、same-family provisional。后续gap定位显示3571部分恢复被清零idle后再抢占，但原LTR rank-prefix与custom fit-scan有接入差异；下一先补prefix组件对照，不包装论文缺陷或扫描量子。 |

| A-stop-rotation-and-cost / 09-14 | d2/d4/d6共62停止点均无预测mean改善且gap不增；12旧raw完成面积闭合，d6含重算区间+3.74/+3.65s、其余−2.12/−2.51s，净+1.62/+1.14s | `O/20260914_recovery_stop_r01/REPORT.md`及surface/completion_cost/observed_completion_area | MODEL_ONLY候选+真实会计，含重算区间不是纯重算成本；新GPU0，无全局Oracle/方法GO。下一恢复重算与KV往返物理成本资格，不继续victim小差异扫描。 |

| compile-domain-r02-host-localization / 09-14 | X同臂+243.390ms=ensure−74.082+apply其余154.433+engine其余162.160+engine外0.879；前3步路由/plan等可观察状态同而先慢44.793ms，随后跨0.5s到达边界、64→96rows | `O/20260912_wisp_olmoe_r01/compile_domain_r02/OBSERVED_COST_LOCALIZATION.json`，只读同两X；新增问题为残留时间源 | [解释](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/compile_domain_r02/OBSERVED_COST_LOCALIZATION.md)。观察定位，未核hidden/KV字节、GC/线程等待/GPUkernel未测；CUDA span不混入host账本。主四格独立有限审计PASS/P0/P1=0、同族provisional；唯一下一host_cost开关四格仍CPU准备/UNRUN。 |

| A-LTR-service-lifecycle / 09-14 | 复用上行同四格：每off有13段、每on有11段恢复后只获1–2新token又被抢占；首输出前再抢占各1/2段。每对step609前32请求/17295输出前缀相同，全部TTFT已发生，TTFT变化不归因提权改变调度 | [生命周期及前缀分析](/private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_components_r01/REPORT.md) | 非新增GPU运行。scope为本custom backend；step650先跳过较早不可行者再恢复尾请求来自fit-scan接入，原LTR prefix-stop不同。新packing组件11CPU项通过；首次旧共同前缀分叉406，可能暂停已开始重算，不预报收益。已冻结1fea4a61…734ef73四格并交单一长任务执行，当前UNRUN；不重复实现恢复义务或宣称已有论文缺陷。 |

| A-KV-roundtrip-sizing / 09-14 | 两d6 least各39抢占，单向有效KV17.99GB、旧生命周期host峰值3.45GB；token模型对称带宽等成本22.50GB/s | `O/20260914_kv_roundtrip_feasibility_r01/REPORT.md`及budget.json | CPU资格，非D2H/native swap实测；连续H2D旧数据不替代双向KV布局。探针仅语法通过，UNRUN/上传0，下一排prefix组后现场检查与独立copy测量。 |

| A-KV-roundtrip-GPU / 09-14 | 205/207/256块gather+D2H+H2D+scatter往返中位16.50/16.74/20.54ms，24/24含warmup内容正确；已释放 | `O/20260914_kv_roundtrip_feasibility_r01/GPU_ADDENDUM.md`及gpu_result.json | ISOLATED_GPU_RUNTIME/MEASUREMENT_ONLY，无请求收益；安装vLLM有native offload，现rotation拒绝connector。下一复用纯native offload基线，先测实际加载/命中/成本，不自写pager。 |

| LTR-JIT-log-correction / 09-14 | 更正旧LTR-component行的episode内JIT断言：默认warning_once覆盖应用预热且可能为磁盘cache加载，不能定位measurement内编译或解释wall波动 | 同组原raw/audit保持不变 | [勘误](../../outputs/admission_capacity/20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md)。请求指标和MEASUREMENT_ONLY不变，纠正audit相应语义。 |
| LTR-packing-native / 09-14 | fit/prefix/prefix/fit四格128请求全完成；prefix吞吐−1.865%/+1.900%、均完成+3.049%/−1.668%、maxITL4.835/4.933→6.732/6.577秒；恢复后0新输出再抢占每次2→6 | `O/20260914_ltr_packing_r01/execution`、analysis | [结果](../../outputs/admission_capacity/20260914_ltr_packing_r01/RESULTS_ADDENDUM.md)。同d6/6656块/200/10/APCoff，旧cohort、组件对照非完整LTR；25/32跨臂输出相同，质量未测，实际计划/资源检查通过，未声称完整planner离线重放。主问题OPEN；下一首新输出恢复保护，同底座最小消融。 |

| A-LTR-prefix-native / 09-14 | 同200/10的fit/prefix/prefix/fit四格128请求：prefix最大ITL6.732/6.577对4.835/4.933s（+39.252%/+33.317%）；吞吐−1.865%/+1.900%、平均完成+3.049%/−1.668%；1–2输出短段11→0，零新输出重抢占2→6 | `O/20260914_ltr_packing_r01/execution`；[独立主分析及首次分叉](/private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_components_r01/REPORT.md) | NATIVE_SERVING/MEASUREMENT_ONLY；首次动作分叉406获两block实际metadata/输出前缀支持，非KV张量相同证明；更忠实prefix未消除部分恢复中断，不能凭短服务段减少称GO。当前组件移植非完整LTR；恢复义务只复用长任务单一草案，不重复Controller，不以旧most跨组结果宣布超越。 |

| host-cost-observer / 09-14 | X off/on/on/off64请求2048输出；on3663 host spans；两on engine8156.657/8486.156ms、thread8154.735/8484.765ms，GC重合46.426/50.228ms不足以解释差值；on/off capture+0.608%/+4.078% | `O/20260912_wisp_olmoe_r01/host_cost_r01/attempt01`，140成员全回读 | [报告](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/host_cost_r01/REPORT.md)。MEASUREMENT_ONLY；4格编译覆盖/观测恢复通过，CPU时间不等于Python瓶颈、GC重合不作可扣收益；off路由/输出全同而on轨迹不同，非噪声界或旧r02根因归因。GPU释放，仅CPU前缀关联。 |

| A-native-prompt-offload / 09-14 | 4格128请求：store12GiB/load2.25GiB，重算21426→2994，两on平均完成+23.83%/+6.73%、wall+23.55%/+5.17%，净收益未成立 | `O/20260914_native_offload_baseline_r01/REPORT.md`及raw/analysis/timing | 原生默认prompt-only/MEASUREMENT_ONLY；第一on尖峰保留，非full-decode/rotation兼容/方法GO；host实际峰值未测。GPU已释放，下一后端调度/执行税定位，不扩大范围抢救。 |

| host-cost-prefix / 09-14 | 两on step1/call137/layer9约20.9ms planner各与gen2 GC重合约14.1ms；前3步engine差32.170ms但GC差仅0.091ms；step10前可观察轨迹同，step11跨2s到达点分为70/102行 | `O/20260912_wisp_olmoe_r01/host_cost_r01/PREFIX_LOCALIZATION.json`，复用同四格，无新增GPU | [定位](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/host_cost_r01/PREFIX_LOCALIZATION.md)。SOURCE_LOCATION_CEILING，非旧r02因果根因/可扣GC成本；限定审计PASS/P0/P1=0，同族provisional。本轮分析闭合，停止追加同类溯源。 |

| A-offload-prefix-cost / 09-14 | 四格32/32输出相同、同臂全schedule signature相同；两组on/off首分叉1026，前缀engine额外3.258/1.594s，scheduler1.331/0.518s | `O/20260914_native_offload_baseline_r01/LOCALIZATION_ADDENDUM.md`与prefix_timing | OBSERVED_LOCALIZATION，非物理布局等价/噪声界；第一on全量保留。下一同on有限clock观测对照，CPU已准备、GPU0。 |
| restore-first-output / 09-14 | off/on/on/off128请求完成；每on27恢复义务全new_output/0中断，maxITL6.669/6.887→4.387/4.141s；吞吐−7.104%/+2.309%、均完成+8.300%/−1.426%，重算74982→96955；各对27/32最大ITL更差 | `O/20260914_restore_completion_r01/execution`及analysis | [结果](../../outputs/admission_capacity/20260914_restore_completion_r01/RESULTS_ADDENDUM.md)。旧d6/6656块/rank-prefix200/10，7556步完整plan/义务重放；同底座因果消融非强基线胜出/新颖性，26/32跨臂输出同、质量未测。完成性已成立，服务摊销/损害转移待定位，主问题OPEN；下一仅CPU区分KV保留与执行优先级。 |

| A-offload-finite-cost / 09-14 | 同native-on profile off/on/on/off128请求，1866步/输出/字节一致；每on17569 span逐步守恒，connector1.200/1.112s、engine剩余30.435/33.205s；profile配对wall+2.217%/+4.968% | `O/20260914_native_offload_cost_r01/REPORT.md`及analysis/raw | MEASUREMENT_ONLY，运行间波动不可扣除；主线程CPU接近wall不等于Python瓶颈，未知engine剩余不可称GPU税。GPU释放，下一限定32共同decode调用CPU/CUDA活动定位。 |

| A-restore-resource-vs-work / 09-14 | 同restore真实首次分叉406/F148状态：fit可同时给恢复994位置及30个decode，完整history预留余3块；guard给恢复1024、0decode、余6块；均无victim | [单状态模型](/private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_components_r01/restore_single_state_actions_r01.json) | 两block原off/on动作均复现；这是CPU当前可行性，非独立未来/加速结论。资源保留不必在该状态独占计算，后续复用长任务唯一guard_residual六格，不重复Controller。 |

| A-offload-decode-window / 09-14 | 默认offload旧四格500..531签名完全匹配，32调用无prefill/重算，宽29/30；有限CPU/CUDA窗口采集器已实现并通过正常/异常关闭恢复检查 | `O/20260914_offload_decode_trace_r01/DECISIONS.md`及expected_window.json | CPU准备/UNRUN，上传0/GPU0，非CUPTI采集成功或性能结论；下一按前序GPU组协调两臂源定位。 |

### 2026-09-14 默认offload共同decode活动定位

`20260914_offload_decode_trace_r01` 两臂COMPLETE；相同32调用/32完整输出一致，各10274 GPU事件可匹配API correlation。窗口648.874→868.054ms，GPU活动并集511.701→511.551ms，额外时间主要在记录GPU活动外；不能归因特定函数或宣称可移除收益。单次off/on且带profiler，顺序/主机漂移未隔离；唯一下一步同窗口反序复测并记录CPU环境。证据上限NATIVE_PROFILER_DIAGNOSTIC，非方法GO，原件及分析见该目录REPORT.md。

| fresh-cohort-FX / 09-14 | 两新cohort/8engine/128请求：capture四配对−3.488/+0.486/+1.085/−0.830%，cohort均值−1.495/+0.129%；D2D−54.068/−54.741%，四配对最大ITL均增加 | `O/20260912_wisp_olmoe_r01/fresh_cohort_r01/attempt01`，222成员回读/冻结分析同 | [结果](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/fresh_cohort_r01/REPORT.md)。原X稳定净收益未成立；7136测量/8032全calls、24GPU边界、0测量compiler/load，fresh同族审计PASS/P0/P1=0/provisional。GPU已释放；下一只准备logical64后映射的数值资格，不重跑相同机制寻找正效。 |

### 2026-09-14 默认offload反序诊断完成

`20260914_offload_decode_trace_reverse_r01` 两格COMPLETE；32调用调度/32完整输出一致，GPU事件10274/臂且全部API correlation匹配。off/on窗口640.596/844.566ms，GPU并集511.943/511.607ms，主机侧差异反序复现。CPU阶段占比不稳，on第527调用同步后66.556ms尖峰；包围窗口cgroup节流增量0，不能排除频率/主机漂移。下一只定位Python调用/GC事件，不猜测性删逻辑；证据上限NATIVE_PROFILER_DIAGNOSTIC。

| restore-token-reservation / 09-14 | 六格192请求/11252步完成：residual/guard吞吐+4.351%/+1.926%，对fit−0.161%/−2.653%；fit maxITL4.920/4.823→residual4.339/4.356s，两对32/32完成更慢；每residual25恢复全兑现、54实际预算改变 | `O/20260914_restore_token_reservation_r01/execution`及analysis | [结果](../../outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md)。旧d6/APCoff/6656块；同组强简单fit仍有服务量优势，首406预算/返回兑现，剩余最大gap97.42%在恢复前。固定旧成本模型只解释实际路径，非Oracle；所有repeat/output差异保留、质量未测。主问题OPEN；下一新cohort3强基线八格CPU准备，GPU UNRUN。 |

| service-window-lifecycle / 09-14 | 复用14格：fit每格11短段/14新token/37851重算；首输出guard为4/4/13767；最新residual为5/5/17505，短段恢复prefix均在下一residency实际重执行。真实409全31共批增量需3/5/6块而F0；固定FCFS子集28/26/25可用已有块继续目标 | `O/20260914_service_window_r01/LATEST.md`、`O/20260914_service_window_model_r01` | 新增只读分账与185行CPU资格模型；10测试通过，前8格独立分账PASS/provisional，新6格不追认旧审计。部分恢复复用与最终丢失分开，输出时间仅engine-return。模型真实排序/host峰值/未知EOS仍未测。GPU新增0，不另造窗口Controller；唯一下一步复用主会话新cohort3 native/most/fit/residual八格。 |

### 2026-09-14 offload主机profile发现观测器成本

`20260914_offload_python_cost_r01` COMPLETE，32调用签名/完整输出匹配；一次gen2 GC68.947ms/collected0。memory_telemetry.state64次累计107.4ms，request_state2050次/get_blocks2050次；嵌套含GC、不可相加或直接归为offload增量。下一只做等价低分配块数观察器消融并无profiler复测，不禁用GC。NATIVE_HOST_PROFILE_DIAGNOSTIC，非方法净收益。
| logical-alignment-qualification / 09-14 | 5请求40输出；176真实层调用+96前缀均逐位同，固定错map finite但不allclose（maxabs0.233704）；384槽/1GiB不变 | `O/20260912_wisp_olmoe_r01/logical_alignment_qualification_r01/attempt01` | [资格结果](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_qualification_r01/REPORT.md)。49成员回读一致；fresh同族限定审计PASS/P0/P1=0/provisional。仅数值资格，无性能/质量GO。原X稳定净收益未成立的裁决保留；下一F/X/logical-X同资源六格尚未运行。 |

| A-reservation-KV-boundary / 09-14 | 新六格两residual的step521：free27=恢复history25+两个peer增长各1；下一个3345还需1且原排名后无resident可驱逐，计算位置仍余598 | [预算增补](/private/tmp/moe-a-recovery-components-20260914/refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_components_r01/TOKEN_RESERVATION_ADDENDUM.md)及token_reservation_commitment_r01.json，复用原六格，无新增GPU | 当前KV/victim资格阻塞，非计算预留不足；不同victim/执行子集未被排除。两个block同一结构事件，不是独立请求样本，不外推速度/新颖性。 |

| service-window-holdout-frontier / 09-14 | cohort3八格256完成：native/most每格短恢复段0，fit11、residual5；most wall25.327/26.579s与maxITL2.814/2.942s均优于residual28.117/28.450、4.291/4.369；全经验Q(g)中residual超越native/most完整策略较优边界0/42、0/42区间 | [最新结果](../../outputs/admission_capacity/20260914_service_window_holdout_r01/REPORT.md)，raw复用原统一八格，GPU新增0 | MEASUREMENT_ONLY/BASELINE_COVERS_TESTED_RESIDUAL；停止当前组件窗口扩展，非family NO-GO/联合SLO/免费在线切换。多数请求完成/间隔仍存在交换，quality未测。模型405前态算出恢复后all-peer缺3/5/6块，fit子集可行但没有方法收益；host只有共享父90GiB快照、无独立hard cap。下一科学问题是native/most在持续/异构/真实EOS域是否仍覆盖边界，不重跑固定长度窗口阈值。 |

### 2026-09-14 KV观察器八格完成，微优化停止

`20260914_kv_observer_cost_r01` 256请求完成，direct平均完成对original：off−2.81/+2.94%，on+5.06/−2.77%，均翻转，无稳定收益。逐步schedule/输出/传输量一致；规范化ID后memory差异仅前69步的到达集合，共享请求状态0差异，严格全状态等价不通过。此实现停止，不归为KV问题NO-GO；下一CPU核对native connector保存/驱逐/恢复契约，禁止直接移除不兼容保护。

### 2026-09-14 Native保存/驱逐契约CPU定位

`20260914_native_store_contract_r01`：store完成用completed_jobs，finished_sending始终空；驱逐只flush已有任务，不自动为未保存victim造任务。封存worker原方法3CPU case通过，fake传输/无GPU。候选须保存登记→原生提交/等待→驱逐两阶段，下一只用真实首次前态核对提前边界及KV保持可行性，不删connector保护。

### 2026-09-14 选择性保存提前边界CPU资格

`20260914_native_store_contract_r01/selective_boundary.json`：两block step328 visible选择3571命中原329victim，31decode/free149→148，释放207块后目标205块静态可容纳；native store-builder原方法+fake host/key/block生成任务，decode允许时3296token保存/10已计算尾部未保存，默认prompt-only3072/234。CPU条件可行性，非GPU/真实hash/净收益；下一单victim真实KV保存恢复接口资格，保留原生保护和成本。
| logical-alignment-FXY / 09-14 | 6engine/96请求：Y/F capture−2.076/−1.704%，Y/X−3.197/−2.409%，双block最大ITL下降；Y/X全进程+1.684/−4.123%翻转 | `O/20260912_wisp_olmoe_r01/logical_alignment_performance_r01/attempt01` | [结果](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_performance_r01/REPORT.md)。1471成员/40输入回读与封存分析同；请求阶段正信号，非稳定净收益/方法GO。下一仅独立B文档113..128逆置臂位置验证，保持实现/预算。 |

### 2026-09-14 单次选择性保存真实接口观察

`20260914_selective_store_once_r01` 两臂64请求完成；328store/329flush/330load/332恢复首新token，保存3296token，实际store432013312/load864026624bytes（两load），victim gap170→104ms；mean+2.37%/wall+1.66%，n=1不支持净收益。25/32完整输出一致、victim一致，7请求动作后分叉；KV保真未测，下一定向保存前/加载后相同逻辑前缀指纹，非方法GO。

### 服务窗口自然长度/EOS执行准备（2026-09-14）

`O/20260914_streaming_recovery_r01/LATEST.md`：CPU准备与暂存完成、GPU_UNRUN。64篇新完整自然文章334–3011 tokens、0.5s持续到达、真实EOS未知/max_output1024；单档4096 usable KV块，native/most/most/native。原6656块档任意32请求声明KV上界6551，保留结构低压力边界。复用已测capture/absence/native后端，只扩open/EOS接口；46相关定向/兼容测试通过。parent90GiB是共享容器硬上限，树RSS仅OBSERVED_ONLY；hostKV offload0。

不重跑已完成LTR/cohort3/offload实验；本组用于查明强基线下的恢复服务量，不承载低插桩性能主张。没有动作/没有实际EOS均原样保留，不按结果选文档/调压/改到达。包SHA1d0e5f05…2cce6，排context-victim整组终态之后，GPU单一入口/共同flock，无后台候卡或自动retry。实际执行若发生，将以新RESULTS.md和终态receipt追加；本条不代表测量结果。

### 2026-09-14 单次保存动作前漂移分解

`20260914_selective_store_once_r01/TIME_PARTITION_ADDENDUM.md`：同到达/同前328步调度，save-on在动作调用开始前已慢0.927040s，完整均完成慢0.610306s，动作后平均剩余时间差−0.316734s；恒等分解非因果校正，n=1既不能归因整体退化，也不能宣称扣除漂移后的收益。原+2.37%保留；下一仍已封存KV保真单格。

| logical-alignment-independent-cohort / 09-14 | 第二组96请求3072输出全完成；Y/F capture −2.663/−2.928%，但Y/X +1.768/−0.0648%、全进程 +4.492/+3.542%；旧X本组已强于F | `O/20260912_wisp_olmoe_r01/logical_alignment_validation_r01/attempt01` | [报告](../../outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_validation_r01/REPORT.md)。1473成员/40输入回读与封存分析一致，全部原件保留。停止稳定Y增量主张，不追加cohort/参数扫描；同臂输出F7/16、X10/16、Y9/16相同，非总体噪声底。真实超显存模型资格另行只读核对。 |

| qwen3-new-gpu-r02-reconciliation / 09-14 | westc旧r02进程全退出；仅4完整分片/4845源张量，第5片残2,515,533,824B；0数值0性能，退出原因UNKNOWN | `O/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/attempt02/readback_20260914_r01` | [核对](../../outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/RECONCILIATION_20260914.md)。55成员归档前后稳定/回读哈希同；原RUNNING/INITIALIZING不改，非科学NO-GO。区别于returned weste r02；新r03复用冻结条件，仅CPU准备。 |

### 2026-09-14 首次恢复的保存前缀真实保真

`20260914_selective_kv_fidelity_r01` 单格32请求/32768输出完成；328保存前与331首load完成后的3296token×16层SHA一致，432013312bytes、206逻辑块，物理映射不同。331 computed3296/output235，332才执行11位置恢复；一次store/两次load，本次只检查首load。哈希新增0.419/0.373s全属诊断，非性能证据。排除该前缀首次搬运损坏，不覆盖其他请求/第二load/质量；净收益仍未验证。下一同非指纹底座完整交错重复，不作漂移扣除。

### 服务窗口：异构context六格只读生命周期接续

`O/20260914_service_window_context_r01/REPORT.md`：原组件context六格完整回读后，本方复用既有helper核验192请求/196608输出，native/most/least全重复零输出丢弃和1–2输出丢弃均0。most每轮8再丢弃段最短11输出，least18段最短5；不改短段阈值复活窗口。most比native吞吐+2.46/+5.89%但均完成+8.49/+5.13%；least比most最大gap−8.52/−5.52%、吞吐−4.30/−7.83%，均完成方向翻转。least高于native/most Q(g)包络的两轮狭小区间不相交；无窗口GO/Oracle，全部简单策略取舍保留。源为重复旧文档的2560/3072异构、固定输出1024、6656KV，host峰值未测，非未知EOS。未修改共享raw/未执行GPU；下一仍已冻结streaming四格，正确认证未恢复。


### 原组件context-victim六格完成：异构长度暴露选择后不可资助

`O/20260914_context_victim_calibration_r01/REPORT.md`：六格192请求/196608输出全部完成，P2560/P3072复用cohort3、O1024、6656实际块，原件已统一回读并持久化，SHA7e83a448…1e9f5d。most/native吞吐+2.461/+5.888%、maxITL减7.291/7.653s，但均完成+8.494/+5.129%；least/most吞吐−4.305/−7.834%、maxITL减0.179/0.117s、均完成方向翻转。每least格同target891–908连续18次选中无法资助的victim，同时有合格可资助候选；891 F31+B200<need234，B211候选可行。most零此类拒绝。属于当前least适配器rank-then-check缺口，非论文算法缺陷/新方法收益。默认关闭的先过滤后排序修复4671调用精确重放、36拒绝before-state选出可资助候选；GPU修复收益UNRUN。fresh同族provisional复核A–D PASS/E–F WARN、无P0/P1。下一仅most/least/least_feasible反序六格，仍保留最强most；本条不新增GPU执行。

### 2026-09-14 单次保存交错重复与混合服务会计

`20260914_selective_store_repeat_r01` 四格128请求完成，均完成on/off +1.713%/−18.779%翻转；block1动作前已快1.671s，同臂全调度/输出相同但off wall27.142→33.637s，净效应UNRESOLVED。目标gap两对−33.60/−50.22%，重算少6591tokens；含重算调用26→21却纯decode1745→1748，135新输出转移，总调用只少2。混合恢复仍服务其他请求，重算量不能直接当可删除串行税。按冻结规则停止追加同域重复，下一CPU把混合正常decode服务纳入动作状态/成本模型。GPU已释放；fresh审阅因模型capacity未执行，非PASS。

### 2026-09-14 保存前缀状态模型的ready边界

`20260914_saved_prefix_model_r01`：原模型已含重算混合decode进度；补原生指定抢占后，无保存两重复329–1868逐步完全匹配。补加载先占块/留waiting/ready后恢复后，保存两重复329–1040匹配，1041均失败：首次load等待2步，第二次实际1步，固定2步近似不成立。末步仍同1866不能覆盖轨迹失败。原least/most/defer预测回归相同。下一CPU定位native提交/完成查询/ready可见时序，不喂未来ready标签，不追加GPU。


原组件后续交接：独立分支 `agent/a-recovery-components-20260914` 本地commit `87460cd3b187323b8fa682c7f989889e19646008` 保存上述实测、分析和默认关闭修复，未push。`O/20260914_funding_filter_comparison_r01/REPORT.md` 已持久化，六格27a392a8…858bc包远端20文件校验完成，27 selector测试/native fixtures/旧4格默认replay通过；主分析6 UNRUN/0 comparisons。排已启动Qwen r03整组终态/明确释放之后，无修复GPU数据/无后台候卡。原件全部保留，context大raw/archive在durable本地+原remote，Git保留其hash及其余分析/决策/输入。


| streaming-recovery / 09-14 | 原N/M/M/N四格256/256完成；实际4096usable块，90GiB共享父预算观察；自然抢占2/5/2/0、强制动作0，9恢复全完成、996–1024新输出/段，零再丢弃 | `O/20260914_streaming_recovery_r01/RESULTS.md`及analysis/resource_summary.json，原归档e4852467…96bd | MEASUREMENT_ONLY / NATIVE_INPROCESS_DIAGNOSTIC；每格6实际EOS末尾、58长度截断，59576返回ID含6EOS。未暴露强基线后短恢复缺口，不调压力/保护期。非客户端/质量/低插桩性能/独立host硬预算；下一复用已排funding-filter资格消融，不另建窗口GPU组。 |

### 2026-09-14 Native异步完成通知状态回放闭合

`20260914_load_ready_contract_r01`：worker执行结束查询finished_recving，scheduler接收后下一步才可提升WAITING_FOR_REMOTE_KVS；computed/占块不等于ready。真实worker事件按perf_counter定位首次331→332、第二次1040→1041；事件回放下两保存重复各1538步调度/空闲块/输出数全匹配。明确使用实际未来完成通知，仅状态机验证，非预测或反事实收益。默认旧三策略回归不变。下一CPU比较原缺席请求优先与资格实现刚抢占victim排队首的恢复排序，保持异步通知未知边界。

### 2026-09-14 单次恢复排序的贡献边界

`20260914_saved_recovery_order_r01`：实测旧资格优先恢复刚抢占3571（332/333），原缺席3640仍1045/1047恢复。CPU原缺席优先且不保存令首服务交换714步，调用1869→1836，但仅两请求完成+75/−76步，均完成步号只降1/32；同排序保存增量−2/−1/+1调用随假设load1/2/4步翻转。不是GPU/墙钟预测，不把排序收益归保存。停止单事件排序GPU扩展，下一盘点既有完整轮转的重复恢复成本空间，保留混合服务与反事实边界。

### 2026-09-14 完整轮转保存量与混合输出比较

`20260914_rotation_save_volume_r01` 复用旧8格；least39事件预算直接复用核验。most44抢占/163764重算tokens却总1315调用，优于least39/137272/1581，不能用重算量评价策略。least/most含重算调用生成4226/4739新输出；累计4.2–4.9s不能全当可删税。按完整块每次全存least17.954GB、most21.418GB单向，最长逻辑前缀合计4.059/12.053GiB仅条件容量盘点。多次保存未被成本空间排除，也未有净收益；下一仅most两阶段native保存/抢占合同与真实前态合法性，保留最强同底座基线，不删旧保护。


### 服务窗口ready状态接口修正

`O/20260914_service_window_ready_r01/REPORT.md`复用保存原件：331/332前target3571同3296computed/206块/235输出，worker通知只在331调度后出现，332才执行11位置（10重算+1decode）。模型新增原生load阻塞状态，pending持块却不计可执行前缀；ready后C_remaining仍未知，不给摊销资格。8定向检查通过，无在线接入/无新GPU/无收益claim。共享ControlPath现场检查已不存在，Qwen实际存活/释放未验证，已暂存funding-filter六格不启动副本。


### 2026-09-14 两阶段保存/抢占 CPU 合同资格

`20260914_staged_save_contract_r01`：复用329冻结前态与native329/330计数，most victim多执行一步后仍具备容量资格；保存3392 tokens/212完整块，10类状态/异步预留检查通过。下一步物理块ID与store成功例为明确CPU fixture，非真实store或性能验证；READY仍依赖native同步屏障才可复用块。GPU_UNRUN，SSH认证前关闭，未启动任务。下一仅一个真实两阶段事件资格，保留most及同相位save-off基线。


### 2026-09-14 单事件原生两阶段适配器实现

`staged_store_once.py`新增独立接口：实际store源块/登记核验、次步状态复核、native抢占flush元数据检查、原缺席请求优先及异步容量保护。导入与封存Scheduler四处AST编译通过，未执行mock-engine/GPU，非性能结果；旧rotation connector拒绝未改。见`20260914_staged_save_contract_r01/ADAPTER_ADDENDUM.md`。下一完成实际schedule接口fixture后封单事件资格包，遵守Qwen/funding-filter顺序。


### 2026-09-14 原生等待路径资格与单事件包

`20260914_staged_store_probe_r01`：执行封存原生queue选择/promote及插桩等待循环前段，completion不可见时不调度、可见后仅target入选，两条件通过；allocation/完整schedule/worker未覆盖。单事件most两阶段off/on本地包已准备，共同16GiB host缓存、旧输入预热，整组flock/installed源码SHA/逐格GPU检查。包文件与语法、CPU合同通过；未上传/未启动，GPU_UNRUN，遵守Qwen及funding-filter原顺序。


原组件异构模型接续：`20260914_context_victim_calibration_r01/model_check_r02.json`，从step90对四个most/least实测格逐步匹配4978调度步；固定6000 horizon，无未来事件驱动状态。条件least_feasible预测target0020902首输出916/915→894，重算83510/83517→89761/86779，末步1444→1451/1447。是CPU结构预测，不是时延排名或GPU结果；保留不利全局成本，原六格不改。r01 observed-end horizon已单独更正，数值完全相同，原件保留。SSH认证前断开，Qwen终态UNKNOWN，资金过滤组仍STAGED/GPU_UNRUN。


| qwen3-r03-interrupted / 09-14 22:32 | 2完整分片/2325张量，第三残1,934,622,720B，0数值/0性能；当前实例GPU已更换、旧三进程不在当前实例 | `O/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/attempt03/START_ADDENDUM.md` | 63成员归档4,368,384B/SHA844fe211…8e91b，61载荷/38输入核验；旧RUNNING/INITIALIZING不改，退出原因UNKNOWN。加载中断不作科学NO-GO；新attempt须重验新硬件/空间并尊重短组队列。 |


### 原cohort3八格主结果补记闭合

`20260914_recovery_holdout_comparison_r01/RESULTS_ADDENDUM.md`：原八格256请求/262144输出/13808实际步，most/native吞吐+8.499%/+1.829%、全局maxITL14.398/14.290→2.814/2.942s，均完成+9.605%/+17.415%，27/28请求更晚完成。most在两主轴均覆盖fit/residual，不能扩写所有指标占优；residual/fit吞吐+1.712%/−2.502%翻转。全部跨臂损失/输出差异/计时边界和八点图保留，原raw/合同不改。固定旧native成本模型第一对fit→residual条件均完成/末完成方向均错，实际future schedule诊断非Oracle，不重拟合。MEASUREMENT_ONLY/OPEN；一次fresh限定审计进行中，不追认PASS。下一复用原funding六格，新GPU独立组内比较，原执行方统一回读分析。

| funding-filter-six-cells / 09-14 | 新bd5e5090六格COMPLETE/192请求/196608输出；过滤将每block同一episode的18拒绝降0，首新输出916→894，但重算83510→89761、结束step1444→1451。相对least吞吐−3.011%/+1.272%变号；相对most最长gap更低、吞吐−5.706%/−5.096%，无稳定净收益 | `O/20260914_funding_filter_comparison_r01/analysis_r02.json`、`model_check_r01.json` | 原包27a392a8…858bc、readback7ad278ed…8329原件完整保留；六格内部同资源，旧CTX卡不作时间pair。冻结结构模型7702steps全部匹配，仅结构非时间/Oracle；maxpause从0020902转0020794。当前MEASUREMENT_ONLY，fresh targeted review进行中；后续仅单动作victim分叉诊断，不重复调阈值。 |


### 2026-09-14 两阶段保存/原缺席优先真实单事件完成

`20260914_staged_store_probe_r01` 新GPU两臂64/64完成。target3640均333步首新输出；victim0000001 off1051/on1049步。实际保存/加载各444596224B，少3392重算tokens；重算调用26→24、纯decode1670→1672，总调用仍1794。on均完成+12.558%，但动作前已有+1.140924s漂移，不能归因稳定损害/收益。SINGLE_EVENT_EXECUTION_QUALIFIED / MEASUREMENT_ONLY，非位级保真/重复most策略GO。原件归档bfad2152…0c75保留。下一已有模型解释重复保存的步数/服务增量，不追单事件正时间。


### 服务窗口：资金过滤六格生命周期与完整阈值分析已完成

原六格统一回读7ad278ed…8329/192请求，新GPU bd5e…0bdc内部比较。新增问题是当前历史可资助之后是否兑现有效服务；复用现有生命周期/全阈值分析器，无新GPU。filtered每轮各1段3732重算→2输出→再抢占，以及1段2991重算→0输出→再抢占；most/least各0此类短段。前者首输出保护已经解除后被native淘汰；后者native自动resume未纳入adapter保护，peer增长后缺1块。内部前缀先复用后实际失效，未把混合恢复调用当纯税。相对least吞吐−3.011%/+1.272%符号反转；相对most吞吐−5.706%/−5.096%，换最大gap−72.886/−162.330ms。Q(g)超过同轮most/least完整策略包络4/33、20/33非零区间，全部保留，不反选阈值、不称全域支配或方法GO。报告与精确状态：`O/20260914_service_window_context_r01/funding_r01/REPORT.md`。主问题OPEN，最小窗口仍结构资格；复用A已完成native两阶段off/on原组分析，不重跑此六格或添加窗口矩阵。


### 2026-09-14 保存单事件：收益未到达完成尾部

`20260914_staged_store_probe_r01/STRUCTURAL_ADDENDUM.md`：两臂各1465步调度/空闲KV/输出计数闭合；on使用自身1048完成→1049ready真实通知，仅状态回放。最终token逐请求直接对齐证实仅victim0000001完成提前2步，其余31个完成步相同（含最后target3640）；平均完成步只降0.0625，总步不降。旧三策略模型回归完全一致。限定单事件结构收益不足，不扩为多事件NO-GO；下一才是独立演进的重复两阶段save-on/off与即时most对照模型。


### 2026-09-14 重复两阶段保存结构敏感性

`20260914_repeated_staged_model_r01/summary_corrected.json`：各候选独立状态，38强制动作；即时most1315调用、staged-off1316、假设load1/2/4/8步的save-on1308/1310/1312/1316，均完成步在2→4步变号。保存/host成功及无store墙钟税是假设，非runtime/Oracle/性能预测。结构收益很小，未排除降低每调用成本的净收益；下一重复适配器只验证完整重算-搬运-同步成本余量，不调阈值/不占B Qwen窗口。原summary计数误标即时强制动作为0，已另文件改38，其余数不变。


### 原funding运行前结构预测直接核验

`20260914_funding_filter_comparison_r01/MODEL_PREDICTION_CHECK.md/json`：PARTIAL_DIRECT_CONFIRMATION。两新filtered在step90的稳定请求/文档/prompt/arrival/计数/块/顺序/历史均等于旧CTX block0（F922），均兑现预先冻结的step891 target0020902/victim0020484、31+211>=234、首输出894、target完成1356、重算89761、末步1451。复用原owner每格1362步exact核验，不重复全组回放。旧block1为F921、15请求各多1 output/computed及0018032多1块，该86779/1447分支未实例化，不算预测miss或替换结果。只确认结构预测支持运行前资格判断，非wall/质量/Oracle；原模型/原件未改，新MD仅排版修正，SHA036861ad…b0ee1，JSON5e2d9210…19b08。


### 2026-09-14 重复保存的增量任务接口边界

`20260914_repeated_staged_model_r01/NATIVE_DELTA_ADDENDUM.md`：native builder按next_stored_chunk_idx/缺失keys产生suffix、稀疏或无新job；无job不证明host命中。get_num_new_matched_tokens会清空connector block映射，不可作抢占前只读探测。新增native_store_delta只核验新job物理块与逻辑key登记，不推断旧prefix驻留/完成；8 CPU接口fixture通过，尚未接重复GPU适配器。


### 2026-09-14 单事件复核收束与重复适配器实现

单事件fresh GPT-5.6-Sol审计WARN、P0/P1=0，支持限定执行资格；原action status初始化标签陈旧，原件保留，由group/cell终态与事件链解释。n=1漂移及完整physical-state未匹配不支持性能因果。新staged_store_rotation.py接入重复prepare/commit、实际增量store核验、native miss重算和flush，导入/既有接口检查通过；完整重复执行仍UNRUN，下一做实际闭包状态fixture，未占B GPU。


### 服务窗口：恢复前KV增长证书与声明cap修正

只用filtered before1026：目标H3772/C0，F242，27 peers；恢复4批joint新增65/128/192/243，首输出批缺1。已知peer该批到cap后的256块不能提前借用。固定FCFS prefix22需65/129/192/242，暂停5peer且保留KV，各少4输出机会；最多7批给目标4机会，条件为不额外admit/no早EOS，非在线或GPU时延排序。代码/证书：`O/20260914_service_window_context_r01/funding_1026_prestate_r01`。模型新增声明剩余output cap/context cap，不再计不存在的输出、完成后age停止；上下文H+n与KV H+n−1边界按安装check_stop核对，10定向测试PASS。见`O/20260914_service_window_ready_r01/DECLARED_CAPS_ADDENDUM.md`。

同一接续复用A原staged-store两臂：victim恢复后均696输出至完成，无再丢弃；gap15.739/17.430s的99.27%/99.65%在恢复调用开始前。真实load3392前缀后执行7重算+1decode，pending212块与ready区分；host配置16GiB但实际峰值未知，C_remaining仍UNKNOWN。`STAGED_INTERFACE.md`保留分账，恢复启动后跨度非纯税。此事件不支持延长服务窗口，下一复用A重复most保存结构检查决定是否值得GPU比较；不追加单事件计时重复/窗口矩阵。


### 2026-09-14 重复适配器实际闭包资格与本地包

`20260914_repeated_staged_probe_r01`：实际begin/schedule闭包在捕获前态+假分配器/传输对象下，off/full/increment提交、target变化取消、缺flush拒绝五路径通过。非完整native scheduler/GPU。重复off/on资格包6f412e39…f2c3/21文件已本地准备，相同d6/16GiB缓存与预热；至少两次实际动作才可COMPLETE，n=1不作为性能GO。原件不改，未上传/无driver，排B Qwen整组之后。


### funding 单 victim 独立结构分叉：即时释放量与后续恢复断点

`20260914_funding_filter_comparison_r01/action_branches_r02.json`保留step891全部27个合法可资助victim独立未来；后续仍同least_feasible，非固定真实future。目标首输出全部894，但重算86794–89773、末步1448–1451、最长输出间隔84–93步。11个候选三项均优于默认89761/1451/90；探索性选择0017453后为86902/1448/84，非预注册胜出、在线规则、完整Oracle或GPU计时。`selected_branch_localization_r01.json`定位：默认step1029 peer增长后48<need49，丢弃2991部分重算而无新输出；候选52>=49并输出。候选首动作多释放4块，后续已独立演进（目标700/703旧输出），不能把所有差异归结为纯4块作用。原默认事件语义复用service_window_context/funding实际定位；下一唯一问题是同runtime单决策替换能否兑现结构排序，保留most强基线。GPU当前归B Qwen完整组，本方没有新增driver。


### funding 六格 fresh 完整性复核收束

`20260914_funding_filter_comparison_r01/EXPERIMENT_AUDIT.md/json`：GPT-5.6-Sol fresh same-family/provisional，WARN，P0/P1=0。逐请求/全部九比较/完整计费/current-state selector replay/6656块守恒通过；单一复用文档组和顺序漂移不支持稳定净收益。审计仅六格实测及主表，模型、生命周期和27分叉不在接受范围。原始safe-cap预测标签保留，以旁侧COMPLETE和组终态确定执行状态。下一仍为单victim因果探针；fresh cohort是后续性能确认要求，非新增并行重复任务。


### cohort3原八格限定完整性复核完成

`20260914_recovery_holdout_comparison_r01/audit/EXPERIMENT_AUDIT.md/json`：PASS / integrity pass，same-family/provisional，P0=0/P1=0/P2=1。八格256请求/262144输出及16比较由原件复算一致，主分析字节一致，全部八点绘图复现；报告SHA85d844ad…78a47保持不变。P2限于历史成本模型未自带训练raw/校准脚本哈希，本审计另记训练raw哈希并重建系数，不影响原生主表；不升级为动作预测或Oracle。停止追加本组审计，保留MEASUREMENT_ONLY及均完成/逐请求代价。


### 服务窗口：请求计时与逐步诊断分离，CPU入口通过

`O/20260914_service_window_ready_r01/REQUEST_MEASUREMENT_ADDENDUM.md`：现有performance模式仍包装scheduler/扫描KV，无已验证轻量入口。新增request_measurement（135行），只取真实请求和engine返回时间，保留外部policy hook及成本，未知抢占/重算/恢复为null。3项CPU行为检查通过（内部工作不重置等待、异常/未来到达分母、外部hook与自然EOS）；入口未接现有GPU runner，CPU_PREPARED/GPU_UNRUN，无性能提升claim。A原21文件重复保存包不改，下一按原队列先做执行资格；若再测性能，诊断/轻量配置分开，旧插桩数据不作时间配对。23:17现场6954/6967仍存活，root无GPU启动。


### 服务窗口：原生host缓冲及失败产物保留边界

`O/20260914_service_window_ready_r01/COST_BOUNDARY_ADDENDUM.md`：当前native CPU KV按配置预分配，store/load共享tensor；模型12.5–12.97GB逻辑内容不等于物理host节约。旧单事件两臂drain均0calls，末load已在末调度前约7.87s完成；未测全部资源清空时刻。已有host_budget_observed可复用，不再写监控器。另发现已暂存repeated runner的drain/动作次数检查异常会漏写raw；最小finally补丁已在root独立worktree应用，实际片段两个异常fixture均保留全部请求/部分输出，主树及远端冻结21文件包未动。补丁和证据见同目录failure_retention/NOTE.md；原执行方接续打新包，不替换旧SHA，GPU仍UNRUN。


### 单victim native因果探针已CPU资格并暂存，GPU UNRUN

`20260914_single_victim_runtime_r01`：依据原funding全部27结构分叉，预定step891 full scalar prestate（32 request states、29running、3waiting、tracker历史）匹配时，target0020902只换一次victim0020484→0017453，其后仍least_feasible；most保留六项同组强基线。条件不符保留执行并标UNMATCHED，不能筛格或称新在线策略。CPU事件/失败闭合、原native factory入口及27selector测试通过；包21d162ff…b9ce已暂存53036原独占路径，22文件SHA匹配，无launch/results。GPU现场仍Qwen6967且共同锁busy，按B→A repeated-staged→本组排队。原funding/runtime raw不变，暂存不是性能证据；分析入口收尾中。


单victim分析入口已闭合：analyzer_cpu_checks记录4条旧实际off路径不变、错误目标output计数被拒绝、旧同角色动作前token前缀相等。独立analysis_r01明确6 UNRUN/0 comparison/9对合同，原21d162ff…b9ce包及22文件未改。新分析器按同SHA补入主工作区，仅分析已有原件，不引入新driver。当前仅待前序GPU整组释放。


### 2026-09-15 重复保存r02失败原件保留修复

吸收服务窗口分支提供的精确runner修复；原capture返回后drain/收尾检查异常可丢raw，修复以finally保留一次raw，错误继续上报。两异常fixture复跑通过，21文件仅run_probe变化。新20260915_repeated_staged_probe_r02已暂存/哈希一致，旧r01封存未跑，不新增策略/输入/性能结论。GPU仍等待Qwen6954完整终态。


### 2026-09-15 恢复执行层合流：计算份额的实际增量边界

`O/20260915_recovery_execution_share_r01/REPORT.md`：不重跑funding六格、不再独立victim搜索。新纯执行模型只接收给定target/实际free/原顺序ready peers，抽取most的逐调用历史KV保留＋余额分配；原六格保护建立后378/378实际token map匹配（首次释放前126调用明确排除）。新增ceil(R/B)整数调用约束0/378改变动作，停止本域该规则GPU扩展。复用filtered before1026独立条件演进：restore-first与ready-first均4调用首输出条件，后者只延后一个peer一次机会（前者81）；自然resume承接保护尚未GPU执行，不称完整收益。自然EOS原四格9恢复全部完成、零再丢弃/零强制动作，不降阈值造问题。4针对性测试通过；revision02修正最初禁全共批的弱参考，全部旧raw/初稿保留。下一只与资源方共用其恢复对象验证自然恢复义务，most强基线/固定3s研究停顿约束下平均完成为主目标，无新审计层。


### 2026-09-15 A冻结重复保存模型对新机实测核对

复用r02/execution_weste_26862原方readback，不重算性能主表。新增prediction_check/REPORT.md：off从329起987步schedule/free/output全吻合，调用1316；on固定delay2前775步吻合、1104首偏差，预测1310实际1312，store12,501,123,072/load20,984,102,912B均完全吻合。delay4总调用虽相同但363已轨迹偏离，不事后选参。工作量模型并非延迟模型；下一仍统一四格成本重复，无新GPU组。


### 2026-09-15 A重复模型首偏差纠正

prediction_check/ADDENDUM.md：1102提交/1103完成/1104 ready真实与delay2相同；偏差来自模型漏skipped_waiting保护，提前1103准备/1104提交下一轮转，真实1105才准备。仅补pending_loads禁准备后off987/on983步schedule/free/output全吻合；三旧策略完整字典回归相同。原失败预测保留，修正为post-hoc实现一致性，非新holdout/稳定延迟预测。新四格仍唯一下一实验。


### 2026-09-15 自然恢复执行义务接入原生schedule与资源模型

`O/20260915_recovery_execution_share_r01/REPORT.md`接续：默认关闭protect_native_recovery仅在原生waiting已选中/实际分配成功/登记计算后接义务，不选新target或victim。完整封存native schedule在CPU假分配/假输出对象上执行，两前态off/on1026–1028相同；1029 off丢弃2991重算，on仅hold0019699一次并完成781位置，1030观察新输出解除；失败分配/首次prefill不登记、声明cap完成解除通过，4旧兼容fixtures通过。与资源方当前模型按单一提交钩子合流：默认most1134/filtered1362步逐项符合原件；on most四自然义务但32完成步/93044重算/末1223不变，filtered重算89761→86770/末1451→1448，4请求快3–4步/1请求慢1步。CPU独立future，非毫秒或GPU收益。源码与状态资格在native_integration；shared rotation_native.py未覆盖。只作为共享执行修正，不另开份额/保护参数矩阵或审计，原26862任务仍由原方执行。


### 2026-09-15 A状态模型在新诊断执行复现

20260915_repeated_kv_service_r01/model_check：固定r02 guard修正与delay2、不注入未来通知，新diag-off987/diag-on983步schedule/free/output全部吻合、总调用1316/1312。复用测量方临时readback，raw SHA4648c381…c39b/b52e838b…82af可对正式归档。相同请求/长度新执行，非独立workload holdout；不重复主性能分析，不新增GPU。


### 2026-09-15 同卡重复保存完整服务 ABBA 已完成：保存纳入本域强基线

`O/20260915_repeated_kv_service_r01/analysis/REPORT.md`：原4062整组exit0，诊断off/on及主性能off/on/on/off六格192请求全COMPLETE。固定同most_output两阶段动作、6656实际GPU可用块、实际native16GiB pinned host（两个tensor引用共一份storage），只改prepare阶段保存已物化完整块；全部warmup cache清空。主两对on相对off吞吐+5.86%/+2.03%，平均完成−5.75%/−2.13%，最大engine-return新输出gap−18.37%/−14.93%；均TTFT+2.182/+65.829ms，但所有TTFT在首保存step329前，动作前可见输出前缀全同且时钟漂移−13.644/+214.286ms，不扣漂移/不把TTFT归因save。完整输出26/32相同，6请求action后不同，未宣称隐藏KV或质量一致。

诊断off/on均44恢复、38强制轮转：重执行163835→3731，调用1316→1312；38已完成store共12,501,123,072B、43load共20,984,102,912B，on末态5961有效host块；off有效0，两臂实际均16GiB，末态pending全0，6格post-drain0额外调用。实际RSS/HWM/cgroup单独记、重叠不相加；父cgroup184GiB，memory.peak不可用是唯一字段缺口。诊断与性能分离，无失败被丢弃。

统一恢复承载engine-call入口，Σ恢复前等待/Σ输出停顿=92.45%/94.96%；E到入口仍约61.829/57.481 request-s，非墙钟saving，E只证观察到的资源资金条件不代表马上可调度。保留load dispatch与入口的不同边界及on一次−1.040ms调用内E偏移。off一次3888重执行→2输出→再重执行3890；on无0/1–2段但仍3输出短段，其后真实复用3872host位置，只余19重执行，不能把GPU释放全部算内部工作丢失。

Verdict为单模型/封闭定长/一个压力点/两对重复的REQUEST_LEVEL正信号，保存作该域强基线；不是窗口机制GO、普适SLO或全局Oracle。唯一下一GPU问题固定save=on，只比min_absence_steps=30/0，其余cooldown20、residency30/victim/资源检查不变，检验额外启动等待是否值得；本轮不追加执行。原r02两格只作独立资格记录，不混入主表。原件见同bundle execution_weste_26862/readback；追加统一S字段分析见analysis-call-boundary.json，旧analysis.json及所有raw保留。


### 2026-09-15 主研究决策：保存闭环后只选全局启动节奏

六格原件已按ba558422…6a7269完整归档合并，先前测量方回读的157文件逐字相同复用，只补32日志/包/释放文件；不重跑主分析或GPU。主线接受原保存报告的本域强基线结论，固定问题见 `expert_saturation/PAPER_ARGUMENT.md`，唯一执行版本入口 `expert_saturation/CURRENT_EXPERIMENT.json`。

新增问题是E到S到底哪项门槛阻塞。复用diag-on的selector/snapshot：592次cooldown决定中488次唯一最老target已absence≥30且记录上下文与资源可筹措，覆盖35轮转间隔；62次显式absence门槛阻塞全部在300–421。E仍不保证native立即接受或prepare共同增长。故上一条提出的absence30/0仅是候选建议，主线下一版本改选global cooldown20/0，其余不改。原六格身份从未改变。

新增独立状态CPU screen（同bundle/start_timing_screen）：取消全局cooldown使调用1312→1270，但轮转38→84、加载43→89、加载位置160096→335472、均完成步1167.125→1177.156。模型无物理传输/淘汰/墙钟结论，不能按少42调用选胜。只投入“提前启动与新增切换/peer停顿的净权衡”，不追加victim/窗口/份额机制；固定一个新提前臂资格＋四格ABBA，本地封包，不继承其它排队组为当前主线。低压力无动作与质量未验证边界保留。


### 2026-09-15 测量方absence启动建议经结构证据后合流，不另跑GPU

`O/20260915_recovery_start_timing_r01/model_analysis/REPORT.md`：从真实before98完整纯decode前态独立比较min_absence30/0，默认30的1214步骤与原diag-on schedule/free/output-count全符。0首次prepare329→300、首输出333→304，总调用1312→1309，但mean completion step1167.125→1167.46875，preempt44→47/load43→46，10请求更晚、11更早、11相同。无未来load通知、固定delay2、非毫秒/Oracle/自然EOS预测；未跑GPU。真实selector/prepare CPU检查确认0仍受cooldown、residency、progress和资金约束。

读取统一CURRENT_EXPERIMENT及新增门槛定位后，本方撤销原absence包/队位：62次absence阻塞仅早段，已有488个达到年龄及记录资源条件的机会被global cooldown阻塞，后者信息增量更大。唯一下一实验以CURRENT_EXPERIMENT指定save-on/cooldown20/0为准，absence30不变，执行归原long-task owner。原8ffa8570…83f7包明确SUPERSEDED_GPU_UNRUN，不上传/不启动；这是优先级合流，不将未执行机制判NO-GO。上一保存报告中的absence建议由本条及统一清单取代，已完成保存数据和结论不变。


### 2026-09-15 B 专家保护成本：一次有界复用分析完成

`O/20260912_wisp_olmoe_r01/retention_cost_diagnostic_r01/REPORT.md`：从实际LRU同前态逐调用分叉普通分组，当前加载均不变；每frequency repeat的190实际保护事件下一真实需求受益1078/被挤出重载893，条件差−185，guard为522/454/−68。frequency额外一组78事件中52条件更好；guard不增组119事件中21条件更差，组数不能单独判值。条件future不作独立策略收益，真实原净miss仍−134/−47不改。12episode完整请求互斥分账99.60%–99.73%在引擎内，调用外配对差≤0.610ms；纯H2D/暴露wait/独立控制未测，未由包络换算秒级收益。停止当前cap24频率保护细调，交付身份/成本模型工具；共享池/fullstage/fresh/logical已完成不重启，仅在途Qwen按原合同闭环。


### 2026-09-15 恢复执行层：加载完成通知后未发现额外漏调度

复用service正式diag-on原件，新增问题仅load通知→实际计算→新输出，不重算主服务表。`20260915_recovery_execution_share_r01/native_ready_handoff/analysis.json`：全部43实际load（37主动保护/6自然恢复）均在通知所在engine-call后的紧接schedule执行并同call返回新输出；待执行2–29位置，0额外漏调度。43个begin快照仍WAITING_FOR_REMOTE_KVS，说明状态标签/已占块不能代替native finished集合消费。通知入口至schedule 0.799–2.108ms含详细host观察与CPU处理，不作纯等待/可回收墙钟；不是新运行或新增重复。与当前pinned源顺序一致；通用scheduler失败分支不等于本OffloadingConnector已支持job失败恢复。

据此暂停本批ready交接/份额扩展，删除未接入适配器草稿，无新CPU测试或GPU；旧重算六格继续HELD_GPU_UNRUN且队位已取消。接受用户统一分工，主目标及下一版本只沿用CURRENT_EXPERIMENT：本方不自任主负责人，不改cooldown/target/victim/保护/份额；下一仅复用原方cooldown对照判断是否出现新执行缺口。复算脚本E/probe_native_ready_handoff.py，范围/源码链/仓库相对命令在同X/REPORT.md。


### 2026-09-15 A cohort3保存迁移四格完成

20260915_repeated_kv_cohort3_r01/analysis/REPORT.md：同域不重叠文档、机制/预算不变、ABBA128/128请求完成。保存对不保存均完成−6.5265%/−5.3842%，吞吐+6.6322%/+5.4168%，最大gap−19.7892%/−18.1576%；每对32/32完成更早、26/32文本相同。TTFT+0.270/+4.587ms均发生首次prepare329前（首输出最晚97），不归因/扣除。实际host四格16GiB、38主动轮转、末态jobs0。新文档迁移非新总体/自然EOS/质量验证；不增加同域扫描。限定fresh审阅待返回，原始四格归档ddb371dd…49722。GPU已释放，主线cooldown动作按CURRENT原owner执行。


### 2026-09-15 保存测量方：8个max-gap受损请求分型与原生容量再抢占

`O/20260915_repeated_kv_service_r01/analysis/PEER_COST.md`：新增问题仅为既有两对同8受损请求的生命周期归因；只读六格，不增加运行/重复数。四主格各32/32请求的(call,累计输出数,new token IDs)逐项与同臂诊断相同。6请求两臂均从未被抢占，on最大间隔是329→330连续调用，增量每个+7.005/+0.571ms、完成均更早；0003640仍为298→333原冷恢复，gap+7.846/+28.632ms。不能把这7例叫作恢复饥饿或把单调用差当纯D2H税。0000799才是改变轨迹的长gap，+126.293/+217.662ms，完成差−661.940/+217.570ms。

diag-on0799于983恢复、985—987输出3token、986解除保护；988 selector cooldown/noop且无强制plan。实际free1，0799自身增量0/持244块；2932、3475、0480、2038各需增1块，2932耗尽后3475分配失败，native抢占0799释放244，最终free241，守恒1+244−4=241。下一次实际load仍复用3872位置、余19重执行、204新输出完成，状态释放不等于全部工作丢失。该段L/E/S_call/F=19.948349/19.968877/22.388558/22.458461s；E依赖替换victim、并非直接可执行，诊断时间不混入性能。

复算脚本peer_cost.py/peer_resource_988.py及全请求JSON/原字段JSON同目录，仓库相对源路径，输出x模式。逐事件token计数/时间连接及本事件12项资源一致性条件通过；不追加review/GPU。保存继续作本域强基线；首次输出非持续服务充分条件，但未证明延长保护净正。统一cooldown20/0实验和执行owner不变；这三类代价交由其完整请求结果区分，测量方不另开controller或保护参数。


### 2026-09-15 A cohort3限定复核收尾

20260915_repeated_kv_cohort3_r01/EXPERIMENT_AUDIT.md：fresh reviewer返回WARN/same-family/provisional，无P0/P1，描述性同域迁移结论保留。新增确认同臂重复文本32/32相同、跨臂固定六请求不同；不推断质量或数值原因。ADDENDUM.md纠正raw中policy_hooks_modified=false的解读：只涉及测量helper，caller实际安装staged scheduler adapter，不能写成stock scheduler未修改。归档核验与审阅范围分开。原raw/包/数值不变，本线停止此组审计；统一cooldown已由主方完成并回读，本方不重复主表。


### 2026-09-15 A模型决策预警与主方cooldown实测对照

20260915_repeated_kv_cohort3_r01/analysis/COOLDOWN_MODEL_HANDOFF.md只引用原start_timing_screen和主方新五格analysis，不重算raw/启动模型或GPU。模型预警调用1312→1270但平均完成步1167.125→1177.15625、load43→89；实际两对完成+2.0417%/+1.4104%、吞吐−0.5021%/+0.1244%、最大gap−9.7975%/−10.4133%。方向支持“不能按少调用选策略”，不支持秒级预测或传输因果分账；gap窗口不同不算误差。保留模型为决策筛选，停止单纯拟合精度扩展，主方决定停顿/效率取舍和下一边界。


### 2026-09-15 恢复执行层：取消全局cooldown仍未产生份额争用

复用主方20260915_saved_recovery_start_r01正式diagnostic-eager，唯一新增问题为既定target下计算预算争用；不启动/回读/复算完整服务主表。20260915_recovery_execution_share_r01/eager_execution_boundary/analysis.json：84/84主动commit兑现首输出，83加载恢复仅余2–28位置、1计算调用，另1冷恢复3274位置/4调用；87就绪计算调用全部满足最低目标份额＋全部resident peer≤1024，无KV保留导致peer延后。仅3个纯resident前态输入模型，逐项符合原分配，preserve_calls新增动作0；加载转换只报实际需求，非pre-action预测。

166无目标计算调用为83PREEMPTED/派发＋83WAIT_REMOTE，peer仍推进，不能整体扣为浪费或可回收墙钟。全请求损益直接引用原方两对：eager最大gap−9.7975%/−10.4133%、平均完成+2.0417%/+1.4104%、吞吐−0.5021%/+0.1244%，保持停顿目标及效率代价，不另定SLO/显著性。主分析引用SHA1997da066120f7264ac744392c84e79c349242a1ca90d43bf262976090b83a1e。

更频繁切换没有暴露此份额规则的增量动作，MEASUREMENT_ONLY/本域份额扩展暂停；旧重算六格保持HELD_GPU_UNRUN，不新增GPU/模型/审计。复用E/probe_staged_execution_boundary.py（仅CLI支持指定诊断格），命令和范围写回同X/REPORT.md。首输出后短服务、完整代价与下一负载域由主方/资源模型原owner决定，本方仅在预定义新域实际出现预算冲突或首输出前中断时重开执行修正。


### 2026-09-15 保存测量方：cooldown的间隔频率与累计代价分解

`O/20260915_saved_recovery_start_r01/lifecycle_cost/REPORT.md`：只复用唯一五格原件及原保存diag-on；主表/90恢复/58再抢占/无0或1–2输出等主生命周期结论直接引用原方，不重复运行或增加样本。新增部分是四轻量格128请求可见(call,count,new token IDs)全序列与各自诊断吻合后，将诊断抢占位置标注到轻量实际输出时间。轻量自身未记录抢占，故只作条件标注，不能称直接测得其90/44次抢占或隐藏KV等价。

标注间隔44→90，每段均值约1.30→0.72s，但轻量累计标注间隔57.301549→65.146933 / 57.527558→64.647385 request-s，分别+7.845384/+7.119827；其余生成间隔+6.003270/+2.613267 request-s。TTFT+标注间隔+其余间隔+完成尾部逐请求守恒，128请求残差0，与原主分析完成分母全同；不是墙钟分解、纯排队或DMA税。两对同17请求累计标注间隔增加，其中同8最大间隔反降。1710段数1→3、最大1.53→约0.65s但累计1.53→约1.94s；0799段数3→4而最大/累计均下降，频率本身也不足以判值。

新增短段资源原因：1710/0133/2733分别6/3/4新输出后在669/821/990被native容量抢占，调用开始free0、目标自身增量0，由peer增长分配失败触发；非forced plan。0/1–2缺口仍为空，不改阈值造窗口GO，不把3例归因成全部代价；host复用原方结果直接引用。复算partition.py/partition-annotated.json与short_service.py/json同目录，源码相对、输出x模式；初版字段preemption_count已明确改为diagnostic annotation，科学数值全同且原件保留。

决策增量为“提前恢复的较短单停顿可能伴随累计中断增加”，不是零输出恢复增多；保存仍强基线，cooldown0是否接受交主会话固定目标权衡。下一应检验该取舍在代表性到达/长度/EOS边界是否仍存在，不追加本域扫描或窗口controller。本方完成此MEASUREMENT_ONLY交接、无GPU动作。


### 2026-09-15 B Qwen r04终态与有界支线交接

O/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/attempt04/REPORT.md：原32/16/16/32四格及唯一回收已完成，parent/child exit0，原GPU已释放。真实超单卡显存Qwen3-30B-A3B BF16，同48槽/层与512MiB请求KV预算，16请求执行/416输出/1552位置。static16相对32两块专家payload+4.801%/+3.000%、组数+1.114%/+5.318%、TTFT+39.545%/+34.436%、maxITL−6.741%/−14.407%；平均完成+9.062%/−6.432%、capture+4.410%/−8.698%，仅描述性权衡，无稳定完整效率优胜。纯decode一组仍有31.369–43.354GB专家payload，不能将组数或单调用重复加载0解释为无搬运。

资源、共享加载/资格/预热/IO均保留：CPU专家master57982058496B、GPU专家scratch21743271936B，90GiB共享父cgroup，无每格独立主机硬隔离；runner总寿命13313.962秒，不将单格capture称冷启动完整成本。原47/48全参考allclose与layer47 maxabs0.25不改写；局部分组执行归因通过，不代表质量等价。既有reviewer新raw定向检查PASS/non-fresh/provisional，停止本组复核。完整结果归档c81b9d56…557d，原先失败attempt保持原件。

B的frequency、guard、复用、共享池、fullstage、fresh、logical P/V及本次Qwen均已闭合。旧cap24保护停止细调，逐身份替换债务/请求区间成本工具与模型见20260912_wisp_olmoe_r01/retention_cost_diagnostic_r01/REPORT.md。无新GPU排队；重新进入动作实验须由主会话在指定域证明同预算强无保护底座后的完整成本空间，不从条件少加载或本次Qwen权衡自动推导。非paging家族NO-GO。


### 2026-09-15 save-on启动五格闭环：少长停顿，多切换与均完成代价

O/20260915_saved_recovery_start_r01/RESULTS.md：包341e6ac3…6e52、controller13136，1789410408.127—1789410688.523，diagnostic-eager/current/eager/eager/current均COMPLETE，160请求全完成；唯一归档a0290d0d…75aa0/5 raw SHA已回读。诊断84实际轮转/80个间隔<20、84完成store/89 load，15974006784/43970985984B，90恢复段、4058重执行、1270calls。轻量current1312/eager1270calls，各38/84轮转，host实际同16GiB、有效5961/7617块，pending末态全0。

两主配对eager/current最大gap−9.798%/−10.413%、输出吞吐−0.502%/+0.124%、平均完成+2.042%/+1.410%；23个gap好/同9坏，9个完成也坏；完成改善仅12/14个。输出26/32相同、质量未测。吞吐小幅翻号不是等效或无效证明，不为此追加同域重复。旧diag-on只作不同组描述参考；新90恢复/58再抢占但无0/1–2输出段，短段后真实host复用3776/3920历史，仅重算8/6。统一S_call等待中位1.296575→0.642477s，累计request间隔61.955719→71.147706s，不能当墙钟分解。

研究选择：保存后启动节奏确实改变完整停顿—均完成权衡，current/eager作为同资源简单基线保留；无稳定吞吐增益、无方法GO，不扫第三个cooldown。下一唯一动态运行域资格在20260915_natural_saved_recovery_gate_r01本地准备：64完整自然文章、允许EOS、原8GiB usable KV/16GiB host，预先固定0.2s受控到达、current save-on一格；旧0.5s低压力无额外轮转直接复用，新点无动作就保留并停。必须区分真实无机会与closed-cohort/mixed-prefill实现限制，不能循环加压或把实现范围外判死问题。CURRENT_EXPERIMENT是唯一合同入口，主线不再占用已释放GPU。


### 2026-09-15 保存测量方：后续直接抢占事件的最小接入

`O/20260915_saved_recovery_start_r01/lifecycle_cost/MEASUREMENT_ADDENDUM.md`：共享request_measurement.py新增显式record_preemptions=True，默认关闭。仅链式包装当前_preempt_request，保留原返回/异常并finally恢复已有实例hook或类方法解析；不改schedule/allocator/策略。记录真实engine-call号、请求ID、此前已返回输出数/L、单列native内部计数和方法返回/失败。completed只指方法正常返回，不代表engine/GPU/async flush完成；原因不推断。默认仍unknown计数，启用标SPARSE_PREEMPTION_EVENTS而非无observer。

独立worktree实现后核验共享旧内容未变再同步源码/测试；原3行为＋新增成功/失败两项共5项通过，覆盖内部99但实际返回1、重复输出不重置L、已有hook委托及两种恢复方式、失败partial raw。CPU_VERIFIED/GPU_UNRUN，扰动未测，旧原件和窗口机制结论不变。下一组如需直接归因由原主执行方共同启用两臂并纳入实际成本；无新GPU/控制器或审计矩阵。源码SHA1b322be02505381dedb0aaf47f58513dfef61d60bf98e8d9590e18e3012534e4，必要检查和调用方式在addendum。


### 2026-09-15 自然到达保存资格：共同KV增长出现实际短恢复

O/20260915_natural_saved_recovery_gate_r01/RESULTS.md，接受3603de48…8b00，root controller18977，1789412553.090—1789412635.117；单格64/64完成、59564输出、6EOS/58length，归档674854a0…c6c6/20876821B已唯一回读核验70文件并释放。固定4096usable GPU块＋null实际8592031744B、host16GiB，原64完整文章0.2s到达。25完成store/31load、7075790848/11085545472B，37恢复/22请求；6次prepare在lastarrival前，动态路径资格成立。无性能配对，诊断39.925626s/均完成21.664046s/最大gap3.531065s仅描述。6EOS全部先于首次prepare8.109950s，恢复对象均cap1024，不支持恢复途中EOS泛化。

两次1–2输出再抢占通过独立源定位复用：0406/0748强制恢复target已load2464/3792、重算17/7，输出2/1后native step903/1161抢占；自身增量0、free0，peer各需6/5新块。排除该两例加载失败，支持共同资源增长与执行分配，不自动延长保护。absence≥30的201个固定victim-funded cooldown观测＋115direct观测是必要条件、非独立样本或立即可调Oracle。下一先同资源native完整增量保存资格补强基线；旧prompt-only不冒充full，当前仅CPU兼容性检查/GPU UNRUN。原0.5s低压无动作边界保留，不扫描压力。


### 2026-09-15 native完整保存资格通过，短恢复机制暂停

O/20260915_natural_native_full_gate_r01/RESULTS.md：包71ee97f3…5902，root controller20050，64完整/59581输出/6EOS＋58length，25rotations，3866store/36load全接受完成，21667774464/12043943936B，266重算。3864个store在selected prepare之外、3723含decode、2个合法超旧prepare-prefix，5finished store/9flush；native_calc未覆盖。73文件归档f79e4c60…0c7a/22526879B已唯一回读并于1789414105.591释放整机。host实际16GiB且末态8192有效块，无pending，drain0调用；请求诊断墙钟不得同D排名。

36恢复中16再抢占/20完成，最短再抢占段4输出，0/1—2均0。故D的2短段不能继续支撑窗口/headroom实施，暂停该机制；原生完整保存进入强候选底座，但完整服务谁优尚未测。唯一下一轻量selected/full/full/selected四格由既有prepare_start_contrast恢复执行会话封包、获CURRENT接受后唯一执行/主分析；模型方自主检验跨D/E共同增长预测与free-only基线，近邻方仅核对直接动作碰撞，root统目标与研究选择。无重复诊断或额外参数扫描。


### 2026-09-15 A：共同增长容量排序退化为强简单基线

O/20260915_natural_saved_recovery_gate_r01/joint_growth_model/REPORT.md。仅新增当前前态候选分析，原GPU数据不重复计数；25prepare合法victim全匹配，h=k=16、无未来EOS/释放/通知。最大释放块同分候选25/25达到最佳条件容量余量，复杂排序无容量residual；实际most与最大余量不同19/25但非性能错误证明。most负margin四例818/898/1157/1342为−10/−21/−23/−6，最佳107/117/−23/59；898可换victim补余量，1157所有合法候选不足。只是立即swap空间包络，不含prepare/transfer/新准入/peer服务税，不保证持续服务。旧草稿progress分母错误已纠正并保留INVALID。停止复杂容量predictor，native完整保存资格后再检验不可共同推进状态；不另启动GPU。


### 2026-09-15 A共同增长跨D/E：容量条件不是短恢复预测

O/20260915_natural_native_full_gate_r01/joint_growth_transfer/REPORT.md。模型原字节/h16不变，E25prepare全部纳入，最大释放同分规则25/25容量最优、全部存在非负候选，无复杂容量排序residual。两组各25prepare与原方恢复段唯一关联；即时free+release资金均通过。D共同余量负4例输出5/2/1/8，E负2例输出9/4，不能据负余量预测≤2输出；D/E另有非负margin2/6/0却14/15/13输出再抢占，不能保证16输出。只是跨诊断描述关联，无独立50样本/因果保存归因。停止margin直接触发窗口/延迟规则，下一CPU验证prepare期peer增长和等待单调性，不改主方轻量对照或占GPU。


### 2026-09-15 跨D/E共同增长模型：解释成立，选择增量不足

O/20260915_joint_growth_decision_r01/RESULT.md复用两自然资格全部25＋25首输出保护释放边界，无按结果过滤。G1=sum(max(0,ceil((computed+1)/16)-held))与free比较，D仅1个即时native事件报警；G2的D两报警恰与free==0相同，E所有两步余量至少4块、无对应事件。块内位置解释D902首步不缺/下一步6块，但没有比最简单free-only多出动作集合。两episode观察性结果不是50独立实验或动作净收益；模型降为诊断，暂停扩大predictor或分配GPU动作。后续正常强底座出现可作用的负余量及完整成本空间再重开，不造压力寻找报警。


### 2026-09-15 保存成本会话：native-full逐块复用与未来恢复税

O/20260915_natural_native_full_gate_r01/saving_cost/REPORT.md及analyze.py/analysis.json已实际运行；只复用主方canonical分析，不新增样本、主表或GPU。10332不同注册块各保存一次；36实际load按固定原生连续prefix路径重建5743块访问，3225不同块，2518重复访问(4.917969GiB)，字节和completed身份精确闭合。7107保存块在观察期未被load访问(44无load请求6697＋恢复请求410)，这是事后标签，非可在线免除成本。16再抢占释放41386GPU已计算位置，下一恢复加载复用41248，仅138重算；全部36恢复266重算均为0—15位置尾部。5679仍需990新prefill，不能以2重算位置冒称2位置恢复。否定本格“整段历史计算丢失”的解释，保留重复加载/等待成本；原生保存游标非ready前缀。支持主方已选selected/full轻量四格，未提出新窗口/阈值/GPU组。


### 2026-09-15 A prepare资金变化：纯增长等待不增加余量

O/20260915_natural_native_full_gate_r01/joint_growth_transfer/PREPARE_MARGIN.md。当前状态一拍模型fund_next=fund_now−peer_growth，victim自身新增块随释放抵消；无外部释放/成员变化时不能靠等一拍增加资金。D/E各25prepare各24完全匹配；两误差是D858原生抢占peer5122释放194、E1398末token完成peer2134释放174，原预测/误差保留。48其余态人口/target不变且各推进1位置；不是独立样本。D/E预测严格减19/20，但50格当前可行→不可行0，原commit已有实查，因此不新增guard/预留。CPU最小反例证明纯peer增长可使即时刚好可行变不可行，不当自然性能。下一只在强底座完整成本或可见释放事件能改变动作时扩展，不继续拟合。


### 2026-09-15 专家分页自主接续：group-map阻塞与提交模型

O/20260915_pager_map_cost_r01/REPORT.md为本轮唯一新证据入口；继承专家分页/执行组织独立职责，不转接新B保存语义。独立worktree /private/tmp/moe-paging-cost-20260915，基于de64dae5。仅复用Qwen r04原4格：7584 layer/8656group，固定512B group map的host区间占apply29.258%–33.914%。源码与PyTorch2.11证明其blocking device构造承接同stream前序异步权重/persistent-map等待；该比例不是可回收墙钟。前两格拟合、后两格检验3792首组，required-only模型WAPE73.276%，加当前miss后15.450%；无未来route/EOS输入，仍为同engine探索性切分。第二块static16 map反多130.862ms而apply短380.761ms，未解释完整性能翻号。

最小提交模型saving=min(H,W+min(M,H)-C)允许async胜/败，收益受host launch H限制，不能消除GPU权重依赖。新partial_map_staging.py只改group-map上传，CPU缓冲DMA生命周期/stream及模型边界验证通过、exact frozen apply安装可编译；真实CUDA及完整服务UNRUN，两臂需同持host/GPU各73728B bank（Qwen几何）。已有batched resident-map修复保持基线；共享池X同类blocking路径仅结构核对，不外推Qwen毫秒。

当前不重启frequency/guard扫描。唯一实验建议为有限单层blocking/async资格与完整调用成本探针，先判断是否只移动等待；尚无主方接受执行身份，不封包、不占GPU或重复读回。该新假说和可用组件由本支线自主推进，完整服务取舍/共享排程仍归主方。


### 2026-09-15 C恢复执行：无抢占等待释放的容量边界与条件分配

沿新子会话指令自主研究执行/运行域，不再把next_group空当CPU禁令。复用native-full资格原方主分析，取全部16个恢复后再抢占段，排除9个上层forced，7个native全纳入，未设短输出阈值。X=20260915_recovery_execution_share_r01/native_full_release_boundary/analysis.json：7目标下一decode均可资助，整个resident集均不可共同资助。6个前态min_i(max(0,ceil((C_i+U_i)/16)-A_i))>free；仅调整计算顺序/peer hold、不提前EOS/回收持块/外部释放时，任何请求都不能先到声明cap，最多新增243/224/196/234/190/182输出机会后耗尽。这是无提前EOS的容量界，不是停顿/墙钟上界，U不是已知实际剩余长度。

1377前态为free3、最少到cap需1块；通用条件规则自动选尚余cap16的0433，并为既定target2680持续16调用＋0433到cap合计保留2块。CPU各自推进：简单延长history escrow在6调用/target6输出后缺块且无人完成；条件规则16调用/target16输出、0433到cap，合计200输出机会，5103/0464/0406全16调用无输出。不同停止时刻不作吞吐对比；finisher134持块未计入free，原生store/flush物理释放尚未验证。7个事件目标实际都length结束；新EOS恢复泛化未测。

可复用E/recovery_execution_share.py新增decode_release_envelope/allocate_completion_bridge；E/probe_decode_release_boundary.py给出所有前态/分支/请求代价与源码SHA。4旧测试＋512微型状态所有合法顺序验证容量界＋双对象保留不借完成释放，共6测试PASS。STRUCTURAL_FROM_NATIVE_STATES/CPU组件，非GPU替代轨迹、质量或完整效率收益。原固定域份额暂停结论保留；只有可见cap路径成立才保留一个条件执行候选，不扩大固定窗口/候选搜索。

主方selected/native_full轻量四格仍由其既有prepare_start_contrast代理唯一执行；本方不打包/代启/改变其合同。下一本方最小工作为同native-full调度入口的条件分配与真实回收依赖接入资格，先给资源模型方复用当前界和接口；是否进入完整episode对照须承接强基线效率结果，不把本例CPU计数升级方法。


### 2026-09-15 A commit重检发现两次可改变驱逐动作

O/20260915_natural_native_full_gate_r01/commit_recheck/REPORT.md，纯CPU组件/全50commit检查，各组1次满足READY且free独立资助target：D859 free193/need95仍驱逐233块，E1399 free277/need106仍驱逐184块；保留全部当前peer一步后余量97/171，无pending。规则只用commit当前状态，保留全部现有资格/target guard，不预测EOS或未来释放。提出direct_resume分支，尚未接native或执行；保留victim可能增加后续压力，必须同底座完整对照，不称方法GO。已接受selected/full四格不变；完成后建议最小default-off commit重检资格，不另建controller/压力扫描。


### 2026-09-15 保存成本会话：轻量直接事件累计等待组件

O/20260915_natural_save_scope_timing_r01/output_wait_cost/partition.py与README.md已从独立worktree交付，3项定向CPU检查通过。复用本组既有稀疏事件，不重复主方sparse_segments：按实际输出位置去重间隔，区分含成功抢占/其他生成间隔/首输出前/完成尾部/未闭合尾段；多次抢占同一间隔只收费一次。方法失败、method成功但engine call未返回、零输出终止和未到达请求保留，完整请求与观察期分别守恒。CPU_VERIFIED，尚无该分解的真实GPU结果；不把marked gap称纯恢复税、不投影旧诊断、不改冻结包。原唯一执行方已启动四格，本方无回读/GPU进程，下一只定向分解canonical raw。


### 2026-09-15 A commit重检接入独立adapter副本，CPU分支通过

同commit_recheck/INTEGRATION.md：default-off recheck_commit_funding已接候选完整adapter，direct_resume保留victim、复用target guard/队列提升、独立direct计数，不造preempt/flush/rotation，schedule后清plan；默认关/资金不足沿旧swap。实际候选begin/schedule函数体经AST绑定CPU替身检查direct、off、资金不足、取消、victim pending-load拒绝及清理通过。未跑完整install/真实native或GPU，状态CPU_COMPONENT_CHECKED_GPU_UNRUN。统一diff、候选、检查源码同目录，原共享源码和已接受四格不动；后续按原建议先真实资格再评完整peer代价。


### 2026-09-15 B完成释放：可分配不等于无需flush

O/20260915_natural_native_full_gate_r01/finish_release/REPORT.md及analyze.py/analysis.json，复用全部64完成原件与同hash源码。63个有后续快照的请求均不再held，最后1个无后续snapshot保留未观察；6请求store完成上报晚于终止输出，5为finished时新建，其中4与同step真实flush连接。1582终止后已0held，才注册最后chunk234/store2186/source1257并flush；native正常request_finished返回False，core立即free，store不给finished_sending。推翻“等所有store ack才可free”，也排除“free容量无同步依赖”的假设。C/A可按当前原生free判断容量，沿原生jobs_to_flush执行复用，不能伪造free耗时或从完成上报推DMA终点。本方未实现C allocator/修改pager/GPU。

### 2026-09-15 B轻量四格累计等待分解完成

O/20260915_natural_save_scope_timing_r01/output_wait_cost/RESULTS.md及compare.py/comparison.json/cells.json，复用原方已资格256/256轻量结果，无新样本/主表/回读。full−selected含抢占生成间隔累计+3.030997/+14.044425 request-s，其他生成间隔−24.319303/−1.801406，首输出等待−1.474430/+5.052421，守恒到累计完成−22.762737/+17.295440。抢占28对28、28对27；同0433/0464/0748/0913两对均多等，0748次数减少/相同仍累计与最大gap均变差，不能只用总切换次数解释。只用本格实际稀疏事件，无diag投影；bucket变化非纯恢复税。旧full诊断重算/字节不回填轻量。建议保留selected长停顿强基线，主方若推进下一项则仅验A commit当前free重检，比较其他请求及累计代价，不加窗口/并发传输。


### 2026-09-15 双GPU资源与部署效率：释放显存未购买额外服务

本方独立worktree `/private/tmp/moe-dual-instance-20260915`，沿用双实例host/部署职责，不接管A/B/C恢复。当前报告：[RESULTS.md](/private/tmp/moe-dual-instance-20260915/research/dual_instance_transfer/RESULTS.md)；资源账：[resource_analysis_r02.json](/private/tmp/moe-dual-instance-20260915/research/dual_instance_transfer/results/resource_analysis_r02.json)，对应resource_analysis.py；D1因果定位为同目录trajectory_boundary.py及results/trajectory_boundary_r01.json。只重分析原22格/30 engines/300请求，新增GPU样本0；原32格copy与22格serving不重跑，raw不改。

分页每GPU实际unique专家池4.5GiB（3.9375GiB私有view+0.5625GiB staging已包含），12GiB本地NUMA pinned母本/实例，双份共24GiB。Torch allocated约6.4123GiB；native为13.9221GiB，差7.5098GiB/卡。两者KV均配置1GiB，请求数未增加；三个cohort声明cap保守上限40/320/520MiB均小于511可用块约1022MiB，全部抢占/重算0。native KV来自预算/8192-token启动日志与同模型布局，非storage探针；native host母本未观测写null，测量前HWM不冒称最终峰值。

实际双实例共同窗口：native7.7923/7.7954 req/s，paged4.0107/4.0069 req/s；native平均完成0.3563/0.3895s，paged5.1979/5.2063s。当前部署应保留常驻效率基线；有限cohort非稳态容量倍数、两底座非纯cache-size因果、输出/质量等价未证。native最大ITL0.0148/0.4225s，paged0.2037/0.2012s，保留尖峰，因此不宣称所有停顿要求下Pareto占优。

D1先前−4.006%变化新增定位：初始cache已不同，step0有plan/miss差，第二请求在solo第5步而dual第4步加入；dual共同row首次route顺序/集合差在0.389510/0.444433s前，对侧P0服务到0.750247s才开始。排除“对侧测量期服务传输导致所有早期差异”，不排除双engine准备/环境影响、不将后来全部字节差归因首次route差。7/10分页配对完整需求签名一致，P0 ABBA也一致却变号，不能把所有小波动都归因需求改变。

投入决定：当前没有新共享瓶颈依据，停止联合controller追加投入，也不为更大模型自动跑矩阵。下一唯一条件假说为真实需求超出常驻经验证安全KV容量，而分页释放空间能承接额外服务；单纯超过当前1GiB不成立，native可先扩大KV。只有该资源关系有实际依据时，先资格单实例host/GPU/磁盘峰值，再同双GPU/host上限比较最强合法常驻KV与固定分页池增加KV，评完整请求及所有停顿损益。当前不申请GPU、不封包、不轮询；已接受主线保存范围四格版本/执行权不变。三个去重/KV边界CPU测试通过，全部native容量日志/paged NUMA及storage核对通过，未追加审计。


### 2026-09-15 保存范围轻量闭环：full不升级，下一验证自然域启动节奏

O/20260915_natural_save_scope_timing_r01/RESULTS.md由原唯一执行方prepare_start_contrast交付，root复用主分析不重算。四格256/256，接受8f9f0f34…9de7，controller21125于1789415599.505退出；归档be54df2b…f0c59/6,567,737B，156文件＋29payload核验，GPU与共同锁已明确释放。两配对full/selected最大gap+13.19%/+63.96%、输出吞吐+0.90%/−3.52%、均完成−1.92%/+1.49%，输出−485/−937、序列同42/43条。full的p90/p95也均更长，不能只用多数请求改善忽略尾部。四格1–2输出再抢占2/3/0/0，零输出再抢占均0；不能用E诊断一次无短段推断full消除。全部最大gap跨真实抢占至首个新输出，精确启动/load/重算本轻量未测。EOS7/7/8/7且自身均未抢占，恢复途中EOS未验证。

同GPU实际8592031744B/4096usable＋null、host分配16GiB；full末态有效16GiB，selected2.73/4.88GiB，后者非测量峰值。drain均0调用、8.5–10.6微秒，完整成本不扣除；HWM18.33–18.37GiB与KV重叠，父184GiB不是独立进程树硬预算。原生0.26默认VLLM_USE_SIMPLE_KV_OFFLOAD=0；simple在APC关闭时跳过初始化，非同配置可直接替换的更快基线，近邻方已有源码短addendum。

研究决策：selected/current为本域停顿优先参照，full保留必要强对照，不宣称全指标支配/普遍NO_GO。保持窗口、headroom和增长predictor暂停；旧诊断完全复用。唯一下一自然selected/current20-eager0节奏对照由同执行方CPU准备，最多缺项eager诊断1＋轻量current/eager/eager/current4、320请求；固定输入资源/保存/victim/保护/份额，仅global cooldown不同。尚无新包接受/GPU运行。模型方给可检验的启动及peer代价预测，root选取舍；不因效率小幅变号无限增加同域重复。


### 2026-09-15 下一包接受前补齐原生系统参考

近邻方核验最近原件：D/E/F均安装rotation，旧native baseline是不同d6/prompt-only，精确自然0.2s/4096usable/16GiB域没有native-full且无额外轮转的完整服务数据。因此G尚未接受时修订为diagnostic-eager1＋native_full_native/current/eager反序6轻量，最多448请求；该原生臂只跳过install_rotation，保存仍full、资源/测量相同。current/eager之间仍selected且只改cooldown，第三臂仅评价系统强基线，不用于单因素归因。旧320是接受前草案，不追溯改F或任一已运行组。已有模型START_CADENCE_DECISION确认D有201必要候选/18原间隔、E304/22，但每组3个prepare增长不足，计数不冒充READY/Oracle；跨状态预测指向peer暂停和完成代价而非重算损失，实际结果可否定。G唯一执行方仍prepare_start_contrast，精确新包尚待接受，GPU未启动。


### 2026-09-15 C完成分配组件：块池归还与覆盖屏障分开

C延续恢复执行职责，复用自然native-full与最新selected/full四格，不启动/回读GPU。O/20260915_recovery_execution_share_r01/REPORT.md末节交付verify_completion_bridge_native.py及native_full_release_boundary/native_method_shared.json：给定1377事后target，16次原生schedule CPU分配均与条件模型一致，target16输出机会、peer0433到cap后原生free归还134块；下一resident-only调用23请求、分配22块。人为pending-store条件执行submit→wait→覆盖前边界，未做真实DMA/GPU写入。原基线0433 step1392完成单点池204→338实测支持正常完成释放。纠正此前“须等store完成才回池”：本版request_finished返回False，pool可分配与flush后可覆盖分离。保留3peer各16调用无输出及全部逐请求代价，无完整收益/有限等待/质量保证；两个CPU条件非独立性能样本。共享命令已实际通过，首次夹具未合流开关兼容错误与原结果保留，不改共享rotation。最新主分析full对selected的max-gap两配对更差，full不称全服务最强。唯一接续为向资源模型接口合入实际pool读取及原生覆盖屏障，再按共同强基线决定默认关闭组件资格；不新增victim/阈值搜索或冻结GPU包。


### 2026-09-15 专家分页成本：完整map复用不足，候选需处理两个发布边界

O/20260915_pager_map_cost_r01/shared_reuse/REPORT.md、analysis.json/events.json及E/analyze_shared_map_reuse.py：仅复用原logical P/V四个X运行，新增GPU样本0。3,552次同层可比较转换中384项execution map完整相同为0；含冷shadow强制上传的7,232次原两张map上传仅23次64项local map可跳过（0.318%，5,888B）。本批数据排除完整内容缓存快路径的足够覆盖，不外推active项语义复用或其它运行域，不实现额外cache分支。

同hash强X源码先构造64项CUDA local map再构造384项execution map；第一次blocking factory已承接权重队列等待，只改第二张保留了首barrier。E/shared_map_staging.py因此提供blocking/execution_only/all_async三模式，保留原slot/LRU/权重拷贝/局部map D2D/kernel几何及ordered dispatch，各模式同持host pinned/GPU各28,672B；固定consumer stream，必要DMA完成等待保留。五项CPU语义/时序检查通过，真实CUDA/完整请求UNRUN，不以检查充当方法收益。

显式publication_timeline新增各段准备/分配成本；先前H上限仅在共同准备抵消等条件下成立，预分配可能改变准备时间，不能作为真实实现普遍上界。当前只留一项单层CUDA建议：已有首个有加载/全命中状态，三模式及反序，计完整调用与实际copy/kernel时序，区分少准备、提前提交和等待搬家。局部无空间停该实现，有空间再建议强X完整服务对照；无执行身份、未封包/上传/GPU，旧cap24频率保护停止边界不变。


### 2026-09-15 B限定commit重检的未来成本与READY语义

O/20260915_natural_native_full_gate_r01/commit_recheck_cost/REPORT.md、analyze.py、analysis.json复用A全部2/50改变动作，未改A/GPU。D859 victim0748实际下一load464MiB/8重算/gap0.112243s，E1399 victim3790为366MiB/5/gap0.260893s，均非各轨迹最大gap事件；不作净saving或主目标收益预告。D prepare858 job122在commit时已登记未提交，E prepare1398无新store、此前60job/183块已完成，READY不是host可加载保证。当前direct候选不取消D延迟store，也不能撤销E已完成保存；D2H未提交不叫沉没传输，但仍是该候选保留成本。成本模型单列下一load/尾部重算、后续其他恢复、flush暴露及全请求等待。建议fixed selected/current资格覆盖未提交store分支，不以E旧保存已完成一格冒称覆盖；不另建执行组。


### 2026-09-15 C给条件完成分配计peer代价，暂停直接接入

O/20260915_recovery_execution_share_r01/REPORT.md末节及native_full_release_boundary/peer_service_bound.json：新增E/recovery_execution_share.py::decode_service_envelope和probe_bridge_service_bound.py，复用唯一1377正例、无新GPU样本。给定H=16且target2680/finisher0433每调用各输出一次，F3中2块必需，余1块对应4个块内零余量peer，故任意同条件分配至少3peer整段无输出；最多总200输出机会，已有简单分配达到，24×16计数标尺下至少184未服务机会（不是实际baseline、毫秒或净损失）。8项CPU测试通过，新界与1280个小状态所有末态输出向量枚举一致；共享入口复算一致。条件明确为无提前EOS/释放/抢占/新准入，计算预算足容cohort；范围外拒绝套用，无全局Oracle。A现有direct/swap仍首输出解除，C桥会新增跨首输出义务，非参数兼容补丁。因此撤回上一条直接接入建议，暂停当前bridge runtime/GPU扩展，保留可复用执行/peer成本界。唯一新证据来源继续复用主方G自然节奏诊断；有实际执行冲突再评低peer代价动作，无冲突不降阈值加压。不重复A/B计数/主分析，不更改其包或GPU身份。


### 2026-09-15 主研究取舍：保留commit重检，优先闭合启动/原生参照

root复用B的F/output_wait_cost/RESULTS.md：含抢占间隔累计两对+3.03/+14.04 request-s，抢占28对28/28对27，说明总次数和少重算均不足以选择完整服务。分账为重叠request时间、非纯恢复税，不新增样本。B的commit_recheck_cost也表明A全部两例victim随后gap仅0.112/0.261s，均非当格最大gap；当前commit不前移target既有等待，且保留prepare已登记的原生store成本。故commit重检保留为有依据的基线完善候选，但不据这两例预告主目标收益、不同时塞入G。root选择先闭合实际存在多个受timer影响区间的启动节奏及缺失的同资源原生系统参照。G已接受f1a3bda1…727027/30payload，唯一执行方受委托前台7格，主分析与模型预测对照后再决定；不扩展窗口/predictor或转移到多卡控制器。


### 2026-09-15 专家分页条件调用：前态复位是测映射成本的必要条件

O/20260915_pager_map_cost_r01/shared_reuse/probe_preparation/REPORT.md、prepared.json及E/probe_shared_map_publication.py已交付。原P/1_X的call96→112与967→983完整slot/LRU/map前态，经原planner重现两个目标plan逐项相同；符号执行首次均0错误，有加载状态若不复位重复旧plan则18/56个active专家读错，全命中仍0/8。原因是原18个D2D源随后被H2D覆盖；这否定固定旧plan热循环的测法，非原X运行bug/新策略反事实。

单层driver实际调用原ordered_dispatch/planner/oneshot，每次恢复相同前态；性能baseline使用未修改oneshot且同持bank，处理新增包装成本计入。完整调用含route/plan/copy/map/kernel及GPU drain，复位单列；profiler诊断另跑。合成BF16权重/hidden和64-way softmax后gather概率匹配OLMoE不归一化top8语义，真实route/plan；full64数值参照需额外0.75GiB GPU资格预算，非性能常驻。CPU准备已运行、源码解析通过，真实CUDA/数值/时序/完整请求UNRUN。唯一下一仍为同前态三模式及反序的单层资格，不追加频率模块；执行身份未接受，未封包/上传/启动。


### 2026-09-15 C在自然selected/current定位真实冷恢复份额边界

X=O/20260915_recovery_execution_share_r01/natural_cold_share；current_diagnosis.json＋native_branches.json及同REPORT末节。复用D全部25主动commit，新增问题是首输出前份额：21加载恢复2–20位置/单计算，4冷恢复3030/3044/3078/3094位置/4计算；12resident前态模型分配全部吻合，其中2个preserve_calls改变分配，均为5122不同恢复，非独立重复。step637 R2035/29peer与730 R2048/28peer从同前态分叉原生CPU schedule；相同2调用窗口，旧分配尚余45/56恢复位置，既有整数规则第2调用首输出。预算下界L=max(0,R+Kn−KB)要求延后45/56peer机会，分支恰好达到；总新输出58→14、56→1，两臂scheduled位置均2048。真实baseline两调用映射与raw吻合；候选独立C/A演进但allocator/token为替身，不模拟DMA、未来EOS/准入或上层swap，不能称少调用加速。共享两条复算命令已通过，原结果保留。E/probe_cold_recovery_share.py新增；原执行探针只加L解释字段并对terminal episode明确拒绝跨段取样，不改G。

研究决定：旧固定域无份额动作不能外推D；首输出前整数份额重新有条件候选空间，但尚无墙钟/完整服务/新颖性支持。首输出后的completion-bridge仍暂停，不以本新结果复活。唯一接续复用G已接受诊断和轻量主结果，判断更早启动后是否还存在这种冷恢复债务及其peer代价，再决定一项同强底座份额对照；本方不接管GPU/回读/主分析或增加实验组。最后读取CURRENT时G为原执行方RUNNING、诊断资格已报通过，canonical主结果尚无；该清单不作为本方live进程验证。


### 2026-09-15 主研究闭合G：简单eager有实际尾部价值，停止增加机制

O/20260915_natural_recovery_cadence_r01/RESULTS.md为唯一原执行方主分析，root复用未重算；7格448/448（64诊断＋384轻量，同64文档），接受f1a3bda1…727027。controller23893/shell23894，1789417112.862—1789417603.731；唯一归档25410e86…4feb8b/27,281,847B，241文件＋30payload回读核验，2026-09-14T20:28:14Z整机窗口明确释放。54轮转/54store/58load及43次sub20成功commit证明动作真实，不是收益分数。

两配对eager/current maxgap−63.43%/−53.12%、吞吐+1.066%/−0.706%、均完成−1.473%/+1.443%，每对64请求长度与stop类型全相同、完整序列同42/39条。仅27/28条自身gap好、37/36坏；完成好46/13条。p90/p95改善、最大4.065/2.860→1.487/1.341s，非全分布支配或显著性声明。补齐native-full无额外轮转系统参照max9.017/7.515s，median/p90更低，第二格少1966输出/均完成与TTFT更好，不作等生成量加速或cooldown归因。全部必要控制/存取在wall内；host分配均16GiB，eager末有效6.42/7.87GiB、current3.23/4.71GiB、nativefull16GiB，非过程中峰值；父184GiB非独立硬预算。EOS请求自身仍无抢占，恢复期EOS未覆盖。

模型方G_NATURAL_CADENCE_PREDICTION_CHECK支持必要候选→实际动作→可见停顿改进；peer风险不等于必然总体损失，peer与load主因未区分，不用F分账回填G。root接受selected/eager为本域停顿优先简单基线；保留current/nativefull的其它目标优势，停止第三冷却值、窗口、headroom与增长predictor。commit重检保留基线完善候选，不作为新颖性或当下主实验。

下一唯一高信息工作转为独立输入/较长到达episode：拟128未使用完整文章、同0.2s/4096GPU块/16GiBhost/EOS允许cap1024，native/current/eager反序6轻量（最多768请求），复用既有组合资格；无动作、无恢复EOS不加压或重采样。问题是已测简单取舍能否在host历史更替下迁移，而非新模块。当前仅提议，输入溯源和封包尚未做、GPU UNRUN；不是稳态或论文GO。主问题与长期goal保持OPEN。


### 2026-09-15 C在G eager复现冷恢复边界，交付默认关闭计算钩子

X=O/20260915_recovery_execution_share_r01/natural_eager_share，analysis.json/native_branches.json/hook_branches.json及同REPORT末节。复用G原诊断，主分析/回读不重做、新GPU样本0。54/54主动恢复首输出，52次待执行2–34位置/单计算，2次冷恢复3003/3026位置/四计算；6个resident模型map全部吻合。两处份额差异来自同一5161的不同恢复，较D5122多一个目标但同文档集、策略内生轨迹，非独立输入验证。step740 R1011/n28/K1、824 R2029/n27/K2：preserve_calls首输出在1/2调用，ready_first同期尚欠15/35位置；peer机会代价下界15/35且分支达到，总新输出28→14/54→20、scheduled总位置1024/2048不变。排除eager后计算份额永远无动作的解释，不把少调用写成节省整个混合调用或服务增益。

E/recovery_compute_share.py约80行、默认关闭；复用同Request/State和整数规则，只对上层已选RUNNING/队尾冷恢复目标追加必要peer hold，保留原KV guard、保存/加载/target/victim/保护终点。G两前态与D两前态均经原生CPU schedule钩子执行吻合，allocator/输出/base为限定替身，无DMA/EOS/完整组合GPU资格。3个回调契约检查及共享相对路径复算通过；G runner未接入、接受包未改，native_full_native无hooks不适用。静态接口核对覆盖先base安装/先C卸载与首输出/terminal释放，未增加审计层。

主方G selected/eager停顿优先基线及current/nativefull其它目标优势直接继承，完整服务CPU替代收益、秒级Oracle、恢复期EOS与新颖性未测。主方已选独立128文档/更长到达episode迁移，C不追加份额GPU组/控制器或要求额外诊断，工具保留条件备用：先看新独立范围是否仍有对固定停顿目标有影响的冷恢复冲突，再决定唯一on/off资格。无冲突不降阈值或加压；首输出后completion-bridge继续暂停。


### 2026-09-15 C补齐restore_first：首resident起点决定组件比较边界

O/20260915_recovery_execution_share_r01/REPORT.md末节；natural_cold_share与natural_eager_share各新增simple_baseline_branches.json及resident_baseline_branches.json。E/probe_cold_recovery_share.py仅增加已有restore_first第三分支和first_resident入口选择，不新建controller或GPU组。四个旧map-change前态PC/RF的目标首输出、各请求进度/持块计数完全相同，但这只是晚期窗口；G实际最早可接入为739而非740，晚分支会漏掉RF早期扣留peer的成本。因此正式比较包含D/G全部首个已核实resident重算起点，D25保护中4次、G54中2次，其余单次加载恢复保留空选择，不按候选结果筛选。

D637/730/840/970、G739/824三规则独立执行native CPU schedule，目标PC/RF均在K=2/2/3/3/2/2首输出。同K窗口peer总量ready/PC/RF依次58/13/13、56/0/0、78/78/26、75/75/25、56/41/28、54/19/19。D840/970 PC与原ready完全相同，多52/50peer机会不可算新模块收益，且比RF各多持2KV块；G739 PC比RF多13、比ready仍少15。D637/G824只是将13/19peer从末次提前首次，D730完全相同。RF已覆盖最少调用，但未普遍覆盖恢复期peer服务；四晚期状态不能代替完整组件域。源码/原件hash与全部request终态在JSON，共享便携命令复算selection/cases全一致。

简单条件S=KB−R：peer持续可行且无完成/准入/外部释放时，RF最多min(n,S)，跨调用PC最多min(Kn,S)；实际KV仍须逐调用验证，不能由同K/同位置或多输出推墙钟/数值KV等价。CPU allocator/输出/base替身、不含真实DMA/EOS/完整服务，旧raw不改、新GPU样本0。投入决定保留RF为必要执行组件对照、不用它替换most余额基线；整数最少调用保证不作为独立贡献，默认关闭模块仍无完整服务增量。唯一主接续仍独立文章/较长到达迁移，不因本分析追加三臂GPU；恢复期EOS和性能Oracle未验证，首输出后bridge继续暂停。


### 2026-09-15 C执行组件阶段交接，停止无新轨迹轮询

C会话01a0953f-560d-7c32-b6eb-79ea17e1f786已连续三轮确认同一外部数据依赖：H已具备128新文章输入与封包，但无本地RESULTS、execution/readback/results或主分析；CURRENT仍为G完成，C未获执行身份。83de4adb的首resident六前态对照与85b6322e默认关闭钩子已同步共享入口，现有CPU必要工作完成；这三轮状态检查不是科学进展。停止重复轮询，将本会话goal置blocked，不改变主问题或各机制证据判定。

接续触发：原唯一执行方交付H实际结果，或提供职责内新的执行状态/明确运行任务。H合同仅六轻量，详细KV/重算/加载观察关闭；后续只复用其可观测请求/EOS证据，不能用轻量字段臆测计算份额冲突，也不追加未经主方选择的诊断。当前服务基线仍selected/eager，份额组件GPU/完整服务收益未验证，首输出后bridge继续暂停。已有模型、代码、全部请求损益与复算入口见O/20260915_recovery_execution_share_r01/REPORT.md末节；不重复主方封包、执行、回读和主分析，无本方GPU或后台任务。


### 2026-09-15 H独立输入首次启动：零测量资格遗漏，保留失败并最小修复

O/20260915_natural_cadence_holdout_r01/RESULTS.md及唯一执行方原件已交付。已冻结128新完整文章，排除224旧train文档，460–3064tokens/0.2s/25.4s；包f6bd54ba…c3ff58/34payload。首个native参考在warmup/capture前被safe_static旧requests==64断言拒绝，实际H为128，原件error为open episode upper bounds differ。实际4096usable/free＋null、GPU KV8592031744B、host17179869184B/8192块和full-history均符合，非资源不足/策略负结果。测量0、warmup0，其余5格UNRUN，预声明3%吞吐/5%均完成预算NOT_ASSESSABLE。

controller28488/shell28489，UTC21:06:42.646—21:07:12.948；唯一归档e76af1d1…821824/3592960B，67文件＋34payload回读验证；UTC21:08:48.471整机/共同锁明确释放。原件与首次包不修改。原执行方仅准备独立r02：相同输入/六格/资源/计量/事前评价，资格常量64→128，保留真实pool/layout/null/APC/bytes/初始历史检查；对实际函数加限定engine替身的128通过/错人数拒绝检查。新SHA尚未接受，GPU未再启动。

近邻方另外完成CPU_INTERFACE_READY原型cpu_ltr_style_selected_r01，明确LTRCounters与selected保存后端的接入和必须替换的策略门；它不是已组合native/GPU或完整LTR。该基线兼容工作不混入H，不用近邻尚未适配为候选宣称创新。


### 2026-09-15 H独立输入闭环：两对通过事前预算，幅度减弱，转向兼容近邻

O/20260915_natural_cadence_holdout_r02/RESULTS.md为唯一执行方主分析，root直接复用；新128完整train文章排除224旧文档，固定460–3064tokens、0.2s/25.4s到达、EOS允许/cap1024。六轻量格768/768完成，不是768独立文章。current/eager仅cooldown20/0不同，同4096usable+null GPU KV8592031744B、host16GiB；native full无额外轮转为单独系统参照。复用G路径资格，无新诊断。r02接受b7360986…d520f/35payload，仅修r01资格人数64→128；r01零测量失败原件完整保留，非策略NO_GO。

controller29201/shell29202于1789420535.577—1789421203.827完成；归档33e88e55…61e40a/19995548B，220文件和35payload全部回读核验；2026-09-14T21:27:40.122Z共同锁及两GPU明确释放。主分析dac67aa1…bdcddb96、RESULTS d1743121…c46f，全部六格comparable，无遗漏/失败测量请求。

eager/current两对maxgap−13.596%/−3.124%、实际输出rate+5.439%/+0.073%、均完成−6.263%/+0.492%，满足每对事前3%吞吐损失/5%均完成增幅预算且gap下降。仅为本次两对合同通过，分析标签STABLE_TRADEOFF_CONFIRMED_IN_H不提升为统计稳定或通用保证。第二对绝对gap仅少0.121s；相对G幅度明显减弱，停止为小差无限增加重复。各对少541/560输出，长度同127/126条、完整序列同65/73，非等工作量/质量证明。每请求gap改善87/77、恶化41/51；均完成改善113/35、恶化15/93，保留全部逐请求代价。

原生full/no-extra-rotation vs eager：吞吐+4.31%/+3.44%、均完成−7.05%/−5.73%，107/106请求自身gap更低，但最大11.835/12.011s vs eager2.936/3.760s。最强底座取决于目标，不能用最长停顿一项抹去原生效率/分布优势。两对eager/current预算不追溯换成相对native的胜负条件。

六格末态host有效8192块/16GiB，pending store/load/ack为0；endpoint满不证明过程中live-history替换。HWM18.53–18.55GiB含init/warmup且与KV重叠，父184GiB不是独立进程树预算。必要控制/保存/传输已计入完整服务，drain另列。每格stop6或7请求，其自身成功preempt均0，恢复期EOS仍未覆盖；有限到达、质量和完整内部时间分解/性能Oracle未验证。稀疏事件中短服务真实存在，但不以此推出可避免的完整服务损失。

模型方O/20260915_joint_growth_decision_r01/H_r02_PREDICTION_CHECK.md核验事前late64：early最大gap−45.876%/−20.215%，late−13.596%/−3.124%，两对方向支持；全局最大值恰为late，不增加独立样本。输入与状态共同变化，不归因host-turnover/长度或peer/load。root生成H/paper_view/transfer_budget.pdf及SVG/PNG，仅从G/H唯一主分析作图，非新数据，已目视核验。

研究决策：G/H已测域保留selected/eager停顿优先简单基线，同时保留原生完整保存效率前沿；停止第三冷却值、窗口/headroom/增长predictor。新增证据关闭独立输入上的方向迁移，但不建立新算法贡献。LTR-style原147行CPU接口已完成量子/等待/ID不变性检查，非完整LTR、非native/GPU。唯一下一工作转为兼容原生connector的互斥adapter接入与生命周期资格，随后再接受合理校准后的同保存/预算/目标服务比较。无需预设已有算法bug或全指标胜出。C冷恢复份额仅CPU条件证据，暂不追加GPU。CURRENT/PAPER/STUDY同步；无已接受下一GPU包，长期goal保持OPEN。


### 2026-09-15 LTR-style接入前纠正移植HOL，明确策略与共同后端

O/20260914_ltr_component_probe_r01/cpu_ltr_style_selected_r01/ACTIVE_TARGET_ACTION_SPACE_ADDENDUM.md给出三请求CPU反例：两boosted PREEMPTED分别需10/2块，free0、唯一普通peer持2块；旧原型先锁住早到不可筹措target，持续DEFER，后者虽可单换恢复却不再被扫描。该缺陷属于自定义single-target移植，非官方LTR或GPU性能负结果；早先147行文案勘误，当前已核原型134行/SHA371c6eb5…99fd。原型和反例保持原件，新20260915_natural_ltr_style_component_r01由原执行方实现私有修复：扫描首个可执行boosted目标、接受intent后锁定；busy/不可行不清计数，不加轮转阈值。

root确定同资源、同selected native-save/单换合法能力下比较完整恢复策略，不声称仅trigger/quantum单因素归因。G的residency/progress/max_absences是政策门，不强制进LTR；共同保留ownership、足够full-history、pure-decode非prepare终止、实际store/flush及async-load安全。most_output为明确的移植选择，非完整LTR。固定官方源码984–996和1396–1452显示全局优先级/多请求/多victim，当前不冒充原系统。论文原max_waiting还含TTFT，本研究继续并列计费。

执行接入发现量子可跨首输出后触及下一KV块增长。root要求保留boost优先顺序，允许原生allocator在必要时自然抢占peer并计其代价；不可兑现的自定义reserve/peer-hold不应制造异常终止或直接撤去合法优先级。pending load不执行/不扣Q，真正无分配记录backend censored；target若自然抢占则释放active latch并保留counter参与重选。该CPU/native资格尚在实现，未宣布可执行/GPU通过。无新接受包或GPU运行，H完整终态不变。


### 2026-09-15 LTR-style CPU 接入闭合，删非必要 prepare 预筛；r02 接受但 GPU 未运行

唯一执行方已完成互斥 native adapter，原生调度循环和实际 adapter closures 的 CPU 检查覆盖可执行目标选择、零分配不扣量子、首输出后量子继续、KV/序列槽筹措、自然抢占、保存提交及异步队列边界。allocator、输出和 connector 注册/flush 均有替身；实际 tensor 传输、native load 完成、安装期类型与恢复期 EOS 未验证。不把这些检查写成 native GPU 结果。

模型方 O/20260915_joint_growth_decision_r01/I_PREPARE_SUM_GUARD_ADDENDUM.md 的同前态原生循环反例：free0、victim 持2且下一步无需增长、peer 持1且需增1、target 需2。保留全体增长预筛时 call0 拒绝 prepare，native 仍抢占 peer；只删该预筛时同样抢占 peer，却注册 store 并在下一 call 获得实际 commit READY。保留分支下一 call 才 prepare。提前分支保存前缀16而非32，并多出 victim 抢占及 CPU 重算17位置，不能外推完整服务净收益。结论只是否定该预筛作为共同后端必要条件；保留实际 ownership/store/commit/flush 检查，无新控制器或额外参数扫描。

O/20260915_natural_ltr_style_component_r01 接受包4fc166e8…709837/30文件保持原件。首次26862连接及随后两入口各一次有界诊断均在SSH认证前被关闭：TCP可达，但未收到server version、未到认证、未提交密码，无远端命令/上传/锁/GPU初始化。零测量、单格UNRUN；两GPU/环境/共同锁均UNKNOWN，不推断实例停止或GPU空闲。EXPERIMENT_STOPPED.json保存旧CURRENT终态，连接证据在execution_weste_26862/connection-diagnostic.json。

2026-09-14T22:22:34.860172+00:00 root接受 O/20260915_natural_ltr_style_component_r02：6888c3188d4555b61ce9295d91735da98e81bc46c6a910b282bd0be7ec4c9e73，30 payload，本地和包内逐文件已核验。唯一runtime差分删sum预筛；G输入/公共计量/实际物理合同/其余策略源码不变。r01 CPU receipt a4e9b095…bc0f93直接复用，明确不是r02重跑；新增反例结果88449ead…da84d，源码0c467d28…1919，改后adapter编译通过。r02零连接尝试，ACCEPTED_GPU_UNRUN_RESOURCE_UNKNOWN；无controller、PID、资源占用或后台执行。

当前最强参照仍为G/H selected/eager的停顿取舍以及原生full的效率前沿。唯一未决决策：同后端上合理校准的兼容LTR-style是否覆盖该取舍，而不是先假设新算法必要。模型已核G2650次引擎调用与调度一一对应，阈值单位为调用而非固定墙钟。60个正分配之间的无分配段有55个达到30、0个达到200；未证明全是PREEMPTED，且受既有策略动作截断，不能当LTR反事实。58次真实加载首输出均需1次正分配，两个无加载事件需4次；后两次为同一请求，不是独立文档。

root事前固定T{30,200}×Q{1,10}至多四点，以同窗口eager作发展参照；吞吐≥97%且均完成≤105%才按maxgap选，平手依吞吐、均完成、低T、低Q，不合格不扩网格。预测固定Q10时T30比T200轮转更多且maxgap更小，频率和服务分开核验；未承诺效率方向。H不用于LTR参数选择但已见过，不称盲测。当前只接受r02单格G64/T30/Q10、180s诊断；性能组未接受。入口恢复并核实共享资源及固定源码后由prepare_start_contrast唯一执行。PAPER/STUDY/CURRENT同步，科学状态仍MEASUREMENT_ONLY，长期goal未完成。
