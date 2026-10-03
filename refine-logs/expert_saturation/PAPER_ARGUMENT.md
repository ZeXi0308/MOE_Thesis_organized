# 论文论证｜合法恢复、普通 backfilling 与完整服务代价

**问题与状态。** 给定 GPU 和明确 host 预算，怎样安排恢复及后续执行，在控制生成长停顿时改善完整服务效率？现有证据为 OLMoE / vLLM 0.26 的 `NATIVE_SERVING / MEASUREMENT_ONLY`。主问题 OPEN；尚未形成充分验证的独立方法或 MoE 专属贡献。

**当前断点（2026-10-01 native full-save 无主动轮转对照未过成本条件）。** 同后端原生参考与ordinary-only在原22937同一锁内完成256/256请求，新臂23条实际ASYNC_LOAD→新输出→完成，主动forced rotation=0、无动作同step额外抢占。native/new率1511.666/1422.598 token/s（94.108%），mean flow35.924/37.801s（105.225%），maxgap13.101→11.844s；预定97%率和105%flow条件均未过，不能以tail改善宣称联合收益。gap69改善59恶化，flow9改善119恶化，TTFT18改善110恶化，goodput5点更好15更差，63序列/2stop不同；native抢占51→64。七个完整打印区间load15.066→17.329GB，首尾未知、CUDAcopy非请求延迟；两臂正式期cache写入0不证明零JIT。见[完整结果](experiments/admission_capacity/20260929_commit_recheck/A_NATIVE_BACKFILL_ONLY_PAIR_RESULT_R01_20261001.json)。该已见输入顺序块保留真实可执行动作，但本精确formulation没有通过效率预算；不扩seed/阈值。当前分析目标/队首实际代价及未隔离的策略CPU开销，再决定唯一下一诊断。独立方法仍缺，OPEN / NOT READY。

**此前普通 backfilling 对照（2026-10-01 普通 backfilling 覆盖原目标）。** 新机器同一锁内完成 Q1 / ordinary backfill / primary-first 三臂，384/384 全部完成，44/19 条真实恢复输出链。两候选各自通过相对本组 Q1 的原冻结判据。ordinary/primary-first 的率为1532.668/1545.904 token/s、mean flow36.412/36.932s、maxgap1.894/1.711s；ordinary 在全部20个固定 goodput 点更高、95/128请求gap更小，primary-first 仍保留更低单个最大gap和较高率，不是全指标支配。ordinary有36/44动作发生在同一步原目标首输出事件外，故触发差异确实执行。结论：本运行块不支持把原目标首输出门禁作为独立贡献或达标必要条件；采用普通backfill为更强简单对照。完整抢占为348/496/433，七个已打印完整区间load111.734/164.098/142.684GB，正式期cache写入32/16/32文件；输出/stop变化保留。输入现在已见，一次顺序开发块，不跨两卡拼性能或称统计确认。见[三臂完整结果](experiments/admission_capacity/20260929_commit_recheck/A_WAITER_BACKFILL_TRIPLET_RESULT_R01_20261001.md)。主问题OPEN，NOT READY；下一小问题检查能否在native full-save上执行ordinary合法backfill而关闭主动强制轮转，分清backfill收益与轮转代价，尚未运行。

**此前冻结新输入检验（2026-10-01 冻结新输入通过）。** spare-followup在已见输入off/on和反序on/off两块分别通过原判据后，策略固定，使用排除旧已物化输入后的首128篇源顺序完整文章。新off/on/native三格384/384完成，18条真实followup恢复→输出→完成链，原目标输出在先、无递归或同一步额外抢占。off/on输出率1472.515/1463.713 token/s（99.402%），mean flow39.216/39.524s（100.785%），maxgap2.387→1.844s，全部原判据通过。gap88改善40恶化、flow39改善89恶化、TTFT30改善98恶化；抢占359→465，62输出序列与3停止原因不同。原生full系统率1497.949、flow35.977s、maxgap12.586s；on相对native flow109.858%，不能改用native分母宣称原5%预算通过。见[新输入原件分析](experiments/admission_capacity/20260929_commit_recheck/A_SPARE_FOLLOWUP_FRESH_RESULT_R01_20261001.json)。这是一个固定运行域中的简单参照取舍，已补齐一次冻结后新输入检验；独立机制/近邻完整基线/跨运行域与不确定性仍缺，NOT READY。下一步从实际请求代价定位强简单参照仍未解决的自然状态，不追加确认seed或重扫旧参数。

**历史断点（G64/H128与H1）。** 修复版G64原定四点均完成，唯一预算合格T200/Q1已冻结；其最大gap3.289s差于同组eager1.598s，43/64请求自身gap改善、21/64恶化。Q10事前预测的“低T更多轮转”成立（44比10），而“低T更小最大gap”未获支持（7.256比4.017s）。H128 native/eager/冻结T200Q1已同机完成，LTR输出率91.15%/mean flow111.35%，最大gap3.963s比eager3.773s，迁移预算失败；H已见输入不作盲测。H1随后已通过实际动作资格：269提交检查中2次直接async准入，匹配加载完成且有后续新输出，128/128完整完成；两目标均length终止。首个同输入off/on配对随后完成，数值达标但on的254次提交全部回退、direct为0，不能归因直接恢复收益；反序组随后完成：on有5次实际direct，输出率100.78%、mean flow104.91%、maxgap3.499比2.743s；两块共同判据未通过，停止此版H1性能扩展。详细来源、自然输出变化和未获支持的判据见[最新校准报告](experiments/admission_capacity/20260929_commit_recheck/A_G64_GUARDED_CALIBRATION_20260930.md)。以下2026-09-15 G/H叙述属于历史运行域，不与新机器拼成同组比较。

**解释与最小选择。** 保存状态、开始恢复、返回新输出和后续服务是不同环节。真实保存是共同底座；在已测 G/H 自然输入域，保留 selected 保存＋global cooldown 0（eager）为停顿优先简单参照，并保留原生完整保存的效率优势。停止第三个冷却值、窗口、headroom 和增长 predictor。复杂选择器必须提供强简单基线之外的实际决策价值。

**保存闭环。** [旧定长保存对照](outputs/admission_capacity/20260915_repeated_kv_service_r01/analysis/REPORT.md)有完整请求收益，旧两格仅作原合同资格。[自然完整保存资格](outputs/admission_capacity/20260915_natural_native_full_gate_r01/RESULTS.md)验证真实增量保存；16 次再抢占的历史有 99.667% 随后从 host 复用，不能把再抢占按整段工作丢失计价。[同调度 F 四格](outputs/admission_capacity/20260915_natural_save_scope_timing_r01/RESULTS.md)256/256 完成，full 相对 selected 最大 gap 两对恶化 13.19% / 63.96%，吞吐和平均完成变号。因此 full 保留为强对照，没有升级为全面更好的底座；少重算不能替代完整服务效果。

**决定性结果。** [G](outputs/admission_capacity/20260915_natural_recovery_cadence_r01/RESULTS.md)完成 64 诊断＋384 轻量请求；诊断验证真实 store/load、43 次小于 20 步的轮转间隔及加载后新输出。[H 独立输入](outputs/admission_capacity/20260915_natural_cadence_holdout_r02/RESULTS.md)排除 224 篇既有 train 文章，固定 128 篇新完整文章、460–3064 输入 tokens、0.2 秒到达及 25.4 秒到达跨度，六轻量格 768/768 完成。每组相同 4096 可用 GPU KV 块＋null、实际 GPU KV 8,592,031,744 B、host KV 16 GiB；current/eager 只改变 cooldown 20/0，原生 full 无额外轮转是独立系统参考。

| eager 相对 selected/current | G 配对 0 / 1 | H 配对 0 / 1 |
|---|---:|---:|
| 最大生成 gap | −63.43% / −53.12% | −13.60% / −3.12% |
| 实际输出吞吐 | +1.07% / −0.71% | +5.44% / +0.07% |
| 平均完成时间 | −1.47% / +1.44% | −6.26% / +0.49% |

H 两对均满足事前选定的吞吐损失≤3%、平均完成增幅≤5%，同时最大 gap 下降。它确认本次独立输入上的预算取舍，不是统计稳定、噪声界或业务 SLO；第二对仅减少 0.121 秒。[事前 late64 预测](outputs/admission_capacity/20260915_joint_growth_decision_r01/H_r02_PREDICTION_CHECK.md)方向也获支持，但全局最大值恰在 late 组，不增加独立样本。收益显著减弱，不能归因于 host 历史更替或某个内部等待。[完整取舍图](outputs/admission_capacity/20260915_natural_cadence_holdout_r02/paper_view/transfer_budget.pdf)保留 G/H 全部配对。

**必要代价与边界。** H eager 最大 gap 2.936 / 3.760 秒，原生 full 为 11.835 / 12.011 秒；但原生输出吞吐高 4.31% / 3.44%、平均完成快 7.05% / 5.73%，107 / 106 个请求自身 gap 更低。改善最长停顿伴随停顿分布和效率取舍，不是全指标支配。H eager 比 current 少 541 / 560 个输出，不能称等工作量加速；自由生成质量未测。所有 EOS 结束请求自身均未被抢占，恢复期未知 EOS 仍未覆盖；有限到达不是稳态服务。必要保存/加载/调度成本包含在测量中，drain 单列；各臂末态 host 有效 16 GiB 不证明 live-history 替换。进程 HWM 包含初始化且与 KV 重叠，父 184 GiB 限额不是独立进程树预算。轻量数据未给出完整内部时间分解或性能 Oracle。

**贡献缺口与唯一下一工作。** H 将“简单启动节奏只在选定输入有效”推进为“独立输入上仍满足预声明取舍，但幅度缩小”，没有因此产生新算法贡献。[直接近邻](experiments/admission_capacity/RELATED_WORK_RECOVERY_20260914.md)已覆盖等待提权与服务量控制；剩余决策是：相同保存、预算和目标下，经合理校准的兼容 LTR-style 是否覆盖该取舍？互斥原生 adapter 的 CPU 接入已完成；定向反例纠正提前锁住不可行 target，以及[全体增长预筛](outputs/admission_capacity/20260915_joint_growth_decision_r01/I_PREPARE_SUM_GUARD_ADDENDUM.md)：后者推迟合法 prepare，却没有避免同一 peer 被抢占。修复进入共同可比的基线，不归因于官方 LTR，也不算服务收益。新 G64 诊断已观察到实际 GPU selected 保存、加载、新输出及后续量子服务，见下文；同机完整请求性能及独立方法价值仍未验证。它不是完整 LTR。停止扩展当前冷却/窗口/增长预测器，冷恢复份额候选不追加 GPU 组。

[有界校准选择](outputs/admission_capacity/20260915_joint_growth_decision_r01/LTR_G_CALIBRATION_DECISION.md)已固定为 T∈{30,200}、Q∈{1,10}，以同窗口 eager 为参照；仅在吞吐不低于 97%、平均完成不高于 105% 的合法完整候选中选择最小最大 gap。没有合格点就保留负结果，不扩网格。预测为固定 Q10 时 T30 比 T200 轮转更多且最大 gap 更低，两项分别检验。G 用于开发；H 不用于该参数选择，但已经见过，不能称盲测。性能执行组尚未接受。

**执行与接续。** H r01 零测量资格失败保留，H r02 已完成归档且当时明确释放 GPU。I r01 的旧端点连接失败。已接受 [I r02](outputs/admission_capacity/20260915_natural_ltr_style_component_r02/README.md)，包 6888c318…4c9e73 / 30 文件；在新授权的单 RTX 5090 上，A 已创建私有固定环境，并核验固定 OLMoE revision 的元数据和三个权重分片。首次 G64/T30/Q10 预启动发现 A 私有 reflink 副本短于源文件，[失败收据](experiments/admission_capacity/20260929_commit_recheck/A_G64_PRELAUNCH_FAILURE_20260930.md)保留，`cells=[]`，没有 GPU 请求；它不是机制负结果。随后系统盘完整复制的新身份通过全部模型哈希和离线解析，完成唯一 G64 原生诊断，[资格结果](experiments/admission_capacity/20260929_commit_recheck/A_G64_NATIVE_LIFECYCLE_RESULT_20260930.md)与归档另列。后续工作转向同机 G64 完整服务对照；[统一清单](CURRENT_EXPERIMENT.json)保留版本、资源与断点。

## 当前主张、证据与缺口

| 可写入草稿的主张 | 证据 | 限制/缺口 |
|---|---|---|
| 当前原生资源预留与adapter准入不一致可形成循环等待 | 新诊断F86/N86/R132、27 running全held、662空调度；guard资格与完整校准已完成 | 是移植实现正确性修复，非新调度算法；旧失败未记录精确分配分支 |
| 在固定效率预算下，事前有限LTR-style网格只有一个合格点 | 修复G64四点完整cohort、各同组eager；T200Q1冻结 | 单块开发校准，自然输出不同；不是完整官方LTR复现 |
| 该冻结点没有在已见H128通过服务预算或改善最坏gap | 三格128/128、完整原始记录；91.15%率/111.35%flow、gap3.963比3.773s | 多数请求自身gap改善仍保留，无统计稳定性或对所有运行域的否定 |
| 提交时可在不驱逐计划victim的情况下实际接纳恢复目标 | [H1资格](experiments/admission_capacity/20260929_commit_recheck/A_H1_NATIVE_DIRECT_QUALIFICATION_RESULT_20261001.md)：2条async准入/load/输出链、128完整请求 | 只占269次检查中的2次；两目标均length。off/on、on/off均已完成但共同停顿判据未通过；同前态反事实、质量与方法新颖性未成立 |

完整论文仍缺强简单规则之外的独立机制、相应动作价值与冻结后新数据确认。本表不将实现修复、一次正负结果或图表数量作为可投稿判据。

## 2026-09-29 论文论证增补：提交时的动作边界

[H1 的 CPU 反例](experiments/admission_capacity/20260929_commit_recheck/H1_TOKEN_BUDGET_ADDENDUM.md)表明：恢复目标在提交时有足够 GPU KV 和序列槽，仍可能因为 running 请求先消耗调度 token 预算而比驱逐一个 victim 的旧路径晚到达生成边界。在 31 个 pure-decode peer、32 槽、1024 token 预算、无 host 命中目标需 994 token 的构造前态中，H1 direct 与 off 分别给目标 993 / 994 token。这是实际 H1 gate 加原生调度顺序的条件 CPU 反例，尚非 GPU 输出时刻或完整请求代价。旧 D/E 派生快照没有目标精确 token 和 host 命中，因此不能替代原生测量。另一方面，已恢复的 G 原生诊断逐次状态显示 54 次 READY commit 全部 `free<need`，该已执行轨迹对 H1 的即时 KV 动作机会为 **0/54**；G/H 轻量格缺即时快照，机会仍只界于 `0..816`。本文后续不得称物理可直接恢复自动保证目标首新输出不晚；target、victim 和其他请求的实际输出与完成都须入账。

[独立静态包门禁](experiments/admission_capacity/20260929_commit_recheck/check_package_compatibility.py)确认已接受 H128 与 H1 候选的输入/公共后端相同；已接受 LTR r02 是 G64，只可先作原生生命周期资格，不可拿它与 H1 H128 直接排性能。LTR fair 的 CPU 新版与 r02 的 active-target hold/优先级语义不同，需要自己的原生资格及同输入封包。即使 H1 自然动作后出现正效果，提交时“空块与槽位足够便直恢”本身是强简单规则；独立方法贡献尚未成立，必须有其之外的状态/动作残差和冻结后新数据支持。

会议适配的官方核查见共享 `research_coord/C_DELIVERY.md`：SoCC 在 CCF B，2026 Full Research 截稿已过且未找到 2027 官方 CFP；CCGrid 在 CCF C，2027 主会技术轨 Track 3 接收 model serving / accelerator sharing，摘要 2026-11-24、全文 2026-12-01 AoE。该轨官网未直接使用 Full/Regular 字样，投稿资格须按正式 CFP 和 CCF 长文规则复核。当前证据仍 `CPU_COMPONENT + NATIVE_SERVING_MEASUREMENT_ONLY`，不足以标记可投稿。

## 2026-09-30 自然残留与近邻边界

[G/H 派生原件的只读复算](experiments/admission_capacity/20260929_commit_recheck/A_RECOVERY_20260930.md)显示 H eager 的 1–2 个新输出后再抢占段在两块为 5/8，G 为 1/0；它是真实现象，尚不足以支撑“普遍短恢复抖动”的动机。H block1 一个事后定位请求在再次抢占前仅 0.556 ms 刚输出，随后出现 3.698 s 新输出间隔；同文档 current/native 分别为 3.024/4.376 s，三者轨迹不同。此例不能用输出 age 已超阈值解释即将发生的停顿，也不能离线删除抢占来声称收益。两强臂仍有长停顿，但缺同前态资源与可计价的不同动作。最近邻 [Andes](https://arxiv.org/pdf/2404.16283) 已比较恢复收益和 peer QoE 代价，[TokenFlow](https://arxiv.org/pdf/2510.02758) 已按在线缓冲/消费状态安排恢复，[FastServe](https://arxiv.org/pdf/2305.05920) 已使用等待提权和量子；本线的独立方法主张仍需要这些规则之后的动作残差。下节记录 LTR 风格同后端真实生命周期资格；其性能与方法残留需另做完整服务对照。

## 2026-09-30 新机器原生资格与待测代价

在单 RTX 5090、固定 OLMoE/vLLM 0.26.0、G64 自然到达和相同 selected native 保存后端下，LTR-style T30/Q10 的 [原始会话与审计](experiments/admission_capacity/20260929_commit_recheck/A_G64_NATIVE_LIFECYCLE_RESULT_20260930.md)已独立归档并核 27/27 文件哈希。64/64 请求完成；53 次 READY 强制轮转中有 42 条 victim selected store、目标 native load、首次新输出及之后九次正调度并返回新输出的严格事件链，覆盖 17 个不同目标请求。其余 11 次目标没有本次恢复的 load job，走重算后仍服务；不能把它们写成加载失败。104 次 WAIT_LOAD 意图分为 52 次本步零调度且不扣 quantum、52 次实际正调度且扣一次；WAIT_LOAD 是调度前意图，不保证该次调用保持等待。

53 次强制轮转都需要 KV 资助并选中一个运行中的 victim。该 victim 从 READY 提交到下一新输出的中位间隔约 0.737 s，且常延续到目标量子结束后；这是当前策略下实际 peer 暴露，不是相对 eager 或 full 的因果增量。当前 42 条链是相关的事件 episode，诊断模式有额外详细观测，不能与旧机器 G/H 或未来轻量 arm 的吞吐/停顿直接排名。恢复目标自然 EOS 未出现，容器缺 `memory.peak`，独立 KV tensor-value 与逐 job 传输字节仍未知。下一步以[固定开发判据](outputs/admission_capacity/20260915_joint_growth_decision_r01/LTR_G_CALIBRATION_DECISION.md)在同机 G64 完整服务中比较 native full、selected/eager 与有限 LTR-style 点，报告目标与 peer 的完整请求后果；若简单策略覆盖 LTR，停止把量子和等待提权本身包装为新方法。H128 需 G 选点后才作已见输入上的迁移。

## 2026-09-30 同机 G64 首个完整服务对照

[严格审计的三格首组](experiments/admission_capacity/20260929_commit_recheck/A_G64_PERF_PILOT_RESULT_20260930.md)在同一 RTX 5090、固定 KV/host、OLMoE revision 与 G64 自然到达 cohort 下串行完成 native full、selected/eager、LTR-style T30/Q10，三格均为 64/64 完整请求。实际输出率分别 1474.96/1528.65/1494.66 token/s，平均完成 21.237/20.991/21.445 s，全体请求最大生成间隔 11.165/2.634/7.044 s。T30/Q10 相对 eager 满足事前 97% 输出率和 105% 平均完成预算，但最大 gap 更差；只占四点开发网格的一点，不能选点或声称 LTR 覆盖简单基线。固定 goodput 前沿有交叉：首 token≤20 s、gap≤2 s 的合格请求为 57/58/60，gap≤4 s 为 58/64/61。eager 与 T30/Q10 虽都返回 59,564 token，但 28/64 请求的 token 序列不同；这是完整自然生成服务测量，不能称等工作量加速或输出质量不变。单次顺序运行无统计稳定性；容器 `memory.peak` 未提供。其余三个冻结 G 点测完后才能根据原判据选择开发点，再把选定点迁移至已见的 H128。

## 2026-09-30 Q1 基线活性故障与校准中断

原冻结包的后续 T30/Q1 在固定 180 s 上限仅完成 6/64 请求，15.711 s 后全体无新输出；同一恢复目标自 step 833 起连续 104218 次被优先选择、原生未准入、释放后重新选择。[原件和严格审计](experiments/admission_capacity/20260929_commit_recheck/A_G64_PERF_R02_PARTIAL_AUDIT_20260930_V2.json)验证完整 64 到达和 58 未完成均保留，控制器归档后停止，T200 两点未运行。该结果确立已执行实现的活性失败，但轻量记录不含原生拒绝分支，不能作为 Q=1 prior-art 的固有性能负结论。校准选择与 H 迁移暂停推进，当前工作是有界原生诊断和共同基线修复；修复本身不算独立算法贡献。


条件源码反例已把一种失败路径缩小到两个合法性判据不一致：[CPU 复现](experiments/admission_capacity/20260929_commit_recheck/A_G64_PERF_R02_LIVENESS_COUNTEREXAMPLE_20260930.json)中，raw free 为 6 块、恢复目标需要 6 块、在途预填充还需要 1 块。适配器接受目标并阻止 peer 增长，而原生异步加载按 free−reserved 拒绝；重复选择不会改变状态。这是 pinned 源码上的可检验反例，尚未证明 r02 现场取值。[实际输出曲线](experiments/admission_capacity/20260929_commit_recheck/g64_q1_stall_figure/observed_output_plateau.svg)完整保留 164.29 秒无新输出尾部与仅 6 个完成请求。诊断首次非阻塞申请共享锁失败，未使用 GPU；现场拒绝原因仍待独立诊断。修复候选只补齐适配器的原生资源资格检查，须经 CPU 回归和真实服务验证后进入可比基线，不能计为本文新方法。


## 2026-09-30 原生准入循环的直接定位

[新诊断结果](experiments/admission_capacity/20260929_commit_recheck/A_G64_LIVENESS_RESULT_20260930.md)实际观测到 raw free=86、目标完整历史需要86、在途预填充保留132；原生完整历史检查通过后，实际异步加载需求85大于 free−reserved=−46，返回拒绝。同时27个运行请求全部被适配器hold，无待处理传输，连续662次全局空调度超过1秒。该实验直接支持资源保留冲突导致此实现循环；它不是旧性能轨迹的同前态反事实，不能补写旧运行未记录的现场值。修复候选在获得直接恢复target前检查目标与在途预填充可共同容纳，覆盖已激活目标，CPU八项回归通过。修复属于共同基线的合法性纠正；原型GPU资格尚未执行，首次资源锁申请忙。固定四点校准和H迁移在资格通过前保持未继续状态。


修复后的原生资格随后完成：[严格回读](experiments/admission_capacity/20260929_commit_recheck/A_G64_INFLIGHT_GUARD_QUAL_R01_AUDIT_20260930_V2.json)核对29/29文件与全部64请求。64/64完成，59564输出，58length/6stop；guard对22个不同step/target检查并拒绝3个，首次F147/N89/R136，原目标虽单独可装入但与在途保留不能共同容纳。无停滞快照或观察错误。本次资格仍带观察器，不用于性能排名；可将该修复纳入共同LTR基线继续原四点开发校准，不能以修复前后完成差异主张新调度算法贡献。

## 2026-09-30 修复后T30性能边界

[修复后独立性能组](experiments/admission_capacity/20260929_commit_recheck/A_G64_GUARDED_CALIBRATION_20260930.md)三格均64/64完成。T30Q1/eager/T30Q10实际输出率1441.894/1611.285/1500.548 token/s，mean flow23.822/19.563/21.111s，最大gap1.343/2.652/7.256s。Q1改善gap但效率超过事前97%输出率/105%mean flow预算，Q10也不合格。Q1/Q10与eager分别36/64、21/64输出序列不同；单次顺序组不能作等工作量或统计稳定性结论。T200余下两点尚待收据核验，未选参数、未迁移H。

## 2026-09-30 四点开发校准冻结

[完整修复后校准](experiments/admission_capacity/20260929_commit_recheck/A_G64_GUARDED_CALIBRATION_20260930.md)四点均64/64完成。唯一预算合格点T200/Q1为同组eager的101.37%实际输出率、98.91%mean flow，但最大gap3.289s差于eager1.598s；其它三点不合格。因此冻结T200/Q1作H128近邻迁移比较点，不能写成停顿改善或新方法。T200 eager有57length/7stop、两LTR为58/6，Q1与eager21/64输出序列不同；自然完整服务结果不是等工作量加速。四点皆单次块，无统计稳定性或盲确认。H输入已见。

## 2026-09-30 H128冻结参数迁移未通过

[同机H128结果](experiments/admission_capacity/20260929_commit_recheck/A_H128_GUARDED_TRANSFER_RESULT_20260930.md)三格native/eager/T200Q1均128/128，实际输出率1504.458/1435.104/1308.168 token/s、mean flow35.706/38.222/42.561s、最大gap13.346/3.773/3.963s。冻结LTR对eager未通过97%/105%预算，maxgap也未改善，停止T/Q扩展；但92请求自身gap改善、36恶化，不能抹掉分布取舍。输出序列不同、单个顺序块、已见H输入，无等工作量或统计稳定性结论，不否定完整官方LTR。保留eager停顿参照与native效率参照。下一检验原定H1在同H128是否存在实际direct动作；NO_ACTION即停止性能扩展，不制造负载。

## 2026-10-01 H1实际动作及前态资源来源

[H1资格](experiments/admission_capacity/20260929_commit_recheck/A_H1_NATIVE_DIRECT_QUALIFICATION_RESULT_20261001.md)在已见H128完成128/128请求，269次提交检查中2次直接进入匹配的native async load并随后输出、最终length完成；计划victim当步未被抢占。相邻原生调度揭示两次空闲都由前一步抢占另一个请求产生：229−4=225、217−4=213块。计划victim此前的178/6个新store block已经发生。因此可观察差异是避免随后计划中的第二次驱逐，不能说无抢占/无保存成本恢复。下一冻结同输入off/on、on/off性能对照检验完整效用；简单direct规则与防御性reservation门禁均不自动构成独立创新。

## 2026-10-01 H1轻量首组：数值收益与动作缺失

[首个off/on性能对照](experiments/admission_capacity/20260929_commit_recheck/A_H1_PERFORMANCE_PAIRED_RESULT_R01_20261001.md)各128/128完整完成。on/off实际输出率108.40%、mean flow91.08%、maxgap3.176比3.699s，数值通过开发判据；但on全部254次READY均空闲块不足、实际direct=0，没有H1直接恢复动作可归因。off为216次轮转，两格60请求输出序列和4请求终止原因不同，不能称等工作量加速。多数请求flow改善而52请求自身gap恶化；20点goodput更高仍含分母变化。初始化/预热差异在测量外，正式区间无JIT/告警日志不证明无波动。预定反序B2保留，不以首组有利决定是否运行，不改输入制造动作。

## 2026-10-01 H1反序完成与新的自然选择残留

[完整两块结果](experiments/admission_capacity/20260929_commit_recheck/A_H1_PERFORMANCE_PAIRED_RESULT_R01_20261001.md)共512/512完成。B1 on无direct，数值达标；B2 on有5次async direct，输出率100.78%、mean flow104.91%，maxgap3.499比off2.743s，flow仅8请求改善/120恶化，gap42改善/86恶化，goodput前沿10点提高/10降低。两块均通过的预定判据失败，结束此代码与H128运行域的性能扩展。B2有61个输出序列和2个终止原因不同，不作等工作量或单动作因果解释。

[新prepare计数](experiments/admission_capacity/20260929_commit_recheck/A_H1_PREPARE_FREEFIT_BOUND_R01_20261001.json)中，267次最终资源不足提交有78个prepare时刻存在其它数值可容纳的暂停请求，40个时刻满足原30步缺席条件；77个持续fit时刻的原生队首不fit且没有waiting获调度。它支持下一项小探索：在原轮转proposal处优先尝试实际可接纳的最长等待替代者，失败回原策略。该动作改变目标选择，与H1仅重新检查固定目标不同；它属于容量感知简单基线，独立新意和完整请求收益均未验证。

本线完整研究工作稿现见[paper_a/draft.tex](experiments/admission_capacity/20260929_commit_recheck/paper_a/draft.tex)，包含摘要、问题、假设、实现、完整结果表、相关工作与局限；不是投稿就绪稿。fit-first原生配对包已经上传，两个公共锁申请均忙、没有GPU初始化，仍为UNRUN。

Fit-first的最近动作关系已按[固定LTR原文§4.3](https://arxiv.org/html/2408.15792v1#S4.SS3)与[Andes原文§§4.1–4.3](https://arxiv.org/html/2404.16283v2#S4)复核：等待提权与容量约束下联合选择恢复/驱逐均已有研究。本候选只在原轮转边界改选最老可容纳目标，比全队列排序或净QoE模型更窄，不以此宣称新颖。当前off/on保持保存后端、触发、原victim回退、首输出保护和资源相同，只改变此选择；原生探索已获公共锁运行，完整效果尚未返回。

## 2026-10-01 Fit-first首次原生配对完成

[完整结果](experiments/admission_capacity/20260929_commit_recheck/A_FIT_FIRST_PAIR_RESULT_R01_20261001.md)两格256/256，47真实改选接纳与后续新输出，覆盖26目标。on率+2.35%、mean flow−5.05%，但maxgap3.795→12.186s；80请求自身gap改善、48恶化，固定goodput13点提高/7降低。强制轮转229→157，但全部抢占265→237，非强制残量36→80；不能把省72次强制轮转当全局省72次抢占。67输出序列/3终止原因不同，首次顺序配对不作等工作量或统计确认。当前无条件fit-first未通过预定停顿判据，停止性能扩展；下一项定位最长gap与1259次资金不足prepare拒绝之间的实际关系，并检查容量合格的其它victim是否被当前most-output单选遗漏。完整工作稿已更新，方法贡献仍未成立。

[固定诊断状态复算](experiments/admission_capacity/20260929_commit_recheck/A_H1_CAPACITY_ELIGIBLE_VICTIM_RESIDUAL_R01_20261001.json)进一步发现511/511次资金不足prepare拒绝都存在另一个满足原progress/residency/max_absences约束、且free+held足够的victim；首选决策全部重现。原策略先按most-output单选，再检查资金，因此漏掉简单的容量合格备选。下一工作是加强同后端简单基线，仅在原排名前过滤资金不足者，不改target等待优先级，不同时开启fit-first或H1。它不构成新算法；实际收益仍待原生对照。


## 2026-10-01 容量筛选基线首组结果

同一 H128 off/on 两格256/256完成。79次容量改选准备、78次真实新victim抢占→目标新输出、1次提交资源不足取消。率1428.855→1502.684、mean flow39.447→35.368、最大gap3.764→2.273；首组有利。prepare资金不足拒绝703→0，但总抢占230→350、强制192→302，必须同时报告代价。逐请求gap87改善41恶化，20点goodput均提高；66输出序列不同、0终止原因不同，总token124469/124470。结果`A_CAPACITY_VICTIM_PAIR_RESULT_R01_20261001.json`，原始会话`moe-a-capacity-victim-session-r01-20261001/`（均在20260929_commit_recheck目录）。容量可行性筛选是加强简单基线，不是新算法。固定原包与阈值的反序on/off块已准备，首次try-lock忙而未启动；不把首组当确认，不扩参数。


## 2026-10-01 容量筛选反序结果与唯一下一步

反序B2 on/off全部256/256完成，83真实容量替换均有后续输出；on/off率99.96%、mean flow105.09%、maxgap3.690/3.468s，两块共同判据失败。全抢占off/on268/363，强制231/324；prepare资金不足事件440/0，但零拒绝不等于零容量等待，性能记录没有selector不可行noop。64输出序列、1终止原因不同；保留首组正向结果，不宣称稳定收益。主报告`A_CAPACITY_VICTIM_TWO_PAIRS_RESULT_R01_20261001.md`（20260929_commit_recheck目录）。

新唯一实验为同capacity=true/q1策略first/second A/A，沿用相同输入/资源/预热/缓存流程，回答无策略变化时差异的量级。最近5个不同策略配对第二格meanflow都较低仅是线索，不能归因于顺序。A/A已启动，尚无结果。条件q10 CPU候选需首输出时未来增长预算可用且R=0，持续留出预算，否则回退；3fixture通过，未跑GPU。B2短服务19对中18有保护释放后再次native抢占，但不覆盖前三最大gap，所以不能拿它解释整体长尾。


## 2026-10-01 同策略 A/A：正式服务中的缓存不对称

相同 capacity=true/q1 两格256/256完成；first/second率1440.868/1455.358 token/s、mean flow39.689/36.666s、maxgap2.694/3.537s。第二格flow低7.62%但尾部更差；62输出序列不同、0终止原因不同。同策略也有相当幅度差异，单个策略配对不能直接归因，且不能据一次A/A估计总体方差或扣除其它配对效应。结果见`A_CAPACITY_IDENTICAL_PAIR_RESULT_R01_20261001.json`。

阶段定位`A_CAPACITY_IDENTICAL_PHASES_R01_20261001.json`：第一个token的最早内容分歧在1.176/0.844s，早于两格首次原生抢占6.383/6.457s；0–5s均26到达、2完成、0抢占，第二格已多978token。缓存元数据`A_CAPACITY_IDENTICAL_CACHE_PHASES_R01_20261001.json`显示first正式计时内64个Triton文件写入（2,867,515B），second为0；这与早期分歧重叠，尚非全部差异的因果解释。

唯一下一项：同策略暖缓存A/A，用已完成缓存的两个独立副本作相同起点，保持候选、输入及资源不变。检验正式阶段是否仍写编译缓存、0–5s输出差距及完整分布是否收敛；一组仍非统计确认。条件q10仅CPU就绪，未在GPU运行；本轮不以额外审计扩展门槛。


## 2026-10-01 新设备上的同缓存起点 A/A 完成

用户更新SSH端口后，原环境/模型/数据保留，但物理5090与容器已更换。r02两格在同一新GPU上完成256/256请求；first/second率1436.930/1412.287 token/s、mean flow37.857/38.596s、maxgap2.840/2.210s。59输出序列与1终止原因不同，总token125418/124444。前5s输出4333/4297，最早序列分叉在第39个token且早于首次抢占。两格正式阶段各有16个缓存文件写入（655329/822659B）；只是相同种子缓存的独立副本，不能称完全无JIT。该组描述同策略波动，旧冷缓存与新卡暖缓存不可作缓存因果比较，也不据一组估计总体方差。完整结果：`A_CAPACITY_WARM_IDENTICAL_PAIR_RESULT_R02_20261001.json`、`A_CAPACITY_WARM_IDENTICAL_PHASES_R02_20261001.json`，原始session已取回；541.365s，退出GPU EMPTY。

下一唯一机制实验为资源合格的Q1/Q10保护原生pilot，同新GPU/同H128/同capacity-victim，仅延长恢复后的有效输出保护；已完成second cache作为两格各自相同副本。Q10仅在首输出后已知R=0、纯decode、无pending queue、未来增长块可用时延长；状态改变回退。全部128请求、实际扩展→输出、释放后再次抢占和最终完成均入账。原固定探索预算rate≥97%/flow≤105%/maxgap下降保留，单配对不作确认。`CAPACITY_PROTECTION_PAIR_PLAN_R03_20261001.json`及新包已上传，首次try-lock忙（exec81781 exit75），未启动GPU；不扩A/A矩阵，不扫q。


### 2026-10-01 原生传输日志的可计量范围

`A_PRINTED_TRANSFER_INTERVALS_R01_20261001.json`及`analyze_printed_transfer_intervals_r01.py`补齐8格已打印区间的原生copy证据。每格正式阶段8次打印，首条可能混warmup而单列排除，后7个完整日志区间相加；vLLM每次打印后清零，不能拿末行当累计量或再次累加size_sum。CV B1 off/on的store为31.701/39.326 GB、load70.118/124.187 GB；B2分别store36.834/39.934、load92.164/121.289 GB。新卡同策略A/A first/second store38.372/39.307、load125.842/122.627 GB。这里GB为十进制，只覆盖已打印区间完成的实际搬运；未打印首尾未知，CUDA事件时间不含前置stream等待，也不能相加当请求暴露延迟。它补充容量筛选增加传输的观测代价，仍不是完整episode成本或单动作因果增量。


## 2026-10-01 Q1/Q10 原生保护探索失败及唯一定位实验

完整结果 `A_CAPACITY_PROTECTION_PAIR_RESULT_R03_20261001.json`：同新GPU、同H128、capacity=true，两格128/128完成；Q1/Q10输出123519/124468，率1491.394/1415.183 token/s（94.890%），mean flow37.340/38.116s（102.080%），maxgap2.243/2.529s；原定rate≥97%且maxgap下降未通过。60输出序列、1终止原因不同；flow51改善77恶化、gap54改善74恶化、TTFT27改善101恶化。实际206次延长均产出10个新token、8次未来增长不足退回首输出保护；全抢占375→263、短服务1–2输出后再次抢占22→11，局部现象改善未转化为完整服务收益。停止该Q10实现的性能扩展，不扫q、不自动追加反序块。

`A_CAPACITY_PROTECTION_EXPOSURE_R03_20261001.json`：206个不重叠延长区间共29.661s，其间其它请求仍输出42945token；不是独占GPU或可直接相加的损失。137个延长episode释放后仍有后续抢占，69个直至完成没有。最早内容分歧在1.587s，早于首次抢占；正式缓存写入16/0，因此不能作精确单动作归因或等工作量比较。`A_CAPACITY_PROTECTION_PRINTED_TRANSFERS_R03_20261001.json`：各7个完整打印区间的store39.548/35.316GB、load123.619/85.656GB，CUDA copy时间不等于请求暴露等待，首尾未覆盖仍unknown。

新的唯一问题：延长保护不仅保留目标未来增长，还沿用全局非目标waiting-loop break；是否实际阻挡了当时数值足够的其它等待请求？下一单格保持同Q10策略，只记录实际命中该break、held peers、空闲与目标增长/等待头需求及调度预算，避免全栈诊断。若没有真实命中的可容纳状态，不基于结构猜测放开准入；若存在，再检验一个保护范围更窄的实际动作。该诊断不与旧格做性能排名，也不将数值fit等同原生准入。完整贡献与新输入确认仍缺，未READY。


## 2026-10-01 Q10 全局保护范围：实际队列阻挡成立

同Q10稀疏原生诊断128/128完整完成，168.058s，exit0/GPU EMPTY。1755个延长步全部命中waiting break；174步涉及24个队首、22个保护目标，R0/无transfer job/无skipped queue/pending pushFalse，free足够队首完整历史及目标未来增长，余量5–354块。另123步有peer hold，共1029请求步。首个反例0008645需185块、free235、目标growth0，仍被保护分支挡住；数值可容纳不是原生准入证明。见`A_PROTECTION_SCOPE_DIAG_RESULT_R01_20261001.md`及完整rawsession。

唯一下一项为实际waiting边界的条件提前释放，未知/不足保留Q10；用Q1/条件释放Q10/普通Q10三格同时保留强简单参照和消融。真实原生接纳、首新输出、原目标及peer直到完成均入账。仍为简单基线延伸，无独立贡献主张，不扫q/seed，不把诊断时间与旧性能比较。计划`PROTECTION_YIELD_TRIPLET_PLAN_R01_20261001.json`尚未冻结/上传。


### 条件提前释放三格已固定、首次资源申请延后

`candidate_protection_yield_r01`在实际waiting-loop gate尝试提前释放：复用已有 `_direct_resume_reason` 的状态/slot/真实所有权检查，并要求R0、无注册transfer/未知push、full-history+目标future-growth有资金；其它情况保留原break。记录`protection_yield_to_ready_head`与`yield_head_admission`（包括NO_NATIVE_ADMISSION）。释放不因未接纳而回滚，不承诺10输出。四个实际closure CPU例通过，只是实现检查。

同包Q1/q10_yield/普通Q10顺序，计划`PROTECTION_YIELD_TRIPLET_PLAN_R01_20261001.json` SHA85560f9904fc29d1dde5098c5e9456b3e2e1ecc61a7ec34458713662c34b3022，包manifest b77bdf94bcea2b98fbb7e191a379ba199ad8fd782f50b7b1b26af521ab757503。已上传解压，exec38476首次flock EAGAIN/exit75，无测量session或A GPU作业。下次仅运行既有controller，不重解压，不改策略/输入。GPU未运行，不能声称接纳或性能收益。


## 2026-10-01 条件提前释放三格完成：实际动作成立，尾部判据失败

Q1 / 条件释放Q10 / 普通Q10三格均128/128完成，组445.759s、全部exit0/GPU EMPTY。率1419.121 / 1413.993 / 1404.383 token/s，mean flow37.704 / 38.153 / 38.041s，每请求最大gap的p95为1.772 / 2.857 / 1.924s，cohort maxgap2.038 / 3.656 / 2.313s。条件释放相对Q1率99.639%、meanflow101.189%，但停顿下降判据失败；不能因与普通Q10吞吐略高而替换Q1强参照。

25次提前释放均为真实ASYNC_LOAD_ADMITTED，并有后续新输出及最终完成；admission→新输出中位46.5ms。原保护对象全部完成，10个动作之后原对象还有后续抢占（相关事件，不是因果增量）。总抢占332 / 266 / 249，短1–2输出再次抢占18 / 10 / 12；局部减少没有转成尾部收益。条件释放相对Q1/普通Q10有57/65输出序列不同、均0终止原因差异，输出总数124464 / 124445 / 124464，不能称等工作量加速。

主结果`A_PROTECTION_YIELD_TRIPLET_RESULT_R01_20261001.json`；raw为`moe-a-protection-yield-session-r01-20261001/`。已打印7个完整区间store39.781 / 33.643 / 34.152GB、load115.402 / 87.323 / 85.788GB，首尾未知，CUDAcopy时间不等于暴露请求等待，见`A_PROTECTION_YIELD_PRINTED_TRANSFERS_R01_20261001.json`。停止此条件释放规则的性能扩展，不扫q/seed或自动追加反序块。下一步只从现有raw定位最差停顿的抢占→准入→新输出和原对象/peer代价，再决定有依据的实际动作；全论文仍未READY。


### 最差停顿定位及工作稿

`A_PROTECTION_YIELD_TRIPLET_TARGET_PEER_R01_20261001.json`：最长六个gap共同落在73.97–78.49s，抢占后到记录准入前占各gap的98.36–98.66%，准入到输出48.3–55.5ms。起始抢占五个匹配强制victim交换，一个跟随普通Q10保护释放；联合区间13个保护开始、12个十输出释放、0提前yield。不能严格把整个前段归为单一排队原因或作相同前态因果结论。停止扩展Q10/提前释放，下一唯一可检验线索：一次victim回收后的空闲是否足以同时恢复原target和另一个实际等待者，以及该余量是否在首target新输出前消失；先使用已有H1完整诊断，不新跑审计格。

工作稿已编译 `paper_a/output/pdf/A_recovery_working_draft_20261001.pdf`（10页，保留NOT SUBMISSION READY），主源码`paper_a/draft.tex`，重建`bash paper_a/build.sh`。已检查全页排版及最新图表，无未解析引用/越界。PDF只是当前证据的可读版本，不改变独立贡献/新数据确认缺口。


## 2026-10-02 A：原生 ordinary-only 主机成本诊断完成

授权45495单格156.630s结束，exit0/VERIFIED/GPU EMPTY，46份原件已本地回读（含补充3份缓存元数据）。`A_NATIVE_BACKFILL_CPU_DIAG_RESULT_R01_20261001.json`：128/128完成，20条实际ASYNC恢复→输出→完成，0forced。正式83.160s，互斥begin/hold/schedule_pre/schedule_post累计wall分别2.534779/0.130115/0.022328/0.033352s，总2.720574s（3.27149%），threadCPU总2.655059s，其中begin2.530850s。measurement-end快照与final一致，drain calls0。此为同策略带计时单轨迹，不与原机旧pair作吞吐比较或从4.048s差直接扣除。原native-only失败仍有效。

源码证据：当前open/non-diagnostic/no-forced路径每步创建全running view，通常仅被已禁用分支需要；普通合法动作仍有独立必要扫描。下一项仅按需构造view，保留动作边界owned/transfer/free门禁和保护检查；无动作异常所有权的早失败时点可能延后，不能称全部异常语义等价。同机同cache原timed/lazy timed两臂待冻结，不改变种子/阈值，不作新算法贡献。


## 2026-10-02 A：按需 running view 的同机计时配对完成

`A_NATIVE_BACKFILL_LAZY_PAIR_RESULT_R01_20261002.json` SHA4b64e1990a84a6823f9ad95c6b8ad00a308fd9870bb48d913b2f3e6e51d10a7e。原/lazy各128完成、19/27真实异步输出完成链、0forced，312.167s整组/GPU已释放，78原件本地完整。begin CPU/call736.983→113.571us（15.410%），四段wall4.325838→0.998584s，占自身正式区间4.738%→1.122%；主CPU减半目标及97%率/105%flow服务护栏均过。率比103.225%、flow比100.519%，maxgap11.690→11.731并未改善。gap77好51坏，flow53好75坏，TTFT26好102坏，goodput7高13低；68序列/1stop变化，输出126626/127483，非等工作量加速。

保留较低开销的实现，但不能宣布此前native-relative失败修复。仅去掉无消费者的running view构造；普通动作/保护检查仍在，无动作异常owned的发现时点可能延后。下一项只做去掉计时器的lazy ordinary→native参考同机配对，沿用最初97%率/105%flow/lowermaxgap/真实动作条件；当前未冻结未运行，不改变seed或参数。所有旧失败保留，独立机制贡献仍缺，工作稿NOT READY。


## 2026-10-02 A：去掉计时器的优化 ordinary / 原生参考配对通过原条件

授权45495同组lazy ordinary→native，exec74808整组304.564s、两格退出0/归档VERIFIED/GPU EMPTY。78原件已本地；`A_NATIVE_BACKFILL_LAZY_PERF_PAIR_RESULT_R01_20261002.json` SHA3516ec54f09e4cacdc52a0bad011c351afb9a644502754e11a94966986f65646。两格各128完成，优化ordinary有17/17真实ASYNC准入→新输出→完成、0forced、动作同step无抢占。native/ordinary输出率1444.756/1527.506（105.728%），meanflow37.827/33.688s（89.058%），maxgap14.927/12.916s，最初97%率/105%flow/lowermaxgap等全部9条件通过。

native/ordinary抢占49/59，输出126565/124886，75输出序列及2stop变化；gap111好17坏、flow127好1坏、TTFT123好5坏，全部20固定goodput点更高。两臂459file同cache种子独立副本，init32各、正式/预热finalmtime0（非零编译证明）。printed成本只累计实打印字段：ordinary/native store38.504/37.948GB各7interval；load16.444/13.348GB分别6/7interval，未打印load保留missing不补0；不称全episode成本或请求等待。

这是较低开销实现的一次反序、已见输入配对资格结果，无方差/等工作量/独立新方法/新数据确认主张。此前原版native-only失败保留，不能把不同机器旧差异直接全归CPU修复。下一步只分析本组17个更差gap和唯一flow恶化及17动作后续代价，再选择有自然证据的实质问题；当前无新GPU任务。

### 2026-10-02 — Pressure-point victim pilot

Native append-on-recovery plus tailpreemption exposes short restoredresidences. Directsuffix victimselection obtains a constrainedmaximum-gap gain (12.172 tail/8.415 arrival/6.222 density), with densityrate98.899%/99.096% andflow100.449%/100.954% againstthetwo controls; all128complete. However arrivalretainsbetterp90/p95gap,74densityindividualgapsworsen,and13/20goodputpointsfall. Thisisnotfullrequestdominanceorindependentnovelty. Scoreusescurrentresidenceoutputs/currentphysicalpages. The nextfrozen ablation testsdroppingresidence-reset ormemorynormalization; actualexecutioninprimarysession `moe-a-native-residency-ablation-primary-session-r01-20261002`, notyetresult. Preserveallpreviousnegativecontrols; no threshold/seed rescue.


## A native residence/KV factor ablation — 2026-10-02

Primary3arm completed128 each; full density/residence-only/lifetime-density rate1498.711/1475.048/1468.987,flow36.098/37.270/36.606s,maxgap5.020/8.533/9.081s. All fixed criteria pass, but fullscore goodput higher only12/20 and11/20 points and TTFT regressions persist. Simple factors do not cover this within-block joint result; no novelty/stability/equal-work claim. Canonical A_NATIVE_RESIDENCY_ABLATION_PRIMARY_TRIPLET_RESULT_R01_20261002.json (4d28303883dc9328d8804fd990e2e487a1577feb2f1ba47ab6d112c15c89bc09). Next unchangedrule new128articles37452..54660, two opposite-order primary blocks frozen; first14921 lockbusy before stage/GPU, no accepted job.

### 2026-10-02 A — residence-density fresh two-block confirmation FAILED
Both predeclared opposite-order blocks completed all128 requests per arm. Block1 density/tail maxgap10.071457/8.390893s, rate ratio0.986414, meanflow ratio1.025246; block2 density/tail9.033967/8.357038s,rate0.988446,flow1.008087. Both fail only the original lower-maxgap-vs-tail criterion; neither replaces the other. Goodput density vs tail2higher18lower and9higher11lower across20points. Full outputs differ and are retained. Combined result A_NATIVE_RESIDENCY_FRESH_TWO_BLOCKS_RESULT_R01_20261002.json; both113-file sessions verified locally. Development successes do not support a stable method claim.
Next one actual action is guarded continuation after current-request self-preemption; frozen triplet tail_break,density_break,density_continue nowRUNNING PTY11180 primary22937. Same now-seen128 inputs; requires actual useful successor work AND service criteria against both controls. This is new FCFS bypass behavior, not proof that one skipped call caused the entire long stall. BidKV source-pinned score adaptation implemented/tested locally, GPUUNRUN; not full reproduction or novelty evidence.


### 2026-10-02 A — strong-score control and the current-request feasibility question

Density self-preempt continuation FAILED its service criterion despite 29 actual continuation actions and 319 same-call successor outputs: maximum request gap 10.750 s, versus density break 7.776 s and native tail 10.535 s; goodput is lower at 14/20 points against both. The worst source had only 27 outputs and held 193 pages when preempted; readmission required 194 pages with zero free. Its observed pause to the next recorded running admission was 10.735 s. This was its first preemption, so past pause duration cannot identify that specific decision. Reject this density-continuation optimization; do not tune a pause-history score to the counterexample.

A pinned official default BidKV score adaptation is now the closest measured control. Tail / score break / score continue maximum gaps are 8.781 / 8.366 / 7.181 s; actual output rates 1513.237 / 1480.460 / 1485.885 token/s; mean flow 37.301 / 37.137 / 37.722 s. Both predeclared score and continuation tests pass in this one already-seen block. However, score continue loses goodput at 15/20 points to score break, so both remain controls. This reproduces the score and ties within the qualified native suffix, not the complete upstream system, and is not our independent method. Canonical results: A_NATIVE_SELF_PREEMPT_CONTINUE_TRIPLET_RESULT_R01_20261002.json and A_NATIVE_BIDKV_SCORE_TRIPLET_RESULT_R01_20261002.json.

The next hypothesis is narrower than a new score: avoid selecting the allocation-failed current request when another qualified unprocessed victim exists. With current KV C, free pages F, and next-token growth d, self release leaves immediate readmission short by (C+d)-(F+C)=d-F. All 19 eligible observed BidKV decisions have F=0 and d=1; an alternative can fund that increment under the private-page assumptions. This explains local feasibility, not total service gain: interruption is moved to the alternative victim, which must be accounted through completion. Native tail implicitly has this preference when a suffix remains, so official BidKV integration is being checked before any novelty claim. Frozen current-guard R02 compared score break, score continue, and guarded score on the same seen 128 inputs; completed and all15criteria passed. Guard maxgap7.764s versus9.861/8.382s, rate1479.547 versus1491.631/1481.869, meanflow36.751 versus37.157/37.966s. Fourrealguardactions allgivecurrentoutput; displacedvictimcosts andmixedgoodput14/6,19/1 are retained. Canonical A_NATIVE_CURRENT_GUARD_TRIPLET_RESULT_R02_20261002.json. Two opposite-order fresh128 four-arm blocks are now frozen with native-tail anchor, UNRUN after initial commonlockbusy. No new independent contribution or fresh confirmation is established.

Pinned-source boundary: official BidKV commit `5ee80256d263d58b1e512d9d436d47e9bac564ba` contains the selector but no host scheduler loop. Its `pick_victim(running, policy, ...)` has no allocation-failed-current parameter and ranks the supplied running requests; the selector itself neither excludes that current request nor rolls back a scheduled prefix or continues after self-preemption. The external host's behavior cannot be established from this plugin commit. The pending current-request guard is therefore a native-loop adapter feasibility candidate, not a new BidKV score or a claimed upstream fix; no service benefit is asserted before its complete-request result.


### 2026-10-02 — current guard fresh first block fails the strong-score comparison
Fresh B1R02 completed569.999s,all4exit0/archivesVERIFIED/GPUEMPTY;154rawfiles/247830974Bverifiedlocally,readbackSHA be93979a0e613710727c79a10310c5dd563ebcc62156df56eb5c481e0f8a0dd7. Canonical A_NATIVE_CURRENT_GUARD_FRESH_B1_RESULT_R02_20261002.json SHA cb4e70341413f55713000f0b06523c78e6342e20de21b5f5598a02bd5a04b732. All128/armcompleted,forced0. Tail/BidKVbreak/continue/guard rates1525.683/1494.663/1491.013/1492.263,meanflow37.362/37.367/37.405/37.257s,maxgap8.606/5.997/8.654/6.870s. All7guardchanges real,all7current samecalloutput/displacedvictimlatercompletion.17/18criteriaPASS;FAILonlylowermaxgapvsBidKVbreak. GoodputguardvsTail16higher4lower,vsBreak11/9,vsContinue15/5;outputsequence52/55/58 andstop3/2/2differ. No equalwork orstablemethodclaim.
Longest0059158: ordinaryBidKVdecision step1760 selects anotherrequest,guardineligible/unapplied; firstpreemption at304outputs,206heldblocks. Preempt26.459951→recordedrunning33.313752→newoutput33.329497s,maximumgap6.870386s. Current-onlyguard doesnotcoverthisobservedfailure; exactqueue/loadsplitunknown andnoalternativeactioncausalclaim. See A_NATIVE_CURRENT_GUARD_FRESH_B1_FAILURE_CHAIN_R02_20261002.json. The originalfrozenreverseB2remainsrequired,notarescueblock; samepolicy/input/18criteria. Resource-onlyR02uses1.25GiBafter4x228492707Bcache,900s/cell4800group90GiB. B2R02archiveuploaded,attempt45430LOCK_BUSYbeforestage/session/dedup/compaction. NoAjob.
Closestbaselinegap confirmed in identicalpinned vLLM0.26scheduler: nativePRIORITYcanrollbackalreadyscheduledprefix token/block/spec/encoderplanandindex,while currentFCFSBidKVadapterqualifiesonlysuffix. A separatefullrunningbaselinecandidateisbeingimplementedlocally; UNRUN,notanewscore andnotpartofthese frozenfourarms. PaperNOTREADY.


### 2026-10-02 — frozen current guard: both blocks fail; stop unchanged rule
B2 completed all four arms, 128/128 each, native full saving, zero forced rotation; 574.160 s, all exit0/VERIFIED/GPU EMPTY. Its 154 raw files (250019533 bytes) are verified locally, readback SHA 06cb9645b24fdbfec93780f7ca80e7c38a2b107fed1a1ef4bfd48d17073c446d. Tail/BidKV break/continue/current guard: output rates 1537.946/1503.199/1499.825/1489.107 token/s; mean flow 38.095/38.458/37.698/37.077 s; maximum gaps 8.850/9.549/7.060/9.111 s. Guard executes five real victim changes with same-call current outputs and all displaced victims later complete. It fails three service conditions: rate >=97% of tail (observed96.824%), lower maxgap than tail, and lower maxgap than continue. Goodput guard improves19/20,20/20,13/20 fixed points versus tail/break/continue; output sequences differ63/60/58 and stops6/5/5. These favorable subsets do not replace frozen criteria.
B1 previously failed lower maxgap than BidKV break (guard6.870 vsbreak5.997). Combined A_NATIVE_CURRENT_GUARD_FRESH_TWO_BLOCKS_RESULT_R02_20261002.json retains both independent blocks and all complete-request frontiers; both fail. Stop performance expansion of unchanged current-only guard. No pooling, threshold, seed, or best-block rescue.
Next concrete unit: qualify the stronger full-running BidKV-score adaptation that can roll back already scheduled native decode requests. candidate_native_bidkv_full_running_r01 manifest844caa2fa985d6101ecf74b2f5aae76a778711868f96dd8bd67df0b3d97855e2; only3payloads changed, CPU3/3 pass and pinned native branch AST compiles. Analyzer analyze_native_bidkv_full_running_triplet_r01.py separates rollback/real-preemption qualification from service budget and checks raw same-call absence of victim output. Both are GPU UNRUN; not new method or complete upstream reproduction. One native tail/suffix score/full-running score triplet being prepared, guard and continuation OFF. Paper updated with both frozen failures, still NOT READY. No own GPU job at this checkpoint.


B2 longest-gap follow-up: A_NATIVE_CURRENT_GUARD_FRESH_B2_FAILURE_CHAIN_R02_20261002.json identifies request0056621, gap9.110669579s at output146→147. Its second preemption is step605 (7.568664s), after prior readmissionstep596 (7.423875s) and resumedoutput7.440438s; next recordedreadmission16.662584s andoutput16.678723s. Both decisions select a request other thanfailedcurrent andguardisineligible. Step605 records onlysuffixindices26..28; earlier26positions have unknownidentity/qualification/score inthese records. Thus prefixeligibility is a concrete baselinegap but no offlinebenefitclaim follows. Same-source controlgap tail0.042728/break9.548761/continue4.055909s is descriptive, notsame-state counterfactual.


### 2026-10-02 — full-running BidKV native triplet executed; readback deferred
Native tail / suffix BidKV / full-running BidKV completed in440.250820s, all3exit0/VERIFIED/GPUEMPTY, each128/128complete,0forced. Plan8b8332b9ba36deaac12dd121cb1e05e3733513afb8fb1c73f0b2f522d586fdc2; controller005333a079871e487cd693a9685fac408aa8909b4c3a4754670eb18535eace58; candidate844caa2fa985d6101ecf74b2f5aae76a778711868f96dd8bd67df0b3d97855e2. Guard/continue OFFall, fullrunning OFF/OFF/ON. Lightweight completion metadata shows62/46/52nativevictimdecisions; fullarm17prefix choices,17tokenrefunds,all17absentfromnativeoutputplan/inpreemptedIDs. Independent rawpreemption/output joins and complete-serviceperformance still UNANALYZED. Do not infer benefit from these action counts.
Remote /root/moe-a-native-bidkv-full-running-session-r01-20261002 complete; controllerPTY96980 and monitor75748 ended. Export95851 and9517 each LOCK_BUSY before anymutation; no tar created. Local export_native_bidkv_full_running_r01.py retains exact boundednextaction; next naturalbreak try commonlock once, archive thenlocallyanalyzeusing frozen analyze_native_bidkv_full_running_triplet_r01.py. No current A GPUjob, goalACTIVE. Fullrawisremote; allpreviousfreshnegative rawretainedlocal+remotecompressed. Neverrepeat completedprepare/cleanup/run.


### 2026-10-02 — full-running baseline qualified; service objective failed
A_NATIVE_BIDKV_FULL_RUNNING_TRIPLET_RESULT_R01_20261002.json SHA1c72e389f3c0a6315357be4b3004b8e04a842b6b38409076aea96a25446014d2. All124files/188008598B verified locally, readbackSHA675a63c8863b7549a66a4cf038d91ebcbd657b0189ed6a2332481499e4f7483a. Tail/suffix/full all128complete,0forced; rates1498.437/1479.641/1467.100 token/s,meanflow36.881/37.690/37.180s,maxgap8.627/6.138/9.364s,outputs126617/128154/127200,preempts62/46/52. All17prefixchoices have positive rollback, same-step rawactualpreemption, absentnativeplan, no same-callrawoutput, lateroutput andcompletion. All10 qualificationconditions pass. Service fails lowermaxgapagainstbothcontrols; fullgoodput7higher13lower vsTail and9/11vsSuffix;70/67outputsequences,3/1stopdifferences. This closes a baseline action-space gap; it is not our optimization or complete BidKV reproduction. No stablebenefitclaim.
Fullarm longest0048279 was selected as a nonprefixvictim atstep2929:preempt45.647159→recordedrunning54.994146→output55.009528s. Its laterprefixpreemptionstep3569 after75furtheroutputs causes only0.111431s outputgap. Membershipin17prefixvictims doesnotattributeits earlier9.363532s worstgap to prefixrollback. Rawchain A_NATIVE_BIDKV_FULL_RUNNING_FAILURE_OR_TAIL_CHAIN_R01_20261002.json.
Next single hypothesis differsfromvictimscoring: at allocationfailure, defergrowers onlyif a knownrunningrequest can reach hardoutputcap usingalreadyownedKVslots (computed+remainingcap<=heldcapacity), permittingnearcompletiontofreecapacity withoutwhole-requesteviction. First checknaturalopportunity inall52fullcandidatepressurestates. NofutureEOS orcoefficients; noGPUplan/newclaimyet. Strong simplecomparisonwillbe ordinary useful-work deferral and nativeTail, with closestscorecontrolrequiredifpilotwarrantsconfirmation.


## 2026-10-02 A — native capacity-deferral pilot completed, service failed

Seen-input three-arm pilot, native tail / generic prefix-work deferral / hard-cap capacity-qualified prefix-finish deferral. All 128 requests per arm completed, no forced rotations. Actual actions: 0 / 9091 / 276; all logged actions joined independent raw output/preemption and final native plans. Every finish witness satisfied its predecision remaining-cap/owned-capacity gate. All witnesses later completed at the hard cap, none by EOS and none re-preempted; these are event counts, not distinct requests.

Tail / work / finish output rate: 1504.284 / 1435.459 / 1487.673 tokens/s; mean flow: 36.441 / 38.896 / 37.362 s; max gap: 8.729 / 7.877 / 9.026 s; native preemptions: 55 / 55 / 56. Mechanism qualification passes. Finish fails the maximum-gap comparison against BOTH controls; work vs tail also exceeds the predeclared rate/flow cost limits (0.954247x / 1.067386x). Goodput points higher/lower: work vs tail 4/16, finish vs tail 1/19, finish vs work 9/11. Sequence differences: 58 / 51 / 63; stop differences: 4 / 1 / 3; actual output totals: 126428 / 126247 / 127286. No equal-work speedup claim.

Canonical: 20260929_commit_recheck/A_NATIVE_COMPLETION_DEFERRAL_TRIPLET_RESULT_R01_20261002.json, SHA 6568a67977a3074a4ee4dd609f484d4a666406b32be5593ba3ecf4c88ed8b2f0. Raw 127 files / 198420782 bytes verified locally; compressed archive SHA 06d13953da68c1615ef399d14c071b66a1445f29ff027f69a487e9ca895b13a7. Block completed in 438.169 s; GPU released. Unchanged gate will not be expanded to fresh confirmation. Next action is to locate concrete delay/release chains, not tune a threshold or seed. CacheOPT already uses predicted completion releases, so this pilot does not establish novelty by invoking completion alone.


## 2026-10-02 A — physical victim-size pilot completed; both extremes fail tail

Natural tail trace had55 single-page deficits at free=0, while tail victims held a median132 pages and smallest qualified other victims81. This motivated a direct three-arm seen-input test of native tail / minimum sufficient other victim / maximum sufficient other victim, with all other interventions off.

All128 per arm completed, zero forced rotations;47 minimum and28 maximum choices differed from tail and executed as verified same-call native preemption with current output. Rate1514.360/1506.869/1497.840 tokens/s; mean flow37.375/37.785/37.876s; maximum gap8.490/10.723/10.244s; native preemptions67/127/36. Minimum passes rate/flow budgets but fails maximum-gap versus both controls. Maximum also has worse maximum-gap than tail despite fewer preemptions. Goodput higher/lower: minimum-tail8/12,maximum-tail9/11,minimum-maximum11/9. Sequence differences55/54/63, stop differences0/0/0; actual outputs128296/128294/128287. Not equal-work speedups.

Canonical A_NATIVE_VICTIM_SIZE_TRIPLET_RESULT_R02_20261002.json, SHA5df34b60a0690b84bf27a0314550937b2eb6201bc5106055ec5c89209ad4cab2. All127rawfiles/188672808B verified locally, tarSHAca9f24caab4997a736ad22639bb424c9998362e138fd3832472e8b9d01915a4b. R01 exited before any cell after25.014s at disk-space gate. R02 reused unchanged frozen packages and kept the same1.25GiB reserve/scientific settings, retiring only verified duplicate source-upload archives; completed in438.551s. Experiment cells all exit0; the lingering SSH connection was closed after successful locked export and local raw verification.

Decision: stop unchanged size-only expansion. Fewer native preemptions do not establish shorter recovery stalls, and smaller victims can greatly increase repeated preemption. Analyze concrete worst-request chains. The only next interface check is whether native external-prefix restoration can truthfully load a shorter prefix when the full match cannot be admitted; no new GPU candidate or benefit claim exists yet.


### 2026-10-02 A — one-shot oldest recovery admission: valid triplet, NO FUNDED ACTION
R01 three arms completed128each in433.741s; all exit0/VERIFIED/GPUEMPTY.127files/187940882B local verified,81output hashes. Canonical A_NATIVE_OLDEST_ADMISSION_TRIPLET_RESULT_R01_20261002.json SHA4cc7d53c906294eae7c6219397a14b57b56ff9ebb981c8c174e1944bfa564684. Native/queue-only/fund-design rate1508.103/1506.574/1512.962,meanflow37.467/37.729/36.997,maxgap12.922/11.433/12.708s; outputs128154/128300/127444,preempts47/47/42. Goodput4/16,17/3,18/2 across20points; natural seq57/49/64 andstop2/1/3 differences retained.
All select source0042224 but at6/15/4outputs,not same physical state. Target gaps10.497/10.885/10.142s. Queue-only target already head; fund mode moves head at609, then610 CANCEL_OLDEST_KEEP_OTHER_OR_UNKNOWN_TRANSFER_JOBS. Actual forced0/Q1holds0,so numerical service pass cannot establish funding benefit. No executed-funded mechanism service falsification. R01 does not retain those commit job identities.
Followup A_NATIVE_OLDEST_ADMISSION_ACTION_FOLLOWUP_R01_20261002.json: target later native async→output→1024complete,no repeatedpreemption. Planned victim not evicted by this action; later unrelated native-tail step674 after173outputs causes7.498s pause. No observed funded cost transfer.
Next only refine commit transfer gate based on pinned job source-block ownership and native per-preempted-request flush; known disjoint running STORE jobs may be safe,LOAD/unknown/alias remaincancel. NewR02 candidate pending; no changedthreshold/seed/servicecriterion and no GPU job. CacheOPT/UniBoost already contain close priority/funding/protection actions; no novelty claim. Paper28pagesupdated.


### 2026-10-02 A — R03 actual funded oldest recovery: one-shot criteria pass
Native/queue-only/fund all128 complete,0 failures; block435.047s,GPU released. Gate correction permits4 evidenced private disjoint running STORE jobs; one actual planned-victim native preemption and Q1 native target restoration executed. Canonical A_NATIVE_OLDEST_ADMISSION_TRIPLET_RESULT_R03_20261002.json SHAf76147d85e97cd2885554bee50767987696d6cee6ba6711e9af48c8e35ad9940. Target0042224 gap10.508/10.256/1.713s; anchor-to-output9.155/8.930/.0673s. Different pre-states6/7/71 outputs, so these are descriptive independent trajectories, not same-state causal effects. Victim postpreempt-to-output7.236s, retained in cohort costs. Rates1513.008/1517.654/1509.396; meanflow37.471/37.308/36.135; globalmaxgap11.827/12.756/11.643. Output128148/128154/126304, naturalstops3/3/5; no equal-work speedup. All predeclared one-shot criteria pass, not repeated policy/stability/novelty. R02 zero-cell disk abort and R01 no-action retained. Raw130files/187450903B and81 output hashesverified; tarSHA5d8b90d43f6c7253c7707353d0c0c7237efeb8c335dfd32367f0317ace5f65d3. Next one direct experiment: repeated fixed1s oldest policy native/queue/fund, at most one active episode, actual victim costs/all128 and20goodput points; no threshold search.


### 2026-10-02 A — repeated oldest recovery R02: positive full-cohort pilot
All128 in native/queue-only/fund completed,0failed/unfinished; all20 frozen criteria PASS. Native/queue/fund P95 perrequest maxgap7.938750/9.581802/1.552190s; globalmax12.066355/12.883049/2.187850s; actual rate1507.069617/1507.384391/1497.005644token/s; meanflow37.055118/37.590503/37.487052s. Fund vsnative rate99.3322%,meanflow101.1657%; p95gap~80.45%lower. Complete natural output127401/128154/127286;stops4/3/4; sequence differences59/66/54,stop3/4/1 retained. No equal-work speedup.
Fund58recoveryepisodes,57actualforcednativepreemptions across21targetsources,1DIRECT_READY,0cancel. Everyepisode retired, no samepause retry, all actualcommitnativeadmission/firstoutput/Q1/victimcompletion verified. Native/queue/fund preempts46/48/103. More preemptions accompanied shorter stalls in this observed policy; this is not a universal theorem. Native/queue anchors4/7; independent trajectories, no samephysicalcounterfactual. Fund goodput14higher6lower vs eachcontrol; maxgap57requests improve71worsen vsnative,median.041560→.055986s; full frontier preventsuniform-winclaim.
Canonical A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json SHA b1cc9b0487e3ed372f277606fd763c750d9c3aecc1948aa06dbd58da4482f5da; frozenanalyzer7e954006f8a7028f8a92461d14bdb7360a4e9b04ffe0c77952015d4b799a2d33, candidatebfe6858307624b09dcac0d4e57a47ee965c9279e643b4d98c002ae3fde0a4946. Raw130files/188411114B,81outputhashesverified; tar0cbe2ff369943c43bada213deece807006874ba9b289d4b3aabb3f38c230b47c. Block436.408824s, allGPUEMPTY. R01 zero-cell25.071082s diskabort retained; R02 only deduplicated immutable own completed source paths, preserving allscientificsettings and1.25GiBreserve.
Next unique experiment: challenge against existing strong free-fit ordinary-backfill, native and unchanged repeatedfund. Currentresult is one seen exploratory block, not stable/fresh/new-method/READY evidence; closest priority/funding/protection overlap remains.


### 2026-10-02 — recovery service feasibility model and raw residency evidence

Primary objective: P95 of per-request maximum host output gap, with actual throughput, completion, whole-cohort and displaced-request costs. Three mechanisms considered: cross-phase commitment; joint recovery cost/service quantum (primary); release-driven execution. Existing Q1 funding and old fixedQ10 protections are not new contributions. UniBoost MemGuard is the closest quantum action; adaptive choice must beat fixedQ on the same reservation and selective-admission path.

Focused raw diagnostic:58 recoveries on21 sources;48 reevicted (32 funding another recovery,16 ordinary native);12 obtain1–9 outputs. All6 short native reevictions still own unused slots. Two one-output events retain12/9 slots, but following gaps differ1.233/.152s. Host accepted→firstoutput median49.061ms is not DMA cost. See recovery_quantum_20261002/recovery_residencies.json and figure. No counterfactual service claim or event-level replication.

Pure recovery_lease.py jointly bounds q by full endpoint KV, past observed recovery/decode cost and peer age frontier. Six CPU model checks pass. Separate candidate_native_recovery_lease_r01 is being integrated with whole-q reservations, selective peer admission, reserved token budget/running slot, and clean deferred-queue lifecycle; native GPU performance UNRUN. A focused review found/fixes missing running-slot reservation and persistence of logical skipped requests. No new positive method claim.


Focused replacement working draft: experiments/admission_capacity/20260929_commit_recheck/recovery_quantum_20261002/paper/draft.tex and output/pdf/draft.pdf. Historical thirty-page draft retained. All new GPU comparisons still UNRUN.

Native lease integration completed in separate candidate (26package+6model CPU checks); whole-q KV, one compute token, one running slot and reversible logical deferral share one admission path. Newhost strong/prototype bundles are local and unrun. Healthy32 tasks now genuinely retokenized using fixed local tokenizer, three workload profiles ready, no task-quality result. Action analyzer separates actual post-native plan receipts from independent raw output/preemption joins. Main scientific uncertainty still requires GPU execution; necessary upload authorization remains pending after automatic-review rejection.


### 2026-10-02T09:57:28.795767+00:00 A recovery-quantum iteration: evidence-backed change of formulation

**研究问题。** 有限 KV 与计算预算下，怎样联合恢复、抢占和连续服务，缩短每请求最大输出间隔的 P95，同时计入所有请求的完成代价？

**中心机制。** 恢复前共同选择服务量子与完整物理 KV 增长，保留运行槽位和执行 token，并允许余量内的原生 peer 接纳。历史恢复成本比值和当前 peer 年龄约束量子；未知成本退回 Q1。原型已实际改变 vLLM 调度并运行至完整请求结束。端点容量条件与条件进展性质成立的前提包括有效 ownership/传输、原生可执行性、逐次有效 decode 和租约不被提前释放；不提供无条件 q 输出或墙钟 SLO 保证。

**三项已测结果。** (1) 同机强普通回填对照仍留下尾部停顿：native/ordinary/fund P95为6.965/3.885/1.575秒，fund比ordinary实际输出率低1.52%、平均完成flow高1.85%，86/128请求flow更差，20点goodput网格11胜9负。(2) 六个新租约实验单元的实际动作核对通过，protected目标无抢占，已腾让victim和接纳peer最终均输出并完成，但这不证明没有延迟损失。(3) 自适应与fixed4的排序在两轮中翻转：P95相对差+14.44%/-10.80%，输出率-1.00%/+0.74%，平均flow则+0.35%/+1.62%。已知成本时desired q始终3，共103/103和75/75次。R01最坏同输出序列请求flow增加18.474秒；R02最坏增加16.760秒且序列不同。这些是原生服务观测，不是三项独立创新声明。

**自然任务与成本边界。** Dev16突发4096页三臂均16EOS、7/16正确；512页五臂均16EOS、8/16正确，块内序列相同、无恢复激活。另一组16题稳态512页五臂也均16EOS，但native/ordinary/q1/fixed4/adaptive正确数10/8/8/9/8、每个非native臂与native有10/16序列不同；抢占2/2/2/2/1，仍无恢复动作。不能主张质量等价或把这些差异归因于自适应动作。全部健康计数版KV传输完成统计有效，copy-work允许重叠、不是暴露延迟；原文章组完整传输和全控制器CPU成本仍未测。不同模型/输入/机器数据不混合。

**决定与停止范围。** 停止当前 h/d、rho0.5、q上限16 的自适应选量子 formulation，保留可执行资源机制和全部原始轨迹。证据尚未支持它稳定超过同路径固定Q4，也未证明整个恢复量子家族无价值。最近邻CacheOPT、UniBoost、LTR、BidKV已覆盖优先级/成本/保护等动作，不能将其重新命名为新贡献；完整上游系统和结构消融尚未运行。当前材料是有解释力的原型及足以换 formulation 的具体证据，不是已达到CCF-C投稿标准的方法。

唯一后续研究问题：先量化可观察的victim/peer等待和剩余服务是否解释完整请求损失，再设计替代控制器；不继续为既有轨迹搜索rho或量子阈值。若出现异质恢复成本及可预测的完整服务增益，并在有恢复动作的自然任务上超过固定量子，可重新打开当前方向。尚无oracle界或同前态因果headroom结论。

当前结果索引：`experiments/admission_capacity/20260929_commit_recheck/recovery_quantum_20261002/README.md`；决定记录：`output/RESEARCH_DECISION_20261002.json`；论文：同目录`paper/draft.tex`与`output/pdf/draft.pdf`。22个完成单元合计1360次请求执行，不能当1360个独立任务。全部raw、失败和未运行旧计划保留；所有A GPU控制器退出并释放共同锁。最终6页PDF已双次编译并逐页视觉检查，结果索引和状态已完成更新。
