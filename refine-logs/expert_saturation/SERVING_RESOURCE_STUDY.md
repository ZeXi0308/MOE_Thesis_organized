# 固定 KV 预算下的请求暂停与服务量

本页承接长任务会话 `01a09ba8-3b21-7853-89fd-83c41e309c07`，记录同一个研究问题及真实接续点。历史总裁决仍由 `docs/current/README.md` 与各 sealed verdict 管理；本页不修改它们。共享结果只读引用 [RESULT_LEDGER](experiments/admission_capacity/RESULT_LEDGER.md)，原研究详记在 [RESEARCH_NOTES](experiments/admission_capacity/RESEARCH_NOTES.md)。

## 问题与范围

**固定GPU资源与明确host预算下，怎样安排恢复和后续执行，降低生成长停顿，同时控制完整服务效率损失？**

已测范围：RTX 5090，OLMoE-1B-7B BF16，vLLM 0.26.0，32 个请求，每个 3072 输入 / 1024 输出，50 ms 到达间隔，token budget 1024。早期组使用 KV 16,089,350,144 bytes / 7,671 可用块；当前 d6 组件组使用 13,960,740,864 bytes / 6,656 可用块。各组文档身份和实际资源按原 bundle 固定，不跨文档/资源组作配对排名。早期轮转关闭 prefix cache，后续已完成APC开启的同配置对照；不把早期配置称为默认 vLLM。主要结果并列报告完整请求吞吐与每请求最大 ITL，成本包括 TTFT、平均完成时间、全部失败和未完成请求。没有业务 SLO 时不事后反选阈值宣布成功。

主问题保持 `OPEN`；现有结果为 `NATIVE_SERVING / MEASUREMENT_ONLY`。机制是实际抢占、恢复优先与恢复期间的资源保护；当前实现使用固定冷却和受害者排序。具体实现或排序失败，不等于问题被否定。Expert paging 是同一总体资源问题的另一运行域，其他会话的结果另列，不与本页 resident 长上下文数字直接排名。

后续范围分别扩展到同6656块下的受控P2560/P3072、O1024异构context，以及4096可用块下64篇完整自然文章、0.5s持续到达、允许EOS。这些是不同运行组，各自比较；自然到达组每格仍有58/64请求被1024上限截断，不能称充分覆盖自然结束分布。2026-09-14恢复访问后现场GPU从70fa…eef9变为bd5e…0bdc，新funding六格只做新设备上的组内比较。

## 当前结论与接续点

**2026-09-15 当前接续。** [H 独立输入六格](outputs/admission_capacity/20260915_natural_cadence_holdout_r02/RESULTS.md)已完成 768/768 请求，来自固定 128 篇未用于选择的新完整文章。eager/current 两对最大 gap −13.60%/−3.12%，实际输出吞吐 +5.44%/+0.07%，平均完成 −6.26%/+0.49%，均通过事前吞吐损失≤3%、均完成增幅≤5%且最大 gap 下降的合同。该范围内选择 selected/eager 为停顿优先简单参照；收益小于 [G](outputs/admission_capacity/20260915_natural_recovery_cadence_r01/RESULTS.md) 的 −63.43%/−53.12%，第二对仅降低 0.121 秒，不称统计稳定。原生完整保存/no-extra-rotation 的最大 gap 为 11.835/12.011 秒，eager 为 2.936/3.760 秒，但原生吞吐、平均完成及 107/106 个请求自身 gap 更优，必须保留系统取舍。eager 比 current 少 541/560 个输出；不是等工作量或质量证明。late64 事前预测方向支持但与全局最大值重合，非新增重复。各臂 host 末态有效 16 GiB，不证明 live-history 替换；恢复期 EOS、稳态服务、内部完整时间分解与性能 Oracle 均未覆盖。

H r01 因旧 64 人数资格在 warmup 前失败、零测量原件保留；r02 仅修人数到 128，同合同完成并已归档释放 GPU。此前 [F 保存四格](outputs/admission_capacity/20260915_natural_save_scope_timing_r01/RESULTS.md)已闭合，full 未升级为全服务更好底座，不重跑旧诊断。停止增加 cooldown/window/predictor；剩余决策是兼容且合理校准的 LTR-style 是否覆盖当前取舍。互斥 LTR-style adapter 已完成 CPU 原生调度循环、增长和队列边界检查。[新增反例](outputs/admission_capacity/20260915_joint_growth_decision_r01/I_PREPARE_SUM_GUARD_ADDENDUM.md)证明全体增长预筛会推迟合法 prepare 而未避免同一 peer 抢占；删除这条非必要限制进入基线，不主张服务收益。现已接受 [r02 单格包](outputs/admission_capacity/20260915_natural_ltr_style_component_r02/README.md) 6888c318…4c9e73 / 30 文件，固定 G64/T30/Q10，旧 CPU receipt 明确复用。r01 原件和零测量记录保留；对 26862、53036 的有界连接诊断均在认证前关闭，资源状态 UNKNOWN。r02 零连接尝试、无上传或后台执行；有效入口恢复并现场核验后，唯一下一项为真实保存/加载/量子生命周期诊断。T{30,200}×Q{1,10} 的有界校准与 eager 相对 3%/5% 成本选择已经固定，性能组尚未接受，不冒充完整 LTR。准确接续见[统一清单](CURRENT_EXPERIMENT.json)、[一页论文论证](PAPER_ARGUMENT.md)和[取舍图](outputs/admission_capacity/20260915_natural_cadence_holdout_r02/paper_view/transfer_budget.pdf)。主问题保持 OPEN，下文旧 RUNNING/UNRUN 为历史阶段。

保存已成为本固定运行域强基线：[原六格](outputs/admission_capacity/20260915_repeated_kv_service_r01/analysis/REPORT.md)192请求完成，主两对吞吐+5.86%/+2.03%、均完成−5.75%/−2.13%、最大gap−18.37%/−14.93%。[另一文档集合](outputs/admission_capacity/20260915_repeated_kv_cohort3_r01/analysis/REPORT.md)128请求完成，分别+6.63%/+5.42%、−6.53%/−5.38%、−19.79%/−18.16%；这是相同长度/压力的文档迁移，不是新总体或自然EOS证明。原r02两格只保留执行资格，不混入这些配对。

[启动五格](outputs/admission_capacity/20260915_saved_recovery_start_r01/RESULTS.md)已完成160请求：save-on下只去掉global cooldown20，保留absence30、residency30、victim、保护和份额。eager实际84轮转/89加载，较current的38/43频繁；主两对maxgap−9.798%/−10.413%、吞吐−0.502%/+0.124%、均完成+2.042%/+1.410%。23/32请求gap改善，同9个恶化且完成更晚；每对26/32完整输出相同，质量未验证。调用1312→1270没有兑现稳定吞吐增益，不能依靠结构步数选胜。

新增[生命周期证据](outputs/admission_capacity/20260915_saved_recovery_start_r01/execution_weste_26862/lifecycle/REPORT.md)表明更频繁的有效恢复和peer停顿重分配：新诊断90恢复/58再抢占，无0或1–2输出失败；host历史仍复用，不能把GPU释放都写成永久浪费。统一host调用边界下，单段等待中位缩短约一半，但累计跨请求恢复间隔增加；跨组诊断只作解释，不代替轻量配对。该证据支持停顿—均完成权衡，未支持更换victim或延长保护。

**历史阶段选择（启动五格后，已执行完毕）：** current/eager保留为同资源简单基线，停止同域冷却、文档或seed扫描；随后完成上述0.2s自然selected与native-full资格。原0.5s低压力无额外轮转结论直接保留。最新保存范围轻量四格现已256/256完成、唯一归档回读并释放GPU，主分析由原执行方交付；此处旧准备状态不再作为当前执行入口。

[cohort3原八格补记](outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/RESULTS_ADDENDUM.md)全部完成：most_output对native吞吐+8.50%/+1.83%，最大ITL14.40/14.29→2.81/2.94s，同时平均完成+9.61%/+17.41%，27/28请求更晚完成。most在吞吐和全局最大ITL两轴都优于fit/guard_residual，后两者平均完成更好；停止把当前恢复保护组合包装为独立方法。[全部八点图](outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/analysis/tradeoffs.svg)并列保留第三指标代价。

本八格的[限定完整性复核](outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/audit/EXPERIMENT_AUDIT.md)已完成：PASS、P0/P1=0，same-family/provisional；主分析和全部请求指标由保留原件复现。唯一P2为历史成本模型的训练来源哈希未嵌入原模型，不影响主表。停止追加该组审计，结论仍为MEASUREMENT_ONLY。

[异构context实测](outputs/admission_capacity/20260914_context_victim_calibration_r01/REPORT.md)揭示当前least适配器先排序后验资的缺口：每次连续18步选中无法资助目标的victim，同时存在可行候选。此缺口只在该适配器/运行域被验证；旧同长度前态全部可行的结论没有覆盖异构域。[funding-filter六格](outputs/admission_capacity/20260914_funding_filter_comparison_r01/REPORT.md)在新设备全部完成并回读，保留most与原least。过滤对least吞吐−3.011%/+1.272%、均完成+3.134%/−1.125%翻转；对most吞吐−5.706%/−5.096%，maxITL减少0.073/0.162s，仍是取舍。两个filtered均多6251重算位置和7次调用，合法候选修复没有稳定净收益。[运行前预测核验](outputs/admission_capacity/20260914_funding_filter_comparison_r01/MODEL_PREDICTION_CHECK.md)已闭合：两新filtered初态均与旧block0稳定身份全等，兑现其step894首输出、89761重算、step1451结束的冻结结构预测；旧block1的F921/15请求多1位置起点未实例化，不能按block编号误配。模型支持了本轮运行前资格判断，runtime未消费离线预测，墙钟排序仍未验证。

[持续到达/EOS四格](outputs/admission_capacity/20260914_streaming_recovery_r01/RESULTS.md)中全部9次恢复均持续到请求完成，没有反复短恢复缺口，most额外轮转为0；不改本组压力或阈值寻找动作。该负边界促成的funding-filter对照现已完成，未形成稳定增量，主问题没有被充分否定。后续接续同问题中已登记的原生两阶段KV保存资格与完整恢复成本核对，不再调整当前保护期或候选过滤阈值。

[单事件两阶段保存](outputs/admission_capacity/20260914_staged_store_probe_r01/REPORT.md)已真实完成3392 token、444596224B的store/load；少2次重算调用、多2次decode，总调用不变，仅victim提前2步。单对墙钟退化含动作前漂移，不能归因或宣布收益。[重复动作结构模型](outputs/admission_capacity/20260914_repeated_staged_model_r01/REPORT.md)在假定保存成功、未计物理搬运成本时，相对immediate most-output仅少7/5/3次或多1次调用，随假定load delay 1/2/4/8步变化；平均完成方向在2到4步之间翻转。这是延迟敏感性，非性能Oracle。该历史轮选择的下一实验是当时已暂存的[重复save-off/on资格组](outputs/admission_capacity/20260914_repeated_staged_probe_r01/STATUS.md)：检验多次增量保存、恢复、flush/fallback及完整请求成本。当时GPU_UNRUN、等待Qwen r04；后来经新机迁移以r02完成，当前见上文；即使通过，方法主张仍需同环境immediate most-output强基线。后续资源状态以[等待回执](outputs/admission_capacity/20260914_repeated_staged_probe_r01/RESOURCE_WAIT.md)和现场协调为准，原STATUS的本地准备记录保留。

历史接续于2026-09-14 23:23（UTC+8）记为 `BLOCKED_RESOURCE_BUSY`（已由2026-09-15新机执行解除）：连续三轮实查Qwen r04的6954/6967仍存活、分片增长且共同锁占用。原重复保存包已暂存，尚未启动；本地准备和分析入口已完成。此状态只表示等待前序整组释放GPU，科学结论仍为OPEN。前序终态后重验实际PID/GPU/锁与源码，再按既定队列首次执行，不能根据日志静默或初始化空闲重启/并跑。

## 最小模型

在线请求状态包括：到达时刻、阶段、已计算/已输出长度、实际持有块、最近输出时刻、抢占及恢复状态。资源状态为实际 KV pool、空闲块以及块引用计数。动作只使用该时刻可见状态；降低准入 cap 不会释放已运行请求的块。

关闭共享缓存时，候选恢复的额外块需求是 `max(0, ceil(已有完整 token 历史 / block_size) - 持有块数)`。一次交换必须满足当前空闲块与实际可释放块足以资助恢复。开启共享缓存后，持有块数不等于可释放块数，需按引用及缓存生命周期重新资格化，不能直接解除现实现的检查。

资源守恒：`F_next = F_now - allocations + frees`。时间守恒：`wall = scheduler_inclusive + engine_excluding_scheduler + outside_engine_calls`。decision 已含于 scheduler；混合重算调用同时推进其他 decode，其全跨度不能称为纯重算/GPU成本。请求停顿分为最后输出至抢占边界、恢复前等待、恢复调用跨度；各策略独立演进 KV、请求、路由和输出。

合法动作、正确性、动作可选性与净收益分别判断。只枚举候选可行性不形成 performance Oracle；本页目前没有全动作空间上界。

原生offload还必须区分“已占块/已计算”和“可执行”：worker完成通知到达scheduler后，下一步才可将WAITING_FOR_REMOTE_KVS提升。已有[状态回放](outputs/admission_capacity/20260914_load_ready_contract_r01/REPORT.md)用真实通知验证状态机，不能把这些未来通知作为在线预测输入。混合重算调用中的正常新输出必须随状态推进；[选择性保存重复](outputs/admission_capacity/20260914_selective_store_repeat_r01/REPORT.md)少6591重算位置却只少2次总调用，说明不能从token降幅直接推导墙钟收益。

[固定成本模型的cohort3迁移](outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/analysis/cost_transfer/analysis.json)未重拟合旧native三项系数。它对most/native能描述平均完成变慢、最后完成变快的方向，但第一对residual/fit的条件均完成及末完成方向均错：估计+0.170/+0.143s，实测−0.348/−0.211s。输入是各自真实未来schedule，起点固定step299，不是完整请求指标或动作Oracle；误差足以淹没小改动，不能用它替代真实候选执行排序。

## 已知证据与尚缺环节

- 增加实际 KV 已消除旧端点的秒级暂停，但改变资源。旧 native 最长停顿约 96.4% 来自恢复前等待；这不是跨策略不可减少的下界。
- [最新强基线八项](outputs/admission_capacity/20260913_rotation_strong_baseline_r01/RESULTS_WESTC_ADDENDUM.md)中，most_output 将最大 ITL 从 4.572/4.760 秒降到 1.520/1.020 秒；相对 native 吞吐 −1.362%/+3.241%，平均完成 +5.962%/+1.031%。同资源 fast-headroom 已比较，原生服务量非劣仍未证明。
- 两次 most_output 的实际调度/输出路径相同，但一条恢复调用为 0.766/0.032 秒，不能从日志沉默推断 JIT，也不能删慢样本。同输入/runtime四项重复已完成，旧长调用未复现，但分散wall差仍在；不把两次新结果当噪声界或根因解释。
- 旧原生 baseline 关闭 APC；新四格已测原生APC开/关，开启后两次各复用1008位置，重算7685→6677，但恢复步骤不变。见 [APC实测补记](outputs/admission_capacity/20260914_prefix_cache_baseline_r01/RESULTS_ADDENDUM.md)。
- 已有动作查新表明抢占、aging、恢复优先、资源预留本身不足以构成新颖性；新增共享 [直接近邻核对](experiments/admission_capacity/RELATED_WORK_RECOVERY_20260914.md)还指出LTR、Andes、UniBoost、TokenFlow已有等待提权、净切换收益、有效服务保护及传输排队/合批处理。LTR200/10与prefix组件的同runtime适配已经执行，但完整LTR及其他近邻系统尚未复现，不能把组件适配或APC兼容称作新方法；当前尚未验证独立贡献或代表性连续服务/质量/SLO收益。

- 共享 [d6强基线八格](outputs/admission_capacity/20260914_d6_strong_baselines_r01/REPORT.md)已完成，APC off/6656块：most相对least整批吞吐+2.75%/+9.58%，但平均完成+8.03%/+0.60%；原生平均更好。其新因果分支由A会话接续研究，本会话不重复实现或跨APC配置拼接排名。

- CPU 前态核对已关闭一个实现解释：26个已观察交换前态的506个合法victim全部可资助恢复，未见先排序再检查资金漏掉动作；仅限已测无共享缓存域，见 [候选资格](outputs/admission_capacity/20260914_rotation_candidate_feasibility_r01/REPORT.md)。

## 执行历史（各组保持原合同）

1. 已接续 [四项同路径时间重复](outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/REPORT.md)原冻结包。首次尝试因跨会话启动重叠在初始化失败，零请求测量；[失败补记](outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/INITIALIZATION_FAILURE_ADDENDUM.md)与原件保留。
2. `execution02_after_finite` 四项已完成并回传128/128请求；[实测补记](outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/RESULTS_AFTER_FINITE_ADDENDUM.md)：most/native吞吐+4.087%/−1.640%、平均完成+0.258%/+6.310%，最大ITL4.885/4.711→1.065/1.067秒。
3. 首个forced恢复末调用本轮32.029/32.457ms，旧766ms长调用未见；两most仍同1162步/输出但wall差0.848秒。旧长调用解释不足，未定位主机/设备/GC/JIT因果，不追加同一重复或调参。
4. [APC 开/关原生强基线](outputs/admission_capacity/20260914_prefix_cache_baseline_r01/RESULTS_ADDENDUM.md)四项已完成：吞吐+0.498%/+0.040%，平均完成−0.576%/+0.067%，最大ITL4.663/4.687→4.686/4.709秒，仍是MEASUREMENT_ONLY。
5. [APC兼容四格](outputs/admission_capacity/20260914_apc_rotation_r01/RESULTS_ADDENDUM.md)已全部完成并回读，128/128请求。最大ITL4.760/4.673→1.047/1.042秒；吞吐+1.692%/−0.334%，平均完成+2.586%/+4.791%，31/32及32/32请求完成更慢；held=0，保护分支必要性未测。结论仍是MEASUREMENT_ONLY，不能宣布默认缓存下方法GO。
6. 共享A的d6 step329两次前态资格已完成：请求状态及每次13,619,429,376字节有效KV均相同，512分段中496非空、0差异。约12秒采集开销保留，不作性能结果。其least/most/defer单动作六格已由A完成192/192请求，本会话只读复用analysis.json。least/most均4调用、defer5调用恢复目标；most相对least平均剩余完成+0.040%/−1.842%，选中的0000001请求晚0.788/0.508秒，尚无稳定完整收益。数字从相同前态资格采集结束后起算，不能把约12秒采集暂停扣除后称为未插桩端到端性能；这是单事件条件后续，非全策略或Oracle。
7. [LTR组件首次四格](outputs/admission_capacity/20260914_ltr_component_probe_r01/RESULTS_ADDENDUM.md)已执行/回读128请求；同custom资源后端boost off/on/on/off，d6/APC off/6656块、固定200/10，非完整LTR。吞吐−2.188%/+3.215%，平均完成+2.838%/−3.273%，最大ITL4.692/4.905→4.928/4.868秒，未见一致完整收益。每次on仅一个有效量子，10次连续调度实际产出7新token，未见量子耗尽而零新输出。on/off每对29/32输出完全相同，3条轨迹不同，质量未测；各arm内部repeat32/32输出相同。主问题仍OPEN。一次fresh同族审计P0/P1=0；旧日志仅证明引擎初始化结束后存在JIT hook事件；因覆盖应用预热且warning_once也可能含磁盘缓存加载，不能定位为measurement内编译或归因wall波动，见[JIT语义勘误](outputs/admission_capacity/20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md)。
8. [最长gap定位](outputs/admission_capacity/20260914_ltr_component_probe_r01/analysis/gap_localization/REPORT.md)完成：3571在518–520只重算2985位置，held14步后535再次被抢占；直到649才返回下个token。245调用的输出间隙被两段113/125步未选中分开，未到200提权阈值；未获新输出的部分重算重置了idle。
9. 新发现最近邻实现差异：原LTR在首个不可容纳候选处停止，当前共同后端continue fit-scan。不能把该后端的事件升级为论文算法缺陷。先由原组件方完成fit-scan/rank-prefix强简单对照，root准备统一首次执行；恢复完成义务方案保留CPU草案但暂停封包/GPU，prefix尚未测出残留前不增加保护机制。GPU无当前占用声明，封包后重新协调。

该历史轮 Repository HEAD 为 `c4ae4f5daaea929e1a4358862296d832f1cc67ab`，继承共享 dirty workspace。已读 `docs/current/README.md`、`docs/ideas/README.md`、共享台账、目标最新结果与 GPU 协调记录。不开新题、不自动修改权威入口或 push。

10. [prefix首次四格](outputs/admission_capacity/20260914_ltr_packing_r01/RESULTS_ADDENDUM.md)已完成128请求：prefix/fit吞吐−1.865%/+1.900%，maxITL4.835/4.933→6.732/6.577秒；实际恢复后0新输出再抢占每次2→6。该强简单对照没有消除恢复中断，仍非完整LTR。现恢复[首输出边界消融](outputs/admission_capacity/20260914_restore_completion_r01/EXECUTION_CONTRACT_ADDENDUM.md)CPU实现；只保护已开始恢复至首新输出，不调200/10，不声称方法GO。

11. [首输出恢复四格](outputs/admission_capacity/20260914_restore_completion_r01/RESULTS_ADDENDUM.md)128请求完成；两on共54义务全部以首新输出解除、零中断，maxITL6.669/6.887→4.387/4.141s。但重算74982→96955，各对27/32请求maxITL更差，吞吐−7.104%/+2.309%。完成性成立、服务摊销未成立；不作方法GO，下一仅CPU核对能否分离恢复KV保留和执行优先级，暂不追加GPU或量子扫描。

12. 首个共同406前态已完成[一步资源核对](outputs/admission_capacity/20260914_restore_completion_r01/analysis/localization/first_dispatch.json)：F148，恢复剩余142块，30个resident pending1工作共增3块；994+30在1024预算内可行、无额外victim。实际guard_all在406/407单给恢复1024，408已恢复混合服务；409首输出释放后411再次抢占是另一事件，不混淆归因。下一[预算分配修正](outputs/admission_capacity/20260914_restore_token_reservation_r01/REPORT.md)仅CPU封包：同组fit/guard_all/guard_residual反序六格，不调200/10。当前live HEAD由共享会话更新为de64dae5，各原包SHA不变。

13. 六格包 f2e20f73…cac7450 已首次暂存并启动，前序 A/B 各整组终态及现场空闲已核对，共同 flock 覆盖整组。首输出四格 fresh 审计 [PASS](outputs/admission_capacity/20260914_restore_completion_r01/audit/EXPERIMENT_AUDIT.md)，P0/P1=0；旧 UNRUN/campaign 文案作为历史保留，以 RESULTS_ADDENDUM 与实际开关为准。新六格尚未形成 GPU 结果，不将 1906 步旧规则 CPU 等价回放称为新策略收益。

14. [预算六格](outputs/admission_capacity/20260914_restore_token_reservation_r01/RESULTS_ADDENDUM.md)已回读192请求/11252步全部有效。residual/guard_all吞吐+4.351%/+1.926%、均完成−4.711%/−3.525%；对fit则吞吐−0.161%/−2.653%、均完成+0.753%/+2.613%，maxITL4.920/4.823→4.339/4.356s。实际54步修正、恢复25/25均兑现；第一406可服务的30请求全部产新token，残留最大gap97.42%仍在恢复开始前。当前是权衡，不是方法胜出；不再延长恢复保护或调200/10。下一唯一比较：排除前128文档的新cohort3，原native/most/fit/residual及反序，CPU准备中/GPU UNRUN；同原语料分布，不称跨总体。

15. 新[服务窗口分账与模型](outputs/admission_capacity/20260914_service_window_r01/LATEST.md)已复用最新14格：residual仍5段仅得5新token，17505恢复位置随后再次执行。真实首个短段409前态F0，all31继续1/2/3步需3/5/6新块；固定FCFS fit子集28/26/25含target可在原块内继续，但六peer和原waiting请求承担额外等待。仅局部CPU资格，不能认定加长保护有收益；10定向测试通过。本分支无GPU/driver，不重复Controller，接续点仍为上行新cohort3强基线八格。

15. [新cohort3四臂反序八格](outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/REPORT.md)已首次启动：native/most_output/fit_scan/guard_residual，不改既有规则；固定6656块，原始128文档均排除。冻结包b7557c2e…018516d、24项，实际d6/T的18项engine参数及底层模块已核对相同。当前RUNNING，无新文档收益结论；只检验强简单基线覆盖，暂不加Controller。

预算六格的[限定审计](outputs/admission_capacity/20260914_restore_token_reservation_r01/audit/EXPERIMENT_AUDIT.md)完成：WARN / integrity pass、same-family/provisional，P0/P1=0；同cohort、质量未测及条件因果边界保留。被审查的报告/原件保持原哈希，不追加T审计轮次。

16. 服务窗口协作分支完成[cohort3强基线及全阈值复算](outputs/admission_capacity/20260914_service_window_holdout_r01/REPORT.md)。原统一八格全COMPLETE，native/most均无1–2输出后再抢占段；most的wall与全局maxITL两轮优于residual。全部间隔断点的Q(g)上residual未越过native/most两条完整策略的较优边界（两轮0/42），不能继续把组件短段称为强基线残差。当前组件窗口扩展停止，主问题OPEN；非联合SLO或family NO-GO。下一科学验证应针对持续到达/异构/真实EOS中强简单策略的失效边界，不改200/10或短段阈值救方向。本分支无GPU新组。主机预算[实地边界](outputs/admission_capacity/20260914_host_budget_r01/REPORT.md)：共享父cgroup90GiB/只读无委派，独立进程树硬预算仍未实现。


17. 服务窗口分支完成[持续到达/EOS四格](outputs/admission_capacity/20260914_streaming_recovery_r01/RESULTS.md)：64完整自然文章、8GiB可用KV、N/M/M/N各64完成。自然抢占2/5/2/0、额外轮转全0；9恢复段均持续到完成，996–1024新输出/段，无再丢弃，3次属于首输出前恢复。KV曾短暂耗尽但无持续恢复循环；该档不增加窗口或调整压力参数。每格6实际EOS末尾/58长度截断；共享父90GiB仅观察、未独立硬隔离。原冻结包/失败/原件保留，状态MEASUREMENT_ONLY，非方法GO。下一复用原组件已暂存funding-filter六格先检验最简单资源资格修正，遵守既有Qwen整组队列，不重建控制器。


## 专家分页成本支线交接（2026-09-15）

B本轮[有界成本诊断](outputs/admission_capacity/20260912_wisp_olmoe_r01/retention_cost_diagnostic_r01/REPORT.md)已完成：保护的当次加载不变，收益与被挤出专家的代价均在后续；多一组仍可能条件少搬，不增组也可能条件多搬。完整请求变化主要位于引擎调用内，尚无可将条件字节机会定价为暴露时间的实测分离。当前cap24 frequency/guard不继续参数扫描；成本模型和逐身份分叉工具交接，未声明paging无空间。后续shared-pool、fullstage、fresh、logical P/V均已完成，直接复用已有停止边界，不再执行旧计划。

唯一在途[Qwen r04](outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/attempt04/REPORT.md)也已完成并释放GPU：真实超单卡显存BF16模型、同资源单引擎32/16/16/32，16请求执行/416输出；static16多搬4.801%/3.000%、TTFT增加39.545%/34.436%、最大ITL下降6.741%/14.407%，平均完成+9.062%/−6.432%。描述性权衡，无稳定完整效率优胜或质量结论。B本轮有界任务完成，无新controller或GPU队列；后续实验由主会话按同预算强基线后的明确成本空间决定，不另接管resident恢复/保存主指标。
