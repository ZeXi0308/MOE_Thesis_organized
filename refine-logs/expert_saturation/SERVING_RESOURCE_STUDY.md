# 固定 KV 预算下的请求暂停与服务量

本页承接长任务会话 `01a09ba8-3b21-7853-89fd-83c41e309c07`，记录同一个研究问题及真实接续点。历史总裁决仍由 `docs/current/README.md` 与各 sealed verdict 管理；本页不修改它们。共享结果只读引用 [RESULT_LEDGER](experiments/admission_capacity/RESULT_LEDGER.md)，原研究详记在 [RESEARCH_NOTES](experiments/admission_capacity/RESEARCH_NOTES.md)。

## 问题与范围

**固定实际 KV 预算下，怎样减少已有请求的长暂停，同时保住完整请求服务量？**

当前可验证范围：RTX 5090，OLMoE-1B-7B BF16，vLLM 0.26.0，32 个请求，每个 3072 输入 / 1024 输出，50 ms 到达间隔，token budget 1024，KV 16,089,350,144 bytes / 7,671 可用块。早期轮转关闭 prefix cache，后续已完成APC开启的同配置对照；不把早期配置称为默认 vLLM。主要结果并列报告完整请求吞吐与每请求最大 ITL，成本包括 TTFT、平均完成时间、全部失败和未完成请求。没有业务 SLO 时不事后反选阈值宣布成功。

主问题保持 `OPEN`；现有结果为 `NATIVE_SERVING / MEASUREMENT_ONLY`。机制是实际抢占、恢复优先与恢复期间的资源保护；当前实现使用固定冷却和受害者排序。具体实现或排序失败，不等于问题被否定。Expert paging 是同一总体资源问题的另一运行域，其他会话的结果另列，不与本页 resident 长上下文数字直接排名。

## 最小模型

在线请求状态包括：到达时刻、阶段、已计算/已输出长度、实际持有块、最近输出时刻、抢占及恢复状态。资源状态为实际 KV pool、空闲块以及块引用计数。动作只使用该时刻可见状态；降低准入 cap 不会释放已运行请求的块。

关闭共享缓存时，候选恢复的额外块需求是 `max(0, ceil(已有完整 token 历史 / block_size) - 持有块数)`。一次交换必须满足当前空闲块与实际可释放块足以资助恢复。开启共享缓存后，持有块数不等于可释放块数，需按引用及缓存生命周期重新资格化，不能直接解除现实现的检查。

资源守恒：`F_next = F_now - allocations + frees`。时间守恒：`wall = scheduler_inclusive + engine_excluding_scheduler + outside_engine_calls`。decision 已含于 scheduler；混合重算调用同时推进其他 decode，其全跨度不能称为纯重算/GPU成本。请求停顿分为最后输出至抢占边界、恢复前等待、恢复调用跨度；各策略独立演进 KV、请求、路由和输出。

合法动作、正确性、动作可选性与净收益分别判断。只枚举候选可行性不形成 performance Oracle；本页目前没有全动作空间上界。

## 已知证据与尚缺环节

- 增加实际 KV 已消除旧端点的秒级暂停，但改变资源。旧 native 最长停顿约 96.4% 来自恢复前等待；这不是跨策略不可减少的下界。
- [最新强基线八项](outputs/admission_capacity/20260913_rotation_strong_baseline_r01/RESULTS_WESTC_ADDENDUM.md)中，most_output 将最大 ITL 从 4.572/4.760 秒降到 1.520/1.020 秒；相对 native 吞吐 −1.362%/+3.241%，平均完成 +5.962%/+1.031%。同资源 fast-headroom 已比较，原生服务量非劣仍未证明。
- 两次 most_output 的实际调度/输出路径相同，但一条恢复调用为 0.766/0.032 秒，不能从日志沉默推断 JIT，也不能删慢样本。同输入/runtime四项重复已完成，旧长调用未复现，但分散wall差仍在；不把两次新结果当噪声界或根因解释。
- 旧原生 baseline 关闭 APC；新四格已测原生APC开/关，开启后两次各复用1008位置，重算7685→6677，但恢复步骤不变。见 [APC实测补记](outputs/admission_capacity/20260914_prefix_cache_baseline_r01/RESULTS_ADDENDUM.md)。
- 已有动作查新表明抢占、aging、恢复优先、资源预留本身不足以构成新颖性；新增共享 [直接近邻核对](experiments/admission_capacity/RELATED_WORK_RECOVERY_20260914.md)还指出LTR、Andes、UniBoost、TokenFlow已有等待提权、净切换收益、有效服务保护及传输排队/合批处理。它们的同runtime style基线尚未执行，不能把本轮APC兼容称作新方法；当前尚未验证独立贡献或代表性连续服务/质量/SLO收益。

- 共享 [d6强基线八格](outputs/admission_capacity/20260914_d6_strong_baselines_r01/REPORT.md)已完成，APC off/6656块：most相对least整批吞吐+2.75%/+9.58%，但平均完成+8.03%/+0.60%；原生平均更好。其新因果分支由A会话接续研究，本会话不重复实现或跨APC配置拼接排名。

- CPU 前态核对已关闭一个实现解释：26个已观察交换前态的506个合法victim全部可资助恢复，未见先排序再检查资金漏掉动作；仅限已测无共享缓存域，见 [候选资格](outputs/admission_capacity/20260914_rotation_candidate_feasibility_r01/REPORT.md)。

## 当前一条执行链

1. 已接续 [四项同路径时间重复](outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/REPORT.md)原冻结包。首次尝试因跨会话启动重叠在初始化失败，零请求测量；[失败补记](outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/INITIALIZATION_FAILURE_ADDENDUM.md)与原件保留。
2. `execution02_after_finite` 四项已完成并回传128/128请求；[实测补记](outputs/admission_capacity/20260914_rotation_runtime_repeat_r01/RESULTS_AFTER_FINITE_ADDENDUM.md)：most/native吞吐+4.087%/−1.640%、平均完成+0.258%/+6.310%，最大ITL4.885/4.711→1.065/1.067秒。
3. 首个forced恢复末调用本轮32.029/32.457ms，旧766ms长调用未见；两most仍同1162步/输出但wall差0.848秒。旧长调用解释不足，未定位主机/设备/GC/JIT因果，不追加同一重复或调参。
4. [APC 开/关原生强基线](outputs/admission_capacity/20260914_prefix_cache_baseline_r01/RESULTS_ADDENDUM.md)四项已完成：吞吐+0.498%/+0.040%，平均完成−0.576%/+0.067%，最大ITL4.663/4.687→4.686/4.709秒，仍是MEASUREMENT_ONLY。
5. [APC兼容四格](outputs/admission_capacity/20260914_apc_rotation_r01/RESULTS_ADDENDUM.md)已全部完成并回读，128/128请求。最大ITL4.760/4.673→1.047/1.042秒；吞吐+1.692%/−0.334%，平均完成+2.586%/+4.791%，31/32及32/32请求完成更慢；held=0，保护分支必要性未测。结论仍是MEASUREMENT_ONLY，不能宣布默认缓存下方法GO。
6. 共享A的d6 step329两次前态资格已完成：请求状态及每次13,619,429,376字节有效KV均相同，512分段中496非空、0差异。约12秒采集开销保留，不作性能结果。其least/most/defer单动作六格已由A完成192/192请求，本会话只读复用analysis.json。least/most均4调用、defer5调用恢复目标；most相对least平均剩余完成+0.040%/−1.842%，选中的0000001请求晚0.788/0.508秒，尚无稳定完整收益。数字从相同前态资格采集结束后起算，不能把约12秒采集暂停扣除后称为未插桩端到端性能；这是单事件条件后续，非全策略或Oracle。
7. [LTR组件首次四格](outputs/admission_capacity/20260914_ltr_component_probe_r01/RESULTS_ADDENDUM.md)已执行/回读128请求；同custom资源后端boost off/on/on/off，d6/APC off/6656块、固定200/10，非完整LTR。吞吐−2.188%/+3.215%，平均完成+2.838%/−3.273%，最大ITL4.692/4.905→4.928/4.868秒，未见一致完整收益。每次on仅一个有效量子，10次连续调度实际产出7新token，未见量子耗尽而零新输出。on/off每对29/32输出完全相同，3条轨迹不同，质量未测；各arm内部repeat32/32输出相同。主问题仍OPEN。一次fresh同族审计P0/P1=0；旧日志仅证明引擎初始化结束后存在JIT hook事件；因覆盖应用预热且warning_once也可能含磁盘缓存加载，不能定位为measurement内编译或归因wall波动，见[JIT语义勘误](outputs/admission_capacity/20260914_ltr_component_probe_r01/JIT_LOG_SEMANTICS_ADDENDUM.md)。
8. [最长gap定位](outputs/admission_capacity/20260914_ltr_component_probe_r01/analysis/gap_localization/REPORT.md)完成：3571在518–520只重算2985位置，held14步后535再次被抢占；直到649才返回下个token。245调用的输出间隙被两段113/125步未选中分开，未到200提权阈值；未获新输出的部分重算重置了idle。
9. 新发现最近邻实现差异：原LTR在首个不可容纳候选处停止，当前共同后端continue fit-scan。不能把该后端的事件升级为论文算法缺陷。先由原组件方完成fit-scan/rank-prefix强简单对照，root准备统一首次执行；恢复完成义务方案保留CPU草案但暂停封包/GPU，prefix尚未测出残留前不增加保护机制。GPU无当前占用声明，封包后重新协调。

本轮 Repository HEAD 为 `c4ae4f5daaea929e1a4358862296d832f1cc67ab`，继承共享 dirty workspace。已读 `docs/current/README.md`、`docs/ideas/README.md`、共享台账、目标最新结果与 GPU 协调记录。不开新题、不自动修改权威入口或 push。
