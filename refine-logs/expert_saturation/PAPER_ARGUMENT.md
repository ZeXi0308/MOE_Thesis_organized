# 论文论证｜2026-09-15 保存、启动与独立输入验证闭环

**问题与状态。** 给定 GPU 和明确 host 预算，怎样安排恢复及后续执行，在控制生成长停顿时改善完整服务效率？现有证据为 OLMoE / vLLM 0.26 的 `NATIVE_SERVING / MEASUREMENT_ONLY`。主问题 OPEN；尚未形成充分验证的独立方法或 MoE 专属贡献。

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

**贡献缺口与唯一下一工作。** H 将“简单启动节奏只在选定输入有效”推进为“独立输入上仍满足预声明取舍，但幅度缩小”，没有因此产生新算法贡献。[直接近邻](experiments/admission_capacity/RELATED_WORK_RECOVERY_20260914.md)已覆盖等待提权与服务量控制；剩余决策是：相同保存、预算和目标下，经合理校准的兼容 LTR-style 是否覆盖该取舍？现已完成互斥原生 adapter 的 CPU 接入。定向反例纠正提前锁住不可行 target，以及[全体增长预筛](outputs/admission_capacity/20260915_joint_growth_decision_r01/I_PREPARE_SUM_GUARD_ADDENDUM.md)：后者推迟合法 prepare，却没有避免同一 peer 被抢占。修复进入共同可比的基线，不归因于官方 LTR，也不算服务收益。CPU 使用真实固定 native 调度循环和明确的 allocator/输出/传输替身；实际 GPU 保存、加载与量子生命周期尚未验证。它不是完整 LTR。停止扩展当前冷却/窗口/增长预测器，冷恢复份额候选不追加 GPU 组。

[有界校准选择](outputs/admission_capacity/20260915_joint_growth_decision_r01/LTR_G_CALIBRATION_DECISION.md)已固定为 T∈{30,200}、Q∈{1,10}，以同窗口 eager 为参照；仅在吞吐不低于 97%、平均完成不高于 105% 的合法完整候选中选择最小最大 gap。没有合格点就保留负结果，不扩网格。预测为固定 Q10 时 T30 比 T200 轮转更多且最大 gap 更低，两项分别检验。G 用于开发；H 不用于该参数选择，但已经见过，不能称盲测。性能执行组尚未接受。

**执行与接续。** H r01 零测量资格失败保留，H r02 已完成归档且当时明确释放 GPU。I r01 首次连接失败；随后对两个已提供入口的一次有界诊断均在 SSH 认证前关闭，未提交凭据、上传或初始化，测量 0，资源状态 UNKNOWN。现已接受 [I r02](outputs/admission_capacity/20260915_natural_ltr_style_component_r02/README.md)，包 6888c318…4c9e73 / 30 文件，只删除上述一条额外预筛，复用旧 CPU 检查并明确新增反例的边界。r02 未尝试连接、无后台执行。唯一下一项是有效入口恢复并现场核验后运行 G64/T30/Q10 单格 native 生命周期诊断。[统一清单](CURRENT_EXPERIMENT.json)记录版本、预算和唯一执行方；执行方负责原件与主分析，模型方负责可检验预测，近邻方负责公平基线，root 负责目标与论文取舍。
