# 论文工作稿

`main.tex` 以 **Determinants of Speculative Decoding Efficiency in Resident MoE Inference** 为题，组织实际验证形状、路由占用、草稿覆盖率、服务指标边界和能耗证据。当前是探索性 determinants characterization；eager r02 与 compiled r01 两套完整服务矩阵、同 kernel eager/CUDA-graph 组件时间拟合、配对执行域对照、带插桩 compiled 工作集及实际在线 acceptance observer 已测。完整 compiled 五臂比较已完成 40 个真实 run、零排除；当前 n-gram 域 adaptive 未胜过选定 static0 的等权 TPOT 目标。自然停止小检查已完成但不证明质量等价或任务正确；神经草稿/其他模型域仍待完成。

工作格式为 **IEEEtran 会议双栏**；当前编译版为 **9 页**，含完整五臂结果、全部图表与参考文献。使用 TeX Live 2025 自带 `IEEEtran.cls` V1.8b（2015-08-26）、`conference,letterpaper` 默认 10 pt 和 `IEEEtran.bst`，未修改官方字体大小、正文边距或栏距。作者/单位仍为明确占位文字；正式投稿前补真实信息并复核目标年度规则。当前稿超过常规8页，后续整合多域证据时再压缩篇幅；格式完成不代表研究验证完成。

转换前的 11 页自定义版保存在 `versions/characterization_pre_ieee.tex`、同名 `.pdf`/`.bib`；`characterization_pre_ieee_manifest.json` 记录原始 SHA256，未覆盖旧证据。旧 `.tex` 的相对图路径按本 `paper/` 工作目录解析。

当前源码新增 F3 实测五臂 TPOT 图及 adaptive/static 三指标小表，F4 policy counter-span energy 与 F6 decision/discard 结果用精炼正文报告。完整两次原值保留在图和 CSV；没有把 future-prefix 固定轨迹参考作为实际策略或 oracle 曲线。神经草稿域和全 observer 扰动仍未完成；自然停止小检查保留质量限制，已有 eager/compiled 历史证据保留。

本次 `latexmk` 编译与逐页渲染检查完成，无未定义引用、overfull、裁切或图文重叠。F1 位于第4页，主 compiled F2 与历史 eager F5 位于第5页，历史 eager F4 位于第6页，kernel 与T2在第7页，compiled路由和策略取舍表在第8页，实际F3在第9页；F4/F6实际策略图和actual adaptive F5另有完整外部文件。F3使用 `figures/F3_executed_policies.pdf`，宽度0.64302 textwidth，保持原嵌入图的坐标轴实际尺寸。`export_f3.py` 仅从原PGF中删除一行与正文/图注重复的脚注，全部数据及其余绘图命令不变，原始独立图保留完整注释；不用PDF裁切隐藏文字。导出manifest记录源文件和输出哈希。最终前8页在相同1200像素渲染下与上一版逐像素一致，第9页另行检查通过，文本提取中已无该重复脚注。当前整合前源码保存在 `versions/characterization_pre_five_arm.tex`，更早完整compiled服务整合前的8页稿保存在 `versions/characterization_pre_compiled_service.tex/.pdf`。

在本目录编译：

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

F3论文图已随源码提供；需要重新导出时，在本地项目根目录运行 `python3 -B paper/export_f3.py`。该步骤依赖本地 `code/plot_energy_progress.py`、TeX Live和Poppler，不是GPU运行步骤。

`main.pdf` 是编译预览，原始运行入口包括 `../runs/olmoe_ngram_matrix_r02/` 和 `../runs/olmoe_ngram_compiled_r01/`。前者 eager BF16 OLMoE/ShareGPT/n-gram campaign 已完成 7 个 B × γ0–7 × 2 repetitions，共 56 格、112 个 performance records。每条完整人口为 64 请求 ×128 forced-output tokens，112 条均完成 8,192 输出 token。GPU 持锁 3,091.24 秒，包括启动、预热和诊断。仅 B64 预热，cold shape 与运行顺序影响保留；不声称显著性或稳态 serving capacity。

Compiled r01 同样为 56 格/112 records，全部完成相同 64 个开发请求 ×128 forced tokens；GPU 锁 2,223.7864479348063 秒。V1 compiled 配置、scheduler cap64，逐 B/prompt-shape 32-token warmup，trace/native-return off，原生 observer 仍计入性能。它与 eager 处于不同运行时段和预热条件，不是 graph-only 因果干预。所有 gamma1–7 的 49 格 wave 吞吐低于各自 AR；TPOT speedup 有 15 格大于一（B16gamma4、B32/B64全部正 cap）。正文明确 summed wave windows 不含 inter-wave gaps；另用 full-population denominator 复算，49 格仍全部低于 AR。两个原始重复只作描述，不声称显著性。

证据入口：

- F1：`../figures/F1_olmoe_ngram_r02/route_prediction_scatter.pdf`，跨栏两面板。数据为 `../runs/route_prediction_r02/route_prediction.json`，γ0 的 3,472 层步拟合 q，γ1–7 的 23,568 层步执行留出；同 prompt subset 不是独立负载留出。actual S MAE 1.9145、nominal B(γ+1) MAE 15.2994、actual B × nominal cap MAE 13.6975。全部首步保留；首步占 3.33% 行、25.78% SSE，MAE 5.73 vs 后续 1.78。
- F2 主图：`../figures/F2_olmoe_ngram_compiled_r01/speedup_15c7a01f28d9.pdf`。同目录的 `tpot_speedup_15c7a01f28d9.pdf`、`joules_per_token_15c7a01f28d9.pdf` 为外部补充。精确统计为 `../runs/olmoe_ngram_compiled_matrix_analysis_r01/summary.csv`、`raw_runs.csv`、`numerical_brief.json`、`per_batch_ranges.csv`；零排除，56 格均有两次原值。B1/2/4/8/16/32/64 的正 cap 吞吐范围分别为 0.478835–0.727097、0.579847–0.864068、0.711088–0.857165、0.841169–0.874470、0.892255–0.918428、0.950430–0.962188、0.957835–0.979402。旧 eager F2 保留在 `../figures/F2_olmoe_ngram_r02/speedup_0_01dd8e1e.pdf` 及其原始 CSV。请求完成后不回填，下一 wave 等当前排空，不能称为连续 serving capacity。
- F4 eager r02：`../figures/F4F5_olmoe_ngram_r02/F4_energy_vs_batch.pdf`，跨栏绝对/相对 J/token 曲线，保留每格两个原始点。相对点除以同 B 的 AR 两次中位数，不是配对实验或 CI，未传播分母不确定性。相同 wave 窗口的 NVML device energy，未减 idle，也非整机能耗。原始热图保留在 `../figures/F4_olmoe_ngram_r02/`。
- F5 eager coverage/progress：`../figures/F4F5_olmoe_ngram_r02/F5_availability_progress.pdf`，跨栏三面板含非空草稿频率、可用草稿数和实际 emitted tokens；只标为覆盖/推进组件，不能冒充条件接受 survival 分布或 EWMA 检验。
- F5 离线统计诊断：`../figures/F5F6_tracking_exposure_ratio_olmoe_ngram_r02/` 保存新版 EWMA 接受计数/测试计数比值的固定轨迹重放，旧 `../figures/F5F6_tracking_olmoe_ngram_r02/` 保留。先验 accepted=0.7、tested=1，decay=0.9，无测试曝光不更新；tested 的实际代码口径为每请求 `min(accepted_prefix+1, scheduled_length)` 的和，排除首次拒绝后的未测试位置。该跨 rank 汇总比率不等于几何模型的 rank 条件接受概率；prefix/availability 仍按 eligible step 更新。B1 γ7 的测试位置加权 pre-step bias 从 −0.2285 到 −0.0137，但推进预测 MAE 约为 0.282，不能当作 adaptive 实测或性能收益。
- Kernel r03：`../figures/kernel_cost_r03/kernel_cost_validation_161b773f.pdf` 保留为原始 eager 验证图（当前正文使用下述 graph 验证图，eager 数值仍保留）。`../runs/kernel_cost_fit_r03/kernel_diagnostics.json` 和 `model_comparison.csv` 固定 88 fit / 66 held-out snapshots（B1/4/16/64 vs B2/8/32），实际 S1–148/D8–64。单层真实 BF16 权重 + 合成 activation/路由共现；event 区间 `D+c` 与 max-form 同解，留出 MAE 12.731 µs / MAPE 2.345%，S+c 为 21.673%，constant 为 29.517%。b 不可识别、c=0.153261 ms 仅此单层边界；不当作完整模型 c0、HBM 带宽或固定开销主导。独立单次 profiler 未再做 turnover，不用 event−profile 归因 CPU。
- Compiled r01：`../runs/compiled_route_summary_r01/summary.csv` / `summary.json` 和 `native_source_excerpt.txt`；`../figures/compiled_routes_r01/compiled_routes.pdf` 已作为跨栏主文图插入，正文配精简数字。六条件 γ0/4 × B1/8/32，各 64 唯一请求×32 输出，所有 16 层及 S/P/top-k/hist 守恒通过。γ4 的 valid→physical D 均值为 9.62→26.73、38.91→46.36、59.27→60.90；B1 的 1,569/1,747 步 S1→P5。P 是 model-forward 行数，不含 expert GEMM tile padding。V1 只有 decode mode 为 FULL 且 uniform query length>1 时才把共享 capture size 向 `G_eng+1` 倍数舍入；本固定 cap 诊断中 `G_eng=gamma`。此 OLMoE route flag 不改变该 runner/dispatch；原生 batch table 则会把含 FULL 的 V1 配置改为 PIECEWISE，不适用该舍入分支。仅相同 resolved mode/grid 和形状可沿用 padding 规则，不能外推 D 或服务时间；诊断 scheduler cap32，而实际 performance cap64。路由观测自身有 native 写入、D2H、hist/copy 成本。
- 原生表图模式更正的源码证据：`../runs/environment_20261016/source/vllm/config/vllm.py:819–836`（调用 1273–1274）检查 dynamic speculative config、含 FULL、非 V2 后改为 PIECEWISE；`../runs/environment_20261016/source/config/speculative.py:1337–1338` 的 dynamic 判据为 table `is not None`。`../runs/compiled_route_summary_r01/native_source_excerpt.txt` 保留 `config/compilation.py:1466–1474` 的 FULL-only 舍入条件。实际 `../runs/olmoe_compiled_controller_eval_native_unrestricted_r01/gamma7.log` 有 override、51 个 PIECEWISE capture、无 FULL，网格含 1/2/4；启动记录 6s/0.17GiB。对照 `../runs/olmoe_compiled_fullbudget_calibration_r01/gamma0.log` 为 48 PIECEWISE + 48 FULL，3s/0.21GiB。以上是启动捕获记录；trace-off 的计时 dispatch 占比与实际 P 未记录，不能据此把性能差归因于 P1/P8，也不能把启动时间当服务成本。
- F6 原生 r02 历史 CPU 段仍保留：schedule 占 host decode-step 时间 0.1285–0.3628%，acceptance 更新占 0.0356–0.0707%。当前主文加入真实 adaptive decision-call 计时和弃草稿结果，见下方五臂入口。wrapper/decision/schedule 是嵌套范围，不求和或从墙钟扣除；未计时 native/fallback call 不是零，全 observer 扰动仍待独立消融。
- 草稿覆盖与实际推进：`../runs/derived/draft_progress_r02_complete/`，保留 all_decode、nonterminal 等人口口径。available、scheduled、accepted 和 client emitted 不混用。
- 原生 threshold/unrestricted table 根据 direct-gamma 开发格选择并冻结，实际表引擎上限为 `G_eng=7`，但配置 table 后 V1 将含 FULL 的图模式改为 PIECEWISE，跳过 FULL-path 的 `G_eng+1` capture-size 舍入。因此它与后截断 G7 校准不具有相同图模式/网格，选择也不是同一 capture 域穷尽调优或全空间最优；AR 和 native best-static 保留独立实测。后截断支付完整 G 草拟成本，不能替代强 native threshold。
- 完整 F5 实际在线观察：`../runs/F5_online_compiled_matrix_r01/online_acceptance.json` 与 `../figures/F5_online_compiled_matrix_r01/`。8 engines/112 runs，从真实 post-warmup 状态恢复 240,254 host steps 和 2,032 wave-end snapshots；7,168 terminal request-steps 删失、unknown 0。没有 logged adaptive decision prediction，native availability EWMA 未更新；这是观察与状态恢复证据，不是已学习 availability 或 controller 收益。正文仅加短句，完整图留外部。
- 实际 adaptive F5：`../runs/F5_online_compiled_adaptive_r01/` 核对 19,759 个 alpha-after checkpoint 与 170 个 wave-end state 全部一致。19,589 个真实 decision-progress 记录中 19,419 步可评分、58,152 eligible request-steps；512 terminal、unknown/recovered/token-proxy mismatch 均0。B1/4/16/64 两次加权推进 MAE 分别 0.231269/0.230631、0.176895/0.177053、0.060344/0.069529、0.066154/0.066131 tokens/request-step。这是实际线上预测对有噪声已实现观测的误差，不是 latent truth 或性能收益；正文仅补一句，不增加图。 外部图已完成：`../figures/F5_online_compiled_adaptive_r01/F5_online_prediction_errors.pdf` 与 `F5_online_state_example.pdf`，保留两次 raw 值和连续状态范围。
- Compiled 能耗同样用 summed-wave NVML counter windows：49 个正 cap 格均高于 AR。`full_population_raw.csv` 与 `cell_summary_with_population.csv` 保留全部 112 个有效 counter spans；加入 inter-wave gaps 后仍无正 cap 格低于 AR。counter span 从首 wave start 至末 wave end，端点与 runner population_wall 口径不同。

外部完整成本与固定轨迹诊断（不作为 F3 实测曲线）：

- `../figures/fullbudget_cost_eager_r01/complete_step_costs.pdf`；数据见 `../runs/olmoe_fullbudget_cost_analysis_r01/numerical_brief.json` 和 `fixed_budget7_vs_native_costs.csv`。固定 G=7 的 eager 完整同步 host-step 成本覆盖 32 格，每格仅 1 次运行，计入验证、下一轮草拟和 host/controller 工作。与 r02 native 格的差值只作描述：两组输出长度、重复数、预热、观测代码和时窗不同，不能归因于额外草拟预算，也不能把单重复小差当作稳定动作排序。
- `../figures/stepwise_prefix_eager_r01/stepwise_prefix_diagnostic.pdf`；数据与边界见 `../runs/derived/stepwise_prefix_eager_r01/README.md`、`diagnostic_summary.json` 和 `plot_run_values.csv`。在 eager r02 γ7 的 14 条固定轨迹上，28,306 个步骤支持全部八档比较；每 run 冷先验重放的事前选择与同一步未来接受前缀参考比较，分数为校准成本/平均请求推进量，部分成本按 B 插值。这不是实际在线策略、有效 TPOT、端到端收益或理论上界；动作改变后的后续轨迹未验证。两次轨迹分数相同不消除成本不确定性，小分数差也不能证明在线信息无价值。

Compiled 固定 G=7 校准现已完成：`../runs/olmoe_compiled_fullbudget_calibration_r01/status.json` 为 complete，32 格各 1 次运行，持锁 760.8426720499992 秒。准备产物为 `../runs/controller_preparation_compiled_r01/`；`calibration/calibration.json` 含 32 个完整同步 host-step 成本，SHA256 为 `305a7835a12caaa2dc68bb48fbebd07d772eb63e1b85012280c6973220d57271`，已与 `evaluation_commands.json` 核对。成本已就绪不代表候选策略已获验证。

Compiled成本图已交付：`../figures/fullbudget_cost_compiled_r01/complete_step_costs.pdf`；逐格值、原生两次运行及排序见`../runs/olmoe_compiled_fullbudget_cost_analysis_r01/numerical_brief.json`。32格、36804个eligible步骤、680个wave，复算与协调器表除source路径外完全相同。B1/4/16/64成本范围为7.285–7.521/8.078–8.732/9.989–11.184/12.683–13.797 ms；每格仅一次校准，不把细小非单调差当作稳定排序。两campaign的runner/plugin/cost源码一致，但64/128输出、general warmup0/80（shape均0）、预算与时窗不同，差值不是纯截断开销。

独立compiled参考图为`../figures/stepwise_prefix_compiled_r01/stepwise_prefix_diagnostic.pdf`，数据在`../runs/derived/stepwise_prefix_compiled_r01`；14个run的28432个eligible步骤覆盖8个动作，只使用compiled成本和轨迹。未来prefix参考与过去信息冷先验重放的score差不等于实际TPOT或策略收益；B1零差依赖此单次成本表中cap7最低成本且弱最大推进，不证明全局最优。B32/64重放比分别固定cap7更差的值保留；两次raw汇总重合不意味着独立不确定性消失。两张图已视觉检查；现已完成的实际 F3 使用独立执行结果，继续与这些诊断分开。

冻结时的 `selection/selection_manifest.json` 与 `selection/regime_e7cb8fe905/selection.json` 保留原始 selected_pending_evaluation 状态；当前执行完成状态以 `../runs/olmoe_compiled_controller_comparison_r01/comparison.json` 为准。按 B1/4/16/64 等权开发 effective TPOT 选择：静态 gamma0；dense-style 原生阈值 cutoff0，即 `[[1,64,0]]` 全关；原生 unrestricted 表为 `[[1,15,0],[16,63,4],[64,64,7]]`。AR 与所选静态已分别使用独立 G0 引擎实测；全零阈值仍配置 G_eng=7，且 table 触发 V1 PIECEWISE override，既不能当作独立 G0 AR，也不能当作后截断 G7 的 FULL_AND_PIECEWISE 图配置。表由 direct-gamma 开发格估计选择，非同 capture grid 内穷尽调优，选择分数不是实际策略结果。五臂均已完成各 8 个 run，合计 40 个真实记录、零排除，canonical comparison 为 complete。原生 table 在零预算时跳过草拟，并强制 PIECEWISE；adaptive 先生成 G7 再截断，图/生产边界均不同。这是 whole-policy 完整比较，不能将差值归因于纯选择规则、P1/P8 或单项预算成本。两臂相同配置下的跨臂512/512个完整128-token序列一致，固定长度比较入口为 `../runs/olmoe_compiled_baseline_generation_check_r01/`。static/AR的TPOT中位数比为0.998118/0.999103/0.995163/1.012773，J/token比为0.994698/0.997112/1.002674/0.967094（B1/4/16/64）；这是局部顺序运行波动，不是策略收益或通用噪声阈值。AR B1两次能耗差约8.67%，原值保留；自然停止小检查现已完成，具体计数与质量边界见下段。

完整五臂实测入口（F3/F4/F6）：`../runs/olmoe_compiled_controller_comparison_r01/comparison.json`、`policy_runs.csv`、`policy_summary.csv`、`weighted_objective.csv`、`control_runs.csv` 和 `control_summary.csv`。图在 `../figures/F3F4F6_olmoe_compiled_controller_r01/`：`F3_effective_tpot_92cb309cdb53.pdf` 已加入主文；`F4_counter_span_energy_92cb309cdb53.pdf`、`F6_control_92cb309cdb53.pdf` 为外部完整图，主文报告实际数值和边界。图保留每格两次原点，线是 run median，没有 CI；同一 resident engine 与连续状态不是独立引擎重复。

等权 B1/4/16/64 TPOT 为 AR 7.714910、static0 7.740541、native unrestricted 7.917877、threshold 8.141005、adaptive 8.989071 ms。Adaptive/static 的 TPOT 比依次为 2.227642/1.273748/1.033728/0.959581，完整人口 TPS 比为 0.460511/0.745835/0.916799/0.948388，counter-span J/token 比为 2.393173/1.350436/1.055272/1.083353。B64 的 TPOT 改善未带来吞吐或能耗改善；这一 n-gram 域没有支持 adaptive 的总体目标优势，不称显著或普遍结论。人口 wall 包含 inter-wave gaps；NVML counter span 同样包含 gaps，但端点不等于 runner wall，未减 idle，也非整机能耗。

F6 的 adaptive timed choose 调用均值在各 run 为 39.70–52.14 µs。Discarded/available 比例为 B1 0/0、B4 0.422666/0.422666、B16 0.731301/0.689627、B64 0.430243/0.430243；decode 人口含 terminal/drain。弃草稿数量不是已测 GPU 时间或能量，嵌套 wrapper 不求和。评估为 workload entries64–127，开发选择为0–63，shape warmup来自开发人口；历史 general warmup 已见过48个评估prompt，所以不称 whole-pipeline prompt-unseen。下一步继续已排队的神经草稿/其他目标和代码/摘要负载，availability 与成本均需实测，不因当前候选未胜过 static0 而宣告路线完成。

自然停止小检查入口为 `../runs/olmoe_compiled_natural_generation_check_r01/summary.json` 、`inspection.md` 与 `manual_inspection.md`。已完成 12 cells：AR/static/adaptive × B1/B8 ×2 repeats，每格相同16请求，failed/unfinished/missing 均0。B1 的 AR/static 每次 EOS4、256-token cap12，adaptive 每次 EOS2、cap14；B8 三臂每次均 EOS3、cap13。抽检 B1/repeat0 的固定 workload offset64 起前8个源样本，不按结果选：source288 三臂均超出≤50词要求（空白切分60词、正则59词）并补写用户未提供的个人成就；不据此断言其现实真假，source305 三臂均未等用户请求就给提示，296/297 代码因256上限未完成要求功能；多数固定样本触顶。这些事实不支持质量等价、内容正确或生成分布等价，零运行失败也不替代任务质量。

正文同时报告有限 wave 吞吐与请求级有效 TPOT：后者为 first-to-last chunk 间隔除以输出 token 数减一，再在 run 内请求取均值、跨 run 取中位数。B16/64 的 γ6 已显示二者差异。`../runs/derived/wave_completion_r02_g026_b1_4_16_64/` 的 receipt-time 分解已加入：B16平均post-first span减少0.70788s，被新增barrier项0.71303s抵消；B64分别为0.25573s和0.20864s。此项不是可追回时间上界，queued vs wave实测仍待运行，不能直接外推稳态收益。原自定义版的均匀模型演算随旧版保留，原 8 页版已去掉该非实测段落。最近邻及建模口径见 `../PRIOR_ART.md`。

历史编译记录随 `versions/characterization_pre_ieee.*` 保留；本页顶部仅记录当前版本。Online EWMA 在同一 engine 的 waves/B/repeats 间连续，离线 cold-prior replay 不冒充在线状态。

显存边界已修正：当前 baseline/神经 draft 计划使用同一物理设备容量和显式 6 GiB KV cap；联合权重、metadata 与实际 KV 仍需报告，未通过缩放 KV 配平相同实际总驻留量。

已测资源表在`../runs/engine_resource_summary_r01/engine_resources.csv`：三个OLMoE/ngram campaign共24个engine，eager/compiled初始化后PyTorch allocated约19.9828/19.9908 GiB，KV cap均6 GiB；启动model-loading日志12.89 GiB，compiled graph-capture日志0.10–0.21 GiB。快照在应用warmup前，但内部加载/图捕获后；这些是不同口径，不是NVML驻留、峰值或可相加的内存分解。后续神经草稿的联合资源按真实日志补入，当前不借资产文件大小推测运行峰值。

Compiled 预测窄补充：`../runs/compiled_route_prediction_r01/` 固定 γ0 的 B1/8/32 逐层拟合 q（每层 5,952 token），同 prompt 的 γ4 执行留出。B1/8/32 的 physical D MAE 从 q+S 的 17.675/6.307/1.818 降到 q+P 的 3.133/2.898/1.464；B1 仍有 +2.562 bias、12.864% MAPE。21,920 个同 wave 相邻且均 padded 的 layer-step pairs 中，padding histogram 完全相同；此跨步重复本身不判定步内独立性，也不证明 token 顺序或 cache 命中。正文仅新增短段，未改 eager F1、未拟合新模型或新增图。

## 目标会议与工作格式（2026-10-16 核对）

CCF 最新正式目录为**第七版**（2026-03-31 发布，04-09 勘误）；**ICPADS、HPCC、ICA3PP 均为 C 类**。目录仅覆盖正式 Full/Regular 长文，short/workshop 不替代主会目标。[CCF 发布说明](https://www.ccf.org.cn/Academic_Evaluation/By_category/)、[CCF 当前分类表](https://www.ccf.org.cn/Academic_Evaluation/ARCH_DCP_SS/)。

| 选项 | 与当前稿的适配判断 | 2026 官方稿件要求 |
| --- | --- | --- |
| **ICPADS** | AI Infrastructure and Systems 明列高效推理、AI 负载表征及 LLM 系统，最贴近当前稿。 | IEEE Proceedings 双栏、单倍行距、10 pt；8 页含图表/参考，可付费至 10 页，初稿硬上限 10 页；single-blind。[CFP/格式](https://www.cloud-conf.net/icpads2026/cfp-research-paper.html) |
| **HPCC** | 性能建模、HPC for AI、Green HPC 适合成本/能耗主线。 | IEEE Proceedings；regular full 8 页含图表/参考，可付费至 10 页；single-blind。官网另列 6 页 full 类别，此处按 regular full。[官方 CFP](https://ieee-hyper-intelligence.org/2026/hpcc/) |
| **ICA3PP** | 性能建模、AI 工作负载分析、异构架构相符。 | Springer LNCS 单栏、single-blind；Full 投稿至少 12 页，未明示初稿上限；Regular 终稿 20 页含参考，可付费加 4 页，不能反推初稿上限。[投稿要求](https://hpcn.exeter.ac.uk/ica3pp2026/call4paper.php)、[终稿要求](https://hpcn.exeter.ac.uk/ica3pp2026/Camera-Ready-Paper-Instructions.html) |

**推荐 ICPADS 主会 Regular Paper 的 IEEE 双栏工作格式，按 8 页组织，10 页仅作允许上限。**这是主题适配判断，不保证录用。现已按此工作格式将原 11 页自定义 `article` 转为 IEEEtran 8 页版，保留证据边界和旧版备份；格式就绪不等于研究验证完成。

2026 常规投稿截止分别为 ICPADS 08-15、HPCC 08-05、ICA3PP 07-25，均已过（见各 CFP）；此为初稿格式选择，实际投稿年度与规则须按下一届正式 CFP 复核。

T1/T2 整理：主文新增 `tables/T2_actions.tex` 动作级小表，覆盖 MoESD、Cascade、AdaEAGLE、vLLM 原生和当前实现的信息、动作及草拟成本边界。这不是这些系统的完整数值复现。正文的实际 controller 比较范围改为五臂 AR / native static / native threshold / native unrestricted table / ours；Cascade 仅作已有机制对照，不承诺尚未实现的数值基线，也不声称优于它。

`tables/T1_assumptions_directions.tex` 是暂不嵌入主文的可更新证据表。原 full-availability、alpha=0.7、纯专家权重读取的翻转只是有条件演算；r02 的实际 n-gram availability 与完整服务成本不满足前提，因此没有一个可硬对比的精确预测翻转点。表中分别记录 OLMoE eager 小 B 正收益及大 B 的方向、compiled 49 格吞吐低于 AR、组件与 compiled 路由边界；第二模型 Qwen 和已有第三模型计划仍为 GPU 待测，不从下载资产或 OLMoE 借填结果。

T1 已独立临时编译检查，未作为主文结果表嵌入；T2 随主稿编译。页数与当前视觉检查状态见顶部。

CUDA-graph kernel r01 已纳入：原始 `../runs/olmoe_kernel_graph_r01/measurements.jsonl`；独立拟合及精确配对表位于 `../runs/kernel_cost_fit_graph_r01/`。154 个 case 的路由、权重、环境、kernel、tile、seed、输入与 256 MiB timer 外 turnover 配置全部匹配。88 fit/66 held-out 不变，graph D+c 为 `T[us]=9.01937D+55.73326`，留出 MAE 5.86753 us / MAPE 1.92402%；max-form 为 5.84972 us / 1.92048%，仅 1/88 fit 点走 S 分支、66 test 全走 D 分支，不宣称广泛 compute 项已识别。旧 eager D+c 留出 12.73086 us / 2.34506% 和全部原始证据保留。

正文 kernel 图改为 `../figures/kernel_cost_graph_r01/kernel_cost_validation_258f5f5f.pdf`，继续展示留出预测与强简单模型；实际配对图 `../figures/kernel_cost_graph_r01/kernel_eager_graph_paired.pdf` 作为独立补充入口保留。配对 event graph/eager 中位数 0.79237955，范围 0.48987057--0.86692429；wall 比中位数 0.79430932，154/154 均更低。这些是不同时间窗口中的固定输入单层执行域对照，没有交错随机的独立运行重复或 CI，不是 isolated CPU/Python launch 成本、完整模型 padding 或服务收益。Capture/first replay 不计入正式 repeats，单独记录；旧 r03 profiler 没有 turnover，而 graph profiler 有，禁止差值归因 CPU。T1 保留组件边界；F3 的新实际结果另见上方五臂入口。

接受率定义窄修正：按只读 `code/cost_model.py` 的真实 `AcceptanceEWMA.update`，主文区分几何假设 `alpha_g` 与实际 pooled EWMA 比率 `alpha_bar`，并写出 accepted/tested 分母。Gamma=1 时 tested 与已观察草稿槽数一致，gamma>1 不可直接混用。策略主路径仍为 empirical prefix survival × availability；pooled alpha 只作为初始/未见 rank 的几何先验，并非已验证的 rank 条件概率。该定义修正不改变运行时；后续已纳入完整 compiled F2 与在线 F5 观察，未混用早期 partial 图。当前页数和检查状态见顶部。
