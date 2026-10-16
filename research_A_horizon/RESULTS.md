当前主张：完整初版F2未压过最佳cap，F3显示固定策略取舍依赖负载；继续实证研究，不宣称负载无关优势。
关键证据：r10完整策略/关闭恢复保护的maxgapP99=10.162/5.942s、stall=13.28%/16.80%、thrash=0/6；MC LOAD零抢占、117次有效放宽，TTFTP95=26.308s、gapP99=0.576s。均为单次开发运行。
最强替代解释：调好的静态cap已覆盖收益；MC资格限制、启动编译及新请求LOAD账本重复计费风险影响策略归因，后者实际影响尚未量化。
下一动作：保留Qwen native校准34980；新接续43621待其完整结束并本地归档后，用code-r13执行OLMoE六组前沿；归档60690继续。旧协调器34979已结束，勿恢复或重提。
资源状态：累计持锁4682.358s（78.04min）；r11零格候锁超时已本地归档；r11b等待窗口6h，不计入GPU执行预算。

# HORIZON 研究入口

任务来源：用户附件 `bf3da24c-53a3-4be3-9489-d9c8ef8f3b8b/pasted-text-1.txt`。完整目标保留为可安装/卸载调度模块、两模型×三负载×两档 KV 预算真机实验、F1–F6/T1–T3、LaTeX 长文及逐图复现入口。当前为第 1 周实现阶段，未完成整个目标。

## 固定评价口径（首张新图之前）

主目标为每请求最大相邻输出间隔的 P99，以及 maxgap > 0.2 s 的请求占比；首 token 前等待单列 TTFT，不混入 maxgap。代价为 TTFT P95、输出 token/s、外部到达到完成的均值与 P95。联合 goodput 使用 TTFT ≤ 5 s 且 maxgap ≤ 0.2 s 的已完成请求数除以完整观察时间。这些是本课题固定服务目标，不回写成历史实验的生产 SLO。

全部到达、失败、超时和未完成请求均计数；缺输出或截断轨迹不能当作 maxgap=0。自然 EOS 与 length 终止分别报告，不事后筛等长度。历史 A/A 仅提示均值微小差异不可区分；本机 A/A×3 尚未运行，不能迁移历史噪声地板。运行级比较，不把 token 当作独立重复。

## 2026-10-16：事实快照与实现起点

见 [FACTS.md](FACTS.md)。实际远端 scheduler、KV manager、offload scheduler 的 SHA 与 pinned 源码一致。OLMoE-Instruct revision `7f1c97f440f06ce36705e4f2b843edb5925f4498` 的三片权重存在；Qwen3-8B 尚未准备。旧 B 最大 donor 组的 receipt 仍为 RUNNING，但原 PID 83207/85923 均不存在，当前无实验进程；这是旧运行中断/不完整收据，不是仍在候卡，也不是 HORIZON 结果。没有重提旧组。

裸 PYTHONPATH 导入旧 overlay 缺 libcudart.so.13；复用历史 plan 的 LD_LIBRARY_PATH 后 CPU 导入 exit0：torch 2.11.0+cu130、vLLM 0.26.0、CUDA 未初始化。无需重装或修改共享安装。

实现前发现三个必须准确计费的边界：LOAD 命中仍占目标 GPU 页；R 与 native inflight reservation 重叠，不能重复扣；D 的流体近似不自动构成逐页安全保证。默认 H=16/32/64 为块大小倍数时固定 decode 集合的 D 恰好准确，但新增/恢复目标及非整倍数 EWMA 视野需要单独计费。论文先写条件性容量命题，不能直接声称原提示词中的全局无抖动引理已证明。

CPU 数学内核已实现，三个检查通过，输出见 `runs/cpu-model-20261016.json`：250 请求/32 步在全部16种页相位均需500页；逐页视野计费与 B 的 T/R/G 参考一致且重叠只计一次；victim 跳过租约、到期和全租约回退成立。1步时流体估计16页而同步边界实际需250页；250 peers 加1个恢复目标在32步需要502页。它们支持修正计费，不能直接证明动态准入下的全局无抖动。

原生 install/uninstall、STORE 公共修复、runner 与实验均尚未完成；真实 KNOWN 快照、非尾 victim 本轮回滚还未检查。没有以数学检查替代真机服务评价。下一实现直接接原生 hook，不重做历史探针。

## 2026-10-16：原生接线与首格提交

上述“尚未实现”为前一阶段状态。现已新增 `code/horizon_sched.py`，提供 `install(scheduler,cfg)` 返回的适配器及 `.uninstall()`；只改实例，不改共享安装。原生 waiting allocation 之前做门控，成功后登记租约；尾部逆序跳过租约、全租约时回退；已排入本轮的非尾 victim 退还 token、删除 scheduled maps、修正 req_index。STALL 只改连续 PREEMPTED 前缀，没有 once/8 总动作预算。已加入所有臂共用的零 LOAD 重算游标回退，原生 STORE builder 和 ACK 保持不动；其真机行为仍待验证。

三项 CPU 入口 `code/check_native_cpu.py` 已通过，结果 `runs/cpu-native-20261016.json`：32 步增长场景；从完整 pinned Scheduler 提取的改选/回滚分支实际执行；历史 native KNOWN 同入口快照算术为700页，与 B 原账本一致，UNKNOWN 回原生。它们不是完整 engine 正确性证明，接着直接做真机。

实现范围如实记录：当前支持 B 的同步 Scheduler（async LOAD 保留），尚未支持 C 的 AsyncScheduler；MC 臂需另接既有适配。容量账本每 waiting phase 做一次 O(n) 扫描，查询是 O(1)，非整步 O(1)；逆序 victim 和前缀排序仍为线性操作。EWMA 接 client receipt，但尚未测敏感性。动态无抖动保证未成立。完整两模型矩阵、其他基线、消融及图表仍待完成。

W2 第一版本复用 B 的256篇自然文本，按 Instruct chat template 加文档分析指令；交替512/3072 token，max_tokens1024、ignore_eos=False、min_tokens0。属于声明构造的诊断混合，不冒充生产请求轨迹。初次构造因当前 transformers 的 chat template 默认返回 BatchEncoding 失败；显式 `return_dict=False` 后256请求全部生成，未消耗GPU。

首格计划 `configs/w2-48-native-calibration-r01.json`，native、48 GiB、cap256、10 req/s；测量最多240s，首格包含编译的进程上限600s。runner 整组持公共锁、每格检查数据盘≥10GiB、无重试，完成后先释放锁再归档。远端 session `/root/autodl-tmp/research_A_horizon/runs/w2-48-native-calibration-r01`，controller4035，日志 `runs-launch-r01.log`。已提交，不可按本条重新执行；连接中断先核实4035/receipt与实际子进程。

## 2026-10-16：首个有效自然 EOS 校准结果

r01 初始化失败、无测量请求：新 runner 漏带历史 `VLLM_USE_FLASHINFER_SAMPLER=0`，触发 FlashInfer 架构检查错误。已按原 B runner 补齐 sampler/offline 环境；不改共享安装。r01 controller4035/child5606已退出，独占36.195s；失败归档 SHA `2a7859241197f79d8fdcbbc5784850f999c7956fa9e860b6b1ff6c9755f6edeb` 已取回，未抹掉失败。

r02 使用相同输入与10 req/s，controller6089/child6092均终态、exit0，独占233.196s。256/256请求完成、无失败或未完成，215自然stop、41达到length上限，输出179324 tokens。全部数据在 `runs/w2-48-native-calibration-r02`，原始归档4,405,558B、SHA `ba8c964f967036ecb44e09f7fdca5b9f10bd61049d0ce3367c2d51131c4f96c3`；两端hash一致后清理该已归档session的远端raw/缓存，远端留配置、收据与日志。

| r02 native 指标 | 值 |
|---|---:|
| 被抢占请求 / 总数 | 8 / 256（3.125%） |
| 抢占次数 / 恢复后16输出内再抢占 | 22 / 11 |
| maxgap P99 / stall率 | 1.071124 s / 15.625% |
| TTFT P50 / P95 | 0.230646 / 16.537061 s |
| 完成时间均值 / P95 | 35.169864 / 48.014307 s |
| 输出吞吐 | 2613.445 token/s |
| 联合 goodput | 2.331820 req/s |
| 最后512 token出现周期≤32的请求数 | 1 |

这是单次开发校准，不是候选收益或质量等价结论。计时包含所有观测与初始化后的实际JIT影响；测量初期记录了一次 fused_moe_kernel JIT 警告，没有事后扣除。后续比较保留相同运行设置，必要时针对形状预热成本单独分析。已确认完整安装/卸载在 native配置可执行，但候选 A/L/V 门控尚待真机。

按事前5–15%压力目标，3.125%不足，`calibrate.py`给出下一native=20 req/s。已提交新组 `configs/w2-48-calibration-policy-r03.json`，顺序native→HORIZON(Ha64,Hr32)，其余输入/48GiB/cap256不变；两格只作校准与候选首次真机，不是独立确认。远端 session `/root/autodl-tmp/research_A_horizon/runs/w2-48-calibration-policy-r03`，controller6840，首child6843；公共锁整组占用，实际组前可用20,364,115,968B。**已提交，不重提**。若观察中断，先查6840/6843及receipt；当前没有r03服务结果。

F1历史动机图已生成、目视检查并进入LaTeX：`figures/F1/{plot.py,data.json,F1.pdf,F1.png}`。native重复恢复及三条LOAD样本与另一受控运行的62页增长分别标注，流体62.5仅作数量级参照，不称为引理验证。图与当前校准不混为同组证据。论文编译通过，仍是未完成初稿。

## 2026-10-16：λ 校准完成与首个 HORIZON 对照

r03 两格 exit0，均256/256完成。20 req/s 的 native 抢占17/256=6.64%，落在5–15%区间；固定 OLMoE/48GiB/W2 的 λ=20，第二压力档40。此选择仅使用 native 抢占比例，不依据候选收益。归档8,695,000B、SHA `6019a08881799a6218e8696c567033929ca7433862b70e745e38ec0c87258f85` 已本地核对并解压，远端已清理对应cache及主要raw，配置/收据/日志保留。

| 指标 | native | HORIZON Ha64/Hr32 |
| --- | ---: | ---: |
| maxgap P99 (s) | 11.9948 | 7.5126 |
| stall率 | 12.8906% | 14.8438% |
| TTFT P95 (s) | 26.8218 | 26.4462 |
| output tokens/s | 2698.64 | 2668.23 |
| mean flow (s) | 39.5701 | 40.5541 |
| 抢占请求 | 17 | 10 |
| 恢复后≤16输出内再抢占 | 4 | 0 |
| scheduler μs/step | 1175.31 | 1493.38 |
| stop / length | 221 / 35 | 219 / 37 |

候选161次拒绝新准入、174次拒绝恢复、19次重排；255次target-boundary UNKNOWN退回原生。没有记录到victim跳过，不能据此把收益归给租约victim。尾部降低与stall率升高并存，不能只报P99即宣布主目标全面改善；输出终止分布不同，也不主张生成工作量相等。当前仅探索性一对，运行间不确定性未定。

r04 `configs/w2-48-native-aa-r04.json` 已提交，controller8450，3个相同native格测本工作点波动；检查其现有receipt，禁止重复提交。r03持锁273.344s，前三组累计542.736s，不含其他会话占用。

基线接入本地已实现：`code/baselines.py` 使用C原始三文件的原有installer、Gate及FIFO实现，副本与来源SHA放 `code/vendor_c/sources.json`。所有臂共享STORE和生命周期观测；MC为同步scheduled-prefill资格适配，非C v25异步系统完整复现。尚未真机验证。容量指标更名为peer_committed_ratio和peer_horizon_growth_ratio，后者扣除R/I；采样人口明确为KNOWN gate attempts，不能当逐步均匀采样。

基线验证组r05已以独立版本 `code-r05` 部署，tar SHA `6ea49fa119f2ce840253c60a747f2e62eb1002d20f6687ff94c1ac3b4f175a50`；没有修改r04正在使用的code目录。controller9348，共2格full-budget/MC，整组等公共锁；失败即中止，不自动重试。原有三项CPU检查仍通过，paper加入r03混合取舍并编译通过5页，仍为工作稿。

门控覆盖补充：r03候选的KNOWN新准入417次尝试覆盖全部256个请求（各至少一次允许）；KNOWN恢复185次尝试覆盖10个恢复请求（各至少一次允许）。255次UNKNOWN不能解释为255个请求从未经过门控，具体回退边界尚未逐次归因。

r06前沿扫描计划已在本地准备，尚未提交：static cap=128/160/192，Ha=16/256/1024；复用r03的Ha64点和r04 native重复。cap三档围绕当前约190并发压力及更低占用选取，不根据cap结果事后挑点。仅在首轮全臂齐全后判断是否优于最佳cap；Ha16<Hr32为边界探索，不满足条件命题的视野关系。

## 2026-10-16：本工作点 A/A 完成，基线与前沿扫描推进

r04三格exit0，均256完成，原始归档13,344,034B，SHA `371d0a00f4d19f553fc1e18de4273e6a145c689206b61d61f53b370a51b042ef` 已本地核验。仅清理该已归档远端session的主要raw/warmup/cache，留receipt/config/log；持锁410.831s。原始重复值与样本标准差见 `runs/w2-48-native-aa-r04/variability.json`。

| native重复 | maxgapP99(s) | stall率 | flow均值(s) | output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16.2612 | 17.1875% | 41.1338 | 2653.55 |
| 2 | 15.0478 | 14.4531% | 40.2976 | 2672.49 |
| 3 | 12.2045 | 21.0938% | 40.1332 | 2672.87 |

前三个native被抢占请求为19/19/23（7.42%/7.42%/8.98%），继续支持λ20压力域。首个HORIZON的flow40.554s、TPS2668.23落在重复范围内，按任务口径记为不可区分。其stall14.84%也在该范围内；早先“stall提高”仅指r03那一对，不是稳定方法效应。P99低于这三次native，仍不足以从一个候选格确定总体收益。三次重复只描述本工作点，不把极差作为通用显著性门槛，不追加重复追求显著。

r05 full-budget首格9788正在实测，MC随后执行；r06 controller10198已提交，等公共锁，使用独立code-r05；6格计划不变。F2脚本 `figures/F2/plot.py` 已生成当前可用点及T1逐运行表，缺失臂明确列出，不能将不完整图当最终前沿结论。

W1/W3定义为相同来源、相同文档分析指令下的全短(≤512 token)/长请求先到达后短请求的分组混合，W2为长短交替混合；都是诊断负载，非生产轨迹。F3固定W2所选cap/Ha与外部λ，不在新负载重调策略。builder增加family和source-model参数，跨tokenizer时用源tokenizer解码原始文档，避免用目标模型词表解释OLMoE token IDs；当前尚未生成新模型输入。

W3构造修正（生成与测量之前）：历史源只有128份长前缀，不能把已有短文档填充成256份长请求。采用同W2的256份文档及长度，仅排序为128长→128短，检验时间相关的工作集变化；不拼接、不重复、不填充。此前全长描述为未执行草案，已纠正。

## 2026-10-16：基线首轮完成与评价协议的具体缺口

r05两格exit0、各256完成，归档9,420,256B，SHA `807b6307adb5f1c2531f886d6326598bdb7f6400aea32133b9c2ff01348bccfb` 已本地核验；仅清理本session远端raw/warmup/cache。持锁278.343s。full-budget/MC分别：maxgapP99=0.54655/0.53656s、stall=11.72%/7.03%、TTFTP95=30.719/31.308s、TPS=2593.01/2583.42；均零抢占、零thrash。原生fit预测各511次、零不一致，两个installer均成功卸载。

MC的640次attempt全部 `not_new_local_prefill`，eligible=0、成功放宽=0；共同Instruct前缀在host命中后走新请求LOAD，原同步MC及async_mc.py都有同一新目标排除条件。因此不能靠换异步调度器解决，也不能称此格验证了MC未来峰值策略；图中将注明MC未触发。下一接入应保留原物理与旧请求条件，仅支持可证明保守的新目标LOAD：完整prompt页从入口计入，新请求被LOAD推迟时其未来增长不超过假设立即开始增长的包络。当前尚未实现/验证该扩展，不删掉原始MC结果。

full-budget raw中30个>0.5s间隔共享完全相同的host边界[1.591323,2.137876]s；cell日志测量开始后记录fused_moe_kernel JIT警告。支持一个共同启动停顿的解释，但不把日志时间直接等同设备完成。原预热32/2请求不足以覆盖后续decode形状。新增本地 `--warmup-profile broad`：先用256请求、末64输入token、固定64输出做公共形状预热，然后原32/2预热、drain/reset，再按原自然EOS测量。固定输出只用于预热，不改变测量任务；启动成本完整保留。该协议尚未部署/真机验证，r06继续原冻结版本，不事后减去已有JIT停顿。

F2当前7点数据包括未触发的MC，仍缺cap/Ha扫描；保留为开发图，不作最终优势结论。后续确认需区分legacy/broad预热，不混合成同一协议的重复。

## 2026-10-16：部分前沿回收、MC LOAD适配与空间等待

r06 controller10198已退出，四格均exit0，第五格开始前因数据盘<10GiB停止；Ha256/1024没有启动。完整归档17,796,158B、SHA `20c64c6d502da0cc0e2c57dae6a968922ad8a95b6fd555b102a27f1640e254bc` 已本地核对。早先前三格分段归档SHA `574315b3ae678f185088ae6a0c57d96a077c61742be091bf93e49be6dde7d32d` 亦保留本地；重复是临时空间交接，不是独立实验。末格期间曾压缩已完成格以准备清理缓存，可能引入主机竞争；成本未剔除，后续确认需统一干净协议。所有已归档r06主要raw/warmup/cache清理后，数据盘仍约4.96GiB。本目录此前仅432MiB，其他目录主要是正在准备的模型权重；未删除或修改它们。C结果入口记录用户正在扩容，尚未观察到容量变更。已请求空间协调，当前没有新GPU排队进程。

| r06臂 | TTFTP95(s) | maxgapP99(s) | stall率 | 输出tokens/s | 抢占请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cap128 | 32.3378 | 0.5637 | 10.156% | 2578.47 | 0 |
| cap160 | 26.9754 | 0.5397 | 13.281% | 2674.99 | 0 |
| cap192 | 27.0701 | 11.4621 | 19.922% | 2695.62 | 21 |
| Ha16/Hr32 | 26.8108 | 13.6066 | 16.016% | 2657.05 | 17 |

全部256完成。cap160是当前三档中较有竞争力的选择，但旧预热的共同JIT仍在，不把约0.54s当无抢占系统不可降低的稳态极限。HORIZON尚缺Ha256/1024，先补齐而非提前触发全方法成败判决。续跑计划 `configs/w2-48-frontier-r06b.json` 仅这两格，使用冻结code-r05；尚未提交，不重复r06已完成四格。

`code/mc_load.py` 已实现有来源的同步MC新请求LOAD适配：保留C原目标native fit、旧owner条件、peak_envelope、FIFO、全额虚拟账与源码pins，仅放宽尚未准入的WAITING请求、零历史计算/输出/持有GPU页、正external前缀且异步LOAD的目标资格。完整prompt从h=0计费，推迟新目标增长不超过旧包络假定的立即增长；不允许恢复请求、已准入LOAD或不合格旧owner。新增请求/实际成功计数区分请求动作和真正生效。动态函数转换的目标与super边界精确计数并编译通过，尚未GPU验证。

独立 `code-r07` 已上传，打包SHA `77f30883d91e6ade677e8e8f3e26888f0d9e5046aee6ebc062bc6ee3face38ef`；r07计划native/full-budget/mc-load三格均broad预热，尚未提交。现有code/code-r05保持冻结。broad预热是否覆盖全部形状尚未测量，不先声称JIT问题已解决。

W1/W3实际CPU生成完毕：W1全256条512tokens，W3长度512/3072，按document_id逐条核对W3提示词多重集合与W2完全一致。三负载输入包1,675,437B、SHA `a77add124238979dea8bbfdccc379e8a5715f33c5ac12301bb1099a4f2978b20` 已取回、核验并展开至configs。它们是输入就绪，不是GPU矩阵已完成。

T2六行历史窄干预表已从B原文及指标入口整理，`figures/T2/{build.py,data.json,table.tex}` 保留节名、行号与SHA。涵盖full-tail、once/repeat8、defer、yield、unique8与单donor交换，未重跑旧实验；每行区分观察与机制解释/限制。已加入paper/main.tex；8页工作稿编译通过，新增表格与前沿页面已渲染检查，修正表格字间距和图中相同符号的参数标注。当前仍有未完成实验与相关工作，不是最终论文。

## 2026-10-16：扩容核实与断点接续

重新连接后实测数据盘200GiB、可用158GiB，原磁盘门槛阻塞已解除。核实r06为终态后，仅提交缺失的Ha256/1024到独立r06b；当前Ha256 exit0，Ha1024运行中。r07以相同公共锁排队，依次比较broad预热下native/full-budget/mc-load。旧legacy与新broad协议分开解释，不把预热变化当策略收益。原始数据归档、hash核验后再清理自身缓存。

## 2026-10-16：完整初版F2与实证研究转向

r06b两格256/256完成、零抢占，Ha256 TTFT P95=30.013166s/maxgap P99=0.555594s/TPS=2627.345；Ha1024为34.864182s/0.570116s/2549.335。cap160在这两维均更低。小视野Ha16/64保持略低TTFT但残留长停顿，整条曲线未压过cap160，触发用户预定的实证研究分支；不再预设方法获胜，仍完成两模型矩阵、消融与图表。公共编译停顿及单次开发运行限制保留，不宣称统计优势或全域不可能性。

F2现13点，missing_arms为空；T1逐运行表同步更新，论文题目/摘要/评价/结论已转向。r06b归档8,997,116B，SHA `96ea9e17abe91db31abd6d40608db4b95e191201c8a60d1ee4c23bf4bdb7a26a`，本地解包分析完成；持锁278.036s。r07继续等待其他会话，不抢占其GPU。

## 2026-10-16：第二模型准备

Qwen3-8B固定revision `b968826d9c46dd6066d109eabc6255188de91218`，下载进程3373实际存活；远端直连huggingface.co网络不可达，hf-mirror.com可用。`code/prepare_qwen.py`下载后逐权重核对仓库LFS SHA、核对index引用完整性并记录全部文件SHA；尚未报告下载完成。只写本课题models目录，不复用C的其他Qwen量化模型。

官方config为36层、8KV heads、head_dim128，BF16每token KV=147456B，16token页=2359296B；engine实测页数仍是容量口径。builder新增显式`--disable-thinking`，写入template kwargs和temperature=.7/top_p=.8/top_k=20；测量入口按seed+输入index固定逐请求采样，默认OLMoE仍temperature0。依据官方模型卡：https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md 。源文档仍由OLMoE tokenizer解码后转目标tokenizer；不可把跨模型token长度匹配当语义完全一致。

C基线只在Qwen模型上扩展原model_type守卫，保留原source/layout/physical-fit断言并额外核对Qwen全注意力几何，安装后恢复原函数；vendor源码不改。上述代码通过语法及实际measurement source补丁编译检查，尚待Qwen真机初始化与行为验证，未替换排队中code-r07。

Qwen下载与r07启动短暂重叠；在确认r07 RUNNING后主动SIGSTOP下载3373，后续检查首格日志确认是否仅与初始化重叠，若侵入测量必须标明。组后恢复原进程，不重复下载。

后续收据显示r07 native cell已结束、full-budget已进入测量，故不能假定下载仅与初始化重叠。r07 native按下载I/O混杂的诊断运行处理，不用于确认预热效果的性能比较；full-budget是否重叠的准确STOP时间未记录，保守同样不作无干扰确认。mc-load在下载暂停后执行，可检查动作资格。无需抹除或重复整个组，待诊断结果决定必要的干净对照。

## 2026-10-16：预热诊断与F5/跨负载扩展

r07 native/full-budget测量阶段仍有fused_moe_kernel JIT警告，broad预热未消除该已知成本。保留结果并停止盲目扩展预热；后续主矩阵沿用legacy，启动成本包含在全服务口径内。MC动作资格待本格完成。

F5初版从F2同13个运行点生成，数据/脚本/PDF/PNG见figures/F5。Ha16/64/256/1024的growth预留中位数0.696/1.880/10.124/24.723%；HORIZON整调度器1414–1513us/step，native cap256为1156–1265。门控采样不是均匀每步采样，整调度器差异不是独立hook开销；线性账本尚未满足O(1)维护。已入论文，编译无warning/overfull。

r08计划7格：24GiB W2 native λ10起始校准；48GiB下固定cap160与Ha256/Hr32，在W1 λ20、W2 λ40、W3 λ20各一格，不跨负载调参。Ha256选择依据是W2扫描中首次消除抢占的视野，仍未胜过cap160。采用原code-r05/legacy协议，所有成败均保留。Qwen下载3373继续暂停，待r08结束再恢复，避免引入新的I/O混杂。

## 2026-10-16：MC LOAD适配生效

r07全部256/256完成：native TTFTP95/maxgapP99=28.775/14.638s，26请求被抢占；full-budget为32.342/0.575s、零抢占；mc-load为27.731/0.556s、零抢占，TPS2626.478。MC 603次评估、482符合资格、117次成功放宽，121次因admitted_not_all_running回退，365次峰值拒绝。动作已实际进入分配，不仅是打分；服务收益仍需同协议且无下载混杂的对照，不把本组前两格作为确认。broad后的测量期JIT仍存在，故不将本组并入legacy F2。

归档14,815,011B，SHA `8d3b58a6b93a4e80947fd29804d5b13fd9823181101c4f88e22ca67e2120e8d1`，本地完整核验并analyze；持锁424.549s。r08真实controller4274（shell job4273），首格child4278已运行。

## 2026-10-16：T3近邻动作核对与24GiB初点

T3九项原始来源已核对并记录版本/定位，见figures/T3/data.json及可重建table.tex。MC-SF原文包含shortest-first子集选择，本课题FIFO峰值组件不是完整MC-SF；UniBoost MemGuard有服务阈值前不驱逐保护，WAIT v4含未知长度安全缓冲。已据此改写相关工作，不能以租约或预留本身声称新颖。FastSwitch改变layout/transfer，SuperInfer针对GH200/C2C，Llumnix跨实例，均未冒充实测基线。

r08首格24GiB/W2 λ10完成，39/256=15.234375%请求被抢占、46次抢占，略高于5–15%校准目标；下一点λ5（r09配置已准备，未提交）。r08其余六格继续，不因首格超目标而覆盖或终止。

T3编译后检查第8–9页：跨页表头重复、所有九行可见、无裁切；修正FastSwitch列宽后LaTeX无warning/overfull。全文仍为工作稿，未宣称最终排版验收。r08已完成24GiB native和W1 cap160，W1 HORIZON格child5738运行中；整组结束前不复制变化中的完整归档。

## 2026-10-16：F3重建入口与消融准备

F3脚本按输入token SHA/模型revision核对同场景两臂，仅选择48GiB、legacy、cap160或Ha256/Hr32；W2 λ20为调参数据，W1/W2 2λ/W3为固定参数迁移。当前本地仅2个调参点，6个迁移点明确missing，未伪造完整图，也尚未入论文。r08完成归档后自动补齐。

F4及Hr敏感性计划r10共8格已生成但未提交：Ha64/Hr32参考、关闭恢复门控及victim租约（Hr0/lease_victim0）、关闭准入（Ha0）、关闭stall排序、Hr16、Hr64、EWMA、mc-load legacy。选择Ha64是因已观测恢复事件，避免在Ha256零抢占工作点把无动作当有效消融；迁移主比较仍用Ha256不变。新CLI开关明确对应实现；code-r10冻结包已生成，未覆盖运行中的code-r05。

code-r10 SHA `66b6a3af8f95671e2288ae7f30a6958a3bcdd55762e5f8fdf2d903f7cd00490c` 已上传并在独立code-r10目录核验解包/语法；r09/r10配置部署完成，均未提交。r08仍controller4274运行、当前cell3；Qwen3373保持暂停，须在r08结束后优先恢复。

## 2026-10-16：自然EOS恢复链关联

新增code/recovery_chain.py，将原生抢占返回、GPU页分配、LOAD job提交/主机完成通知、ACK应用、再次分配及首输出按请求/事件窗关联。旧trace缺少ACK/再次分配时间，明确null，不用host completion替代device finish。每次后续抢占切断episode，未恢复/未完成保留；汇总区间和子区间重叠，不能相加。

r03真实raw关联通过单调时间校验：native22个async_LOAD episode，preempt→allocation中位3.904869s，allocation→first_output中位0.276978s，LOAD submit→host complete中位0.072171s；H64为11个episode，分别6.665698/0.335374/0.044515s。不同策略选出的事件人口不同，不当作独立系统重复或因果配对；只能说明仅传输时长不足以概括这些停顿。指标新增recovery_chains，原服务指标不变，F2/F5来源SHA已重建。

本地horizon_sched增加稀疏recovery_ack_applied与recovery_reentry_allocated记录，不改原生ACK行为；未覆盖任何已运行或已冻结的code-r05/r07/r10。后续需以新快照执行验证新观测。

Qwen尚未运行前修正采样种子关联：当前run_cell使用seed+document_id哈希，而非输入index，保证W3仅重排W2文档时同文档保留同一随机种子。默认OLMoE greedy不变；这不保证不同批次的浮点输出完全相同。修改只在本地新版本，已冻结r10仅用于OLMoE而不受影响。

## 2026-10-16：第二模型与观测代码快照

code-r11 SHA `4e2604b0613a7e71029da8eefd3c9a440342fcc5fd2ad6cdabb54aeb6139637a`，40,233B，远端独立目录已核验部署。实际measurement源码补丁编译通过，文档重排后逐文档种子保持不变、OLMoE默认仍greedy。三项原生CPU检查上轮通过；新增ACK观测尚待GPU。现有r10快照保留，后续可使用r11运行相同消融配置并记录实际代码SHA。

r08前六格exit0，第七格child8384运行中，controller4274实际存活；Qwen3373仍暂停。尚未取回变化中的整组raw，结束后先恢复下载再归档核验与F3。论文最新编译无warning/overfull。

## 2026-10-16：F3完整固定参数迁移结果

r08七格均256/256完成，归档28,532,178B，SHA `f8b0decc6fd0d63b53115b0406998095ad586df138e7c35782586eb8e12d301f` 本地核验，持锁931.975s。F3八点齐全（两调参点+六迁移点），输入/模型一致性检查通过。

|负载|cap160 stall / H256 stall|cap160 TTFTP95 / H256 TTFTP95|
|---|---|---|
|W1|12.890625% / 12.890625%|11.180638s / 0.186348s|
|W2 λ|13.28125% / 13.28125%|26.975438s / 30.013166s|
|W2 2λ|2.34375% / 3.90625%|32.572747s / 34.809985s|
|W3|7.8125% / 0.390625%|36.947587s / 32.129201s|

W3的cap160发生22请求抢占，H256零抢占；W2 2λ则cap零抢占、H256有4请求抢占。长请求集中到达下cap不再保证与W2相同的KV工作集。单次迁移结果支持运行域相关取舍而非普遍优势；仍有公共JIT、输出长度/终止人口差异，未证明质量等价。W3 meanflow cap/H=46.282/47.578s，不能只保留有利尾部。W1 TPS4186.652/4576.459而flow20.789/21.947s，同样披露代价。

## 2026-10-16：Qwen等内容镜像续传

原HF镜像下载恢复后仍约MB/s级，两个partial共约1GiB，且曾有断连。原进程3373核实后SIGTERM，确认/proc消失，旧收据保留为qwen3-8b-download-first-attempt.json。ModelScope仓库五片权重size/SHA256与HF固定revision完全一致，采用逐文件Revision定位，在HF private cache内写入独立partial；新进程9620执行fetch_qwen_mirror.py，3线程64MiB Range，严格验证Content-Range、总长度、最终SHA后才转成完整blob。此为资产来源替换，不改变模型权重；完成后仍需prepare_qwen.py补齐snapshot并校验全部文件，未声称模型就绪。

核查更正：HF原下载的partial带随机后缀，镜像脚本使用规范SHA.incomplete名称，因此本次没有复用旧partial。核实时新文件已全部超过对应旧文件进度，保留运行中的9620，不重启下载。旧partial仍保留，待完整blob逐SHA验证后才清理。当前数据盘约125GiB可用。

## 全运行指标表 T1

`figures/T1/build.py`从各session的metrics和receipt重建`data.json`与`all-runs.tsv`；当前25格，24格完整、1格初始化失败。保留失败行且未产生的服务指标为null；不将不同预热协议或校准条件合并。r07前两格标记下载重叠诊断。每个源metrics保存SHA256；表含完整请求人口、尾延迟、停顿率、吞吐、完成时间、联合goodput、抢占/抖动、调度成本及自然终止/长度上限计数。当前只覆盖已测OLMoE运行，不能代替尚未完成的两模型矩阵。

## F4 消融分析入口

新增`figures/F4/plot.py`，固定读取r10执行计划并核对输入token SHA、模型revision及实际配置。前四格用于Full / 关闭恢复保护 / 关闭准入 / 关闭stall排序；其余Hr敏感性、EWMA与MC保留在数据及T1，不混入组件对比。关闭恢复保护同时令Hr=0与lease_victim=0，因此包含恢复门控与victim租约两部分，不能称为仅关闭victim。未有测量时仅写缺失状态，不生成虚构图。当前八格均未运行。

## 当前 Instruct 恢复链图

`figures/F1/recovery.py`从r03现有metrics抽取22个native与11个H64/32恢复episode，生成独立recovery-data.json/PDF/PNG并进入论文。native的抢占→分配、LOAD→主机完成、分配→首输出中位数分别3.904869/0.072171/0.276978s；H64为6.665698/0.044515/0.335374s。图支持“传输本身不足以解释长停顿”，不支持策略间匹配因果差异。区间重叠，不可相加；旧轨迹ACK缺失不填零。图与论文页已目视检查，LaTeX二次编译无Warning/Overfull；完整论文仍待剩余实验和最终验收。

## Qwen资产准备已接续

CPU进程12459执行`finish_qwen_assets.py`，等待已核实身份的下载9620退出，仅在镜像收据COMPLETE后执行code-r11中的prepare_qwen/build_workload。固定revision不变；W1/W2/W3均使用non-thinking模板及既定采样配置，W3与W2按document_id核对完全相同的prompt多重集。输入独占创建，不覆盖旧配置；失败停止，不自动重试、不提交GPU。收据models/qwen-assets.json、日志qwen-assets.log。脚本SHA256 d601003003f3e133fb00494846fc017e1657b992dd33a81d458ebf18873af068。当前状态WAITING_DOWNLOAD，尚未宣称输入就绪。

## 在途LOAD增长记账修正（本地，尚未真机执行）

检查pinned scheduler在allocate_slots后将async LOAD放入_inflight_prefills的路径，发现allocated()增量记账包含该owner的H步增长，但下一步_build()只保留其历史缺口。可复现边界：16-token历史、已持1页、H=32，增量demand=2页，重建变0。现统一为所有running∪inflight owner均计H步endpoint deficit；ACK后集合取并集避免重复收费。历史KNOWN快照仍为700页并通过；三项CPU门槛通过，新增LOAD分配→重建→ACK成员变化一致性检查。capacity参考将增量字段明确为owner_increment_blocks；旧B参考只含running增长的差异在检查中显式计算。

这是本课题adapter的记账修复，不是vLLM缺陷。已运行r01–r08和远端code-r11保持不变；旧结果不得直接标为修正版。后续须部署新快照并先用r10完整臂验证，再解释组件消融。修正不构成动态无抖动证明，也未解决每步O(1)维护要求。

code-r12已独立部署，SHA256 `38f8f9c3eed04842d8b19ce02e3cae20d41ac961e4f54f774dabecc476f89c88`；尚未执行GPU格。下载9620与资产准备12459均核实存活。

## r09/r10接续队列已提交

进程12998执行configs/launch-r09-r10.sh，当前经ps核实存活，等待资产进程12459退出并检查qwen-assets.json为COMPLETE。随后以code-r12运行r09（OLMoE24GiB λ5，1格）及r10（OLMoE48GiB λ20，8格）。每组沿用run_group的公共锁等待1800s、逐格磁盘/GPU占用检查及失败停止，不自动重试。此时尚未申请GPU锁、未执行新格。日志runs/queue-r09-r10.log；连接中断后先核实12998及两组receipt，不重复提交。T1新增实际command导出的code_snapshot字段；F4要求同组code-r12，防止修复前后版本混用。

论文实现与评价段已按当前证据更新：删除“尚无整机验证”和“扩展预热正在准备”的过时表述；保留原账本的实测边界，新增LOAD增长修复及GPU待验证说明，并披露r07下载重叠、JIT仍存在和MC LOAD实际触发情况。引言改为服务取舍及当前实证范围。二次LaTeX编译无Warning/Overfull，修改页已渲染目视核对；全矩阵与最终成稿仍未完成。

Qwen首轮校准配置`configs/qwen-calibration-r11.json`已部署但未提交：W2，native cap256，48GiB λ10与24GiB λ5，各256请求、自然EOS、上限1024。两档仅为初始探点，不继承OLMoE的压力判定；各自根据实际抢占比例决定下一次二分，目标5–15%。使用code-r12及qwen-w2.json；必须先核实qwen-assets COMPLETE和既有r09/r10终态，避免重复或模型下载重叠。run_cell会填入cap/output_tokens_by_request，Qwen采样参数来自负载配置。

T1已增加可编译LaTeX补充材料：运行索引、服务指标、请求人口与终止、机制/成本、预留采样五表，25格/六页；paper/supplement.pdf已逐页目视核对。初始化失败的未测值保留为空，不替换为0；r07重叠诊断与代码执行目录明示。此表仍是当前已测子集，不代表完整矩阵。

## 权重尾段并行下载

前三片、第五片与tokenizer已完成原SHA校验，仅第四片剩余约1.3GiB。资产进程12459先SIGSTOP，原下载9620随后SIGTERM并确认/proc消失；旧收据保存在modelscope-download-sequential.json。进程18225执行finish_qwen_ranges.py，对第四片保留前缀后的互斥64MiB区间使用4线程；严格核对206/Content-Range/长度，全部完成后顺序合并并校验整片SHA，随后恢复12459。失败同样写ERROR再恢复接续，让队列显式停止，禁止把缺失资产当完成。脚本SHA256 `67d758408bfc0b838b1358d2f7567afb53986827fb20489f149313b47a12f2af`。

## Qwen资产就绪，r09进入公共锁队列

四路尾段进程18225已正常结束，modelscope-download.json=COMPLETE；资产12459恢复后完成固定revision快照逐文件验证及W1/W2/W3构造，qwen-assets.json=COMPLETE且W2/W3逐文档prompt完全相同。BF16 KV=147456B/token。输入与收据归档qwen-assets.tar.gz为1,627,010B，SHA256 `1de5b7308490c70e6d20faa812b5847691e92033abf5ef2df3597318a9acf535`，本地已核对并解包，三份输入逐SHA复核通过。

r09控制器19899已由队列12998启动，当前经ps与receipt核实WAITING_LOCK，GPU另有PID19266使用；未声称持锁或完成校准。r10由原队列在r09成功终态后启动，不重复提交。Qwen校准r11尚未提交。

Qwen接续队列20361已启动并经ps确认存活，执行configs/launch-qwen-r11.sh：等待12998结束并检查资产/r09/r10全部COMPLETE，再执行两格qwen-calibration-r11（code-r12、公共锁）。任一前置失败会停止，不自动绕过。当前尚未启动Qwen控制器。

六份输入统计保存configs/workload-summary.json及各自SHA：OLMoE W1均512，W2/W3均值1792（128×512+128×3072）；Qwen W1范围459–512、均值505.84，W2/W3范围459–3072、均值1781.05，均为128个>512的长请求。Qwen仅39条恰为3072，不将长度上界写成所有请求精确长度。跨模型分别校准λ，不假定同到达率代表同压力。

论文模型节补入动态容量不变量F_t≥E_t(L_t)及事件归纳证明，L_t为最长剩余租约。前提为所有running/inflight owner完整收费、所有entrant精确门控、分配不超已收费endpoint、每步extent最多增长1、无未计分配/共享变化。固定Hr、Ha≥Hr满足视野关系，但不能绕过这些前提。EWMA缩短不改变旧租约，门控还须覆盖旧租约最长剩余期；当前adaptive路径及UNKNOWN bypass未证满足，因此不宣称实现全局无抖动。两次LaTeX编译无Warning/Overfull，修改两页已渲染检查。

## 租约生命周期维护优化（本地，未部署）

当前本地adapter将begin()的全租约扫描改为按到期步索引，每步仅处理该步到期桶；旧到期记录不能取消同一请求的新租约。完成、抢占和reset沿用原生生命周期hook清理；capture晚到的终止输出不再重新创建已清理receipt。已核对pinned _free_request/_free_blocks路径。既有三项CPU门槛通过，victim检查覆盖旧租约续期、删除后到期、reset和晚到receipt；真实KNOWN仍为700页。复杂度为每步O(1+k)处理k条到期记录、每租约摊销O(1)，不等于整个调度步O(1)：账本重建、victim和stall前缀扫描仍未解决。远端code-r12与已提交r09/r10/r11保持原快照；此优化尚无GPU证据。

校准分析器已支持--kv-gib筛选混合预算组，修正Qwen r11两预算不能独立分析的问题；指标按cell名称关联而非zip位置，未测格单列、重复session拒绝。以已有原始归档生成的metrics验证：OLMoE24GiB λ10抢占39/256，下一λ5；48GiB λ20抢占17/256，校准λ20、高档λ40。未把Qwen尚未执行的初始探点当校准完成。r09控制器19899及接续12998/20361当前经ps核实仍存活，r09 WAITING_LOCK且0格。

## 扩展前沿配置已生成（未提交）

code/build_frontier_plan.py从完整native cap256、抢占率5–15%的W2实测格生成十臂：native、三个静态cap、full-budget、MC LOAD适配与Ha16/64/256/1024（Hr32）。校准receipt/metrics/inputs与目标输入SHA写入配置；W1/W3沿用W2的λ，同模型/采样/输出语义必须一致，W2还要求prompt token完全一致。当前生成OLMoE48GiB的W1/W2/W3各十格，λ20，cap128/160/192，configs/olmoe48-*-corrected-frontier.json，均未部署或提交。24GiB尚未进入压力带及跨模型套用校准的拒绝路径已检查。此配置补齐完整负载前沿入口，不把旧账本结果标成新版本，也不改变当前code-r12队列；实际执行版本须在r10结果检查后确定。

## r09：24GiB压力校准完成

native cap256、W2 λ5，256请求全部完成，26/256请求抢占（10.15625%），进入既定5–15%压力区；因此24GiB基准λ5、高档λ10，无需追加二分。maxgap P99=1.054998s，stall=49/256（19.140625%），TTFT P95=13.016411s，TPS=2211.928840，flow均值26.472086s，216自然EOS/40达到上限。测量期仍有JIT，按既定口径保留。

28个恢复episode均为async LOAD，28个ACK应用与28个再次分配边界已记录；recovery_chain按本进程时钟完成重建，无边界次序断言失败。这证明新增观测在该原生格能工作，不代表HORIZON恢复门控已真机验证。

r09持锁153.469407s，累计3572.429s（59.54min）。本地raw归档4,208,763B，SHA256 `6515045ac0d015828d90a41b3b80a919ccf853bd3f804dfc79c3531432380d29`，解包与分析完成后才清理远端本组cache/raw，保留配置、日志、收据与压缩归档。r10控制器23722已启动，经ps确认存活、WAITING_LOCK；12998与20361仍为原接续队列，无重复提交。

依据r09生成24GiB W1/W2/W3各十臂配置（λ5），cap64/80/96为48GiB三档按KV预算减半的初始扫描，不声称已调优；W1/W3沿用同档避免按负载重新选择。三个olmoe24-*-corrected-frontier.json均只在本地准备，尚未执行。

r09结果已纳入T1（26格、25完整）与论文校准段；补充PDF六页全部目视核对，主文修改页通过检查，二次编译无Warning/Overfull。r10首格完整策略退出码0，1971步、574次门控记录，恢复拒绝123、准入拒绝179、stall重排19、unknown_target_boundary375；无victim跳过计数。此为修正版在原生GPU引擎执行及动作触发证据，尚未分析完整服务指标或归因UNKNOWN。第二格PID24629正在运行，整组未完成，不提前宣称消融结论。

F1新增complete_chain.py/complete-chain.pdf：r09全部28个单LOAD恢复episode具备七个主机边界，六段相邻区间逐episode加和核对通过。平均总恢复0.588524s，其中抢占→分配0.473789s、LOAD→主机完成0.054158s；其余分配→提交0.002436s、主机完成→ACK0.007382s、ACK→再次分配0.007849s、再次分配→首输出0.042910s。图中均值堆叠可相加，区别于旧recovery.pdf的重叠区间；保留全部episode点和源SHA，不将episode当独立运行。图已进入论文并渲染检查，二次编译无Warning/Overfull。r10前两格退出码0，第三格PID25040运行中。

## r10：修正版消融与恢复视野敏感性完成

八格均256请求完整结束，统一code-r12、48GiB/W2 λ20、legacy预热。原始归档36,025,684B，SHA256 `2c19d831df625ee498e31742a93b477124bd2c49f551b4441c20c583cadc4d0e`，本地核验解包与analyze完成后清理远端本组raw/cache。持锁1109.928623s，累计4682.358s。

|策略|gap P99(s)|stall(%)|TTFT P95(s)|TPS|flow均值(s)|thrash|EOS/上限|
|---|---:|---:|---:|---:|---:|---:|---|
|Full64/32|10.162233|13.281250|30.026942|2615.460798|42.275357|0|208/48|
|关闭恢复保护|5.942010|16.796875|30.139981|2645.457053|41.638598|6|217/39|
|关闭准入|12.804277|19.531250|26.661527|2653.551913|40.291346|0|216/40|
|关闭stall排序|12.318247|17.968750|29.209783|2645.782488|41.454444|0|217/39|
|Hr16|6.236569|12.890625|27.703951|2665.989448|40.375009|0|218/38|
|Hr64|10.376188|11.328125|30.167551|2631.299877|40.695533|0|216/40|
|EWMA|8.583574|16.406250|28.977797|2630.265153|41.640429|0|212/44|
|MC LOAD|0.575637|10.937500|26.307954|2624.250745|40.134654|0|222/34|

判断：恢复保护减少本次thrash与stall频率，却未改善极端gap；Full没有victim跳过，不能把差异归因victim租约。Hr扫描没有在两项主目标上统一最优。MC LOAD实际117次成功放宽、零抢占，值得作为剩余矩阵强基线；不同臂自然EOS与上限比例不同，不能声称工作量/质量等价。每臂仅单次，未估计运行级不确定性，不宣称显著或全域支配。F4四组件图已生成，8格全指标与SHA保留data.json；论文纳入取舍与敏感性，旧F2曲线保持原版隔离。Qwen控制器29079经ps确认存活，WAITING_LOCK，0格。

OLMoE24/W2十臂前沿接续队列已启动PID30031，经ps确认存活，等待Qwen控制器29079结束并检查r10/r11均COMPLETE；随后用冻结code-r12申请公共锁。脚本configs/launch-olmoe24-frontier.sh SHA256 `08eb3c39a4b0bc6c81edbb204cbead4e080a75595e9eb38231119e743f0b0d54`，配置SHA256 `65f48ddffb19441d981ca385b2c6f071a7ce1a9660b30597103a13756dbd4984`，远端逐一核验。固定λ5、cap64/80/96、full/MC LOAD/Ha16/64/256/1024；组内逐格检查磁盘和GPU占用，失败停止不重试。当前仅接续脚本存活，前沿控制器未提交。Qwen仍WAITING_LOCK；磁盘96GiB可用。

T1扩展到34格（33完整、1初始化失败），包括r10全部结果；运行索引新增no victim lease/no stall ordering/EWMA标记，避免同Ha/Hr消融标签混淆。补充PDF六页检查通过，二次编译无Warning/Overfull。

## r10请求类别分解（后验描述）

figures/F4/request_classes.py按既有512/3072-token两类各128请求分解八格；跨臂核对request/document/prompt SHA/arrival完全一致，逐类stall与输出token和全体指标一致。Full短/长stall均13.28125%，关闭恢复保护分别16.40625%/17.1875%；Full分类gapP99=9.749415/10.237487s，对照5.117521/6.134842s。因此本次频率/严重性取舍在两类中方向一致，未被汇总隐藏。长请求上限终止43对36，输出工作量不同，不能据此隔离调度因果或证明质量等价。原始SHA、16行分类指标保存request-classes.json/tsv，论文注明后验与单运行限制；编译及两页目视检查通过。Qwen29079和接续30031存活，Qwen仍WAITING_LOCK、0格。

修正版十臂前沿绘图入口F2/plot_corrected.py已完成，逐格核对计划与实际版本、输入、终态，按session独立输出，原F2不变。当前OLMoE24/W2返回十格未测状态，未生成虚构图；完整绘图分支待真实十臂数据到齐后验证。

## OLMoE完整前沿接续已提交

五组既有十臂配置已上传并逐SHA核验，顺序为48/W2、24/W1、48/W1、24/W3、48/W3，均沿用冻结code-r12与各预算的W2校准λ。控制器31718执行code/continue_olmoe_matrix.py，等待现有30031（同时核对/proc启动tick，避免PID复用）结束；前置24/W2必须COMPLETE并已有本地归档SHA/字节数确认才继续。每组仍独立申请公共GPU锁、每格检查≥10GiB，失败停止且不重试。累计已用持锁时间加下一组所有单格超时及终止宽限不得超过12小时。

脚本SHA256 `fbdb8735d0a47beb9c7f33e2125032045a9e7d74257720804e1e7b1f7de5b9bb`，配置SHA见configs/olmoe-matrix-continuation-sha.json。每组完成后队列等待该session/local-archive-verified.json，须在下载并核对archive.json的sha256和bytes后由本地归档流程写入；队列自身不清理任何原始数据。日志runs/queue-olmoe-matrix.log。当前是五组接续队列已提交，不是50格已运行；不得重复启动。

## 本地归档接续已启动

code/archive_matrix.py按Qwen r11、24/W2及其余五组顺序等待终态归档，下载后流式核对SHA256/字节数、限制解包到同名session、拒绝覆盖不同本地证据，并运行analyze.py。只有COMPLETE组通过本地核验后才清理远端该组cache及归档中明确存在的raw/horizon/offload/warmup文件，再写local-archive-verified.json放行矩阵队列；STOPPED组保留失败归档后停止，不放行。实际旧r09归档与已有解包内容逐文件一致验证通过。

本地PID54246、SSH复用master53490已用ps确认存活，日志runs/archive-matrix-local-r02.log，状态WAITING_ARCHIVE qwen-calibration-r11。首个本地归档进程53808因非交互SSH找不到python而退出，保留archive-matrix-local.log；已修正为/root/miniconda3/bin/python后启动54246，无GPU重提。SSH socket在/private/tmp/horizon-matrix-ssh，不存储密码。远端Qwen29079及30031/31718队列仍存活；当前归档传输和清理分支尚未在新组上实际执行，不能称本组归档已完成。

修正版F2六子图入口figures/F2/plot_matrix.py已接入现有逐格验证器；严格要求同模型revision、同code snapshot、24/48GiB×W1/W2/W3六个唯一面板，同预算W1/W3沿用W2到达率。共享对数y轴，cap及H档使用不同标记，逐面板保留实际参数。OLMoE当前返回60格未测的JSON状态，没有生成PDF/PNG；完整渲染及版面QA仍待真实数据。相同入口可用于Qwen F6，不混合历史账本版本。

## r11候锁超时，r11b接续

Qwen r11实际STOPPED，error为Public GPU lock wait expired，cells=[]且无acquired_unix_s；29079、30031、31718均已从/proc消失。公共锁由另一存活实验控制器持有，属于资源等待，不是Qwen兼容性或模型失败。自动归档已完成失败收据下载、SHA和字节数验证与解包，702B、SHA256 `5fe7f5fdde5fe23615a171680cdfa0e2f7e838c2a5cf3fd98595281012653b10`，metrics=[]。本地归档54246随后按STOPPED规则退出，未写成功确认，未清理原始数据；旧接续30031/31718按前置失败停止。累计持锁仍4682.358s。

在核实上述进程终态后创建新session qwen-calibration-r11b，除session路径外配置与r11完全相同，code-r12不变。新的code/continue_matrix_r11b.py将候锁窗口从1800延至21600秒，按Qwen→OLMoE24/W2→48/W2→24/W1→48/W1→24/W3→48/W3执行，每组完整且本地归档校验后才放行下一组，无自动重试。计划/输入SHA写configs/matrix-r11b-sha.json并在远端核验，未覆盖旧失败session。预算逐组按累计持锁时间加下一组超时与宽限检查12h上限。

新总接续PID34979、Qwen控制器34980、本地归档60690均已实际ps核实存活。远端日志runs/queue-matrix-r11b.log，本地runs/archive-matrix-local-r11b.log，当前WAITING_LOCK/WAITING_ARCHIVE、0格。archive_matrix.py新增--qwen-session以明确接续r11b；未修改归档校验/失败停止/限域清理规则。后续不要重启旧30031/31718队列或重复提交新session。

论文实验设置补齐硬件/软件、固定KV与host offload预算、fresh-engine预热/重置、开环到达与提交滞后计入方式、host receipt计时和240s测量边界。Qwen段按资产实测统计写明长度范围与39条恰为3072、非思考采样及按document identity派生seed，不把准备完成写成Serving验证。依据run_cell.py、B原测量源码和workload-summary核对。LaTeX编译无Warning/Overfull，修改第5–7页渲染检查通过；全文仍为17页工作草稿，后续完整矩阵及最终浮动图排版未完成。Qwen34980和总接续34979经ps核实存活，r11b仍WAITING_LOCK、0格。

复现文档新增当前模块使用入口：以实际install/receipt/uninstall实现为准，说明pinned实例安装、内部request ID与同进程monotonic时间、正输出增量回调、逆序卸载hook、固定视野与EWMA开关区别，明确卸载不替代引擎取消/drain/shutdown。区分GPU冻结code-r12与本地未部署到期桶优化，保留O(1)维护尚未满足的限制。旧复现条目整体标为历史执行记录，避免旧PID/排队描述被误当当前重提指令；代码和已提交实验未改变。

## 本地恢复 reentry 增量更新（未部署）

按 pinned scheduler 的 allocate→running.append 边界实现完整 LOAD 的快速路径：账本保存非 RUNNING、历史块需求为零的在途 owner；同步 reentry 在对象、历史 token 数及持有块数不变时仅更新 n，不重复收取 horizon demand，也不重建其他 owner。缓存行消费一次；变化/缺失/异步或重复回调退回账本失效。失败 LOAD ACK 可能释放物理块，安装后的 ACK wrapper 在调用原生前记录失败状态、调用后使账本失效。

check_cpu.py 与 check_native_cpu.py 通过。新增检查将快速路径与完整重建的 demand/R/I/n/native_reserved 比较，并覆盖持有块变化、重复/异步回调，以及原生 ACK 消费失败标志后的失效处理。当前只支持这条受限路径的 CPU 等价性，不证明完整原生运行等价或服务收益；每步全体 owner/块重建、victim/prefix 线性操作仍在，O(1) 整体要求未满足。GPU 队列继续使用冻结 code-r12，不包含此优化。

F5/plot.py 新增 --plan/--code-snapshot 入口，复用修正版 F2 的逐组验证，按本组实际策略标签输出独立 cost 图及来源指标 SHA。缺失或非有限成本不会当作零成本绘点。对 olmoe24-w2-corrected-frontier 实际执行返回十格未测 JSON，未生成 PDF/PNG；完整渲染仍待真实数据。原无参数入口继续读取旧 F2，原始图未覆盖。

本地未部署租约维护补充：续租、原生完成与抢占回调通过 `_unlease` 同时删除 lease 和对应到期桶成员，空桶立即删除，避免失效记录驻留到旧期限。删除使用字典/集合定位，预期常数时间；同一步到期多个租约仍需逐个处理。两项 CPU 检查通过，新增断言覆盖续租保留同桶其他 owner、取消及重复取消后的桶状态。冻结 code-r12 与已提交 GPU 计划未变，未声称整体 O(1) 或实测性能收益。

## 本地新请求 LOAD 入队重复计费修复（未部署）

CPU 构造用例发现：无抢占历史的请求若已作为 inflight owner 被账本计入，随后同步进入 RUNNING，旧 allocated 分支仅凭 recovery pending 判断重复进入，会将 H=32 的 2 页需求重复加到 4 页。账本现保存已计入 owner 集合；已存在 owner 使用受限 reentry 快速路径或使账本失效，新增分配才添加贡献。用例先在修改前失败，再在修复后通过，并将 demand/R/I/n/native_reserved 与完整重建比较，重复回调也退回失效。check_cpu.py 与 check_native_cpu.py 通过。

这是本地适配器的构造状态检查，不是原生 vLLM 缺陷，也不证明该路径已在当前 GPU 负载发生；尚需原生轨迹验证。冻结 code-r12 和既有实验结果保持原版本，不能将本地修复收益归入已测数据。整体 O(1) 要求仍未满足。

原始 r10 轨迹补查：八臂各有 255 次 accepted LOAD 在请求首次记录的抢占之前提交；例如首批 lookup matched=32、asynchronous=true。逐 job 与源 SHA 保存在 runs/w2-48-ablations-r10/new-load-origins.json。这证明新请求 LOAD 路径真实存在，但旧日志未记录其 reentry 时缓存账本是否存在，不能据此量化重复计费或改写历史服务指标。为避免未经处理的风险扩展到新矩阵，已核实身份后仅 SIGSTOP 本课题协调进程34979；native Qwen子进程34980继续候锁，本地归档60690继续运行。远端 runs/matrix-r11b.hold.json 记录原因。恢复六组前沿前需验证并冻结修复版本、更新接续；不要直接 SIGCONT 旧协调器启动 code-r12，也不要重复提交 Qwen。

## code-r13冻结与矩阵接续替换

补充32-token命中、512-token历史的部分LOAD用例：分配扩页后必须失效，重建后H32需求为2页而非重复计入旧历史；与完整LOAD/失败ACK/重复回调等现有检查一起通过。增加new/recovery各自的cached/invalidated reentry计数，以便首组原生运行检查实际触发。

新快照runs/code-r13.tar.gz为44,411B，SHA256 `458ea0aca8c5e6d23b4548ec84ff9b0ac3f2467e73df294b4a7ec1b923ae3e37`；以code-r12为基，仅替换horizon_sched.py与check_native_cpu.py，包含已通过CPU检查的到期桶、失败ACK失效和owner去重/reentry处理。逐文件SHA见configs/code-r13-sha.json，远端全部核验。尚无code-r13 GPU结果，不宣称性能改进或整体O(1)。

确认六个目标session均不存在后，只结束已挂起的本课题协调器34979（启动tick2655225332）；原Qwen34980（tick2655225335）保留。新code/continue_matrix_r13.py启动PID43621，实际ps存活，日志runs/queue-matrix-r13.log显示WAITING_EXISTING_QWEN。它等待原Qwen终态和本地归档，不重提Qwen；随后按configs/matrix-r13-sha.json六组原配置使用code-r13，逐组校验源码/计划/输入SHA、12h预算、公共锁及归档确认，失败不重试。本地60690实际存活继续等待原Qwen归档。旧代码、历史测量及失败归档保留；r09/r10与新矩阵须按版本分别解释。

论文版本边界已补充code-r13的新请求LOAD去重修复及CPU证据，明确r10仍是code-r12测量且受未量化的缓存账本风险限制；不把255次无更早已记录抢占的LOAD等同于255次重复计费。F4正文和图注明确code-r12。主文最终两次编译无Warning/Overfull，仍17页；修改及相邻第5/6/12/13/14页已渲染检查。全文仍为待完整矩阵的工作草稿。新协调器43621及原Qwen34980实际存活，后者仍WAITING_LOCK、0格。

## 新版F3双倍到达率配置入口

figures/F3/prepare_transfer.py从同版本完整W2十臂结果中接受显式cap/H选择和服务取舍理由，保存选择点与计划/指标/收据SHA，复制两臂配置后仅翻倍到达率并使用新session；不提交GPU任务，不覆盖已有配置。W1/W3须沿用相同参数，W2来源点明确为选择数据。实际使用当前未测olmoe48/W2配置验证，返回未测错误且不创建输出；仅完成语法及缺失数据分支检查，完整生成分支和新版F3绘图仍待实测。此次没有确定新cap/H，也未修改已排队快照或计划。

## 新版F3固定参数绘图入口（未实测出图）

新增 figures/F3/plot_corrected.py，接收W1/W2/W3前沿和prepare_transfer生成的两臂高到达率计划，验证选择来源SHA、版本、模型、预算、参数及自然生成输入。数据齐全后绘制stall率与TTFT；缺项只写JSON，新输出基名避免残留图误用。CLI已运行；临时构造数据验证八点缺失和高到达率臂偷改H被拒绝。此为读取边界检查，完整实测分支、出图视觉检查和PDF导出仍未完成；没有确定cap/H，没有新GPU结果，也未改变远端队列。

F3读取集成补查：在自动删除的临时目录中构造三组十臂前沿及两臂高到达率收据/输入/指标，调用真实prepare_transfer.prepare、F2.read_panel和F3.collect（仅重定向ROOT），正确汇总四种条件八点且无缺项；将高到达率臂runner改为code-r12后被拒绝。首次临时fixture因本地文件系统大小写不敏感，使W2.json计划覆盖w2.json输入而触发SHA失败，改为plan-W2.json后通过，研究文件未受影响。此检查不产生实验结论；真实数据读取与图像布局仍待实测。

新版F3增加PDF与PNG同步导出，并将PDF纳入新输出基名的覆盖检查。临时合成八点经真实main绘图分支输出，预览标题明确标注SYNTHETIC LAYOUT CHECK - NOT EXPERIMENTAL RESULTS；PDF经Poppler渲染检查，标题、图例、坐标与单位无裁切。预览仅在/private/tmp，不纳入实验图或论文。真实数据读取与最终图像仍待矩阵完成，未确定实际cap/H。
