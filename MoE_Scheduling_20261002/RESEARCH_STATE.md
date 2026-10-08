# MoE 专家分页与请求调度优化

2026-10-02，探索阶段；独立工作目录，不改已有方向权威结论。

## 最新状态（后文保留历史过程）

当前目标因GPU连接受阻而标记BLOCKED（未完成）：连续三个goal轮次确认connect.weste.seetacloud.com:41307返回Connection refused；等待实例恢复或新SSH地址。上传/运行授权已解决，本地四格服务包已准备且未启动，既有源码与实测结果均保留。恢复后先确认远端状态再执行该组，不重复旧负结果、不把数值修复算作论文机制完成。

2026-10-02 新增基线修复已完成单层GPU验证：`wave_src`在多波之间保留原token/top-k的FC2贡献，最后一次native归约，单波仍走原路径。`deferred_reduction_gpu_r01.json`的8个OLMoE尺寸case全部PASS，与全权重参考及逆序波组逐位一致；旧多波最大误差0.00390625。冷加载微基准新/旧均值比0.99281–1.00585，没有明确速度收益，且M2048 allocation峰值多约48MiB。它是数值实现修复，不是新的论文机制；完整模型输出等价仍未证明。四格standard/deferred/deferred/standard已准备，但服务源码上传连接重置、之后41307端口Connection refused，故尚未启动，无本方GPU任务在途；已请求实例状态/新地址，上传运行授权仍有效。详见`WAVE_EXPERIMENT.md`。共享常驻core＋跨层transient槽候选有Mira HOT/STAGE及fastllm全局槽直接近邻，未继续开发。研究目标仍未完成。

2026-10-02 等总池开发＋独立输入两组共12格均COMPLETE，原始结果已取回/校验并全部分析。开发23对24均值drained−1.24%、输出率+3.04%；独立48–63却drained+37.28%、输出率−11.50%，两次同向且同臂输出完全一致，0053输出75→349造成长尾；20在独立集输出率+7.13%但flow+10.57%，不能事后挑它为普适最优。报告 `PARTITION_RESULTS_R01.md`、`PARTITION_HOLDOUT_RESULTS_R01.md`，两组六池总量均5.5GiB、allocated/reserved峰值一致，仍是人工压力。holdout归档SHA55dfb101…62452e；开发SHA605ad41a…626d51。查新VAMP已直接覆盖分配失败时成本比较→专家转KV，停止普通grow-only研究；`live_grow_src`仅未上传未执行草稿，不能称可运行成果（`C_PRIOR_ART.md`）。后缀性能方向维持停止；真实FlashAttention r03 6/6 PASS只是局部正确性。无本方GPU任务在途；下一中心机制尚未选定，研究目标未完成，不以工程基线/少量负结果拼成论文。后文排队/运行为历史。

2026-10-02 八格主对照已全部 COMPLETE 并取回：`execution_suffix_compact_41307_r01`，归档SHA `190943ebd11e1bb929f70d691a2c67c13383ebab2079614f829046db2ae6b39f`。AR/N1/MLP-only horizon/跨层compact horizon均值输出率85.0266/84.8071/85.0466/82.6144，compact相对AR−2.84%、N1−2.59%，两个配对均负；MLP-only对AR仅+0.024%且两配对翻转。停止当前后缀策略性能调参，不启动更大模型来挽救本结果。跨层压缩195次固定动作已真实执行、自然EOS和KV恢复可推进，但完整模型greedy位等价未证明；独立真实FlashAttention/独立slot写入小测试排队中，仅排除接口错误，不是性能补偿。下一候选转向同总池5.5GiB的专家/KV静态边际曲线：24槽+1GiB、23槽+1.1875GiB、20槽+1.75GiB，AR与其它资源相同；独立 `partition_src`/`launch_partition.py` 已上传并以controller24228排队到 `results_partition_r01`（24/23/20/20/23/24），真实初始化将核对两池总字节；不把静态划分称创新、没有开发活页管理器。独立attention测试r02在case前因缺省KV布局未设失败，日志已取回；已用原生HND设置修复，r03在共享锁队列中（tool session20597），不改服务源码。正式报告 `SUFFIX_COMPACTION_RESULTS_R01.md` 已完成，图 `figures/suffix_compact_r01_service.pdf` 已生成；此前“排队/运行中”均为历史。

2026-10-02 跨层压缩接续：`suffix_compact_src` 已实现第0决策层之后一次压缩 hidden/residual/positions、FlashAttention query/seq 与独立 KV slot_mapping，后续层实际缩短执行，最终散回原 sampler 形状；独立源码复核完成，41307 上真实 torch/vLLM 的14项测试全部通过。`results_suffix_compact_diag_r01` 因共享锁忙 ABORT_LOCK_BUSY，没有测量；不覆盖该记录。`run_compact_sequence.py` 已在远端排队，先 `results_suffix_compact_diag_r02`（n2fixed0c），实际执行成功后再运行 `results_suffix_compact_r01` 八格 AR/N1/N4horizon/N4horizon+compact/反序，使用整组 flock，无插队。自身状态 `compact_sequence_status.json` 与组 `group_status.json` 足够监控；勿读取另一实验组详细计划，自动审批已拒绝该独立来源。本任务上传/运行授权仍有效，无需再次提问。B/C 备线接口核对见 `mechanism_fallback_feasibility.md`：现阶段没有可直接替换的新小接口。等待本组真实压缩证据，不扩大旧策略调参。

2026-10-02 41307 实测更新：上传许可已由用户明确解除；本段之后的 BLOCKED/尚无测量均为历史。AR/N1/N2/N2/N1/AR 六格完成，见 `NGRAM_SHORT_RESULTS_41307.md`：N1平均输出率+2.55%但两配对方向翻转；N2完整时间−6.69%伴随输出−6.50%，吞吐仅+0.09%。后缀撤销原型三格 r01 完成，N2horizon仅+0.58%吞吐且正确7/16对off8/16、总搬运增加，不能正结论。r02 N1off/N4horizon/N4horizon/N1off也完成：N4horizon配对吞吐+2.18%/+1.33%，完整时间+11.41%/+13.46%、输出+13.84%/+14.97%，均8/16正确但内容不同；615/669条草稿行被撤销、局部冷专家字节实际减少，完整搬运仍增加。当前没有本方GPU测量在途，正在做一次有界的后续attention/router真实跨层压缩优化以检验残留成本，并确认备线实际接口；不将小幅吞吐与变长输出称作方法成功。归档及执行状态见 `execution_41307.json`。本研究目标仍未完成。

2026-10-02 新目标接手：用户已明确授权上传 N1/N2 包、后续原型与输入到新指定 41307 并运行，历史 45495 上传阻塞不再描述当前任务。41307 是全新 RTX 5090 环境，已上传原冻结短草稿包，正在安装 vLLM 0.26.0 和下载原 revision 的 OLMoE 权重；尚无新 GPU 测量。已提出三机制并选择真实路由可见后的合法草稿后缀撤销，备线为阶段执行合批；`suffix_src` 已开始原生原型（保持幸存 top-k，MoE 活行真实 gather/execute/scatter，重新打包 prefix logits，native 回退），CPU 测试通过，GPU 未运行。参见 `mechanism_candidates_review.md`、`prior_art_action_comparison.md`、`execution_41307.json`。文献已纠正 SpecMoEOff 有动态长度，EcoSpec 覆盖预测边际专家成本选节点；不将这些概念当新贡献。

目标现已标记 BLOCKED（未完成）：同一上传授权阻塞连续三轮仍未解除，用户尚未明确答复此前对709KiB实验包上传至45495的许可问题。冻结包SHA仍为6b9e27a09f61e92e3c4bbca17c2b2341f7b6c03f35657d802b8d9acf9a389a74；本地没有results_ngram_short_r01。上一轮只读远端确认包与stage均不存在、未启动该组。已有成本/正确性/局部上界分析已完成，没有剩余必要本地工作可以替代真实六格对照；停止重复轮询/扩大离线分析。恢复条件是明确允许上传该实验包；届时重新核验整组资源并执行原冻结AR/N1/N2/N2/N1/AR，不改变CCF论文目标，不将工程基线或局部上界宣称为独立方法完成。

本轮完成一次有界本地松弛分析，属于progress：`analysis/draft_row_relaxation.py` / `results_ngram_r01/draft_row_relaxation.json` 对两个N4重复均得到相同结构。允许每层任意删至多实际draft数D行（不需要也不假设request-row identity）时，mixed144次中44次仍无法省冷专家；mixed总局部字节上界5.285/69.080GB，puredecode467.694/641.452GB，后者过宽。它不构成合法后缀策略或全episode性能上界，不支持方法GO/NO-GO。停止继续离线oracle，下一步仍是固定N1/N2实测。此前上传审批问题尚未获用户明确答复，没有上传重试或新GPU运行；不能把自动goal接续当作对此代码包外传的批准。

短N1/N2组六格目前本地冻结、GPU未运行。SCP上传被自动审批拒绝：现有GPU实验授权未被认可为向45495发送本次研究代码包的明确授权。已向用户异步请求此次上传许可，未绕过/重试；远端尚未收到本包。两机现场均有其它组。等待期间只做本地成本建模与有界的行后缀撤销可行性审查，不把源代码可行性算作实验收益。

本地补充完成：`analysis/ngram_cost_model.py` 对N4完整阶段守恒和两次配对盈亏关系核对通过，结果 `results_ngram_r01/ngram_cost_model.json`；固定已观察输出时N4还需节省0.7065/0.3908s才能追平同序AR，仅代数阈值，非可实现收益。NGRAM_EXPERIMENT.md 已加入专家集合并集成本、前缀单调截断/bonus与KV不变量及off-by-one反例。`analysis/layer_suffix_feasibility.txt`64行定位本地runner候选行映射和采样前hook，同时明确其非已运行版本资格、sampler/CPU offload提交边界未知；原生metadata尾端取logits使简单缩短draft数不正确。没有新controller或GPU结果。唯一下一步仍是获上传许可及整组资源后执行已冻结六格，不扩展参数搜索。

最新：N4四格68021已COMPLETE并本地完整归档`results_ngram_r01`，SHA `a34d6b22a759089928ff6ff2d60d3bca9ede1f84bb813d6bdcd12da0181c6254`（16,904,168B）；64请求隔离、资源和draft/chunk资格均PASS。AR/N4平均drained17.8867/19.7996s（+10.69%），实际rate79.1658/76.9808（−2.76%），flow12.5182/13.2414s（+5.78%）；host最坏chunk gap1.2698/0.5968s，N4 token ITL正确留null。全部16EOS、同8题正确；同臂16/16输出相同、跨臂6/16相同，1416/1524输出，非等工作量。N4每格1580实际scheduled draft rows、183个多token chunks、403个超过单token receipt的输出（不是精确accepted）。puredecode168→127步，但专家总595.612→718.409GB（+20.62%），fixed4无净吞吐收益。行身份仍未资格化，不能把新增bytes精确归某个draft尾部。

当前无本方在途GPU任务；唯一下一直接对照是更短固定长度1/2（AR/N1/N2/N2/N1/AR），在同已探索32..47输入和相同资源下验证简单强baseline，先于任何layer-cutoff controller。`ngram_short_src`已冻结，包35文件725732B，SHA `6b9e27a09f61e92e3c4bbca17c2b2341f7b6c03f35657d802b8d9acf9a389a74`。45495现场有C组controller68564，22937有A计算进程80209；未启动本方测量，不在其它组重复之间插队。holdout48..63未跑。目标独立CCF方法仍未完成，本轮属于progress：完成一格因果诊断、六格尾块预留及四格真实N4，保存全部负结果与边界。

N4四格最新实际状态：D两组66671/67169均退出、completion receipt显示整组清空/锁释放后，现场GPU空闲；已在45495启动本方controller68021，1790885290.462 Unix，目录`/dev/shm/moe-scheduling-20261002-ngram-r01`，AR/N4/N4/AR，source包9d9dfcc4…60ef不变。当前RUNNING，后文STAGED等待为历史。不要重复启动或更改在途源码；等待完整结果后归档分析。

唯一下一直接实验已冻结并STAGED_NOT_LAUNCHED：原生fixed ng2/5/4对AR16，顺序AR/N4/N4/AR，两臂均CPU/GPU KV1GiB、Q2048、expert24、cap16、同旧warmup输入/32/Q512但各自算法预热。源`ngram_src`/`launch_ngram.py`，包35文件725675B，SHA `9d9dfcc47cf6172d36fe31e84c20ed0f10137279cd488f87e62288b06f4960ef`，45495 `/dev/shm/moe-scheduling-20261002-ngram-r01`。启动前GPU已有D组66722，assert忙而未建launch.json/results；22937也有A组。不是运行失败/性能结果，不插队。S/N共同关闭不支持spec的row observer；仍计全部kernel成本，行身份未资格化。capture保留draft IDs/counts及真实多token chunk；分析已将未解析ITL置null、明确host chunk gap，真实AR旧指标一致核对通过。下一步整组资源空闲后核对manifest并直接启动该stage，不重复mkdir/改源。

转向依据仅是有界可行性：固定2/5/4在已有r02 S轨迹的原生post-first-output 1400个prefix中735有候选，假定原串不变时请求内decode pass条件计数1400→1047（−25.21%），非batch调用/实际接受率/GPU加速。结果`analysis/prompt_lookup_opportunity.json`。更窄的潜在动作是验证进入层内后按真实route/驻留撤销draft尾部、保留前缀全部原top-k；未实现。SpecMoEOff/EVICT覆盖既有spec主干，AcceptMoE已覆盖逐层router/驻留指导删专家，故不能笼统称cache-aware创新；本三篇未见相同删token后缀动作，greedy边界可考察，随机采样分布保持未证。先跑已有固定N4再决定，详见NGRAM_EXPERIMENT.md与prior_notes.md。

最新裁决：单块预留组六臂71905已全部COMPLETE并归档`results_kv_reserve_r01`，tar SHA `c7b25e0c2835ff41bb72b2f699c9a19773df865f735a80ae5258619e9d2325c0`（37,387,019 bytes）。所有96请求隔离、同GPU KV1GiB/512块及动作receipt资格通过；P两次各2次实际预留、11次容量拒绝。0044恢复step57→52，局部动作成功；0047却81→87且停顿更长。P/S平均drained20.974/19.259s（+8.90%）、实际rate71.528/73.540（−2.74%）、flow14.120/13.476s（+4.78%）、worstgap1.671/1.298s（+28.74%）。P7/16正确，S/H8/16；同臂重复16/16输出一致，P/S11/16一致。停止当前单块预留方案，不运行为它准备的48..63 holdout，不把局部修补包装成论文方法。无本方在途GPU任务，目标仍未完成。

单块预留组六臂已启动：22937 `/dev/shm/moe-scheduling-20261002-kv-reserve-r01`，controller71905，S16/H10/P16/P16/H10/S16，源包37文件SHA `669f944be2ef3c5eb9dc9856bb4d6886e8ca8794fe9159b5cca29babba695271`。45495已有其它C组，故本组全在22937fresh engine，只做组内性能比较。实现是73行guarded原生allocate wrapper，CPU模型/接口/AST/CLI检查通过；GPU效果待结果。

最新进展：单格诊断61113已COMPLETE，本地`results_kv_restore_diag_r01`完整归档SHA256 `502759aad253cd1381fc6a9a9f1c4d6bca182acf1ea23bb2d2c999b88210f3df`。capture_alignment通过；0044在step52已finished_recving，但free0，剩余3tokens的allocate在52–55失败，到57才成功多1块。由此确认局部恢复后容量饥饿，而非将全部等待归DMA。正在实现固定总512块内、仅已知一块尾部的最小预留，并准备S16/H10/P16/P16/H10/S16直接对照；协议`KV_RESERVE_EXPERIMENT.md`。诊断不是性能重复，策略尚未GPU运行。60062已COMPLETE后才启动本方61113，未插队；下文STAGED/WAIT状态均为保留历史。

目前已有自然EOS请求的工程收益，但尚无独立CCF B/C论文方法。强静态prefill预算2048相对512，两次平均wall降低12.07%、meanflow降低14.77%、实际token/s提高10.07%；输出长度也改变，不能称等工作量加速。固定2048的cap16/10/8正反序已在22937六臂COMPLETE、全归档`results_headroom_r01`，SHA63bdec81392682e88b82d2f27170988620a060f9136916a8556ca4083f3e6116。cap10消除抢占、最坏gap均值0.921→0.335s，但wall+0.57%、meanflow+2.25%；只是停顿取舍，不是全面加速。cap8的wall+10.44%、flow+8.94%，停止本组cap网格。

原生KV r01三臂正反序已全部完成并本地归档`results_kv_swap_r01`：S16相对R16平均drained+0.18%、实际rate−0.05%、flow−2.20%；S有跨请求fewshot前缀复用，不是纯抢占恢复因果实验。旧r01仍完整保留，不覆盖。最小隔离r02也已六臂COMPLETE并归档`results_kv_recovery_r02`，SHA256 c3877b2c18a284fffc1299a144450db38572e01f203db0987b9c98f36513ffcc，controller55074结束。所有96请求盐/起始位置资格通过；S额外CPU KV1GiB、实际load262MiB/store1552MiB，无跨请求复用。S/R平均drained−5.20%、实际rate+3.50%、flow−1.60%，但worstgap+43.35%、每请求最大gap的P95为2.244倍；输出1416/1443，不是等工作量加速。全部自然16EOS、同8题正确；S/R12条完整输出相同。见`KV_RECOVERY_RESULTS.md`，尚无独立方法GO。

当前无本方在途GPU任务，22937旧r01目录仍仅STAGED_NOT_LAUNCHED。唯一下一问题是原生CPU恢复的额外调度等待：首个共同victim0044同step49抢占，R51恢复、S57恢复；计算707→3位置而缺席0.213→1.044s。先定位可见原生分支及已有恢复优先/余量强基线，不加新预测器/参数网格，不把缺席全归DMA。开启既有swap不作为创新。旧加载重排、分组流水、成本准入模型的具体失败均保留，详见下文。

唯一单格诊断已实现并暂存：`kv_restore_diag_src`/`launch_kv_restore_diag.py`，只给S16的0044在measurement step49–57增加host只读状态/finished_recving/free/held blocks/allocate返回观察，capture step与时间对齐必须通过；不改变策略，不作为性能重复。语法与CLI通过，未执行CPU模拟，尚未GPU运行。包36文件SHA256 `46db910869f7178c95ad045ba124a42d3d1ff092313f6917b89e4f385363e6e8`，45495目录`/dev/shm/moe-scheduling-20261002-kv-restore-diag-r01` STAGED_NOT_LAUNCHED：启动前发现其它C math_runway4096 controller60062，未创建本方launch.json/results。待该整组结束再现场领取，禁止用瞬时GPU空闲插队。相邻C `qualified_serving_v3/restore_runway_policy_v2.py`仅connector=None/fullISL=True的重算门控，已读为简单近邻，不直接转用到本异步路径。

资源接续：实际确认60062此前是WAITING_RESOURCES并每15s释放锁，RAM不足26GiB，非GPU测量收尾。为释放本方占用，已对45495的kv-swap-r01、kv-recovery-r02全部145+145个文件逐项核对保留tar内容与展开原件，SHA如上；只移除remote `results`展开副本，约901MB，保留remote tar、源码与`results_archive_receipt.json`，本地完整raw不变。随后C在1790882888启动唯一GPU子进程60427，实际RUNNING；本方诊断仍未启动。后续读取本方旧结果用本地目录或remote验证tar，不把展开目录不存在当实验缺失。

## 问题和现状

问题：容量受限的 MoE expert offload 中，基于本层已经产生的真实 top-k，优先补齐请求最后的专家依赖，能否在强简单调度之后仍减少完整请求等待？资源是 GPU 专家缓存与串行/有限并发的权重搬运。所有专家贡献必须保留；后续 route、KV、batch 必须随每个执行策略独立推进。

当前状态 MEASUREMENT_ONLY；直接优化试验正在运行。r01两臂已完成：cap24的3840个decode层行中531个早于最后一组完成（13.83%），提前行剩余窗口中位3.601ms；纯decode为227/1488（15.26%）、中位1.385ms；cap64负控0/3840。两臂各16请求/256强制输出token，无preemption。尚无本线性能收益。异步 top-k join / 跨层推进已有 QLLM、AMoE 先例；cached-first / high-load-first 有 HybriMoE 先例，详见 prior_notes.md。旧 JoinStream 的局部窗口未转化成完整收益，不能改名复活。

继承仓库 HEAD 76d6d888de42081c63cd440a8a67623161d8181f；共享工作区有其他会话修改，本线只私有复用 pager、row-context、capture。已读 AGENTS.md、docs/current/README.md、docs/ideas/README.md、共享结果台账与相关近期状态/原始结果。旧 docs/current 停留在较早方向，不自动覆盖近期实测。

## 第一个最小实验（先于结果声明）

- 唯一最弱环节：同批请求的全部 top-k 专家完成时点是否存在可利用的分散。
- 45495 单张 RTX 5090 32 GiB，vLLM 0.26.0 eager / BF16 / Triton；上游 WiSP pager 固定容量。
- OLMoE-1B-7B-0924-Instruct，revision 7f1c97f440f06ce36705e4f2b843edb5925f4498；GSM8K 原序前16条，保留完整660–740-token提示，不按结果筛选。
- expert-major 默认分组，retention=none；16请求同时到达，token budget512，KV1GiB；每请求强制16输出token。此合同用于有界 decode 诊断，不是自然EOS完成、答案质量或任务服务指标。
- 两个顺序运行：cap24（受限缓存诊断），cap64（全驻留负控）。OLMoE本身能装入32GiB，cap24是人为限制，不能表述为自然显存不足。
- 私有 observer 只在measurement、真实decode row身份确定时记录 start / group-complete / layer-end CUDA events；episode末统一同步。ready必须包含全部top-k，未知映射和未完成调用排除。
- 指标：各层请求的首次全部贡献完成组、ready→layer-end局部窗口、首组冷加载与全驻留请求共等待覆盖、实际加载字节/组数、观测全episode时间。局部窗口互相重叠，禁止直接累加成请求收益。事件间隔含host提交间隙。
- 会计：完整请求时间 = 排队 + prefill + decode非MoE + 暴露MoE + sampling + idle/barrier；本轮只观测其中局部路径，不扣减完整时间。
- 所有尝试和失败保留；原始结果不覆盖。资源锁覆盖整个两臂运行组，初始化/两臂边界检查GPU，busy或query失败则停本组，不动其他作业。

## 裁决与下一步

本轮证据上限 LOCAL_LAYER_OPPORTUNITY_ONLY，底层是真实native eager serving，不等于实现了请求级异步调度。cap64负责检查缓存/分组导致的窗口是否消失。若局部窗口很小，停止本加载顺序 formulation；若有显著窗口，再在固定本层路由上比较 cached/ready-first、high-load-first、oldest-request-first、最简单依赖补齐贪心与离线Oracle，以筛选是否有残差。离线结果仅为结构上界。

只有上述残差存在才实现同一异步依赖引擎的策略对照，并计入kernel/attention碎片、重复加载和后续层争用。论文级证据还需要自然完成、质量、真实容量不足的大模型运行域、第二routing regime、steady/bursty、受控重复及近邻强基线。当前不承诺录用或已有独立贡献。

## 已据结果收敛（2026-10-02）

固定当前单层的加载顺序上界：最简单补齐greedy在K=24预算下完成3771行，精确oracle3777行；pure decode二者均1481行。继续复杂selector没有足够理由，未建设跨层异步引擎。原始结果在results_r01，metrics_scoped.json纠正了“固定长度完成不等于自然完成”的字段，原metrics.json保留。

用户明确要求直接优化，已完成固定24槽的12专家小组copy/compute流水两组A/B/C/C/B/A（第二组三臂共同异步map）。全部12cells完成，C比同组A平均慢4.22%和3.56%，停止本实现，见OPTIMIZATION_RESULTS.md。两个完整结果目录results_opt_r01/results_opt_r02与源码包均保留，不覆盖原始结果。

用户进一步要求更建模化、系统化并以CCF B/C投稿为目标。现唯一主线为成本模型驱动prefill准入；模型135行、native hook已接入，native16/static2/4/8/age_gate/model六臂采用不同于校准的新16请求。协议ADMISSION_EXPERIMENT.md，源码admission_src，远端/root/moe-scheduling-20261002-admission-r01随后在共同锁下启动controller33543，六臂全COMPLETE。原始结果全部取回results_admission_r01，归档SHA7918388367c0f08c3bd88aa1bde081cfacc44e5feff58f4aac2fc2e687b564e0。native/model wall8.9096/9.5093s、meanflow7.0836/7.5434s，模型分别慢6.73%/6.49%；全部16请求512token，零抢占，模型与native10/16输出序列相同。停止当前准入启发式，不细调阈值。


下一唯一直接实验：MoE扫描成本的prefill粒度摊销。同engine最大4096和共同训练warmup512，实际token budget512/2048/4096正反序。模型约束是有限专家容量导致mixed近全扫描，优化动作改变扫描次数而非缓存身份。先用static强参照测曲面；若无净收益不建设复杂controller。旧训练trace的层容量重分配cap8..64精确DP最多少0.58% miss且增4.68%组，不续GPU；旧热点保护动作不换名重启。

粒度组六臂已启动：远端/root/moe-scheduling-20261002-budget-r01，controller38791，源budget_src与launch_budget.py，统一engine4096/warmup512；初格b512完整8.9823s。保持整组锁，等待全部正反序完成。

粒度组六臂已COMPLETE并本地归档，SHA5e0d705a3ccf59f0bc3782e0c312a816e358699f4ff2c14904140cfffa683288；results_budget_r01/metrics.json。4096相对512两重复平均wall−27.50%、flow−33.03%、专家bytes−35.86%；但maxITL约238→327ms，峰值allocated增加250,381,312bytes。prefill-bearing22→3步、puredecode49→61步，完整代价都已入账。此为已知token-budget动作的优化候选，不宣称新算法。

自然完成确认已启动：/dev/shm/moe-scheduling-20261002-natural-r01，controller42296，三预算正反序六臂，新的source32..47，ignore_eos=False/min0/max512，旧训练warmup固定32。源natural_src，launch_natural.py，NATURAL_EXPERIMENT.md。结果转到本方独立/dev/shm目录，因为/root只余约0.9GB而shm余16GB，现有模型/其它会话目录不改。等待完整六臂及质量分析，不能把当前27.5%强制长度收益自动外推。

自然组六臂全COMPLETE，controller42296结束；本地results_natural_r01及归档92a7f58e0766e64b0119d24f42ece4f7c6157ea7a5d0e07043dc6ac7a08f1bb1。全部16/16 EOS、0length cap；三个预算均相同8题正确，同预算重复输出16/16相同，跨预算输出/长度有分歧。2048相对512两次均正：平均wall21.411→18.826s(−12.07%)、flow14.941→12.734s(−14.77%)、实际token/s69.637→76.653(+10.07%)，输出1491→1443，不能叫等工作量加速。4096平均19.482s且第二次20.774s，flow有paired退化，不选最好格当主结果。当前2048仅强静态优化基线，独立CCF B/C贡献未达成。

秒级停顿定位在pause_diagnostic.json：六臂最长gap均匹配当请求原生抢占；512一次抢占却2.562/2.606s，2048两次却最长0.859/0.833s。次数不等于严重性，期间peer仍有输出。当前没有后台GPU任务，不自动启动新的controller；待据本组完整成本证据确定下一唯一动作。
