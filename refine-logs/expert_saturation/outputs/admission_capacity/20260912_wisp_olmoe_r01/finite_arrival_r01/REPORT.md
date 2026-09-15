# 有限到达U/M/F/X：未复现稳定X/F优势

2026-09-14。八格U/M/F/X/X/F/M/U全部完成；[冻结分析](attempt01/analysis.json)为`DESCRIPTIVE_FINITE_COHORT`、issues为空。当前裁决`MEASUREMENT_ONLY`：X相对F的capture变化为+3.394%/−0.948%，完整进程分别慢1.771%/0.636%。没有稳定X/F净收益证据。

本轮沿用11份runtime和10份CPU helper；16篇新文档每篇128-token前缀、自由生成固定32输出，0.25s计划到达、prefill32、token budget160、384 expert槽/1GiB KV。每个新引擎只先用第一篇文档做2输出warmup，之后保留各自cache并连续测量，不在请求间reset。四臂重复使用同一16篇文档，两新引擎/臂；不是16个独立实验或稳态负载。具体合同见[protocol](protocol.json)。

原始128请求/4096输出全部保留；计划16384个prefill位置加3968个后续decode位置，共20352个调度位置；6960测量层调用、7856全部层调用。另有8条warmup请求/16输出，24次GPU边界检查通过。完整包16055476B、SHA256 `490f5c48ce9a89a16adc14b7350690713b4f59e80ed8a4bd018803053e9cfbe4`，162份[清单文件](attempt01/retrieval_manifest.json)逐字节回读。没有覆盖旧F/X结果或删除慢格。

| 次序/模式 | capture s | 平均TTFT s | 平均请求完成 s | 最大请求ITL s | 完整进程 s |
|---|---:|---:|---:|---:|---:|
| 0_uniform | 10.955155 | 1.884834 | 8.480277 | 1.013314 | 48.456407 |
| 1_selected | 9.702533 | 0.973883 | 6.910176 | 0.784850 | 49.763743 |
| 2_fullstage | 8.540936 | 0.894750 | 5.810298 | 0.474095 | 45.422487 |
| 3_oneshot | 8.830808 | 0.836272 | 6.053348 | 0.757324 | 46.226908 |
| 4_oneshot | 8.425140 | 0.828220 | 5.639868 | 0.197616 | 47.358481 |
| 5_fullstage | 8.505746 | 0.820049 | 5.700292 | 0.199253 | 47.059357 |
| 6_selected | 8.807463 | 0.898970 | 6.024461 | 0.211199 | 46.129835 |
| 7_uniform | 8.665572 | 0.883837 | 5.921499 | 0.214046 | 44.950532 |

完整进程包括初始化、唯一warmup、capture、采集写盘与shutdown；capture从计划首到达开始，计入有限到达启动瞬态和排空，不含引擎初始化。第一格M的进程成本另有已确认外部初始化重叠，见下文。

| X相对同半程基线 | capture正序/反序 | 完整进程正序/反序 | H2D正序/反序 |
|---|---:|---:|---:|
| uniform | -19.391% / -2.775% | -4.601% / +5.357% | +23.260% / +11.240% |
| selected | -8.985% / -4.341% | -7.107% / +2.663% | +13.883% / +9.434% |
| fullstage | +3.394% / -0.948% | +1.771% / +0.636% | +2.384% / -2.131% |

X/F的D2D分别减少52.061%/53.862%，但其capture符号翻转，不能用字节减少代替完整请求收益。X/U、X/M的capture均下降，但完整进程方向翻转；首U的大量延迟尖峰、首M的启动重叠以及同臂漂移，阻止把这些百分比升级为可靠因果增益。

| 同臂第二次相对第一次 | capture变化 | 输出序列相同 | route hash相同 |
|---|---:|---:|---|
| uniform | -20.900% | 10/16 | False |
| selected | -9.225% | 9/16 | False |
| fullstage | -0.412% | 10/16 | False |
| oneshot | -4.594% | 8/16 | False |

这些是两次运行间的已观察差，不能称总体噪声底，不能据此设3×门槛或计算显著性。跨策略相同输出仅7–11/16，全部route hash不同；各策略独立推进，计时差会改变实际到达合批、route、cache和H2D。固定输入和计划到达相同，不表示后续执行轨迹相同。没有按输出是否一致筛选请求。

延迟定位见[host_call_localization.json](host_call_localization.json)。U0−U7 capture多2.289583s，其中layer0 apply多2.902965s、其他层少0.639160s、engine内apply外多0.021894s、engine外多0.003885s；各项为互斥host包络。U0最高五个step为3/8/14/15/17，layer0 host_apply为598.8/822.0/278.6/755.3/576.0ms，而对应load CUDA跨度约9.7–12.4ms。M1 step22同样出现593.8ms layer0包络。该定位不将重叠CUDA/host跨度相加，也不把未细分项称纯CPU或编译。

U0/U7前四步的row身份、route、resident集合、groups与loads一致，step3仍为783.560ms/205.826ms；后续U为49/54步、M为54/55步，轨迹已经不同。日志有fused_moe_kernel JIT告警，但当前非verbose监控采用warning_once，未逐次记录shape/callID/编译耗时。已核对当前jit_monitor.py为17059B、SHA `cadf567b611e42424e3a2badd2d07da5d2047f14942ca139011b9fe4b707409d`，与本地stock/valid-window参考一致。因此冷JIT是有依据的解释候选，不是全部尖峰的已证实归因；不能从总时间直接扣掉这些包络构造假想收益。

外部rotation PID13921在1789318202.793933–1789318214.593186运行11.799253s并进入CUDA/NCCL初始化，随后显存预检失败。它仅与1_selected完整进程1789318197.413978–1789318247.178917重叠；该格warmup起1789318231.253574，capture起1789318232.287354，分别在外部退出16.660s/17.694s之后，直接时间重叠均0。1_selected第二次边界检查在外部进程存活期间仍PASS，外部CUDA初始化发生更晚。故完整进程成本存在已知隔离缺陷，不得减去11.799s或断言对后续测量没有影响；其余七格不与该子进程重叠。原失败由原会话保留，本组不改判或删除任何格。

| 裁决字段 | 当前判定 |
|---|---|
| Verdict | MEASUREMENT_ONLY；有限到达域未证明X/F稳定增益 |
| Evidence type | NATIVE_SERVING调用路径、单模型/单卡人工专家池、有限到达描述性测量 |
| What was measured | 8新引擎、128请求/4096输出、实际内存/搬运/全进程成本和同臂漂移 |
| What was not measured | JIT对逐次尖峰的独立归因、受控环境中的稳定收益、生产SLO/质量/真实超显存部署/EP |
| Strongest baseline | 本次普通完整stage F已与X接近且process两次均更短；不代表优化FreeToken或全局cache系统 |
| Oracle/headroom | 未测统一目标Oracle；D2D减半未成为稳定完整收益 |
| Claim ceiling | 所测有限cohort的原始描述；无显著性、非劣效、稳态或方法GO |
| Failure category | 执行收益未稳定；形状/JIT/host包络未分清；首M初始化隔离失败 |
| Resurrection condition | 控制可测运行变异后，仍存在强简单基线未覆盖的完整请求差异；不换阈值抹掉当前结果 |
| One next smallest experiment | 先对普通U做一次受控的编译事件定位，记录逐次JIT事件/shape与engine/层调用位置；只改变观察与已声明的编译cache初态，保留冷/暖完整成本，辨别上述尖峰来源，再决定是否继续X机制 |

直接回答：移除五次warmup并改为有限时钟到达后，当前数据仍不能证明X提供稳定净收益；F是必须保留的强简单基线。当前应先关闭运行变异的来源，不继续增加调度器或按有利半程选择结论。GPU已交给后续rotation长任务，本组没有存活controller。

后续只读安装源码补充：Triton runtime在compiler返回磁盘缓存命中结果后同样调用jit_post_compile_hook。因此上述日志只能证明该hook触发，不能证明发生实际重新编译；新定位使用compiler listener的cache_hit与CompileTimer，并单列kernel handle首次初始化。旧数据和计时不改写。
