# 对齐接纳档位：steady 出现请求级信号，因果归因仍未闭合

2026-09-08。**24 个原生 vLLM 测量 episode 已完成，768 次请求执行全部完成；使用的是同一批32条文本，不是768个独立样本。** 本任务连接远端时，既有冻结 campaign 已在运行；本任务完成逐引擎回传、原始时间戳重算和解释，不将其记为另一次新运行。

本轮回答的问题是：使用 `[8,16,24,32]` 档位的普通 ITL 反馈，能否在完整请求成本下超过同次引擎的简单静态档位？答案是：**在这批 steady 请求中，两次重复都超过最好已测静态点；bursty 两次都更差。** 这保留了一个需要同期旧规则对照的探索信号，还不能把收益归因于捕获桶对齐，也没有专家感知方法的证据。

证据来自单张 RTX5090、OLMoE BF16 固定 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`、vLLM0.26.0、Torch2.11.0+cu130/CUDA13.0、Transformers5.15.1。每条请求为128输入和128固定输出tokens，忽略EOS；steady每50ms到达一条，bursty分四组各8条，均在1.55秒内到达完毕后排空。主SLO沿用TTFT≤200ms且平均请求TPOT≤9ms。模型、引擎和两次运行的实际源码均在 `readback/`；执行源码由各引擎 environment hash核对，不以同期修改中的本地runner冒充。

`goodput = 联合达标的完成请求数 / episode墙钟时间`。该分母包含实际排队、prefill、decode、host采集、反馈决策和循环成本。平均TPOT以首末token间隔除以127计算，不等于逐token间隔均满足9ms。没有将步骤当作独立样本，也没有从局部padding扣出未经执行的请求收益。

| Arm | steady正序 / 反序 goodput | bursty正序 / 反序 goodput |
|---|---:|---:|
| static8 | 2.0300 / 2.0250 | 2.2543 / 2.2441 |
| static16 | 3.8415 / 2.4258 | 12.1721 / 12.2677 |
| static24 | 1.8909 / 2.2955 | 12.4473 / 12.5988 |
| static32 | 1.5309 / 1.5296 | 12.4782 / 12.5416 |
| shadow32 | 1.1443 / 1.5355 | 12.3990 / 12.5509 |
| aligned feedback32 | 3.9721 / 3.6275 | 11.9690 / 12.0718 |

单位为联合达标请求/秒。全部静态点均报告；“最好点”是这个有限扫描的实测上包络，不是训练后的在线策略或动态Oracle。正序继续作为预先指定的canonical，反序作为保留的重复，未按结果交换。

| 同引擎比较 | 最好已测静态点 | 静态 / 反馈达标数 | feedback相对静态goodput |
|---|---:|---:|---:|
| steady正序 | 16 | 11 / 14（各32条） | +3.3992% |
| steady反序 | 16 | 7 / 13 | +49.5405% |
| bursty正序 | 32 | 32 / 32 | −4.0804% |
| bursty反序 | 24 | 32 / 32 | −4.1828% |

这个结果改变了一点认识：原四步ITL规则在旧档位上的失败，不能自动外推到替换档位后的全部请求结果；本次steady存在越过静态点的有限信号。不过增益从3.4%到49.5%，静态16达标数也从11变成7，说明9ms门槛附近的计数对重复敏感。反馈steady平均TPOT中位数为8.678/8.643ms，TTFT中位数则为203.0/21.9ms；仅看中位数会漏掉请求分布变化。bursty两次反馈都32条达标，goodput仍降低约4.1%，是完整episode排空变慢，不能用“全达标”掩盖成本。

已有预设SLO网格也限定了这个信号：保持TTFT200ms，将评价中的平均TPOT约束放宽到12ms或16ms时，steady反馈相对当次最好静态点分别落后62.93%/61.25%（正序/反序）。这是对原轨迹的描述性重算，没有改变运行时9ms反馈阈值，也没有替换主SLO。它说明本次固定9ms规则不是普遍吞吐改进；不等于按新SLO重新运行的反馈也会得到相同结果。

反馈改变的目标和实际活跃数也不是同一回事。steady正序首次32→24时actual active为12、等待为0；后续16→8时active为16且等待3。反序首次32→24时active13、等待0，16→8时active17、等待1。因此不能把首次目标写入当作首次限制接纳，也不能将非抢占排空视作即时降低执行宽度。独立causal检查从已调度token和completion重建既有decode是否全部继续推进，并核对每次决策仅使用当时已交付的历史。

原F1–F4均保留，判据局限写在 [ADDENDUM.md](ADDENDUM.md)：F1把目标cap和实际width混用，还把旧中位数范围当作复现容忍区间；F2/F3缺同次旧ladder，并且caps变化同时改变warmup集合。F4的“仍不能赢静态点”在两次steady被本次数据推翻，在两次bursty保持。这个划分不能升级成“对齐导致收益”：不同降档步幅、动作时机、实际排空和预热都是尚未分离的解释。padding是从日志捕获集合推导的诊断，未测HBM或专家拥塞。

具体来说，steady的padding waste相对历史旧规则在正序从0.3014降到0.2066，在反序却从0.1570升到0.2516；请求级改善没有伴随一致的padding下降。当前两次feedback的waste还都高于同引擎static16的0.0920/0.0936。这进一步削弱了“减少padding解释全部收益”的叙述。static24的pure步实际宽度中位数仅为steady14、bursty16；只取实际width17–24时，每cell有8–107步，中位耗时约9.291–9.295ms，略低于旧区间不能单独否定阶梯。

**归档情况。** 两个已完成引擎分别打包回传，本地与远端SHA256一致；24个测量raw JSON均可读，包含请求ID、输入token标识、所有输出时间戳、scheduler/output events和动作记录。配置、环境、运行源码包、stdout/stderr合并日志、逐cell进程检查及campaign零退出状态均保留，见 [TRANSFER.json](TRANSFER.json)。远端原件未删。两个引擎另各执行12个warmup，继承runner只保存摘要并丢弃成功warmup完整raw；这些轨迹未持久化，无法补传。本轮测量数据已完整回传，但不能声称全部阶段raw完整。

**唯一下一实验。** 同一引擎共同预热 `[8,12,16,24,32]`，排空后分别真实执行旧ladder、新ladder及同次静态点；第二引擎完整反转测量次序。工作区同期任务已准备 [32-episode配对方案](../20260908_capture_ladder_paired_r01/DECISIONS.md)，本报告不把该方案当成已经执行的结果。它直接区分“旧规则在同一当前环境下也会改善”和“档位替换保留额外收益”，不引入新predictor。若收益变号，保留全部重复并先做同配置受控复测；若稳定无增益，停止此运行域的档位替换，不按收益调整阈值或文本。

| 结束字段 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；steady出现有限请求级探索信号，未建立方法因果解释 |
| Evidence type | REQUEST_LEVEL，原生vLLM同步进程内arrival driver；两次新引擎、24测量episodes |
| What was measured | 完整请求TTFT/平均TPOT/goodput、真实调度与动作时序、实际width及日志派生padding |
| What was not measured | 同期旧ladder、专家信号增量、任务质量、自然EOS、第二模型/运行域、EP或生产服务 |
| Strongest baseline | 本次同引擎static8/16/24/32的实测上包络；尚非prior-art完整实现 |
| Oracle/headroom | 未测动态Oracle；仅有上述steady静态残差，幅度不稳定 |
| Claim ceiling | 单模型单卡有限cohort中的策略比较；不主张padding因果、MoE特有机制或论文GO |
| Failure category | bursty完整成本后无收益；历史比较缺对照，F1判据不支持结构判死；warmup归档有缺口 |
| Resurrection condition | 本次改法若在配对实验停止，仅在新自然运行域/执行成本产生可重复动作空间时重开 |
| One next experiment | 同引擎旧/新ladder、共同warmup和静态上包络的配对实验 |

复算入口为本目录 [analyze_ladder.py](analyze_ladder.py)，从 `readback/results/` 的24个raw cell重算；24/24通过身份、保存指标一致性、历史截止、既有decode继续推进及step/receipt对齐检查。完整逐请求指标、全部静态比较和实际width分布见 [分析结果](analysis/analysis.json)。输出目录必须新建：

```bash
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260908_aligned_ladder_r01/analyze_ladder.py --output-dir /tmp/moe-aligned-ladder-reanalysis
```

未修改 `docs/current/README.md`，未覆盖封存原始数据，未push。
