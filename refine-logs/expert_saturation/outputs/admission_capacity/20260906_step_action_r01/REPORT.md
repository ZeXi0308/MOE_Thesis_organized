# 固定 decode 边界的非抢占升档对照

2026-09-06；HEAD `2a37765`，实验实现尚未提交。

**Verdict：`MEASUREMENT_ONLY / NO_STABLE_UC_ACTION_SPLIT`。** 在本 custom runtime、两个已探索文本组和固定 SLO 下，没有得到可重复的 U/C 动作分界；不进入复杂 predictor。动作本身有效，且吞吐—TPOT 权衡存在，但不能升级为专家感知容量方法 GO，也不是整个 admission problem family 的 NO-GO。

## 问题与执行

完成第 6 次 decode 后，在普通离散状态相同时，保持 cap6 与升到 cap8 的请求级响应是否依文本组而稳定改变？U/C 是否有成为动作前区分信号的证据？

复用 offset0、offset32 各 16 条真实 WikiText 输入，128 prompt / 16 固定输出 tokens，OLMoE-1B-7B-0924 固定 revision、BF16，单 RTX 5090，Torch 2.8.0 / Transformers 4.57.6。steady/bursty 到达区间均为 0–1.5s，TTFT≤5s、mean TPOT≤0.2s。八个 fresh model 进程按预定回文顺序执行，每进程四个独立 KV episode，共 **32 cells、16 个 hold/up 配对、32 条不同输入文本**；不是 512 个独立研究样本，也不声称 article-disjoint。

两臂统一预热宽度 6、8；每轮最多一个 FCFS prefill，再执行所有 active 请求。hold 在同一边界记录同类观测；升档只增加接纳，不暂停、抢占或驱逐已有请求。没有返回 cap6 的时间参数，没有在线 U/C selector。完整请求分母已包含 KV 整理、采集、动作记录与实际生效延迟，未重复加局部耗时。

运行顺序及完整命令见 [DECISIONS.md](DECISIONS.md)、[run_campaign.sh](run_campaign.sh) 和各运行目录 commands.txt；核心原始数据见 [gpu_results](gpu_results/)，日志见 [run.log](run.log)。第一次启动因另一 GPU 进程而在加载前停止，0 个 cell；保留于 [startup_blocked](startup_blocked/)，状态为 `UNRUN`，不是负结果。之后新输出目录的 32/32 cells 全部完成。

## 对齐与完整成本

- 16/16 配对的动作前 request membership、可见输出前缀、decode progress、KV 长度、waiting/future 身份相同；32 个边界均为 active6、waiting10、future0，progress=[6,5,4,3,2,1]、KV=[134,133,132,131,130,129]。这不是逐张量 exact-KV snapshot。
- 升档后的实际 active 与 decode batch 均达到 8，hold 最大为 6；升档应用到实际生效耗时 **0.408–0.543s**。动作合法且 action space 暴露。
- latest step5 的逐层压力在同一文本组的 8 个 ON cells 内完全一致；层均 U/C：c0 为 **0.503906 / 5.166667**，c32 为 **0.486328 / 5.083333**。使用包含零负载专家的 C 分母；压力是返回 logits 重建的 proxy，不是 HBM 流量或硬件拥塞。
- 最近延迟和等待年龄仍有差别。跨文本组的离散 shape 匹配不足以证明普通连续状态已经控制，更不能把文本差异归因于 U/C。四步历史宽度混合，不作为当前主信号。
- 从 raw 重算指标，与 canonical analysis 语义完全一致；32/32 请求身份、全 active 执行、采集截止及前后 GPU 进程检查通过。跨八进程的模型、输入预算、SLO、预热、实际执行源码、软件版本和线程配置一致。GPU 隔离仅在边界观察，未声称连续监控。
- `requested=step5.completed≤observed≤applied≤step6.start` 在全部 cells 成立；近期 model/iteration/ITL 字段确实来自 step5。没有将未来压力或 ON 压力转贴到 OFF。
- ON 的显式压力统计 helper 占 episode 墙钟 **0.95%–1.10%**，已计入分母；这不包括全部 router-logit 返回代价，也不能替代独立 ON/OFF 完整成本比较。

## 结果

升档相对保持 6 的 goodput 变化，保留全部反序重复：

| 文本组 / 到达 / 采集 | 第一次 | 第二次 |
|---|---:|---:|
| c0 / steady / OFF | +15.70% | −52.59% |
| c0 / steady / ON | −36.57% | −37.32% |
| c0 / bursty / OFF | −0.32% | −36.44% |
| c0 / bursty / ON | −52.97% | −36.42% |
| c32 / steady / OFF | −0.54% | −59.44% |
| c32 / steady / ON | +15.93% | −60.35% |
| c32 / bursty / OFF | +17.79% | −60.17% |
| c32 / bursty / ON | +16.96% | −61.07% |

**16/16 升档都缩短 episode 3.67%–15.11%，改善 TTFT 中位数，但 TPOT 中位数全部增加 17.34–43.15ms。** goodput 仅 4/16 为正，第二重复 8/8 为负。hold 达标 15–16/16，up 为 6–16/16；32 cells 中 18 个全通过、没有全失败，整体有 SLO 区分度。hold 唯一未达标请求违反 TTFT；其余主要权衡来自升档后的 TPOT。

不能把所有变号都归于门槛微抖动：升档有 6/16 cells 出现距 0.2s 门槛不足 1ms 的请求，最近仅 0.060ms；c0 第一次 bursty/OFF 的 −0.32% 涉及两个仅超标 0.268/0.633ms 的请求。但 c32 第二重复四个 cells 都有 10 个 TPOT 失败，中位 TPOT 为 208.6–215.2ms，存在连续指标的明显恶化。保持原 SLO，没有选取收益更好的阈值或替换原运行。

## 研究判断与唯一下一步

证据上限为 `CUSTOM_CONTINUOUS_RUNTIME` 中的请求级测量。最强已执行简单对照是 hold6；本轮没有完成普通状态反馈策略、U/C 增量策略、nearest-prior-art 动作对照或 action-conditioned Oracle。可执行的升档缩短整个 episode，不等于提高联合 SLO goodput。两组可重现的 U/C 差别尚未形成稳定的动作收益差别，普通时延变化与 post-action TPOT 风险仍未分离。

当前失败类别是**动作收益不稳定、U/C 增量未验证**，不是动作非法或实验无效。暂停在此 custom 短请求域增加 predictor；仅当代表性 runtime 中出现可重复的请求级容量边界、且强简单策略仍留下可选择的 residual 时重开专家信号选择实验。

已有另一份顺序执行的原生 vLLM 32-cell 数据现在均 COMPLETE：压缩到达时 cap6/8 与队列实际暴露，但全部请求通过原 SLO，goodput 退化为完成吞吐；不重跑这些短请求。**唯一下一实验：原生 vLLM，32 条真实请求、128 prompt / 256 固定输出 tokens，沿用压缩到达与 SLO5s/0.2s，cap6/8 做一次 ABBA 四个独立 episode。** 目的仅是用更长的实际请求占用检验排队/TPOT 是否进入 SLO 边界，先不加 U/C 或 controller；需要匹配容纳 384 tokens 的模型长度配置。该下一实验在本记录中为 `UNRUN`。

### 补充：按最新原生结果收敛下一实验

以上长输出方案尚未执行。读取[原生记录的最终解释限制](../20260906_native_transfer_r01/REPORT.md)后，**下一步改为共同引擎配置下的 cap6/8 接纳对照**：固定 `EngineArgs.max_num_seqs=8` 与编译/内存配置，只在已排空的 episode 边界设置接纳上限；复用原 16 条短请求、快速 steady/bursty 和明确覆盖实际形状的 warmup。因为现有两臂的引擎 cap 同时改变 CUDA graph 捕获计划，先分离这个具体混杂，比同时扩展请求数量和输出长度更能改变判断。暂不执行在线 8→6。

新运行同时保留旧 SLO，并使用原生报告依据全部快速到达请求合并 p75、向上取整得到的探索 SLO：TTFT≤0.20s、mean TPOT≤0.009s。阈值在新结果前固定，不代表业务要求，不回写旧运行。本补充取代上一段的长输出下一步；共同配置对照仍为 `UNRUN`。

本轮还有一项机制边界：每轮最多接纳一个请求，前六次 decode 尚未因 cap6 阻挡接纳。因此 step6 升8 相当于该入口的静态 cap8 后续执行；固定边界解决了比较前沿，却不构成新动态方法。16 个配对完成吞吐提高 3.81%–17.79%，TTFT 中位数下降 9.51%–29.18%，TPOT 中位数增加 10.04%–25.08%。同 cohort、同动作的未来成员序列与最终输出在各八次执行中一致，ON 的逐步逐层压力也重现；普通运行速度与 SLO 结果仍有变化。当前只能说 **U/C 动作增量未建立**，不能据此否定它与普通时延交互的可能性。

核心配对数据见 [analysis.json](analysis.json)，输入和实际执行版本核对见 [verification.json](verification.json)。重算使用 `python analyze_fixed_frontier.py --output /tmp/moe-step-action-reanalysis.json`（目标必须不存在）；执行命令与源码来源均已保留，无需复制新 source snapshot。
