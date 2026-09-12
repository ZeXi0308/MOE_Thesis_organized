# 从时延反馈降档转向容量约束的资格化

2026-09-06；`agent/publish-current-moe-code@2a37765`，工作区含未提交研究代码及结果。
状态：**OPEN / 新运行域尚未测量 / 无方法或 CCF-B GO**。

问题锚：在有限单卡资源上，为 MoE 原生推理找到一个具有完整请求后果、强简单基线
之后仍有空间、动作可以真实执行的研究问题。现有负结果用于缩小问题，不改写成收益。

本轮读取权威入口 `docs/current/README.md`、`docs/ideas/README.md`及以下较新的方向
报告；历史 sealed 结论保留。采用 research-refine 的问题锚和最小机制方法，并按用户
要求仅作一次定向评议、保留一个方案文件；不进行评分追逐、多轮协议扩展或 `.aris`。

## 已有证据要求怎样改变投资

| 结果 | 已关闭的解释 | 尚未关闭 |
|---|---|---|
| 短输出两个 cohort 的固定8均胜6 | 当前6/8域没有动态策略动机 | 其它实际资源受限域 |
| 单次32→16共24 episodes，8组均未超过hold32/static16 | 当前四步ITL触发降档 formulation不值得继续 | 其它动作的Oracle；不能称整个接纳家族无空间 |
| 原生companion探针36次调用，2 target，固定M16/layer3；目标输出均逐位一致 | 所测同宽原生算子中未出现数值外部性 | 不同shape及上游路径；不逐层换参数寻找正例 |

后两项为本轮读取的共享工作区新结果，本轮没有重复执行。单次降档写入时active
为13/14/16、waiting=0，均无需排空，因此不能继续用“drain过慢”解释其失败。
首次约束接纳的机会在写入后115–1001ms出现，也不等于动作已经改善请求。
hold/down前缀未完全匹配；这些是端到端策略比较，不是同一KV状态下的精确单步因果效应。

来源：[单次动作及重复](../../expert_saturation/outputs/admission_capacity/20260906_native_single_action_r01/REPORT.md)、
[原生companion](../../expert_saturation/outputs/native_companion/20260906_layer3_r01/REPORT.md)。

## 唯一 Primary：先测自然容量压力，再决定是否存在新的回收动作

首个研究问题：自然合法长上下文下，是否同时出现持续active、KV block紧张、等待或
抢占，并进入TTFT/TPOT/完成时间？当前全驻留vLLM中的专家物理状态不变，cap16/32
比较最多资格化容量压力，不能回答expert/KV联合回收是否成立。

通过之后的机制假说才是：“账面可分配”与“在请求需要之前能安全取得”的容量差，
是否带来稳定的完整请求代价？这仍然没有实验数据。

这比历史 U/C → 下一步时延 → 调cap少依赖一个未验证的选择器，但不因此自动成立。
稀缺资源必须是实际 HBM/传输带宽；需要先从原生运行证明，不能把显存预分配量、
模型权重+KV字节下界或手工缩小cache当成自然瓶颈。

最小下一实验只资格化运行域：同一原生引擎配置、同一组真实长输入、cap16/32，
一个合法长上下文压力条件和一个短上下文负控，反序重复；先确认输入+最大输出
不超过模型支持的上下文。保持模型、精度、expert residency、token预算和设备资源一致。
压力条件不得通过人为降低显存利用率制造。具体配置冻结后才运行，当前不新增GPU任务。

记录互斥的模型/专家、KV物理池、KV实际占用及空闲块、workspace/graph成本，连同
实际接纳等待、完成时间、TTFT、TPOT、KV不足事件及重算/抢占。预留显存与使用中块
不能重复相加。原生引擎若发生抢占，记录该现象并标明本臂不满足严格非抢占语义；
不跳过正在decode的请求。先看连续响应及失败原因，旧短请求的9ms阈值不用于制造新域正例。

这个实验最多证明资源约束存在。它不证明expert offload有收益，也不证明需要MoE专用
机制。若只有普通KV饱和，按token/KV预算接纳即可解释和解决，应降级为工程结论。
仅仅“本臂专家全驻留”不自动判死尚未测过的驻留动作；如果压低并发解决内存却留下
明显吞吐代价，则先检查现有expert驻留动作的可执行性与成本，不直接新建pager。

## 正信号后的最小动作假说

动作目标可以从“看到慢就降低并发”改为“接纳前只使用已经兑现的可用容量”。
最强简单对照先包括固定cap、按prompt/已知最大输出预算预留KV的接纳规则。
只有这些规则留下稳定空间，且专家驻留是差异来源，才考虑改变expert/KV预算的接口。

一个条件化接口是：仅在资源不再被正在执行的kernel或活跃KV引用后，允许释放/转换
对应物理容量，成功后再更新可接纳预算；不把预计稍后回收的字节提前承诺给请求。
容量至少区分已承诺、操作进行中、可回收、已发布可用，发布点发生在真实回收完成后。
初版只采用固定触发及一个reclaim primitive，以no-op、最好静态分配和KV-only guard
为对照。活跃KV地址/内容、权重、路由及请求持续推进保持合法。转换若需要全局排空，必须把
等待计入成本；若更改物理映射或kernel路径，同样完整计费和验证。
这不是已实现方案，也不保证SLO；已知最大输出预算最多支持相应资源可行性边界。

只保留这一新动作的资格问题，不同时实现pager、复杂predictor和多动作controller。
零成本或只看逻辑字节的离线Oracle只能是上界；方法结果必须独立执行未来KV/队列/
路由/完成轨迹，并与完整成本的简单规则及最近邻实现比较。

## 最近邻已经覆盖什么

| 来源 | 已覆盖 | 本方案不能直接据为贡献 |
|---|---|---|
| [WiSP v2](https://arxiv.org/html/2606.21868v2)，§3.3–3.5 | expert/KV边际价值分配、admission floor、drained barrier两池resize | 联合分配、KV预留本身 |
| [FluxMoE](https://arxiv.org/html/2604.02715v1)，§3/4.3 | 虚拟化专家地址、流式驻留、根据KV压力调专家预算 | 动态expert residency本身 |
| [vAttention](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf) | 保持连续虚拟地址、按需分配KV物理内存 | 固定地址/动态物理页本身 |

WiSP使用排空点是已读到的实现条件，不足以证明自然持续请求中存在显著损失。
同样，把FluxMoE、vAttention与接纳规则组合也不自动新颖。MoE残余的最低证据是：
控制active set、序列长度、KV blocks、prefill/decode和抢占状态后，expert侧实际驻留页、
换入换出字节或暴露等待发生变化，并稳定对应请求损失；仅U/C变化不够。若只能做
episode级策略比较，要保留不同未来状态，不能把普通状态未匹配的数据称为隔离归因。
待证残余必须同时包含：
实际运行中难以及时回收的容量、现有机制后仍存在的请求后果、一个能安全且净正地
改变该后果的动作。若只获得一般内存压力曲线，不能写成新的MoE系统机制。

## 收敛、停止和备选

- 当前最弱链路：自然压力及其完整请求暴露路径是否存在。先测这一项。
- Continue：压力在正常配置下重复出现，普通token/KV规则之后仍有可解释空间。
- Stop：全驻留本已足够；仅KV饱和且简单规则覆盖；转换成本超过收益；或最近邻覆盖。
- Reopen：新的自然模型/合法工作量/硬件改变资源约束，而非换阈值、挑层或挑有利repeat。
- 条件备选只有测量论文：若观察到稳定的容量回收时序边界，却没有低成本动作，先形成
  跨运行域的因果表征；它仍需新发现和请求后果，负结果数量本身不构成论文。

旧profile四次诊断仍UNRUN，可用于解释已有短请求长间隔，但不再作为必须完成后
才能改变研究方向的前置门；它也不提供本新内存问题的证据。

本轮只读新GPU结果、核验原始论文并细化方案，没有上传、执行新GPU、修改权威账本
或覆盖原始artifact。结论是有依据的下一研究投资，不是已成立的CCF-B idea。

## 一次精简评议及处理

GPT-5.6-Sol / xhigh，same-family provisional；仅检查方向，没有给出实验或论文GO。
采用其三项建议：把首轮收紧到压力资格化、要求expert侧物理证据、动作仅保留一个
固定触发的回收原语。对“expert全驻留即停止此域”作限定：它只限制首轮证据层级，
不能判死从未执行的驻留动作。无需再用多轮评分推进一个尚无新数据的假说。

<details>
<summary>原始评议</summary>

主建议：保留这个 cheap probe，但把当前 Primary 定义为“自然 HBM 压力与回收屏障暴露资格化”。“持续活跃请求下的安全可回收性边界”可以作为通过后的机制假说。全驻留 vLLM 中专家物理状态不会变化，cap16/32 本身只能证明运行域存在，无法证明 expert/KV 联合回收或新颖性。

对 [PROPOSAL.md](/Users/leandrozhao/Desktop/、++++++++/refine-logs/independent_ideas_20260905/direction_refinement_20260906/PROPOSAL.md) 建议只改三处：

1. 将首个问题收紧为：自然合法长上下文下，是否出现持续 active、KV block 紧张、等待或抢占，并进入 TTFT/TPOT/完成时间。若现象完全由 KV used/allocated blocks、active、prefill/decode 状态解释，且 expert 始终全驻留，结论就是普通 KV/并发饱和，停止此运行域。

2. 明写 MoE 残余的最低证据：匹配 active set、序列长度、KV blocks、prefill/decode 比例和抢占状态后，必须看到 expert 侧物理量变化，例如 resident pages、真实换入换出字节或暴露的 expert 等待，并稳定对应请求损失。route/U/C、模型名称或权重占用下界都不够。

3. 若上述残余存在，只实现一个固定触发的 reclaim primitive 和 `committed / in-flight / reclaimable / published-free` 账本；回收完成后才向 admission 发布容量，保持活跃 KV 地址、内容和请求推进。用 no-op、最好静态分配或 KV-only guard 作对照，不同时实现 pager、预测器和 controller。

这个方向比旧的“ITL/U/C 触发降 cap”更好，因为它直接追问可执行容量何时兑现，避开了已失败的 selector 链。但 WiSP、FluxMoE 和 vAttention 已覆盖联合分配、expert residency 与稳定 KV 地址等组成部分；当前 novelty residual 仍未成立。停止结论只限本次 OLMoE、RTX5090、native vLLM 0.26 的自然压力运行域。

</details>
