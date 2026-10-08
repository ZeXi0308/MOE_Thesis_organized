# 层内后缀撤销 r01：动作成立，完整收益未成立

2026-10-02。**本组已真实减少所选层的冷专家加载并完成合法短前缀采样，但还没有形成完整服务收益。** Horizon 相对同组 N2off 的实际输出率仅 +0.58%，同时 drained +5.66%、mean flow +7.49%、专家总搬运 +5.51%，正确题数从 8/16 降到 7/16。固定全部撤销 fixed0 的实际输出率 −2.08%。三臂各只有一次运行，不能宣称稳健加速或质量等价。

本报告分析 [r01 完整结果](execution_suffix_41307_r01/results_suffix_r01/suffix_metrics.json)与[补充分布/输出比较](execution_suffix_41307_r01/results_suffix_r01/suffix_r01_details.json)，不改变原始 capture 或原型源码。自动表见 [suffix_summary.md](execution_suffix_41307_r01/results_suffix_r01/suffix_summary.md)。计时说明按随结果归档的源码核对，未从完整时间扣除任何异常长点。

## 配置与完整请求结果

顺序为 N2off/fixed0/horizon，三个独立 engine，全组 COMPLETE，均 16/16 自然 stop、无 length cap。输入为已见 source32–47，greedy、max512、同时到达；RTX 5090 32GB、OLMoE、expert24、cap16、Q2048、engine4096，GPU KV 与 pinned CPU KV 各 1GiB/512块，expert scratch 均 4,831,838,208 B。每臂执行其自身算法的 Q512/固定32输出 warmup。所有预算检查通过，末态无 pending transfer/store；OLMoE 可以全驻留，expert24 仍是人工压力域。

三臂均使用原生 N2 prompt lookup 与同一 suffix 底座。N2off 记录行身份和成本但不裁；fixed0 在第0层把符合条件的草稿全部撤销；horizon 在第0层按当前冷专家并集和已完成工作校准的代价选择合法后缀。r01 只缩短 MoE 活行与 sampler metadata，attention 等中间计算保持原形状。

| 臂 | capture / served / drained（s） | 实际 token/s | Mean flow（s） | Mean TTFT（s） | Host chunk gap p95 / max（s） | 输出 token | 正确 |
|---|---:|---:|---:|---:|---:|---:|---:|
| N2off | 16.0876 / 16.0874 / 16.0889 | 82.293 | 10.5991 | 2.3799 | 0.1996 / 0.2599 | 1324 | 8/16 |
| fixed0 | 17.8187 / 17.8179 / 17.8211 | 80.579 | 11.7287 | 2.3697 | 0.1802 / 0.8011 | 1436 | 9/16 |
| horizon | 16.9974 / 16.9967 / 16.9993 | 82.768 | 11.3930 | 2.4907 | 0.1993 / 0.2609 | 1407 | 7/16 |

| 相对 N2off | Drained | 实际输出率 | Mean flow | Mean TTFT | 最大 host gap | 输出量 | 专家总搬运 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed0 | +10.77% | −2.08% | +10.66% | −0.43% | +208.23% | +8.46% | +2.58% |
| horizon | +5.66% | +0.58% | +7.49% | +4.66% | +0.38% | +6.27% | +5.51% |

输出长度不同，以上是实际 episode 比较，不是等工作量加速。N2off/horizon 有多 token receipt，内部 token ITL 不可解析；fixed0 的 1436 个 receipt 都是单 token。host chunk gap 排除 TTFT，不将多 token 同次返回解释为零 ITL。Horizon 对最大 gap 基本持平，不能称尾延迟改善。

## 真实动作与局部加载核对

| 臂 | 决策 / 裁剪步 | 被裁 request-step / 不同请求 | 原提案 / 保留草稿行 | 被删草稿行 | Native 接受草稿 | Sampler 提交 / 最终返回 token |
|---|---:|---:|---:|---:|---:|---:|
| N2off | 127 / 0 | 0 / 0 | 794 / 794 | 0 | 281 | 1324 / 1324 |
| fixed0 | 190 / 190 | 766 / 16 | 1532 / 0 | 1532 | 0 | 1436 / 1436 |
| horizon | 137 / 52 | 89 / 15 | 910 / 776 | 134 | 274 | 1407 / 1407 |

Native 接受与 sampler 提交由真正执行的 bookkeeping 记录，位于 stop/EOS filtering 之前；本组三臂总提交量恰好均与最终返回 ID 数相同。最终输出率始终用实际返回 ID。被裁掉的位置未获 target 完整验证，不计为拒绝样本，亦不根据 host chunk 反推接受率。

Horizon 在 137 个 eligible 步中裁剪 52 次（37.96%）；其余 9 次由必留行覆盖全部冷专家而直接保留，76 次没有正预测收益。fixed0 有 8 个裁剪步没有节省当前层冷专家，直接验证“删 token 不一定省加载”。Horizon 的 52 个真实动作均消除了至少一名当前冷专家。

| 臂 | 决定层同状态冷字节差总和（GB） | 当前 call 实际 after-load 核对 | 整个 episode 第0层实际 copy（GB） | 整个 episode 全层实际 copy（GB） |
|---|---:|---:|---:|---:|
| N2off | 0 | 127/127 | 44.7952 | 597.1347 |
| fixed0 | 17.4651 | 190/190 | 47.5634 | 612.5613 |
| horizon | 3.9133 | 137/137 | 45.7515 | 630.0516 |

核对使用决定层实际已有路由/入口驻留，比较 receipt 的裁后冷字节与同一 pager call 的真实 `weight_copy_bytes`、loaded experts 和每名 entry miss 只加载一次。所有决定层核对通过，全部测量 pager 字节也与完整 metrics 一致。因此不是只改掩码而没有减少本次加载。

但 17.465/3.913GB 是**各自策略实际到达状态中，本次不裁与本次裁后的局部集合差**，不是相对 N2off 的完整 episode 实测节省。即使只看第0层，全程实际搬运仍增加；后续轮次、并发/KV、输出和缓存轨迹都已改变。后续层仅汇报其真实执行，不把死行的新路由当作未裁策略的路线。

Horizon 冻结的接受 survival 为 0.489510/0.307692，来自该臂尚未裁剪的 warmup（141 个提案机会、位置1/2成功69/43），没有使用测量期结果重估接受先验。当前层 transfer proxy 节省合计 0.078656s，乘16得到预测剩余节省 1.258496s；模型另估少提交49.413 token、未来服务代价0.684336s、动作开销价格0.286292s，净预测0.287868s。这些是动作评分账，不是实测时间差。特别是乘16没有观测后层反事实，不能用它解释为已实现1.258s收益。fixed0 warmup 已固定裁零，接受先验只是无样本平滑0.5/0.5；其动作由固定规则强制执行，不用这条先验作性能证据。

## 全 episode 成本为什么仍然增加

阶段互斥，GB=10^9字节；decode 调度位置包含未必提交的验证候选。

| 臂 / 阶段 | Engine calls | Prefill / decode 调度位置 | Engine wall（s） | 实际专家 copy（GB） | Groups |
|---|---:|---:|---:|---:|---:|
| N2off / pure_prefill | 1 | 2048 / 0 | 0.275779 | 7.889486 | 48 |
| N2off / mixed | 7 | 9067 / 90 | 1.549250 | 55.452893 | 336 |
| N2off / pure_decode | 129 | 0 / 1731 | 14.244586 | 533.792293 | 4170 |
| fixed0 / pure_prefill | 1 | 2048 / 0 | 0.221921 | 7.876903 | 48 |
| fixed0 / mixed | 8 | 9075 / 144 | 1.659197 | 61.517857 | 383 |
| fixed0 / pure_decode | 188 | 0 / 2821 | 15.915040 | 543.166562 | 5118 |
| horizon / pure_prefill | 1 | 2048 / 0 | 0.269532 | 7.889486 | 48 |
| horizon / mixed | 8 | 9067 / 113 | 1.733517 | 63.342379 | 384 |
| horizon / pure_decode | 136 | 0 / 1914 | 14.975043 | 558.819705 | 4365 |

N2off/fixed0/horizon 总 engine calls 为137/197/145，总 engine wall 为16.069614/17.796159/16.978092s；drained 减 engine wall 的余量为19.298/24.932/21.237ms，保留在总成本中。N2off 末尾 native drain 实际调用1次、0.528ms；另外两格调用0次，仍保留其真实 cleanup/drained 时间。

固定撤销虽降低单个已到达层的执行需求，却增加后续轮次；并且 r01 attention 仍按原提案形状执行，不能把1532个被删 MoE 行当成1532个完整 token 的全部计算都省去。Horizon pure-decode 由129增至136步，实际专家 copy 增25.027GB、wall 增0.730s；mixed 再增7.889GB、0.184s。总搬运 N2off→horizon 增32.917GB，不能用局部3.913GB覆盖它。

| 臂 | Native 抢占 | KV load / store（B） | 峰值 GPU allocated（B） | 峰值 reserved（B） |
|---|---:|---:|---:|---:|
| N2off | 0 | 0 / 1,619,001,344 | 7,044,519,424 | 7,501,512,704 |
| fixed0 | 2 | 178,257,920 / 1,631,584,256 | 7,053,047,296 | 7,501,512,704 |
| horizon | 0 | 0 / 1,625,292,800 | 7,044,519,424 | 7,501,512,704 |

逻辑资源池相同；fixed0 实际临时 allocated 峰值增加8,527,872B，不能写成逐字节相同峰值。没有把 KV 保存、恢复或末尾 drain 从成本中排除。

## 决策与重排实测开销、长点及包含性

下表对每格所有测量决策/采样钩子计时作统计，单次分布单位为毫秒，总计单位为秒。分位数沿用现有分析的线性插值，不把127/190/137次调用视作独立性能重复。

| 臂 / 计时 | 次数 | 总计（s） | Mean（ms） | Median（ms） | P95（ms） | Max（ms） |
|---|---:|---:|---:|---:|---:|---:|
| N2off / decision | 127 | 0.135630 | 1.06796 | 0.12473 | 1.52066 | 99.66308 |
| fixed0 / decision | 190 | 0.043958 | 0.23136 | 0.13913 | 0.24597 | 4.20336 |
| horizon / decision | 137 | 0.042755 | 0.31208 | 0.16666 | 1.49822 | 5.50562 |
| N2off / sample hook | 137 | 0.000542 | 0.00396 | 0.00391 | 0.00573 | 0.00725 |
| fixed0 / sample hook | 197 | 0.016234 | 0.08241 | 0.08476 | 0.09576 | 0.20292 |
| horizon / sample hook | 145 | 0.021937 | 0.15129 | 0.00404 | 0.18740 | 13.97769 |

Horizon 52次真正裁剪的 decision mean/median/p95/max 为0.15674/0.15543/0.22834/0.43675ms；裁剪步 sample hook 为0.41550/0.16412/0.19376/13.97769ms。全部决策之和占各自 drained 的0.8430%/0.2467%/0.2515%；sample hook 为0.0034%/0.0911%/0.1290%。这些是计时窗口的实际占比，不是从另一臂扣得的边际额外开销。

N2off 最大 decision 位于 suffix state162、测量 scheduler/engine call99：decision99.663ms，该层 `host_apply`105.670ms、`route_to_host`0.033ms、整个 engine call192.041ms。它发生在 pure-decode，原7行全部保留。现有字段只定位到主机决策包围区间，不能归因给 Python GC、OS调度、JIT 或某个函数。Horizon 的最大 sample hook 位于 state82（测量第19步），为13.978ms。两点都完整保留。

在纯代数地固定 horizon 已观察1407个输出时，其相对 N2off rate 的余量只对应约98.178ms，与 off 单个99.663ms主机长点同量级。这里**不删除该点、不修正总时间、不声称它造成全部差异**；这进一步说明一次运行中的+0.58%不足以建立稳健收益，且质量已出现下降。

已核对 [归档执行器](execution_suffix_41307_r01/suffix_src/wisp_expert_groups.py) 与 [controller](execution_suffix_41307_r01/suffix_src/suffix_controller.py)：`route_to_host_ms` 在 `ids.to(cpu)`/`tolist` 后、`controller.select` 前结束，**不含决策和 gather**；`decision_host_s` 包含 planner 与 `set_keep_drafts`，不含 select 后的完整活行索引构造/gather/scatter；`host_apply_ms` 包含这些外层工作与 kernel 提交/可能等待。`sample_repack_host_ms` 是原生 sampler 前的校验/metadata重排及张量操作提交窗口，不是新增 GPU kernel 时间的独立测量。CUDA load span、这些 host 子窗口、engine wall 和 drained 有包含/重叠关系，不能再相加，也没有单独测得 gather/scatter 的全部边际成本。

## 质量、序列差异与 fixed0 是否等于 AR

同组 fixed0/N2off、horizon/N2off 分别只有6/16完整 token 序列相同、10/16最终提取答案相同。Horizon/fixed0 为4/16完整序列、9/16答案相同。评分继续使用相同的历史 last-number 字符串匹配。

唯一正确性翻转如下，全部错误题与内容变化都保留：

| 比较 | 请求 | 基线答案 / 正误 | 候选答案 / 正误 |
|---|---|---|---|
| fixed0 / N2off | source0038 | 1.5 / 错 | 10 / 对 |
| horizon / N2off | source0036 | 75 / 对 | 37.5 / 错 |

与此前同机已完成六格固定 AR/N1/N2 的输出比较只用于内容核对，不把旧计时拼成本组同场实验。每个旧臂的两次输出完全相同，因此下面的匹配数对其两个重复均成立。

| r01 臂 | 输出 / 正确 | 与旧 AR 完整序列 / 答案相同 | 与旧 N1 完整序列 / 答案相同 | 与旧 N2 完整序列 / 答案相同 |
|---|---:|---:|---:|---:|
| N2off | 1324 / 8 | 5 / 12 | 8 / 11 | 16 / 16 |
| fixed0 | 1436 / 9 | 5 / 11 | 6 / 12 | 6 / 10 |
| horizon | 1407 / 7 | 4 / 11 | 6 / 10 | 6 / 10 |

N2off 与两次旧 N2 不仅输出16/16相同，提案数397/草稿行794、专家字节597,134,671,872、groups4554和阶段步数也相同，支持 off 接线未改变本输入上的观测轨迹。它不证明所有可能状态均正确。

**fixed0 不能当作已证明等价的 AR。** 旧 AR 为1416 token/8题正确；fixed0 为1436/9，仅5/16完整序列相同。固定撤销后的采样确实每次只提交一个 token、保留 anchor logits，所有 eligible 草稿均被撤销；但系统仍按 N2 进行提案、原长度调度/KV分配、未缩短的 attention 与不同 warmup，MoE有gather/scatter，batch/group形状与后续状态也可能改变。BF16形状/归约敏感是合理候选解释，尚没有相同前缀下 logits/argmax 对照把它与实现错误、状态更新差异分开。

例如 source0038 的 fixed0/旧AR 共用前100个输出 token 后分歧，答案10对13，正确性变好；这既不能证明裁剪提高质量，也不能反向证明原型有语义错误。当前“sampler/KV接口可连续执行、输出自然结束、当前层加载符合集合账”是已测事实，完整 AR 等价尚未证实。后续若声称不改变 greedy语义，应定位少数相同前缀的首次分歧，核对正确位置的 logits、原始 top-k、有效KV长度与两种执行形状；无需把相同答案误当作等价证据。

## 本轮结论与边界

本轮新增的正证据是：合法后缀动作已经改变真实专家加载，原生短采样与后续请求推进可完成；在线模型也能在共享专家集合下产生非零动作。负证据是：即便局部冷字节减少，完整运行仍增加轮次、总专家搬运和完成时间；horizon 乘层数的预测没有转成质量可接受且超过波动的有效输出收益。

当前 r01 不支持论文主结果或稳健加速，亦不把整个层内执行方向判死。只缩短 MoE 而保留全长 attention 是明确尚未消除的实现成本；这为进一步改变实际执行粒度提供动机，不能事后把未实现节省加回本组。强参照仍需同场 AR、N1、N2，并保留质量和首次分歧定位；r01 的各一次运行保持原样。

复现本报告基础分析：

```bash
/private/tmp/moe-c-input-env/bin/python -B analyze_suffix_group.py \
  --results execution_suffix_41307_r01/results_suffix_r01
```

详细开销分布、全部组内及与六格短草稿结果的逐请求差异保存在 [suffix_r01_details.json](execution_suffix_41307_r01/results_suffix_r01/suffix_r01_details.json)。上述分析未启动新的 GPU 任务。
