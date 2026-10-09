# F新动作：输出物化时机与粒度（运行前冻结）

2026-10-08；沿用../AGENTS.md所指工作区基准，保留全部C1/排序结论，不再调C1。唯一新动作是改变可选输出对象的物化时机与粒度；不改生成、停止检测、GPU、KV或准入。本文件在新性能结果产生前冻结。

问题：相同完整语义能否以更少完整前端CPU交付，而代价不只是把昂贵工作和重请求拖后？主指标仍为轻token实际source emit→客户端解析P99，联合评价完整server/client CPU秒、逐请求交付/gap/完成、RSS、全量排空吞吐、SSE次数和字节。不设统一收益百分比或产品SLO；75ms为研究等待预算而非用户SLO，定时器受同一事件循环调度，实际超期如实报告。

## 先验成本与原型

原生stream_interval门控在detokenizer.update、停止检查和LogprobsProcessor.update_from_output之后：它已经省掉CompletionOutput/RequestOutput、collector交接、部分chat封套/JSON调用和重复封套字节。所有逐token解码、top-k条目、UTF-8上下文修正、API条目对象和JSON数据本体仍必须交付。仅把logprobs放晚不等于消除成本。

强简单T/B/S均在原生RequestState.make_request_output之前门控，不用“先构造再丢弃”弱化基线。首token立即、finish/STOP/ABORT立即；累计token游标保留完整DELTA，状态每个engine输出原时刻推进。以最老pending的首次时间设置真正event-loop timer；没有下一token也刷新，后续token不延长期限。范围为DELTA/n=1、文本、无parser/streaming-input/prompt-logprobs/transfer metadata；不支持路径明确拒绝。

唯一候选L沿用S的同一释放规则，仅延迟heavy sample-logprob容器物化：每步立即推进detokenization/stop和数值cumulative_logprob，保存原始数组；输出前合并raw数组并调用原生逐position处理，保留UTF-8上下文与全部条目。可省的是若干外层调用，不能省每position实际条目工作；raw数组合并复制和双次数值累加计入完整CPU。L不是动态方法，若无增量直接结束，不重命名成新策略。

## 冻结对照与工作点

| arm | 规则 |
| --- | --- |
| native1 / native2 / native4 | 原生stream_interval=1/2/4，含原生首token与finish例外 |
| token4 | 4个pending token或75ms，早期物化门控 |
| bytes1024 | 1024B pending原始payload或75ms；8B/sample ID + 原生logprob数组nbytes，不是网络字节 |
| class_static | light立即，heavy=token4 |
| lazy_static | 与class_static同规则，额外延迟并合并heavy logprob物化；唯一候选 |

75ms在已知high每请求token间隔约56ms、medium约12ms之间形成有意义的计数/期限对照；不是看新结果后选择。native2/4都保留，避免只比较不合适的大interval。每负载顺序：1,2,4,T,B,S,L,L,S,B,T,4,2,1（各独立进程，两次运行）。无cache、priority或线程改动，统一原生fixed1 handler与GC freeze。

输入仅复用既有high与medium：high SHA cb813b551ed43b8aafbfedda87d7f3bba57fbd9cb4e6d993d17d5a63a728a32b，medium SHA f735ed337c4f635337cd954c45335d691ef7970beb0fd36d70f879aed619abc5。时间、批次突发、内容、到达、重请求集合保持。原GPU未开logprobs及原始批内顺序缺失的限制延续，不把回放吞吐视为容量或GPU收益。

阶段预算：一次语义检查及两固定负载共28个短CPU运行，预计15–25分钟服务器墙钟；实现/分析约60–90分钟主动工作，0GPU。frontend/client/producer仍固定核2/4/6，主机共享。不做参数扫描，不因结果不利补跑；接线失败保留并修复，失败组不混作正式配对。

## 检查、解释与决定

先通过原生语义检查：Unicode、stop string、engine STOP、pending abort、无下一token的timer。比较每步detok/cumulative/abort触发与flatten文本、逐position logprobs、finish/stop、usage、DONE；允许chunk边界和封套字节改变。性能原件保留完整client/collector/source时间、输出摘要、完成状态；记录全前端CPU和少量物化计数，不增逐函数日志。首次冷成本保留，全部请求与排空纳入；超时/失败不删除。

source迟到P99超过对应原批间隔中位数时标时序失配并保留，不补跑；所有分位数/最大误差均报告。timer只限制待物化期限，不保证客户端75ms内看到，报告实际超期。CPU包含timer物化；OutputProcessor子阶段不含timer部分，不能代替总CPU。

S/L为主要同规则配对，分别报告两次，使用run为重复单位；对每请求P99/gap/完成保留background/long子群及最坏退化，不把token当独立样本。不同方案的内部轨迹允许变化，不能拼接离线时间当反事实。

动态方法只在当前可观察年龄/原始大小/既有成本能预先区分强简单方案的反复错误动作，且该失配造成完整CPU或逐请求取舍损失时才考虑。若L只是延后同样工作、总CPU与代价不优于强简单，或没有增量状态证据，明确结束F，不再补命名、调参或GPU验证。若有稳定独立增量，才另立一个有因果预测的最小动态候选或真实服务阶段。
