# 固定资源下，首次入场顺序是否改变完整请求代价？

2026-09-08，GPU执行前冻结；当前8个正式episode、8个warmup均UNRUN。

Repository HEAD：`2a37765fe522b1d74609a686f1d327ede7619a50`，工作区含既有未跟踪研究文件。
权威入口：`docs/current/README.md`、`docs/ideas/README.md`，以及本线程
`../prefill_budget_midpoint_r01/REPORT.md`、`../next_prefill_budget/REPORT.md`。
不改科学权威入口，也不修改其他线程实验或原始数据。

继承事实：同域256与512静态token预算都呈现TTFT/TPOT/完成时间和部分ITL尾部之间的
权衡。上一轮四格raw各有3个首次入场机会：最老waiting请求为2048-token长prompt，
同时已有128-token短prompt等待。这仅说明队列动作空间存在，不是性能Oracle。
其他Codex当前研究cap32原生抢占/重算与KV安全cap；本轮动作是固定cap/token预算的
首次入场排序，范围不同。运行前还要确认其他GPU campaign已结束，而不只观察空闲间隙。

唯一问题：对真实已到达、从未开始的请求优先处理短prompt，能否改善完整请求代价，
还是把等待从短请求转移到长请求？最弱因果环节是waiting顺序的真实request-level效果。
不加入专家信号、预测器、抢占、动态cap或新的预算点。

## 动作、基线和负控

固定同一个native FCFS scheduler、cap8、runtime与编译token budget1024。
在每次原生schedule之前，仅对`waiting`中的新请求按已知prompt长度稳定排序；同长度
保留原相对顺序。FCFS臂执行同样的状态读取、日志与deque重建操作，保留原顺序。
两个臂都保持同一个queue对象及request对象；running列表、其中partial prefill/decode、
skipped_waiting和原生running调度优先级不变。发现blocked/preempted/已开始的waiting
请求即保存失败并停止，不能绕过不支持状态继续出收益。真实schedule结果更新ever-started集合。

排序只读取已在native等待队列的prompt长度，不读取未来到达、生成长度、route或completion。
输入输出最大长度相同，但策略不使用输出长度。每个episode从完全排空引擎开始，独立生成
后续KV、batch、route、token与completion；不复用另一策略轨迹。

两组输入全部逐字节复用前次真实执行包：mixed为`[128,2048]×8`，all_short为16×128。
均为相同16篇文章的既有token IDs，50ms间隔到达、固定128输出、seed20260905。
全短组是排序no-op负控：同长度必须保持队列顺序。它只能给出此负控下的运行差异，
不是所有mixed差异的噪声上界。

OLMoE-1B-7B-0924，模型和tokenizer revision均
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`；BF16，vLLM0.26.0，单RTX5090。
max_model_len4096、memutil0.70、chunked prefill、无prefix caching、同步in-process、
stream_interval1、compiled engine。源码hash必须与`runtime_reference.json`一致。

两个新引擎各先执行共同warmup：all_short FCFS、all_short短优先、mixed FCFS、mixed短优先。
forward正式顺序同warmup；reverse正式顺序完全反转。共8正式+8预热，正式128次请求执行；
只有16篇重复使用的文章，不能称128个独立样本。所有运行全量保留，无按结果选择canonical。

## 会计与停止规则

`request latency = host submission lag + add-return到首次调度 + 首次调度到首次token + 后续decode receipt跨度`。
四段共享host时钟并互斥；中间段不是纯GPU时间，也不能当作可全部移除的queue Oracle。
主指标是全体/短/长请求TTFT、平均TPOT、request latency、完成数和episode wall；ITL p95/p99
为描述性补充。参考SLO固定TTFT5s、平均TPOT0.2s，仅记录，不根据结果换阈值。
决策/日志/执行开销全部进入wall和request分母，warmup与正式分开。

核对实际首次调度顺序、重排事件、身份与到达时间对齐；保留原有decode推进、无抢占、
无KV调整、每次engine.step唯一schedule及全部空输出return检查。出现动作语义错误或
measurement错误时标INVALID；动作未暴露则不能判排序失败。长请求成本必须单列。

若两个block的平均请求完成代价均下降至少3%，同时长请求尾部与wall未出现同等量级
恶化，可继续研究这一普通基线的适用域；这不是显著性阈值或方法GO。否则报告明确权衡；
若变号或接近负控波动，唯一下一步优先受控重复，不换阈值调到正结果。
无论如何不自动增加新排序器；未测Oracle，不宣称专家感知收益、新颖性、任务质量保持、
token等价加速、生产HTTP、第二模型或多卡EP。Claim ceiling：单域REQUEST_LEVEL探索测量。
Reopen条件：新的自然到达/长度混合使真实waiting动作或完整请求后果改变；仅换seed不算新Idea。

## 执行、回传

复用既有runner/capture/metrics，只新增queue helper及必要接入，不构建通用控制器。
全新远端目录，每次一个engine；forward全量回传并验证后才运行reverse。
保留输入、执行源码、配置、版本、准确命令、环境、所有raw、stdout/stderr和OS退出码。
本地验证归档字节及核心JSON后，继续保留远端原件。没有GPU数据时仅报告UNRUN。
