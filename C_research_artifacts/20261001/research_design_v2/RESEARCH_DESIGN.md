# KV 容量约束下请求准入调度的研究设计

2026 年 10 月 1 日。本文是本轮研究设计与新增证据的独立增补，不覆盖原始结果、共享台账或权威裁决。唯一 Primary 保留 **M0 FIFO 退休容量包络**，但投入范围收缩为验证其执行契约和健康工作负载上的净价值；目前不成立“新峰值算法”或“MoE 专属调度”的论文主张。本轮新增实验是 CPU 同信息集动作挑战，远端仅只读恢复已有结果。未启动新的 GPU campaign。

## 一页事实快照

| 项目 | 本轮核实的事实及解释 |
|---|---|
| 主仓库 | `MOE_Thesis_organized` HEAD `76d6d888de42081c63cd440a8a67623161d8181f`；共享论证、CURRENT、台账和 A 代码有未提交改动。保持不动。 |
| 最新 C 源码 | `/private/tmp/moe-research-c-20260929-v2` 初读 HEAD `f5b27506fc6aa1612e81b20abca2bddc5497b47b`；fresh 分析当时未跟踪。本轮期间其他会话继续修改并提交该工作树，后续已观察到 `df508795ef7f7287e90cc1d7cb3130bbd0e21491`。本次实验锚定逐文件 SHA，不声称整树保持不变，亦未将这些并发改动归入本轮。 |
| 过程和证据 | 已按 AGENTS、current README、PAPER_ARGUMENT、CURRENT、RESULT_LEDGER 和指定 C 文档恢复事实。current README 的 8 月状态是历史；9—10 月原始回执决定本轮实测状态。 |
| 开发证据 | 四格 512/512 完成；M0 相对 FIFO goodput +14.22%/+11.70%，flow −5.94%/−4.80%。仍是已见输入。 |
| E0 已完成 | fresh 顺序 native/FIFO/M0/M0/FIFO/native，768/768 完成，六格 exit 0；本地、远端 `block-receipt.json` 一致，终态 UTC 08:54:51。不得重抽输入或重跑。 |
| E0 新判断 | M0/FIFO goodput +11.85%/+30.87%，flow −0.28%/−4.94%。相对 native goodput +10.35%/+26.24%，实际输出速率 −6.13%/−4.51%，flow +2.81%/−1.57%。通过原 E0 的继续资格，不是全指标优胜。 |
| 真实动作 | fresh M0 104/102 次超过完整上界规则的真实准入；物理峰值 4034/4029 块，包络 4095/4096；0 抢占、0 记录到的条件违反。 |
| 反证与缺口 | fresh 各格仍有 119—123/128 个 length 终止；输出 token 序列改变。旧八格 359/1024 次有至少 512-token 短周期重复。自然 EOS 不是质量合格证据。 |
| 当前原型 | 只支持同步、单卡、非 speculative、无共享前缀、无 offload；条件失约抛异常，尚无 safe drain，取消也未支持。不能宣称一般部署保证。 |
| 资源与他人作业 | RTX 5090 `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`，32607 MiB；锁 `2304:29005388732`。UTC 09:08:55 A 暖缓存两格已完成，后续 Q1/Q10 r03 正 RUNNING，controller 6236、GPU PID 6360。没有抢锁、重建锁、停止作业或接管 A。 |
| 最弱因果环节 | 扣除普通 hard-cap 时间容量打包和 decoder 服务保护后，M0 是否还有可复现的系统贡献；现有收益是否依赖大量撞 cap 的生成。 |

完整回收指标、代码版本、运行时与成本见 [证据笔记](evidence_notes.md) 和 [原样回收的 fresh 分析](fresh_analysis_recovered.json)。A 现场状态有时间戳，见 [只读回执](remote_A_receipts.txt)，不是持续占用保证。远端已核 C 适配器哈希与本地冻结值一致，见 [源文件和回执路径](remote_receipt_paths.txt)。

## 当前证据怎样约束论文

| fresh 执行 | 20/4 合格 | Goodput 请求每秒 | Episode 秒 | 平均 flow 秒 | length / stop | 抢占 |
|---|---:|---:|---:|---:|---:|---:|
| native 1 | 54 | 0.665782 | 81.108 | 35.963 | 123 / 5 | 43 |
| FIFO 1 | 57 | 0.656831 | 86.780 | 37.079 | 120 / 8 | 0 |
| M0 1 | 63 | 0.734660 | 85.754 | 36.974 | 122 / 6 | 0 |
| M0 2 | 69 | 0.822365 | 83.904 | 35.760 | 120 / 8 | 0 |
| FIFO 2 | 55 | 0.628381 | 87.526 | 37.618 | 119 / 9 | 0 |
| native 2 | 53 | 0.651421 | 81.361 | 36.331 | 122 / 6 | 34 |

原定 E0 判据是相对 FIFO 两对 goodput 同向且完整性成立，它已满足。不能事后把新提出的 3% 吞吐预算用于改判 E0 无效；但如果今后选择“相对 native 输出率至少 97%”为工程继续条件，则当前两对都未满足。这是有效的性能取舍，不是无效实验。

M0/FIFO 的全部请求都满足 gap≤4 秒，该比较中的联合 goodput 实际由 TTFT 决定；native 各有 12 个请求不满足 gap≤4 秒，最坏 gap 14.367/14.209 秒。M0 相对 native 的截止曲线有交叉，不支持全面支配。相对 FIFO 的输出序列不同为 58/61 个请求，长度不同 2/3 个；第一对 76/128 个请求 flow 恶化，不能用平均值掩盖。两次 M0 运行不是两个独立抽样的 episode，只是同一 fresh cohort 的重复执行。

此前无条件 first-fit 单格 goodput 相对两次 FIFO 为 −8.7%/−13.1%，78 个被越过请求平均 TTFT 增加 1.42/1.63 秒，最大 flow 恶化 13.82/14.06 秒。A 的 fit-first 也有尾部失败。它们支持停止无条件跳队，不证明有预约的 backfill 永远无效。

## 三个候选与唯一选择

| 候选 | 问题和在线可见状态 | 动作与受限保证 | 最强对照与自然机会 | 完整成本与最便宜证伪 |
|---|---|---|---|---|
| **M0 Primary** | 总 hard-cap 上界把不同时刻的峰值相加。用 prompt、当前 computed/output、cap、phase、物理 ownership、native 预留和 token/槽预算。 | 保持 FIFO；全部 resident 为纯 decoder 时，按退休峰值选前缀，新请求计完整上界；依赖旧 decoder 每轮一个输出。容量保证和执行进度保证分开。 | 同服务保护的 FIFO；同信息的全局 hard-cap packing；CacheOPT 嵌入组件。已观测 fresh 104/102 次额外准入。 | 决策、保护造成的 prefill 等待、输出变化、全部 drain、失约/失败均计入。先 CPU 动作等价挑战；下一步健康任务上的 native/FIFO/M0 小块。 |
| **M1 条件扩展** | 混合 prefill/decode 阻止 M0 放宽。保留已承诺 decoder 集 D，未完成 prefill 为 P。 | 峰值 `max E_D + Σ full(P及新请求)`；为 D 留每步服务，并为 P 留最小可兑现 prefill 额度。旧承诺跨重新分类持续存在。 | M0、同 decoder 保护的完整上界 FIFO、普通 decode-first/chunked-prefill。本轮重建有77/80次混合上界可放入队头，覆盖49/52个队头，但机会窗口合计仅1.717/1.769秒；只是结构条件，不是新策略结果。 | 不能只看更小代数上界；可能增加 prefill 饥饿、保护和计数税。先 16—32 请求 shadow 机会探针；机会存在才 native M0/M1。 |
| **M2 备线** | FIFO 队头不可装而后继可装，已知 cap/进度可用于最晚预约；不读真实剩余输出。 | 只允许保持队头容量、序列槽及 prefill 服务预约的有限回填；保证首先以调度轮次表述。 | EASY、保守 backfill、aging、已有 first-fit 负控。M0两格有4613/4590次队头容量阻塞，其中2820/3527次存在可装后继；没有验证加入后仍能兑现队头预约。 | batch 扩大使每轮变慢，轮次不等于墙钟 TTFT；还需入场/离场开销和长请求最差代价。先预约反例和实际 HoL 请求等待诊断，不重跑无条件 first-fit。 |

混合阶段机会的可观测调度间隔合计占waiting时间约2.38%/2.47%，到下一真实all-decode状态中位约22.96/22.55ms，最长77.71/67.57ms。相关状态次数不是独立重复，也不能把这约2.4%当可加速上界；改策略会改变后续轨迹。当前不足以使M1优先于健康生成与强基线。

本轮不实现 M1 或 M2。即使 M0 算法叙事收缩，也不以更复杂机制自动逃避同信息简单基线；M1 只有在自然混合阶段构成经常发生的执行障碍时进入候选。

## 最近邻等价性挑战及创新剩余

[CacheOPT 原文](https://arxiv.org/html/2503.13773v1#S3.SS3)已使用未来容量复用；其逐对 embedding 与全局精确峰值判定是不同层次。仅把原文预测输出换成 hard cap，并不自动使原 embedding 算法与 M0 一致。另一方面，**若 B3 按同一逐轮占用轨迹、相同 FIFO、候选新请求同样计完整矩形上界，以及相同 native 合法性判定进行全局 hard-cap 时间容量打包，则它和 M0 是同一个可行性条件。** 端点求最大值只是更省求值点。

这一等价有明确边界：更强的 packing 可以对新请求自身的增长和退休也建模，而当前 M0 对它们计常量完整上界。它可能产生不同动作，但必须先给出可兑现的 prefill 与 decode 进度，不能凭同信息就假设提前完成。不能故意把 B3 限成较弱的 pairwise embedding 再宣称 M0 首创未来容量。

本轮 [CPU 协议](mechanism_probe/PROTOCOL.md)、[实现与逐状态挑战](mechanism_probe/probe.py)直接导入冻结 M0 的函数和准入路径；独立全轮枚举器使用相同保守计数约定，比较峰值与 FIFO 前缀。[已运行结果](mechanism_probe/result.json)：2,971个小状态穷举和1,000个固定seed状态，峰值和真实M0准入路径的动作差异均为0；fresh两格12,345次调用全部可恢复，其中11,990个纯decode状态的峰值和准入前缀长度差异均为0，队列/居民对齐失败为0。这支持限定域中的动作等价，反对独立峰值算法叙事；不是vLLM GPU安全验证，也不是完整CacheOPT复现。两个手工状态另证明M0与CacheOPT完整需求逐对组件接受集互不支配，12种计数/buffer组合均保持；不能外推到完整CacheOPT。[近邻机制分析](prior_art_notes.md)与 [CacheOPT hard-cap 组件](prior_art_components.py)区分原文规则、组件实现与完整系统。

[WAIT 和 Nested WAIT](https://arxiv.org/html/2504.11320v4#S5.SS1)的未知长度分阶段等待、常驻边界和投影容量检查与 M1 有独立重叠，不能只测 CacheOPT 后宣称覆盖全部近邻。[作者项目](https://or4llm.github.io/)链接的代码是模拟器工程实现，不能直接称为本工作 pinned vLLM 的官方复现。后续 B4 至少包括：CacheOPT hard-cap embedding 的原生组件移植，以及 Nested WAIT 阶段规则的原生移植；若暂时无法完成，两者均标 UNRUN，主矩阵不宣称全面超过原系统。原文使用预测或估计的信息、FIFO 是否改变、驱逐/重算和对短长请求的动作必须逐项列出。

[经典 backfill 比较](https://www.mcs.anl.gov/~kettimut/publications/iwpp02.pdf)已讨论队头预约与更保守的预约，故 M2 的名字和“保留队头再回填”均不构成新贡献。[PagedAttention](https://arxiv.org/abs/2309.06180)解决块管理与共享；本文可能的剩余是受限 runtime 的可执行容量契约及其真实代价，是否足以成论文仍需证据。

## 系统状态与容量推导

限定配置为单 GPU、同步非 speculative decode、每轮采样一个 token、没有 host offload、APC、encoder、多模态、异步/PP。`P_i` 为 prompt 数，`O_i` 为已输出数，`C_i` 为 computed 数，`M_i` 为 max_tokens，块长 `b=16`。纯 decoder 要满足 `0<O_i<M_i`、`C_i=P_i+O_i−1`、下一次正好计算一个 token，且无占位、在途或 speculative tokens。

当前 M0 使用保守计数 `a_i=P_i+O_i`、剩余 cap `r_i=M_i−O_i`：

```text
q_i(k) = ceil((a_i+k)/b) · 1[k≤r_i]
E_D = max over k≥0 of Σ q_i(k)
U_j = ceil((P_j+M_j)/b)
E_D + Σ U_j(new admitted requests) ≤ usable capacity
```

这是源码的保守上界，不是精确 allocator 轨迹。它比当前 computed KV 多计一个 token，并在 cap 端点仍保留该请求；最后一次分配不能因即将结束而提前删除。实际 pinned 一 token路径在产生最后输出前计算的是前一个输出位置，具体 off-by-one 与同轮分配、采样、free 顺序须独立核对。不能为了缩小公式在未验证时删去末点。

端点充分性：在相邻退休端点之间，存活集合不变，每项阶梯函数单调不减；区间最大值在下一次释放前。因而检查 `k=0` 及每个 `r_i` 足够。这个证明依赖所定义的同速服务轨迹，不证明原生调度必然兑现轨迹。

物理池为 4097 块，1 块 null，4096 可用；OLMoE 每块 2,097,152 字节，实际 tensor 共 **8,592,031,744 字节**，可用块对应 8 GiB。逻辑完整上界和超过 4096 不是物理溢出。逻辑 reserve 不是提前实际分配，也不是释放 HBM。源码核对在当前无connector路径 `reserved_blocks=0`，watermark/lookahead亦为0；native full-input预留仍属于allocator的进一步合法性判断；不能把已包含在 `U_j` 的同一 prompt 预留再从 envelope 容量扣一遍。实际 allocate 的 raw-free/in-flight 保留计数仍必须通过 native validate，模型 envelope 接受并不强迫原生接受。

状态包括 FIFO waiting、running、phase、物理 block ownership、native 预留、已承诺的 protected decoder 集及其下一次输出计数。三个不变量分别是：

1. **内存可行性**：物理已拥有与本轮合法分配不超过可用池；证书覆盖所有未来未归还承诺。
2. **进度可行性**：protected decoder 始终是可调度的运行前缀，本轮每个获得一个 token，并在下轮输出数加一或已原生释放；1024-token budget 与最多 32 槽在此配置提供余量。
3. **服务效用**：前两个不变量不保证 goodput 或公平性；必须用所有外部请求的完成结果单独检验。

## 原生接口和失效处理

现有入口为 `C_NATIVE_RETIREMENT_ADMISSION_V1.py:install(engine)`，对 `scheduler.schedule` 和 `_free_request` 作可恢复包装，复用 `C_NATIVE_MAX_BOUND_ADMISSION.py` 的账本、正常释放与容量检查；由原生 `schedule → kv_cache_manager.allocate_slots → model execution → update_from_output → _free_request` 拥有实际资源。`max_num_running_reqs` 暂时限制 FIFO 可进入前缀，不替代原生 allocate。精确文件路径和源码指纹见证据笔记。

```text
observe scheduler and allocator state
verify last step's protected outputs or native release
retain existing protection while any request is still in prefill
if all residents are pure decoders:
    certify progress budget and block-rounded retirement peak
    choose longest FIFO prefix under peak + new full bounds and slot limit
else:
    admit no new requests
native validate and allocate for the permitted prefix
execute; verify protected one-token service and actual ownership
on normal finish or EOS: native free, then reconcile logical ledger
```

**已实现的失败语义**是记录 violation 后抛 `RuntimeError`；恢复 monkeypatch 不会偿还已借用的容量。取消在当前 free 包装中也不属于接受的终止状态。故现版本是受限实验原型，失约 run 应标 INVALID_EXPERIMENT，并保留全部到达、失败和未完成，不能删去失败请求再报性能。

**待实现的最小 drain 设计**不改变准入算法：保存最后有效证书及尚未履行的 decoder 义务；一旦外部前提将改变，先关闭新增准入、禁止扩大不受保护请求的资源占用，继续为原 protected 集兑现同步单 token 服务。只有 ownership 可核对、protected 全部仍 eligible、token budget 足够、证书剩余轨迹有效时允许此 drain；它们均在有限 hard cap 内释放后，剩余请求恢复完整上界 FIFO。P→D 只在有明确首输出/计算边界并获得后续预算时加入下一证书，不能靠重新分组忘记旧债。

如果检测发生在已错误分配之后，或某受保护请求不能推进，以上 drain 前提就未成立；不得声称安全回退。此时保留 abort 作为原型边界，终止状态完整计账，等待单独修复。取消只有在 native 已完成释放、没有在途占位且 ledger 同步后才能减少证书；提前 EOS 同理只减少占用。未来 native 移植应在所有臂共享正确性修复后重新冻结，绝不改写旧原件。

最小反例：两名受保护 decoder 而 token budget=1；运行队列把一个 decoder 排除；在保护前插入耗尽预算的 prefill；cap 触块却先按结束删占用；取消只删账本而 native 尚未释放。前面三项证明 memory-fit 不蕴含 progress-fit，最后两项证明释放边界不可凭逻辑推测。M1 还需 `|D| + prefill_min ≤ T` 且 prefill 本身合法，超出时保留旧 D、停止新增；不能饿死 P。M2 还须为队头 prefill 和槽位预留，只有容量预约不足以保证首输出。

## 模型数据与运行域冻结方案

**已经实跑的 M_A**：`allenai/OLMoE-1B-7B-0924`，model/tokenizer revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，BF16，Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0、vLLM 0.26.0、Transformers 5.15.1、25 CPU 线程。参数为 max_model_len 4096、max_num_seqs 32、max_num_batched_tokens 1024、chunked prefill 开、APC 关、FCFS、async 关、seed 20260905、greedy、min_tokens 0、max_tokens 1024、自然 EOS。不是 instruction 模型，不为其伪造 chat template。

E0 数据为已有 WikiText 文章规则的 rank 832—959，128 请求，prompt 414—3066 tokens；Poisson 5 请求/秒只抽一次，seed 20261001，最后到达 22.682329 秒。`workload.json` SHA `9576395c9540c71be86dd182df03ab6d39cf1c961a6a24dc851d40991cf0a985`。历史来源 receipt 和整片 Arrow/Parquet 等价仍有缺口；称“相对记录的 C 策略选择新输入”，不称完整盲确认。

**选定待资格的 M_B**：`Qwen/Qwen2.5-1.5B-Instruct`，BF16，单卡，先沿用相同同步 runtime。官方模型卡提供 1.54B 参数、28 层、2 KV heads 和 chat template 使用方法；这支持其作为小 instruction 候选，不能代替本机运行资格。[官方模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

本轮受限目录核查只发现 OLMoE 权重缓存，未发现该 Qwen 权重；远端数据盘剩余约 1.97 GB，不能默认可再落盘约 3 GB BF16 权重。第二模型状态为 **SELECTED_FOR_QUALIFICATION / GPU_UNRUN**，revision、下载字节与 tokenizer 哈希须在首次运行前冻结；不改删其他组文件、不自动下载大权重。正式报告不能写成“第二模型已可运行”。其每块字节数必须从 native KV layout 实测，不能硬套 OLMoE 2 MiB/块。

真实任务选择：交互问答使用 [SQuAD 1.1 官方数据与评价](https://rajpurkar.github.io/SQuAD-explorer/)，摘要使用 [EdinburghNLP XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)。数据不是目前已经下载冻结的结果。SQuAD 按文章/上下文分组切分，XSum 按文档 ID 切分，使用 SHA256(`20261001-health-v1` + 文档ID)排序后划开发/测试；所有过滤只基于输入、上下文长度和许可证，不根据输出质量或收益筛选。先保存原始来源、split、ID、文本及 tokenizer 后 token IDs。不存在的准备脚本不冒充已有资产。

健康资格先用 16 个问答加 16 个摘要；问答指令要求只根据上下文回答，摘要使用同一条不含参考答案的摘要指令。Qwen 使用自身 `apply_chat_template(..., add_generation_prompt=True)`，每臂相同 prompt IDs、greedy、EOS、cap。问答 EM/token-F1；摘要 ROUGE-L 并人工盲审事实支持，ROUGE 不等于事实正确。记录 16-token 内周期重复、EOS/length、空输出与截断，不把 token ID 不同直接当质量下降。

健康 pilot 在看策略性能前冻结筛查门：32 请求中空输出为 0，至少 28 个自然 EOS，长度≥128 的周期≤16重复不超过 1 个；16 个 QA 至少 12 个答案正确，16 个摘要至少 12 个无明显事实错误。它只是开发资格阈值，不是统计质量等价门。未达标先定位 template/tokenizer/模型任务适配，不凭调度结果删样本。正式质量比较保留全部样本，采用开发集冻结的非劣容忍度和配对不确定性；质量未通过不以性能补偿。

## E0 到 E7 实验矩阵

以下 UNRUN 是执行状态，不是机制负结论。共用计量约定见下一节。`W` 指上述 C 最新 worktree 中 `refine-logs/expert_saturation/experiments/admission_capacity/20260929_c_baseline_delivery`；`C` 指父目录。命令清单见 [commands.sh](commands.sh)。

| 实验 | 唯一问题与实际或拟冻结配置 | 策略和重复 | 入口与当前状态 | 预算与继续停止判据 |
|---|---|---|---|---|
| **E0** | fresh 输入是否保留对 FIFO 收益；M_A、原 8 GiB usable KV、128 请求、Poisson 5/s、20/4。 | native/FIFO/M0/M0/FIFO/native；同输入各 2 次。 | 已有 `C_NATIVE_RETIREMENT_FRESH_BLOCK_V1.py` 六格全部完成；只使用 `...FRESH_ANALYZE_V1.py` 回收结果。 | 已用 841.391 秒 wall≈0.234 GPUh；原两对同向规则通过。两重复不提供总体确认，不再跑。 |
| **E1** | 自然生成是否健康、收益是否依赖撞 cap。先 M_B 的 16 QA+16摘要健康筛查；保留 M_A 旧压力结果作负面背景，再做相同任务模板资格。 | 首健康资格 native 1 格；合格后一个 32 请求混合任务 pilot，native/FIFO/M0/M0/FIFO/native。正式任务分开并独立抽样，cap 256/1024，EOS 自然。 | 现有 cell 固定 OLMoE 和 128 请求，不能直接改参数冒充泛化。新通用任务 cell 和数据转换器 **待实现**；模型/数据未缓存冻结，GPU_UNRUN。 | 先申请单资格+六格、每格硬上限 900 秒、共≤1.75 GPUh；预计费用须用首资格实际 t 更新。健康失败停性能；合格却无增量，缩域或停止方法投入，不强迫撞 cap。 |
| **E2** | M0 的真实额外准入、mixed-phase 可行而被挡、M0 剩余 HoL 各有多少请求后果。 | 复用日志；缺字段时最多 1 个 32 请求诊断。只有自然 mixed 机会才 M0/M1/M1/M0；M2 不同时实现。 | 本轮 `mechanism_probe/probe.py` 结构挑战已实施；未记录字段不能填零。轻量 mixed/HoL producer **待实现 / GPU_UNRUN**。 | shadow 最多 0.25 GPUh；条件小对照≤1 GPUh，均非本轮授权。结构机会无真实准入先查执行性；约3%—5%改善只作下一阶段弱信号。 |
| **E3** | 强简单和近邻是否覆盖收益。M_A 既有开发集先用；健康 M_B 仅用开发集。 | B0 native；B1 full FIFO；B2固定并发{4,8,16,32}或水位{0,64,128,256块}两个有限网格；B3全局hard-cap packing；B4 CacheOPT组件与 Nested WAIT移植；P=M0。各调参点3独立开发episode，策略冻结后核心至少5 episode。 | B0/B1/M0已原生；B3与CacheOPT组件本轮CPU，native服务UNRUN；B2统一可配置入口、两个B4 native移植 **待实现**。 | 每基线最多4个开发配置，P不追加特征搜索；B2两表均保留，再事前选一个主列。B3动作等价则作为别名报告，不堆GPU重复证明不同新算法。未移植的B4不能写“已全面超过”。 |
| **E4** | 冻结策略的跨模型任务完整服务收益。M_A/M_B，QA/摘要，0.6/0.9/1.1×共同native参考到达率，512请求起，完整drain。 | B0/B1/事前选B2/CacheOPT组件/Nested WAIT/P，共6列×2模型×2任务×3负载×5独立episode=360run；若P=M1再加M0则420run。原五列300run仅保留为预算算例，不满足本设计两个近邻的比较合同。 | 通用矩阵 runner、质量合并及episode统计 **待实现 / UNRUN**。先一个模型一个任务完整小块，不一次铺开。 | 正式t未知；预算=1.2×run数×实测t。128请求旧均值140.116秒外推360run为16.81h，仅算例，不能用于512请求/第二模型授权。每格若6分钟则43.2h含20%余量；300run的原始五列算例为36h。 |
| **E5** | 收益来自退休包络还是服务保护。近容量健康任务，同FIFO和日志。 | full FIFO、相同保护full FIFO、M0；选M1才加M1；2任务×3臂×5episode=30run，可复用已冻结同配置核心格。 | protected-full adapter、同成本计时入口 **待实现 / UNRUN**。取消保护的unsafe版本仅CPU/shadow。 | 30×t×1.2；没有保护对照则不写包络独立因果收益。报告逻辑少留块→实际新增准入→TTFT→decode/重算→完整请求链。 |
| **E6** | 哪条边界击穿主张。一个模型两任务近容量；KV 0.75/1/1.25；cap256/1024/2048及异质混合；短长混合prompt；Poisson与固定burst。 | 一次改变一个因素；每边界B0/B1/强简单/P，5独立episode；先4个最能证伪的非锚点约160run，不作全因子。 | 重用未来通用runner，独立冻结配置 **待实现 / UNRUN**。cap2048需输入+cap≤上下文，不能默默截上下文。 | 160×t×1.2 上限建议，按前阶段逐块决定；质量/安全任何失败停止外推。必报长短请求最差等待/gap/flow；M2才报预约兑现和迟到成本。 |
| **E7** | 真实运行语义、低压税、失约处理和成本边界。低负载、无压力、early EOS、取消、跨块、预算不足、decoder中断。 | CPU小状态/定向native事件各一次；低压和无压力完整服务各5episode；仅测一个最终P。 | 本轮CPU峰值与真实源路径挑战；safe-drain实现和native故障注入 **待实现 / GPU_UNRUN**。 | native资格最多8格×900秒=2h提案；性能成本按实测t。取消/中断只abort则仍受限原型；不得据其他性能掩盖失败。 |

E4 的 native 参考容量在模型/任务开发集统一估计：先量无上游饥饿的完整请求服务率，再以少量开放到达点核对积压趋势，冻结 `λ_ref`；之后所有策略用同一到达序列和相同比例。短 burst 不称稳态。选近容量锚点的持续实验，测量窗口不少于 10 分钟、至少覆盖若干倍原生高分位 flow，预先固定停止时刻，再完整 drain；如果样本不足，不主张稳定 P99。

## 指标统计和成本合同

有限 episode 主指标继续是 `全部外部到达中同时满足TTFT和max host-return gap且完成的请求数 / episode含drain秒数`。持续测量则固定到达窗口请求集合、时间分母与超时，等其结局后计失败；不能把两种分母拼表。拒绝、超时、无输出、未完成都计入；全体均满足gap时注明判据退化。20/4 仅是旧实验主点，真实任务阈值由独立开发数据和需求冻结，报告附近曲线与宽前沿，不在测试结果中选最好门槛。

并列报告 TTFT、flow、每请求最大可观测 host-return gap、输出 tokens/s、完成率、队列长度、实际KV峰值、抢占/重算、决策时间和额外内存、EOS/length、任务质量。没有逐token时间戳时不称ITL；自然输出分叉时不称等工作量加速。保存/加载及JIT在正式区间发生就进入分母，预热流程和种子缓存起点在各臂相同；A/A已说明相同种子缓存也可能有正式阶段编译。

独立单位是 episode，不是 token、相关请求、额外准入事件或20个阈值。5次是初始预算：用不进入最终测试的pilot方差，按事前希望辨别的最小工程效应冻结最终重复数和总预算；主比较为P−B1及P−事前强简单，B0成本单列，按episode配对差/比及区间报告。样本少则明确区间不稳定；不能显著后停、也不能不显著就无限追加。测试按请求/文档与开发分离，同episode跨策略共享输入和到达，运行顺序分块平衡。

建议未来健康探索的工程继续条件为：质量和安全通过、相对事前强简单goodput点估计至少约3%且两重复方向不反转，实际tokens/s下降≤3%，mean flow增加≤5%。这是分配后续预算的弱信号；不宣称接收门槛或统计确认。固定点的区间包括明显负效果时标不确定，按已冻结总预算完成而非换阈值。native效率损失始终另外报告，不能用更慢的FIFO遮掩。

现有 M0 `decision_seconds` 是调度前适配器的墙钟累计，不是进程全部CPU时间，也不含日志构造、原生调度、输出处理的完整税。已从 `steps[].decision_s` 复算：M0_1 p50/p95/p99为100.922/129.937/170.495微秒，M0_2为98.951/123.912/157.549微秒；累计0.589546/0.572630秒。进程CPU和额外内存没有隔离估计，记 **UNMEASURED**。诊断和性能路径分开，所有对照共享必要正确性修复和记录。现有每轮保存steps带来内存增长，部署版不能假设日志免费。只有端到端测到决策税可观，才做端点缓存/增量优化；新旧逐状态动作一致后再独立量净收益。

## 首个最小实验和三分支决策

本轮首个新增实验已限定为 **E3 的 CPU hard-cap packing 等价挑战**，协议先于执行写入 `mechanism_probe/PROTOCOL.md`。命令实际存在，见 commands.sh 的默认 CPU 路径。它不需要新资源授权，不初始化GPU，也不覆盖旧证据。

| 结果 | 解释 | 后续唯一动作 |
|---|---|---|
| M0出现合法的不同动作且能解释 | 只有当输入信息、候选profile、FIFO、native合法性一致才是差异；先排除计数或边界bug。差异本身不是性能。 | 构造最小native状态资格，验证新动作的可执行性，之后才健康任务小块。 |
| 全部动作相同 | 反对峰值判据的独立算法创新；保留工程契约和服务边界问题。不能用更弱近邻做稻草人。 | **健康instruction任务 native/FIFO/M0 小块**，看是否值得投入系统契约贡献；若无健康收益停止方法campaign。 |
| 源码或状态对齐失败 | INVALID_EXPERIMENT，不是方法失败；保留出错状态和已有结果。 | 只修最小对齐后重新冻结一个CPU挑战；不通过切换负载救结论。 |

下一GPU工作当前 **UNRUN**，先补 M_B 的权重/数据/tokenizer与一格资格，不接管A窗口。单资格健康通过后才执行同输入独立演化的六格；若健康收益为负，保留全请求并缩小至明确压力域或结束方法主线；若OOM/进度违约/计量缺失为无效，只修实现不解释为有效性能；正结果仅进入强近邻与保护消融，仍不自动展开360run。

## 论文主张与证据对应

| 可争取的主张 | 核心图表 | 已有证据 | 必须补的证据 |
|---|---|---|---|
| 保守完整预留形成真实准入损失 | 图1 逻辑预留与物理容量、blocked请求等待 | 旧FIFO轨迹及fresh真实额外准入 | 健康任务、按episode的时间/请求质量；不把逻辑差当HBM释放 |
| 执行契约决定退休容量何时可兑现 | 图2 状态机、端点容量轨迹和失约反例 | 原生资格、CPU挑战、protected检查 | 同保护FIFO、safe drain、混合phase/取消边界、与近邻的不可等价系统约束 |
| 固定资源下完整请求的净收益及取舍 | 图3 goodput与TTFT/flow/gap | 开发四格和fresh六格 | 双模型真实任务、强简单/两个必要近邻、独立episode及质量 |
| 包络与decoder保护的各自作用 | 图4 三臂直接消融 | 当前尚缺 | E5，不能由与native差异代替 |
| KV、cap和负载定义适用域 | 图5 窄敏感性及持续队列 | 单压力点，仅有限到达 | E6、持续开放到达、overload保留 |
| 没有隐藏质量、长请求和实现成本 | 表6 质量/最差请求/决策与CPU内存 | 输出变化、first-fit尾损、部分计时 | 任务质量、长短分层、全部失败、共享日志成本和原型限制 |

论文至多三个贡献：保守预留与服务代价规律；一个有独立差异且可兑现的执行契约；原生净收益与失败边界。第二项若被同信息近邻覆盖，就改成测量/边界论文或停止，不拼接M1与M2制造机制复杂度。

最可能被拒的三项实质问题：**工作负载生成退化**，用E1真实任务、质量指标、cap松紧和早晚结束分层检验；**机制等价且基线不全**，用B3动作挑战、CacheOPT及Nested WAIT原生移植和E5保护消融检验；**保证只在成功轨迹上成立且收益含成本/顺序混淆**，用E7取消/失约drain、共同缓存与日志成本、独立episode和平衡顺序检验。这三项未关闭前，不以现有百分比判断可投稿。

## 投稿与执行里程碑

[CCGrid 2027 CFP](https://hpcclab.org/ccgrid27-call-for-papers/) Track 3 包含 inference/model serving；官网摘要截止 2026-11-24 AoE、全文 2026-12-01 AoE。[投稿说明](https://hpcclab.org/ccgrid-2027-instructions-for-submission/)为双盲，最多10页含图表和参考文献。这里只确认主题和规则，不表明当前结果达到接收水平。

CCF官方分类页的搜索索引正文列 CCGRID、ICPADS 为C类；2026第七版官方公告列 Full/Regular 长文边界。官方页直开/正式PDF仍未完整获取，学校认定仍待核实，不能把搜索抓取说成已完成最终资格核验。详见近邻笔记中的官方来源与读取限制。

前48小时完成E0恢复、CPU等价挑战及健康资格准备；第一周只在健康现象成立后做原生近邻、保护消融和安全drain；第二至三周在策略冻结后分块做确认与窄敏感性；第四周仅补最关键证据并组织全文与复现包。时间表按证据推进。当前报告完成不等于全文方法已成立，不沿用历史“100小时”作为新campaign授权。

直接回答：**M0 是最值得继续一个有界实验的调度机制，值得检验的是同步执行进度约束下的容量承诺能否换来健康请求的净收益。它与一般未来容量打包没有已证明的新公式差异，与 CacheOPT/Nested WAIT 的具体区别在硬上限、分块保守profile和受保护执行条件的组合。下一条最可能推翻其投入价值的证据，是正常EOS、输出质量合格的instruction任务中，相同保护的完整上界或强简单规则已经覆盖全部收益。**

交付核对：commands.sh 的summary与reanalyze-e0均已实际执行；E0在新的临时目录重算得到的JSON与回收副本逐字节一致。脚本语法、报告本地链接、结果计数和原分析SHA均通过核对。未再次运行GPU；本会话SSH连接已关闭，A作业未受干预。
