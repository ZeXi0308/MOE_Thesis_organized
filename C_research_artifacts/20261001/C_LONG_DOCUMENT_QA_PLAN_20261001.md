# LongBench MultiFieldQA-en：固定 16 条输出资格计划

## 本次问题与判断边界

此前原生 GSM8K128 在 4096 可用块中只观察到 932 块峰值、零抢占，已停止该容量方向。本次只检验一个不同的任务域：已冻结 OLMoE Instruct 在官方短答案长文档 QA 提示的 4k 适配下，会产生什么自然结束、全文 F1 和具体错误。它不是新控制器实验，也不以另选输入来挽救 GSM8K 结论。

预期短答案指令与 64-token 官方上限下，大部分请求可自然结束；是否有任务意义必须结合全体 F1、输出全文和被截断证据判断。一条非零 F1 或一次 EOS 不构成资格通过。若出现系统性的空输出、无关续写、重复或模板异常，保留全部结果并停止这套适配的直接服务结论；不追加提示、seed、cap 或按答案筛行的修补格。即使存在有用回答，也不能自动进入策略实验，仍需自然资源瓶颈与强基线错失的实际动作机会。

## 固定来源与输入

采用[官方来源说明](C_LONG_DOCUMENT_QA_DOMAIN_CHECK_20261001.md)中的代码 `2e00731f8d0bff23dc4325161044d0ed8af94c1e`、数据 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`。镜像只作为传输，完整 ZIP 113932529 字节，SHA-256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` 与官方相符；任务原件 150 行、4483926 字节、SHA `0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f`。直接 HF 连接失败 V1 原件保留；镜像 V2 于锁内完成 46.684 s，随后释放资源，无 GPU 推理。

仅取 JSONL 源序 0–15，未按长度、答案、压力、截断或模型输出筛选。官方 `multifieldqa_en` 模板和最大生成 64；原始模板提示超 3500 tokens 时，分别解码首尾 1750 并拼接，再套 OLMoE 单 user/generation chat 模板。没有 GSM8K 的额外 assistant `Answer:`。3500 借鉴官方 4k Llama2 配置，OLMoE/chat/vLLM 均属本地适配，不称官方全 LongBench 成绩。

输入目录 `20261001_c_long_document_qa_inputs_v1`，workload SHA `504e28d18cc3fd959cedd5763df169d2e17ec74f768d69e7456fdc752d09c1b4`。前 16 条有 14 个不同 context 哈希、两组各重复两次；12 条截断。原始提示 1449–13145 tokens，最终 chat 1462–3513，最大 P+64 为 3577。全 16 独立完整上限共 3279 块，小于 4096，即使 APC 关闭也无已知 cap 容量压力。不得把本格称容量优化证据。

输入的 `proposed_runtime.status` 保留构建时的 `NOT_FROZEN_OR_EXECUTED` 历史元数据；实际执行配置现在由本计划、cell/launcher 与独立 freeze 文件锁定。正式运行前再次用远端 AutoTokenizer 验证每条原提示分词、头尾解码、chat 和最终 IDs。所有输入均保留，字面/归一化答案出现只作弱诊断，不作为证据完整或删行依据。独立人工输入证据说明也不得更改这一集合。

## 单一 GPU 单元

- 同一备用 GPU `GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e`；现有非阻塞锁 inode `2304:25841682495`，锁忙即退，无队列循环。
- 已验证 Instruct revision `7f1c97f440f06ce36705e4f2b843edb5925f4498` RAM cache，BF16；复用当前私有 vLLM 0.26.0 环境。只建立已验证权重的私有 hardlink stage，结束后删除 stage，持久 cache 仍计入 RAM。
- 全 16 外部到达 t=0、完整排空；max_num_seqs=128、batch=1024、4096 可用 KV 块、APC/chunked prefill 开启、FCFS、full-ISL reserve；无 offload/async/speculative。
- Greedy、seed 20260905、min_tokens=0、max_tokens=64、EOS 启用、无文本 stop。首尾各做一次最多 16-token 暖机后 reset APC；正式测量 deadline 180 s、包含加载清理的整个 owned child deadline 360 s。
- 记录全部原文/IDs、结束原因、逐 token host 返回时间、真实 KV/队列调度边界、抢占、drain、初始化/暖机/正式服务分段时间。计时用于本格完整成本记录；若出现 inference JIT，明确保留，不补跑以取更好时间。
- 输出唯一根 `c-instruct-longbench-qualification-dev-v1`；完成或失败都保留，不重跑同根。锁覆盖 setup、child、drain、child 回收、GPU release 和自有 stage 清理。

## 冻结分析和下一决定

使用 `C_LONG_DOCUMENT_QA_METRIC_V1.py` 的官方 English QA 归一化/F1，完整输出对每条所有 gold references 取最大 F1，全 16 分母平均并乘 100。另报 normalized exact match 辅助、逐题 F1、空输出、EOS/cap、长度、每条截断和全部文本。缺失/未完成留在分母、总状态 INCOMPLETE；只有合格 runtime EOS receipt 才推断自然 EOS。不得做最后数字抽取、首行截断或按正确性重算服务指标。

本格属于已见开发资格，非留出确认。12 条头尾截断可能删掉答题依据；不能把低 F1 全归因于模型，也不能将常识回答归因于长文档理解。执行后先分析这一小格，再决定该域是否仍值得原生资源观察；目前没有冻结额外 cohort、容量格或新策略。
