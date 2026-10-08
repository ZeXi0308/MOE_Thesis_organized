# 证据事实与 E0–E7 可复用入口

2026-10-01，只读核对后新写；不改变原件或权威裁决。本笔记初稿形成于CPU挑战结束前，下文当时标为待实现的packing组件、mixed/HoL结构探针和决策分位数，已在本轮完成；最新裁决以同目录RESEARCH_DESIGN.md与mechanism_probe/REPORT.md为准，native/GPU近邻仍UNRUN。已读用户附件、R/AGENTS.md、docs/current/README.md、PAPER_ARGUMENT.md、CURRENT_EXPERIMENT.json、RESULT_LEDGER.md，以及指定 C 计划/结果。C 及新目录层未发现 AGENTS.md。这里 R=`/Users/zhaozhenyu/Desktop/毕业设计/MOE_Thesis_organized`；C=`/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001`；D=`/private/tmp/moe-research-c-20260929-v2/refine-logs/expert_saturation/experiments/admission_capacity/20260929_c_baseline_delivery`。

## 一页事实快照

| 项目 | 已核事实与结论边界 |
|---|---|
| 唯一 Primary | M0：FIFO 不改，只松弛容量证书；现阶段是受限同步原生准入原型，尚无独立方法新颖性或健康任务收益证明。|
| 开发证据 | 旧卡 FIFO/M0/M0/FIFO 512/512 完成；goodput +14.22%/+11.70%，flow −5.94%/−4.80%；不是未见确认。|
| E0 最新状态 | 固定 fresh 六格已 COMPLETE，2026-10-01 08:40:50.552–08:54:51.944 UTC，768/768 完整完成；按原顺序 native/FIFO/M0/M0/FIFO/native，不得重跑或重抽。|
| E0 支持 M0 | 对 FIFO 的两组 20/4 goodput +11.85%/+30.87%，flow −0.28%/−4.94%，20/20 前沿点提高；实际额外准入 104/102，0 抢占，0 条件违例，128 准入/释放，完整 drain。包络峰 4095/4096、物理峰 4034/4029 块。支持预定继续分支，不能据两个同 cohort 重复估计总体 CI。|
| E0 限制/反证 | 对 native 仅 5/20、10/20 前沿点更好，实际 output rate −6.13%/−4.51%，flow +2.81%/−1.57%。若把新研究示例的 rate≥97% 预算用于 native 比较，两组均不达标；该预算不是旧 E0 预定判死规则。M0 全部 gap<1 s，联合 goodput 在本块退化成 TTFT 判据。|
| 质量/输出差异 | fresh M0 length 122/128、120/128；与 FIFO 输出 ID 不同 58/61，请求长度不同 2/3。旧八格诊断共 359/1024 有≥512 token 短周期重复，86 条整段为同一 token。旧诊断不能直接当 fresh 的重复统计；自然 EOS 不等于健康任务。|
| 已失败邻近机制 | C 无条件 first-fit：60 次越过，78 个请求被越过，主 goodput −8.7%/−13.1%，被越过者最大 flow 恶化 13.82/14.06 s；不重跑调参。A fit-first：max gap 3.795→12.186 s，亦是警示而非 M2 已判死。|
| 当前 UNRUN | 健康 instruction 双任务；当前无 offload 第二模型资格；M1/M2 动作；同信息 hard-cap 时间打包/最近邻基线；保护服务消融；5 独立 episode 确认；开销分位数/取消/中断安全 drain。计划和 CPU 检查不算这些 GPU 结果。|
| 状态滞后 | C/C_PAPER_DRAFT.md 和 checkpoint 仍写 fresh 下一步，实际被本轮完成回执取代；旧草稿 H1 GPU UNRUN 被 A 的较新资格/两配对结果取代。docs/current 的八月冻结只作历史。输入 JSON 的 FRESH_INPUTS_GPU_UNRUN 是不可变创建状态，不能覆盖运行回执。|
| A 任务/资源 | CURRENT 的 x：warm A/A r02 COMPLETE_ANALYZED；y 原记录 Q1/Q10 r03 锁阻塞。root 最新现场 09:08:55 UTC：warm 回执 CELLS_COMPLETE、两格 exit0；Q1/Q10 r03 RUNNING，controller6236、GPU pid6360 22300 MiB，见同目录 remote_A_receipts.txt。不声称 A 全部结束，不抢锁、不中断。|
| 授权边界 | 历史计划写共享 100 h，不等于本轮新 campaign 授权。当前用户明确先设计、复用已完成格；仅给 SSH 不自动重启 GPU campaign。E0 回收/本地核对可完成，后续 GPU 仍需实际授权范围和组级锁。|

**最弱因果环节：** M0 在几乎固定上限且可能退化的生成域，确实增加合法准入；尚未证明在健康任务和相同信息集强基线之后仍有有用请求收益。先做健康输出检查与 hard-cap packing 等价性挑战，信息增益高于扩矩阵或 M1/M2 控制器。

## E0 六格实际数据、配置和成本

| 顺序 | 合格/128 | goodput req/s | episode s | mean flow s | length/stop | preempt |
|---|---:|---:|---:|---:|---:|---:|
| native1 | 54 | .665782 | 81.108 | 35.963 | 123/5 | 43 |
| FIFO1 | 57 | .656831 | 86.780 | 37.079 | 120/8 | 0 |
| M0_1 | 63 | .734660 | 85.754 | 36.974 | 122/6 | 0 |
| M0_2 | 69 | .822365 | 83.904 | 35.760 | 120/8 | 0 |
| FIFO2 | 55 | .628381 | 87.526 | 37.618 | 119/9 | 0 |
| native2 | 53 | .651421 | 81.361 | 36.331 | 122/6 | 34 |

- 模型 `allenai/OLMoE-1B-7B-0924`，model/tokenizer revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，BF16；Python 3.12.3、torch 2.11.0+cu130、CUDA13、vLLM0.26.0、Transformers5.15.1、25 CPU threads。
- 新卡 RTX5090 `GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36`；旧卡 `GPU-3fc910c2…` 仅背景。block16；4096 可用 +1 null；每块2 MiB，实际配置 KV allocation `4097*2*1024*1024=8,592,031,744` B，可用8GiB。无 offload、prefix cache关闭、async关闭、non-spec、32序列、token budget1024、max context4096、chunked prefill开启、reserve_full_isl开启。
- Wikitext103 raw train，源码声明 revision `b08601e04326c79dfdd32d625aee71d232d685c3`；完整文章 eligible ranks832–959；prompt414–3066，资格 ceiling3072；一次 Poisson5/s、arrival seed20261001、span22.682329s；greedy seed20260905、EOS开启、min0、max1024。来源是已钉住本地 Parquet 字节，未声称六份历史 receipt 完整或 Arrow/Parquet 全片等价。
- 6个 launcher wall（包括初始化、warmup、服务、drain、退出）按顺序：231.294、122.512、120.430、120.171、127.964、118.323 s；整组841.391s=14.023min=0.23372 GPUh。均值140.116s/格，第一格明显更贵。M0 调度前适配器决策计时小计 .589546/.572630 s；它来自 perf_counter，不是进程 CPU 总时长，也不是全部 instrumentation税。
- 机械按本次均值估计300格+20%=14.01GPUh，只能作为旧128请求/该模型的预算刻度；新模型、512请求、持续到达须先测新资格时长，不是已授权额度。

## 代码和证据版本

R HEAD `76d6d88`，有 A 当前文档、adapter、20260929_commit_recheck 等未提交内容；只读保留。C worktree初读HEAD `f5b2750`；此处为读取时快照。共享工作树随后被其它会话提交，最终观察HEAD `df508795`且clean；本轮按固定适配器SHA解释，不将并发改动归为本轮。早期 `code-checkpoint-manifest.json` 仅为 `8920787`，不能代表最新 retirement 源码。

已核本地 D 的真实源字节：

| 文件 | SHA256 |
|---|---|
| C_NATIVE_RETIREMENT_ADMISSION_V1.py | 261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6 |
| C_NATIVE_MAX_BOUND_ADMISSION.py | b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee |
| C_NATIVE_RETIREMENT_FRESH_ANALYZE_V1.py | 0315e313c54b9b1dc39b377edd5ce91cc4a9578a6e5cbc1ac879e9959fe2041e |
| native_retirement_fresh_analysis_v1.json | 4042ead64598fe67b747c57fc0011e8925af67f9e5522fc208e7d532f9f7c4f6 |

fresh 分析原字节复制到本目录 `fresh_analysis_recovered.json`，相等且 SHA 同上；没有重算或覆盖来源。冻结 cell SHA `90c0065e8f004f98c636c1453caacaf678033144b96e9e74a708943e8b13703b`，block SHA `3233aebf21bf66e8d79befaae327c6d369a017e51a2a9ed1a3cf41e051110373`。最终依据是 C/c-native-retirement-fresh-v1/{block-start,block-receipt}.json、6个 launcher-receipt 和两份 retirement-envelope.json。回执核对两 M0 均 violations=[]、drained=true。

## E0–E7 入口与缺口

| 实验 | 可直接复用的真实入口 | 当前缺口及运行边界 |
|---|---|---|
| E0 | D/C_NATIVE_RETIREMENT_FRESH_{CELL,BLOCK,CONTRACT,ANALYZE}_V1.py；C/20261001_c_retirement_fresh_inputs_v1；C/c-native-retirement-fresh-v1 | COMPLETE；复用已有分析。BLOCK 是一次性硬编码启动器，禁止再执行现有输出身份。|
| E1 | D/C_SUSTAINED_TOKEN_PATTERN_DIAGNOSTIC.py 的最长短周期算法；raw requests/output_token_ids；已有 native capture/metrics | 旧 analyzer要求八格audit，不能直接假称fresh已测。健康问答+摘要/代码数据、质量评分与instruction模板验收脚本待实现。模型/数据下载不是本轮自动授权。|
| E2 | D/C_NATIVE_RETIREMENT_ENVELOPE_PROBE_V1.py（--previous、--output）；旧 native_retirement_envelope_opportunity_v1.json、native_bound_hol_opportunity_v1.json | 旧探针针对完整上界轨迹，仅证明结构机会，不能当M1/M2反事实。mixed-phase义务/机会分类与HoL预约反例脚本待实现；本轮16–32请求pilot仍UNRUN。|
| E3 | D/C_NATIVE_MAX_BOUND_ADMISSION.py、D/C_NATIVE_RETIREMENT_ADMISSION_V1.py；B0 cell运行路径真实存在 | 固定并发/水位校准、hard-cap时间容量打包、WAIT/CacheOPT可比组件待实现/资格；无已完成最近邻基线，不编造官方复现。|
| E4 | 现有native capture、metrics、完整drain和原始输出结构 | 泛化episode/模型参数化runner待实现；现有FRESH cell明确锁死128、OLMoE revision、inputSHA、GPU和max3072，不能直接改命令跑第二模型。300run只是规划，0正式新campaign执行。|
| E5 | 两个真实adapter可作为full-bound/M0起点 | 同decoder保护的full-bound臂、统一保护日志/成本路径待实现；不以旧 B1 未显式保护代表保护消融。|
| E6 | 既有输入完整request/document身份、arrivals和max_output参数语义 | KV/cap/prompt/arrival窄矩阵的冻结输入与参数化配置待实现；当前只有一个KV点、一个cap点，无敏感性结果。|
| E7 | retirement条件检查、120个小CPU端点例子的历史资格；两M0真实进度/释放记录；已有steps决策分位数见下 | 取消、预算不足、已承诺decoder进度中断与安全drain反例、整进程CPU和独立额外host内存待实现/测量；切回full-bound不是偿还旧承诺。|

离线复算 E0 的**真实命令**如下（仅在有具体复算需要时；本轮已复用旧分析，未重复执行），`--output`必须是新的不存在路径：

```sh
python3 /private/tmp/moe-research-c-20260929-v2/refine-logs/expert_saturation/experiments/admission_capacity/20260929_c_baseline_delivery/C_NATIVE_RETIREMENT_FRESH_ANALYZE_V1.py \
  --root '/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/c-native-retirement-fresh-v1' \
  --output '/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/research_design_v2/fresh_recompute_addendum.json'
```

已有 GPU cell CLI 为 `C_NATIVE_RETIREMENT_FRESH_CELL_V1.py --parent-package … --inputs-dir … --arm native_full|native_max_bound|native_retirement --output-dir …`，但必须由组级launcher持既有锁调用，且只适用原冻结身份；不能拿它充当 E1–E7 可立即开跑的通用命令。实际远端解释器 `/root/autodl-tmp/c-vllm-v026-venv/bin/python`，base `/root/autodl-tmp/c-research-20260930`；不再启动已 COMPLETE 的 E0。

第二模型可用性只做了有界本地核对：默认HF cache与/private/tmp顶层未发现Qwen权重。共享台账中的Qwen3-30B-A3B BF16是旧专家分页48槽/层、512MiB请求KV的r04实验，不能冒充当前无offload instruction第二模型资格。真正模型缓存/数据/授权需现场核查；未验证的模型revision或数据路径应保持待定。

## 已有 steps 的决策成本分位数补记

仅只读汇总 fresh 两份 retirement-envelope.json 的 `steps[].decision_s`，没有增加运行或修改原件。按排序后索引 `(n−1)*p` 线性插值；全部调度调用纳入，两个 sum 与原 receipt `decision_seconds` 完全相同。

| M0 | 调用数 | p50 μs | p95 μs | p99 μs | max μs | 调度前计时小计 s |
|---|---:|---:|---:|---:|---:|---:|
| retirement1 | 6175 | 100.922 | 129.937 | 170.495 | 408.676 | .5895458423 |
| retirement2 | 6170 | 98.951 | 123.912 | 157.549 | 664.588 | .5726299994 |

源码计时从 gate.schedule 入口到调用 native `original_schedule` 前；不含 native schedule、其后的进度检查/事件append/stepsappend、free钩子和日志落盘。含在线状态核对与准入计算，不能进一步全部归因于峰值公式。性能 episode 已含真实在线总成本；上述分位数不能替代端到端消融。

现有 memory-before/after-init/after 提供 GPU 参数、KV、torch allocated/reserved及峰值；两 M0 的 KV storage均8,592,031,744B、参数13,838,323,712B，after torch allocated均22,537,967,104B；这些桶互相包含，不能相加。它们不能辨别策略新增的 Python `active/protected/expected_outputs` 和累积 `steps/events` 的独立 host 开销；没有匹配的策略对象大小或进程CPU基线，**额外host内存与整进程CPU总时长为未测**。不从固定KV tensor大小推出策略内存零开销。
