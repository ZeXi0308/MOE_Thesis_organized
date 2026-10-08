# C：LongBench MultiFieldQA-en 前 16 条输出资格结果

**结论：固定适配能产生任务相关回答与自然 EOS；尚无容量策略收益证据。** 冻结源序前 16 条全部完成，官方全文词级 F1 为 **52.00435/100**，13 条自然 EOS、3 条达到 64-token 上限，没有空输出。四条归一化 exact match；F1 不是答对率。三条封顶都是相关、连贯解释在句中截断，未见先前长续写实验中的重复循环。本格是开发资格观察，不是官方完整 LongBench 分数、未见确认或新控制器实验。

## 固定运行与完整分母

[事前计划](C_LONG_DOCUMENT_QA_PLAN_20261001.md)和执行代码在 `402302e` 冻结，输入证据说明在 `628c191`、GPU 输出前完成。模型是 `allenai/OLMoE-1B-7B-0924-Instruct` 修订 `7f1c97f440f06ce36705e4f2b843edb5925f4498`，BF16、greedy、seed 20260905、EOS 开启、无文本停止符、max_tokens=64。使用官方任务提示，3500-token 头尾截断后套 OLMoE chat；这套 4k 适配不等同官方列出的其他模型配置。源序 0–15 全数保留、同时外部到达、完整排空，没有按答案或长度筛选。

同一 RTX 5090 `GPU-3fc910c2…`、私有 vLLM 0.26.0；128 sequence、1024 batch tokens、4096 可用 KV 块，原生 FCFS、APC 和 chunked prefill 开启。冷缓存正式测量前执行已冻结首尾暖机并 reset APC。已有非阻塞资源锁覆盖加载至退出清理，未改公共环境。

| 指标 | 实测 |
| --- | ---: |
| 全部外部请求 / 完成 / 缺失 | 16 / 16 / 0 |
| 官方全文 mean F1 × 100 | 52.004350 |
| 归一化 exact match | 4/16 |
| 自然 EOS / 长度封顶 / 空输出 | 13 / 3 / 0 |
| 实际输出 token | 395 |
| 原生抢占 | 0 |
| after-schedule 占用峰值 / 可用块 | 1941 / 4096 |
| after-schedule running 峰值 | 10 |
| 调度调用 | 106 |
| 正式生成 host 观察区间 | 1.374292 s |

[完整运行汇总](long_document_qa_runtime_v1.json)给出：输出 token 均值24.6875，host TTFT/到达至完成均值0.554871/0.918041 s，每请求最大不同host返回间隔均值0.024991 s、最大0.029982 s。同一次host返回内的多个token不能分解为独立GPU间隔。waiting在call42、0.922008 s后首次清零且未重现，首次完成早于此（0.173257 s）；队列等待不能单独归因于KV。

模型准备11.874581 s、engine初始化40.448311 s、正式测量前warmup0.214030 s、measurement wrapper1.400347 s、shutdown0.710644 s、整个child60.554141 s。日志保留初始化compile9.72 s与MoE默认配置警告，没有检出inference JIT警告；单次带观察器时间仍不是稳态吞吐。生成观察区间包含在measurement wrapper内，不能重复相加。

逐请求全文、参考答案、原始 IDs 和结束原因保留在[完整评分 JSON](long_document_qa_qualification_v1.json)与原件中。

| 源序 | 输出 token | 结束 | 全文 F1 | 输出要点与限制 |
| ---: | ---: | --- | ---: | --- |
| 0 | 7 | EOS | 1.000000 | South West Ultras fan club |
| 1 | 20 | EOS | 0.444444 | 回答不需要 ISR，并补充条件 |
| 2 | 45 | EOS | 0.457143 | STM/STS，并额外列举光谱、计算和装置 |
| 3 | 64 | cap | 0.363636 | ICD 解释中途截断；关键原文支持已被输入截断删除 |
| 4 | 14 | EOS | 0.434783 | 防止变形，未给参考答案的锥形截面理由 |
| 5 | 64 | cap | 0.217391 | 先回答抑制 Kondo 效应，继续解释至截断 |
| 6 | 41 | EOS | 0.368421 | 电力日常用途的较长概述 |
| 7 | 4 | EOS | 1.000000 | Vice Admiral |
| 8 | 4 | EOS | 0.000000 | 1-0 是表中赛后累计战绩，参考选比赛比分 15-3 |
| 9 | 11 | EOS | 0.888889 | K3、K4、K5 |
| 10 | 30 | EOS | 0.774194 | 能切换电子态，省略改变吸附位置的机制 |
| 11 | 9 | EOS | 1.000000 | 3-D printing and software development；保留输入的人物关联不清 |
| 12 | 8 | EOS | 0.000000 | 90-120 mcg/day，没有女性/男性对应关系；官方归一化无重叠 |
| 13 | 4 | EOS | 0.333333 | Joule 对功率单位是错误回答，词重叠仍得部分 F1 |
| 14 | 64 | cap | 0.038462 | 先答 flexibility，再继续解释至截断 |
| 15 | 6 | EOS | 1.000000 | Jacob C. Landau |

不剪掉多余解释，不改 gold、单位归一化或 cap。[输出解读](C_LONG_DOCUMENT_QA_OUTPUT_NOTE_20261001.md)说明 F1 与事实正确性可能分离，以及 #8 的比分/战绩歧义。它不替代冻结的官方评分。

## 输入证据与容量边界

最终 prompt 为 1462–3513 tokens，12/16 做了头尾截断，14 个不同原始 context。GPU 输出前的[人工输入证据记录](C_LONG_DOCUMENT_QA_INPUT_EVIDENCE_20261001.md)判定 12 条关键支持保留、3 条被移除（#3/#12/#13）、1 条人物关联不明（#11）；全部样本仍保留。因此低分不能全归因于模型，常识性回答也不能证明模型读到了被删除的证据。

全部 16 条的独立完整输出上限预留为 3279 块，小于 4096，即使关闭 APC 也不能形成已知 cap 的容量不足。实测峰值 1941、零抢占与此一致。该结果不证明更长队列下仍无压力，也不证明准入控制会有收益；没有把 APC 关闭或 KV 缩小来制造压力。较长输入和短答案的混合 prefill/decode 过程需要结合真实队列观察，不能把峰值 running=10 直接解释为资源浪费。

## 原件与执行状态

运行 `2026-10-01T16:01:08.877540Z` 至 `16:02:12.374681Z`，SSH 62051 退出 0，child 12619 回收并退出 0，无超时或错误。GPU 退出后 2 MiB、无 compute process，私有 stage 已删除；13,838,721,960 字节持久权重 RAM cache 仍保留并计入内存。

原始根 `c-instruct-longbench-qualification-dev-v1` 在远端和稳定目录均保留，不重跑。完整压缩原件 446841 字节，SHA-256 `f83a7f928f7a5bf7a23effbbcf23cacf0dd361609022147ea807a20195009106`；本地复制后哈希相符。稳定目录为 `/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/`。执行 freeze SHA `964f6a270ddd6e7516ea99ff69b788bbe939a262a5ae945a1253d29f49688545`，workload SHA `504e28d18cc3fd959cedd5763df169d2e17ec74f768d69e7456fdc752d09c1b4`。

评分复算（在本文件目录，`$C_RAW` 指向稳定原始根）：

```sh
/private/tmp/moe-c-input-env/bin/python C_LONG_DOCUMENT_QA_ANALYZE_V1.py \
  --input-dir 20261001_c_long_document_qa_inputs_v1 \
  --run-dir "$C_RAW/native" \
  --metadata-dir 20261001_c_instruct_model_metadata_v1 \
  --output long_document_qa_qualification_v1.json
```

## 对下一步的约束

本格没有系统性空输出、无关续写或模板故障，故不触发事前的适配停止规则。它支持在保持任务协议和合理原生设置的条件下继续判断真实资源问题，不能直接支持新方法。唯一下一步为[完整官方150条的原生观察](C_LONG_DOCUMENT_QA_FULL_PLAN_20261001.md)，保留相同配置并预先规定无容量阻塞则停止该域。当前 GSM8K 容量方向继续停止；未选择新控制器。旧峰值公式的新颖性仍撤回，整个论文目标尚未完成。
