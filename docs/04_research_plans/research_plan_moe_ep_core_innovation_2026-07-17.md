# MoE EP 唯一核心创新严格研究计划

## 研究目标与查询类型

这是一个 depth-first 查询：目标不是罗列方向，而是从同一组 MoE EP 证据出发，以 backend semantics、数值机制、通信表示、系统瓶颈和发表新颖性等不同视角独立搜索，最终只选一个可证伪、可实现、适合硕士资源约束的核心机制。

需要回答的事实包括：最新本地实验（尤其 2026-07-17 CreditReduce P0-1 正式失败）对旧 QuotaEP-H、mixed precision 与 uniform FP8 baseline 的含义；DeepEP、NCCL EP、Hybrid-EP、MORI 等真实 combine/dispatch wire unit；2024-2026 年 mixed-precision collective、FP4/FP8 EP、compression/aggregation、tail-latency 控制的 prior art；Mac、单 GPU、多 GPU和多节点四级证据边界；候选机制的算法、ABI、buffer layout、复杂度、break-even 和硬停止条件。

## 三种研究方法及选择

方法一是从旧 QuotaEP-H 延伸，优点是复用资产多，缺点是已受到 uniform FP8 支配和 fixed quota quality tax 的直接威胁。方法二是从 backend 原语和真实 wire semantics 逆向寻找遗漏机制，优点是系统真实性最高。方法三是从最新负结果反推尚未被 baseline 吸收的新 observation，优点是最符合证据驱动且可低成本证伪。最终采用方法二与方法三结合：不默认 mixed precision，先用 backend 和负结果约束候选空间，再用独立 novelty/implementation 红队筛选。

## 并行研究流

研究流 A：本地证据审计。完整读取用户指定材料、四组 grouped-owner formal/round-robin artifacts，以及最新 CreditReduce P0-1 正式结果；输出时间有序的证据表、冲突结论和可继承 observation。

研究流 B：backend semantics。审查 DeepEP、NCCL EP、Hybrid-EP、MORI/MoRI 的官方代码、文档和论文，定位 dispatch/combine 的 wire unit、reduction order、LL/HT 路径、dtype、buffer contract、completion/overlap。

研究流 C：表示与聚合机制。搜索 2024-07-17 至 2026-07-17 的 FP4/FP8 EP、compressed collective、residual/sketch/low-rank/progressive representation 等工作，提出不依赖“BF16 recast 有害”的替代机制并做 prior-art collision。

研究流 D：系统瓶颈与反方。研究 MoE serving 中 combine/dispatch/expert compute/receiver/barrier 的真实瓶颈证据，主动检验“uniform FP8 已足够”“mixed precision 不回本”“该题只能做 characterization”等反方。

研究流 E：候选生成与红队。基于 A-D 的约束独立产生至少三类候选，再按 novelty、causality、observability、implementation、falsifiability、resource realism 淘汰。

## 微信公众号与网页检索策略

必须使用 `wechat-article-search` skill，时间范围设为 `2024-07-17 2026-07-17`，关键词包括“DeepEP MoE 通信优化”“NCCL EP Hybrid EP”“MoE FP8 FP4 通信”“MoE 推理 系统 优化”“分层 all-to-all MoE”。公众号结果只用于发现中文技术解读、版本线索和术语，不作为核心 novelty 的唯一证据；所有关键机制再用论文原文、官方仓库、正式文档和 release/commit 交叉验证。

网页检索与子 Agent 搜索采用同一时间范围，优先官方 GitHub、arXiv/OpenReview、USENIX/NSDI/ATC、NVIDIA/AMD 官方技术博客。关键事实至少由官方代码或原论文之一支持；时间敏感的 backend 事实以 2026-07-17 最新公开版本为准。

## 预期输出与必要性

输出严格按 A-K 结构：当前核心与缺口、三个候选、唯一主方案、完整设计、三条贡献、可证伪假设、实验矩阵、碰撞矩阵、3-5 个前置实验、标题摘要叙事、最终判决。每项都区分 [Observed]/[Inferred]/[Hypothesis]。本步骤不可拆除，因为只有同时覆盖本地证据、backend、prior art 与负结果，才能避免顺从性地复活已失败方向。

## 综合规则

任何依赖“预测未来拥塞”“训练一个控制器”“backend 将来支持 mixed dtype collective”而无当前可执行 contract 的路线标记 BLOCKED。任何被 uniform FP8、native HT/LL 或静态策略在 matched quality/matched wire 下支配的路线标记 KILLED。最终方案必须先有 Mac 上可直接杀死的 P0，再允许进入单 GPU；没有多 GPU 时不得声称 actual wire、completion、TPOT/P99。若所有系统方案均不通过，允许最终判决为 characterization/negative-result thesis，但不得虚构一个主创新。