# G 最邻近工作与可证伪贡献边界

审计日期：2026-10-08。仅使用论文原文、作者站点和 NVIDIA 官方资料。本文是文献边界审计，不是当前机器的可执行性或性能证据；硬件、精度和安装版本须由环境审计确认。

## 已有工作实际覆盖了什么

| 主源 | 已有决策及测量边界 | 对 G 的直接约束 |
|---|---|---|
| [RaMP, 2026，§IV–V](https://arxiv.org/html/2604.26039v1) | 由 token 数及路由直方图选择 kernel；wave cost model 的目标是局部时间。自研路径为 H200 SM90a 的 W8A8 FP8；服务实验使用 vLLM v0.9、`--enforce-eager`。每步首层 GPU 求选择后有一次 `.item()`，其后各层复用。 | 不能将其称作 BF16 RTX 可直接使用的候选，也不能声称首次利用路由选 kernel。论文的 4.3 KB 是 cost-model 系数大小，不能解释为全部 kernel/graph 的真实驻留成本。未见计划集合—KV—完整请求 SLO 的联合选择实验。 |
| [DA-MoE, 2026，§IV–V](https://arxiv.org/html/2607.23099v1) | 各 token 桶离线调优路由样例；在线 GPU 构建、排序直方图并匹配样例，以 conditional CUDA Graph 选择 kernel。实验为 B200、CUDA 13.1、FlashInfer 0.6.6，DeepSeek-V3/Kimi K2。主指标为 fused-MoE 与完整 MoE 层延迟。 | GPU 路由分派和多分支 graph 已有先例。文中的“end-to-end MoE-layer”不能当作完整请求 goodput。其 B200/TRTLLM kernel 不能默认兼容当前卡及 BF16；G 不应为了复现而新建 conditional graph 框架。 |
| [TensorRT-LLM 官方 graph 桶研究](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md) | 明确比较固定 `x2/+64/+8` graph 集合、padding、KV 容量、吞吐和启动成本。说明共享池由最大 graph 决定，差异主要来自池外 graph 数据；`+8` 相对 `+64` 的 KV tokens 在两个模型下降 17.2%/8.6%。Future Work 已提出用 serving logs 的并发分布自动选择 capture 桶。 | “更多 graph 更快却挤占 KV”本身已不是新发现；“按历史直方图自动挑桶”也是直接后续。其 GB200/NVFP4 数值不可迁移到当前环境，但应是强方法基线。 |
| [TASO: Time and Space Optimization，2020，§4.4–4.5](https://arxiv.org/pdf/2005.10709) | 通过 ILP 选择逐层 primitive 和 layout，在内存预算下优化时间；还显式建模跨层复用 workspace，其容量按最大需求而非简单求和。实验为嵌入式 CNN 推理。 | 不能将“内存约束全局 kernel 选择”“用 max 表示共享 workspace”单独作为创新。G 须证明 MoE 服务中保留计划集合与 KV 造成新的、可测的选择错误。勿混淆同名的 DNN 图替换 TASO。 |
| [μ-cuDNN，作者原文](https://htor.inf.ethz.ch/publications/img/ucudnn.pdf) | 在 workspace 约束下用 DP/ILP 联合选择卷积算法及 micro-batch 分解，保持运算语义。 | 算法选择与 batch 的联合优化也有先例。G 的差异只能来自固定调度策略下 KV 容量改变实际服务 batch，而不能泛称首次联合优化 batch 和 kernel。 |

[TensorRT-LLM 内存文档](https://nvidia.github.io/TensorRT-LLM/reference/memory.html)另说明：activation 内存存在复用和 profile 最大值，paged KV 在初始化时预分配。由此不能从某次 `reserved` 减少推断 KV 自动增加；应验证当前后端启动顺序及最终 blocks。

## 半页贡献假说（待实验，不是已成立的论文主张）

**缺口。** 上述局部选择器优化一次调用；已有 graph 调优已认识 KV 机会成本。仍可检验的窄缺口是：部署决策的成本对象应是完整计划集合，而候选集合改变 KV 容量后，又改变固定 scheduler 所产生的 batch 分布。这两个耦合是否会让逐层最快方案、可加成本模型或固定历史直方图选错，尚需本环境证据；本次有限审计不能证明全领域无人研究。

**静态组合边界。** 不能声称“调好的静态组合原则上解决不了”：给定部署存在最优静态组合，G 的首个原型也输出静态组合。能够主张的是一种可迁移的选择过程：在少量测量下，利用集合成本与容量反馈，可靠地得到接近该最优点的组合。若一个简单固定规则已取得同等结果，论文主张应撤回，而不是把它排除出基线。

**第一候选。** 仅在 A/B/C 通过后，保留少量合法且非支配的计划。用新进程、固定 capture 顺序测量集合 `S` 的真实驻留 `M(S)`，以及实际可分配 `K(S)`；不将单计划峰值相加。由相同输入和路由测得执行代价，先以已有轨迹形成少量候选，在确认容量约束能影响目标负载后，再对候选的真实 KV 容量做短服务校准，更新条件分布 `H(batch, route | S, K(S), arrivals)`。按预定 SLO 的预测 goodput 选一个组合并启动；每次校正须由增量决策价值和阶段预算支持，不实现在线驱逐或动态扩容。最终完整服务验证预测，不能拿局部时间代替结果。

**推翻条件。** 共享成本与容量反馈不改变候选排序；KV blocks 虽变但自然负载不受影响；固定直方图或简单固定组合与候选等效；或收益完全来自修复默认桶/过度 workspace 预留。任一成立即停止当前配置的论文探索。2026-10-08 重审后不再采用统一 +10% 门槛；判断须结合效应、运行波动、部署价值与方法增量。旧 SLO 缺应用依据，只作探索口径。

## 与主实验连接的最小证据

1. 记录两个以上实际执行路径的同输入、同路由正确性与局部时延；若同一路径只是参数别名，不能通过门槛 A。
2. 至少测紧凑集、局部最快集、简单预算固定集和至多一个候选的完整集合成本。逐项区分常驻 graph 数据、真实复用 workspace、allocator 历史与瞬时峰值；最终 blocks 为容量依据。
3. 只有 C 成立才做预固定低压/容量压力服务对照，主实验使 KV 随真实成本自然变化；固定 KV 仅作辅助归因。至少展示一次去掉集合成本或容量条件分布后是否真的选错，才可能支持方法贡献。
4. 若模型与自然轨迹从未进入容量敏感区，文献中的大模型正结果不构成换模型追结果的理由。应根据实际证据作域内收束或因证据不足暂存，不把未运行记为负结果。

本文未声称通过立项门槛，未起草论文摘要。
