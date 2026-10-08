# 两条替代 formulation 的有界查新

2026-10-02，后缀撤销八格完整服务比较为负之后。只核对直接动作与最小执行入口，没有实现新系统或运行新 GPU 组。

| 候选 | 直接先例与差异 | 当前决定 |
|---|---|---|
| 仅常驻专家生成近似 draft，由完整原 target 验证 | [SpecMoE §III-B](https://arxiv.org/html/2604.10152#S3.SS2) 已明确使用常驻专家、草稿阶段零专家搬运、免训练 drafter、完整 target 验证及常驻草稿专家更新；甚至包括 affinity 替代。不是 SpecMoEOff，也不是 MoE-Spec | 中心动作直接覆盖，不另起名字实施 |
| 精确保留 BF16 的专家传输压缩，GPU 解码后运行原 MoE | [ZipServ](https://arxiv.org/html/2603.17435v1) 已有 BaseExp、3-bit 指数编码及完整 BF16 例外；[DFloat11](https://arxiv.org/html/2504.11651v2#S3.SS2) 已有无损权重压缩与 GPU 解码；[ZipMoE](https://arxiv.org/html/2601.21198v1#S2.SS2) 已用于 MoE 专家卸载 | 接口可以移植，但目前只有运行域/工程差异，不造新编码器作为创新 |

[MoE-Spec](https://arxiv.org/html/2602.16052#S3) 修改验证阶段的专家贡献，与“近似仅发生于草稿、target完整”不同。[Draft & Verify](https://aclanthology.org/2024.acl-long.607/) 则已有免训练跳层自推测。当前 CPU ngram 接口若改为 resident self-draft，还需要 GPU draft loop、隔离近似路由与 tentative KV；不是切换一个 pager 标志。

压缩的最小原生接入位置为 `ensure_resident()`：用 compressed H2D + GPU decode 替代 BF16 `copy_`，解码到原 `scratch_w13/scratch_w2`，保持现有专家计算。原 BF16 scratch 仍需驻留，因此它不释放 KV 容量；压缩流暂存还会增加瞬时资源。不同论文报道的压缩后大小约为原始的68%–74%，其中 ZipMoE 的 Jetson UMA/SSD 域不能直接外推当前 PCIe 域。

串行路径的一阶盈亏条件为 `rB/P + B/D + delta < B/P`：r为压缩比例，P为真实传输率，D为GPU解码输出率，delta为额外提交/启动成本。r=0.70、忽略delta时需D>3.33P。若仅把当前约49.84GB/s的加载事件表观窗口率代入敏感性计算，12MiB专家留下约76微秒解码及提交余量；这个窗口不是独立纯DMA带宽，也不是已经测得的压缩收益。

本次没有找到足以直接替换的原创中心动作。下一步等总字节的专家/KV边际曲线属于另一候选的实际成本实验，不借这两条已知动作包装创新。

## 追加：ngram匹配位置携带历史路由

有限核查未支持将其升为中心机制：利用prompt-lookup返回的历史source positions，查已完成token的逐层route，再预测新draft的物理cache代价，本质上替换了EcoSpec式验证前代价预测器；后续预取也与ExpertFlow预测路由→cache/prefetch路径重合。相同token在新上下文中的route仍是预测，不是已知未来。未发现相同source-position接口不等于证明新颖；本次不实现。若以后已有独立方法需要免训练预测参照，可作为消融。来源沿用[EcoSpec §4](https://arxiv.org/html/2607.12696v1#S4)、[ExpertFlow §3.2–3.4](https://arxiv.org/html/2410.17954v2#S3.SS2)。

## 追加：常驻core与跨层共享执行槽

候选是将原16层各24槽改为各20个常驻槽加64个跨层共享临时槽，总384槽不变，完整active union一次执行，减少wave/sort/reduce；代价是长期驻留容量下降、以后可能多搬专家。核心权衡为本次少波的节省是否超过新增加载与未来替换成本。没有运行本候选，不能从旧固定route推导其完整服务收益。

本次不升为中心机制：[Mira §IV-C](https://arxiv.org/html/2609.38090v1#S4.SS3)已有同专家预算内HOT长驻＋STAGE短驻、随层经过回收；[fastllm作者实现](https://github.com/ztxz16/fastllm/blob/master/docs/cuda-expert-cache.md)已有跨层global key-to-slot、先保护当前命中、按slot索引直接计算。按真实U消除多波是尚待检验的窄目标差异，但固定20＋64容易退化为普通缓存保护，必须首先面对同384槽global-LRU单波参照，不能只赢固定每层24而声称新机制。

最小接口陷阱也已明确：vLLM0.26 alignment在ignore_invalid_experts=True时按expert_map结果索引num_experts大小数组，logical64＋physical320..383不能直接使用。可先按64逻辑专家排序再映射输出expert_ids，或扩大排序域到384；无需新GEMM。仅源码可行性，未GPU验证。对应[原生alignment源码](https://github.com/vllm-project/vllm/blob/v0.26.0/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu)。
